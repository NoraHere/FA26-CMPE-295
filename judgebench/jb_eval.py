"""JudgeBench (GPT split) evaluation with first-token logit judges.

Tasks (--tasks, default math,code): math (livebench-math 56, mmlu-pro-math 11), code (livecodebench 42),
reasoning (livebench-reasoning 98), knowledge (the 13 non-math MMLU-Pro subjects, 11 pairs each = 143).

Standalone until the team qjudge harness is available; mirrors its protocol:
thinking off, no sampling, score = first-token logit margin, label tokens chosen
by measurement on fixed calibration prompts (never on JudgeBench items).

Modes
  pointwise: margin = logit(CORRECT) - logit(INCORRECT) on each response;
             the response with the higher margin wins (exact tie = wrong).
  pairwise:  m = logit(A) - logit(B) in original and reversed order;
             debiased = (m_orig - m_rev) / 2  (> 0 means response_A wins).

Gold labels are only read in score_*() after the model call; they never enter a prompt.

Usage (GPU node only):
  python jb_eval.py --mode both --limit 4
  python jb_eval.py --mode both --grid g05     # INT4 grid 5 of 16 (see quant_grids.py)
  python jb_eval.py --tasks knowledge,reasoning --method gptq --grid g05   # see quant_methods.py
"""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from quant_grids import apply_grid, parse_grid, target_linears

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "/home/019111677/models/Qwen3-1.7B"
DEFAULT_DATA = "/home/019111677/datasets/JudgeBench/data/gpt-00000-of-00001.jsonl"

MMLU_PRO_NON_MATH = ["biology", "business", "chemistry", "computer science", "economics", "engineering",
                     "health", "history", "law", "other", "philosophy", "physics", "psychology"]
TASK_OF_SOURCE = {
    "livebench-math": "math",
    "mmlu-pro-math": "math",
    "livecodebench": "code",
    "livebench-reasoning": "reasoning",
    **{f"mmlu-pro-{subject}": "knowledge" for subject in MMLU_PRO_NON_MATH},
}
EXPECTED_COUNTS = {"livebench-math": 56, "mmlu-pro-math": 11, "livecodebench": 42, "livebench-reasoning": 98,
                   **{f"mmlu-pro-{subject}": 11 for subject in MMLU_PRO_NON_MATH}}
TASKS = ["math", "code", "reasoning", "knowledge"]
DEFAULT_TASKS = "math,code"
TRUNC_MARKER = "\n...[truncated]"

# Prompts. POINTWISE_TEMPLATE should be replaced by the qjudge prompt once that repo is available.
POINTWISE_TEMPLATE = """You are grading a response to a problem.

Problem:
{question}

Response:
{response}

Is the response's final answer correct? Answer with exactly one word: CORRECT or INCORRECT."""

PAIRWISE_TEMPLATE = """You are an impartial judge comparing two responses to the same problem.

Problem:
{question}

Response A:
{response_a}

Response B:
{response_b}

Which response is correct? Answer with exactly one letter: A or B."""

# Label candidates: first token of each spelling is used; the variant set with the most
# probability mass on the calibration prompts below wins.
LABEL_VARIANTS = {
    "pointwise": [("CORRECT", "INCORRECT"), (" CORRECT", " INCORRECT"), ("Correct", "Incorrect")],
    "pairwise": [("A", "B"), (" A", " B")],
}

# Calibration prompts: trivial, not from JudgeBench or GSM8K.
CALIBRATION = {
    "pointwise": [
        {"question": "What is 2 + 3?", "response": "2 + 3 = 5. The answer is 5."},
        {"question": "What is 4 * 6?", "response": "4 * 6 = 20. The answer is 20."},
    ],
    "pairwise": [
        {"question": "What is 7 - 2?", "response_a": "The answer is 5.", "response_b": "The answer is 4."},
        {"question": "What is 9 + 1?", "response_a": "The answer is 11.", "response_b": "The answer is 10."},
    ],
}


# ---------------------------------------------------------------- data

def load_items(data_path, limit, seed, tasks=("math", "code")):
    with open(data_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    items = [dict(r, task=TASK_OF_SOURCE[r["source"]]) for r in rows
             if TASK_OF_SOURCE.get(r["source"]) in tasks]
    counts = Counter(r["source"] for r in items)
    expected = {k: v for k, v in EXPECTED_COUNTS.items() if TASK_OF_SOURCE[k] in tasks}
    if dict(counts) != expected:
        raise SystemExit(f"Unexpected source counts {dict(counts)}, expected {expected}")
    for r in items:
        if r["label"] not in ("A>B", "B>A"):
            raise SystemExit(f"Unexpected label {r['label']!r} for {r['pair_id']}")
    items.sort(key=lambda r: r["pair_id"])
    if limit:
        # stratified: equal share per task (remainder to the first tasks), fixed seed
        rng = random.Random(seed)
        picked = []
        for i, task in enumerate(tasks):
            n = limit // len(tasks) + (i < limit % len(tasks))
            picked += rng.sample([r for r in items if r["task"] == task], n)
        items = picked
    return items


def cluster_key(item):
    # original_id is null for livebench/livecodebench; every question there is unique,
    # so those pairs are their own cluster.
    if item.get("original_id") is not None:
        return f"oid:{item['original_id']}"
    return f"pair:{item['pair_id']}"


# ---------------------------------------------------------------- judge

class Judge:
    def __init__(self, model_path, dtype_name, max_len):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA not available; run this on a GPU node via sbatch/srun.")
        major, _ = torch.cuda.get_device_capability(0)
        if dtype_name == "bfloat16" and major < 8:
            raise SystemExit(f"{torch.cuda.get_device_name(0)} lacks bf16 support; use --dtype float16 "
                             "or request an A100/H100.")
        self.dtype = getattr(torch, dtype_name)
        self.tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=self.dtype, device_map={"": 0}, local_files_only=True
        ).eval()
        self.max_len = max_len
        self.labels = {}

    def format_chat(self, user_text):
        # optional system message (set judge.system, e.g. ifeval_score.py's team-style prompt)
        msgs = ([{"role": "system", "content": self.system}] if getattr(self, "system", None) else []) \
            + [{"role": "user", "content": user_text}]
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def n_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def build(self, template, question, responses):
        """Format the prompt, truncating responses (never the question or instruction) to fit.

        responses: dict placeholder -> text. Returns (chat_text, {placeholder: tokens_cut}).
        """
        resp_ids = {k: self.tok(v, add_special_tokens=False).input_ids for k, v in responses.items()}
        empty = self.format_chat(template.format(question=question, **{k: "" for k in responses}))
        budget = self.max_len - self.n_tokens(empty) - len(responses) * self.n_tokens(TRUNC_MARKER)
        if budget <= 0:
            raise ValueError("Question + instruction alone exceed max_len")
        keep = {k: len(v) for k, v in resp_ids.items()}
        for _ in range(5):
            while sum(keep.values()) > budget:
                # trim the currently longest response by one token (water-filling)
                k = max(keep, key=keep.get)
                keep[k] -= 1
            cut = {k: len(resp_ids[k]) - keep[k] for k in resp_ids}
            texts = {
                k: responses[k] if cut[k] == 0 else self.tok.decode(resp_ids[k][: keep[k]]) + TRUNC_MARKER
                for k in responses
            }
            chat = self.format_chat(template.format(question=question, **texts))
            n = self.n_tokens(chat)
            if n <= self.max_len:
                return chat, cut, n
            budget -= n - self.max_len  # decode/re-encode can shift boundaries slightly
        raise ValueError(f"Prompt still {n} > {self.max_len} tokens after truncation")

    def next_logits(self, chat_text):
        torch = self.torch
        ids = self.tok(chat_text, return_tensors="pt", add_special_tokens=False).input_ids.to(0)
        with torch.inference_mode():
            out = self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        return out.logits[0, -1].float()

    def calibrate(self, mode):
        """Pick label tokens by probability mass on calibration prompts."""
        torch = self.torch
        template = POINTWISE_TEMPLATE if mode == "pointwise" else PAIRWISE_TEMPLATE
        probs = []
        for c in CALIBRATION[mode]:
            fields = {k: v for k, v in c.items() if k != "question"}
            chat, _, _ = self.build(template, c["question"], fields)
            probs.append(torch.softmax(self.next_logits(chat), -1))
        report = []
        best = None
        for pos, neg in LABEL_VARIANTS[mode]:
            pos_ids = self.tok.encode(pos, add_special_tokens=False)
            neg_ids = self.tok.encode(neg, add_special_tokens=False)
            if pos_ids[0] == neg_ids[0]:
                report.append({"variant": [pos, neg], "skipped": "same first token"})
                continue
            mass = sum(float(p[pos_ids[0]] + p[neg_ids[0]]) for p in probs) / len(probs)
            entry = {
                "variant": [pos, neg],
                "token_ids": [pos_ids[0], neg_ids[0]],
                "tokens": [self.tok.decode([pos_ids[0]]), self.tok.decode([neg_ids[0]])],
                "single_token": [len(pos_ids) == 1, len(neg_ids) == 1],
                "mean_mass": mass,
            }
            report.append(entry)
            if best is None or mass > best["mean_mass"]:
                best = entry
        if best is None:
            raise SystemExit(f"No usable label variant for {mode}")
        self.labels[mode] = best
        return {"chosen": best, "candidates": report}

    def margin(self, mode, chat_text):
        torch = self.torch
        logits = self.next_logits(chat_text)
        pos_id, neg_id = self.labels[mode]["token_ids"]
        probs = torch.softmax(logits, -1)
        mass = float(probs[[pos_id, neg_id]].sum())
        # what the judge would actually say first; diagnoses low label mass
        top_p, top_i = probs.topk(5)
        top = [[self.tok.decode([int(i)]), round(float(p), 4)] for p, i in zip(top_p, top_i)]
        return float(logits[pos_id] - logits[neg_id]), mass, top


# ---------------------------------------------------------------- per-item runs

def run_pointwise(judge, item):
    rec = {}
    for side in ("A", "B"):
        chat, cut, n = judge.build(
            POINTWISE_TEMPLATE, item["question"], {"response": item[f"response_{side}"]})
        m, mass, top = judge.margin("pointwise", chat)
        rec[f"margin_{side}"] = m
        rec[f"label_mass_{side}"] = mass
        rec[f"top5_{side}"] = top
        rec[f"trunc_tokens_{side}"] = cut["response"]
        rec[f"prompt_tokens_{side}"] = n
    return rec


def run_pairwise(judge, item):
    rec = {}
    orders = {"orig": (item["response_A"], item["response_B"]), "rev": (item["response_B"], item["response_A"])}
    for name, (first, second) in orders.items():
        chat, cut, n = judge.build(
            PAIRWISE_TEMPLATE, item["question"], {"response_a": first, "response_b": second})
        m, mass, top = judge.margin("pairwise", chat)
        rec[f"m_{name}"] = m
        rec[f"label_mass_{name}"] = mass
        rec[f"top5_{name}"] = top
        rec[f"trunc_tokens_{name}"] = [cut["response_a"], cut["response_b"]]
        rec[f"prompt_tokens_{name}"] = n
    return rec


def pick(score):
    """score > 0 -> A, < 0 -> B, exactly 0 -> tie."""
    return "A" if score > 0 else "B" if score < 0 else "tie"


def score_pointwise(rec, gold):
    p = pick(rec["margin_A"] - rec["margin_B"])
    return {"pick": p, "correct": p == gold, "tie": p == "tie",
            "truncated": rec["trunc_tokens_A"] > 0 or rec["trunc_tokens_B"] > 0}


def score_pairwise(rec, gold):
    p_orig = pick(rec["m_orig"])
    # in reversed order slot A holds response_B, so map back to original identity
    p_rev = {"A": "B", "B": "A", "tie": "tie"}[pick(rec["m_rev"])]
    p_deb = pick((rec["m_orig"] - rec["m_rev"]) / 2)
    return {
        "debiased_score": (rec["m_orig"] - rec["m_rev"]) / 2,
        "pick_orig": p_orig, "pick_rev": p_rev, "pick_debiased": p_deb,
        "correct_orig": p_orig == gold, "correct_rev": p_rev == gold, "correct_debiased": p_deb == gold,
        "consistent": p_orig == p_rev and p_orig != "tie",
        "first_slot_orig": rec["m_orig"] > 0, "first_slot_rev": rec["m_rev"] > 0,
        "truncated": any(rec["trunc_tokens_orig"]) or any(rec["trunc_tokens_rev"]),
    }


# ---------------------------------------------------------------- stats

def cluster_bootstrap(values, clusters, n_boot, seed):
    """Mean of values with a 95% percentile CI, resampling clusters with replacement."""
    by_c = defaultdict(list)
    for v, c in zip(values, clusters):
        by_c[c].append(float(v))
    groups = [(sum(v), len(v)) for v in by_c.values()]
    point = sum(s for s, _ in groups) / sum(n for _, n in groups)
    rng = random.Random(seed)
    k = len(groups)
    stats = []
    for _ in range(n_boot):
        draw = [groups[rng.randrange(k)] for _ in range(k)]
        stats.append(sum(s for s, _ in draw) / sum(n for _, n in draw))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[min(n_boot - 1, int(0.975 * n_boot))]
    return {"mean": point, "ci95": [lo, hi], "n_items": len(values), "n_clusters": k}


METRICS = {
    "pointwise": {
        "accuracy": lambda r: r["correct"],
        "tie_rate": lambda r: r["tie"],
        "truncated_rate": lambda r: r["truncated"],
        # probability on the two label tokens; low mass = margin read from the tail
        "label_mass_mean": lambda r: (r["label_mass_A"] + r["label_mass_B"]) / 2,
        "low_mass_rate": lambda r: ((r["label_mass_A"] < 0.5) + (r["label_mass_B"] < 0.5)) / 2,
    },
    "pairwise": {
        "accuracy_orig": lambda r: r["correct_orig"],
        "accuracy_rev": lambda r: r["correct_rev"],
        "accuracy_debiased": lambda r: r["correct_debiased"],
        "order_consistency": lambda r: r["consistent"],
        # share of decisions (2 per pair) that picked slot A
        "first_slot_pick_rate": lambda r: (r["first_slot_orig"] + r["first_slot_rev"]) / 2,
        "truncated_rate": lambda r: r["truncated"],
        "label_mass_mean": lambda r: (r["label_mass_orig"] + r["label_mass_rev"]) / 2,
        "low_mass_rate": lambda r: ((r["label_mass_orig"] < 0.5) + (r["label_mass_rev"] < 0.5)) / 2,
    },
}


def summarize(records, n_boot, seed):
    out = {}
    for mode, metrics in METRICS.items():
        recs = [r for r in records if r["mode"] == mode]
        if not recs:
            continue
        out[mode] = {}
        for group in ["all"] + TASKS:
            sub = recs if group == "all" else [r for r in recs if r["task"] == group]
            if not sub:
                continue
            clusters = [r["cluster"] for r in sub]
            out[mode][group] = {
                name: cluster_bootstrap([f(r) for r in sub], clusters, n_boot, seed)
                for name, f in metrics.items()
            }
            out[mode][group]["n_truncated_items"] = sum(r["truncated"] for r in sub)
    return out


def write_markdown(path, meta, summary):
    lines = [f"# JudgeBench eval: {meta['run_name']}", ""]
    lines += [f"- model: `{meta['model_path']}` ({meta['dtype']}, grid {meta.get('grid', 'bf16')}, "
              f"method {meta.get('method', 'rtn')}), tasks {meta.get('tasks', DEFAULT_TASKS)}, "
              f"GPU: {meta.get('gpu')}",
              f"- items: {meta['n_items']} pairs, max_len {meta['max_len']}, "
              f"bootstrap {meta['n_boot']} resamples clustered by original_id (pair_id if null)",
              f"- torch {meta.get('torch')}, transformers {meta.get('transformers')}", ""]
    for mode, chosen in meta.get("labels", {}).items():
        c = chosen["chosen"]
        lines.append(f"- {mode} label tokens: {c['tokens']} ids {c['token_ids']}, "
                     f"calibration mass {c['mean_mass']:.3f}")
    lines.append("")
    for mode, groups in summary.items():
        lines += [f"## {mode}", "", "| group | metric | mean | 95% CI | items | clusters |",
                  "|---|---|---|---|---|---|"]
        for group, metrics in groups.items():
            for name, v in metrics.items():
                if name == "n_truncated_items":
                    continue
                lines.append(f"| {group} | {name} | {v['mean']:.3f} | "
                             f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['n_items']} | {v['n_clusters']} |")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pointwise", "pairwise", "both"], default="both")
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--data-path", default=DEFAULT_DATA)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--tasks", default=DEFAULT_TASKS, help=f"comma list of {TASKS}")
    ap.add_argument("--limit", type=int, default=0, help="subsample stratified by task; 0 = all pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--summarize-only", action="store_true", help="recompute summary from items.jsonl")
    ap.add_argument("--grid", default="bf16",
                    help="weight grid from quant_grids.py: bf16 (default), g00..g15 (INT4), i8g00..i8g15 (INT8)")
    ap.add_argument("--method", choices=["rtn", "gptq", "mnr"], default="rtn",
                    help="rounding on the grid: rtn (nearest), gptq, mnr (margin-neutral); see quant_methods.py")
    ap.add_argument("--calib-samples", type=int, default=128, help="gptq/mnr: wikitext-2 windows of 512 tokens")
    ap.add_argument("--tune-problems", type=int, default=32, help="mnr: S9 development problems for tuning")
    args = ap.parse_args()
    parse_grid(args.grid)  # fail fast on a bad name
    tasks = args.tasks.split(",")
    if not tasks or any(t not in TASKS for t in tasks):
        raise SystemExit(f"--tasks must be a comma list of {TASKS}")
    if args.grid != "bf16" and args.dtype != "bfloat16":
        raise SystemExit("--grid rounds from the BF16 weights; use --dtype bfloat16")
    if args.method != "rtn" and args.grid == "bf16":
        raise SystemExit(f"--method {args.method} needs a --grid")

    modes = ["pointwise", "pairwise"] if args.mode == "both" else [args.mode]
    model_tag = Path(args.model_path).name.lower()
    grid_tag = ("" if args.grid == "bf16" else f"_{args.grid}" if args.method == "rtn"
                else f"_{args.method}-{args.grid}")
    task_tag = "" if args.tasks == DEFAULT_TASKS else "_" + "-".join(tasks)
    run_name = args.run_name or (f"jb_{model_tag}_{args.dtype}{grid_tag}{task_tag}_{args.mode}"
                                 f"_n{args.limit or 'all'}")
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    items_path = run_dir / "items.jsonl"
    meta_path = run_dir / "meta.json"

    items = load_items(args.data_path, args.limit, args.seed, tasks)
    print(f"[data] {len(items)} pairs: {dict(Counter(r['task'] for r in items))}", flush=True)

    done = set()
    if items_path.exists():
        with items_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["pair_id"], r["mode"]))
    print(f"[resume] {len(done)} (pair, mode) records already in {items_path}", flush=True)

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for key, default, new in [("grid", "bf16", args.grid), ("method", "rtn", args.method),
                              ("tasks", DEFAULT_TASKS, args.tasks)]:
        if meta and meta.get(key, default) != new:
            raise SystemExit(f"{run_dir} was run with {key} {meta.get(key, default)}, not {new}; "
                             "use a new --run-name.")
    meta.update({
        "grid": args.grid, "method": args.method, "tasks": args.tasks,
        "run_name": run_name, "model_path": args.model_path, "data_path": args.data_path,
        "dtype": args.dtype, "max_len": args.max_len, "limit": args.limit, "seed": args.seed,
        "n_boot": args.n_boot, "n_items": len(items), "modes": modes,
        "pair_ids": [r["pair_id"] for r in items],
        "prompt_sha256": {
            "pointwise": hashlib.sha256(POINTWISE_TEMPLATE.encode()).hexdigest()[:16],
            "pairwise": hashlib.sha256(PAIRWISE_TEMPLATE.encode()).hexdigest()[:16],
        },
    })

    todo = [(r, m) for m in modes for r in items if (r["pair_id"], m) not in done]
    if todo and not args.summarize_only:
        import torch
        import transformers
        judge = Judge(args.model_path, args.dtype, args.max_len)
        if args.grid != "bf16" and args.method == "rtn":
            linears = target_linears(judge.model)
            # in place: each weight is rounded from itself
            meta["weight_err_over_scale"] = apply_grid(linears, {n: m.weight for n, m in linears}, args.grid)
            print(f"[grid] {args.grid}: {len(linears)} linear layers rounded, "
                  f"max err/s {meta['weight_err_over_scale']:.3f}", flush=True)
        elif args.grid != "bf16":
            from quant_methods import Calibrated
            ctx = Calibrated(judge, args.method, calib_samples=args.calib_samples,
                             tune_problems=args.tune_problems)
            meta["quant"] = dict(ctx.meta, **ctx.quantize(judge, args.grid))
            del ctx
            torch.cuda.empty_cache()
            print(f"[grid] {args.method} {args.grid}: {json.dumps(meta['quant'])}", flush=True)
        meta.update({"torch": torch.__version__, "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0]})
        labels = meta.setdefault("labels", {})
        for mode in modes:
            cal = judge.calibrate(mode)
            prev = labels.get(mode)
            if prev and prev["chosen"]["token_ids"] != cal["chosen"]["token_ids"]:
                raise SystemExit(f"Label tokens changed vs existing run ({prev['chosen']['token_ids']} -> "
                                 f"{cal['chosen']['token_ids']}); use a new --run-name.")
            labels[mode] = cal
            print(f"[labels] {mode}: {cal['chosen']['tokens']} ids {cal['chosen']['token_ids']} "
                  f"mass {cal['chosen']['mean_mass']:.3f}", flush=True)
        meta_path.write_text(json.dumps(meta, indent=2))

        t0 = time.perf_counter()
        with items_path.open("a", encoding="utf-8") as f:
            for i, (item, mode) in enumerate(todo, 1):
                t = time.perf_counter()
                rec = run_pointwise(judge, item) if mode == "pointwise" else run_pairwise(judge, item)
                gold = item["label"][0]  # "A>B" -> "A"; used only for scoring, after the model call
                rec.update(score_pointwise(rec, gold) if mode == "pointwise" else score_pairwise(rec, gold))
                rec.update({"pair_id": item["pair_id"], "mode": mode, "task": item["task"],
                            "source": item["source"], "cluster": cluster_key(item), "gold": gold,
                            "seconds": time.perf_counter() - t})
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[{i}/{len(todo)}] {mode} {item['task']} {item['pair_id'][:8]} "
                      f"{ {k: rec[k] for k in ('pick', 'pick_orig', 'pick_rev', 'pick_debiased') if k in rec} } "
                      f"gold={gold} {rec['seconds']:.2f}s", flush=True)
        print(f"[done] {len(todo)} records in {time.perf_counter() - t0:.1f}s", flush=True)

    keep = {r["pair_id"] for r in items}
    with items_path.open(encoding="utf-8") as f:
        records = [r for r in map(json.loads, f) if r["pair_id"] in keep and r["mode"] in modes]
    summary = summarize(records, args.n_boot, args.seed)
    meta_path.write_text(json.dumps(meta, indent=2))
    (run_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
    write_markdown(run_dir / "summary.md", meta, summary)
    print((run_dir / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
