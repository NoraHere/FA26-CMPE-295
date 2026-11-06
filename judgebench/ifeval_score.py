"""IFEval step 3 (GPU): score every judge-set response under each precision / rounding condition.

Pointwise, team protocol (q-judge core/judge.py): system + user prompt, thinking off, no sampling,
first-token logit(CORRECT) - logit(INCORRECT); label tokens from jb_eval's calibration prompts on the
BF16 judge. Labels (audit.jsonl) are never read here.

Conditions (--conditions):
  primary, the team's versions (q-judge):
    bf16
    tao_w8, tao_w4   torchao Int8/Int4 weight-only, group 128, lm_head excluded (real kernels)
    rtn_w4           q-judge simquant grid (min-max incl. 0, scale range/15, integer zero), fake quant
    gptq_w4          q-judge GPTQ on that grid; calibration = judge-format IFEval development prompts
    mnr_w4           gptq_w4 + margin-neutral flips tuned on IFEval development notes
  secondary, ours:
    i8g00, g00       offset-shifted RTN grids (quant_grids.py)
    gptq-g00, mnr-g00  GPTQ / MNR on grid g00 with wikitext-2 calibration and S9 tuning
Every condition starts from the BF16 weights. Output: results/<run>/scores.jsonl rows
{cond, item_id, margin, label_mass}; resumable per condition. --with-notes also scores
data/ifeval/notes.jsonl (add-on).

Usage (GPU node only):
  python ifeval_score.py --model-path ~/models/Qwen3-4B --limit-items 16    # smoke
  python ifeval_score.py --model-path ~/models/Qwen3-4B
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from quant_grids import apply_grid, parse_grid, target_linears

HERE = Path(__file__).resolve().parent
DEFAULT_CONDITIONS = "bf16,tao_w8,tao_w4,rtn_w4,gptq_w4,mnr_w4,i8g00,g00,gptq-g00"
TEAM = {"tao_w8": ("tao", 8), "tao_w4": ("tao", 4), "rtn_w4": ("team_rtn", 4), "rtn_w8": ("team_rtn", 8),
        "gptq_w4": ("team_gptq", 4), "mnr_w4": ("team_mnr", 4)}
CALIB_MAXLEN = 1024  # q-judge simquant.py

IF_SYSTEM = ("You are a strict grader of instruction following. Decide whether the candidate response "
             "follows every instruction in the user's request exactly. "
             "Reply with exactly one word: CORRECT or INCORRECT.")

IF_TEMPLATE = """Instructions:
{question}

Candidate response:
{response}

Does the candidate response follow every instruction exactly? Reply with exactly one word: CORRECT or INCORRECT."""


def parse_condition(cond):
    """-> (kind, arg): ('rtn', grid) | ('gptq'|'mnr', grid) | (team kind, bits)."""
    if cond in TEAM:
        return TEAM[cond]
    method, _, grid = cond.rpartition("-")
    method = method or "rtn"
    if method not in ("rtn", "gptq", "mnr") or (method != "rtn" and grid == "bf16"):
        raise ValueError(f"bad condition {cond!r}")
    parse_grid(grid)
    return method, grid


def load_tao(model_path, bits):
    """The team's tao_w8 / tao_w4 (q-judge core/judge.py), loaded as a second model."""
    import torch
    from transformers import AutoModelForCausalLM, TorchAoConfig
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig
    cfg = Int4WeightOnlyConfig(group_size=128) if bits == 4 else Int8WeightOnlyConfig(group_size=128)
    return AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
        quantization_config=TorchAoConfig(quant_type=cfg, modules_to_not_convert=["lm_head"])).eval()


def dev_calibration(judge, d, n):
    """q-judge recipe on IFEval: n judge-format development items spread over the sorted ids, each
    rendered with the judge's chat template and cut to CALIB_MAXLEN tokens."""
    with (d / "items.jsonl").open(encoding="utf-8") as f:
        rows = sorted((r for r in map(json.loads, f) if r["partition"] == "development"),
                      key=lambda r: r["item_id"])
    rows = rows[::max(1, len(rows) // n)][:n]
    ids = []
    for r in rows:
        chat = judge.format_chat(IF_TEMPLATE.format(question=r["question"], response=r["response"]))
        ids.append(judge.torch.tensor(judge.tok(chat, add_special_tokens=False).input_ids[:CALIB_MAXLEN]))
    return ids, [r["item_id"] for r in rows]


def dev_tuning(judge, d, n_prompts, batch_size):
    """MNR tuning set: rule-breaking development responses with and without the note, first n prompts."""
    from quant_methods import Tuning
    with (d / "items.jsonl").open(encoding="utf-8") as f:
        items = {r["item_id"]: r for r in map(json.loads, f) if r["partition"] == "development"}
    with (d / "notes.jsonl").open(encoding="utf-8") as f:
        notes = sorted((r for r in map(json.loads, f) if r["note_of"] in items), key=lambda r: r["item_id"])
    keys = sorted({r["key"] for r in notes})[:n_prompts]
    notes = [r for r in notes if r["key"] in keys]
    chats, pairs = [], []
    for r in notes:
        base = items[r["note_of"]]
        for row in (r, base):
            chats.append(judge.build(IF_TEMPLATE, row["question"], {"response": row["response"]})[0])
        pairs.append((len(chats) - 2, len(chats) - 1))
    return Tuning.from_chats(chats, pairs, len(keys), batch_size, "ifeval development notes")


def git_rev():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "."], cwd=HERE) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="/home/019111677/models/Qwen3-4B")
    ap.add_argument("--data-dir", default=str(HERE / "data" / "ifeval"))
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    ap.add_argument("--with-notes", action="store_true")
    ap.add_argument("--limit-items", type=int, default=0, help="smoke: first N items only")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--tune-problems", type=int, default=32, help="mnr tuning: S9 problems / IFEval prompts")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    conds = args.conditions.split(",")
    for c in conds:
        parse_condition(c)
    d = Path(args.data_dir)
    files = ["items.jsonl"] + (["notes.jsonl"] if args.with_notes else [])
    items = []
    for name in files:
        with (d / name).open(encoding="utf-8") as f:
            sub = [json.loads(line) for line in f]
        items += sub[:args.limit_items] if args.limit_items else sub

    model_tag = Path(args.model_path).name.lower()
    run_dir = Path(args.out_dir) / (args.run_name or f"ifeval_team_{model_tag}" + ("_smoke" if args.limit_items else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path, meta_path = run_dir / "scores.jsonl", run_dir / "meta.json"
    done = {}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for r in map(json.loads, f):
                done.setdefault(r["cond"], set()).add(r["item_id"])
    ids = {it["item_id"] for it in items}
    todo = [c for c in conds if not ids <= done.get(c, set())]
    print(f"[data] {len(items)} items from {files}; conditions todo {todo}", flush=True)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"model_path": args.model_path, "data_dir": args.data_dir, "files": files,
                 "n_items": len(items), "conditions": conds, "max_len": args.max_len,
                 "system": IF_SYSTEM, "template": IF_TEMPLATE,
                 "build_meta": json.loads((d / "build_meta.json").read_text())["sha256"]})
    meta.setdefault("provenance", []).append({"git": git_rev(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                                              "todo": todo, "time": time.strftime("%Y-%m-%d %H:%M:%S")})

    if todo:
        import torch
        import transformers
        from jb_eval import Judge
        from s9_score_grids import score_all

        judge = Judge(args.model_path, "bfloat16", args.max_len)
        judge.tok.padding_side = "left"
        meta["labels"] = {"pointwise": judge.calibrate("pointwise")}  # jb_eval probes, as before
        judge.system = IF_SYSTEM
        try:
            import torchao
            tao_version = torchao.__version__
        except ImportError:
            tao_version = None
        meta.update({"torch": torch.__version__, "transformers": transformers.__version__,
                     "torchao": tao_version, "gpu": torch.cuda.get_device_name(0)})
        built = [judge.build(IF_TEMPLATE, it["question"], {"response": it["response"]}) for it in items]
        meta["n_truncated"] = sum(cut["response"] > 0 for _, cut, _ in built)
        # score in length order (less padding), write in item order
        order = sorted(range(len(items)), key=lambda i: built[i][2])
        chats = [built[i][0] for i in order]
        linears = target_linears(judge.model)
        original = {n: m.weight.detach().to("cpu") for n, m in linears}
        ctxs = {}
        meta.setdefault("quant", {})
        for cond in todo:
            t0 = time.time()
            kind, arg = parse_condition(cond)
            apply_grid(linears, original, "bf16")
            bf16_model = None
            if kind == "rtn":
                meta["quant"][cond] = {"weight_err_over_scale": apply_grid(linears, original, arg)}
            elif kind == "tao":
                before = torch.cuda.memory_allocated()
                bf16_model, judge.model = judge.model, load_tao(args.model_path, arg)
                meta["quant"][cond] = {"toolchain": "torchao", "bits": arg, "group_size": 128, "excluded": ["lm_head"],
                                       "weight_mem_mib": (torch.cuda.memory_allocated() - before) >> 20}
            elif kind == "team_rtn":
                from quant_methods import team_rtn
                with torch.no_grad():
                    for n, m in linears:
                        m.weight.copy_(team_rtn(original[n].to(m.weight.device), arg).to(m.weight.dtype))
                meta["quant"][cond] = {"grid": "q-judge simquant min-max incl. 0, integer zero", "bits": arg}
            else:
                from quant_methods import Calibrated
                team = kind.startswith("team_")
                method = kind.replace("team_", "")
                key = kind
                if key not in ctxs:
                    if team:
                        calib_ids, calib_items = dev_calibration(judge, d, args.calib_samples)
                        tuning = dev_tuning(judge, d, args.tune_problems, 8) if method == "mnr" else None
                        ctxs[key] = Calibrated(judge, method, calib_ids=calib_ids, tuning=tuning,
                                               calib_desc=f"ifeval development, judge format, <= {CALIB_MAXLEN} tokens")
                        ctxs[key].meta["calib_item_ids"] = calib_items
                    else:  # S9 tuning chats use jb_eval's math prompt, without the IFEval system line
                        judge.system = None
                        try:
                            ctxs[key] = Calibrated(judge, method, calib_samples=args.calib_samples,
                                                   tune_problems=args.tune_problems)
                        finally:
                            judge.system = IF_SYSTEM
                q = ctxs[key].quantize(judge, None, team_bits=arg) if team else ctxs[key].quantize(judge, arg)
                meta["quant"][cond] = dict(ctxs[key].meta, **q)
            scores = score_all(judge, chats, args.batch_size)
            if bf16_model is not None:
                del judge.model
                judge.model = bf16_model
                torch.cuda.empty_cache()
            by_item = [None] * len(items)
            for pos, i in enumerate(order):
                by_item[i] = scores[pos]
            with out_path.open("a", encoding="utf-8") as f:
                for it, (margin, mass) in zip(items, by_item):
                    if it["item_id"] not in done.get(cond, set()):
                        f.write(json.dumps({"cond": cond, "item_id": it["item_id"], "margin": margin,
                                            "label_mass": mass}) + "\n")
            accepted = sum(m > 0 for m, _ in by_item)
            print(f"[{cond}] {len(items)} items in {time.time() - t0:.0f}s; accepted {accepted}/{len(items)}; "
                  f"mean label mass {sum(x for _, x in by_item) / len(items):.3f}", flush=True)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
