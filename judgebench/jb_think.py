"""Exploratory: does letting the judge think first help on JudgeBench? (BF16, pointwise only)

Same pointwise prompt as jb_eval.py, but with Qwen3 thinking on: the judge first writes a greedy
<think> ... </think> trace (capped at --think-budget new tokens; a capped trace is closed with
"</think>"), then the score is the usual first-token logit(CORRECT) - logit(INCORRECT) right after
"</think>\\n\\n". The response with the higher margin wins (exact tie = wrong). Greedy decoding, no
sampling, so it stays deterministic; not the team protocol (thinking off), so results are exploratory.

Output: results/<run>/items.jsonl (one row per pair x side, with the trace), summary.json/.md including
the paired accuracy change vs the thinking-off run of the same judge (--compare-run), if present.

Usage (GPU node only):
  python jb_think.py --model-path ~/models/Qwen3-8B --tasks reasoning
"""

import argparse
import json
import time
from pathlib import Path

from jb_eval import POINTWISE_TEMPLATE, Judge, cluster_bootstrap, cluster_key, load_items, pick

HERE = Path(__file__).resolve().parent
THINK_END = "</think>"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="/home/019111677/models/Qwen3-8B")
    ap.add_argument("--data-path", default="/home/019111677/datasets/JudgeBench/data/gpt-00000-of-00001.jsonl")
    ap.add_argument("--tasks", default="reasoning")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--think-budget", type=int, default=4096, help="max new tokens of thinking")
    ap.add_argument("--max-len", type=int, default=8192, help="prompt budget before thinking")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--compare-run", default=None, help="thinking-off jb_eval run dir (default: matching name)")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    model_tag = Path(args.model_path).name.lower()
    task_tag = "-".join(tasks)
    run_dir = Path(args.out_dir) / (args.run_name or f"jb_think_{model_tag}_{task_tag}_n{args.limit or 'all'}")
    run_dir.mkdir(parents=True, exist_ok=True)
    items_path = run_dir / "items.jsonl"
    items = load_items(args.data_path, args.limit, args.seed, tasks)
    done = set()
    if items_path.exists():
        with items_path.open(encoding="utf-8") as f:
            done = {(r["pair_id"], r["side"]) for r in map(json.loads, f)}
    todo = [(it, side) for it in items for side in ("A", "B") if (it["pair_id"], side) not in done]
    print(f"[data] {len(items)} pairs, {len(todo)} responses to judge", flush=True)
    meta = {"model_path": args.model_path, "tasks": args.tasks, "limit": args.limit,
            "think_budget": args.think_budget, "max_len": args.max_len, "decoding": "greedy",
            "n_items": len(items)}

    if todo:
        import torch
        judge = Judge(args.model_path, "bfloat16", args.max_len)
        meta["labels"] = judge.calibrate("pointwise")  # thinking off, as in jb_eval
        pos_id, neg_id = judge.labels["pointwise"]["token_ids"]
        end_id = judge.tok.convert_tokens_to_ids(THINK_END)
        # same prompt, thinking on
        judge.format_chat = lambda text: judge.tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
            enable_thinking=True)
        judge.tok.padding_side = "left"
        t0 = time.time()
        for k in range(0, len(todo), args.batch_size):
            batch = todo[k:k + args.batch_size]
            built = [judge.build(POINTWISE_TEMPLATE, it["question"], {"response": it[f"response_{side}"]})
                     for it, side in batch]
            enc = judge.tok([c for c, _, _ in built], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(0)
            with torch.inference_mode():
                gen = judge.model.generate(**enc, max_new_tokens=args.think_budget, do_sample=False,
                                           temperature=None, top_p=None, top_k=None,
                                           eos_token_id=[end_id, judge.tok.eos_token_id],
                                           pad_token_id=judge.tok.pad_token_id)
            new = gen[:, enc.input_ids.shape[1]:]
            finals, traces = [], []
            for (chat, cut, _), row in zip(built, new):
                drop = {judge.tok.pad_token_id, judge.tok.eos_token_id}
                ids = [t for t in row.tolist() if t not in drop]  # keeps the leading <think>
                closed = end_id in ids
                trace = judge.tok.decode(ids[:ids.index(end_id)] if closed else ids)
                traces.append((trace, len(ids), not closed))
                finals.append(chat + trace + ("" if trace.endswith("\n") else "\n") + THINK_END + "\n\n")
            enc2 = judge.tok(finals, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
            pos = (enc2.attention_mask.cumsum(-1) - 1).clamp_min(0)
            with torch.inference_mode():
                logits = judge.model(**enc2, position_ids=pos, use_cache=False,
                                     logits_to_keep=1).logits[:, -1].float()
            probs = torch.softmax(logits, -1)
            with items_path.open("a", encoding="utf-8") as f:
                for j, ((it, side), (trace, n_tok, capped)) in enumerate(zip(batch, traces)):
                    f.write(json.dumps({
                        "pair_id": it["pair_id"], "side": side, "task": it["task"], "source": it["source"],
                        "cluster": cluster_key(it), "margin": float(logits[j, pos_id] - logits[j, neg_id]),
                        "label_mass": float(probs[j, [pos_id, neg_id]].sum()), "think_tokens": n_tok,
                        "capped": capped, "trunc_tokens": built[j][1]["response"], "trace": trace}) + "\n")
            print(f"[{k + len(batch)}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
        meta["seconds"] = round(time.time() - t0, 1)

    with items_path.open(encoding="utf-8") as f:
        rows = {(r["pair_id"], r["side"]): r for r in map(json.loads, f)}
    recs = []
    for it in items:
        a, b = rows[it["pair_id"], "A"], rows[it["pair_id"], "B"]
        p = pick(a["margin"] - b["margin"])
        recs.append({"pair_id": it["pair_id"], "task": it["task"], "cluster": cluster_key(it),
                     "correct": p == it["label"][0], "pick": p,  # gold read only here, after the model call
                     "capped": a["capped"] or b["capped"], "mass": (a["label_mass"] + b["label_mass"]) / 2,
                     "think_tokens": (a["think_tokens"] + b["think_tokens"]) / 2})
    clusters = [r["cluster"] for r in recs]
    summary = {name: cluster_bootstrap([f(r) for r in recs], clusters, args.n_boot, args.seed)
               for name, f in [("accuracy", lambda r: r["correct"]), ("capped_rate", lambda r: r["capped"]),
                               ("label_mass_mean", lambda r: r["mass"]),
                               ("think_tokens_mean", lambda r: r["think_tokens"])]}
    off_dir = Path(args.compare_run) if args.compare_run else (
        Path(args.out_dir) / f"jb_{model_tag}_bfloat16_{task_tag}_both_n{args.limit or 'all'}")
    if (off_dir / "items.jsonl").exists():
        with (off_dir / "items.jsonl").open(encoding="utf-8") as f:
            off = {r["pair_id"]: r for r in map(json.loads, f) if r["mode"] == "pointwise"}
        if all(r["pair_id"] in off for r in recs):
            summary["accuracy_thinking_off"] = cluster_bootstrap(
                [off[r["pair_id"]]["correct"] for r in recs], clusters, args.n_boot, args.seed)
            summary["change_vs_thinking_off"] = cluster_bootstrap(
                [r["correct"] - off[r["pair_id"]]["correct"] for r in recs], clusters, args.n_boot, args.seed)
            meta["compare_run"] = str(off_dir)
    (run_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
    lines = [f"# Thinking-on pointwise judge: {run_dir.name}", "",
             f"- model `{args.model_path}`, tasks {args.tasks}, {len(items)} pairs, greedy, "
             f"think budget {args.think_budget}; exploratory (team protocol is thinking off)", "",
             "| metric | mean | 95% CI |", "|---|---|---|"]
    lines += [f"| {k} | {v['mean']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |" for k, v in summary.items()]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
