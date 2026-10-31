"""S9: score the injection set at BF16 and on 16 offset-shifted INT4 grids.

Grids: see quant_grids.py (INT4 g00..g15, same scale and worst-case error, only the offset
differs; optional INT8 control i8g00..i8g15 with the same construction).

Output: results/<run>/items.jsonl, one row per (grid, item) in the format s9_lottery_stats.py reads,
plus meta.json (label tokens, per-grid weight error). Resumable per grid.

Usage (GPU node only):
  python s9_score_grids.py --items data/s9_injection/items.jsonl --partitions development,locked
"""

import argparse
import json
import time
from pathlib import Path

from jb_eval import POINTWISE_TEMPLATE, Judge
from quant_grids import GROUP, N_GRIDS, apply_grid, target_linears

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "/home/019111677/models/Qwen3-4B"


def score_all(judge, chats, batch_size):
    """First-token CORRECT - INCORRECT margin and label mass for each chat (left-padded batches)."""
    torch = judge.torch
    tok = judge.tok
    pos_id, neg_id = judge.labels["pointwise"]["token_ids"]
    out = []
    for i in range(0, len(chats), batch_size):
        enc = tok(chats[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        ids, mask = enc.input_ids.to(0), enc.attention_mask.to(0)
        pos = (mask.cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            logits = judge.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                 use_cache=False, logits_to_keep=1).logits[:, -1].float()
        probs = torch.softmax(logits, -1)
        for k in range(logits.shape[0]):
            out.append((float(logits[k, pos_id] - logits[k, neg_id]), float(probs[k, [pos_id, neg_id]].sum())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default=str(HERE / "data" / "s9_injection" / "items.jsonl"))
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--partitions", default="development,locked", help="comma list, or 'all'")
    ap.add_argument("--limit-problems", type=int, default=0, help="smoke test: first N problems only")
    ap.add_argument("--grids", type=int, default=N_GRIDS, help="score INT4 grids g00..g(N-1) (plus bf16)")
    ap.add_argument("--int8-grids", type=int, default=0, help="INT8 control grids i8g00..i8g(N-1)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    with open(args.items, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.partitions != "all":
        keep = set(args.partitions.split(","))
        items = [it for it in items if it["partition"] in keep]
    if args.limit_problems:
        pids = list(dict.fromkeys(it["problem_id"] for it in items))[: args.limit_problems]
        items = [it for it in items if it["problem_id"] in set(pids)]
    if not items:
        raise SystemExit("no items selected")

    model_tag = Path(args.model_path).name.lower()
    run_name = args.run_name or f"s9_{model_tag}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path, meta_path = run_dir / "items.jsonl", run_dir / "meta.json"

    done = set()
    if rows_path.exists():
        with rows_path.open(encoding="utf-8") as f:
            done = {json.loads(line)["grid"] for line in f}
    grids = (["bf16"] + [f"g{g:02d}" for g in range(args.grids)]
             + [f"i8g{g:02d}" for g in range(args.int8_grids)])
    todo = [g for g in grids if g not in done]
    print(f"[data] {len(items)} items, {len({it['problem_id'] for it in items})} problems; "
          f"grids done {sorted(done)}, todo {todo}", flush=True)

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"run_name": run_name, "model_path": args.model_path, "items": args.items,
                 "partitions": args.partitions, "limit_problems": args.limit_problems,
                 "n_items": len(items), "group_size": GROUP, "n_grids": args.grids,
                 "n_int8_grids": args.int8_grids,
                 "batch_size": args.batch_size, "max_len": args.max_len})
    if todo:
        import torch
        import transformers

        judge = Judge(args.model_path, "bfloat16", args.max_len)
        judge.tok.padding_side = "left"
        meta["labels"] = {"pointwise": judge.calibrate("pointwise")}
        meta.update({"torch": torch.__version__, "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0)})
        chats = [judge.build(POINTWISE_TEMPLATE, it["question"], {"response": it["response"]})[0]
                 for it in items]
        linears = target_linears(judge.model)
        original = {n: m.weight.detach().clone() for n, m in linears}
        meta.setdefault("weight_err_over_scale", {})
        print(f"[model] {len(linears)} linear layers quantized per grid", flush=True)

        for grid in todo:
            t0 = time.time()
            err = apply_grid(linears, original, grid)
            if grid != "bf16":
                meta["weight_err_over_scale"][grid] = err
            scores = score_all(judge, chats, args.batch_size)
            with rows_path.open("a", encoding="utf-8") as f:
                for it, (margin, mass) in zip(items, scores):
                    f.write(json.dumps({"grid": grid, "problem_id": it["problem_id"],
                                        "answer_id": it["answer_id"], "note_id": it["note_id"],
                                        "heldout": it["heldout"], "partition": it["partition"],
                                        "margin": margin, "label_mass": mass}) + "\n")
            acc = sum(m < 0 for (m, _), it in zip(scores, items) if it["note_id"] is None)
            n0 = sum(it["note_id"] is None for it in items)
            print(f"[grid {grid}] {len(items)} items in {time.time() - t0:.0f}s; "
                  f"no-note wrong answers rejected {acc}/{n0}; "
                  f"max err/s {meta['weight_err_over_scale'].get(grid, 0):.3f}", flush=True)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {rows_path}", flush=True)


if __name__ == "__main__":
    main()
