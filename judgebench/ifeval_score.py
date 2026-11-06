"""IFEval step 3 (GPU): score every judge-set response under each precision / rounding condition.

Pointwise, same protocol as jb_eval (thinking off, no sampling, first-token
logit(CORRECT) - logit(INCORRECT), label tokens from jb_eval's calibration prompts on the BF16 judge),
with an instruction-following prompt. Labels (audit.jsonl) are never read here.

Conditions (--conditions): bf16, RTN grids (g00, i8g00, ... from quant_grids.py) and calibrated
rounding on a grid (gptq-g00, mnr-g00, see quant_methods.py). Every condition starts from the BF16
weights. Output: results/<run>/scores.jsonl rows {cond, item_id, margin, label_mass}; resumable per
condition. --with-notes also scores data/ifeval/notes.jsonl (add-on).

Usage (GPU node only):
  python ifeval_score.py --model-path ~/models/Qwen3-4B --limit-items 16    # smoke
  python ifeval_score.py --model-path ~/models/Qwen3-4B
"""

import argparse
import json
import time
from pathlib import Path

from quant_grids import apply_grid, parse_grid, target_linears

HERE = Path(__file__).resolve().parent
DEFAULT_CONDITIONS = "bf16,i8g00,g00,gptq-g00,mnr-g00"

IF_TEMPLATE = """You are grading whether a response follows the user's instructions.

Instructions given to the assistant:
{question}

Response:
{response}

Does the response follow every instruction above exactly? Answer with exactly one word: CORRECT or INCORRECT."""


def parse_condition(cond):
    """'gptq-g00' -> ('gptq', 'g00'); 'g00' -> ('rtn', 'g00'); 'bf16' -> ('rtn', 'bf16')."""
    method, _, grid = cond.rpartition("-")
    method = method or "rtn"
    if method not in ("rtn", "gptq", "mnr") or (method != "rtn" and grid == "bf16"):
        raise ValueError(f"bad condition {cond!r}")
    parse_grid(grid)
    return method, grid


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
    ap.add_argument("--tune-problems", type=int, default=32)
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
    run_dir = Path(args.out_dir) / (args.run_name or f"ifeval_{model_tag}" + ("_smoke" if args.limit_items else ""))
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
                 "template": IF_TEMPLATE, "build_meta": json.loads((d / "build_meta.json").read_text())["sha256"]})

    if todo:
        import torch
        import transformers
        from jb_eval import Judge
        from s9_score_grids import score_all

        judge = Judge(args.model_path, "bfloat16", args.max_len)
        judge.tok.padding_side = "left"
        meta["labels"] = {"pointwise": judge.calibrate("pointwise")}
        meta.update({"torch": torch.__version__, "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0)})
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
            method, grid = parse_condition(cond)
            apply_grid(linears, original, "bf16")
            if method == "rtn":
                meta["quant"][cond] = {"weight_err_over_scale": apply_grid(linears, original, grid)}
            else:
                if method not in ctxs:
                    from quant_methods import Calibrated
                    apply_grid(linears, original, "bf16")
                    ctxs[method] = Calibrated(judge, method, calib_samples=args.calib_samples,
                                              tune_problems=args.tune_problems)
                meta["quant"][cond] = dict(ctxs[method].meta, **ctxs[method].quantize(judge, grid))
            scores = score_all(judge, chats, args.batch_size)
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
