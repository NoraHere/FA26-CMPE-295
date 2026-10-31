"""Compare jb_eval runs on shifted weight grids against the BF16 run of the same judge.

Per grid: pointwise accuracy, pairwise debiased accuracy, agreement with BF16 picks, and the paired
accuracy change vs BF16 (95% bootstrap CI, clustered like jb_eval). CPU only.

Usage:
  python jb_compare.py --model-tag qwen3-4b
"""

import argparse
import json
import statistics
from pathlib import Path

from jb_eval import cluster_bootstrap
from quant_grids import grid_family

HERE = Path(__file__).resolve().parent
PICK = {"pointwise": "pick", "pairwise": "pick_debiased"}
CORRECT = {"pointwise": "correct", "pairwise": "correct_debiased"}


def load(run_dir):
    with (run_dir / "items.jsonl").open(encoding="utf-8") as f:
        return {(r["pair_id"], r["mode"]): r for r in map(json.loads, f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-tag", default="qwen3-4b")
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res = Path(args.results)
    prefix = f"jb_{args.model_tag}_bfloat16"
    ref = load(res / f"{prefix}_both_nall")
    runs = {}
    for d in sorted(res.glob(f"{prefix}_*_both_nall")):
        grid = d.name[len(prefix) + 1: -len("_both_nall")]
        recs = load(d)
        if set(recs) == set(ref):
            runs[grid] = recs
        else:
            print(f"[skip] {d.name}: {len(recs)}/{len(ref)} records")

    out = {"model_tag": args.model_tag, "n_pairs": len(ref) // 2, "grids": {}}
    lines = [f"# JudgeBench grid comparison: {args.model_tag}", "",
             f"- {len(ref) // 2} pairs; reference: `{prefix}_both_nall`; "
             f"paired 95% bootstrap CI ({args.n_boot} resamples, clustered as in jb_eval)", ""]
    for mode in ("pointwise", "pairwise"):
        keys = sorted(k for k in ref if k[1] == mode)
        clusters = [ref[k]["cluster"] for k in keys]
        base = cluster_bootstrap([ref[k][CORRECT[mode]] for k in keys], clusters, args.n_boot, args.seed)
        lines += [f"## {mode} (BF16 accuracy {base['mean']:.3f} "
                  f"[{base['ci95'][0]:.3f}, {base['ci95'][1]:.3f}])", "",
                  "| grid | accuracy | change vs BF16 | 95% CI | agreement with BF16 |", "|---|---|---|---|---|"]
        accs = {}
        for grid, recs in runs.items():
            acc = sum(recs[k][CORRECT[mode]] for k in keys) / len(keys)
            diff = cluster_bootstrap([recs[k][CORRECT[mode]] - ref[k][CORRECT[mode]] for k in keys],
                                     clusters, args.n_boot, args.seed)
            agree = sum(recs[k][PICK[mode]] == ref[k][PICK[mode]] for k in keys) / len(keys)
            accs.setdefault(grid_family(grid), []).append(acc)
            out["grids"].setdefault(grid, {})[mode] = {"accuracy": acc, "diff_vs_bf16": diff,
                                                       "agreement_with_bf16": agree}
            lines.append(f"| {grid} | {acc:.3f} | {diff['mean']:+.3f} | "
                         f"[{diff['ci95'][0]:+.3f}, {diff['ci95'][1]:+.3f}] | {agree:.3f} |")
        for fam, xs in accs.items():
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            lines.append(f"\n- {fam}: {len(xs)} grids, accuracy mean {statistics.mean(xs):.3f}, "
                         f"SD {sd:.3f}, range {min(xs):.3f} to {max(xs):.3f}")
        lines.append("")
    stem = res / f"jb_compare_{args.model_tag}"
    stem.with_suffix(".json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
