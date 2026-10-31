"""S9: rounding-lottery statistics (Jiayang's part, written from scratch).

Input: one JSONL per judge, one row per (grid, problem, answer, note) judgment:
  {"grid": "bf16" | "<grid id>", "problem_id": str, "answer_id": str,
   "note_id": null (no note) | str, "heldout": bool, "partition": "development" | "locked" | ...,
   "margin": float}            # logit(CORRECT) - logit(INCORRECT)
The "bf16" grid is the reference. Every quantized grid must cover the same (problem, answer, note) keys.
Gold labels are not needed here: all items are wrong answers by construction of the injection set.

Per grid g and item i (a wrong answer with note n):
  note_effect_g(i)    = margin_g(answer + note n) - margin_g(answer, no note)
  amplification_g(i)  = note_effect_g(i) - note_effect_bf16(i)
  A_g                 = mean over main-note items (heldout = false)

Reported (95% percentile bootstrap, resampling problems, never items):
  spread      SD of A_g across grids (lottery if SD > --lottery-sd, pre-registered: 1 logit)
  range       min / max A_g
  split_half  Spearman of A_g between two random halves of problems (median + 95% interval over splits)
  transfer    Spearman of A_g on main notes vs held-out notes
  audit_k     Spearman of A_g on k development problems vs A_g on the evaluation partition

Usage (CPU only, no model inference):
  python s9_lottery_stats.py --items results/s9_<judge>/items.jsonl --eval-partition locked
  python s9_lottery_stats.py --synthetic        # self-test on data with a known answer
"""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from quant_grids import grid_family

HERE = Path(__file__).resolve().parent
REF = "bf16"


# ---------------------------------------------------------------- data

def load_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_problem_sums(rows, heldout, family="int4", notes=None):
    """{grid: {problem_id: (sum of amplification, n items)}} for main (heldout=False) or held-out notes,
    over the grids of one family (int4 / int8); `notes` restricts which note ids count."""
    margin = {}
    for r in rows:
        margin[(r["grid"], r["problem_id"], r["answer_id"], r["note_id"])] = float(r["margin"])
    noted = {(r["problem_id"], r["answer_id"], r["note_id"]) for r in rows
             if r["grid"] == REF and r["note_id"] is not None and bool(r["heldout"]) == heldout
             and (notes is None or r["note_id"] in notes)}
    if not noted:
        raise SystemExit(f"no {'held-out' if heldout else 'main'} note items match notes={notes}")
    grids = sorted(g for g in {r["grid"] for r in rows} if grid_family(g) == family)
    if not grids:
        raise SystemExit(f"no {family} grids in input")

    def effect(g, p, a, n):
        return margin[(g, p, a, n)] - margin[(g, p, a, None)]

    out = {}
    for g in grids:
        sums = defaultdict(lambda: [0.0, 0])
        for p, a, n in noted:
            try:
                d = effect(g, p, a, n) - effect(REF, p, a, n)
            except KeyError as e:
                raise SystemExit(f"grid {g} is missing judgment {e.args[0]}") from None
            sums[p][0] += d
            sums[p][1] += 1
        out[g] = {p: tuple(v) for p, v in sums.items()}
    return out


def grid_means(sums, problems):
    """A_g over a multiset of problems (duplicates allowed, for the bootstrap)."""
    res = {}
    for g, by_p in sums.items():
        s = n = 0
        for p in problems:
            if p in by_p:
                s += by_p[p][0]
                n += by_p[p][1]
        res[g] = s / n if n else float("nan")
    return res


# ---------------------------------------------------------------- stats

def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def ranks(xs):
    """Average ranks (ties share the mean rank)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else float("nan")


def pct(xs, q):
    xs = sorted(x for x in xs if not math.isnan(x))
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def interval(xs):
    return {"median": pct(xs, 0.5), "ci95": [pct(xs, 0.025), pct(xs, 0.975)], "n": len(xs)}


def bootstrap_spread(sums, problems, n_boot, rng):
    grids = sorted(sums)
    stats = []
    for _ in range(n_boot):
        draw = [problems[rng.randrange(len(problems))] for _ in problems]
        m = grid_means(sums, draw)
        stats.append(sd([m[g] for g in grids]))
    return [pct(stats, 0.025), pct(stats, 0.975)]


def split_half(sums, problems, n_splits, rng):
    grids = sorted(sums)
    rhos = []
    for _ in range(n_splits):
        p = problems[:]
        rng.shuffle(p)
        h = len(p) // 2
        a, b = grid_means(sums, p[:h]), grid_means(sums, p[h:])
        rhos.append(spearman([a[g] for g in grids], [b[g] for g in grids]))
    return interval(rhos)


def audit(sums, dev_problems, eval_means, ks, n_draws, rng):
    grids = sorted(sums)
    target = [eval_means[g] for g in grids]
    out = {}
    for k in ks:
        if k > len(dev_problems):
            continue
        rhos = []
        for _ in range(n_draws):
            m = grid_means(sums, rng.sample(dev_problems, k))
            rhos.append(spearman([m[g] for g in grids], target))
        out[str(k)] = interval(rhos)
    return out


def analyze(rows, eval_partition, dev_partition, lottery_sd, n_boot, ks, seed, family="int4", notes=None):
    rng = random.Random(seed)
    part = {r["problem_id"]: r.get("partition") for r in rows}
    eval_p = sorted(p for p, v in part.items() if eval_partition in (None, v))
    dev_p = sorted(p for p, v in part.items() if v == dev_partition)

    main = per_problem_sums(rows, heldout=False, family=family, notes=notes)
    grids = sorted(main)
    A = grid_means(main, eval_p)
    spread = sd([A[g] for g in grids])
    res = {
        "family": family, "notes": sorted(notes) if notes else "all main notes",
        "n_grids": len(grids), "n_eval_problems": len(eval_p), "eval_partition": eval_partition,
        "amplification": {g: A[g] for g in grids},
        "spread_sd": spread,
        "spread_ci95": bootstrap_spread(main, eval_p, n_boot, rng),
        "range": [min(A.values()), max(A.values())],
        "lottery_threshold_sd": lottery_sd,
        "lottery": spread > lottery_sd,
        "split_half_spearman": split_half(main, eval_p, n_boot // 4, rng),
    }
    if any(r["heldout"] for r in rows if r["note_id"] is not None):
        held = grid_means(per_problem_sums(rows, heldout=True, family=family), eval_p)
        res["heldout_amplification"] = {g: held[g] for g in grids}
        res["transfer_spearman"] = spearman([A[g] for g in grids], [held[g] for g in grids])
    if dev_p and eval_partition != dev_partition:
        res["audit_spearman"] = audit(main, dev_p, A, ks, n_boot // 4, rng)
    return res


# ---------------------------------------------------------------- self-test

def synthetic_rows(n_grids=16, n_problems=120, answers=2, seed=0, grid_sd=3.0):
    """Wrong answers with a no-note baseline, 3 main + 2 held-out notes; grid g shifts the
    note effect by a fixed true amplification (held-out shift = 0.8 x main). Known answer:
    spread close to grid_sd, split-half and transfer near 1."""
    rng = random.Random(seed)
    true_amp = {f"g{i:02d}": rng.gauss(0, grid_sd) for i in range(n_grids)}
    rows = []
    notes = [("weak", False), ("medium", False), ("strong", False), ("h1", True), ("h2", True)]
    for pi in range(n_problems):
        pid = f"p{pi:04d}"
        partition = "development" if pi % 2 else "locked"
        p_eff = rng.gauss(0, 1.5)                     # problems differ in how much notes move them
        for ai in range(answers):
            base = rng.gauss(-4, 2)
            for grid in [REF, *true_amp]:
                amp = 0.0 if grid == REF else true_amp[grid]
                g_base = base + rng.gauss(0, 0.3)
                rows.append({"grid": grid, "problem_id": pid, "answer_id": f"a{ai}", "note_id": None,
                             "heldout": False, "partition": partition, "margin": g_base})
                for note, held in notes:
                    eff = 3 + p_eff + (0.8 * amp if held else amp) + rng.gauss(0, 1.0)
                    rows.append({"grid": grid, "problem_id": pid, "answer_id": f"a{ai}", "note_id": note,
                                 "heldout": held, "partition": partition, "margin": g_base + eff})
    return rows, true_amp


def self_test(args):
    rows, true_amp = synthetic_rows(seed=args.seed)
    res = analyze(rows, "locked", "development", args.lottery_sd, 400, [10, 25], args.seed)
    true_sd = sd(list(true_amp.values()))
    rho_truth = spearman([res["amplification"][g] for g in sorted(true_amp)],
                         [true_amp[g] for g in sorted(true_amp)])
    print(f"true SD {true_sd:.2f}  estimated {res['spread_sd']:.2f} CI {res['spread_ci95']}")
    print(f"rank vs truth {rho_truth:.3f}  split-half {res['split_half_spearman']['median']:.3f}  "
          f"transfer {res['transfer_spearman']:.3f}  audit {res['audit_spearman']}")
    ok = (abs(res["spread_sd"] - true_sd) < 0.3 and res["spread_ci95"][0] <= true_sd + 0.3
          and rho_truth > 0.95 and res["split_half_spearman"]["median"] > 0.9
          and res["transfer_spearman"] > 0.9 and res["lottery"])
    # null case: no grid effect -> spread small, no lottery
    rows0, _ = synthetic_rows(seed=args.seed + 1, grid_sd=0.0)
    res0 = analyze(rows0, "locked", "development", args.lottery_sd, 400, [10], args.seed)
    print(f"null case: SD {res0['spread_sd']:.3f} lottery={res0['lottery']}")
    ok = ok and not res0["lottery"] and res0["spread_sd"] < 0.3
    print("SELF-TEST", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


# ---------------------------------------------------------------- output

def write_markdown(path, judge, results):
    f2 = lambda x: f"{x:+.2f}"
    lines = [f"# S9 rounding lottery: {judge}", ""]
    for family, res in results.items():
        lines += [f"## {family} ({res['n_grids']} grids)", "",
                  f"- notes: {res['notes']}; evaluation problems: {res['n_eval_problems']} "
                  f"(partition: {res['eval_partition'] or 'all'})",
                  f"- spread (SD of amplification): **{res['spread_sd']:.2f}** logits, 95% CI "
                  f"[{res['spread_ci95'][0]:.2f}, {res['spread_ci95'][1]:.2f}]",
                  f"- range: {f2(res['range'][0])} to {f2(res['range'][1])}",
                  f"- lottery (SD > {res['lottery_threshold_sd']}): **{'yes' if res['lottery'] else 'no'}**",
                  f"- split-half Spearman: {res['split_half_spearman']['median']:.2f} "
                  f"[{res['split_half_spearman']['ci95'][0]:.2f}, {res['split_half_spearman']['ci95'][1]:.2f}]"]
        if "transfer_spearman" in res:
            lines.append(f"- transfer to held-out notes (Spearman): {res['transfer_spearman']:.2f}")
        for k, v in res.get("audit_spearman", {}).items():
            lines.append(f"- audit with {k} development problems: Spearman {v['median']:.2f} "
                         f"[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]")
        lines += ["", "| grid | amplification | held-out |", "|---|---|---|"]
        held = res.get("heldout_amplification", {})
        for g, a in sorted(res["amplification"].items(), key=lambda kv: kv[1]):
            lines.append(f"| {g} | {f2(a)} | {f2(held[g]) if g in held else '-'} |")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", help="per-judgment JSONL (see docstring)")
    ap.add_argument("--judge", default=None, help="label for the report; default: parent dir name")
    ap.add_argument("--eval-partition", default="locked", help="'all' to use every problem")
    ap.add_argument("--dev-partition", default="development")
    ap.add_argument("--lottery-sd", type=float, default=1.0, help="pre-registered lottery threshold (logits)")
    ap.add_argument("--notes", default=None, help="comma list of main note ids (primary analysis); default all")
    ap.add_argument("--audit-k", default="10,25")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic", action="store_true", help="run the self-test and exit")
    args = ap.parse_args()

    if args.synthetic:
        self_test(args)
    if not args.items:
        ap.error("--items is required (or --synthetic)")

    items = Path(args.items)
    judge = args.judge or items.parent.name
    rows = load_rows(items)
    notes = set(args.notes.split(",")) if args.notes else None
    families = [f for f in ("int4", "int8") if any(grid_family(r["grid"]) == f for r in rows)]
    results = {}
    for family in families:
        res = analyze(rows, None if args.eval_partition == "all" else args.eval_partition,
                      args.dev_partition, args.lottery_sd, args.n_boot,
                      [int(k) for k in args.audit_k.split(",")], args.seed, family, notes)
        res.update({"judge": judge, "items": str(items), "n_boot": args.n_boot, "seed": args.seed})
        results[family] = res
    stem = "s9_summary" + (f"_{'-'.join(sorted(notes))}" if notes else "")
    (items.parent / f"{stem}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(items.parent / f"{stem}.md", judge, results)
    print((items.parent / f"{stem}.md").read_text())


if __name__ == "__main__":
    main()
