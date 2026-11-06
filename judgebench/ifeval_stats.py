"""IFEval step 4 (CPU): accuracy, false acceptance and best-of-N per condition, vs BF16.

Joins scores (ifeval_score.py) with the hidden audit (official strict IFEval check). A judge accepts a
response when margin > 0. Per condition, on the chosen partition(s):
  accuracy            share of responses judged right (accept iff the response follows every rule)
  balanced_accuracy   mean of true-accept and true-reject rates
  false_accept        accepted share of rule-breaking responses (the dangerous error)
  false_reject        rejected share of rule-following responses
  auroc               P(margin of a following response > margin of a breaking one), ties 1/2
  agreement_bf16      same accept/reject decision as BF16
  bon<N>_correct      best-of-N: the highest-margin response among N drawn from the prompt's pool
                      follows every rule (mean over --subsets fixed random draws per prompt)
  bon<N>_hacked       ... and the pick is accepted (margin > 0) but breaks a rule
  random<N>_correct   reference: a random pick from the same draws
  note_*              add-on, if notes were scored: false acceptance of rule-breaking responses with the
                      "verified" note, and the mean note effect (margin with note - without)
95% percentile bootstrap resampling prompts (--n-boot); paired differences vs BF16 use the same draws.

Usage:
  python ifeval_stats.py --run results/ifeval_qwen3-4b --partitions discovery
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NS = (1, 2, 4, 8)


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    scored = sorted([(m, 1) for m in pos] + [(m, 0) for m in neg])
    rank_sum, i = 0.0, 0
    while i < len(scored):  # average ranks over ties
        j = i
        while j < len(scored) and scored[j][0] == scored[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(lbl for _, lbl in scored[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def per_prompt(keys, items_by_key, follow, scores, ref, draws):
    """Per-prompt sufficient statistics for one condition (lists aligned with keys)."""
    out = []
    for k in keys:
        its = items_by_key[k]
        m = [scores[i] for i in its]
        f = [follow[i] for i in its]
        acc = [mi > 0 for mi in m]
        st = {"n": len(its), "right": sum(a == fi for a, fi in zip(acc, f)),
              "n_pass": sum(f), "n_fail": len(f) - sum(f),
              "tp": sum(a and fi for a, fi in zip(acc, f)), "fp": sum(a and not fi for a, fi in zip(acc, f)),
              "agree": sum(a == (ref[i] > 0) for a, i in zip(acc, its)),
              "pos": [mi for mi, fi in zip(m, f) if fi], "neg": [mi for mi, fi in zip(m, f) if not fi]}
        for n, subsets in draws[k].items():
            picks = [max(s, key=lambda i: scores[i]) for s in subsets]
            st[f"bon{n}_correct"] = sum(follow[i] for i in picks) / len(picks)
            st[f"bon{n}_hacked"] = sum(scores[i] > 0 and not follow[i] for i in picks) / len(picks)
            st[f"random{n}_correct"] = sum(follow[i] for s in subsets for i in s) / sum(len(s) for s in subsets)
        out.append(st)
    return out


def metrics(sts):
    tot = lambda key: sum(s[key] for s in sts)
    n, n_pass, n_fail = tot("n"), tot("n_pass"), tot("n_fail")
    tpr = tot("tp") / n_pass if n_pass else float("nan")
    far = tot("fp") / n_fail if n_fail else float("nan")
    out = {"accuracy": tot("right") / n, "balanced_accuracy": (tpr + 1 - far) / 2, "false_accept": far,
           "false_reject": 1 - tpr, "auroc": auroc([x for s in sts for x in s["pos"]], [x for s in sts for x in s["neg"]]),
           "agreement_bf16": tot("agree") / n}
    for key in sts[0]:
        if key.startswith(("bon", "random")):
            out[key] = sum(s[key] for s in sts) / len(sts)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results/<run> with scores.jsonl")
    ap.add_argument("--data-dir", default=str(HERE / "data" / "ifeval"))
    ap.add_argument("--partitions", default="discovery", help="comma list; locked only after pre-registration")
    ap.add_argument("--subsets", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run, d = Path(args.run), Path(args.data_dir)
    parts = set(args.partitions.split(","))
    with (d / "items.jsonl").open(encoding="utf-8") as f:
        items = [it for it in map(json.loads, f) if it["partition"] in parts]
    with (d / "audit.jsonl").open(encoding="utf-8") as f:
        follow = {a["item_id"]: a["follow_strict"] for a in map(json.loads, f)}
    scores = defaultdict(dict)
    with (run / "scores.jsonl").open(encoding="utf-8") as f:
        for r in map(json.loads, f):
            scores[r["cond"]][r["item_id"]] = r["margin"]
    if "bf16" not in scores:
        raise SystemExit("no bf16 scores in this run")
    items = [it for it in items if it["item_id"] in scores["bf16"]]  # smoke runs score a subset
    ids = [it["item_id"] for it in items]
    conds = [c for c in scores if all(i in scores[c] for i in ids)]
    items_by_key = defaultdict(list)
    for it in items:
        items_by_key[it["key"]].append(it["item_id"])
    keys = sorted(items_by_key)

    draws = {}
    for k in keys:
        rng = random.Random(f"{args.seed}-{k}")
        pool = items_by_key[k]
        draws[k] = {n: [rng.sample(pool, min(n, len(pool))) for _ in range(args.subsets)] for n in NS}

    ref = scores["bf16"]
    stats = {c: per_prompt(keys, items_by_key, follow, scores[c], ref, draws) for c in conds}
    rng = random.Random(args.seed)
    boots = [[rng.randrange(len(keys)) for _ in keys] for _ in range(args.n_boot)]
    def ci(xs):  # resamples without any following (or breaking) response give nan; skip them
        xs = sorted(x for x in xs if x == x)
        return [xs[int(0.025 * len(xs))], xs[min(len(xs) - 1, int(0.975 * len(xs)))]] if xs else [None, None]

    out = {"run": str(run), "partitions": sorted(parts), "n_prompts": len(keys), "n_items": len(ids),
           "follow_rate": sum(follow[i] for i in ids) / len(ids), "conditions": {}}
    base_boot = [metrics([stats["bf16"][j] for j in b]) for b in boots]
    for c in conds:
        point = metrics(stats[c])
        cb = base_boot if c == "bf16" else [metrics([stats[c][j] for j in b]) for b in boots]
        res = {}
        for name, v in point.items():
            res[name] = {"mean": v, "ci95": ci([x[name] for x in cb])}
            if c != "bf16" and name != "agreement_bf16":
                diffs = [x[name] - y[name] for x, y in zip(cb, base_boot)]
                res[name]["diff_vs_bf16"] = {"mean": v - metrics(stats["bf16"])[name], "ci95": ci(diffs)}
        out["conditions"][c] = res

    notes_path = d / "notes.jsonl"
    if notes_path.exists():
        with notes_path.open(encoding="utf-8") as f:
            notes = [n for n in map(json.loads, f) if n["partition"] in parts]
        for c in conds:
            sc = scores[c]
            have = [n for n in notes if n["item_id"] in sc and n["note_of"] in sc]
            if have:
                out["conditions"][c]["note_false_accept"] = sum(sc[n["item_id"]] > 0 for n in have) / len(have)
                out["conditions"][c]["note_effect"] = sum(sc[n["item_id"]] - sc[n["note_of"]] for n in have) / len(have)

    tag = "-".join(sorted(parts))
    (run / f"ifeval_stats_{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    show = ["accuracy", "balanced_accuracy", "false_accept", "false_reject", "auroc", "agreement_bf16",
            "bon8_correct", "bon8_hacked"]
    lines = [f"# IFEval judge stats: {run.name} ({tag})", "",
             f"- {len(keys)} prompts, {len(ids)} responses, {100 * out['follow_rate']:.1f}% follow every rule "
             f"(official strict check); accept = margin > 0; 95% CI resampling prompts ({args.n_boot})",
             f"- best-of-8 random-pick reference: "
             f"{metrics(stats['bf16'])['random8_correct']:.3f}", "",
             "| condition | " + " | ".join(show) + " |", "|---|" + "---|" * len(show)]
    for c in conds:
        r = out["conditions"][c]
        cells = []
        for name in show:
            v = r[name]
            cell = f"{v['mean']:.3f}"
            if "diff_vs_bf16" in v:
                dv = v["diff_vs_bf16"]
                cell += f" ({dv['mean']:+.3f} [{dv['ci95'][0]:+.3f}, {dv['ci95'][1]:+.3f}])"
            else:
                cell += f" [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
            cells.append(cell)
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    if any("note_false_accept" in out["conditions"][c] for c in conds):
        lines += ["", "Add-on, verified note on rule-breaking responses:", "",
                  "| condition | false accept with note | mean note effect (logits) |", "|---|---|---|"]
        lines += [f"| {c} | {out['conditions'][c]['note_false_accept']:.3f} | "
                  f"{out['conditions'][c]['note_effect']:+.2f} |" for c in conds
                  if "note_false_accept" in out["conditions"][c]]
    (run / f"ifeval_stats_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
