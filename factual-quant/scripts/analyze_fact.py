"""Summarize fact_eval.py results: TruthfulQA (noref / ref) and RewardBench 2 Factuality.

Bootstrap CIs resample questions (TruthfulQA) or prompts (RB2), not individual answers.
Writes results/fact_summary.json and prints a table.
"""
import json, os
import numpy as np
import pandas as pd

R = os.path.expanduser("~/qjudge/results")
FAMS = ["Qwen3-4B", "Qwen3-8B", "Qwen2.5-7B-Instruct", "Phi-4-mini-instruct"]
QUANTS = ["", "-int8", "-int4", "-rtn4", "-gptq4", "-w4a16", "-llmc-gptq-w4a16", "-llmc-awq-w4a16", "-hqq4", "-mnr4", "-gptaq4"]  # rtn4/gptq4: simquant; w4a16: Red Hat GPTQ checkpoint (Qwen3-4B only)
# H100 runs (run_fact_h100.sbatch): compared with BF16 on the same GPU type
H100 = [(f + "-h100", ["", "-int8", "-fp8", "-w8a8"]) for f in ("Qwen3-8B", "Phi-4-mini-instruct")]
B = 2000


def auroc(score, label):
    r = pd.Series(score).rank().to_numpy()
    n1 = label.sum(); n0 = len(label) - n1
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def tqa_stats(d):
    pos = d.kind.eq("correct").to_numpy(); acc = (d.m_yes > 0).to_numpy()
    tpr, far = acc[pos].mean(), acc[~pos].mean()
    return dict(bal_acc=(tpr + 1 - far) / 2, accept_correct=tpr, accept_incorrect=far, auroc=auroc(d.m_yes.to_numpy(), pos))


def boot(d, unit, seed=0):
    """Question-level bootstrap of balanced accuracy and false acceptance, from per-question counts."""
    d = d.assign(pos=d.kind.eq("correct"), acc=d.m_yes > 0)
    c = d.groupby(unit).apply(lambda g: pd.Series(dict(
        np_=g.pos.sum(), ap=(g.pos & g.acc).sum(), nn=(~g.pos).sum(), an=(~g.pos & g.acc).sum()))).to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(c), (B, len(c)))
    s = c[idx].sum(1)                                  # B x 4
    tpr, far = s[:, 1] / s[:, 0], s[:, 3] / s[:, 2]
    q = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    return dict(bal_acc=q((tpr + 1 - far) / 2), accept_incorrect=q(far))


def rb2_wins(d):
    g = d.groupby("pid")
    return (g.apply(lambda x: x[x.kind == "chosen"].m_yes.max() > x[x.kind == "rejected"].m_yes.max())).rename("win")


def main():
    out = {}
    print(f"{'judge':<26}{'cond':<7}{'bal acc':>18}{'AUROC':>8}{'acc corr':>9}{'acc wrong':>10}{'agree':>8}")
    groups = [(f, [f + q for q in QUANTS]) for f in FAMS]
    groups += [(b, [b.replace("-h100", q + "-h100") for q in qs]) for b, qs in H100]
    for fam, members in groups:
        if not os.path.exists(f"{R}/{fam}/truthfulqa.csv"):
            print(fam, "missing\n"); continue
        base_t = pd.read_csv(f"{R}/{fam}/truthfulqa.csv")
        base_r = rb2_wins(pd.read_csv(f"{R}/{fam}/rb2_factuality.csv"))
        for m in members:
            if not os.path.exists(f"{R}/{m}/truthfulqa.csv"):
                print(m, "missing"); continue
            t = pd.read_csv(f"{R}/{m}/truthfulqa.csv")
            meta = json.load(open(f"{R}/{m}/fact_meta.json"))
            res = dict(weights_GiB=meta["weights_GiB"], items_per_s=meta["prompts"] / meta["score_s"])
            for cond in ("noref", "ref"):
                d = t[t.cond == cond].reset_index(drop=True)
                b = base_t[base_t.cond == cond].reset_index(drop=True)
                s = tqa_stats(d)
                s["agree_bf16"] = float(((d.m_yes > 0) == (b.m_yes > 0)).mean())
                s["ci"] = boot(d, "qid")
                res[f"tqa_{cond}"] = s
                print(f"{m:<26}{cond:<7}{s['bal_acc']:>8.1%} [{s['ci']['bal_acc'][0]:.0%},{s['ci']['bal_acc'][1]:.0%}]"
                      f"{s['auroc']:>8.3f}{s['accept_correct']:>9.1%}{s['accept_incorrect']:>10.1%}{s['agree_bf16']:>8.1%}")
            rd = pd.read_csv(f"{R}/{m}/rb2_factuality.csv")
            w = rb2_wins(rd)
            wr = w.to_numpy(dtype=float)
            rng = np.random.default_rng(0)
            ci = np.quantile([rng.choice(wr, len(wr)).mean() for _ in range(B)], [.025, .975])
            acc = (rd.m_yes > 0)
            res["rb2"] = dict(best_of_4=float(wr.mean()), ci=ci.tolist(),
                              agree_bf16=float((w == base_r.loc[w.index]).mean()),
                              accept_chosen=float(acc[rd.kind == "chosen"].mean()),
                              accept_rejected=float(acc[rd.kind == "rejected"].mean()))
            s = res["rb2"]
            print(f"{m:<26}{'rb2':<7}{s['best_of_4']:>8.1%} [{ci[0]:.0%},{ci[1]:.0%}]{'':>8}"
                  f"{s['accept_chosen']:>9.1%}{s['accept_rejected']:>10.1%}{s['agree_bf16']:>8.1%}"
                  f"   {res['weights_GiB']:.1f} GiB, {res['items_per_s']:.1f} items/s")
            jbf, jbb = f"{R}/{m}/judgebench_knowledge.csv", f"{R}/{fam}/judgebench_knowledge.csv"
            if os.path.exists(jbf) and os.path.exists(jbb):
                j, jb0 = pd.read_csv(jbf), pd.read_csv(jbb)
                dc = j.debiased_correct.to_numpy(dtype=float)
                rng = np.random.default_rng(0)
                ci = np.quantile([rng.choice(dc, len(dc)).mean() for _ in range(B)], [.025, .975])
                agree = float((((j.m_goodA > 0) == (jb0.m_goodA > 0)).mean()
                               + ((j.m_goodB > 0) == (jb0.m_goodB > 0)).mean()) / 2)
                res["jb"] = dict(debiased=float(dc.mean()), ci=ci.tolist(),
                                 consistent=float(j.consistent_correct.mean()),
                                 picks_A_both=float(j.picks_A_both.mean()), picks_B_both=float(j.picks_B_both.mean()),
                                 agree_bf16=agree)
                s = res["jb"]
                print(f"{m:<26}{'jb':<7}{s['debiased']:>8.1%} [{ci[0]:.0%},{ci[1]:.0%}]  both-orders {s['consistent']:.1%}"
                      f"  A-both {s['picks_A_both']:.1%}  B-both {s['picks_B_both']:.1%}{s['agree_bf16']:>8.1%}")
            out[m] = res
        print()
    json.dump(out, open(f"{R}/fact_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
