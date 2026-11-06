"""Calibrated rounding on the shifted grids of quant_grids.py: GPTQ and margin-neutral rounding (MNR).

Both keep grid g exactly (per-group scale and offset from the BF16 weights) and only choose which
grid level each weight lands on, so memory and worst-case range match plain RTN on the same grid.

gptq: Frantar et al. (2023). Column-by-column rounding with error feedback through the inverse
      Hessian H = sum x x^T of each layer's inputs on calibration text (wikitext-2 train, never a
      test set). Decoder blocks are processed in order, each fed by the already-quantized blocks.
mnr:  margin-neutral rounding (the team's S10 idea, own implementation). Start from gptq, then flip
      a few weights to their other neighboring level so the change in the injected-note effect,
      predicted to first order with the BF16 judge's gradient, cancels the measured amplification
      (note effect of the quantized judge minus BF16) on a tuning set. Flips are taken in order of
      smallest layer-output error increase per unit of predicted correction; a few secant rounds
      re-measure and adjust the number of flips. Tuning set: S9 development problems, main notes
      only (held-out notes and the locked partition are never used).

Usage (from jb_eval.py / s9_score_grids.py, model on GPU holding BF16 weights):
  ctx = Calibrated(judge, "mnr", calib_samples=128)
  stats = ctx.quantize(judge, "g05")
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

from quant_grids import GROUP, grid_params, parse_grid, target_linears

HERE = Path(__file__).resolve().parent
DEFAULT_CALIB = "/home/019111677/datasets/wikitext/wikitext-2-raw-v1/train.jsonl"
DEFAULT_TUNE = str(HERE / "data" / "s9_injection" / "items.jsonl")
MAIN_NOTES = ("weak", "medium", "strong")


# ---------------------------------------------------------------- data

def load_calibration(tok, n_samples, seqlen, seed, path=DEFAULT_CALIB):
    """n_samples random windows of seqlen tokens from wikitext-2 train, as a (n, seqlen) tensor."""
    import torch
    with open(path, encoding="utf-8") as f:
        text = "".join(json.loads(line)["text"] for line in f)
    ids = tok(text[:4_000_000], add_special_tokens=False, return_tensors="pt").input_ids[0]
    rng = random.Random(seed)
    starts = [rng.randrange(len(ids) - seqlen) for _ in range(n_samples)]
    return torch.stack([ids[s:s + seqlen] for s in starts])


class Tuning:
    """S9 development items (no note + main notes) and the judge's mean note effect on them."""

    def __init__(self, judge, path, n_problems, batch_size):
        from jb_eval import POINTWISE_TEMPLATE
        with open(path, encoding="utf-8") as f:
            rows = [r for r in map(json.loads, f) if r["partition"] == "development" and not r["heldout"]
                    and (r["note_id"] is None or r["note_id"] in MAIN_NOTES)]
        pids = set(list(dict.fromkeys(r["problem_id"] for r in rows))[:n_problems])
        rows = [r for r in rows if r["problem_id"] in pids]
        base = {(r["problem_id"], r["answer_id"]): i for i, r in enumerate(rows) if r["note_id"] is None}
        self.pairs = [(i, base[r["problem_id"], r["answer_id"]]) for i, r in enumerate(rows) if r["note_id"]]
        self.chats = [judge.build(POINTWISE_TEMPLATE, r["question"], {"response": r["response"]})[0]
                      for r in rows]
        self.n_problems = len(pids)
        self.batch_size = batch_size

    def note_effect(self, judge):
        from s9_score_grids import score_all
        side, judge.tok.padding_side = judge.tok.padding_side, "left"
        try:
            m = [x for x, _ in score_all(judge, self.chats, self.batch_size)]
        finally:
            judge.tok.padding_side = side
        return sum(m[i] - m[b] for i, b in self.pairs) / len(self.pairs)

    def coefficients(self):
        """d(note effect)/d(margin of each chat)."""
        coef = defaultdict(float)
        for i, b in self.pairs:
            coef[i] += 1 / len(self.pairs)
            coef[b] -= 1 / len(self.pairs)
        return coef


# ---------------------------------------------------------------- gradients (BF16 judge)

def note_effect_grads(judge, tuning, batch_size):
    """Gradient of the mean note effect w.r.t. every decoder Linear weight, kept on the CPU."""
    torch = judge.torch
    model, tok = judge.model, judge.tok
    linears = target_linears(model)
    pos_id, neg_id = judge.labels["pointwise"]["token_ids"]
    coef = tuning.coefficients()
    idx = sorted(coef)

    for p in model.parameters():
        p.requires_grad_(False)
    for _, m in linears:
        m.weight.requires_grad_(True)
    # train mode is needed for checkpointing; zero every dropout so the forward pass is unchanged
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "attention_dropout"):
            mod.attention_dropout = 0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    side, tok.padding_side = tok.padding_side, "left"
    try:
        for k in range(0, len(idx), batch_size):
            sel = idx[k:k + batch_size]
            enc = tok([tuning.chats[i] for i in sel], return_tensors="pt", padding=True,
                      add_special_tokens=False)
            ids, mask = enc.input_ids.to(0), enc.attention_mask.to(0)
            pos = (mask.cumsum(-1) - 1).clamp_min(0)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                           use_cache=False, logits_to_keep=1).logits[:, -1].float()
            w = torch.tensor([coef[i] for i in sel], device=logits.device)
            ((logits[:, pos_id] - logits[:, neg_id]) * w).sum().backward()
    finally:
        tok.padding_side = side
        model.gradient_checkpointing_disable()
        model.eval()
    grads = {}
    for n, m in linears:
        grads[n] = m.weight.grad.detach().to("cpu")
        m.weight.grad = None
        m.weight.requires_grad_(False)
    torch.cuda.empty_cache()
    return grads


# ---------------------------------------------------------------- GPTQ

def gptq_quantize(W, H, s, off, levels, percdamp=0.01, blocksize=GROUP):
    """GPTQ of a (out, in) float weight on fixed per-group grid (s, off); returns the dequantized weight."""
    import torch
    W = W.clone()
    H = H.clone()
    out_f, in_f = W.shape
    S, OFF = s[..., 0], off[..., 0]  # (out, in / GROUP)
    if not torch.isfinite(H).all():
        raise RuntimeError(f"GPTQ: Hessian has {int((~torch.isfinite(H)).sum())} non-finite entries")
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = percdamp * torch.mean(torch.diag(H))
    eye = torch.arange(in_f, device=W.device)
    Hinv = None
    # fp32 first (as before); float64 only if fp32 Cholesky fails, then cast back
    for dtype in (torch.float32, torch.float64):
        for attempt in range(5):
            Hd = H.to(dtype)
            Hd[eye, eye] += damp * 10 ** attempt
            L, info = torch.linalg.cholesky_ex(Hd)
            if info == 0:
                Hi, info = torch.linalg.cholesky_ex(torch.cholesky_inverse(L), upper=True)
                if info == 0 and torch.isfinite(Hi).all():
                    Hinv = Hi.float()
                    break
        if Hinv is not None:
            break
    if Hinv is None:
        raise RuntimeError("GPTQ: Hessian not positive definite even with 1e4 x damping in float64")
    Q = torch.zeros_like(W)
    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            grp = (i1 + i) // GROUP
            w = W1[:, i]
            q = torch.clamp(torch.round((w - OFF[:, grp]) / S[:, grp]), 0, levels - 1) * S[:, grp] + OFF[:, grp]
            Q[:, i1 + i] = q
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q


def flip_candidates(W0, Q, H, G, s, off, levels, k):
    """Per direction (+1/-1 = raises/lowers the note effect): the k flips with the smallest
    output-error increase per unit of predicted note-effect change.

    A flip moves a weight to the other grid level next to its BF16 value. Error increase of one flip
    by d on (row, col): 2 d (E H)[row, col] + d^2 H[col, col], E = Q - W0, normalized by the layer's
    total error sum(E H * E) so layers are comparable. Returns {dir: (ratio, |pred|, flat idx, old, new)}.
    """
    import torch
    out_f, in_f = W0.shape
    S = s.expand(-1, -1, GROUP).reshape(out_f, in_f)
    OFF = off.expand(-1, -1, GROUP).reshape(out_f, in_f)
    cur = torch.round((Q - OFF) / S)
    lo = torch.floor((W0 - OFF) / S).clamp(0, levels - 1)
    hi = (lo + 1).clamp(max=levels - 1)
    alt = torch.where(cur == lo, hi, torch.where(cur == hi, lo, cur))
    delta = (alt - cur) * S
    del cur, lo, hi, alt, OFF
    pred = G * delta
    E = Q - W0
    EH = E @ H
    total = (EH * E).sum().clamp_min(1e-12)
    cost = (2 * delta * EH + delta ** 2 * torch.diag(H)) / total
    del EH, E
    ratio = cost / pred.abs()
    out = {}
    for d in (1, -1):
        r = torch.where(pred * d > 0, ratio, torch.full_like(ratio, float("inf"))).view(-1)
        vals, idx = torch.topk(r, min(k, r.numel()), largest=False)
        ok = torch.isfinite(vals)
        vals, idx = vals[ok], idx[ok]
        out[d] = (vals.cpu(), pred.view(-1)[idx].abs().cpu(), idx.cpu(),
                  Q.view(-1)[idx].cpu(), (Q + delta).view(-1)[idx].cpu())
        del r
    return out


def capture_inputs(model, calib):
    """Hidden states and keyword arguments entering decoder block 0 for each calibration sample."""
    import torch

    class Stop(Exception):
        pass

    caught = []

    def hook(_mod, args, kwargs):
        kw = dict(kwargs)
        x = args[0] if args else kw.pop("hidden_states")
        caught.append((x.detach(), kw))
        raise Stop

    h = model.model.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            for ids in calib:
                try:
                    model(input_ids=ids.unsqueeze(0).to(0), use_cache=False)
                except Stop:
                    pass
    finally:
        h.remove()
    return caught


def gptq_pass(model, calib_inputs, bits, g, grads=None, k=262144, percdamp=0.01):
    """Quantize every decoder Linear in place with GPTQ on grid (bits, g).

    With grads (mnr), also returns flip candidates per direction, tagged with the Linear's index
    in target_linears order. Returns (stats, candidates, modules).
    """
    import torch
    levels = 2 ** bits
    names = {id(m): n for n, m in target_linears(model)}
    modules, cands = [], {1: [], -1: []}
    xs = [x for x, _ in calib_inputs]
    kws = [kw for _, kw in calib_inputs]
    t0 = time.time()
    with torch.no_grad():
        for layer in model.model.layers:
            subset = [(n, m) for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)]
            H = {n: torch.zeros(m.in_features, m.in_features, device=m.weight.device) for n, m in subset}

            def make_hook(n):
                def hook(_mod, inp, _out):
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    H[n].addmm_(x.T, x)
                return hook

            hooks = [m.register_forward_hook(make_hook(n)) for n, m in subset]
            for x, kw in zip(xs, kws):
                layer(x, **kw)
            for h in hooks:
                h.remove()
            for n, m in subset:
                W0 = m.weight.data.float()
                s, off = grid_params(m.weight.data, g, bits)
                try:
                    Q = gptq_quantize(W0, H[n], s, off, levels, percdamp)
                except RuntimeError as e:
                    raise RuntimeError(f"{names[id(m)]}: {e}") from e
                if grads is not None:
                    G = grads[names[id(m)]].to(W0.device).float()
                    for d, c in flip_candidates(W0, Q, H[n], G, s, off, levels, k).items():
                        cands[d].append((len(modules), *c))
                    del G
                modules.append(m)
                m.weight.data.copy_(Q.to(m.weight.dtype))
                del H[n], W0, Q
            for i, (x, kw) in enumerate(zip(xs, kws)):
                out = layer(x, **kw)
                xs[i] = out[0] if isinstance(out, tuple) else out
    torch.cuda.empty_cache()
    return {"gptq_seconds": round(time.time() - t0, 1)}, cands, modules


# ---------------------------------------------------------------- margin-neutral flips

def neutralize(judge, tuning, ne_bf16, cands, modules, rounds=4, tol=0.1):
    """Flip GPTQ weights (in place) until the tuning-set amplification is within tol logits of 0."""
    torch = judge.torch
    a0 = tuning.note_effect(judge) - ne_bf16
    stats = {"amplification_gptq": a0, "history": [[0.0, 0, a0]]}
    if abs(a0) <= tol:
        stats.update(amplification_final=a0, n_flips=0, frac_weights_flipped=0.0)
        return stats
    d = -1 if a0 > 0 else 1
    lid = torch.cat([torch.full((len(c[1]),), j, dtype=torch.int32) for j, *c in cands[d]])
    ratio, pred, idx, old, new = (torch.cat([c[i] for _, *c in cands[d]]) for i in range(5))
    order = torch.argsort(ratio)
    lid, pred, idx, old, new = lid[order], pred[order], idx[order], old[order], new[order]
    cum = torch.cumsum(pred.double(), 0)
    applied = 0

    def set_count(n):
        nonlocal applied
        a, b, vals = (applied, n, new) if n > applied else (n, applied, old)
        with torch.no_grad():
            for j in torch.unique(lid[a:b]).tolist():
                sel = torch.nonzero(lid[a:b] == j).squeeze(1) + a
                w = modules[j].weight
                w.view(-1)[idx[sel].to(w.device)] = vals[sel].to(w.device, w.dtype)
        applied = n

    target = abs(a0)
    a_now = a0
    for _ in range(rounds):
        n = int(torch.searchsorted(cum, torch.tensor(target, dtype=cum.dtype)).clamp(max=len(cum)))
        set_count(n)
        a_now = tuning.note_effect(judge) - ne_bf16
        p_now = float(cum[n - 1]) if n else 0.0
        stats["history"].append([p_now, n, a_now])
        if abs(a_now) <= tol:
            break
        (p1, _, a1), (p2, _, a2) = stats["history"][-2:]
        if a2 == a1:
            break
        target = min(max(p2 - a2 * (p2 - p1) / (a2 - a1), 0.0), float(cum[-1]))
    total = sum(m.weight.numel() for m in modules)
    stats.update(amplification_final=a_now, n_flips=applied, frac_weights_flipped=applied / total,
                 candidates=len(cum), budget_exhausted=applied == len(cum))
    return stats


# ---------------------------------------------------------------- entry point

class Calibrated:
    """Calibration inputs (and, for mnr, the tuning set and BF16 gradients), reused across grids."""

    def __init__(self, judge, method, calib_samples=128, seqlen=512, seed=0, tune_items=DEFAULT_TUNE,
                 tune_problems=32, batch_size=8, k=262144):
        assert method in ("gptq", "mnr")
        self.method, self.k = method, k
        t0 = time.time()
        calib = load_calibration(judge.tok, calib_samples, seqlen, seed)
        self.inputs = capture_inputs(judge.model, calib)
        self.meta = {"method": method, "calib": DEFAULT_CALIB, "calib_samples": calib_samples,
                     "calib_seqlen": seqlen, "calib_seed": seed}
        if method == "mnr":
            if "pointwise" not in judge.labels:
                judge.calibrate("pointwise")
            self.tuning = Tuning(judge, tune_items, tune_problems, batch_size)
            self.ne_bf16 = self.tuning.note_effect(judge)
            self.grads = note_effect_grads(judge, self.tuning, max(1, batch_size // 2))
            self.meta.update(tune_items=tune_items, tune_problems=self.tuning.n_problems,
                             tune_pairs=len(self.tuning.pairs), note_effect_bf16=self.ne_bf16)
        self.meta["setup_seconds"] = round(time.time() - t0, 1)

    def quantize(self, judge, grid):
        """Quantize judge.model (holding BF16 weights) in place on grid; returns stats."""
        bits, g = parse_grid(grid)
        grads = self.grads if self.method == "mnr" else None
        stats, cands, modules = gptq_pass(judge.model, self.inputs, bits, g, grads, self.k)
        if self.method == "mnr":
            stats.update(neutralize(judge, self.tuning, self.ne_bf16, cands, modules))
        return stats


def self_test():
    """Random layer: GPTQ stays on the grid and beats RTN on output error; flips move the right way."""
    import torch
    from quant_grids import fake_quant
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_f, in_f, n = 64, 4 * GROUP, 2048
    W = torch.randn(out_f, in_f, device=dev)
    X = torch.randn(n, in_f, device=dev) @ torch.randn(in_f, in_f, device=dev) / in_f ** 0.5  # correlated
    H = X.T @ X
    for g in (0, 5):
        s, off = grid_params(W, g, 4)
        Q = gptq_quantize(W, H, s, off, 16)
        R = fake_quant(W, g, 4)[0].float()
        S = s.expand(-1, -1, GROUP).reshape(out_f, in_f)
        O = off.expand(-1, -1, GROUP).reshape(out_f, in_f)
        code = (Q - O) / S
        assert torch.allclose(code, code.round(), atol=1e-3) and code.min() > -1e-3 and code.max() < 15 + 1e-3
        e_gptq, e_rtn = float(((Q - W) @ X.T).norm()), float(((R - W) @ X.T).norm())
        assert e_gptq < e_rtn, (e_gptq, e_rtn)
        G = torch.randn_like(W)
        c = flip_candidates(W, Q, H, G, s, off, 16, 100)
        for d, (ratio, pred, idx, old, new) in c.items():
            assert len(idx) > 0 and torch.all(torch.diff(ratio) >= 0)
            assert torch.all(d * G.view(-1)[idx].cpu() * (new - old) > 0)
            assert torch.allclose(old, Q.view(-1)[idx].cpu())
        print(f"[self-test] grid g{g:02d}: output error GPTQ {e_gptq:.1f} vs RTN {e_rtn:.1f}; "
              f"{len(c[1][2])}+{len(c[-1][2])} flip candidates ok")
    print("[self-test] passed")


if __name__ == "__main__":
    self_test()
