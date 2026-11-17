"""GPTAQ-style asymmetric calibration (Li et al., 2025, "GPTQv2"), simulated like the team's gptq_w4.

GPTQ (team simquant.gptq_model) quantizes each decoder layer to reproduce ITS OWN output on the
inputs X~ that the already-quantized earlier layers produce:  min_Q || X~ Q^T - X~ W^T ||^2.
GPTAQ instead targets the FULL-PRECISION model's output, so each layer also absorbs the error
that earlier layers introduced:                               min_Q || X~ Q^T - X W^T ||^2.
Splitting off the least-squares part, that equals  || X~ (Q - W*)^T ||^2 + const  with
    W* = W (X^T X~) H^-1 = W + W (X - X~)^T X~ H^-1,   H = X~^T X~,
so this runs the team's unchanged GPTQ routine (same grid, group 128, damping 0.01, no act-order)
on the target W* with Hessian H. alpha scales the correction (1 = full asymmetric target);
cdamp is the ridge used only for the W* solve. With alpha 1 and GPTQ's 1% damping the
correction blew up (|W*-W|/|W| up to 5.5 on Qwen3-4B), so it is regularised.

This is our implementation of the objective, not the authors' code (their column-wise update
reaches the same objective without forming W*). X and X~ come from the same calibration tokens:
for every layer, each calibration sequence is run through the not-yet-quantized layer twice, once
from the full-precision stream and once from the quantized stream, and the inputs of every Linear
are paired row by row.
"""
import math

import torch
import torch.nn as nn


class _Stop(Exception):
    pass


@torch.no_grad()
def gptaq_model(model, tok, calib, sq, bits=4, alpha=1.0, cdamp=None, dev="cuda"):
    """cdamp: ridge for the W* solve only, as a fraction of mean diag(H) (default: GPTQ's DAMP)."""
    cdamp = sq.DAMP if cdamp is None else cdamp
    layers = model.model.layers
    cache = []

    def catcher(hidden_states, *a, **kw):
        cache.append((hidden_states, kw))
        raise _Stop

    orig_fwd = layers[0].forward
    layers[0].forward = catcher
    for text in calib:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=sq.CALIB_MAXLEN).to(dev)
        try:
            model(**ids, use_cache=False)
        except _Stop:
            pass
    layers[0].forward = orig_fwd
    ntok = sum(h.shape[1] for h, _ in cache)
    print(f"[gptaq] captured {len(cache)} calibration sequences, {ntok} tokens", flush=True)

    qc = list(cache)                  # quantized stream (inputs the quantized model sees)
    fc = [(h.clone(), kw) for h, kw in cache]   # full-precision stream
    out = lambda o: o[0] if isinstance(o, tuple) else o
    stats, corr = {}, {}
    for li, layer in enumerate(layers):
        lin = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
        H = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in lin.items()}
        A = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in lin.items()}
        cur = {}

        def mk(n):
            def hook(mod, inp, o):
                cur[n] = inp[0].reshape(-1, inp[0].shape[-1]).float()
            return hook

        hs = [m.register_forward_hook(mk(n)) for n, m in lin.items()]
        new_fc = []
        for (hq, kw), (hf, kwf) in zip(qc, fc):
            layer(hq, **kw)
            xq = dict(cur)
            new_fc.append((out(layer(hf, **kwf)), kwf))   # FP layer output = next FP input
            for n in lin:
                xt, x = xq[n], cur[n]
                H[n] += xt.t() @ xt
                A[n] += (x - xt).t() @ xt
        for h in hs:
            h.remove()
        fc = new_fc
        for n, m in lin.items():
            Hn, An = H[n] / ntok, A[n] / ntok
            W = m.weight.data.float()
            if alpha:
                Hd = Hn.clone()
                dead = torch.diag(Hd) == 0
                Hd[dead, dead] = 1
                Hd += cdamp * torch.mean(torch.diag(Hd)) * torch.eye(Hd.shape[0], device=dev)
                L = torch.linalg.cholesky(Hd)
                # W* = W + alpha * W A H^-1   (A H^-1 via a Cholesky solve on the transpose)
                delta = torch.cholesky_solve((W @ An).t(), L).t()
                corr[f"{li}.{n}"] = float(delta.norm() / W.norm())
                W = W + alpha * delta
            Q, loss = sq.gptq(W, 2 * Hn, bits)
            m.weight.data = Q.to(m.weight.dtype)
            stats[f"{li}.{n}"] = round(loss, 4)
        del H, A
        qc = [(out(layer(h, **kw)), kw) for h, kw in qc]    # through the QUANTIZED layer
        if li % 4 == 0:
            print(f"[gptaq] layer {li}/{len(layers)} done, mean |W*-W|/|W| "
                  f"{sum(v for k, v in corr.items() if k.startswith(f'{li}.')) / max(1, len(lin)):.4f}",
                  flush=True)
    return stats, corr
