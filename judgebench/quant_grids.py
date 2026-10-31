"""Offset-shifted round-to-nearest weight grids (fake quantization), shared by jb_eval and S9.

Every nn.Linear inside the decoder blocks is rounded; embeddings and lm_head stay BF16.
Per group of GROUP consecutive input weights, with L = 2**bits levels:
  s = (max - min) / (L - 2),   levels_g = min - (g / N_GRIDS) s + j s,   j = 0..L-1
One step of slack keeps [min, max] covered for every shift, so all grids of a bit width share the
same scale and worst-case error (s / 2); only the offset differs.

Grid names: "bf16" (no rounding), "g05" (INT4, shift 5/16), "i8g05" (INT8, shift 5/16).
"""

import re

GROUP = 128
N_GRIDS = 16
_NAME = re.compile(r"^(i8)?g(\d+)$")


def parse_grid(name):
    """'g05' -> (4, 5); 'i8g05' -> (8, 5); 'bf16' -> None."""
    if name == "bf16":
        return None
    m = _NAME.match(name)
    if not m or not 0 <= int(m.group(2)) < N_GRIDS:
        raise ValueError(f"bad grid name {name!r}; expected bf16, g00..g{N_GRIDS - 1}, i8g00..i8g{N_GRIDS - 1}")
    return (8 if m.group(1) else 4), int(m.group(2))


def grid_family(name):
    """'g05' -> 'int4', 'i8g05' -> 'int8', 'bf16' -> 'bf16'."""
    spec = parse_grid(name)
    return "bf16" if spec is None else f"int{spec[0]}"


def target_linears(model):
    import torch
    return [(n, m) for n, m in model.model.layers.named_modules() if isinstance(m, torch.nn.Linear)]


def fake_quant(w, g, bits=4):
    """Shifted asymmetric RTN of a (out, in) weight; returns (dequantized weight, max |err| / s)."""
    import torch
    levels = 2 ** bits
    out_f, in_f = w.shape
    assert in_f % GROUP == 0, f"in_features {in_f} not divisible by {GROUP}"
    x = w.float().reshape(out_f, in_f // GROUP, GROUP)
    lo = x.amin(-1, keepdim=True)
    hi = x.amax(-1, keepdim=True)
    s = ((hi - lo) / (levels - 2)).clamp_min(1e-8)
    off = lo - (g / N_GRIDS) * s
    q = torch.clamp(torch.round((x - off) / s), 0, levels - 1)
    deq = q * s + off
    err = float(((deq - x).abs() / s).max())
    return deq.reshape(out_f, in_f).to(w.dtype), err


def apply_grid(linears, original, name):
    """Write grid `name` into the given linears from their BF16 originals; returns max |err| / s."""
    import torch
    spec = parse_grid(name)
    errs = [0.0]
    with torch.no_grad():
        for n, m in linears:
            if spec is None:
                m.weight.copy_(original[n])
            else:
                deq, e = fake_quant(original[n], spec[1], spec[0])
                m.weight.copy_(deq)
                errs.append(e)
    return max(errs)
