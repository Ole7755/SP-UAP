"""Scale, color, and mixup transformations used in refinement."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from . import rotating_single_proxy as r

METHODS = {
    "control": {"kind": "identity", "scope": "whole"},
    "scale_color17_whole": {"kind": "scale_color", "scope": "whole", "ratio": 1.7, "color_p": 0.8, "span": 0.5},
    "scale_mixup_clean": {"kind": "scale_mixup", "scope": "clean", "ratio": 1.7, "color_p": 0.8, "span": 0.5},
}


def draw(rng, lo=0.0, hi=1.0):
    return lo + (hi - lo) * torch.rand((), generator=rng).item()


def integer(rng, n):
    return int(torch.randint(n, (1,), generator=rng).item())


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


def scale(x, rng, ratio):
    if draw(rng) > 0.9:
        return x
    h, w = x.shape[-2:]
    factor = draw(rng, 1 / ratio, ratio)
    if factor < 1:
        hh = max(2, round(h * factor))
        ww = max(2, round(w * factor))
        dy = integer(rng, h - hh + 1)
        dx = integer(rng, w - ww + 1)
        return F.pad(resize(x, (hh, ww)), (dx, w - ww - dx, dy, h - hh - dy))

    hh = max(2, min(h, round(h / factor)))
    ww = max(2, min(w, round(w / factor)))
    dy = integer(rng, h - hh + 1)
    dx = integer(rng, w - ww + 1)
    return resize(x[:, :, dy:dy + hh, dx:dx + ww], (h, w))


def color(x, rng, p=0.8, span=0.5):
    if draw(rng) > p:
        return x
    op = integer(rng, 5)
    factor = draw(rng, 1 - span, 1 + span)
    if op == 0:
        return x.flip(-1)
    if op == 1:
        return x.flip(-2)

    gray = (x * x.new_tensor([0.2989, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    if op == 2:
        y = x * factor
    elif op == 3:
        mean = gray.mean((2, 3), keepdim=True)
        y = mean + (x - mean) * factor
    else:
        y = gray + (x - gray) * factor
    return y.clamp(0, 1)


def transform(x, cfg, rng):
    kind = cfg["kind"]
    if kind == "identity":
        return x
    x = scale(x, rng, cfg["ratio"])
    x = color(x, rng, cfg.get("color_p", 0.8), cfg.get("span", 0.5))
    if kind == "scale_mixup":
        other = x[torch.randperm(len(x), generator=rng).to(x.device)]
        alpha = draw(rng, 0.2, 0.5)
        x = (1 - alpha) * x + alpha * other
    return x


def gradient(model, x, y, delta, phase, augmentation, seed):
    if augmentation["kind"] == "identity":
        return r.gradient(model, x, y, delta, phase, seed)

    v = delta.detach().requires_grad_(True)
    rng = torch.Generator().manual_seed(seed + 98765431)
    if augmentation["scope"] == "clean":
        value = (transform(x, augmentation, rng) + v).clamp(0, 1)
    else:
        value = transform((x + v).clamp(0, 1), augmentation, rng)

    rng = torch.Generator().manual_seed(seed)
    if torch.rand((), generator=rng).item() < phase["shift_probability"]:
        limit = phase["shift_pixels"]
        dy, dx = torch.randint(-limit, limit + 1, (2,), generator=rng).tolist()
        value = torch.roll(value, (dy, dx), (2, 3))

    logits = model(value)
    z = logits - logits.mean(1, keepdim=True)
    loss = (z / z.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6))[:, phase["target_class"]].mean()
    (g,) = torch.autograd.grad(loss, v)
    assert torch.isfinite(loss) and g.isfinite().all()
    return loss.detach(), g.detach()
