"""Perturbation initialization, cross-entropy warmup, and logit refinement."""

from __future__ import annotations

import torch
from torch.nn import functional as F

EPS = 12 / 255
_FLOAT32_EPS = torch.tensor(EPS, dtype=torch.float32)
if _FLOAT32_EPS.item() > EPS:
    _FLOAT32_EPS = torch.nextafter(_FLOAT32_EPS, torch.tensor(0.0))
BOUND = _FLOAT32_EPS.item()

WARMUP = {
    "epochs": 20,
    "loss": "ce",
    "step_pixels": 2.0,
    "microbatch": 8,
    "momentum": 0.9,
    "shift_probability": 0.0,
}


def initial_delta(seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.empty(1, 3, 224, 224, device="cuda").uniform_(-EPS, EPS, generator=generator)


def gradient(model, x, y, delta, phase, transform_seed):
    v = delta.detach().requires_grad_(True)
    value = (x + v).clamp(0, 1)
    if phase["shift_probability"]:
        rng = torch.Generator().manual_seed(transform_seed)
        if torch.rand((), generator=rng).item() < phase["shift_probability"]:
            limit = phase["shift_pixels"]
            dy, dx = torch.randint(-limit, limit + 1, (2,), generator=rng).tolist()
            value = torch.roll(value, (dy, dx), (2, 3))

    logits = model(value)
    if phase["loss"] == "ce":
        loss = F.cross_entropy(logits, y)
    else:
        z = logits - logits.mean(1, keepdim=True)
        loss = (z / z.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6))[:, phase["target_class"]].mean()
    (g,) = torch.autograd.grad(loss, v)
    assert torch.isfinite(loss) and g.isfinite().all()
    return loss.detach(), g.detach()
