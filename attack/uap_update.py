"""Two-stage single-surrogate UAP optimizer."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

from . import augmentation_proxy_v1 as a
from . import rotating_single_proxy as r


EFFECTIVE_BATCH = 128
LOGGER = logging.getLogger(__name__)


def generate_uap(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    refinement: Mapping,
    *,
    seed: int,
    legacy_vit_warmup: bool = False,
) -> torch.Tensor:
    """Warm up with true-label cross-entropy, then refine with target logits."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The surrogate must be frozen and in evaluation mode")
    if (
        images.ndim != 4
        or images.shape[1:] != (3, 224, 224)
        or len(images) not in (500, 2000)
        or not images.is_cuda
        or images.dtype != torch.float32
        or labels.shape != (len(images),)
        or labels.dtype != torch.long
        or labels.device != images.device
    ):
        raise ValueError("Expected 500 or 2000 CUDA pixel tensors and matching ImageNet labels")

    warmup = r.WARMUP
    # The 2,000-image ViT warmup uses the ordinary float32 clamp.
    warmup_bound = r.EPS if legacy_vit_warmup else r.BOUND
    delta = r.initial_delta(seed)
    stages = (
        ("ce20", warmup, warmup["epochs"], warmup_bound),
        ("refine", refinement, refinement["refine_epochs"], r.BOUND),
    )
    for name, phase, epochs, bound in stages:
        delta.clamp_(-bound, bound)
        momentum = torch.zeros_like(delta)
        for epoch in range(epochs):
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(len(images), generator=generator).to(images.device)
            for start in range(0, len(order), EFFECTIVE_BATCH):
                ids = order[start:start + EFFECTIVE_BATCH]
                gradient = torch.zeros_like(delta)
                for offset in range(0, len(ids), phase["microbatch"]):
                    idx = ids[offset:offset + phase["microbatch"]]
                    transform_seed = seed * 1000000000 + epoch * 1000000 + start * 1000 + offset
                    if name == "ce20":
                        _, part = r.gradient(model, images[idx], labels[idx], delta, phase, transform_seed)
                    else:
                        _, part = a.gradient(
                            model, images[idx], labels[idx], delta, phase, phase["augmentation"], transform_seed,
                        )
                    gradient.add_(part, alpha=len(idx) / len(ids))
                momentum.mul_(phase["momentum"]).add_(gradient / gradient.abs().mean().clamp_min(1e-12))
                delta.add_(phase["step_pixels"] / 255 * momentum.sign()).clamp_(-bound, bound)
            LOGGER.info("%s epoch %d/%d", name, epoch + 1, epochs)
    return delta.detach().cpu()
