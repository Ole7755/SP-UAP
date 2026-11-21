"""Generate a universal perturbation using the paper's surrogate presets."""

from __future__ import annotations

import torch

from .augmentation_proxy_v1 import METHODS
from .uap_update import generate_uap as _generate_uap

__all__ = ["SURROGATES", "generate_paper_uap"]

SURROGATES = {
    "vit": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "swin": "swin_tiny_patch4_window7_224.ms_in1k",
    "deit": "deit_base_patch16_224.fb_in1k",
    "cait": "cait_s24_224.fb_dist_in1k",
    "levit": "levit_384.fb_dist_in1k",
}
_PAPER_SEEDS = {"vit": 0, "swin": 2, "deit": 2, "cait": 2, "levit": 0}
_REFINEMENT_DEFAULTS = {
    "refine_epochs": 40,
    "loss": "normalized_logit",
    "target_class": 109,
    "step_pixels": 0.5,
    "microbatch": 64,
    "momentum": 0.9,
    "shift_probability": 0.7,
    "shift_pixels": 4,
    "augmentation": METHODS["scale_color17_whole"],
}
_REFINEMENT_OVERRIDES = {
    "vit": {"refine_epochs": 5, "augmentation": METHODS["control"]},
    "swin": {"shift_probability": 1.0},
    "deit": {"shift_pixels": 8, "augmentation": METHODS["scale_mixup_clean"]},
    "levit": {"refine_epochs": 80, "step_pixels": 0.25, "shift_probability": 1.0, "shift_pixels": 8},
}


def generate_paper_uap(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    source: str,
    seed: int | None = None,
) -> torch.Tensor:
    """Return a CPU perturbation of shape (1, 3, 224, 224) at the [0, 1] pixel scale.

    Select the surrogate with a key from ``SURROGATES``. Supply 500 or 2000
    float32 RGB images of shape (N, 3, 224, 224) in [0, 1] and N long labels
    in the model's ImageNet class order. The frozen, eval-mode model and both
    tensors must be on the current CUDA device. The model must normalize its
    input with the checkpoint's statistics and return class logits. Run with
    gradients enabled for the perturbation. The paper runs disabled fused
    attention in the surrogate, TF32 in CUDA and cuDNN, and cuDNN benchmarking,
    and used deterministic cuDNN kernels.

    With ``seed=None``, use the paper's seed for the 2000-image setting.
    """
    if source not in _PAPER_SEEDS:
        raise ValueError(f"Unknown surrogate {source!r}. Choose from {', '.join(SURROGATES)}.")
    if seed is None:
        seed = _PAPER_SEEDS[source]
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("The seed must be an integer in [0, 2**32).")

    refinement = {**_REFINEMENT_DEFAULTS, **_REFINEMENT_OVERRIDES.get(source, {})}
    return _generate_uap(
        model,
        images,
        labels,
        refinement,
        seed=seed,
        legacy_vit_warmup=source == "vit" and len(images) == 2000,
    )
