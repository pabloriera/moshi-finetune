"""Build student LMs and inspect exact shapes without allocating model weights."""

import torch
from moshi.models.lm import LMModel

from .config import Architecture


def build_model(architecture: Architecture, *, device="cpu", checkpointing=False):
    return LMModel(
        **architecture.lm_kwargs(checkpointing=checkpointing),
        device=device,
        dtype=torch.float32,
    )


def parameter_report(architecture: Architecture):
    with torch.device("meta"):
        model = build_model(architecture, device="meta")
    counts = {
        "temporal_transformer": 0,
        "depth_transformer": 0,
        "embeddings_interfaces_heads": 0,
    }
    for name, parameter in model.named_parameters():
        group = (
            "temporal_transformer"
            if name.startswith("transformer.")
            else "depth_transformer"
            if name.startswith("depformer.")
            else "embeddings_interfaces_heads"
        )
        counts[group] += parameter.numel()
        if not parameter.is_meta:
            raise RuntimeError(f"Parameter inspection allocated a tensor: {name}")
    total = sum(counts.values())
    return {
        "parameters": total,
        "groups": counts,
        "bf16_weights_gib": total * 2 / 1024**3,
        "mixed_precision_adam_state_estimate_gib": total * 16 / 1024**3,
        "memory_note": "16 bytes/parameter estimate; excludes hidden matchers, activations, Mimi, allocator, and optimizer temporaries",
        "mimi_included": False,
    }
