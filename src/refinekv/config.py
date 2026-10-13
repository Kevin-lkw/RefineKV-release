"""Explicit algorithm configuration, independent of inference frameworks."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RefineKVConfig:
    """Settings for diagonal-ellipsoid selection and sparse decode.

    ``eps`` is a condition threshold, not a fixed token budget.
    ``any`` expands a page when any query head in its KV-head group selects it.
    Prompt anchors are optional and use the same threshold and reduction.
    """

    block_size: int = 32
    eps: float = 0.01
    gqa_reduction: str = "any"
    anchor_tokens: int = 0
    full_attention_layers: int = 0
    backend: str = "auto"
    summary_chunk_blocks: int = 512
    collect_stats: bool = False

    def __post_init__(self):
        if self.block_size not in (16, 32, 64):
            raise ValueError("block_size must be 16, 32, or 64")
        if not math.isfinite(self.eps) or self.eps < 0:
            raise ValueError("eps must be finite and nonnegative")
        if self.gqa_reduction not in ("mean", "max", "any"):
            raise ValueError("gqa_reduction must be 'mean', 'max', or 'any'")
        if self.backend not in ("auto", "torch", "triton"):
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        for name in ("anchor_tokens", "full_attention_layers"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.summary_chunk_blocks, int) or self.summary_chunk_blocks <= 0:
            raise ValueError("summary_chunk_blocks must be a positive integer")
