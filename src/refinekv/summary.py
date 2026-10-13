"""Prompt summaries for continuous K/V storage."""

from dataclasses import dataclass

import torch

from .config import RefineKVConfig


@dataclass
class BlockSummary:
    """Per-KV-head page summaries; original K/V remain available for refinement.

    K/V inputs have shape ``(kv_heads, tokens, head_dim)``. Page storage has
    shape ``(kv_heads, pages, block_size, head_dim)`` and may share the cache
    allocation. Summaries belong to one immutable prompt prefix.
    """

    keys: torch.Tensor
    values: torch.Tensor
    counts: torch.Tensor
    key_mean: torch.Tensor
    value_mean: torch.Tensor
    ellipsoid: torch.Tensor
    value_norm: torch.Tensor
    max_value_norm: torch.Tensor
    prompt_length: int
    block_size: int

    @property
    def num_blocks(self):
        return self.counts.numel()


def _pages(tensor, block_size):
    heads, length, dim = tensor.shape
    padding = (-length) % block_size
    if padding:
        tensor = torch.cat([tensor, tensor.new_zeros(heads, padding, dim)], dim=1)
    pages = tensor.reshape(heads, -1, block_size, dim)
    if pages.stride(3) != 1 or pages.stride(2) != dim or pages.stride(1) != block_size * dim:
        pages = pages.contiguous()
    return pages


@torch.no_grad()
def build_summary(
    keys: torch.Tensor, values: torch.Tensor, config: RefineKVConfig | None = None
) -> BlockSummary:
    """Build BF16 summaries with a centered, cancellation-free radius.

    FP32 radius construction is chunked to bound temporary memory. The
    folded squared-axis vector is rounded upward before BF16 storage.
    """
    config = config or RefineKVConfig()
    if keys.ndim != 3 or keys.shape != values.shape or min(keys.shape) <= 0:
        raise ValueError(
            "keys and values must have the same nonempty (kv_heads, tokens, dim) shape"
        )
    if keys.device != values.device or keys.dtype != values.dtype:
        raise ValueError("keys and values must have the same device and dtype")
    if not keys.is_floating_point():
        raise TypeError("keys and values must be floating point")

    block = config.block_size
    length = keys.shape[1]
    key_pages, value_pages = _pages(keys.detach(), block), _pages(values.detach(), block)
    n_blocks = key_pages.shape[1]
    counts = torch.full((n_blocks,), block, device=keys.device, dtype=torch.int64)
    last_valid = length - (n_blocks - 1) * block
    counts[-1] = last_valid
    sizes = counts.float()[None, :, None]
    key_mean = (key_pages.sum(2, dtype=torch.float32) / sizes).to(torch.bfloat16)
    value_mean = (value_pages.sum(2, dtype=torch.float32) / sizes).to(torch.bfloat16)
    key_max, key_min = key_pages.amax(2), key_pages.amin(2)
    if last_valid < block:
        tail = key_pages[:, -1, :last_valid]
        key_max[:, -1], key_min[:, -1] = tail.amax(1), tail.amin(1)
    width = torch.maximum(key_max.float() - key_mean.float(), key_mean.float() - key_min.float())
    squared_width = width.clamp_min(1e-6).square().to(torch.bfloat16)
    ellipsoid = torch.empty_like(squared_width)
    for start in range(0, n_blocks, config.summary_chunk_blocks):
        end = min(start + config.summary_chunk_blocks, n_blocks)
        # Always copy: .float() alone aliases FP32 prompt/cache storage.
        centered = key_pages[:, start:end].to(dtype=torch.float32, copy=True)
        centered.sub_(key_mean[:, start:end].float().unsqueeze(2))
        centered.mul_(torch.rsqrt(squared_width[:, start:end].float()).unsqueeze(2))
        centered.square_()
        distances = centered.sum(-1)
        if end == n_blocks and last_valid < block:
            distances[:, -1, last_valid:] = -torch.inf
        radius_squared = distances.amax(-1).clamp_min(0)
        # Match the existing centered-rho builder, including its sqrt/square rounding.
        radius_squared = radius_squared.sqrt().square()
        ellipsoid[:, start:end] = (
            radius_squared.unsqueeze(-1) * squared_width[:, start:end].float() * (1 + 2**-8)
        ).to(torch.bfloat16)
    norms = torch.linalg.vector_norm(value_pages, dim=-1, dtype=torch.float32)
    if last_valid < block:
        norms[:, -1, last_valid:] = -torch.inf
    value_norm = norms.amax(-1)
    return BlockSummary(
        keys=key_pages,
        values=value_pages,
        counts=counts,
        key_mean=key_mean,
        value_mean=value_mean,
        ellipsoid=ellipsoid,
        value_norm=value_norm,
        max_value_norm=value_norm.amax(-1),
        prompt_length=length,
        block_size=block,
    )
