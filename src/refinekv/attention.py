"""Framework-independent single-token attention entry points."""

from typing import NamedTuple
import importlib.util

import torch

from .config import RefineKVConfig
from .summary import BlockSummary


class AttentionResult(NamedTuple):
    output: torch.Tensor
    selected: torch.Tensor | None


def _backend(query, config):
    if config.backend == "torch":
        return "torch"
    available = query.is_cuda and importlib.util.find_spec("triton") is not None
    if config.backend == "triton" and not available:
        raise RuntimeError("the Triton backend requires CUDA and the 'cuda' installation extra")
    if not available:
        return "torch"
    if query.dtype != torch.bfloat16:
        if config.backend == "triton":
            raise TypeError("the Triton backend requires BF16 queries")
        return "torch"
    dim, group = query.shape[-1], query.shape[1]
    if dim < 32 or dim > 256 or dim % 32 or group > 16:
        if config.backend == "triton":
            raise ValueError(
                "Triton requires head_dim in 32..256 (multiple of 32) and GQA group <= 16"
            )
        return "torch"
    if torch.cuda.get_device_capability(query.device)[0] < 8:
        if config.backend == "triton":
            raise RuntimeError("the Triton BF16 backend requires compute capability 8.0 or newer")
        return "torch"
    return "triton"


def _validate(query, summary, config, anchor_mask=None):
    if query.ndim != 4 or query.shape[2] != 1 or min(query.shape) <= 0:
        raise ValueError("query must have shape (kv_heads, group_size, 1, head_dim)")
    if query.shape[0] != summary.keys.shape[0] or query.shape[-1] != summary.keys.shape[-1]:
        raise ValueError("query heads and dimension must match the summary")
    if query.device != summary.keys.device:
        raise ValueError("query and summary must be on the same device")
    if config.block_size != summary.block_size:
        raise ValueError("config.block_size must match the summary")
    if anchor_mask is not None:
        if anchor_mask.dtype != torch.bool or tuple(anchor_mask.shape) != (
            query.shape[0],
            summary.num_blocks,
        ):
            raise ValueError("anchor_mask must be boolean with shape (kv_heads, blocks)")
        if anchor_mask.device != query.device:
            raise ValueError("anchor_mask must be on the query device")


@torch.no_grad()
def select_blocks(query, summary, config=None, *, workspace=None):
    """Return a boolean ``(kv_heads, blocks)`` mask for one query."""
    config = config or RefineKVConfig(block_size=summary.block_size)
    _validate(query, summary, config)
    if _backend(query, config) == "triton":
        from .kernels.runtime import select_blocks_triton

        return select_blocks_triton(query, summary, config, workspace)
    from .reference import select_blocks_reference

    return select_blocks_reference(query, summary, config)[0]


@torch.no_grad()
def sparse_attention(
    query: torch.Tensor,
    summary: BlockSummary,
    key_suffix: torch.Tensor,
    value_suffix: torch.Tensor,
    config: RefineKVConfig | None = None,
    *,
    anchor_mask: torch.Tensor | None = None,
    workspace: dict | None = None,
    return_selected: bool = False,
    suffix_length: torch.Tensor | None = None,
) -> AttentionResult:
    """Attend to representatives, selected prompt pages, and the exact suffix.

    ``query`` has shape ``(kv_heads, group_size, 1, head_dim)``. Suffix K/V
    have shape ``(kv_heads, suffix_tokens, head_dim)``. No causal mask is
    required because the supplied suffix must contain only visible tokens.
    ``suffix_length`` optionally gives the valid prefix of a fixed-capacity
    suffix buffer as a device-side int32 scalar for CUDA Graph replay.
    CUDA outputs backed by a workspace are reused on the next call; clone
    them if retaining them beyond that call.
    """
    config = config or RefineKVConfig(block_size=summary.block_size)
    _validate(query, summary, config, anchor_mask)
    expected = (query.shape[0], query.shape[-1])
    if key_suffix.ndim != 3 or key_suffix.shape != value_suffix.shape:
        raise ValueError("suffix K/V must have matching (kv_heads, tokens, head_dim) shapes")
    if (key_suffix.shape[0], key_suffix.shape[-1]) != expected:
        raise ValueError("suffix heads and dimension must match the query")
    if key_suffix.device != query.device or value_suffix.device != query.device:
        raise ValueError("suffix K/V must be on the query device")
    backend = _backend(query, config)
    if suffix_length is not None:
        if (suffix_length.shape != () or suffix_length.device != query.device
                or suffix_length.dtype != torch.int32):
            raise ValueError("suffix_length must be a scalar int32 tensor on the query device")
        if backend != "triton":
            raise ValueError("device-side suffix_length requires the Triton backend")
    if backend == "triton":
        from .kernels.runtime import sparse_attention_triton

        return sparse_attention_triton(
            query,
            summary,
            key_suffix,
            value_suffix,
            config,
            anchor_mask,
            workspace,
            return_selected,
            suffix_length,
        )
    from .reference import sparse_attention_reference

    output, selected = sparse_attention_reference(
        query, summary, key_suffix, value_suffix, config, anchor_mask
    )
    return AttentionResult(output, selected if return_selected else None)
