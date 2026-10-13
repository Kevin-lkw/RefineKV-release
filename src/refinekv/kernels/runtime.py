"""Launch the production split kernels without process-global configuration."""

import torch
import triton

from ..attention import AttentionResult
from .attention import (
    attention_reduce_kernel,
    finalize_pages_kernel,
    finalize_representatives_kernel,
)
from .selection import (
    selection_finalize_kernel,
    selection_reduce_kernel,
    selection_stats_kernel,
)


_BLOCK = 32
_SELECT_PROGRAMS = 48
_REP_PROGRAMS = 48
_PAGE_PROGRAMS = 64
_REDUCTION_CODES = {"mean": 0, "max": 1, "any": 2}


def _buffer(workspace, name, shape, device, dtype):
    tensor = workspace.get(name) if workspace is not None else None
    if (
        tensor is None
        or tuple(tensor.shape) != tuple(shape)
        or tensor.device != device
        or tensor.dtype != dtype
    ):
        tensor = torch.empty(shape, device=device, dtype=dtype)
        if workspace is not None:
            workspace[name] = tensor
    return tensor


def _statistics(query, summary, workspace, *, reduce=False):
    heads, groups, _, dim = query.shape
    rows = heads * groups
    q = query.reshape(rows, dim).contiguous()
    blocks = summary.num_blocks
    programs = min(_SELECT_PROGRAMS, triton.cdiv(blocks, _BLOCK))
    span = triton.cdiv(triton.cdiv(blocks, _BLOCK), programs)
    s = _buffer(workspace, "scores", (rows, blocks), q.device, torch.float32)
    delta = _buffer(workspace, "delta", (rows, blocks), q.device, torch.float32)
    partial = _buffer(workspace, "selection_partial", (4, rows, programs), q.device, torch.float32)
    selection_stats_kernel[(heads, programs)](
        q_ptr=q,
        k_bar_ptr=summary.key_mean,
        w2_ptr=summary.ellipsoid,
        counts_ptr=summary.counts,
        s_cache_ptr=s,
        delta_cache_ptr=delta,
        z_m_ptr=partial[0],
        z_l_ptr=partial[1],
        c_m_ptr=partial[2],
        c_l_ptr=partial[3],
        n_blocks=blocks,
        span=span,
        num_p=programs,
        group_size=groups,
        head_dim=dim,
        scale=dim**-0.5,
        delta_scale=dim**-0.5 * (1 + 2**-8),
        BLOCK_G=16,
        BLOCK_B=_BLOCK,
        BLOCK_D=triton.next_power_of_2(dim),
        NUM_STAGES=3,
        num_warps=4,
    )
    global_stats = None
    if reduce:
        global_stats = _buffer(workspace, "selection_global", (4, rows), q.device, torch.float32)
        selection_reduce_kernel[(rows,)](
            partial[0],
            partial[1],
            partial[2],
            partial[3],
            global_stats[0],
            global_stats[1],
            global_stats[2],
            global_stats[3],
            programs,
            BLOCK_C=triton.next_power_of_2(programs),
            num_warps=4,
        )
    return q, s, delta, partial, global_stats, programs


def select_blocks_triton(query, summary, config, workspace=None):
    heads, groups, _, _ = query.shape
    q, s, delta, _, global_stats, _ = _statistics(query, summary, workspace, reduce=True)
    blocks = summary.num_blocks
    selected = _buffer(workspace, "selection_mask", (heads * groups, blocks), q.device, torch.bool)
    logits = _buffer(workspace, "selection_logits", selected.shape, q.device, torch.float32)
    selection_finalize_kernel[(heads, triton.cdiv(blocks, _BLOCK))](
        s_cache_ptr=s,
        delta_cache_ptr=delta,
        v_norm_ptr=summary.value_norm,
        v_norm_all_ptr=summary.max_value_norm,
        counts_ptr=summary.counts,
        z_m_ptr=global_stats[0],
        z_l_ptr=global_stats[1],
        c_m_ptr=global_stats[2],
        c_l_ptr=global_stats[3],
        selected_ptr=selected,
        z_out_ptr=logits,
        n_blocks=blocks,
        group_size=groups,
        eps=config.eps,
        BLOCK_G=triton.next_power_of_2(groups),
        BLOCK_B=_BLOCK,
        GQA_REDUCTION=_REDUCTION_CODES[config.gqa_reduction],
        num_warps=4,
    )
    # Own the result: a caller may retain it while the workspace is reused.
    return selected.reshape(heads, groups, blocks)[:, 0].clone()


def sparse_attention_triton(
    query, summary, key_suffix, value_suffix, config, anchor_mask, workspace, return_selected,
    suffix_length=None,
):
    heads, groups, _, dim = query.shape
    rows, blocks = heads * groups, summary.num_blocks
    q, s, delta, selection_partial, _, selection_programs = _statistics(query, summary, workspace)
    rep_programs = min(_REP_PROGRAMS, triton.cdiv(blocks, _BLOCK))
    page_programs = min(_PAGE_PROGRAMS, blocks)
    slots = rep_programs + page_programs
    span = triton.cdiv(triton.cdiv(blocks, _BLOCK), rep_programs)
    partial_acc = _buffer(workspace, "attention_acc", (rows, slots, dim), q.device, torch.float32)
    partial_m = _buffer(workspace, "attention_m", (rows, slots), q.device, torch.float32)
    partial_l = _buffer(workspace, "attention_l", (rows, slots), q.device, torch.float32)
    mask = _buffer(workspace, "head_mask", (heads, blocks), q.device, torch.int8)
    use_anchor = anchor_mask is not None
    if use_anchor:
        anchor_mask = anchor_mask.contiguous()
    else:
        anchor_mask = _buffer(workspace, "anchor_dummy", (1,), q.device, torch.bool)
    finalize_representatives_kernel[(heads, rep_programs)](
        v_bar_ptr=summary.value_mean,
        s_cache_ptr=s,
        delta_cache_ptr=delta,
        v_norm_ptr=summary.value_norm,
        v_norm_all_ptr=summary.max_value_norm,
        counts_ptr=summary.counts,
        z_m_part_ptr=selection_partial[0],
        z_l_part_ptr=selection_partial[1],
        c_m_part_ptr=selection_partial[2],
        c_l_part_ptr=selection_partial[3],
        anchor_mask_ptr=anchor_mask,
        mask_out_ptr=mask,
        partial_acc_ptr=partial_acc,
        partial_m_ptr=partial_m,
        partial_l_ptr=partial_l,
        n_blocks=blocks,
        n_sel_chunks=selection_programs,
        span=span,
        num_ptot=slots,
        group_size=groups,
        head_dim=dim,
        eps=config.eps,
        BLOCK_M=16,
        BLOCK_N=_BLOCK,
        BLOCK_D=triton.next_power_of_2(dim),
        BLOCK_SC=triton.next_power_of_2(selection_programs),
        NUM_STAGES=1,
        USE_ANCHOR=use_anchor,
        GQA_REDUCTION=_REDUCTION_CODES[config.gqa_reduction],
        num_warps=4,
    )
    if key_suffix.stride(-1) != 1:
        key_suffix = key_suffix.contiguous()
    if value_suffix.stride(-1) != 1:
        value_suffix = value_suffix.contiguous()
    if suffix_length is None:
        suffix_length = _buffer(workspace, "suffix_length", (), q.device, torch.int32)
        suffix_length.fill_(key_suffix.shape[1])
    finalize_pages_kernel[(heads, page_programs)](
        q_ptr=q,
        k_block_ptr=summary.keys,
        v_block_ptr=summary.values,
        mask_ptr=mask,
        k_suffix_ptr=key_suffix,
        v_suffix_ptr=value_suffix,
        partial_acc_ptr=partial_acc,
        partial_m_ptr=partial_m,
        partial_l_ptr=partial_l,
        n_blocks=blocks,
        num_pa=rep_programs,
        num_pb=page_programs,
        num_ptot=slots,
        suffix_len_ptr=suffix_length,
        k_block_head_stride=summary.keys.stride(0),
        v_block_head_stride=summary.values.stride(0),
        k_suffix_head_stride=key_suffix.stride(0),
        k_suffix_token_stride=key_suffix.stride(1),
        v_suffix_head_stride=value_suffix.stride(0),
        v_suffix_token_stride=value_suffix.stride(1),
        group_size=groups,
        head_dim=dim,
        scale=dim**-0.5,
        BLOCK_M=16,
        BLOCK_N=_BLOCK,
        BLOCK_D=triton.next_power_of_2(dim),
        MAX_ITER=triton.next_power_of_2(max(triton.cdiv(blocks, page_programs), 2)),
        PAGE_SIZE=summary.block_size,
        num_warps=4,
    )
    output = _buffer(workspace, "output", (rows, dim), q.device, query.dtype)
    attention_reduce_kernel[(rows, triton.cdiv(dim, 32))](
        partial_acc,
        partial_m,
        partial_l,
        output,
        slots,
        dim,
        BLOCK_C=triton.next_power_of_2(slots),
        D_TILE=32,
        num_warps=2,
    )
    return AttentionResult(
        output.reshape(heads, groups, 1, dim), mask.bool() if return_selected else None
    )
