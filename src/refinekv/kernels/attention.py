"""Split representative/page attention with online-softmax reduction."""

import triton
import triton.language as tl

from .selection import _condition_terms, _finite


@triton.jit(
    do_not_specialize=["n_blocks", "n_sel_chunks", "span", "num_ptot"],
    do_not_specialize_on_alignment=["n_blocks", "n_sel_chunks", "span", "num_ptot"],
)
def finalize_representatives_kernel(
    v_bar_ptr,
    s_cache_ptr,
    delta_cache_ptr,
    v_norm_ptr,
    v_norm_all_ptr,
    counts_ptr,
    z_m_part_ptr,
    z_l_part_ptr,
    c_m_part_ptr,
    c_l_part_ptr,
    anchor_mask_ptr,
    mask_out_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    n_blocks,
    n_sel_chunks,
    span,
    num_ptot,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SC: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_ANCHOR: tl.constexpr,
    GQA_REDUCTION: tl.constexpr,
):
    kv_head = tl.program_id(0)
    p = tl.program_id(1)
    m_off = tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    m_mask = m_off < group_size
    d_mask = d_off < head_dim
    row = kv_head * group_size + m_off
    sc_off = tl.arange(0, BLOCK_SC)
    sc_mask = sc_off < n_sel_chunks
    part_off = row[:, None] * n_sel_chunks + sc_off[None, :]
    part_mask = m_mask[:, None] & sc_mask[None, :]
    z_m_part = tl.load(z_m_part_ptr + part_off, mask=part_mask, other=-float("inf"))
    z_l_part = tl.load(z_l_part_ptr + part_off, mask=part_mask, other=0.0)
    c_m_part = tl.load(c_m_part_ptr + part_off, mask=part_mask, other=-float("inf"))
    c_l_part = tl.load(c_l_part_ptr + part_off, mask=part_mask, other=0.0)
    z_m = tl.max(z_m_part, axis=1)
    c_m = tl.max(c_m_part, axis=1)
    z_l = tl.sum(z_l_part * tl.exp(z_m_part - z_m[:, None]), axis=1)
    c_l = tl.sum(c_l_part * tl.exp(c_m_part - c_m[:, None]), axis=1)
    z_m = tl.where(m_mask, z_m, 0.0)
    z_l = tl.where(m_mask, z_l, 1.0)
    c_m = tl.where(m_mask, c_m, 0.0)
    c_l = tl.where(m_mask, c_l, 1.0)
    stats_finite = _finite(z_m) & _finite(z_l) & _finite(c_m) & _finite(c_l)
    bad_global = tl.sum((m_mask & ~stats_finite).to(tl.int32), axis=0) > 0
    b_all = tl.load(v_norm_all_ptr + kv_head).to(tl.float32)
    softmax_m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    softmax_l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for i in tl.range(0, span, num_stages=NUM_STAGES):
        chunk = p * span + i
        block_idx = chunk * BLOCK_N + n_off
        block_mask = block_idx < n_blocks
        count = tl.load(counts_ptr + block_idx, mask=block_mask, other=0)
        active_block = block_mask & (count > 0)
        active_2d = m_mask[:, None] & active_block[None, :]
        s = tl.load(
            s_cache_ptr + row[:, None] * n_blocks + block_idx[None, :], mask=active_2d, other=0.0
        ).to(tl.float32)
        delta = tl.load(
            delta_cache_ptr + row[:, None] * n_blocks + block_idx[None, :],
            mask=active_2d,
            other=0.0,
        ).to(tl.float32)
        z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
        b_c = tl.load(v_norm_ptr + kv_head * n_blocks + block_idx, mask=active_block, other=0.0).to(
            tl.float32
        )
        term1, term2 = _condition_terms(z, delta, z_m, z_l, c_m, c_l, b_c, b_all)
        condition = tl.where(active_2d, term1 + term2, 0.0)
        if GQA_REDUCTION == 0:
            selected = tl.sum(condition, axis=0) / group_size > eps
        elif GQA_REDUCTION == 1:
            selected = tl.max(condition, axis=0) > eps
        else:
            selected = tl.sum((condition > eps).to(tl.int32), axis=0) > 0
        bad_local = tl.sum((active_2d & ~_finite(condition)).to(tl.int32), axis=0) > 0
        selected = selected | bad_global | bad_local
        if USE_ANCHOR:
            anchor_selected = tl.load(
                anchor_mask_ptr + kv_head * n_blocks + block_idx, mask=block_mask, other=0
            ).to(tl.int1)
            selected = selected | anchor_selected
        selected = selected & active_block
        tl.store(
            mask_out_ptr + kv_head * n_blocks + block_idx,
            tl.where(selected, count, 0).to(tl.int8),
            mask=block_mask,
        )
        active_rep = active_block & ~selected
        rep_scores = tl.where(m_mask[:, None] & active_rep[None, :], z, -float("inf"))
        has_rep = tl.sum(active_rep.to(tl.int32), axis=0) > 0
        rep_m = tl.max(rep_scores, axis=1)
        new_m = tl.where(has_rep & m_mask, tl.maximum(softmax_m, rep_m), softmax_m)
        alpha = tl.where(has_rep & m_mask, tl.exp(softmax_m - new_m), 1.0)
        beta = tl.where(
            m_mask[:, None] & active_rep[None, :], tl.exp(rep_scores - new_m[:, None]), 0.0
        )
        rep_off = (kv_head * n_blocks + block_idx[:, None]) * head_dim + d_off[None, :]
        v_rep = tl.load(
            v_bar_ptr + rep_off, mask=active_rep[:, None] & d_mask[None, :], other=0.0
        ).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(beta.to(tl.bfloat16), v_rep, out_dtype=tl.float32)
        softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
        softmax_m = new_m
    partial_off = row * num_ptot + p
    tl.store(partial_m_ptr + partial_off, softmax_m, mask=m_mask)
    tl.store(partial_l_ptr + partial_off, softmax_l, mask=m_mask)
    tl.store(
        partial_acc_ptr + partial_off[:, None] * head_dim + d_off[None, :],
        acc,
        mask=m_mask[:, None] & d_mask[None, :],
    )


@triton.jit(
    do_not_specialize=[
        "n_blocks",
        "num_pa",
        "num_pb",
        "num_ptot",
        "k_block_head_stride",
        "v_block_head_stride",
        "k_suffix_head_stride",
        "k_suffix_token_stride",
        "v_suffix_head_stride",
        "v_suffix_token_stride",
    ],
    do_not_specialize_on_alignment=[
        "n_blocks",
        "num_pa",
        "num_pb",
        "num_ptot",
        "k_block_head_stride",
        "v_block_head_stride",
        "k_suffix_head_stride",
        "k_suffix_token_stride",
        "v_suffix_head_stride",
        "v_suffix_token_stride",
    ],
)
def finalize_pages_kernel(
    q_ptr,
    k_block_ptr,
    v_block_ptr,
    mask_ptr,
    k_suffix_ptr,
    v_suffix_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    n_blocks,
    num_pa,
    num_pb,
    num_ptot,
    suffix_len_ptr,
    k_block_head_stride,
    v_block_head_stride,
    k_suffix_head_stride,
    k_suffix_token_stride,
    v_suffix_head_stride,
    v_suffix_token_stride,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_ITER: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    kv_head = tl.program_id(0)
    p = tl.program_id(1)
    m_off = tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    p_off = tl.arange(0, PAGE_SIZE)
    i_off = tl.arange(0, MAX_ITER)
    m_mask = m_off < group_size
    d_mask = d_off < head_dim
    row = kv_head * group_size + m_off
    q = tl.load(
        q_ptr + row[:, None] * head_dim + d_off[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    cand = p + i_off * num_pb
    cand_mask = cand < n_blocks
    sel = tl.load(mask_ptr + kv_head * n_blocks + cand, mask=cand_mask, other=0)
    sel_bool = (sel > 0) & cand_mask
    sel_cum = tl.cumsum(sel_bool.to(tl.int32), axis=0)
    n_my = tl.sum(sel_bool.to(tl.int32), axis=0)
    softmax_m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    softmax_l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for sel_j in tl.range(0, n_my, num_stages=2):
        hit = (sel_cum == sel_j + 1) & sel_bool
        page = tl.sum(tl.where(hit, cand, 0), axis=0)
        page_count = tl.sum(tl.where(hit, sel.to(tl.int32), 0), axis=0)
        token_active = p_off < page_count
        token_off = (page * PAGE_SIZE + p_off[:, None]) * head_dim + d_off[None, :]
        page_k = tl.load(
            k_block_ptr + kv_head * k_block_head_stride + token_off,
            mask=token_active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        page_v = tl.load(
            v_block_ptr + kv_head * v_block_head_stride + token_off,
            mask=token_active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        page_scores = tl.dot(q, tl.trans(page_k), out_dtype=tl.float32) * scale
        page_scores = tl.where(m_mask[:, None] & token_active[None, :], page_scores, -float("inf"))
        page_m = tl.max(page_scores, axis=1)
        page_new_m = tl.where(m_mask, tl.maximum(softmax_m, page_m), softmax_m)
        page_alpha = tl.where(m_mask, tl.exp(softmax_m - page_new_m), 1.0)
        page_beta = tl.where(
            m_mask[:, None] & token_active[None, :], tl.exp(page_scores - page_new_m[:, None]), 0.0
        )
        acc = acc * page_alpha[:, None] + tl.dot(
            page_beta.to(tl.bfloat16), page_v, out_dtype=tl.float32
        )
        softmax_l = softmax_l * page_alpha + tl.sum(page_beta, axis=1)
        softmax_m = page_new_m
    suffix_len = tl.load(suffix_len_ptr)
    suffix_start = p * BLOCK_N
    while suffix_start < suffix_len:
        suffix_idx = suffix_start + n_off
        suffix_active = suffix_idx < suffix_len
        k_suffix_off = (
            kv_head * k_suffix_head_stride
            + suffix_idx[:, None] * k_suffix_token_stride
            + d_off[None, :]
        )
        k = tl.load(
            k_suffix_ptr + k_suffix_off, mask=suffix_active[:, None] & d_mask[None, :], other=0.0
        ).to(tl.bfloat16)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        scores = tl.where(m_mask[:, None] & suffix_active[None, :], scores, -float("inf"))
        tile_m = tl.max(scores, axis=1)
        new_m = tl.where(m_mask, tl.maximum(softmax_m, tile_m), softmax_m)
        alpha = tl.where(m_mask, tl.exp(softmax_m - new_m), 1.0)
        beta = tl.where(
            m_mask[:, None] & suffix_active[None, :], tl.exp(scores - new_m[:, None]), 0.0
        )
        v_suffix_off = (
            kv_head * v_suffix_head_stride
            + suffix_idx[:, None] * v_suffix_token_stride
            + d_off[None, :]
        )
        v = tl.load(
            v_suffix_ptr + v_suffix_off, mask=suffix_active[:, None] & d_mask[None, :], other=0.0
        ).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(beta.to(tl.bfloat16), v, out_dtype=tl.float32)
        softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
        softmax_m = new_m
        suffix_start += num_pb * BLOCK_N
    partial_off = row * num_ptot + (num_pa + p)
    tl.store(partial_m_ptr + partial_off, softmax_m, mask=m_mask)
    tl.store(partial_l_ptr + partial_off, softmax_l, mask=m_mask)
    did_work = (n_my > 0) | (p * BLOCK_N < suffix_len)
    tl.store(
        partial_acc_ptr + partial_off[:, None] * head_dim + d_off[None, :],
        acc,
        mask=m_mask[:, None] & d_mask[None, :] & did_work,
    )


@triton.jit(
    do_not_specialize=["n_slots", "head_dim_rt"],
    do_not_specialize_on_alignment=["n_slots", "head_dim_rt"],
)
def attention_reduce_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    out_ptr,
    n_slots,
    head_dim_rt,
    BLOCK_C: tl.constexpr,
    D_TILE: tl.constexpr,
):
    """Core's stage2 reduce with a (rows, d-tiles) grid instead of (rows,):
    the original's 32 programs leave the machine idle; the m/l merge is
    replicated per tile (tiny) and each program reduces one D_TILE slice."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < n_slots
    m = tl.load(partial_m_ptr + row * n_slots + c, mask=c_mask, other=-float("inf"))
    l = tl.load(partial_l_ptr + row * n_slots + c, mask=c_mask, other=0.0)
    gm = tl.max(m, axis=0)
    gm_safe = tl.where(gm > -float("inf"), gm, 0.0)
    w = tl.where(c_mask, tl.exp(m - gm_safe), 0.0)
    denom = tl.sum(l * w, axis=0)
    d = tile * D_TILE + tl.arange(0, D_TILE)
    d_mask = d < head_dim_rt
    live = c_mask & (l > 0.0)
    acc = tl.load(
        partial_acc_ptr + (row * n_slots + c)[:, None] * head_dim_rt + d[None, :],
        mask=live[:, None] & d_mask[None, :],
        other=0.0,
    )
    out = tl.sum(acc * w[:, None], axis=0) / tl.maximum(denom, 1e-30)
    tl.store(out_ptr + row * head_dim_rt + d, out.to(out_ptr.dtype.element_ty), mask=d_mask)
