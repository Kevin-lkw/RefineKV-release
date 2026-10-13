"""Stable diagonal-ellipsoid selection kernels."""

import triton
import triton.language as tl

_DIRECT_DELTA_MAX = tl.constexpr(10.0)
_LOG_TWO = tl.constexpr(0.6931471805599453)


@triton.jit
def _finite(value):
    return (value == value) & (tl.abs(value) < float("inf"))


@triton.jit
def _log_cosh(delta):
    """Stable log(cosh(delta)); delta is a nonnegative radius bound."""
    clipped = tl.minimum(delta, _DIRECT_DELTA_MAX)
    direct = tl.log(0.5 * (tl.exp(clipped) + tl.exp(-clipped)))
    large = delta + tl.log(0.5 * (1.0 + tl.exp(-2.0 * delta)))
    return tl.where(delta <= _DIRECT_DELTA_MAX, direct, large)


@triton.jit
def _condition_terms(z, delta, z_m, z_l, c_m, c_l, b_c, b_all):
    """Return the two nonnegative selection-condition terms."""
    clipped = tl.minimum(delta, _DIRECT_DELTA_MAX)
    t = tl.exp(clipped)
    tanh_direct = (t - 1.0) / (t + 1.0)
    exp_neg = tl.exp(-delta)
    tanh_large = (1.0 - exp_neg) / (1.0 + exp_neg)
    tanh_half = tl.where(delta <= _DIRECT_DELTA_MAX, tanh_direct, tanh_large)
    cosh_minus_one = 0.5 * (t + 1.0 / t) - 1.0
    term1_direct = 2.0 * b_all * tl.exp(z - c_m[:, None]) * cosh_minus_one / c_l[:, None]
    one_minus_exp_neg = tl.maximum(1.0 - exp_neg, 1e-30)
    log_cosh_minus_one = delta + 2.0 * tl.log(one_minus_exp_neg) - _LOG_TWO
    term1_large = 2.0 * b_all * tl.exp(z - c_m[:, None] + log_cosh_minus_one) / c_l[:, None]
    term1 = tl.where(delta <= _DIRECT_DELTA_MAX, term1_direct, term1_large)
    term2 = 2.0 * b_c[None, :] * tl.exp(z - z_m[:, None]) * tanh_half / z_l[:, None]
    return (term1, term2)


@triton.jit(
    do_not_specialize=["n_blocks", "span", "num_p"],
    do_not_specialize_on_alignment=["n_blocks", "span", "num_p"],
)
def selection_stats_kernel(
    q_ptr,
    k_bar_ptr,
    w2_ptr,
    counts_ptr,
    s_cache_ptr,
    delta_cache_ptr,
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    n_blocks,
    span,
    num_p,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    delta_scale: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    kv_head = tl.program_id(0)
    p = tl.program_id(1)
    g = tl.arange(0, BLOCK_G)
    d = tl.arange(0, BLOCK_D)
    g_mask = g < group_size
    d_mask = d < head_dim
    row = kv_head * group_size + g
    q_bf = tl.load(
        q_ptr + row[:, None] * head_dim + d[None, :],
        mask=g_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    q_f32 = q_bf.to(tl.float32)
    q2_bf = (q_f32 * q_f32).to(tl.bfloat16)
    run_zm = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    run_zl = tl.zeros((BLOCK_G,), tl.float32)
    run_cm = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    run_cl = tl.zeros((BLOCK_G,), tl.float32)
    for i in tl.range(0, span, num_stages=NUM_STAGES):
        chunk = p * span + i
        b = chunk * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b < n_blocks
        count = tl.load(counts_ptr + b, mask=b_mask, other=0)
        active = b_mask & (count > 0)
        tile_off = (kv_head * n_blocks + b[None, :]) * head_dim + d[:, None]
        tile_mask = active[None, :] & d_mask[:, None]
        k_bar = tl.load(k_bar_ptr + tile_off, mask=tile_mask, other=0.0)
        w2 = tl.load(w2_ptr + tile_off, mask=tile_mask, other=0.0)
        s = tl.dot(q_bf, k_bar) * scale
        qw2 = tl.dot(q2_bf, w2)
        delta = tl.sqrt(tl.maximum(qw2, 0.0)) * delta_scale
        z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
        active_2d = g_mask[:, None] & active[None, :]
        z = tl.where(active_2d, z, -float("inf"))
        zc = z + _log_cosh(delta)
        delta_store = delta
        tl.store(
            s_cache_ptr + row[:, None] * n_blocks + b[None, :],
            s.to(s_cache_ptr.dtype.element_ty),
            mask=active_2d,
        )
        tl.store(
            delta_cache_ptr + row[:, None] * n_blocks + b[None, :],
            delta_store.to(delta_cache_ptr.dtype.element_ty),
            mask=active_2d,
        )
        new_zm = tl.maximum(run_zm, tl.max(z, axis=1))
        zm_safe = tl.where(new_zm > -float("inf"), new_zm, 0.0)
        run_zl = run_zl * tl.exp(run_zm - zm_safe) + tl.sum(tl.exp(z - zm_safe[:, None]), axis=1)
        run_zm = new_zm
        new_cm = tl.maximum(run_cm, tl.max(zc, axis=1))
        cm_safe = tl.where(new_cm > -float("inf"), new_cm, 0.0)
        run_cl = run_cl * tl.exp(run_cm - cm_safe) + tl.sum(tl.exp(zc - cm_safe[:, None]), axis=1)
        run_cm = new_cm
    partial_off = row * num_p + p
    tl.store(z_m_ptr + partial_off, run_zm, mask=g_mask)
    tl.store(z_l_ptr + partial_off, run_zl, mask=g_mask)
    tl.store(c_m_ptr + partial_off, run_cm, mask=g_mask)
    tl.store(c_l_ptr + partial_off, run_cl, mask=g_mask)


@triton.jit(do_not_specialize=["n_chunks"], do_not_specialize_on_alignment=["n_chunks"])
def selection_reduce_kernel(
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    global_z_m_ptr,
    global_z_l_ptr,
    global_c_m_ptr,
    global_c_l_ptr,
    n_chunks,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    mask = c < n_chunks
    z_m = tl.load(z_m_ptr + row * n_chunks + c, mask=mask, other=-float("inf"))
    z_l = tl.load(z_l_ptr + row * n_chunks + c, mask=mask, other=0.0)
    c_m = tl.load(c_m_ptr + row * n_chunks + c, mask=mask, other=-float("inf"))
    c_l = tl.load(c_l_ptr + row * n_chunks + c, mask=mask, other=0.0)
    global_z_m = tl.max(z_m, axis=0)
    global_c_m = tl.max(c_m, axis=0)
    global_z_l = tl.sum(z_l * tl.exp(z_m - global_z_m), axis=0)
    global_c_l = tl.sum(c_l * tl.exp(c_m - global_c_m), axis=0)
    tl.store(global_z_m_ptr + row, global_z_m)
    tl.store(global_z_l_ptr + row, global_z_l)
    tl.store(global_c_m_ptr + row, global_c_m)
    tl.store(global_c_l_ptr + row, global_c_l)


@triton.jit(do_not_specialize=["n_blocks"], do_not_specialize_on_alignment=["n_blocks"])
def selection_finalize_kernel(
    s_cache_ptr,
    delta_cache_ptr,
    v_norm_ptr,
    v_norm_all_ptr,
    counts_ptr,
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    selected_ptr,
    z_out_ptr,
    n_blocks,
    group_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_B: tl.constexpr,
    GQA_REDUCTION: tl.constexpr,
):
    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)
    g = tl.arange(0, BLOCK_G)
    b = chunk * BLOCK_B + tl.arange(0, BLOCK_B)
    g_mask = g < group_size
    b_mask = b < n_blocks
    row = kv_head * group_size + g
    count = tl.load(counts_ptr + b, mask=b_mask, other=0)
    active = b_mask & (count > 0)
    mask_2d = g_mask[:, None] & active[None, :]
    s = tl.load(s_cache_ptr + row[:, None] * n_blocks + b[None, :], mask=mask_2d, other=0.0).to(
        tl.float32
    )
    delta = tl.load(
        delta_cache_ptr + row[:, None] * n_blocks + b[None, :], mask=mask_2d, other=0.0
    ).to(tl.float32)
    z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    z_m = tl.load(z_m_ptr + row, mask=g_mask, other=0.0).to(tl.float32)
    z_l = tl.load(z_l_ptr + row, mask=g_mask, other=1.0).to(tl.float32)
    c_m = tl.load(c_m_ptr + row, mask=g_mask, other=0.0).to(tl.float32)
    c_l = tl.load(c_l_ptr + row, mask=g_mask, other=1.0).to(tl.float32)
    b_c = tl.load(v_norm_ptr + kv_head * n_blocks + b, mask=active, other=0.0).to(tl.float32)
    b_all = tl.load(v_norm_all_ptr + kv_head).to(tl.float32)
    term1, term2 = _condition_terms(z, delta, z_m, z_l, c_m, c_l, b_c, b_all)
    condition = tl.where(mask_2d, term1 + term2, 0.0)
    if GQA_REDUCTION == 0:
        selected = tl.sum(condition, axis=0) / group_size > eps
    elif GQA_REDUCTION == 1:
        selected = tl.max(condition, axis=0) > eps
    else:
        selected = tl.sum((condition > eps).to(tl.int32), axis=0) > 0
    stats_finite = _finite(z_m) & _finite(z_l) & _finite(c_m) & _finite(c_l)
    bad_global = tl.sum((g_mask & ~stats_finite).to(tl.int32), axis=0) > 0
    bad_local = tl.sum((mask_2d & ~_finite(condition)).to(tl.int32), axis=0) > 0
    selected = (selected | bad_global | bad_local) & active
    tl.store(
        selected_ptr + row[:, None] * n_blocks + b[None, :],
        selected[None, :],
        mask=g_mask[:, None] & b_mask[None, :],
    )
    tl.store(
        z_out_ptr + row[:, None] * n_blocks + b[None, :], z, mask=g_mask[:, None] & b_mask[None, :]
    )
