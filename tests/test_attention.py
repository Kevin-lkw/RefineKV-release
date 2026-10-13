from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from refinekv import RefineKVConfig, build_summary, select_blocks, sparse_attention


def _inputs(device="cpu", length=71, dim=64):
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(shape):
        return torch.randn(shape, device=device, generator=generator, dtype=torch.bfloat16)

    return normal((2, 4, 1, dim)), normal((2, length, dim)), normal((2, length, dim))


@pytest.mark.parametrize("block", [16, 32, 64])
def test_partial_page_summaries_ignore_padding_and_preserve_layout(block):
    _, keys, values = _inputs(length=block + 3)
    # A noncontiguous prefix of a larger cache must remain kernel-addressable.
    keys = torch.cat([keys, torch.zeros_like(keys)], 1)[:, : block + 3]
    config = RefineKVConfig(block_size=block, summary_chunk_blocks=1)
    summary = build_summary(keys, values, config)
    assert summary.counts.tolist() == [block, 3]
    torch.testing.assert_close(
        summary.key_mean[:, -1], keys[:, -3:].float().mean(1).to(torch.bfloat16)
    )
    torch.testing.assert_close(
        summary.value_mean[:, -1], values[:, -3:].float().mean(1).to(torch.bfloat16)
    )
    assert summary.keys.stride(2) == keys.shape[-1]
    assert summary.keys.stride(3) == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [64, 67])
def test_summary_preserves_input_cache_and_exact_keys(dtype, length):
    generator = torch.Generator().manual_seed(17)
    key_cache = torch.randn(2, 96, 64, generator=generator).to(dtype)
    value_cache = torch.randn(2, 96, 64, generator=generator).to(dtype)
    original_keys, original_values = key_cache.clone(), value_cache.clone()
    keys, values = key_cache[:, :length], value_cache[:, :length]
    config = RefineKVConfig(backend="torch", summary_chunk_blocks=1)
    summary = build_summary(keys, values, config)
    assert torch.equal(key_cache, original_keys)
    assert torch.equal(value_cache, original_values)
    torch.testing.assert_close(
        summary.keys.reshape(2, -1, 64)[:, :length], original_keys[:, :length],
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        summary.values.reshape(2, -1, 64)[:, :length], original_values[:, :length],
        rtol=0, atol=0,
    )


def test_centered_ellipsoid_handles_constant_axes_and_bounds_sampled_queries():
    q, keys, values = _inputs(length=65, dim=128)
    keys[..., :3] = torch.tensor([96, 48, -80], dtype=torch.bfloat16)
    summary = build_summary(keys, values, RefineKVConfig(summary_chunk_blocks=1))
    assert torch.isfinite(summary.ellipsoid).all()
    centered = summary.keys.float() - summary.key_mean.float().unsqueeze(2)
    exact = torch.einsum("hgd,hbtd->hgbt", q[:, :, 0].float(), centered).abs()
    valid = torch.arange(summary.block_size)[None] < summary.counts[:, None]
    exact = exact.masked_fill(~valid[None, None], 0).amax(-1)
    bound = torch.einsum(
        "hgd,hbd->hgb", q[:, :, 0].float().square(), summary.ellipsoid.float()
    ).sqrt()
    assert (exact <= bound + 1e-5).all()


@pytest.mark.parametrize("suffix_length", [0, 5])
def test_full_expansion_matches_dense_attention(suffix_length):
    q, keys, values = _inputs()
    config = RefineKVConfig(backend="torch", eps=0)
    summary = build_summary(keys, values, config)
    result = sparse_attention(
        q, summary, keys[:, :suffix_length], values[:, :suffix_length], config, return_selected=True
    )
    assert result.selected.all()
    exact_keys = torch.cat([keys, keys[:, :suffix_length]], 1).repeat_interleave(4, 0)
    exact_values = torch.cat([values, values[:, :suffix_length]], 1).repeat_interleave(4, 0)
    expected = F.scaled_dot_product_attention(
        q.reshape(8, 1, 64).float(), exact_keys.float(), exact_values.float()
    )
    torch.testing.assert_close(
        result.output.float().reshape(8, 1, 64), expected, atol=0.004, rtol=0.02
    )


def test_representative_multiplicity_uses_actual_partial_page_count():
    config = RefineKVConfig(backend="torch", eps=1e6, block_size=16)
    keys = torch.zeros(1, 19, 64, dtype=torch.bfloat16)
    values = torch.zeros_like(keys)
    values[:, :16] = 1
    values[:, 16:] = 3
    summary = build_summary(keys, values, config)
    q = torch.zeros(1, 2, 1, 64)
    result = sparse_attention(q, summary, keys[:, :0], values[:, :0], config, return_selected=True)
    assert not result.selected.any()
    torch.testing.assert_close(result.output, torch.full_like(result.output, (16 + 9) / 19))


def test_anchor_union_expands_only_requested_pages():
    q, keys, values = _inputs()
    config = RefineKVConfig(backend="torch", eps=1e6)
    summary = build_summary(keys, values, config)
    anchor = torch.zeros(2, summary.num_blocks, dtype=torch.bool)
    anchor[:, -1] = True
    result = sparse_attention(
        q, summary, keys[:, :0], values[:, :0], config, anchor_mask=anchor, return_selected=True
    )
    assert torch.equal(result.selected, anchor)
    assert torch.isfinite(result.output).all()


def test_large_finite_delta_stays_finite():
    q, keys, values = _inputs()
    q = (q.float() * 200).to(torch.bfloat16)
    summary = build_summary(keys, values)
    selected = select_blocks(q, summary, RefineKVConfig(backend="torch"))
    result = sparse_attention(
        q, summary, keys[:, :0], values[:, :0], RefineKVConfig(backend="torch")
    )
    assert selected.shape == (2, summary.num_blocks)
    assert torch.isfinite(result.output).all()


@pytest.mark.parametrize(
    "kwargs", [{"eps": float("nan")}, {"eps": -1}, {"block_size": 10}, {"anchor_tokens": -1}]
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RefineKVConfig(**kwargs)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "block,reduction,eps,length,dim",
    [
        (16, "any", 0, 71, 64),
        (32, "mean", 0.1, 71, 128),
        (64, "max", 1e6, 71, 64),
        (16, "any", 0.2, 25001, 128),
    ],
)
def test_triton_matches_reference_and_reuses_workspace(block, reduction, eps, length, dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    q, keys, values = _inputs("cuda", length, dim)
    config = RefineKVConfig(backend="triton", block_size=block, eps=eps, gqa_reduction=reduction)
    summary = build_summary(keys, values, config)
    anchor = torch.zeros(2, summary.num_blocks, device="cuda", dtype=torch.bool)
    anchor[:, -1] = True
    workspace = {}
    selected = select_blocks(q, summary, config, workspace=workspace)
    expected_selected = select_blocks(q, summary, replace(config, backend="torch"))
    assert torch.equal(selected, expected_selected)
    for suffix_length in (0, 5):
        actual = sparse_attention(
            q,
            summary,
            keys[:, :suffix_length],
            values[:, :suffix_length],
            config,
            anchor_mask=anchor,
            workspace=workspace,
            return_selected=True,
        )
        expected = sparse_attention(
            q,
            summary,
            keys[:, :suffix_length],
            values[:, :suffix_length],
            replace(config, backend="torch"),
            anchor_mask=anchor,
            return_selected=True,
        )
        assert torch.equal(actual.selected, expected.selected)
        torch.testing.assert_close(actual.output, expected.output, atol=0.004, rtol=0.02)


@pytest.mark.cuda
def test_device_suffix_length_masks_fixed_capacity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    q, keys, values = _inputs("cuda", length=71, dim=128)
    config = RefineKVConfig(backend="triton", gqa_reduction="mean")
    summary = build_summary(keys, values, config)
    suffix_keys, suffix_values = _inputs("cuda", length=16, dim=128)[1:]
    visible = torch.zeros((), device="cuda", dtype=torch.int32)
    for count in (0, 1, 5, 16):
        visible.fill_(count)
        actual = sparse_attention(
            q, summary, suffix_keys, suffix_values, config, suffix_length=visible
        ).output.clone()
        expected = sparse_attention(
            q, summary, suffix_keys[:, :count], suffix_values[:, :count], config
        ).output.clone()
        assert torch.equal(actual, expected)
