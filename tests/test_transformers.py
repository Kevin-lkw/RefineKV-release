import pytest
import torch

transformers = pytest.importorskip("transformers")
from refinekv import RefineKV, RefineKVConfig  # noqa: E402

pytestmark = pytest.mark.transformers


def _model(family="llama", device="cpu"):
    torch.manual_seed(9)
    config_type, model_type = (
        (transformers.LlamaConfig, transformers.LlamaForCausalLM)
        if family == "llama"
        else (transformers.Qwen2Config, transformers.Qwen2ForCausalLM)
    )
    config = config_type(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=None,
    )
    model = model_type(config).eval().to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    return model


def _generate(model, length=39, static=False):
    ids = torch.arange(length, device=model.device)[None] % 61 + 3
    options = {"cache_implementation": "static"} if static else {}
    return model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=4,
        do_sample=False,
        **options,
    )


@pytest.mark.parametrize("family", ["llama", "qwen2"])
@pytest.mark.parametrize("static", [False, True])
def test_native_generation_with_anchors_and_request_reset(family, static):
    model = _model(family)
    original = model.config._attn_implementation
    original_hooks = len(model._forward_pre_hooks), len(model._forward_hooks)
    with RefineKV(
        model, RefineKVConfig(backend="torch", anchor_tokens=3, eps=1e6, collect_stats=True)
    ) as session:
        output = _generate(model, static=static)
        assert output.shape == (1, 43)
        if static:
            assert session._cache.layers[0].__class__.__name__ == "_VisibleStaticLayer"
        assert 0 < session.stats["equivalent_access_ratio"] <= 1
        assert session.stats["decode_attention_calls"] == 6
        assert all(state.anchor_mask is not None for state in session._layers.values())
        # A fresh generation in the same context must rebuild a different prompt.
        shorter = _generate(model, length=17, static=static)
        assert shorter.shape == (1, 21)
        assert session.stats["decode_attention_calls"] == 6
        assert all(state.prompt_length == 17 for state in session._layers.values())
    assert model.config._attn_implementation == original
    assert original_hooks == (len(model._forward_pre_hooks), len(model._forward_hooks))
    assert not session._layers
    assert not hasattr(model, "_refinekv_session")
    assert all(not hasattr(layer.self_attn, "_refinekv_session") for layer in model.model.layers)


def test_generation_config_static_uses_visible_cache():
    model = _model()
    model.generation_config.cache_implementation = "static"
    with RefineKV(model, RefineKVConfig(backend="torch")) as session:
        result = _generate(model)
        assert result.shape == (1, 43)
        assert session._cache.layers[0].__class__.__name__ == "_VisibleStaticLayer"


def test_stock_static_cache_is_rejected():
    model = _model()
    ids = torch.arange(39)[None] % 61 + 3
    cache = transformers.StaticCache(config=model.config, max_cache_len=43)
    with RefineKV(model):
        with pytest.raises(ValueError, match="stock StaticCache is unsupported"):
            model.generate(input_ids=ids, max_new_tokens=4, past_key_values=cache)


def test_full_layers_match_native_generation_and_other_models_are_untouched():
    model = _model()
    other = _model()
    with torch.inference_mode():
        expected = _generate(model)
    other_implementation = other.config._attn_implementation
    with RefineKV(model, RefineKVConfig(full_attention_layers=2, collect_stats=True)) as session:
        actual = _generate(model)
        assert other.config._attn_implementation == other_implementation
    assert torch.equal(actual, expected)
    assert session.stats["equivalent_access_ratio"] == 1


def test_exception_and_nested_context_restore_model():
    model = _model()
    original = model.config._attn_implementation
    with pytest.raises(ValueError, match="unpadded"):
        with RefineKV(model):
            with pytest.raises(RuntimeError, match="active"):
                with RefineKV(model):
                    pass
            model.generate(
                input_ids=torch.tensor([[0, 3, 4]]),
                attention_mask=torch.tensor([[0, 1, 1]]),
                max_new_tokens=2,
            )
    assert model.config._attn_implementation == original
    assert not hasattr(model, "_refinekv_session")


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_anchors_use_tail_projection_when_large_projection_rounds_differently(family, monkeypatch):
    model = _model(family)
    # A zero query has zero condition and selects no pages. Simulate a
    # length-dependent projection discrepancy only in the full prompt GEMM.
    for layer in model.model.layers:
        projection = layer.self_attn.q_proj
        with torch.no_grad():
            projection.weight.zero_()
            if projection.bias is not None:
                projection.bias.zero_()
        original = projection.forward

        def perturbed(hidden, original=original):
            result = original(hidden)
            return result + 1 if hidden.shape[1] > 2 else result

        monkeypatch.setattr(projection, "forward", perturbed)
    with RefineKV(model, RefineKVConfig(backend="torch", eps=0, anchor_tokens=2)) as session:
        _generate(model)
        assert all(not state.anchor_mask.any() for state in session._layers.values())


@pytest.mark.cuda
@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_cuda_native_generation_static_and_dynamic(family):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    model = _model(family, "cuda")
    config = RefineKVConfig(backend="triton", eps=1e6, anchor_tokens=2)
    with RefineKV(model, config):
        dynamic = _generate(model)
    with RefineKV(model, config):
        static = _generate(model, static=True)
    assert torch.equal(dynamic, static)


@pytest.mark.cuda
@pytest.mark.parametrize("family", ["llama", "qwen2"])
@pytest.mark.parametrize("reduction,anchors", [("mean", 0), ("any", 2)])
def test_cuda_graph_fixed_decode_matches_eager(family, reduction, anchors):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    model = _model(family, "cuda")
    if family == "qwen2":
        model.generation_config.repetition_penalty = 1.05
    config = RefineKVConfig(
        backend="triton", eps=0.01, gqa_reduction=reduction, anchor_tokens=anchors
    )
    requests = []
    for prompt_length in (39, 17):
        ids = torch.arange(prompt_length, device="cuda")[None] % 61 + 3
        options = dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=8,
            min_new_tokens=8,
            eos_token_id=None,
            forced_eos_token_id=None,
            do_sample=False,
        )
        with RefineKV(model, config):
            eager = model.generate(**options)
        requests.append((options, eager))
    with RefineKV(model, config, cuda_graph=True):
        for options, eager in requests:
            assert torch.equal(model.generate(**options), eager)


@pytest.mark.cuda
def test_cuda_graph_rejects_variable_output_and_stats():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    model = _model(device="cuda")
    ids = torch.tensor([[3, 4, 5]], device="cuda")
    with RefineKV(model, RefineKVConfig(backend="triton"), cuda_graph=True):
        with pytest.raises(ValueError, match="fixed output"):
            model.generate(input_ids=ids, max_new_tokens=8, eos_token_id=None)
    with RefineKV(model, RefineKVConfig(backend="triton", collect_stats=True), cuda_graph=True):
        with pytest.raises(ValueError, match="no stats"):
            model.generate(
                input_ids=ids, max_new_tokens=8, min_new_tokens=8, eos_token_id=None
            )
    model.generation_config.no_repeat_ngram_size = 2
    with RefineKV(model, RefineKVConfig(backend="triton"), cuda_graph=True):
        with pytest.raises(ValueError, match="logits processors"):
            model.generate(
                input_ids=ids, max_new_tokens=8, min_new_tokens=8, eos_token_id=None
            )
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.do_sample = True
    with RefineKV(model, RefineKVConfig(backend="triton"), cuda_graph=True):
        with pytest.raises(ValueError, match="greedy"):
            model.generate(
                input_ids=ids, max_new_tokens=8, min_new_tokens=8, eos_token_id=None
            )
