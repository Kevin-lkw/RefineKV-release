"""Scoped Transformers integration through its registered attention interface."""

from dataclasses import dataclass, field
from functools import wraps
import math

import torch

try:
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.cache_utils import Cache, StaticCache, StaticLayer
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
except ImportError as error:
    raise ImportError(
        "install RefineKV with the 'transformers' extra to use this adapter"
    ) from error

from ..attention import select_blocks, sparse_attention
from ..config import RefineKVConfig
from ..summary import BlockSummary, build_summary

_MISSING = object()


class _VisibleStaticLayer(StaticLayer):
    """Preallocate KV while exposing only tokens that have been written."""

    is_compileable = False

    def __init__(self, max_cache_len):
        super().__init__(max_cache_len)
        self.visible = 0

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.max_batch_size, self.num_heads = key_states.shape[:2]
        self.k_head_dim, self.v_head_dim = key_states.shape[-1], value_states.shape[-1]
        self.keys = torch.empty(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.k_head_dim),
            dtype=self.dtype, device=self.device,
        )
        self.values = torch.empty(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.v_head_dim),
            dtype=self.dtype, device=self.device,
        )
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        end = self.visible + key_states.shape[-2]
        if end > self.max_cache_len:
            raise ValueError("visible KV cache capacity exceeded")
        self.keys[:, :, self.visible:end].copy_(key_states)
        self.values[:, :, self.visible:end].copy_(value_states)
        self.visible = end
        return self.keys[:, :, :end], self.values[:, :, :end]

    def get_mask_sizes(self, cache_position):
        return self.visible + cache_position.shape[0], 0

    def get_seq_length(self):
        return self.visible


def make_visible_static_cache(model, max_cache_len):
    """Create an in-place cache with a mask-free visible prefix for prefill."""
    if not isinstance(max_cache_len, int) or max_cache_len <= 0:
        raise ValueError("max_cache_len must be a positive integer")
    return Cache(layers=[
        _VisibleStaticLayer(max_cache_len) for _ in model.model.layers
    ])


class _GraphStaticLayer(StaticLayer):
    """Fixed KV addresses with a prompt-only view before graph decode."""

    is_compileable = False

    def __init__(self, max_cache_len):
        super().__init__(max_cache_len)
        self.visible = 0
        self.graph_decode = False

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if not self.graph_decode:
            if self.visible or key_states.shape[-2] > self.max_cache_len:
                raise ValueError("graph cache requires one complete prompt prefill")
            end = key_states.shape[-2]
            self.keys[:, :, :end].copy_(key_states)
            self.values[:, :, :end].copy_(value_states)
            self.visible = end
            return self.keys[:, :, :end], self.values[:, :, :end]
        position = cache_kwargs["cache_position"]
        self.keys.index_copy_(2, position, key_states)
        self.values.index_copy_(2, position, value_states)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position):
        return (self.max_cache_len if self.graph_decode else cache_position.shape[0]), 0

    def get_seq_length(self):
        return self.visible


@dataclass
class _LayerState:
    prompt_length: int
    visible_length: int
    summary: BlockSummary | None = None
    anchor_mask: torch.Tensor | None = None
    workspace: dict = field(default_factory=dict)


def _attention_forward(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs
):
    session = getattr(module, "_refinekv_session", None)
    if session is None:
        raise RuntimeError("use the RefineKV attention interface inside a RefineKV context")
    return session._attention(module, query, key, value, attention_mask, dropout, scaling, kwargs)


def _register():
    # Add a named implementation; never replace Transformers' eager/SDPA functions.
    AttentionInterface.register("refinekv", _attention_forward)
    AttentionMaskInterface.register("refinekv", AttentionMaskInterface()["sdpa"])


class RefineKV:
    """Use RefineKV for dense prefill followed by single-sequence cached decode.

    Example::

        with RefineKV(model, RefineKVConfig(anchor_tokens=32)):
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    The model must be in evaluation mode. Llama and Qwen2/Qwen2.5 model
    families are supported. Each generation starts with an empty cache and
    one unpadded prompt. DynamicCache and a visible-prefix static cache are
    supported. State is scoped to this model and is released when the context exits. Set
    ``cuda_graph=True`` for fixed-length greedy CUDA decode with a private
    preallocated cache; the default is eager generation.
    """

    def __init__(self, model, config: RefineKVConfig | None = None, *, cuda_graph=False):
        self.model = model
        self.config = config or RefineKVConfig()
        self.cuda_graph = cuda_graph
        self._graph_mode = False
        self._graph_suffix_length = None
        self._layers = {}
        self._anchor_queries = {}
        self._handles = []
        self._cache = None
        self._active = False
        self._totals = dict(accessed=0, dense=0, decode_attention_calls=0)

    @property
    def stats(self):
        """Decode-only equivalent KV access, materialized on explicit request.

        An unexpanded page counts as one representative. This is an
        algorithmic access ratio, not physical memory usage or measured IO.
        Reading this property synchronizes CUDA counters when stats are on.
        """
        totals = {
            name: value.item() if torch.is_tensor(value) else value
            for name, value in self._totals.items()
        }
        totals["equivalent_access_ratio"] = (
            totals["accessed"] / totals["dense"] if totals["dense"] else None
        )
        return totals

    def __enter__(self):
        if self._active or hasattr(self.model, "_refinekv_session"):
            raise RuntimeError("this model already has an active RefineKV context")
        if self.model.training:
            raise ValueError("RefineKV is inference-only; call model.eval() first")
        if self.model.config.model_type not in ("llama", "qwen2"):
            raise ValueError("the Transformers adapter supports model_type='llama' or 'qwen2'")
        if getattr(self.model.config, "use_sliding_window", False):
            raise ValueError("sliding-window attention is not supported")
        if getattr(self.model.config, "output_attentions", False):
            raise ValueError("RefineKV does not return attention weights")
        self._modules = list(self.model.model.layers)
        if any(
            getattr(layer.self_attn, "sliding_window", None) is not None for layer in self._modules
        ):
            raise ValueError("sliding-window attention is not supported")
        _register()
        self._original_implementation = self.model.config._attn_implementation
        self._original_layer_implementations = [
            layer.self_attn.config._attn_implementation for layer in self._modules
        ]
        self._inference = torch.inference_mode()
        self._inference.__enter__()
        try:
            self.model.set_attn_implementation("refinekv")
            self.model._refinekv_session = self
            self._module_indices = {}
            for index, layer in enumerate(self._modules):
                layer.self_attn._refinekv_session = self
                # Map module identities instead of modifying architecture-specific layer indices.
                self._module_indices[id(layer.self_attn)] = index
                if self.config.anchor_tokens and index >= self.config.full_attention_layers:
                    self._handles.append(
                        layer.self_attn.register_forward_pre_hook(
                            self._capture_anchor_queries, with_kwargs=True
                        )
                    )
            self._original_generate_attribute = self.model.__dict__.get("generate", _MISSING)
            original_generate = self.model.generate

            @wraps(original_generate)
            def generate(*args, **kwargs):
                self._layers.clear()
                self._anchor_queries.clear()
                self._cache = None
                self._totals = dict(accessed=0, dense=0, decode_attention_calls=0)
                if self.cuda_graph:
                    return self._generate_cuda_graph(*args, **kwargs)
                if isinstance(kwargs.get("past_key_values"), StaticCache):
                    raise ValueError(
                        "stock StaticCache is unsupported; use cache_implementation='static' "
                        "or make_visible_static_cache"
                    )
                generation = kwargs.get("generation_config") or self.model.generation_config
                cache_implementation = kwargs.get(
                    "cache_implementation", generation.cache_implementation
                )
                if cache_implementation == "static":
                    if kwargs.get("past_key_values") is not None:
                        raise ValueError("pass either a static cache request or past_key_values")
                    inputs = kwargs.get("input_ids")
                    if inputs is None and args:
                        inputs = args[0]
                    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
                        raise ValueError("static cache generation requires input_ids of shape (1, tokens)")
                    prompt_length = inputs.shape[1]
                    max_new_tokens = kwargs.get("max_new_tokens", generation.max_new_tokens)
                    if max_new_tokens is not None:
                        capacity = prompt_length + int(max_new_tokens)
                    else:
                        capacity = int(kwargs.get("max_length", generation.max_length))
                    if capacity <= prompt_length:
                        raise ValueError("static cache capacity must exceed prompt length")
                    kwargs["past_key_values"] = make_visible_static_cache(self.model, capacity)
                    # An explicit cache and cache_implementation cannot coexist in generate.
                    kwargs["cache_implementation"] = None
                # Summary construction must run outside compiled model forwards.
                kwargs["disable_compile"] = True
                return original_generate(*args, **kwargs)

            self.model.generate = generate
            self._handles.append(
                self.model.register_forward_pre_hook(self._before_forward, with_kwargs=True)
            )
            self._handles.append(
                self.model.register_forward_hook(self._after_forward, with_kwargs=True)
            )
            self._active = True
            return self
        except BaseException:
            self._restore()
            self._inference.__exit__(None, None, None)
            raise

    def _restore(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for layer, original in zip(self._modules, self._original_layer_implementations):
            if getattr(layer.self_attn, "_refinekv_session", None) is self:
                del layer.self_attn._refinekv_session
            layer.self_attn.config._attn_implementation = original
        if getattr(self.model, "_refinekv_session", None) is self:
            del self.model._refinekv_session
        if hasattr(self, "_original_generate_attribute"):
            if self._original_generate_attribute is _MISSING:
                self.model.__dict__.pop("generate", None)
            else:
                self.model.generate = self._original_generate_attribute
        self.model.config._attn_implementation = self._original_implementation
        self._layers.clear()
        self._anchor_queries.clear()
        self._cache = None
        self._graph_mode = False
        self._graph_suffix_length = None
        self._active = False

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._restore()
        finally:
            self._inference.__exit__(exc_type, exc_value, traceback)
        return False

    def _generate_cuda_graph(self, *args, **kwargs):
        """Capture one fixed-length greedy decode step and replay it on CUDA."""
        allowed = {
            "input_ids", "attention_mask", "max_new_tokens", "min_new_tokens", "do_sample",
            "num_beams", "use_cache", "eos_token_id", "forced_eos_token_id", "pad_token_id",
            "cache_implementation", "disable_compile", "repetition_penalty",
        }
        unsupported = set(kwargs) - allowed
        if unsupported:
            raise ValueError(f"CUDA Graph generation does not support {sorted(unsupported)}")
        if args:
            if len(args) != 1 or "input_ids" in kwargs:
                raise ValueError("pass one input_ids tensor to CUDA Graph generation")
            input_ids = args[0]
        else:
            input_ids = kwargs.get("input_ids")
        if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
                or input_ids.shape[0] != 1 or input_ids.dtype != torch.long):
            raise ValueError("CUDA Graph generation requires one input_ids sequence")
        max_new_tokens = kwargs.get("max_new_tokens")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 4:
            raise ValueError("CUDA Graph generation requires max_new_tokens >= 4")
        if (kwargs.get("min_new_tokens") != max_new_tokens
                or "eos_token_id" not in kwargs or kwargs["eos_token_id"] is not None):
            raise ValueError(
                "CUDA Graph generation requires fixed output: "
                "min_new_tokens=max_new_tokens and eos_token_id=None"
            )
        generation = self.model.generation_config
        if (kwargs.get("do_sample", generation.do_sample)
                or (kwargs.get("num_beams", generation.num_beams) or 1) != 1
                or kwargs.get("use_cache", generation.use_cache)
                not in (None, True)
                or kwargs.get("forced_eos_token_id", generation.forced_eos_token_id) is not None
                or kwargs.get("cache_implementation", generation.cache_implementation)
                not in (None, "static")
                or (generation.num_return_sequences or 1) != 1):
            raise ValueError("CUDA Graph generation supports greedy, single-beam, static-cache decode only")
        penalty = kwargs.get(
            "repetition_penalty", getattr(generation, "repetition_penalty", 1.0)
        )
        penalty = 1.0 if penalty is None else float(penalty)
        if not math.isfinite(penalty) or penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive")
        unsupported_processors = {
            "encoder_repetition_penalty": (None, 1.0),
            "no_repeat_ngram_size": (None, 0),
            "bad_words_ids": (None, []),
            "suppress_tokens": (None, []),
            "begin_suppress_tokens": (None, []),
            "forced_bos_token_id": (None,),
            "sequence_bias": (None, {}),
            "force_words_ids": (None, []),
            "constraints": (None, []),
            "watermarking_config": (None,),
        }
        active = [
            name for name, defaults in unsupported_processors.items()
            if getattr(generation, name, None) not in defaults
        ]
        if active:
            raise ValueError(f"CUDA Graph generation does not support logits processors {active}")
        if (not input_ids.is_cuda or input_ids.device != self.model.device
                or self.model.dtype != torch.bfloat16 or self.config.backend == "torch"
                or self.config.collect_stats or self.config.full_attention_layers):
            raise ValueError("CUDA Graph requires a CUDA BF16 model, Triton backend, no stats, and no dense layers")
        mask = kwargs.get("attention_mask")
        if mask is not None and (
            not isinstance(mask, torch.Tensor) or mask.shape != input_ids.shape
            or mask.device != input_ids.device or not bool(mask.all())
        ):
            raise ValueError("CUDA Graph generation requires an unpadded prompt")

        prompt_length = input_ids.shape[1]
        cache = Cache(layers=[
            _GraphStaticLayer(prompt_length + max_new_tokens) for _ in self.model.model.layers
        ])
        prefill = self.model(
            input_ids=input_ids, attention_mask=mask, past_key_values=cache,
            use_cache=True, logits_to_keep=1,
        )
        seen = None
        if penalty != 1.0:
            seen = torch.zeros_like(prefill.logits[:, -1], dtype=torch.bool)
            seen.scatter_(1, input_ids, True)

        def greedy(scores):
            scores = scores.float()
            if seen is not None:
                adjusted = torch.where(scores < 0, scores * penalty, scores / penalty)
                scores = torch.where(seen, adjusted, scores)
            return scores.argmax(-1, keepdim=True)

        token = greedy(prefill.logits[:, -1]).clone()
        output_tokens = torch.empty(max_new_tokens, device=input_ids.device, dtype=input_ids.dtype)
        output_tokens[:1].copy_(token.reshape(1))
        del prefill
        for layer in cache.layers:
            layer.graph_decode = True
        position = torch.tensor([prompt_length], device=input_ids.device, dtype=torch.long)
        output_index = torch.ones(1, device=input_ids.device, dtype=torch.long)
        self._graph_suffix_length = torch.zeros((), device=input_ids.device, dtype=torch.int32)
        self._graph_mode = True

        def step():
            self._graph_suffix_length.add_(1)
            if seen is not None:
                seen.scatter_(1, token, True)
            result = self.model(
                input_ids=token, attention_mask=None, past_key_values=cache,
                cache_position=position, position_ids=position.unsqueeze(0),
                use_cache=True, logits_to_keep=1,
            )
            following = greedy(result.logits[:, -1])
            output_tokens.index_copy_(0, output_index, following.reshape(1))
            token.copy_(following)
            position.add_(1)
            output_index.add_(1)

        try:
            warmup_steps = min(2, max_new_tokens - 2)
            stream = torch.cuda.Stream(device=input_ids.device)
            stream.wait_stream(torch.cuda.current_stream(input_ids.device))
            with torch.cuda.stream(stream):
                for _ in range(warmup_steps):
                    step()
            torch.cuda.current_stream(input_ids.device).wait_stream(stream)
            remaining = max_new_tokens - 1 - warmup_steps
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            # Capture records the work; only replays produce retained outputs.
            for _ in range(remaining):
                graph.replay()
            torch.cuda.synchronize(input_ids.device)
            return torch.cat((input_ids, output_tokens.view(1, -1)), dim=1)
        finally:
            self._graph_mode = False
            self._graph_suffix_length = None

    def _before_forward(self, module, args, kwargs):
        inputs = kwargs.get("input_ids")
        if inputs is None:
            inputs = kwargs.get("inputs_embeds")
        if inputs is None and args:
            inputs = args[0]
        if inputs is not None and inputs.shape[0] != 1:
            raise ValueError("RefineKV requires batch_size=1 and num_beams=1")
        if not kwargs.get("use_cache", self.model.config.use_cache):
            raise ValueError("RefineKV requires use_cache=True")
        if kwargs.get("output_attentions", False):
            raise ValueError("RefineKV does not return attention weights")
        cache = kwargs.get("past_key_values")
        if cache is None or cache is not self._cache:
            if cache is not None and cache.get_seq_length() != 0:
                raise ValueError("start each RefineKV generation with an empty KV cache")
            mask = kwargs.get("attention_mask")
            if isinstance(mask, dict):
                mask = mask.get("full_attention")
            if mask is not None:
                if mask.ndim == 2:
                    valid_mask = bool(mask.all())
                elif mask.ndim == 4 and inputs is not None:
                    # StaticCache generation prepares the causal mask before model.forward.
                    # Its final prompt row must expose every real prompt token.
                    last = mask[0, 0, -1, : inputs.shape[1]]
                    valid_mask = (
                        bool(last.all()) if mask.dtype == torch.bool else bool((last == 0).all())
                    )
                else:
                    valid_mask = False
                if not valid_mask:
                    raise ValueError("RefineKV requires an unpadded prompt")
            self._layers.clear()
            self._anchor_queries.clear()
            self._totals = dict(accessed=0, dense=0, decode_attention_calls=0)
            self._cache = cache

    def _after_forward(self, module, args, kwargs, output):
        cache = getattr(output, "past_key_values", None)
        if cache is not None:
            self._cache = cache

    def _capture_anchor_queries(self, module, args, kwargs):
        layer_index = self._module_indices[id(module)]
        if layer_index in self._layers:
            return
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        positions = kwargs.get("position_embeddings")
        if positions is None and len(args) > 1:
            positions = args[1]
        if hidden is None or positions is None:
            raise RuntimeError("prompt anchors require hidden states and RoPE embeddings")
        count = min(self.config.anchor_tokens, hidden.shape[1])
        tail = hidden[:, -count:]
        query = module.q_proj(tail).view(*tail.shape[:-1], -1, module.head_dim).transpose(1, 2)
        if self.model.config.model_type == "qwen2":
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        else:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        cos, sin = positions
        query, _ = apply_rotary_pos_emb(query, query, cos[..., -count:, :], sin[..., -count:, :])
        # Match the original tail projection. A full-prompt GEMM can round
        # differently in BF16, changing the fixed mask at threshold boundaries.
        self._anchor_queries[layer_index] = query[0].detach()

    def _dense_decode(self, module, query, key, value, mask, scaling, kwargs, visible_length):
        # Preserve the original explicit-mask SDPA path, including KV-head
        # repetition. The mask-free GQA path can produce different BF16 outputs.
        if mask is None:
            mask = query.new_full((1, 1, 1, key.shape[2]), torch.finfo(query.dtype).min)
            mask[..., :visible_length] = 0
        elif mask.dtype == torch.bool:
            mask = query.new_zeros(mask.shape).masked_fill(~mask, torch.finfo(query.dtype).min)
        return sdpa_attention_forward(
            module, query, key, value, mask, dropout=0, scaling=scaling, **kwargs
        )

    def _attention(self, module, query, key, value, mask, dropout, scaling, kwargs):
        if query.shape[0] != 1:
            raise ValueError("RefineKV requires batch_size=1")
        layer_index = self._module_indices[id(module)]
        state = self._layers.get(layer_index)
        if state is None:
            prompt_length = query.shape[2]
            output = sdpa_attention_forward(
                module, query, key, value, mask, dropout=0, scaling=scaling, **kwargs
            )
            state = _LayerState(prompt_length, prompt_length)
            if layer_index >= self.config.full_attention_layers:
                state.summary = build_summary(
                    key[0, :, :prompt_length], value[0, :, :prompt_length], self.config
                )
                heads = key.shape[1]
                groups = query.shape[1] // heads
                anchors = min(self.config.anchor_tokens, prompt_length)
                anchor_queries = self._anchor_queries.pop(layer_index, None)
                if anchors and anchor_queries is None:
                    raise RuntimeError(f"missing prompt anchor queries in layer {layer_index}")
                for position in range(anchors):
                    q = anchor_queries[:, position : position + 1].reshape(
                        heads, groups, 1, query.shape[-1]
                    )
                    selected = select_blocks(
                        q, state.summary, self.config, workspace=state.workspace
                    )
                    state.anchor_mask = (
                        selected if state.anchor_mask is None else state.anchor_mask | selected
                    )
            self._layers[layer_index] = state
            return output
        if query.shape[2] != 1:
            raise ValueError("RefineKV supports one full prefill followed by single-token decode")
        if self._graph_mode:
            if state.summary is None:
                raise ValueError("CUDA Graph does not support dense attention layers")
            heads, groups, dim = key.shape[1], query.shape[1] // key.shape[1], query.shape[-1]
            result = sparse_attention(
                query[0].reshape(heads, groups, 1, dim), state.summary,
                key[0, :, state.prompt_length:], value[0, :, state.prompt_length:],
                self.config, anchor_mask=state.anchor_mask, workspace=state.workspace,
                suffix_length=self._graph_suffix_length,
            )
            return result.output.reshape(query.shape[1], 1, dim).transpose(0, 1).unsqueeze(0), None
        state.visible_length += 1
        if key.shape[2] < state.visible_length:
            raise ValueError("the KV cache no longer contains the full visible prefix")
        if state.summary is None:
            if self.config.collect_stats:
                dense = query.shape[1] * state.visible_length
                self._record(dense, dense)
            return self._dense_decode(
                module, query, key, value, mask, scaling, kwargs, state.visible_length
            )
        heads, groups, dim = key.shape[1], query.shape[1] // key.shape[1], query.shape[-1]
        grouped = query[0].reshape(heads, groups, 1, dim)
        result = sparse_attention(
            grouped,
            state.summary,
            key[0, :, state.prompt_length : state.visible_length],
            value[0, :, state.prompt_length : state.visible_length],
            self.config,
            anchor_mask=state.anchor_mask,
            workspace=state.workspace,
            return_selected=self.config.collect_stats,
        )
        if self.config.collect_stats:
            selected = result.selected
            accessed = ((selected * state.summary.counts).sum() + (~selected).sum()) * groups
            accessed = accessed + query.shape[1] * (state.visible_length - state.prompt_length)
            self._record(accessed, query.shape[1] * state.visible_length)
        output = result.output.reshape(query.shape[1], 1, dim).transpose(0, 1).unsqueeze(0)
        return output, None

    def _record(self, accessed, dense):
        self._totals["accessed"] = self._totals["accessed"] + accessed
        self._totals["dense"] += dense
        self._totals["decode_attention_calls"] += 1
