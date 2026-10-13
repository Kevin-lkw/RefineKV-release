# RefineKV

Implementation of **RefineKV: Query-Adaptive KV Cache Compression via
Attention-Error Bounds**.

RefineKV reduces long-context decode work by adapting KV access to each query.
Different queries need different parts of the context, so it retains the original
KV cache and selectively refines contiguous prompt blocks at each decode step.

For vLLM serving, see the separate experimental
[RefineKV-vllm plugin](https://github.com/Kevin-lkw/RefineKV-vllm).

## How it works

1. **Summarize.** After prefill, cache block mean K/V, counts, value norms,
   and centered diagonal-ellipsoid bounds.
2. **Select.** The attention-output error bound yields a query-dependent risk
   score combining **importance** (approximate attention mass) and **distortion**
   (bounded logit deviation), evaluated using only block summaries.
3. **Refine.** Expand blocks whose risk exceeds `eps` (the paper's `τ`). Others
   contribute count-weighted mean K/V. Selected blocks and generated tokens use
   token-level attention; Triton merges contributions with online softmax.

GQA `any` expands a block if any query head in its KV group selects it.
Prompt-query anchors keep tail-query selections expanded throughout generation.
Risk evaluation uses stable log-space arithmetic.

**RefineKV reduces decode access and computation, while retaining the full KV
cache.** Equivalent access is not allocated memory or measured speedup. `eps`
is a block-risk threshold, not a fixed token budget or a total generation-error
bound.

The paper evaluates Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct on LongBench,
RULER, and HELMET, reporting up to **11.7× attention-phase speedup** and
**1.76× decode speedup**. These are paper results on RTX PRO 6000 Blackwell.
The release benchmark uses a separate runtime, described below.

## Demos

These silent MP4s are uploaded as GitHub video attachments.

### Figure 3: how RefineKV selects KV blocks

https://github.com/user-attachments/assets/79a81278-2e34-4fef-8cec-c19db89af354

### Long-context decoding

https://github.com/user-attachments/assets/31753e09-710e-46db-baef-dbd29ef51630

On a real 130,801-token LongBench-v2 prompt, Llama-3.1-8B-Instruct generates
256 tokens with Full Attention and with the separate RefineKV vLLM plugin.
Median time per output token over three measured requests was **23.03 ms** and
**12.56 ms**, respectively (**1.83×** speedup).

## Install

From this directory:

```bash
pip install -e ".[transformers,cuda]"
# PyTorch core / CPU reference only:
pip install -e .
```

Recommended reproduction environment: **Python 3.10, PyTorch 2.10.0+cu128,
Triton 3.6.0, Transformers 5.1.0**. This combination passed 31 repository tests
and 20 real-model comparisons against saved original-implementation results,
with bitwise-equal tokens, logits, masks, summaries, and access counters.

Dependency ranges: Python 3.10+, PyTorch 2.10–2.13, Transformers 5.1–5.2,
Triton 3.6–3.7. PyTorch 2.13 / Triton 3.7.1 / CUDA 13.0 was also validated;
cross-version model results can differ. CUDA requires BF16 on NVIDIA Ampere or
newer. The PyTorch backend is a correctness reference.

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from refinekv import RefineKV, RefineKVConfig

model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda").eval()
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain how attention works."}],
    tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

config = RefineKVConfig(block_size=32, eps=0.01, anchor_tokens=32)
with RefineKV(model, config):
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

Or use a Hugging Face identifier / local model directory:

```bash
python examples/generate.py --model Qwen/Qwen2.5-7B-Instruct \
  --eps 0.01 --anchor-tokens 32 --static-cache
```

Generation uses the standard Transformers path by default. The context resets
each request and restores the model on exit. It supports **Llama / Qwen2 /
Qwen2.5**, one unpadded sequence, greedy or sampling generation, and dynamic or
static KV caches. Requesting `cache_implementation="static"` inside a RefineKV
context uses RefineKV's visible-prefix, in-place cache. Its prefill sees only
written tokens, avoiding the large 4-D mask produced by stock Transformers
`StaticCache` at long context lengths. Full Attention comparisons can use
`make_visible_static_cache(model, prompt_length + max_new_tokens)` to share
the same cache behavior by passing it as `past_key_values` to `model.generate`.
RefineKV requires `num_beams=1`, `use_cache=True`, and an empty incoming cache;
sliding-window attention and `torch.compile` are unsupported.
For Qwen contexts above 32K, the paper uses static YaRN with factor 4.

## Compare decode speed

At 128K total context, the release benchmark measures **128 generated tokens**
and the **last 125 decode steps**: **1.886×** versus Full Attention. See
[speed/README.md](speed/README.md) for the protocol, other windows, and
the opt-in CUDA Graph path used for these measurements.

## Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `block_size` | `32` | Prompt block size: 16, 32, or 64 |
| `eps` | `0.01` | Risk threshold `τ`; smaller typically expands more blocks |
| `gqa_reduction` | `"any"` | Combine query-head decisions: `any`, `max`, or `mean` |
| `anchor_tokens` | `0` | Union selections from the final N prompt queries |
| `full_attention_layers` | `0` | Keep the first N layers dense |
| `backend` | `"auto"` | Triton when supported, otherwise PyTorch; also `torch` / `triton` |
| `summary_chunk_blocks` | `512` | Block count per FP32 summary-construction chunk |
| `collect_stats` | `False` | Collect decode-only equivalent access counters |

Paper settings: blocks of **32**, GQA **any**, `eps` **0.01 or 0.1**;
**32 anchors for RULER/HELMET**, **0 for LongBench**. These are experiment
settings; tune for your workload. CUDA supports GQA groups up to 16 and head
dimensions 32–256 in multiples of 32.

Read `session.stats` with `collect_stats=True`: unexpanded blocks count as one
representative, expanded blocks and suffixes by actual tokens. Counters include
dense decode layers, exclude prefill, and synchronize CUDA; disable for timing.

## Tensor API

The core works independently of Transformers:

```python
from refinekv import RefineKVConfig, build_summary, sparse_attention

config = RefineKVConfig()
# Prompt K/V: (kv_heads, prompt_tokens, head_dim)
summary = build_summary(prompt_keys, prompt_values, config)
# Query: (kv_heads, group_size, 1, head_dim)
# Suffix K/V: (kv_heads, visible_generated_tokens, head_dim)
output, selected = sparse_attention(
    query, summary, suffix_keys, suffix_values, config, return_selected=True,
)
```

`select_blocks` returns a mask usable as `anchor_mask`. Supply only visible
suffix tokens and keep prompt K/V immutable. Optional `workspace={}` reuses CUDA
buffers; clone retained outputs before the next call.

## Layout and tests

```text
src/refinekv/
  config.py                    settings
  summary.py                   block summaries and centered bounds
  attention.py / reference.py  tensor API and PyTorch reference
  kernels/                     Triton selection and mixed attention
  integrations/transformers.py model.generate adapter
examples/generate.py            inference example
tests/                         correctness and integration tests
```

```bash
pip install -e ".[transformers,cuda,dev]"
pytest -q
```

Tests cover input preservation, partial blocks, bounds, anchor union, dense and
Triton/reference agreement, and state restoration. Random model tests require
no downloads. For CPU: `pytest -m "not cuda"`.

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution.
