# Decode speed benchmark

This is the release protocol for comparing **Full Attention** with **RefineKV
mean/Q0 + CUDA Graph**. The headline setting generates 128 tokens at 128K total
context and measures the last 125 decode steps. An inclusive decode window is
reported alongside it to show the effect of Graph setup. No prefill latency is
reported.

## Reproduce

From the repository root, install the CUDA dependencies and run the benchmark
on a CUDA GPU that can hold the model and the 128K KV cache:

```bash
pip install -e ".[transformers,cuda]"
python examples/benchmark_decode.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output speed_128.json
```

The default is 128 output tokens. To run the longer, supplementary setting:

```bash
python examples/benchmark_decode.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output-tokens 256 --output speed_256.json
```

`--model` also accepts a local model directory. The script writes each measured
request's timing, generated token IDs, median latency, speedup, environment,
and a retrieval sanity check to JSON. `--context-length` changes **prompt plus
output** length; its default is 131,072. `--warmup` and `--repeats` default to
one and three. Run the methods on the same GPU with no competing workload.

## Optional CUDA Graph decode

`RefineKV(model, config)` uses ordinary Transformers generation by default.
Pass `cuda_graph=True` to opt into fixed-length CUDA Graph decode. The benchmark
script enables it explicitly for its RefineKV measurements. This release path
is independent of the paper's optimized runner and does not include its
`multi2` selected-page kernel.

```python
with RefineKV(model, config, cuda_graph=True):
    output = model.generate(
        **inputs, max_new_tokens=128, min_new_tokens=128,
        eos_token_id=None, do_sample=False,
    )
```

Graph requires a CUDA BF16 model, Triton on an Ampere-or-newer GPU, one
unpadded sequence, greedy single-beam decode, at least four output tokens,
`min_new_tokens=max_new_tokens`, `eos_token_id=None`, no dense leading layers,
and no access-statistics collection. Its cache must be static or unspecified.
The Graph path applies the model's `repetition_penalty` (or an explicit
override); other custom logits processors are unsupported. It uses a private
preallocated static cache and captures separately for each request. The output
length is fixed, so EOS does not stop generation. Warmup and capture are part of
request time and are included in the all-decode window below.

For the CLI, add `--cuda-graph` to `examples/generate.py`. It uses ordinary
generation unless that flag is present. Use `--gqa-reduction mean` and
`--anchor-tokens 0` for the LongBench mean/Q0 algorithm setting.

## Protocol

| Item | Setting |
| --- | --- |
| Model | Llama-3.1-8B-Instruct, BF16, Transformers SDPA, greedy generation |
| Prompt | Exact-length synthetic retrieval prompt with the answer in the middle |
| Total length | 131,072 tokens: prompt 130,944 + output 128, or prompt 130,816 + output 256 |
| RefineKV | Triton, block size 32, `eps=0.01`, GQA reduction `mean`, zero anchors (mean/Q0), zero dense leading layers, access statistics off |
| Full Attention | Eager, mask-free native-GQA SDPA; visible-prefix, in-place static KV cache |
| RefineKV runtime | CUDA Graph explicitly enabled; private fixed-address, in-place static KV cache; no `multi2` kernel |
| Requests | One untimed warmup per method, then three measured requests per method; order alternates Full/RefineKV and RefineKV/Full |
| Summary | Median synchronized wall time per decode step for each method; speedup = Full median / RefineKV median |

The first generated token comes from the prompt forward pass. Thus 128 generated
tokens have 127 decode intervals, and 256 generated tokens have 255. Each method
uses the same prompt, generation length, and model. CUDA is synchronized at the
start and end of each measured window; the timer spans the full sequence of
decode steps, not individual GPU kernels.

Two windows are reported for every request:

1. **All decode:** from the end of the prompt forward pass to generation end.
   For RefineKV this includes two eager decode warmups, Graph capture, and
   replays. It spans 127 or 255 decode intervals.
2. **Steady decode:** from after Full's second eager decode forward, or after
   RefineKV's Graph capture, to generation end. It spans 125 or 253 decode
   intervals; RefineKV executes only Graph replays in this window.

The steady window compares warmed decode paths. The all-decode window exposes
Graph setup cost over the same generated length. Full runs eager while RefineKV
uses a Graph, so these figures describe the **release runtime comparison**, not
an isolated comparison of the attention kernels. The script does not use the
stock Transformers Full Attention `StaticCache`: it calls the shared
`make_visible_static_cache` helper, keeping SDPA mask-free while writing KV
in place.

## Measured results

Measured on **NVIDIA RTX PRO 6000 Blackwell Server Edition** with PyTorch
2.10.0, Transformers 5.2.0, Triton 3.6.0, and RefineKV 0.1.2. Values are
medians of three measured requests, in milliseconds per decode step.

| Generated tokens | Window | Full Attention | RefineKV Graph | Full / RefineKV |
| ---: | --- | ---: | ---: | ---: |
| **128** | **Steady, last 125** | **26.807** | **14.210** | **1.886×** |
| 128 | All 127 | 26.859 | 14.580 | 1.842× |
| 256 | Steady, last 253 | 26.818 | 14.222 | 1.886× |
| 256 | All 255 | 26.825 | 14.421 | 1.860× |

The archived measurements are in [results.json](results.json). The two
128-token windows came from separate paired runs with identical settings; the
two 256-token windows came from the same three paired requests. The current
benchmark script records both windows in each request, so reproduction can
show small run-to-run differences. Both methods' answers contained the target
code in all three measured requests in each run. This is a basic retrieval
sanity check, not a quality evaluation.

After extracting the visible-prefix cache into the shared library helper, one
paired 128K regression request measured 26.920 ms/token for Full and
14.214 ms/token for RefineKV Graph over the last 125 steps (1.894×). Both
answers passed the retrieval sanity check. This single-request check confirms
the cache refactor did not materially change the three-request medians above;
it does not replace them. Its raw request data is in
[regression_128_visible_static.json](regression_128_visible_static.json).

The 256-token result shows the same steady decode speedup as the 128-token
result to three decimal places. Its all-decode speedup is closer to the steady
number because the setup work is spread over more decode steps. Results for
1,024 generated tokens have not been measured here.

These results use one synthetic prompt and one model/GPU setting. They should
not be compared directly with the paper's 126-step LongBench macro average:
that result used a different workload and optimized runner, including the
`multi2` kernel. For other cache and algorithm experiments,
`examples/benchmark.py` remains exploratory. Its static mode now uses the
same visible-prefix cache for Full and RefineKV, while its timing protocol
differs from this benchmark.
