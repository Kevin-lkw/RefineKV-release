"""Compare Full and RefineKV Graph over inclusive and steady-state decode windows."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import synthetic_inputs
from refinekv import RefineKV, RefineKVConfig, make_visible_static_cache, __version__


def run_full(model, inputs, options):
    output_tokens = options["max_new_tokens"]
    capacity = inputs["input_ids"].shape[1] + output_tokens
    cache = make_visible_static_cache(model, capacity)
    calls = 0
    boundaries = {}

    def mark_forward(module, args, output):
        nonlocal calls
        calls += 1
        if calls in (1, 3):
            torch.cuda.synchronize(model.device)
            boundaries[calls] = time.perf_counter()

    handle = model.register_forward_hook(mark_forward)
    try:
        with torch.inference_mode():
            output = model.generate(**inputs, **options, past_key_values=cache)
            torch.cuda.synchronize(model.device)
            end = time.perf_counter()
    finally:
        handle.remove()
    if calls != output_tokens or set(boundaries) != {1, 3} or output.shape[1] != capacity:
        raise RuntimeError(f"unexpected Full generation length: {calls} forwards")
    return {
        "all_decode_ms_per_token": (end - boundaries[1]) * 1000 / (output_tokens - 1),
        "steady_decode_ms_per_token": (end - boundaries[3]) * 1000 / (output_tokens - 3),
        "tokens": output[0, -output_tokens:].cpu().tolist(),
    }


def run_refinekv_graph(model, inputs, config, options):
    # The release path captures after two eager decode warmups. Mark both
    # boundaries so one request provides inclusive and replay-only timing.
    output_tokens = options["max_new_tokens"]
    original_graph = torch.cuda.CUDAGraph
    captured = []
    calls = 0
    prefill_boundary = None

    class TimedGraph(original_graph):
        def __init__(self):
            super().__init__()
            self.replays = 0
            captured.append(self)

        def capture_end(self, *args, **kwargs):
            result = super().capture_end(*args, **kwargs)
            torch.cuda.synchronize(model.device)
            self.replay_boundary = time.perf_counter()
            return result

        def replay(self):
            result = super().replay()
            self.replays += 1
            return result

    def mark_forward(module, args, output):
        nonlocal calls, prefill_boundary
        calls += 1
        if calls == 1:
            torch.cuda.synchronize(model.device)
            prefill_boundary = time.perf_counter()

    handle = model.register_forward_hook(mark_forward)
    try:
        torch.cuda.CUDAGraph = TimedGraph
        with RefineKV(model, config, cuda_graph=True):
            output = model.generate(**inputs, **options)
            torch.cuda.synchronize(model.device)
            end = time.perf_counter()
    finally:
        handle.remove()
        torch.cuda.CUDAGraph = original_graph
    if (len(captured) != 1 or calls != 4 or prefill_boundary is None
            or captured[0].replays != output_tokens - 3
            or output.shape[1] != inputs["input_ids"].shape[1] + output_tokens):
        raise RuntimeError(f"unexpected RefineKV Graph forward/replay count: {calls}")
    return {
        "all_decode_ms_per_token": (end - prefill_boundary) * 1000 / (output_tokens - 1),
        "steady_decode_ms_per_token": (end - captured[0].replay_boundary) * 1000 / (output_tokens - 3),
        "tokens": output[0, -output_tokens:].cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face ID or local model directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-length", type=int, default=131072,
                        help="Prompt plus output tokens (default: 131072)")
    parser.add_argument("--output-tokens", type=int, default=128,
                        help="Fixed generated-token count (default: 128)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Write raw samples and tokens as JSON")
    args = parser.parse_args()
    if args.output_tokens < 4 or args.context_length <= args.output_tokens or args.warmup < 1 or args.repeats < 1:
        parser.error("output tokens must be at least 4, context length must exceed output tokens, and warmup/repeats must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this benchmark requires CUDA")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    if args.context_length > model.config.max_position_embeddings:
        parser.error("context length exceeds model max_position_embeddings")
    inputs = synthetic_inputs(tokenizer, args.context_length - args.output_tokens, device)
    config = RefineKVConfig(
        backend="triton", eps=0.01, block_size=32, gqa_reduction="mean", anchor_tokens=0
    )
    options = dict(
        max_new_tokens=args.output_tokens, min_new_tokens=args.output_tokens, eos_token_id=None,
        forced_eos_token_id=None, do_sample=False, num_beams=1,
        use_cache=True, disable_compile=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    methods = {
        "Full Attention": lambda: run_full(model, inputs, options),
        "RefineKV Graph": lambda: run_refinekv_graph(model, inputs, config, options),
    }
    for method in methods:
        for _ in range(args.warmup):
            methods[method]()
    samples = {method: [] for method in methods}
    metrics = ("all_decode_ms_per_token", "steady_decode_ms_per_token")
    for repeat in range(args.repeats):
        order = list(methods) if repeat % 2 == 0 else list(reversed(methods))
        for method in order:
            row = methods[method]()
            samples[method].append(row)
            print(
                f"{repeat + 1} {method}: all {row['all_decode_ms_per_token']:.3f}, "
                f"steady {row['steady_decode_ms_per_token']:.3f} ms/token", flush=True
            )
    medians = {
        method: {metric: statistics.median(row[metric] for row in rows) for metric in metrics}
        for method, rows in samples.items()
    }
    report = {
        "model": args.model, "gpu": torch.cuda.get_device_name(device),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "triton"
        )} | {"refinekv": __version__},
        "context_tokens": args.context_length,
        "prompt_tokens": inputs["input_ids"].shape[1],
        "output_tokens": args.output_tokens,
        "all_decode_steps": args.output_tokens - 1,
        "steady_decode_steps": args.output_tokens - 3,
        "timing": {
            "all_decode_ms_per_token": "Synchronized wall time after prefill forward to generation end; includes two eager decode warmups and Graph capture for RefineKV",
            "steady_decode_ms_per_token": "Synchronized wall time after Full second decode forward or RefineKV Graph capture to generation end",
        },
        "full_cache": "visible-prefix in-place static cache, mask-free native-GQA SDPA",
        "refinekv_cache": "private fixed-address in-place static cache",
        "refinekv_graph": True,
        "refinekv_multi2": False,
        "config": vars(config), "warmup": args.warmup, "repeats": args.repeats,
        "prompt_source": "synthetic retrieval",
        "samples": samples, "medians": medians,
        "speedup": {
            metric: medians["Full Attention"][metric] / medians["RefineKV Graph"][metric]
            for metric in metrics
        },
        "answer_contains_target": {
            method: [
                "SILVER-6419-ORCHID" in tokenizer.decode(row["tokens"], skip_special_tokens=True)
                for row in rows
            ]
            for method, rows in samples.items()
        },
    }
    print(
        "Full / RefineKV Graph: "
        f"all {report['speedup']['all_decode_ms_per_token']:.3f}x, "
        f"steady {report['speedup']['steady_decode_ms_per_token']:.3f}x", flush=True
    )
    print(f"Retrieval sanity: {report['answer_contains_target']}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
