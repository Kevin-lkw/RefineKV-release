"""Explore cache and algorithm settings with 127-interval inclusive decode timing.

For the standard 128-output-token, last-125-step comparison, use benchmark_decode.py.
"""

import argparse
from contextlib import nullcontext
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from refinekv import RefineKV, RefineKVConfig, make_visible_static_cache, __version__


def synthetic_inputs(tokenizer, length, device):
    """Build an exact-length synthetic retrieval prompt, retaining its target and question."""
    marker = "REFINEKV_BENCHMARK_DOCUMENT"
    question = "\nWhat is the verification code for record 420? Give only the code."
    content = marker + question
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        before, after = rendered.split(marker)
        prefix = tokenizer.encode(before, add_special_tokens=False)
    else:
        prefix = tokenizer.encode("", add_special_tokens=True)
        after = question
    suffix = tokenizer.encode(after, add_special_tokens=False)
    target = tokenizer.encode(
        "\nRecord 420 verification code: SILVER-6419-ORCHID.\n", add_special_tokens=False
    )
    filler = tokenizer.encode(
        "\nOrdinary reference record: department 17, inventory 203, no verification code.\n",
        add_special_tokens=False,
    )
    available = length - len(prefix) - len(target) - len(suffix)
    if available < 0 or not filler:
        raise ValueError("context length is too short for the synthetic prompt")
    body = (filler * (available // len(filler) + 1))[:available]
    midpoint = available // 2
    ids = prefix + body[:midpoint] + target + body[midpoint:] + suffix
    input_ids = torch.tensor([ids], device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def run_once(model, inputs, config, options, *, collect_stats=False, cuda_graph=False):
    """Synchronize at the prefill boundary and request end, never on each decode step."""
    session_config = replace(config, collect_stats=collect_stats) if config else None
    context = RefineKV(model, session_config, cuda_graph=cuda_graph and not collect_stats) if config else nullcontext()
    generate_options = dict(options)
    if config is None and options["cache_implementation"] == "static":
        capacity = inputs["input_ids"].shape[1] + options["max_new_tokens"]
        generate_options["past_key_values"] = make_visible_static_cache(model, capacity)
        generate_options["cache_implementation"] = None
    with torch.inference_mode(), context as session:
        first_forward_end = None

        def mark_prefill(module, args, output):
            nonlocal first_forward_end
            if first_forward_end is None:
                torch.cuda.synchronize(model.device)
                first_forward_end = time.perf_counter()

        handle = model.register_forward_hook(mark_prefill)
        try:
            torch.cuda.synchronize(model.device)
            start = time.perf_counter()
            output = model.generate(**inputs, **generate_options)
            torch.cuda.synchronize(model.device)
            end = time.perf_counter()
        finally:
            handle.remove()
        if first_forward_end is None:
            raise RuntimeError("generation did not execute a model forward")
        tokens = output.shape[1] - inputs["input_ids"].shape[1]
        if tokens != options["max_new_tokens"]:
            raise RuntimeError(f"expected fixed output length, got {tokens} tokens")
        decode_steps = tokens - 1  # The first output token is produced by prefill.
        decode_seconds = end - first_forward_end
        sample = {
            "prefill_setup_ms": (first_forward_end - start) * 1000,
            "decode_ms_per_token": decode_seconds * 1000 / decode_steps,
            "decode_tokens_per_second": decode_steps / decode_seconds,
            "total_seconds": end - start,
            "output_tokens": tokens,
            "decode_steps": decode_steps,
        }
        stats = session.stats if collect_stats else None
    return sample, output[0, inputs["input_ids"].shape[1] :].cpu(), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face ID or local model directory")
    parser.add_argument("--device", default="cuda:0")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        help="Total tokens including output, e.g. 32768 65536 131072 (default: 131072)",
    )
    prompts.add_argument("--prompt-file", type=Path, help="Use this UTF-8 text as the user prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--anchor-tokens", type=int, default=32)
    parser.add_argument("--gqa-reduction", choices=("mean", "max", "any"), default="any")
    parser.add_argument("--full-attention-layers", type=int, default=0)
    parser.add_argument("--cuda-graph", action="store_true", help="Graph the RefineKV decode path only")
    parser.add_argument(
        "--cache",
        choices=("dynamic", "static"),
        default="static",
        help="KV cache implementation (default: static)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Collect access in a separate untimed run"
    )
    parser.add_argument(
        "--output", type=Path, help="Save environment, samples, medians, and answers as JSON"
    )
    args = parser.parse_args()
    if args.cuda_graph and (args.cache != "static" or args.full_attention_layers):
        parser.error("--cuda-graph requires --cache static and zero full-attention layers")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this latency example requires a CUDA GPU and the cuda installation extra")
    minimum_tokens = 4 if args.cuda_graph else 2
    if args.max_new_tokens < minimum_tokens or args.warmup < 1 or args.repeats < 1:
        parser.error(
            f"use at least {minimum_tokens} output tokens, 1 warmup, and 1 repeat"
        )
    lengths = args.context_lengths or [131072]
    if not args.prompt_file and any(length <= args.max_new_tokens for length in lengths):
        parser.error("context lengths must exceed the output-token budget")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    config = RefineKVConfig(
        backend="triton",
        eps=args.eps,
        block_size=args.block_size,
        anchor_tokens=args.anchor_tokens,
        gqa_reduction=args.gqa_reduction,
        full_attention_layers=args.full_attention_layers,
    )
    options = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "disable_compile": True,
        "eos_token_id": None,
        "forced_eos_token_id": None,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        "cache_implementation": args.cache,
    }
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
        if tokenizer.chat_template:
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_cases = [(inputs["input_ids"].shape[1], inputs)]
    else:
        prompt_cases = [
            (
                length - args.max_new_tokens,
                synthetic_inputs(tokenizer, length - args.max_new_tokens, device),
            )
            for length in lengths
        ]
    for length, _ in prompt_cases:
        if length + args.max_new_tokens > model.config.max_position_embeddings:
            parser.error(
                "prompt + output exceeds model max_position_embeddings; use shorter input or a long-context model config"
            )
    report = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(device),
        "versions": {
            **{
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "triton")
            },
            "refinekv": __version__,
        },
        "config": vars(config),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cache": options["cache_implementation"],
        "full_cache": (
            "RefineKV visible-prefix static" if args.cache == "static"
            else "Transformers DynamicCache"
        ),
        "refinekv_cache": (
            "RefineKV private Graph static" if args.cuda_graph
            else "RefineKV visible-prefix static" if args.cache == "static"
            else "Transformers DynamicCache"
        ),
        "context_length_definition": "prompt tokens + generated tokens",
        "compile": False,
        "refinekv_cuda_graph": args.cuda_graph,
        "prompt_source": str(args.prompt_file) if args.prompt_file else "synthetic retrieval",
        "timing": "Synchronized wall time; prefill/setup ends after first model forward; decode includes generation-loop overhead and boundary synchronization.",
        "results": [],
    }
    print(
        f"GPU: {report['gpu']} | BF16 | eager SDPA baseline | "
        f"RefineKV {'CUDA Graph' if args.cuda_graph else 'eager'} | "
        f"{report['cache']} cache"
        f"{' (visible-prefix)' if args.cache == 'static' else ''}",
        flush=True,
    )
    print(
        "Medians; decode excludes the token produced by prefill. Statistics are untimed.",
        flush=True,
    )
    print(
        f"Forced {args.max_new_tokens} output tokens, ignoring EOS, for equal-length timing.",
        flush=True,
    )
    print(
        f"{'Context':>7} {'Prompt':>7}  {'Method':<15} {'Prefill/setup ms':>16} {'Decode ms/tok':>14} {'Decode tok/s':>13} {'Total s':>9}",
        flush=True,
    )
    methods = {"Full Attention": None, "RefineKV": config}
    for length, inputs in prompt_cases:
        samples = {name: [] for name in methods}
        answers = {}
        for name, method in methods.items():
            for _ in range(args.warmup):
                run_once(model, inputs, method, options, cuda_graph=args.cuda_graph)
        # Alternate order to reduce systematic drift between methods.
        for repeat in range(args.repeats):
            order = list(methods) if repeat % 2 == 0 else list(reversed(methods))
            for name in order:
                sample, tokens, _ = run_once(
                    model, inputs, methods[name], options, cuda_graph=args.cuda_graph
                )
                samples[name].append(sample)
                answers[name] = tokenizer.decode(tokens, skip_special_tokens=True)
        medians = {
            name: {key: statistics.median(s[key] for s in runs) for key in runs[0]}
            for name, runs in samples.items()
        }
        for name, values in medians.items():
            print(
                f"{length + args.max_new_tokens:7d} {length:7d}  {name:<15} {values['prefill_setup_ms']:16.2f} {values['decode_ms_per_token']:14.2f} {values['decode_tokens_per_second']:13.2f} {values['total_seconds']:9.3f}",
                flush=True,
            )
        decode_speedup = (
            medians["Full Attention"]["decode_ms_per_token"]
            / medians["RefineKV"]["decode_ms_per_token"]
        )
        total_speedup = (
            medians["Full Attention"]["total_seconds"] / medians["RefineKV"]["total_seconds"]
        )
        print(
            f"         RefineKV speedup: decode {decode_speedup:.2f}x | total {total_speedup:.2f}x",
            flush=True,
        )
        access = None
        if args.stats:
            _, _, access = run_once(model, inputs, config, options, collect_stats=True)
            print(
                f"         Decode equivalent KV access: {access['equivalent_access_ratio']:.2%}",
                flush=True,
            )
        for name, answer in answers.items():
            preview = answer[:160] + ("..." if len(answer) > 160 else "")
            print(f"         {name} output preview: {preview!r}", flush=True)
        report["results"].append(
            {
                "context_tokens": length + args.max_new_tokens,
                "prompt_tokens": length,
                "samples": samples,
                "medians": medians,
                "decode_speedup": decode_speedup,
                "total_speedup": total_speedup,
                "answers": answers,
                "access_stats": access,
            }
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
