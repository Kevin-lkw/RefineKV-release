"""Generate with RefineKV using a model from Hugging Face or a local directory."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from refinekv import RefineKV, RefineKVConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="Explain how attention works.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--anchor-tokens", type=int, default=0)
    parser.add_argument("--gqa-reduction", choices=("mean", "max", "any"), default="any")
    parser.add_argument("--backend", choices=("auto", "torch", "triton"), default="auto")
    parser.add_argument(
        "--static-cache", action="store_true",
        help="Use RefineKV's visible-prefix in-place static cache",
    )
    parser.add_argument("--cuda-graph", action="store_true", help="Fixed-length greedy CUDA Graph decode")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    if args.cuda_graph and (args.stats or args.backend == "torch" or not args.device.startswith("cuda")):
        parser.error("CUDA Graph decode requires CUDA, Triton, and stats disabled")
    if args.cuda_graph and args.max_new_tokens < 4:
        parser.error("CUDA Graph decode requires at least 4 output tokens")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
        .to(args.device)
        .eval()
    )
    if tokenizer.chat_template:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = inputs.to(args.device)
    config = RefineKVConfig(
        eps=args.eps,
        block_size=args.block_size,
        anchor_tokens=args.anchor_tokens,
        gqa_reduction=args.gqa_reduction,
        backend=args.backend,
        collect_stats=args.stats,
    )
    options = {"cache_implementation": "static"} if args.static_cache or args.cuda_graph else {}
    if args.cuda_graph:
        options.update(min_new_tokens=args.max_new_tokens, eos_token_id=None)
    with RefineKV(model, config, cuda_graph=args.cuda_graph) as session:
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
            **options,
        )
    print(tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True))
    if args.stats:
        print(session.stats)


if __name__ == "__main__":
    main()
