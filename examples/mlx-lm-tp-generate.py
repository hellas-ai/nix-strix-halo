#!/usr/bin/env python3

import argparse
import json

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import sharded_load


parser = argparse.ArgumentParser(
    description="Run a reference or tensor-parallel MLX-LM generation."
)
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", default="Reply with exactly: TP works")
parser.add_argument("--max-tokens", type=int, default=16)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument(
    "--reference",
    action="store_true",
    help="Run the same generation without tensor parallelism.",
)
args = parser.parse_args()
if args.max_tokens <= 0:
    parser.error("max-tokens must be positive")
if args.temperature < 0:
    parser.error("temperature must be non-negative")

if args.reference:
    world = None
    model, tokenizer = load(args.model)
else:
    world = mx.distributed.init(strict=True, backend="jaccl")
    model, tokenizer = sharded_load(args.model, tensor_group=world)

tokens = []
text = []
for response in stream_generate(
    model,
    tokenizer,
    prompt=args.prompt,
    max_tokens=args.max_tokens,
    sampler=make_sampler(temp=args.temperature),
):
    text.append(response.text)
    if response.generation_tokens > len(tokens):
        tokens.append(int(response.token))

if world is None or world.rank() == 0:
    print(
        json.dumps(
            {
                "mode": "reference" if args.reference else "tensor-parallel",
                "text": "".join(text),
                "tokens": tokens,
            },
            ensure_ascii=False,
        )
    )
