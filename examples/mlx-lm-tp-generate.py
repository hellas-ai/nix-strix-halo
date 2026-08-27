#!/usr/bin/env python3

import argparse

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import sharded_load


parser = argparse.ArgumentParser(
    description="Run one MLX-LM generation with tensor-parallel weights."
)
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", default="Reply with exactly: TP works")
parser.add_argument("--max-tokens", type=int, default=16)
parser.add_argument("--temperature", type=float, default=0.0)
args = parser.parse_args()
if args.max_tokens <= 0:
    parser.error("max-tokens must be positive")
if args.temperature < 0:
    parser.error("temperature must be non-negative")

world = mx.distributed.init(strict=True, backend="jaccl")
model, tokenizer = sharded_load(args.model, tensor_group=world)
response = generate(
    model,
    tokenizer,
    prompt=args.prompt,
    max_tokens=args.max_tokens,
    sampler=make_sampler(temp=args.temperature),
    verbose=False,
)
if world.rank() == 0:
    print(response)
