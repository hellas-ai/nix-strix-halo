#!/usr/bin/env python3

import argparse
import time

import mlx.core as mx


parser = argparse.ArgumentParser()
parser.add_argument("--elements", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=1)
parser.add_argument("--warmup", type=int, default=0)
args = parser.parse_args()
if args.elements <= 0 or args.iterations <= 0 or args.warmup < 0:
    parser.error("elements and iterations must be positive; warmup must be non-negative")

world = mx.distributed.init(strict=True, backend="jaccl")
value = mx.full((args.elements,), world.rank() + 1, dtype=mx.int32)

for _ in range(args.warmup):
    mx.eval(mx.distributed.all_sum(value, group=world))

# Synchronize ranks before timing without assuming synchronized host clocks.
mx.eval(mx.distributed.all_sum(mx.array([1]), group=world))
started = time.perf_counter()
for _ in range(args.iterations):
    result = mx.distributed.all_sum(value, group=world)
    mx.eval(result)
elapsed = time.perf_counter() - started

expected = world.size() * (world.size() + 1) // 2
if not bool(mx.all(result == expected)):
    raise RuntimeError(f"rank {world.rank()}: incorrect all-sum result")

print(
    f"rank={world.rank()} size={world.size()} "
    f"elements={result.size} iterations={args.iterations} "
    f"value={result[0].item()} seconds={elapsed:.6f} "
    f"payload_gib_s={args.elements * 4 * args.iterations / elapsed / 2**30:.3f}"
)
