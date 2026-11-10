#!/usr/bin/env python3
"""GPU qualification of the runtime-length indexer and page-table kernels.

Checks, for many guarded widths, row counts, pool sizes and exponent regimes:

  1. `_paged_logits_wmma` / `_paged_logits` with runtime WIDTH/PAGE_COLS/PAGES are
     BIT-IDENTICAL (int32 view, so NaN payloads and -inf count) to the frozen
     constexpr kernels in indexer-constexpr-reference.py, including positions
     past each row's length and invalid page-table entries.
  2. The installed kernels compile ONE variant for all widths (device cache size).
  3. `page_table_from_req_to_token` equals the torch expression it replaces and
     compiles one variant per page size.
  4. First-call latency per width (JSON `timing` events): the reference pays its
     JIT at every new width bucket, the installed kernel only at the first.

    sglang-python indexer-recompile-gpu.py RUNTIME             # GPU, needs the sentinel
    sglang-python indexer-recompile-gpu.py RUNTIME --cpu-interpret   # smoke test, CPU

The GPU mode refuses to run unless the coordinator's GPU-FREE-FOR-MICROBENCH
sentinel exists. It does not take /tmp/ds41-gpu.lock: like the K2/Q microbenches it
may share a node with a loaded-but-idle serving process (< 5 GB of GPU memory).
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

SENTINEL = Path(
    "/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/GPU-FREE-FOR-MICROBENCH"
)
parser = argparse.ArgumentParser()
parser.add_argument("runtime")
parser.add_argument("--cpu-interpret", action="store_true")
args = parser.parse_args()
runtime = Path(args.runtime).resolve()

if args.cpu_interpret:
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
else:
    if not SENTINEL.exists():
        raise SystemExit(f"refusing to touch a GPU: {SENTINEL} does not exist")

import torch  # noqa: E402

DEVICE = "cpu" if args.cpu_interpret else "cuda"
DSV4 = runtime / "lib/python3.13/site-packages/sglang/kernels/ops/attention/dsv4"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new = load(DSV4 / "fp4_indexer_gfx1151.py", "indexer_runtime_ints")
ref = load(Path(__file__).with_name("indexer-constexpr-reference.py"), "indexer_constexpr")
glue = load(DSV4 / "attn_glue_hip.py", "attn_glue_runtime")
if not args.cpu_interpret:
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
    torch.cuda.set_per_process_memory_fraction(0.03)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def build(rows, heads, width, page_cols, pages, regime, seed, lengths_offset=0):
    gen = torch.Generator().manual_seed(seed)

    def rand(shape, lo, hi, dtype):
        return torch.randint(lo, hi, shape, generator=gen, dtype=torch.int32).to(dtype)

    # normal: tight exponents (WMMA fast path); wide: any byte, which exercises the
    # scalar fallback guard, subnormal products, Inf and NaN scales.
    lo, hi = (124, 131) if regime == "normal" else (0, 256)
    q = rand((rows, heads, 64), 0, 256, torch.uint8)
    q_scale = rand((rows, 1, 4, 16, 4), lo, hi, torch.uint8)
    k = rand((pages, 1, 4, 64, 16), 0, 256, torch.uint8)
    k_scale = rand((pages, 1, 4, 64), lo, hi, torch.uint8)
    weights = torch.randn((rows, heads), generator=gen).to(torch.bfloat16)
    # Invalid page ids (-2, -1, >= pages) must read as -inf in both kernels.
    table = rand((rows, page_cols), -2, pages + 2, torch.int32)
    lengths = rand((rows + lengths_offset,), 0, width + 130, torch.int32)
    lengths[lengths_offset] = width
    if rows > 1:
        lengths[lengths_offset + 1] = 0
    if rows > 2:
        lengths[lengths_offset + 2] = width + 100
    tensors = [t.to(DEVICE) for t in (q, q_scale, k, k_scale, weights, table, lengths)]
    # Row-chunked scoring hands the kernel `lengths[rows]`: slice AFTER the copy so
    # the pointer really sits 4 * offset bytes past an aligned base.
    tensors[6] = tensors[6][lengths_offset:]
    return tuple(tensors)


def run(module, tensors, width, scale, fill):
    out = torch.full((tensors[0].shape[0], width), fill, dtype=torch.float32, device=DEVICE)
    q, q_scale, k, k_scale, weights, table, lengths = tensors
    module.paged_fp4_logits(q, q_scale, k, k_scale, weights, table, lengths, scale, out)
    return out


def same_bits(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


if args.cpu_interpret:
    WIDTHS = [(64, 5), (128, 6), (256, 8)]  # (width, page_cols)
    ROWS = (1, 3)
    POOLS = (3,)
else:
    guarded = [(4096 * n, 64 * n + 4) for n in (1, 2, 3, 4, 7, 8, 15, 16, 33)]
    guarded += [(256, 8), (768, 16), (4352, 72)]
    ragged = [(100, 6), (4100, 70), (8192, 128)]  # unguarded: no +4, odd columns
    WIDTHS = guarded + ragged
    ROWS = (1, 5, 37, 320)
    POOLS = (3, 257, 8191)

failures = []
checked = 0
seed = 0
for heads, kernel_name in ((32, "_paged_logits_wmma"), (16, "_paged_logits"), (64, "_paged_logits")):
    for width, page_cols in WIDTHS:
        for pages in POOLS:
            for rows in ROWS:
                if rows * width > 16 * 2**20:
                    continue
                for regime, offset in (("normal", 0), ("wide", 0), ("normal", 1), ("wide", 3)):
                    seed += 1
                    tensors = build(rows, heads, width, page_cols, pages, regime, seed, offset)
                    t0 = time.perf_counter()
                    expected = run(ref, tensors, width, 0.5, float("nan"))
                    sync()
                    t1 = time.perf_counter()
                    got = run(new, tensors, width, 0.5, -123.0)
                    sync()
                    t2 = time.perf_counter()
                    checked += 1
                    if not same_bits(expected, got):
                        diff = (expected.view(torch.int32) != got.view(torch.int32)).sum()
                        failures.append(
                            dict(kernel=kernel_name, heads=heads, width=width, page_cols=page_cols,
                                 pages=pages, rows=rows, regime=regime, lengths_offset=offset,
                                 differing=int(diff))
                        )
                    if regime == "normal" and offset == 0 and rows == ROWS[0] and pages == POOLS[0]:
                        emit("timing", kernel=kernel_name, heads=heads, width=width,
                             reference_s=round(t1 - t0, 3), runtime_ints_s=round(t2 - t1, 3))

emit("bitwise", cases=checked, failures=len(failures))
for failure in failures[:20]:
    emit("mismatch", **failure)

# Variant counts: one Triton variant per kernel/head-count, whatever the width.
variants = {}
if not args.cpu_interpret:
    device = torch.cuda.current_device()
    for name in ("_paged_logits_wmma", "_paged_logits"):
        for tag, module in (("runtime_ints", new), ("constexpr_reference", ref)):
            variants[f"{name}.{tag}"] = len(getattr(module, name).device_caches[device][0])
    emit("variants", **variants)
    assert variants["_paged_logits_wmma.runtime_ints"] == 1, variants
    assert variants["_paged_logits.runtime_ints"] == 2, variants

# page_table_from_req_to_token against the torch expression it replaces.
gen = torch.Generator().manual_seed(7)
table_failures = []
page_sizes = (64, 256)
cols = (1, 2, 3, 5, 9, 17, 33, 64, 65, 200, 1000, 2048, 4097) if not args.cpu_interpret else (1, 5, 33)
r2t_width = 1 << 20 if not args.cpu_interpret else 1 << 12
for page_size in page_sizes:
    r2t = torch.randint(-5, 2**30, (6, r2t_width), generator=gen, dtype=torch.int32).to(DEVICE)
    row_ids = torch.tensor([3, 0, 5, 3], dtype=torch.int64, device=DEVICE)
    for num_cols in cols:
        max_seq_len = max(1, num_cols * page_size - int(torch.randint(0, page_size, (1,), generator=gen)))
        got = glue.page_table_from_req_to_token(r2t, row_ids, max_seq_len, page_size)
        want = torch.div(
            r2t[row_ids][:, :max_seq_len:page_size], page_size, rounding_mode="floor"
        ).to(torch.int32)
        if not torch.equal(got, want):
            table_failures.append(dict(page_size=page_size, num_cols=num_cols))
emit("page_table", cases=len(page_sizes) * len(cols), failures=len(table_failures))
if not args.cpu_interpret:
    count = len(glue._page_table_from_req_to_token_kernel.device_caches[torch.cuda.current_device()][0])
    emit("page_table_variants", count=count, page_sizes=len(page_sizes))
    assert count == len(page_sizes), count

if failures or table_failures:
    raise SystemExit(1)
emit("pass", note="runtime-int kernels are bit-identical to the constexpr reference")
