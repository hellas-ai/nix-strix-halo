#!/usr/bin/env python3
"""Installed-runtime check: length growth must not create new Triton variants.

CPU only. `_paged_logits_wmma` (and the scalar `_paged_logits`) used to take the
guarded page-table width, page-table columns and pool page count as
`tl.constexpr`, so every 4096-position growth of the context JIT-compiled a new
variant (about 1.4 s each, on every rank, in the middle of a prefill). They are
runtime integers now, excluded from Triton's divisibility specialization.

This script feeds the installed kernels' real Python call sites with CPU tensors
and asks Triton's own specializer for the cache key (the hash that names the
on-disk cache directory) of every launch. It never compiles or launches.

    sglang-python indexer-recompile-cpu.py /nix/store/...-sglang-v41-rocm-gfx1151
"""

import importlib.util
import sys
from pathlib import Path

import torch
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend
from triton.runtime.jit import create_function_from_signature

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"
DSV4 = SITE / "sglang/kernels/ops/attention/dsv4"
BACKEND = make_backend(GPUTarget("hip", "gfx1151", 32))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hash(kernel, args, kwargs):
    """ASTSource hash of one launch: signature, constexprs, specialization."""
    binder = create_function_from_signature(kernel.signature, kernel.params, BACKEND)
    bound, specialization, _ = binder(*args, **kwargs)
    options = BACKEND.parse_options(kwargs)
    _, signature, constexprs, attrs = kernel._pack_args(
        BACKEND, kwargs, bound, specialization, options
    )
    return ASTSource(kernel, signature, constexprs, attrs).hash()


indexer = load(DSV4 / "fp4_indexer_gfx1151.py", "indexer_gfx1151_under_test")
glue = load(DSV4 / "attn_glue_hip.py", "attn_glue_hip_under_test")

params = {p.name: p for p in indexer._paged_logits_wmma.params}
for name in ("WIDTH", "PAGE_COLS", "PAGES"):
    assert not params[name].is_constexpr, f"{name} is a constexpr again"
    assert params[name].do_not_specialize, f"{name} is specialized again"

REAL = {
    "_paged_logits_wmma": indexer._paged_logits_wmma,
    "_paged_logits": indexer._paged_logits,
    "_page_table_from_req_to_token_kernel": glue._page_table_from_req_to_token_kernel,
}
HASHES = {name: set() for name in REAL}
LAUNCHES = {name: 0 for name in REAL}


class Recorder:
    """Stands in for a kernel at its call site: hash the launch, never run it."""

    def __init__(self, name):
        self.name = name

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            HASHES[self.name].add(source_hash(REAL[self.name], args, kwargs))
            LAUNCHES[self.name] += 1

        return launch


indexer._paged_logits_wmma = Recorder("_paged_logits_wmma")
indexer._paged_logits = Recorder("_paged_logits")
glue._page_table_from_req_to_token_kernel = Recorder(
    "_page_table_from_req_to_token_kernel"
)


def guarded(logical_width, bucket):
    pages = max(4, (logical_width + bucket - 1) // bucket * bucket)
    return pages * 64, pages + 4


def call(heads, width, page_cols, pages, rows, lengths_offset=0):
    u8 = {"dtype": torch.uint8}
    q = torch.empty((rows, heads, 64), **u8)
    q_scale = torch.empty((rows, 1, 4, 16, 4), **u8)
    k = torch.empty((pages, 1, 4, 64, 16), **u8)
    k_scale = torch.empty((pages, 1, 4, 64), **u8)
    weights = torch.empty((rows, heads), dtype=torch.bfloat16)
    table = torch.empty((rows, page_cols), dtype=torch.int32)
    # Row-chunked scoring passes `lengths[rows]`: a pointer 4 * offset bytes past an
    # aligned base, i.e. 16-byte aligned only for some chunks.
    lengths = torch.empty((rows + lengths_offset,), dtype=torch.int32)[lengths_offset:]
    out = torch.empty((rows, width), dtype=torch.float32)
    indexer.paged_fp4_logits(q, q_scale, k, k_scale, weights, table, lengths, 1.0, out)


widths = set()
for bucket in (4, 64):  # page_table_bucket of the generic and low-ratio callers
    for logical in list(range(1, 400)) + [4096, 8191, 16385, 16384 * 4 + 1]:
        widths.add(guarded(logical, bucket))
for width, page_cols in sorted(widths):
    for pool_pages in (1, 2, 4095, 8191, 16000):
        for rows in (1, 7, 40):
            call(32, width, page_cols, pool_pages, rows)
    # An unguarded caller and a ragged one must not split the cache either.
    call(32, width, page_cols - 4, 3, 2)
    call(32, width + 16, page_cols + 1, 3, 2)
    for offset in (1, 2, 3, 273):  # sliced length vectors, aligned and not
        call(32, width, page_cols, 3, 4, lengths_offset=offset)
for width, page_cols in sorted(widths)[:40]:
    call(16, width, page_cols, 5, 3)
    call(48, width, page_cols, 6, 3)

wmma = HASHES["_paged_logits_wmma"]
assert len(wmma) == 1, f"{len(wmma)} _paged_logits_wmma variants"
# The scalar kernel specializes on nothing length-related: one variant per head count.
scalar = HASHES["_paged_logits"]
assert len(scalar) == 2, f"{len(scalar)} _paged_logits variants for 2 head counts"

# _page_table_from_req_to_token: the tile was next_power_of_2(num_cols), one
# variant per power of two plus a divisibility split. PAGE is a constexpr of the
# deployment (one value per server), so count variants per page size.
request_to_token = torch.empty((2, 1 << 21), dtype=torch.int32)
rows = torch.tensor([1, 0], dtype=torch.int64)
table = HASHES["_page_table_from_req_to_token_kernel"]
per_page_size = {}
for page_size in (64, 128, 256):
    table.clear()
    for max_seq_len in list(range(1, 2100)) + [4097, 65535, 1 << 20, 1 << 21]:
        glue.page_table_from_req_to_token(request_to_token, rows, max_seq_len, page_size)
    per_page_size[page_size] = len(table)
    assert len(table) == 1, f"{len(table)} page-table variants at page size {page_size}"

# The bf16-shadow activation dequantizer is shape-generic too: fp8_kernel cannot be
# imported without a device, so extract it (its decoder imports fine).
import ast  # noqa: E402
import sys as _sys  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

_sys.path.insert(0, str(SITE))
from sglang.kernels.ops.gemm.deepseek_v41_gemv import _e4m3fn_to_fp32  # noqa: E402

fp8_path = SITE / "sglang/kernels/ops/gemm/fp8_kernel.py"
tree = ast.parse(fp8_path.read_text(), filename=str(fp8_path))
node = next(
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_dequant_e4m3fn_group32_bf16"
)
dequant_ns = {"triton": triton, "tl": tl, "_e4m3fn_to_fp32": _e4m3fn_to_fp32}
exec(  # noqa: S102 -- installed kernel source, never launched here
    compile(
        ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
        str(fp8_path),
        "exec",
    ),
    dequant_ns,
)
dequant = dequant_ns["_dequant_e4m3fn_group32_bf16"]
dequant_hashes = set()
for k_dim in (576, 1280, 2048, 5120, 6144, 32, 64, 4096):
    for m_rows in (1, 16, 300, 1536):
        q = torch.empty((m_rows, k_dim), dtype=torch.float8_e4m3fn)
        sc = torch.empty((m_rows, k_dim // 32), dtype=torch.float32)
        o = torch.empty((m_rows, k_dim), dtype=torch.bfloat16)
        dequant_hashes.add(
            source_hash(
                dequant,
                (q, sc, o, k_dim, q.stride(0), sc.stride(0), sc.stride(1), o.stride(0)),
                dict(BLOCK=1024, num_warps=4),
            )
        )
assert len(dequant_hashes) == 1, f"{len(dequant_hashes)} dequantizer variants"

print(
    "indexer-recompile-cpu OK:",
    f"{LAUNCHES['_paged_logits_wmma']} wmma launches over {len(widths)} widths -> "
    f"{len(wmma)} variant;",
    f"{LAUNCHES['_paged_logits']} scalar launches -> {len(scalar)} variants;",
    f"{LAUNCHES['_page_table_from_req_to_token_kernel']} page-table launches -> "
    f"{per_page_size} variants per page size;",
    f"bf16-shadow dequantizer {len(dequant_hashes)} variant",
)
