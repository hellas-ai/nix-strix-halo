"""gfx1151 decode RMSNorm / routed-add fusion: bitwise equality with the torch chain.

Run on a gfx1151 GPU with the runtime's sglang-python (``sglang-python
rmsnorm-decode.py RUNTIME [--source DIR]``).  ``--source`` prepends a patched
site-packages root so an unbuilt tree can be tested.  Uses < 1 GB of GPU memory.

What is asserted (every comparison is on raw bit patterns):
  * native_rmsnorm_decode(tree="match") == RMSNorm.forward_native chain for every
    supported width, row count 1..16, weight dtype and a set of adversarial and
    heavy-tailed inputs;
  * tree="invariant" rows are independent of the batch size;
  * the same holds inside a replayed CUDA graph with changed inputs;
  * native_routed_add_ == ``routed += shared``;
  * tl.rsqrt == torch.rsqrt on a dense sweep of positive FP32 values.
It also reports (not asserts) how many rows of the *deployed torch chain* change
with the batch size, i.e. the batch dependence the "match" tree reproduces.
"""

import json
import sys
from pathlib import Path

import torch

args = sys.argv[1:]
source = None
if "--source" in args:
    i = args.index("--source")
    source = args[i + 1]
    del args[i : i + 2]
if source:
    sys.path.insert(0, source)

import triton
import triton.language as tl

from sglang.kernels.ops.layernorm import native_rmsnorm_decode as nr
from sglang.kernels.ops.layernorm import native_routed_add as ra

torch.cuda.set_per_process_memory_fraction(0.05)
assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
assert torch.cuda.mem_get_info()[0] > 3 * 2**30, "not enough free GPU memory"
EPS = 1e-20
DEV = "cuda"


def chain(x, w, eps=EPS):
    """RMSNorm.forward_native for a plain row batch (the deployed path)."""
    x = x.contiguous()
    orig = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (x * w).to(orig)


def bits(t):
    return t.view(torch.int16)


def inputs(rows, width, kind, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(rows, width, generator=g)
    if kind == "randn":
        x = base * 3
    elif kind == "heavy_tail":
        scale = torch.exp(torch.randn(width, generator=g) * 1.5)
        x = base * scale
    elif kind == "massive_channels":
        x = base
        x[:, :: max(1, width // 7)] *= 300.0
    elif kind == "row_scale_sweep":
        x = base * torch.logspace(-6, 4, rows, base=10.0)[:, None]
    elif kind == "zeros_rows":
        x = base
        x[::2] = 0
    elif kind == "tiny":
        x = base * 1e-12
    elif kind == "huge":
        x = base * 1e15
    elif kind == "sparse":
        x = base * (torch.rand(rows, width, generator=g) < 0.02)
    elif kind == "bf16_ties":
        # values on bf16 rounding boundaries after scaling stress the final cast
        x = (base * 4).to(torch.bfloat16).float() * (1 + 2**-9)
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16).to(DEV)


KINDS = (
    "randn",
    "heavy_tail",
    "massive_channels",
    "row_scale_sweep",
    "zeros_rows",
    "tiny",
    "huge",
    "sparse",
    "bf16_ties",
)


def check_match():
    total = elems = 0
    for width in nr.SUPPORTED_WIDTHS:
        for wdtype in (torch.bfloat16, torch.float32):
            wg = torch.Generator(device="cpu").manual_seed(width)
            w = (torch.randn(width, generator=wg) * 0.4 + 1.0).to(wdtype).to(DEV)
            for rows in (1, 2, 3, 4, 5, 8, 12, 16):
                for k, kind in enumerate(KINDS):
                    for seed in range(3):
                        x = inputs(rows, width, kind, 1000 * k + seed)
                        got = nr.native_rmsnorm_decode(x, w, EPS, tree="match")
                        assert got is not None, (width, rows)
                        want = chain(x, w)
                        total += 1
                        elems += want.numel()
                        if not torch.equal(bits(got), bits(want)):
                            diff = (bits(got) != bits(want)).sum().item()
                            raise AssertionError(
                                f"mismatch width={width} rows={rows} kind={kind} w={wdtype} "
                                f"seed={seed}: {diff}/{want.numel()} elements"
                            )
    print(json.dumps({"event": "match", "calls": total, "elements": elems, "bitwise_equal": True}), flush=True)


def check_strided():
    width, full = 1280, 1792
    w = torch.randn(width, device=DEV).to(torch.bfloat16)
    for rows in (2, 3, 4, 8):
        big = (torch.randn(rows, full, device=DEV) * 3).to(torch.bfloat16)
        x = big[:, :width]
        assert not x.is_contiguous()
        got = nr.native_rmsnorm_decode(x, w, EPS, tree="match")
        assert got is not None
        assert torch.equal(bits(got), bits(chain(x, w)))
    print(json.dumps({"event": "strided_q_lora_slice", "bitwise_equal": True}), flush=True)


def check_fallbacks():
    w = torch.ones(5120, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(1, 5120, device=DEV).to(torch.bfloat16)
    assert nr.native_rmsnorm_decode(torch.randn(1, 5120, device=DEV), w, EPS) is None  # fp32 x
    assert nr.native_rmsnorm_decode(torch.randn(17, 5120, device=DEV).to(torch.bfloat16), w, EPS) is None
    assert nr.native_rmsnorm_decode(torch.randn(1, 4096, device=DEV).to(torch.bfloat16), w[:4096], EPS) is None
    assert nr.native_rmsnorm_decode(x, w.to(torch.float16), EPS) is None
    assert nr.native_rmsnorm_decode(x.cpu(), w.cpu(), EPS) is None
    assert nr.native_rmsnorm_decode(x[:, ::2], w[:2560], EPS) is None
    print(json.dumps({"event": "fallbacks", "ok": True}), flush=True)


def check_invariant_tree():
    changed_rows_torch = 0
    rows_total = 0
    for width in (5120, 1280):
        w = (torch.randn(width, device=DEV) * 0.4 + 1).to(torch.bfloat16)
        for seed in range(300):
            x = inputs(4, width, "randn", 7000 + seed)
            one = [chain(x[r : r + 1], w) for r in range(4)]
            for rows in (2, 3, 4):
                batched_torch = chain(x[:rows], w)
                batched = nr.native_rmsnorm_decode(x[:rows], w, EPS, tree="invariant")
                for r in range(rows):
                    solo = nr.native_rmsnorm_decode(x[r : r + 1], w, EPS, tree="invariant")
                    assert torch.equal(bits(batched[r : r + 1]), bits(solo))
                    # the invariant tree is the deployed one-row tree
                    assert torch.equal(bits(solo), bits(one[r]))
                    rows_total += 1
                    changed_rows_torch += int(not torch.equal(bits(batched_torch[r : r + 1]), bits(one[r])))
    print(
        json.dumps(
            {
                "event": "invariant_tree",
                "invariant_rows_equal_batch1": True,
                "deployed_torch_rows_checked": rows_total,
                "deployed_torch_rows_whose_bf16_output_differs_from_batch1": changed_rows_torch,
            }
        ),
        flush=True,
    )


def check_graph():
    for width, rows in ((5120, 1), (5120, 2), (5120, 4), (1280, 1), (1280, 4)):
        w = (torch.randn(width, device=DEV) * 0.4 + 1).to(torch.bfloat16)
        x = torch.randn(rows, width, device=DEV).to(torch.bfloat16)
        nr.native_rmsnorm_decode(x, w, EPS)  # compile outside capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = nr.native_rmsnorm_decode(x, w, EPS)
        for k, kind in enumerate(("randn", "heavy_tail", "row_scale_sweep", "randn")):
            x.copy_(inputs(rows, width, kind, 31 + k))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(bits(out), bits(chain(x, w))), (width, rows, kind)
    print(json.dumps({"event": "graph_replay", "bitwise_equal": True}), flush=True)


def check_routed_add():
    n = 0
    for rows in (1, 2, 3, 4, 8, 16):
        for seed in range(20):
            g = torch.Generator(device="cpu").manual_seed(seed * 31 + rows)
            routed = (torch.randn(rows, 5120, generator=g) * (10.0 ** (seed % 7 - 3))).to(DEV)
            shared = (torch.randn(rows, 5120, generator=g) * 2).to(torch.bfloat16).to(DEV)
            want = routed.clone()
            want += shared
            got = routed.clone()
            assert ra.native_routed_add_(got, shared)
            assert torch.equal(got.view(torch.int32), want.view(torch.int32))
            n += 1
    assert not ra.native_routed_add_(torch.zeros(17, 5120, device=DEV), torch.zeros(17, 5120, device=DEV, dtype=torch.bfloat16))
    assert not ra.native_routed_add_(torch.zeros(1, 5120, device=DEV, dtype=torch.bfloat16), torch.zeros(1, 5120, device=DEV, dtype=torch.bfloat16))
    print(json.dumps({"event": "routed_add", "cases": n, "bitwise_equal": True}), flush=True)


@triton.jit
def _rsqrt_probe_f64(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(X + i, mask=m, other=1.0).to(tl.float64)
    tl.store(Y + i, tl.rsqrt(x).to(tl.float32), mask=m)


@triton.jit
def _rsqrt_probe_f32(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    tl.store(Y + i, tl.rsqrt(tl.load(X + i, mask=m, other=1.0)), mask=m)


def check_rsqrt():
    # torch.rsqrt(float32) on this build is (float)rsqrt((double)x); the kernel must use the same.
    # Dense sweep: all exponents, random mantissas (subnormals reported separately).
    g = torch.Generator(device="cpu").manual_seed(1)
    exp = torch.randint(0, 255, (1 << 22,), generator=g, dtype=torch.int32)
    man = torch.randint(0, 1 << 23, (1 << 22,), generator=g, dtype=torch.int32)
    x = ((exp << 23) | man).view(torch.float32).to(DEV)
    want = torch.rsqrt(x).view(torch.int32)
    normal = (exp > 0)
    out = {}
    for name, kernel in (("f64_path", _rsqrt_probe_f64), ("f32_v_rsq_path", _rsqrt_probe_f32)):
        y = torch.empty_like(x)
        kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), 1024, enable_fp_fusion=False)
        differs = (y.view(torch.int32) != want).cpu()
        out[name] = {"mismatches": int(differs.sum()), "mismatches_normal": int((differs & normal).sum())}
    print(json.dumps({"event": "rsqrt_sweep", "values": x.numel(), **out}), flush=True)
    assert out["f64_path"]["mismatches_normal"] == 0


def main():
    check_rsqrt()
    check_match()
    check_strided()
    check_fallbacks()
    check_invariant_tree()
    check_graph()
    check_routed_add()
    print(json.dumps({"event": "complete"}), flush=True)


if __name__ == "__main__":
    main()
