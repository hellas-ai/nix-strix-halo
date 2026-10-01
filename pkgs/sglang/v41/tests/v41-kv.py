"""V41 native KV writer/reader regression; bounded GPU execution, no model weights."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.update(
    SGLANG_USE_AITER="0",
    SGLANG_DSV4_KV_LAYOUT="v41",
    SGLANG_DSV4_COMPRESSED_KV_LAYOUT="fp8",
    SGLANG_HACK_FLASHMLA_BACKEND="triton",
)
import torch

FIXED = {"BLOCK_H": 16, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1}


def quantize(rows, policy):
    assert rows.dtype == torch.bfloat16 and rows.shape[-1] == 512
    x = rows.float().reshape(-1, 16, 32)
    assert torch.isfinite(x).all()
    maxima = x.abs().amax(-1)
    # Deliberately different policies. Legacy writer floors AFTER division; official/current writer BEFORE.
    raw = (
        (maxima / 448).clamp_min(1e-4)
        if policy == "writer"
        else maxima.clamp_min(1e-4) / 448
    )
    assert policy in ("writer", "official")
    exps = torch.tensor(
        [math.ceil(math.log2(v)) for v in raw.flatten().tolist()], dtype=torch.int32
    ).reshape(raw.shape)
    scales = torch.tensor(
        [math.ldexp(1.0, e) for e in exps.flatten().tolist()], dtype=torch.float32
    ).reshape(exps.shape)
    codes = (
        (x / scales[..., None])
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(-1, 512)
    )
    return codes, (exps + 127).to(torch.uint8)


def decode(codes, scales):
    # Literal OCP finite E4M3 encoding, independent of Torch's float8 decoder.
    z = codes.to(torch.int32)
    m = z & 7
    e = (z >> 3) & 15
    assert not ((e == 15) & (m == 7)).any()
    values = torch.where(
        e == 0,
        m.double() * 2**-9,
        (1 + m.double() / 8) * torch.pow(2.0, e.double() - 7),
    )
    values = torch.where((z & 128) != 0, -values, values)
    values = (
        values.reshape(-1, 16, 32) * torch.pow(2.0, scales.double() - 127)[..., None]
    )
    return values.reshape(-1, 512).bfloat16()


def pack(raw, rows, slots, page_size, policy="official"):
    c, s = quantize(rows, policy)
    for row, slot in enumerate(slots.tolist()):
        page, off = divmod(slot, page_size)
        raw[page, off * 512 : (off + 1) * 512] = c[row]
        raw[page, page_size * 512 + off * 16 : page_size * 512 + (off + 1) * 16] = s[
            row
        ]
    return decode(c, s)


def unpack(raw, slots, page_size):
    c = []
    s = []
    for slot in slots.tolist():
        page, off = divmod(slot, page_size)
        c.append(raw[page, off * 512 : (off + 1) * 512])
        s.append(
            raw[page, page_size * 512 + off * 16 : page_size * 512 + (off + 1) * 16]
        )
    return torch.stack(c), torch.stack(s)


def attention(q, kv_groups, indices, lengths, sink, scale):
    # Global max is an independent real-valued reference. Probability/output casts
    # match the stated BF16 policy; block reduction/exp implementation is not a GPU oracle.
    output = []
    for row in range(len(q)):
        selected = []
        for kv, idx, lens in zip(kv_groups, indices, lengths):
            ids = idx[row, : int(lens[row])]
            ids = ids[ids >= 0].long()
            selected.append(kv[ids].double())
        values = torch.cat(selected)
        assert len(values) > 0
        score = q[row].double() @ values.T * scale
        mx = score.max(-1).values
        p = torch.exp(score - mx[:, None])
        denom = p.sum(-1) + torch.exp(sink.double() - mx)
        out = p.bfloat16().double() @ values / denom[:, None]
        output.append(out.bfloat16())
    return torch.stack(output)


def numeric(actual, expected):
    assert (
        actual.shape == expected.shape
        and torch.isfinite(actual).all()
        and torch.isfinite(expected).all()
    )
    d = (actual.double() - expected.double()).abs()
    rel = float(d.norm() / expected.double().norm().clamp_min(1e-30))
    ok = bool((d <= 0.012 + 0.02 * expected.double().abs()).all()) and rel < 0.004
    return {
        "passed": ok,
        "max_abs": float(d.max()),
        "relative_l2": rel,
        "rtol": 0.02,
        "atol": 0.012,
        "relative_l2_limit": 0.004,
    }


def policy_cases():
    zero = torch.zeros((1, 512), dtype=torch.bfloat16)
    tiny = torch.full((1, 512), 2**-28, dtype=torch.bfloat16)
    tiny[:, 1::2].neg_()
    exponents = (-28, -24, -20, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8)
    mixed = torch.cat(
        [torch.full((32,), 2.0**e, dtype=torch.bfloat16) for e in exponents]
    ).reshape(1, 512)
    mixed[:, 1::2].neg_()
    return {"zero": zero, "tiny": tiny, "mixed_groups": mixed}


def emit(event, **kw):
    print(json.dumps({"event": event, **kw}), flush=True)


def digest(x):
    return hashlib.sha256(
        x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def bits(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    )


def gate(label, passed, **kw):
    emit("gate", label=label, passed=bool(passed), **kw)
    assert passed, label


class FixedKernel:
    def __init__(self, jit, name):
        self.jit = jit
        self.name = name
        self.calls = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls += 1
            return self.jit[grid](*args, **kwargs, **FIXED)

        return launch


def make_pool(KVLayout, page, n_pages=3):
    stride = KVLayout.V41.page_bytes(page)
    raw = torch.full((n_pages, stride), 173, dtype=torch.uint8)
    view = raw[:, : page * 528].view(n_pages, page, 1, 528)
    assert view.stride(0) == stride
    return raw, view


def writer_checks(KVLayout, data):
    from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
    from sglang.kernels.ops.attention.dsv4.compress import (
        CompressorDecodePlan,
        compress_norm_rope_store,
    )
    from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla

    page = 16
    raw, _view = make_pool(KVLayout, page)
    slots = torch.tensor([page - 1], dtype=torch.int64)
    expected = raw.clone()
    pack(expected, data["norm_bf16"], slots, page, policy="official")
    gpu = raw.cuda()
    freq = torch.ones((1, 32), dtype=torch.complex64, device="cuda")
    pos = torch.zeros(1, dtype=torch.int64, device="cuda")
    loc = slots.to(torch.int32).cuda()
    inputs = [data["raw"].cuda(), data["weight"].cuda()]
    before = [digest(x) for x in inputs]
    fused_k_norm_rope_flashmla(
        *inputs, data["eps"], freq, pos, loc, gpu, page, layout=KVLayout.V41
    )
    torch.cuda.synchronize()
    actual = gpu.cpu().clone()
    gate(
        "actual_main_writer_synthetic_unit_norm_exact_all_bytes",
        bits(actual, expected),
        changed=int((actual != expected).sum()),
        floor_active_groups=0,
    )
    gate("main_writer_inputs_unchanged", before == [digest(x) for x in inputs])
    gpu.fill_(173)
    plan_bytes = (
        torch.tensor([[1, 0, 0, 0]], dtype=torch.int32).view(torch.uint8).cuda()
    )
    plan = CompressorDecodePlan(1, plan_bytes)
    compress_norm_rope_store(
        inputs[0],
        plan,
        norm_weight=inputs[1],
        norm_eps=data["eps"],
        freq_cis=freq,
        out_loc=slots.cuda(),
        kvcache=gpu,
        page_size=page,
        layout=KVLayout.V41,
    )
    torch.cuda.synchronize()
    actual = gpu.cpu().clone()
    gate(
        "actual_compressed_writer_synthetic_unit_norm_exact_all_bytes",
        bits(actual, expected),
        changed=int((actual != expected).sum()),
    )
    # Exact unit RMS with varying norm weights exercises the floor in both
    # producer entry points, including zero/tiny/mixed groups.
    unit = torch.ones_like(inputs[0])
    for label, norm_weight in policy_cases().items():
        expected = raw.clone()
        pack(expected, norm_weight, slots, page, policy="official")
        for producer in ("main", "compressed"):
            gpu.fill_(173)
            if producer == "main":
                fused_k_norm_rope_flashmla(
                    unit,
                    norm_weight.flatten().cuda(),
                    0.0,
                    freq,
                    pos,
                    slots.int().cuda(),
                    gpu,
                    page,
                    layout=KVLayout.V41,
                )
            else:
                compress_norm_rope_store(
                    unit,
                    plan,
                    norm_weight=norm_weight.flatten().cuda(),
                    norm_eps=0.0,
                    freq_cis=freq,
                    out_loc=slots.cuda(),
                    kvcache=gpu,
                    page_size=page,
                    layout=KVLayout.V41,
                )
            torch.cuda.synchronize()
            gate(producer + ":" + label + ":floor_bytes", bits(gpu.cpu(), expected))
    cases = policy_cases()
    cases["unit_norm"] = data["norm_bf16"]
    rows = torch.cat(list(cases.values()))
    for page in (16, 64, 256):
        raw, _ = make_pool(KVLayout, page)
        slots = torch.tensor(
            [page - 1, page, page * 2 - 1, page * 2], dtype=torch.int64
        )
        expected = raw.clone()
        pack(expected, rows, slots, page, policy="official")
        gpu = raw.cuda()
        fused_store_cache(
            rows.cuda(),
            gpu,
            slots.cuda(),
            page_size=page,
            type="flashmla",
            layout=KVLayout.V41,
        )
        torch.cuda.synchronize()
        actual = gpu.cpu().clone()
        gate(
            f"actual_store_page{page}_boundary_padding",
            bits(actual, expected),
            changed=int((actual != expected).sum()),
            page_stride=raw.stride(0),
        )
        c, s = unpack(actual, slots, page)
        official_c, official_s = quantize(rows, "official")
        emit(
            "official_policy_equality",
            page=page,
            cases=list(cases),
            payload_changes_per_row=(c != official_c).sum(-1).tolist(),
            scale_changes_per_row=(s != official_s).sum(-1).tolist(),
            reconstructed_value_changes_per_row=(
                decode(c, s) != decode(official_c, official_s)
            )
            .sum(-1)
            .tolist(),
            scope="Candidate must match official floor policy; old i50 expectation retained separately",
        )
        if page == 16:
            rotated = rows.clone()
            tail = rows[:, 448:].float().reshape(4, 32, 2)
            rotated[:, 448:] = (
                torch.stack(
                    (
                        tail[:, :, 0] * 0.0 - tail[:, :, 1] * 1.0,
                        tail[:, :, 0] * 1.0 + tail[:, :, 1] * 0.0,
                    ),
                    -1,
                )
                .reshape(4, 64)
                .bfloat16()
            )
            expected = raw.clone()
            pack(expected, rotated, slots, page, policy="official")
            gpu.fill_(173)
            fused_store_cache(
                rows.cuda(),
                gpu,
                slots.cuda(),
                page_size=page,
                type="flashmla",
                layout=KVLayout.V41,
                freqs_cis=torch.full((4, 32), 1j, dtype=torch.complex64, device="cuda"),
            )
            torch.cuda.synchronize()
            gate("actual_store_quarter_turn_BF16", bits(gpu.cpu(), expected))
            x = rows.cuda()
            loc = slots.cuda()
            fused_store_cache(
                x, gpu, loc, page_size=page, type="flashmla", layout=KVLayout.V41
            )
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fused_store_cache(
                    x, gpu, loc, page_size=page, type="flashmla", layout=KVLayout.V41
                )
            for label, values, positions in [
                ("A", rows, slots),
                ("B", -rows, slots.flip(0)),
                ("A_return", rows, slots),
            ]:
                x.copy_(values)
                loc.copy_(positions)
                gpu.fill_(173)
                g.replay()
                torch.cuda.synchronize()
                snapshot = gpu.cpu().clone()
                expected = raw.clone()
                pack(expected, values, positions, page, policy="official")
                gate("writer_graph_" + label, bits(snapshot, expected))
    return gpu


def exercise_case(
    m,
    name,
    raws,
    pages,
    qs,
    idxs,
    lens,
    sink,
    scale,
    split=False,
    low=False,
    graph=True,
):
    # Each case owns all tensors and snapshots. Page layouts remain noncontiguous.
    gpu_raw = [r.cuda() for r in raws]
    cache = [
        r[:, : p * 528].view(len(r), p, 1, 528).view(torch.float8_e4m3fn)
        for r, p in zip(gpu_raw, pages)
    ]
    kv = [decode(*unpack(r, torch.arange(len(r) * p), p)) for r, p in zip(raws, pages)]
    q = qs.cuda()
    ii = [x.cuda() for x in idxs]
    ll = [x.cuda() for x in lens]
    ss = sink.cuda()
    all_inputs = [q, ss, *gpu_raw, *ii, *ll]
    before = [digest(x) for x in all_inputs]

    def run():
        if len(cache) == 1:
            return m.fused_gather_attn_decode_dsv4(
                q, cache[0], ii[0], pages[0], scale, ll[0], ss
            )[0]
        fn = (
            m.fused_gather_attn_decode_dsv4_dual_scope_low_overhead
            if low
            else m.fused_gather_attn_decode_dsv4_dual_scope
        )
        kw = {} if low else {"force_no_splitk": not split}
        return fn(
            q,
            cache[0],
            ii[0],
            pages[0],
            cache[1],
            ii[1],
            pages[1],
            scale,
            ll[0],
            ll[1],
            ss,
            **kw,
        )[0]

    expected = attention(qs, kv, idxs, lens, sink, scale)
    out = run()
    torch.cuda.synchronize()
    a = out.cpu().clone()
    verdict = numeric(a, expected)
    gate(
        name + ":independent_native_operand_reference",
        passed=verdict.pop("passed"),
        **verdict,
    )
    gate(name + ":inputs_unchanged", before == [digest(x) for x in all_inputs])
    if not graph:
        return
    # Warm exact kernels then capture same addresses. Index/page ordering and lengths
    # change for B; values and query change too. No shared output alias is retained.
    out = run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured = run()
    g.replay()
    torch.cuda.synchronize()
    a_graph = captured.cpu().clone()
    gate(name + ":graph_A", bits(a, a_graph))
    bq = (-qs).contiguous()
    bi = [x.flip(1).contiguous() for x in idxs]
    bl = [torch.clamp(x - 1, min=1) for x in lens]
    q.copy_(bq)
    [x.copy_(y) for x, y in zip(ii, bi)]
    [x.copy_(y) for x, y in zip(ll, bl)]
    # Keep at least one valid selected key after flip + shortened length.
    assert all(
        (ids[row, : int(length[row])] >= 0).any()
        for ids, length in zip(bi, bl)
        for row in range(len(qs))
    )
    b_eager = run()
    torch.cuda.synchronize()
    b_owned = b_eager.cpu().clone()
    b_expected = attention(bq, kv, bi, bl, sink, scale)
    v = numeric(b_owned, b_expected)
    gate(name + ":B_reference", passed=v.pop("passed"), **v)
    g.replay()
    torch.cuda.synchronize()
    b_graph = captured.cpu().clone()
    gate(name + ":graph_B_owned", bits(b_owned, b_graph))
    q.copy_(qs)
    [x.copy_(y) for x, y in zip(ii, idxs)]
    [x.copy_(y) for x, y in zip(ll, lens)]
    g.replay()
    torch.cuda.synchronize()
    a_return = captured.cpu().clone()
    gate(name + ":graph_A_return", bits(a, a_return) and bits(a_graph, a))
    gate(name + ":all_inputs_restored", before == [digest(x) for x in all_inputs])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime")
    args = parser.parse_args()
    runtime = Path(args.runtime).resolve()
    # Exact RMS=1 with eps=0 removes a platform transcendental premise from bytes.
    raw = torch.ones((1, 512), dtype=torch.bfloat16)
    raw[:, 1::2] = -1
    data = {
        "raw": raw,
        "weight": torch.ones(512, dtype=torch.bfloat16),
        "norm_bf16": raw.clone(),
        "eps": 0.0,
        "q": torch.zeros((1, 16, 512), dtype=torch.bfloat16),
        "sink": torch.zeros(16),
        "softmax_scale": 512**-0.5,
    }
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(0.03, 0)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.kernels.ops.attention.nsa_triton_decode import (
        triton_mla_kernels_decode_fused as m,
    )

    assert str(Path(m.__file__).resolve()).startswith(str(runtime) + "/")
    names = (
        "_fused_gather_attn_dsv4_kernel",
        "_fused_gather_attn_dsv4_splitk_kernel",
        "_fused_gather_attn_dsv4_dual_scope_kernel",
        "_fused_gather_attn_dsv4_dual_scope_splitk_kernel",
    )
    for name in names:
        setattr(m, name, FixedKernel(getattr(m, name).fn, name))

    # Split8 combine has its own supported fixed dimensions; no autotuning sweep.
    class Combine:
        def __getitem__(self, grid):
            return lambda *a, **k: combine[grid](
                *a, **k, BLOCK_H=16, BLOCK_D=512, num_warps=4, num_stages=1
            )

    combine = m._combine_splitk_kernel_8_optimized.fn
    m._combine_splitk_kernel_8_optimized = Combine()
    emit(
        "start",
        runtime=str(runtime),
        config=FIXED,
        gpu_allocator_fraction=0.03,
        scope="V41 reader/writer contract only; official floor policy; no model/throughput qualification",
    )
    start = time.monotonic()
    writer_checks(KVLayout, data)
    raw, _ = make_pool(KVLayout, 16)
    pack(raw, data["norm_bf16"], torch.tensor([0]), 16)
    # Fill otherwise unselected rows as valid zero encodings rather than sentinel NaNs.
    pack(raw, torch.zeros((48, 512), dtype=torch.bfloat16), torch.arange(48), 16)
    pack(raw, data["norm_bf16"], torch.tensor([0]), 16)
    exercise_case(
        m,
        "synthetic_unit_norm_single",
        [raw],
        [16],
        data["q"],
        [torch.zeros((1, 1), dtype=torch.int32)],
        [torch.ones(1, dtype=torch.int32)],
        data["sink"],
        data["softmax_scale"],
        graph=False,
    )
    # Distinct scale groups, page boundaries/reordering, masks and per-query lengths.
    pools = []
    for page in (16, 64):
        raw, _ = make_pool(KVLayout, page)
        row = torch.arange(len(raw) * page)[:, None]
        col = torch.arange(512)[None, :]
        vals = (((row * 3 + col) % 13) - 6).float() * torch.pow(
            2.0, ((col // 32) % 16 - 12).float()
        )
        pack(raw, vals.bfloat16(), torch.arange(len(raw) * page), page)
        pools.append(raw)
    for name, widths, split, low in [
        ("single_nosplit", [65], False, False),
        ("single_split", [8192], True, False),
        ("dual_nosplit", [65, 67], False, False),
        ("dual_split", [128, 1024], True, False),
        ("dual_low_split", [128, 1024], True, True),
    ]:
        raws = pools[: len(widths)]
        pages = [16, 64][: len(widths)]
        ids = []
        lens = []
        for width, page in zip(widths, pages):
            x = (torch.arange(2 * width).reshape(2, width) * 17 % (3 * page)).int()
            x[:, 9::17] = -1
            ids.append(x)
            lens.append(torch.tensor([width, width - 2], dtype=torch.int32))
        # Split paths use uniform scores to independently test partition/denominator
        # composition. The no-split dual case additionally exercises nonuniform softmax.
        q = torch.zeros((2, 16, 512), dtype=torch.bfloat16)
        sink = torch.zeros(16)
        if name == "dual_nosplit":
            q[:, :, 33] = 0.03125
            q[:, :, 353] = -0.0625
            sink.fill_(0.25)
        exercise_case(m, name, raws, pages, q, ids, lens, sink, 512**-0.5, split, low)
    # Unchanged V4 dispatch: literal all-one KV and one key + unit sink weight.
    raw = torch.zeros((2, KVLayout.V4.page_bytes(16)), dtype=torch.uint8)
    raw[0, :448] = 56
    raw[0, 448:576] = torch.ones(64, dtype=torch.bfloat16).view(torch.uint8)
    raw[0, 16 * 576 : 16 * 576 + 7] = 127
    v4 = raw.cuda()[:, : 16 * 584].view(2, 16, 1, 584)
    result, _ = m.fused_gather_attn_decode_dsv4(
        torch.zeros((1, 16, 512), dtype=torch.bfloat16, device="cuda"),
        v4,
        torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        16,
        512**-0.5,
        attn_sink=torch.zeros(16, device="cuda"),
    )
    torch.cuda.synchronize()
    gate(
        "V4_literal_unchanged",
        bits(result.cpu(), torch.full((1, 16, 512), 0.5, dtype=torch.bfloat16)),
    )
    gate(
        "all_four_native_reader_paths_executed",
        all(getattr(m, n).calls > 0 for n in names),
        calls={n: getattr(m, n).calls for n in names},
    )
    emit(
        "complete",
        passed=True,
        full_model_acceptance=False,
        seconds=time.monotonic() - start,
        peak_gpu_allocated=torch.cuda.max_memory_allocated(),
        peak_gpu_reserved=torch.cuda.max_memory_reserved(),
        gpu_allocator_fraction=0.03,
        fixed_config=FIXED,
    )


if __name__ == "__main__":
    main()
