#!/usr/bin/env python3
"""GPU unit test of the HIP block attention (0085) against the installed Triton kernel and a float32 reference.

Bounded error, stated: the kernels differ only in the fp32 summation order inside the matrix-core accumulators, so the outputs agree to bf16
rounding. The test requires, on realistic-magnitude inputs, (a) the HIP kernel's error against a float32 reference to be no larger than 1.5x the
installed kernel's error (max and rms), (b) the same lonely rows (empty key set -> output 0, lse +inf), (c) the log-sum-exp within 0.05 absolute.

    SGLANG_DSV41_HIP_ATTENTION=1 sglang-python hip-block-attention-gpu.py        # needs a free gfx1151 GPU
"""
import os
import sys

import torch

os.environ["SGLANG_DSV41_HIP_ATTENTION"] = "1"
from sglang.kernels.ops.attention.nsa_triton_decode import dsv41_hip_attention as hip  # noqa: E402
from sglang.kernels.ops.attention.nsa_triton_decode import triton_fp8_attention_fwd  # noqa: E402

dev = "cuda"
torch.manual_seed(0)


def make_cache(P, page):
    """Realistic KV: bf16 values with per-32 amax scaling to e4m3, E8M0 scales (exponent codes 110..130)."""
    x = torch.randn(P, page, 512, device=dev) * torch.rand(P, page, 16, 1, device=dev).repeat_interleave(32, dim=2).view(P, page, 512) * 4
    xg = x.view(P, page, 16, 32)
    amax = xg.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    code = (torch.ceil(torch.log2(amax / 448.0)) + 127).clamp(100, 134)
    q8 = (xg / torch.exp2(code - 127)).clamp(-448, 448).to(torch.float8_e4m3fn)
    data = q8.view(P, page, 512).view(torch.uint8)
    raw = torch.cat([data.reshape(P, -1), code.squeeze(-1).to(torch.uint8).reshape(P, -1)], dim=1)
    deq = (q8.to(torch.float32) * torch.exp2(code - 127)).view(P, page, 512)
    return raw.contiguous().view(P, page, 1, 528), deq


def reference(q, caches, idxs, lens, sink, sm):
    T = q.shape[0]
    qf = q.reshape(T, 16, 512).float()
    out = torch.zeros(T, 16, 512, device=dev)
    lse = torch.zeros(T, 16, device=dev)
    for t in range(T):
        ks = []
        for deq, idx, ln in zip(caches, idxs, lens):
            n = int(ln[t]) if ln is not None else idx.shape[1]
            ii = idx[t, :n].long()
            ii = ii[ii >= 0]
            if len(ii):
                ks.append(deq.reshape(-1, 512)[ii])
        if not ks:
            lse[t] = float("inf")
            continue
        K = torch.cat(ks)
        s = (qf[t] @ K.T) * sm
        m = s.max(1).values
        p = torch.exp(s - m[:, None])
        l = p.sum(1)
        out[t] = (p @ K) / (l + (torch.exp(sink - m) if sink is not None else 0))[:, None]
        lse[t] = m + torch.log(l)
    return out, lse


def case(name, T, topk_m=128, topk_e=512, page_m=256, page_e=128, hole=0.05, lens="rand", extra=True, sink=True, qscale=0.25):
    cm, dm = make_cache(5, page_m)
    q = (torch.randn(T, 1, 16, 512, device=dev) * qscale).to(torch.bfloat16)
    snk = torch.randn(16, device=dev) * 0.5 if sink else None

    def idxs(P, page, topk):
        i = torch.randint(0, P * page, (T, topk), device=dev, dtype=torch.int32)
        i = torch.where(torch.rand(T, topk, device=dev) < hole, torch.full_like(i, -1), i).contiguous()
        if lens == "full":
            ln = torch.full((T,), topk, device=dev, dtype=torch.int32)
        elif lens == "rand":
            ln = torch.randint(0, topk + 1, (T,), device=dev, dtype=torch.int32)
        else:
            ln = torch.zeros(T, device=dev, dtype=torch.int32)
            ln[::3] = topk
        return i, ln

    im, lm = idxs(5, page_m, topk_m)
    kw = dict(q=q, k_cache=cm, head_dim_v=512, softmax_scale=512 ** -0.5, indices=im.view(T, 1, -1), attn_sink=snk, topk_length=lm)
    caches, ids, lns, ce = [dm], [im], [lm], None
    if extra:
        cE, dE = make_cache(11, page_e)
        ie, le = idxs(11, page_e, topk_e)
        kw.update(extra_k_cache=cE, extra_indices_in_kvcache=ie.view(T, 1, -1), extra_topk_length=le)
        caches.append(dE); ids.append(ie); lns.append(le)
    sm = 512 ** -0.5
    ref_o, ref_l = reference(q, caches, ids, lns, snk, sm)
    ho, hl = hip.attention(q, cm, im.view(T, 1, -1), lm, kw.get("extra_k_cache"), kw.get("extra_indices_in_kvcache"), kw.get("extra_topk_length"), snk, sm)
    to, tl_ = triton_fp8_attention_fwd(**kw)
    hl, tl_ = hl.reshape(T, 16), tl_.reshape(T, 16)
    fin = torch.isfinite(ref_l)

    def err(o, l):
        e = (o.reshape(T, 16, 512).float() - ref_o).abs()
        le = torch.where(fin, (l - ref_l).abs(), torch.zeros_like(l))
        return e.max().item(), e.pow(2).mean().sqrt().item(), le.max().item(), bool((torch.isinf(l) == torch.isinf(ref_l)).all()) and not torch.isnan(o).any().item()
    h, t = err(ho, hl), err(to, tl_)
    ok = h[3] and h[0] <= 1.5 * t[0] + 1e-3 and h[1] <= 1.5 * t[1] + 1e-5 and h[2] <= 0.05
    print(f"{'ok  ' if ok else 'FAIL'} {name:28s} T={T:4d} |ref|max {ref_o.abs().max().item():7.2f}  hip max {h[0]:.4f} rms {h[1]:.5f} lse {h[2]:.4f}"
          f"   triton max {t[0]:.4f} rms {t[1]:.5f} lse {t[2]:.4f}")
    return ok


results = [
    case("dual, random lengths", 96),
    case("dual, full lengths", 96, lens="full"),
    case("a third of the rows empty", 96, lens="third"),
    case("no holes", 64, hole=0.0, lens="full"),
    case("heavy holes", 64, hole=0.6, lens="full"),
    case("main scope only", 64, extra=False),
    case("no attention sink", 64, sink=False),
    case("extra page 256", 64, page_e=256),
    case("short windows", 64, topk_m=128, topk_e=512, lens="rand"),
    case("large logits", 64, qscale=2.0, lens="full"),
    case("1 row", 1), case("5 rows", 5), case("17 rows", 17),
    case("prefill chunk", 1536, lens="full"),
]
print("PASS" if all(results) else "FAIL")
sys.exit(0 if all(results) else 1)
