#!/usr/bin/env python3
"""Check the DS4 non-AITER mHC route against independent CPU FP64 sums."""

import json
import os
from types import SimpleNamespace

os.environ["SGLANG_USE_AITER"] = "0"
os.environ["SGLANG_OPT_USE_TILELANG_MHC_PRE"] = "0"
os.environ["SGLANG_OPT_USE_TILELANG_MHC_POST"] = "0"

import torch
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer


def reference(x, weight, scale, bias, rms_eps):
    x, weight, scale, bias = [t.cpu().double() for t in (x, weight, scale, bias)]
    flat = x.flatten(1)
    mix = (flat @ weight.T) / (flat.square().mean(-1, keepdim=True) + rms_eps).sqrt()
    pre = (scale[0] * mix[:, :4] + bias[:4]).sigmoid() + 1e-6
    post = 2 * (scale[1] * mix[:, 4:8] + bias[4:8]).sigmoid()
    logits = (scale[2] * mix[:, 8:] + bias[8:]).reshape(-1, 4, 4)
    comb = (logits - logits.amax(-1, keepdim=True)).exp()
    comb = comb / comb.sum(-1, keepdim=True) + 1e-6
    for step in range(20):
        if step:
            comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return (pre.unsqueeze(-1) * x).sum(1), post, comb


def main():
    torch.set_num_threads(4)
    torch.manual_seed(431)
    layer = SimpleNamespace(
        hc_mult=4, rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20
    )
    for tokens, rms_eps in ((n, eps) for eps in (1e-6, 1e-20) for n in (1, 5, 33, 256)):
        layer.rms_norm_eps = rms_eps
        for bias_scale in (1, 10):
            x = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(24, 16384, device="cuda") * 0.01
            scale = torch.full((3,), 0.1, device="cuda")
            bias = torch.randn(24, device="cuda") * bias_scale
            actual = DeepseekV4DecoderLayer.hc_pre(layer, x, weight, scale, bias)
            expected = reference(x, weight, scale, bias, rms_eps)
            assert actual[3] is False
            # This portable route leaves the caller's RMSNorm unfused, even
            # when supplied. It must return the same activation and flag.
            norm = SimpleNamespace(
                weight=torch.randn(4096, device="cuda", dtype=torch.bfloat16),
                variance_epsilon=rms_eps,
            )
            with_norm = DeepseekV4DecoderLayer.hc_pre(
                layer, x, weight, scale, bias, norm=norm
            )
            assert with_norm[3] is False
            for a, b in zip(with_norm[:3], actual[:3], strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for i, (a, b) in enumerate(zip(actual[:3], expected, strict=True)):
                torch.testing.assert_close(
                    a.cpu().double(),
                    b,
                    rtol=0.01 if i == 0 else 1e-4,
                    atol=0.03 if i == 0 else 2e-6,
                )
            residual = torch.randn_like(x)
            y, post, comb, _ = actual
            result = DeepseekV4DecoderLayer.hc_post(layer, y, residual, post, comb)
            post_ref = post.cpu().double().unsqueeze(-1) * y.cpu().double().unsqueeze(
                1
            ) + torch.einsum(
                "sij,sih->sjh", comb.cpu().double(), residual.cpu().double()
            )
            torch.testing.assert_close(
                result.cpu().double(), post_ref, rtol=0.01, atol=0.03
            )
            if tokens in (1, 5):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = DeepseekV4DecoderLayer.hc_pre(
                        layer, x, weight, scale, bias
                    )
                for _ in range(4):
                    graph.replay()
                    for a, b in zip(captured[:3], actual[:3], strict=True):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "rms_eps": rms_eps,
                        "bias_scale": bias_scale,
                        "passed": True,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
