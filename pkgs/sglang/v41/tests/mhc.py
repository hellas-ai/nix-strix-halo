"""V4.1 portable mHC: literal FP64 reference and changing-data graph replay."""
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['SGLANG_USE_AITER'] = '0'
os.environ['SGLANG_OPT_USE_TILELANG_MHC_PRE'] = '0'
os.environ['SGLANG_OPT_USE_TILELANG_MHC_POST'] = '0'

import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer as Layer


def coeff_reference(x, weight, scale, bias, rms_eps):
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
    return pre, post, comb


def norm_reference(x, weight, eps):
    # The BF16 combine is observable before the caller's RMSNorm.
    x = x.to(torch.bfloat16).float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (x * weight.cpu().float()).to(torch.bfloat16)


def main():
    runtime = Path(sys.argv[1]).resolve()
    assert Path(inspect.getfile(Layer)).resolve().is_relative_to(runtime)
    assert 'aiter.dist.device_communicators.custom_all_reduce' not in sys.modules
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx1151'
    torch.cuda.set_per_process_memory_fraction(0.03)
    torch.set_num_threads(4)
    torch.manual_seed(431)
    layer = SimpleNamespace(hc_mult=4, rms_norm_eps=1e-20, hc_eps=1e-6,
                            hc_sinkhorn_iters=20, hc_pre_from_prev_sublayer=True)
    print(json.dumps({'event':'provenance','runtime':str(runtime),'torch':torch.__version__}), flush=True)
    cases = 0
    maximum = 0.0
    for tokens in (0, 1, 2, 5, 33, 128):
        for eps in (1e-6, 1e-20):
            layer.rms_norm_eps = eps
            for bias_scale in (1, 10):
                x = torch.randn(tokens, 4, 5120, device='cuda', dtype=torch.bfloat16)
                if tokens > 1:
                    x[0].zero_()
                fn = torch.randn(24, 20480, device='cuda') * .01
                scale = torch.tensor([.1, .2, .125], device='cuda')
                bias = torch.randn(24, device='cuda') * bias_scale
                norm = RMSNorm(5120, eps=eps).to(device='cuda', dtype=torch.bfloat16)
                norm.weight.data.uniform_(.5, 1.5)
                actual = Layer.hc_pre(layer, x, fn, scale, bias, norm=norm)
                assert actual[3] is False
                pre, post, comb = coeff_reference(x, fn, scale, bias, eps)
                expected = (pre.unsqueeze(-1) * x.cpu().double()).sum(1)
                for a, b in zip(actual[:3], (expected, post, comb), strict=True):
                    torch.testing.assert_close(a.cpu().double(), b,
                        rtol=.01 if a.dtype == torch.bfloat16 else 1e-4,
                        atol=.03 if a.dtype == torch.bfloat16 else 2e-6)
                stats = Layer._hc_mix_stats(layer, x, fn, scale, bias)
                for a, b in zip(stats, (pre, post, comb), strict=True):
                    torch.testing.assert_close(a.cpu().double(), b, rtol=1e-4, atol=2e-6)
                    if a.numel():
                        maximum = max(maximum, float((a.cpu().double() - b).abs().max()))
                residual = torch.randn_like(x)
                y, p, c, _ = actual
                result = Layer.hc_post(layer, y, residual, p, c)
                ref_post = p.cpu().double().unsqueeze(-1) * y.cpu().double().unsqueeze(1)
                ref_post += torch.einsum('sij,sih->sjh', c.cpu().double(), residual.cpu().double())
                torch.testing.assert_close(result.cpu().double(), ref_post, rtol=.01, atol=.03)
                if tokens:
                    # V4.1 consumes the PREVIOUS sublayer's pre coefficients.
                    previous_pre = torch.rand(tokens, 4, device='cuda')
                    combined = Layer._hc_combine(layer, x, previous_pre, norm)
                    previous_ref = (previous_pre.cpu().double().unsqueeze(-1) * x.cpu().double()).sum(1)
                    torch.testing.assert_close(combined.cpu(), norm_reference(previous_ref, norm.weight, eps), rtol=.015, atol=.03)
                    initial = Layer._hc_combine(layer, x, None, norm)
                    torch.testing.assert_close(initial.cpu(), norm_reference(x[:,0].cpu(), norm.weight, eps), rtol=.015, atol=.03)
                if tokens in (1, 5) and eps == 1e-20 and bias_scale == 1:
                    def forward():
                        a = Layer._hc_mix_stats(layer, x, fn, scale, bias)
                        b = Layer.hc_post(layer, y, residual, a[1], a[2])
                        return (*a, b)
                    forward()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = forward()
                    for change in (0.25, -0.75, 1.5):
                        x.mul_(change)
                        residual.add_(change)
                        bias.add_(change)
                        eager = forward()
                        graph.replay()
                        for a, b in zip(captured, eager, strict=True):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                cases += 1
                print(json.dumps({'tokens':tokens,'eps':eps,'bias_scale':bias_scale,'passed':True}), flush=True)
    print(json.dumps({'event':'complete','cases':cases,'max_coefficient_abs_error':maximum,
                      'max_memory_allocated':torch.cuda.max_memory_allocated()}), flush=True)


if __name__ == '__main__':
    main()
