"""Fixtures for the DeepSeek-V4.1 TP4 routed-expert decode microbenchmarks (rank-0 geometry).

Shapes: E=384 experts, hidden K1=5120, per-rank intermediate H=576:
  w13 [E,1152,2560] u8 (gate rows 0:576 then up rows 576:1152; low nibble = even k)   s13 [E,1152,160] e8m0
  w2  [E,5120,288]  u8                                                                 s2  [E,5120,18]  e8m0
"""
import json, mmap, os, struct
import torch

E, K1, H, N1, N2, TOPK = 384, 5120, 576, 1152, 5120, 6
CKPT = "/models/DeepSeek-V4.1-Flash-hf-dba1be0a"

def gen_scales(shape, device, lo=112, hi=124, gen=None):
    """E8M0 bytes. Real DeepSeek scales are narrow; lo/hi can be widened (including 0/255) for corner tests."""
    return torch.randint(lo, hi + 1, shape, dtype=torch.uint8, device=device, generator=gen)

def synth_layer(device, seed=0, scale_lo=112, scale_hi=124):
    g = torch.Generator(device=device); g.manual_seed(seed)
    w13 = torch.randint(0, 256, (E, N1, K1 // 2), dtype=torch.uint8, device=device, generator=g)
    w2 = torch.randint(0, 256, (E, N2, H // 2), dtype=torch.uint8, device=device, generator=g)
    s13 = gen_scales((E, N1, K1 // 32), device, scale_lo, scale_hi, g)
    s2 = gen_scales((E, N2, H // 32), device, scale_lo, scale_hi, g)
    return w13, w2, s13, s2

def _st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def _read_tensor(path, hdr, base, name):
    info = hdr[name]; b, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + b); raw = f.read(e - b)
    return raw, info["shape"]

def real_experts(layer=5, experts=range(48), rank=0, tp=4):
    """Return (w13 [n,1152,2560] u8, s13 [n,1152,160] u8, w2 [n,5120,288] u8, s2 [n,5120,18] u8) CPU tensors for TP rank slice."""
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    out = {k: [] for k in ("w13", "s13", "w2", "s2")}
    cache = {}
    def tens(name):
        f = idx[name]
        if f not in cache: cache[f] = _st_header(f"{CKPT}/{f}")
        hdr, base = cache[f]
        raw, shape = _read_tensor(f"{CKPT}/{f}", hdr, base, name)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape)
    r0, r1 = rank * (2304 // tp), (rank + 1) * (2304 // tp)
    for e in experts:
        p = f"layers.{layer}.ffn.experts.{e}."
        w1, w3 = tens(p + "w1.weight")[r0:r1], tens(p + "w3.weight")[r0:r1]
        s1, s3 = tens(p + "w1.scale")[r0:r1], tens(p + "w3.scale")[r0:r1]
        w2 = tens(p + "w2.weight")[:, rank * 288:(rank + 1) * 288]
        s2 = tens(p + "w2.scale")[:, rank * 18:(rank + 1) * 18]
        out["w13"].append(torch.cat([w1, w3], 0)); out["s13"].append(torch.cat([s1, s3], 0))
        out["w2"].append(w2.contiguous()); out["s2"].append(s2.contiguous())
    return tuple(torch.stack(out[k]) for k in ("w13", "s13", "w2", "s2"))

def tile_to_full(real, device):
    """Tile n real experts across all 384 slots."""
    w13, s13, w2, s2 = real
    n = w13.shape[0]
    sel = torch.arange(E) % n
    return (w13[sel].to(device), w2[sel].to(device), s13[sel].to(device), s2[sel].to(device))

def realistic_activations(M, device, seed=1, dtype=torch.bfloat16):
    """Post-RMSNorm-like hidden states: heavy tailed with a few outlier channels (synthetic; not captured from the model)."""
    g = torch.Generator(device=device); g.manual_seed(seed)
    x = torch.randn(M, K1, device=device, generator=g)
    chan = torch.exp(0.6 * torch.randn(K1, device=device, generator=g))
    chan[torch.randint(0, K1, (8,), device=device, generator=g)] *= 25.0
    x = x * chan * 0.8
    return x.to(dtype)

def make_routes(M, device, seed=2, routing="random"):
    """topk ids [M,6] int32 (distinct within a token) and normalised fp32 weights.

    routing: 'random'  -> independent uniform top-6 per token (about 22 distinct experts at M=4)
             'shared'  -> every token uses the same 6 experts in a different slot order (6 distinct experts)
             'mixed'   -> 3 experts shared by all tokens + 3 private ones per token
    """
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    ids = torch.empty(M, TOPK, dtype=torch.int32)
    base = torch.randperm(E, generator=g)[:TOPK]
    for t in range(M):
        if routing == "random":
            row = torch.randperm(E, generator=g)[:TOPK]
        elif routing == "shared":
            row = base[torch.randperm(TOPK, generator=g)]
        elif routing == "mixed":
            priv = torch.randperm(E, generator=g)
            priv = priv[~torch.isin(priv, base[:3])][:3]
            row = torch.cat([base[:3], priv])[torch.randperm(TOPK, generator=g)]
        else:
            raise ValueError(routing)
        ids[t] = row.to(torch.int32)
    w = torch.rand(M, TOPK, generator=g) + 0.05
    w = (w / w.sum(1, keepdim=True)).float()
    return ids.to(device), w.to(device)
