"""Compile-only gfx1151 ISA of the draft-chain kernels (TOPK=3) for the tile configs the draft row counts use. No GPU.
Run: scripts/run_overlay.sh <overlay> t3-draft/isa_draft.py <overlay>"""
import json, sys, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from tcompile import compile_gfx1151, isa_stats, loop_hist, classify
SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"
spec = importlib.util.spec_from_file_location("dsv41_mxfp4_draft_isa", SITE / "sglang/kernels/ops/moe/dsv41_mxfp4_draft.py")
D = importlib.util.module_from_spec(spec); sys.modules[spec.name] = D; spec.loader.exec_module(D)
k2 = sys.modules["sglang.kernels.ops.moe.dsv41_mxfp4_decode"]
wm = dict(matrix_instr_nonkdim=16, kpack=1)
out = {}

def rec(tag, ck):
    asm = ck.asm['amdgcn']
    st = isa_stats(ck); st['lds_shared'] = ck.metadata.shared
    lh = loop_hist(asm)
    if lh:
        h, lab, n = lh
        st['loop_instr'] = n; st.update({f'loop_{k}': v for k, v in classify(h).items()})
    st['lines'] = len(asm.splitlines())
    out[tag] = st
    print(tag, st, flush=True)
    (HERE / "isa").mkdir(exist_ok=True)
    (HERE / "isa" / f"{tag}.amdgcn").write_text(asm)

def gate_up(BN, BK, w, s, rp):
    spec_ = {"X": ('ptr', 'bf16'), "W": ('ptr', 'u8'), "S": ('ptr', 'u8'), "IDS": ('ptr', 'i32'), "Y": ('ptr', 'bf16'), "M": ('i32',),
             "K": ('const', 5120), "N": ('const', 1152), "TOPK": ('const', 3), "BLOCK_N": ('const', BN), "BLOCK_K": ('const', BK), "RP": ('const', rp)}
    return compile_gfx1151(k2._gate_up_wmma, spec_, dict(num_warps=w, num_stages=s, **wm))

def down(BN, BK, w, s, rp):
    spec_ = {"A": ('ptr', 'bf16'), "W": ('ptr', 'u8'), "S": ('ptr', 'u8'), "IDS": ('ptr', 'i32'), "TW": ('ptr', 'fp32'), "Y": ('ptr', 'bf16'),
             "M": ('i32',), "K": ('const', 576), "N": ('const', 5120), "TOPK": ('const', 3), "BLOCK_N": ('const', BN), "BLOCK_K": ('const', BK), "RP": ('const', rp)}
    return compile_gfx1151(D._down_wmma_weighted, spec_, dict(num_warps=w, num_stages=s, **wm))

# draft row counts: 3 (batch 1), 6 (batch 2); the same tables serve 1..16
for rows in (1, 2, 3, 4, 5, 6, 7, 8):
    g = k2._cfg(k2.GATE_UP_CFG, rows, 1152, 5120, "GATE_UP"); d = k2._cfg(k2.DOWN_CFG, rows, 5120, 576, "DOWN"); rp = D._route_pad(rows)
    rec(f"G_rows{rows}_BN{g['BLOCK_N']}_BK{g['BLOCK_K']}_w{g['num_warps']}_s{g['num_stages']}_RP{rp}", gate_up(g['BLOCK_N'], g['BLOCK_K'], g['num_warps'], g['num_stages'], rp))
    rec(f"Dw_rows{rows}_BN{d['BLOCK_N']}_BK{d['BLOCK_K']}_w{d['num_warps']}_s{d['num_stages']}_RP{rp}", down(d['BLOCK_N'], d['BLOCK_K'], d['num_warps'], d['num_stages'], rp))

rec("S_bf16_swiglu", compile_gfx1151(D._swiglu_bf16, {"GU": ('ptr', 'bf16'), "Y": ('ptr', 'bf16'), "H": ('const', 576), "LIMIT": ('const', 10.0), "BLOCK": ('const', 512)},
                                    dict(num_warps=4, num_stages=1, enable_fp_fusion=False)))
rec("C_bf16_combine", compile_gfx1151(D._combine_bf16, {"RO": ('ptr', 'bf16'), "OUT": ('ptr', 'bf16'), "factor": ('fp32',), "N": ('const', 5120), "TOPK": ('const', 3),
                                                       "BLOCK": ('const', 512), "HAS_SCALE": ('const', True)}, dict(num_warps=2, num_stages=1, enable_fp_fusion=False)))
rec("mask_ids", compile_gfx1151(D._mask_ids, {"IDS": ('ptr', 'i32'), "IDS_OUT": ('ptr', 'i32'), "LIVE": ('ptr', 'i32'), "n_ids": ('i32',), "TOPK": ('const', 3), "IDP": ('const', 64)},
                                dict(num_warps=1, num_stages=1)))
json.dump(out, open(HERE / "isa_draft.json", "w"), indent=1)
