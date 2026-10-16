"""Compile-only gfx1151 ISA of kernel P (`_prep_operand`) and its live-row variant (`_prep_operand_live`). No GPU.
Run: S-exact-rows/scripts/run_overlay.sh <overlay> t1-mask/isa_prep.py <overlay>"""
import os, sys, json, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from tcompile import compile_gfx1151, isa_stats
SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"
spec = importlib.util.spec_from_file_location("dsv41_isa", SITE / "sglang/kernels/ops/moe/dsv41_mxfp4_decode.py")
P = importlib.util.module_from_spec(spec); sys.modules[spec.name] = P; spec.loader.exec_module(P)
opts = dict(num_warps=1, num_stages=1, enable_fp_fusion=False)
base = {"X": ('ptr', 'bf16'), "Y": ('ptr', 'bf16'), "n_groups": ('i32d',), "G": ('const', 8), "eps": ('const', P.EPS), "fp8_max": ('const', P.FP8_MAX)}
live = {"X": ('ptr', 'bf16'), "Y": ('ptr', 'bf16'), "n_groups": ('i32d',), "IDS": ('ptr', 'i32'), "IDS_OUT": ('ptr', 'i32'),
        "LIVE": ('ptr', 'i32'), "n_ids": ('i32',), "G": ('const', 8), "eps": ('const', P.EPS), "fp8_max": ('const', P.FP8_MAX),
        "TOPK": ('const', 6), "IDP": ('const', P.MAX_ROUTES)}
out = {}
for name, fn, spec_ in (("_prep_operand", P._prep_operand, base), ("_prep_operand_live", P._prep_operand_live, live)):
    ck = compile_gfx1151(fn, spec_, opts)
    asm = ck.asm['amdgcn']
    st = isa_stats(ck)
    st["lines"] = len(asm.splitlines())
    st["global_loads"] = asm.count("global_load")
    st["global_stores"] = asm.count("global_store")
    out[name] = st
    (HERE / f"isa_{name}.amdgcn").write_text(asm)
    print(name, st)
json.dump(out, open(HERE / "isa_prep.json", "w"), indent=1)
