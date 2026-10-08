"""Compile-only gfx1151 ISA of the (unchanged) mHC post kernel: the same code serves every row count (grid z = rows)."""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sglang.kernels.ops.layernorm.native_hc_post as P
from tcompile import compile_gfx1151, isa_stats

spec = {"X": ('ptr', 'bf16'), "R": ('ptr', 'bf16'), "P": ('ptr', 'fp32'), "C": ('ptr', 'fp32'), "Y": ('ptr', 'bf16'),
        "H": ('const', 5120), "BLOCK": ('const', 256)}
ck = compile_gfx1151(P._hc_post_c2, spec, dict(num_warps=4, num_stages=1, enable_fp_fusion=False, allow_flush_denorm=False))
asm = ck.asm['amdgcn']
st = isa_stats(ck)
sg = re.search(r'\.sgpr_count:\s+(\d+)', asm)
spills = len(re.findall(r'scratch_store|; Spill', asm))
valu = len([l for l in asm.splitlines() if re.match(r'\s+v_', l)])
print(f"_hc_post_c2 H=5120 BLOCK=256 w4: VGPR {st.get('NumVgprs')} SGPR {sg.group(1) if sg else None} scratch {st.get('ScratchSize')} LDS {st.get('LDSByteSize')} "
      f"occupancy {st.get('Occupancy')} spill-like stores {spills} VALU instructions (straight-line, no loop) {valu}")
