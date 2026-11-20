import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttgir_probe_lib import *
for name, n, k, bk in SHAPES:
    out = {}
    for t in (1, 2):
        kern, rows, tag = kernel_for(mod, t)
        ck = compile_gfx1151(kern, specs(kern, n, k, t, rows, bk), dict(num_warps=4, num_stages=1, enable_fp_fusion=False))
        out[t] = sorted(set(reductions(ck.asm['ttgir'])))
    # compare encodings ignoring the ROWS extent (first dim) of the shape and the threadsPerWarp/warpsPerCTA entry of the rows dim
    import re
    def enc(sig):
        m = re.search(r"sizePerThread = \[([^\]]*)\], threadsPerWarp = \[([^\]]*)\], warpsPerCTA = \[([^\]]*)\], order = \[([^\]]*)\]", sig)
        shape = re.match(r'tensor<([0-9x]+)x', sig).group(1)
        return shape, m.groups() if m else None
    print(name, "c1:", [(a, enc(s)) for a, s in out[1]])
    print(name, "c2:", [(a, enc(s)) for a, s in out[2]])

print("--- C1 with ROWS=8 vs C2 (reduction encodings)")
import isa_gemv
isa_gemv.C1_ROWS = 8
for name, n, k, bk in SHAPES:
    kern, rows, tag = kernel_for(mod, 1)
    assert rows == 8
    ck1 = compile_gfx1151(kern, specs(kern, n, k, 1, rows, bk), dict(num_warps=4, num_stages=1, enable_fp_fusion=False))
    kern2, rows2, _ = kernel_for(mod, 2)
    ck2 = compile_gfx1151(kern2, specs(kern2, n, k, 2, rows2, bk), dict(num_warps=4, num_stages=1, enable_fp_fusion=False))
    e1 = [(a, enc(s)[1]) for a, s in sorted(set(reductions(ck1.asm['ttgir'])))]
    e2 = [(a, enc(s)[1]) for a, s in sorted(set(reductions(ck2.asm['ttgir'])))]
    print(f"{name:15} C1(ROWS=8) reduction encodings identical to C2's: {e1 == e2}")
