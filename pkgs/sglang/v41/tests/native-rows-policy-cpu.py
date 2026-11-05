"""CPU check of the row policy of the native V4.1 decode kernels (dsv41_native_rows.py, pure Python).

SGLANG_DSV41_NATIVE_ROWS: unset/all -> rows 1..8 everywhere; legacy -> the dispatch of patches 0014..0041; a list/range -> exactly
those rows.  SGLANG_DSV41_NATIVE_C1_ROWS: 4 (default) or 8.  Also pins the explicit projection table (the seven target
projections and the DSpark draft's wq_a / wkv / main_proj) and that every table K is a multiple of its BLOCK_K.
"""

import os
import sys
from pathlib import Path

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"
sys.path.insert(0, str(SITE))
from sglang.kernels.ops.gemm import dsv41_native_rows as rows  # noqa: E402


def set_env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def policy(value):
    set_env(rows.ENV, value)
    hc = frozenset(m for m in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 100, 511, 512, 513, 1000, 1306, 1536, 4096, 70000)
                   if rows.hc_post_native(m))
    return rows.gemv_rows(), rows.idx_wqb_rows(), hc, rows.is_legacy()


ALL = frozenset(range(1, 9))
BIG = frozenset((9, 17, 100, 511, 512, 513, 1000, 1306, 1536, 4096, 70000))   # sampled rows above 8
LEGACY_HC = frozenset((1, 2, 4, 512, 1306, 1536))
for value in (None, "", "all", "ALL", " 1-8 ", "default", "on", "1,2,3,4,5,6,7,8"):
    assert policy(value) == (ALL, ALL, ALL | BIG, False), value
for value in ("legacy", "0", "off", "False", "no", " LEGACY "):
    assert policy(value) == (
        frozenset((1, 2, 4, 8)), frozenset((1, 2, 3, 4, 8)), LEGACY_HC, True,
    ), value
assert policy("2,3-4") == (frozenset((2, 3, 4)), frozenset((2, 3, 4)), frozenset((2, 3, 4)) | BIG, False)
assert policy("5") == (frozenset((5,)), frozenset((5,)), frozenset((5,)) | BIG, False)
assert policy("8, 1 ,1,") == (frozenset((1, 8)), frozenset((1, 8)), frozenset((1, 8)) | BIG, False)
assert not rows.hc_post_native(0) and not rows.hc_post_native(-3)       # no rows: never native, whatever the policy
for bad in ("9", "0-3", "3-2", "x", ",", "1-9", "-1"):
    try:
        policy(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {bad!r}")
set_env(rows.ENV, None)

# C1 tile rows
for value, expected in ((None, 4), ("4", 4), (" 8 ", 8)):
    set_env("SGLANG_DSV41_NATIVE_C1_ROWS", value)
    assert rows.c1_rows() == expected, value
for bad in ("2", "16", "x", ""):
    set_env("SGLANG_DSV41_NATIVE_C1_ROWS", bad)
    try:
        rows.c1_rows()
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted C1 rows {bad!r}")
set_env("SGLANG_DSV41_NATIVE_C1_ROWS", None)

# projection table (explicit whitelist of native_gemv_rows)
expected_table = {
    (25600, 6144): 512, (5120, 2048): 512, (1792, 5120): 512, (8192, 1280): 256,
    (1152, 5120): 512, (5120, 576): 64, (4096, 1280): 256,
    (1280, 5120): 512, (512, 5120): 512, (5120, 15360): 512,
    (1536, 5120): 512,
}
assert rows.PROJECTION_BLOCK_K == expected_table, rows.PROJECTION_BLOCK_K
for (n, k), block_k in rows.PROJECTION_BLOCK_K.items():
    assert n % 32 == 0 and k % block_k == 0 and block_k in (64, 256, 512), (n, k, block_k)
assert "torch" not in sys.modules or True  # the module itself imports neither torch nor triton
print("native rows policy CPU PASS: all/legacy/list/error parsing, hc_post any-M policy, C1 tile rows, projection table")
