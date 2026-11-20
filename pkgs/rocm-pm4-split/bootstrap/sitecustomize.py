"""DS41 PM4 experiment bootstrap.

Runs the interpreter's own (Nix) sitecustomize first, then, only when DS41_PM4_CLR, DS41_PM4_ROCR and
DS41_PM4_SITE are all set, preloads the retained-PM4 HSA and HIP runtimes into rocm_sdk before anything
imports torch, the same way Codex's qualified PM4 bootstrap does (rocr, then amd_comgr, then clr), so torch
binds to them and only one copy of each is mapped. Fails closed: a broken preload exits 78 instead of
silently running the stock runtime.
"""
import os
import sys

_own = os.path.join(sys.base_prefix, "lib", f"python{sys.version_info[0]}.{sys.version_info[1]}", "site-packages", "sitecustomize.py")
if os.path.exists(_own):
    with open(_own) as _f:
        exec(compile(_f.read(), _own, "exec"))

_clr, _rocr, _site = (os.environ.get(k) for k in ("DS41_PM4_CLR", "DS41_PM4_ROCR", "DS41_PM4_SITE"))
if _clr and _rocr and _site and os.path.isdir(_site) and "torch" not in sys.modules:
    try:
        import ctypes
        import site

        site.addsitedir(_site)
        import rocm_sdk

        rocm_sdk._ALL_CDLLS["hsa-runtime64"] = ctypes.CDLL(_rocr, mode=ctypes.RTLD_GLOBAL)
        rocm_sdk.preload_libraries("amd_comgr")
        rocm_sdk._ALL_CDLLS["amdhip64"] = ctypes.CDLL(_clr, mode=ctypes.RTLD_GLOBAL)
        _id = os.environ.get("SGLANG_CACHE_NUMERICS_ID", "")
        if _id and not _id.endswith("-pm4"):
            os.environ["SGLANG_CACHE_NUMERICS_ID"] = _id + "-pm4"
    except BaseException as _error:
        print("DS41_PM4_BOOTSTRAP_FATAL " + repr(_error), file=sys.stderr, flush=True)
        os._exit(78)
