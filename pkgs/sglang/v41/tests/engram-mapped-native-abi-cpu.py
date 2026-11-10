#!/usr/bin/env python3
"""CPU-only dlopen/signature and invalid-geometry check for 0073 C ABI."""

import ctypes
import os
from pathlib import Path


source_root = os.environ.get("DS41_ENGRAM_SOURCE_ROOT")
library = (Path(source_root) / "sglang/srt/layers/libengram_mapped_cache.so"
           if source_root else
           Path(__file__).resolve().parent / "native-build/libengram_mapped_cache.so")
lib = ctypes.CDLL(str(library))
lib.engram_mapped_create.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t,
]
lib.engram_mapped_create.restype = ctypes.c_int
lib.engram_mapped_launch.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t,
]
lib.engram_mapped_launch.restype = ctypes.c_int
lib.engram_mapped_destroy.argtypes = [ctypes.c_void_p]
lib.engram_mapped_slots_used.argtypes = [ctypes.c_void_p]
lib.engram_mapped_slots_used.restype = ctypes.c_int

handle = ctypes.c_void_p()
error = ctypes.create_string_buffer(256)
code = lib.engram_mapped_create(
    -1, -1, 0, 0, 10, 0, 10, 0, 1, 0,
    ctypes.byref(handle), error, len(error),
)
assert code == 1 and not handle.value
assert b"invalid bounded Engram cache geometry" in error.value
assert lib.engram_mapped_slots_used(None) == -1
lib.engram_mapped_destroy(None)
print("PASS: mapped C ABI loads, rejects geometry before HIP init, exports launch/destroy/diagnostic")
