{
  lib,
  cpu ? null,
  preferVectorWidth512 ? cpu == "znver5",
}:

# Optional CPU tuning overlay. Appends `-march=<cpu> -mtune=<cpu>` (and
# `-mprefer-vector-width=512` on znver5) to NIX_CFLAGS_COMPILE_AFTER for
# the few packages where retuning is known to matter (llama-cpp, openblas,
# torch, numpy).
#
# Intentionally minimal — per-CPU rebuilds are a bad cost/benefit trade
# until proven otherwise. Heavier knobs (CPU-specific openblas target,
# llvm target narrowing, etc.) belong elsewhere when needed.

let
  appendCompileFlags =
    flags: pkg:
    pkg.overrideAttrs (old: {
      env = (old.env or { }) // {
        NIX_CFLAGS_COMPILE_AFTER = (old.env.NIX_CFLAGS_COMPILE_AFTER or "") + flags;
      };
    });
in

if cpu == null then
  _: _: { }
else
  let
    cpuFlags = " -march=${cpu} -mtune=${cpu}";
    vectorFlags = lib.optionalString preferVectorWidth512 " -mprefer-vector-width=512";
    compileFlags = cpuFlags + vectorFlags;
    tune = appendCompileFlags compileFlags;
  in
  _: prev: {
    llama-cpp = tune prev.llama-cpp;
  }
