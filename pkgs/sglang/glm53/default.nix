{
  lib,
  pythonPackages,
  rocmSdk,
  sglang-rocm,
  mscclpp-rocm,
  numactl,
  rdma-core,
}:

# GLM-5.3-Flash campaign runtime: sglang-rocm (same SGLang source and kernels) with extra runtime patches.
#   0001-0003  Codex's FP8 decode/prefill experiments: contiguous batch-one dense GEMV, prefill tiles, TP4 expert GEMV
#   0004       expert guard/grid for batches 1-4 (vendored, not applied yet)
#   0005       opt-in MSCCL++ small-message all-reduce (the DS4 v41 0090 patch rebased onto this SGLang;
#              SGLANG_DSV41_MSAR=1, see sglang/srt/distributed/device_communicators/dsv41_msar.py)
#   0006       opt-in single-launch Triton (add+)RMSNorm for ROCm without vLLM/AITER (SGLANG_ROCM_TRITON_RMSNORM=1),
#              replacing ~11 torch kernels per norm
let
  basePatches =
    let
      dir = ../patches;
      names = lib.sort (a: b: a < b) (
        builtins.filter (lib.hasSuffix ".patch") (builtins.attrNames (builtins.readDir dir))
      );
    in
    map (n: dir + "/${n}") names;
  base = sglang-rocm.override {
    runtimePatches = basePatches ++ [
      ./patches/0001-glm53-fp8-contiguous-decode.patch
      ./patches/0002-glm53-fp8-prefill-config.patch
      ./patches/0003-glm53-fp8-moe-gemv.patch
      ./patches/0005-glm53-mscclpp-allreduce.patch
      ./patches/0006-rocm-triton-rmsnorm.patch
    ];
  };
  sitePackages = pythonPackages.python.sitePackages;
in
base.overrideAttrs (old: {
  postInstall = old.postInstall + ''
    # Prebuild the opt-in MSCCL++ small-message all-reduce (0005). libibverbs is dlopen'ed by MSCCL++; the path is pinned.
    ${rocmSdk}/bin/therock-hip-clang++ -x hip --offload-arch=gfx1151 \
      -std=c++17 -O2 -fPIC -shared -I${mscclpp-rocm}/include \
      -DMSAR_IBV_SO='"${rdma-core}/lib/libibverbs.so.1"' \
      "$out/${sitePackages}/sglang/srt/distributed/device_communicators/dsv41_msar.hip.cpp" \
      -o "$out/${sitePackages}/sglang/srt/distributed/device_communicators/libdsv41_msar.so" \
      -L${mscclpp-rocm}/lib -lmscclpp -L${rocmSdk}/lib -lamdhip64 \
      -Wl,-rpath,${mscclpp-rocm}/lib -Wl,-rpath,${rocmSdk}/lib -Wl,-rpath,${numactl}/lib -Wl,-rpath,${rdma-core}/lib
  '';
  postFixup = (old.postFixup or "") + ''
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${../v41/tests/msar-dispatch-cpu.py} "$out"
  '';
})
