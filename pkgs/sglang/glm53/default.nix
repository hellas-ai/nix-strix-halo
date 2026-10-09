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
#   0004       expert GEMV guard/grid for batches 1-4 (needed once decode runs more than one request)
#   0005       opt-in MSCCL++ small-message all-reduce (the DS4 v41 0090 patch rebased onto this SGLang;
#              SGLANG_DSV41_MSAR=1, see sglang/srt/distributed/device_communicators/dsv41_msar.py)
#   0006       opt-in single-launch Triton (add+)RMSNorm for ROCm without vLLM/AITER (SGLANG_ROCM_TRITON_RMSNORM=1),
#              replacing ~11 torch kernels per norm
#   0007       opt-in fused linear-attention projections under the FP8 checkpoint (SGLANG_GLM53_FUSED_QKVBFG=1): its
#              modules_to_not_convert lists the fused module names, so q/k/v/beta/f_a/g_a run as one BF16 GEMM
#   0008       opt-in lossy int8 TP4 ring all-reduce for prefill-sized messages (SGLANG_GLM53_COMPRESSED_RING=1), the
#              DS4 v41 fused compressed ring generalised to any hidden size, hooked into GroupCoordinator.all_reduce
#   0009       opt-in GC controls inside scheduler processes: SGLANG_GLM53_GC_THRESHOLD, SGLANG_GLM53_GC_LOG_SECS
#   0010       opt-in: linear-attention extend takes its token count from the CPU (SGLANG_GLM53_KDA_NOSYNC=1|check),
#              removing a device-to-host sync per layer from every eager prefill
#   0011       opt-in WMMA block-FP8 GEMV for 2..8 rows on the GLM dense shapes (SGLANG_GLM53_WMMA_GEMV=1), the DS4 v41
#              kernel with GLM's 128x128 block scales; for MTP verify and small prefills
#   0012       SGLANG_GLM53_MOE_GEMV_MAX_ROWS (default 4): rows served by the TP4 FP8 expert GEMV (0003/0004) before
#              the generic fused_moe kernel (~0.9 s for a 12-token step)
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
      ./patches/0004-glm53-fp8-moe-batch4.patch
      ./patches/0005-glm53-mscclpp-allreduce.patch
      ./patches/0006-rocm-triton-rmsnorm.patch
      ./patches/0007-glm53-fused-qkvbfg-under-fp8.patch
      ./patches/0008-glm53-compressed-ring-prefill.patch
      ./patches/0009-glm53-scheduler-gc-controls.patch
      ./patches/0010-glm53-kda-extend-no-sync.patch
      ./patches/0011-glm53-wmma-fp8-gemv-rows.patch
      ./patches/0012-glm53-moe-gemv-max-rows.patch
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
