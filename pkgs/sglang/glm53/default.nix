{
  lib,
  pythonPackages,
  rocmSdk,
  sglang-rocm,
  mscclpp-rocm,
  numactl,
  rdma-core,
  openssl,
}:

# GLM-5.3-Flash compatibility patches and the qualified fixed-BN16 MoE prefill
# implementation, plus numerically qualified native MTP support. The serving
# launcher selects the measured production profile. KDA column layout remains
# excluded; the shared speculative infrastructure does not qualify DFlash.
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
      ./patches/0013-glm53-dsa-kpool-verify-hip.patch
      ./patches/0014-glm53-router-gemv.patch
      ./patches/0015-glm53-bf16-shadow-prefill.patch
      ./patches/0016-glm53-moe-wmma-prefill.patch
      ./patches/0053-glm53-chunked-owned-row-admission.patch
      ./patches/0066-glm53-always-think-tool-grammar.patch
      ./patches/0067-glm53-explicit-reasoning-effort-default.patch
      ./patches/0068-glm53-qualified-fixed-attention-configs.patch
      ./patches/0069-glm53-mooncake-complete-hybrid-state.patch
      ./patches/0070-glm53-mooncake-chunk-checkpoint-write-through.patch
      ./patches/0071-glm53-mooncake-single-kda-restore-destination.patch
      ./patches/0072-glm53-mooncake-byte-diagnostics.patch
      ./patches/0073-glm53-mooncake-kda-producer-fence.patch
      ./patches/0074-glm53-mooncake-bounded-disk-restore.patch
      ./patches/0040-glm53-mtp-verify-widths.patch
      ./patches/0041-glm53-hip-spec-sampling.patch
      ./patches/0042-glm53-dflash-verify-widths.patch
      ./patches/0043-glm53-sampled-draft-graph-tp.patch
      ./patches/0044-glm53-dflash-hip-sampling-tp.patch
      ./patches/0045-glm53-parallel-spec-final-sampling.patch
      ./patches/0046-glm53-live-graph-rows.patch
      ./patches/0047-glm53-live-mtp-draft-rows.patch
      ./patches/0048-glm53-actual-width-metrics.patch
      ./patches/0049-glm53-point-mass-draft-tp.patch
      ./patches/0075-glm53-row-invariant-fp8-gemv.patch
      ./patches/0076-glm53-verify-decode-math.patch
      ./patches/0077-glm53-verify-indexer-decode-math.patch
      ./patches/0078-glm53-mooncake-async-restore.patch
    ];
  };
  sitePackages = pythonPackages.python.sitePackages;
in
base.overrideAttrs (old: {
  postInstall = old.postInstall + ''
    # Native radix-page hashing is a CPU JIT. Pin both its headers and runtime
    # libcrypto instead of relying on the service host's compiler search path.
    substituteInPlace "$out/${sitePackages}/sglang/srt/mem_cache/cpp_utils/native_hash.py" \
      --replace-fail 'extra_cflags = ["-O3"' 'extra_cflags = ["-I${lib.getDev openssl}/include", "-O3"' \
      --replace-fail 'extra_ldflags=["-lcrypto"]' 'extra_ldflags=["${lib.getLib openssl}/lib/libcrypto.so"]'
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
      "$out/bin/sglang-python" ${./tests/mooncake-glm-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${../../../lib/bench/test-glm53-always-think-tool-grammar.py} \
      "$out/${sitePackages}/sglang/srt/entrypoints/openai/serving_chat.py"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${../../../lib/bench/test-glm53-explicit-reasoning-effort.py} \
      "$out/${sitePackages}/sglang/srt/entrypoints/openai/serving_chat.py"
    ROCP_TOOL_LIBRARIES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${../../../lib/bench/test-glm53-chunked-admission.py} \
      --source "$out/${sitePackages}/sglang"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/msar-dispatch-cpu.py} "$out"
    ROCP_TOOL_LIBRARIES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/moe-wmma-dispatch-cpu.py} \
      --source "$out/${sitePackages}/sglang"
    ROCP_TOOL_LIBRARIES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 TRITON_INTERPRET=1 \
      "$out/bin/sglang-python" ${./tests/spec-sampling-cpu.py}
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/test-verify-indexer-decode-math.py} \
      --source "$out/${sitePackages}/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py"
  '';
})
