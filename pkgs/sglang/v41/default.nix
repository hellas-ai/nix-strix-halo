{
  lib,
  callPackage,
  fetchurl,
  pythonPackages,
  rocmSdk,
  sglang-rocm,
  mscclpp-rocm,
  numactl,
  rdma-core,
}:

let
  revision = "fdf14605e5791ad3bcb93b27a495d447f2746838";
  version = "0.5.20.post1.dev41308";
  src = fetchurl {
    name = "sglang-${revision}.tar.gz";
    url = "https://github.com/sgl-project/sglang/archive/${revision}.tar.gz";
    hash = "sha256-oXyEki1hlrdnqFW7ySzYZNQJiIPRGFrU41VyyzO/mAQ=";
  };
  nativeExtensions = callPackage ./native-extensions.nix {
    inherit
      src
      revision
      version
      pythonPackages
      ;
  };
  base = sglang-rocm.override {
    runtimePatches = [
      ./patches/0001-portable-mxfp4.patch
      ./patches/0002-portable-mhc.patch
      ./patches/0003-fp4-indexer.patch
      ./patches/0004-fp8-ue8m0.patch
      ./patches/0005-engram-store.patch
      ./patches/0006-engram-loader.patch
      ./patches/0007-engram-model.patch
      ./patches/0008-vision-reference-precision.patch
      ./patches/0009-bounded-weight-loading.patch
      ./patches/0010-swa-autotune-head-tiles.patch
      ./patches/0011-rccl-graph-usage-mode.patch
      ./patches/0012-topk-logical-ties.patch
      ./patches/0013-engram-decode-graphs.patch
      ./patches/0014-v41-native-fp8-gemv.patch
      ./patches/0015-stable-candidate-block-ties.patch
      ./patches/0016-native-hc-projection.patch
      ./patches/0017-v41-native-shared-expert-gemv.patch
      ./patches/0018-c2-gemv-eight-row-tiles.patch
      ./patches/0019-native-gemv-exact-fp8-conversion.patch
      ./patches/0020-native-hc-post-c2.patch
      ./patches/0021-mxfp4-routed-scale-ownership.patch
      ./patches/0022-engram-graph-buckets.patch
      ./patches/0023-mxfp4-live-row-pairs.patch
      ./patches/0024-native-fp8-c4-c8.patch
      ./patches/0025-chunked-owned-row-admission.patch
      ./patches/0026-software-fp8-rounding.patch
      ./patches/0027-openai-byte-tokenizer-cache.patch
      ./patches/0028-engram-capacity-prefixes.patch
      ./patches/0029-native-hc-projection-rows.patch
      ./patches/0030-v41-triton-kv-reader.patch
      ./patches/0031-v41-kv-scale-floor.patch
      ./patches/0032-v41-routed-operand-policy.patch
      ./patches/0033-mxfp4-prefill-m32.patch
      ./patches/0034-native-hc-post-c1-c4.patch
      ./patches/0035-fp8-prefill-exact-conversion.patch
      ./patches/0036-native-hc-post-prefill.patch
      ./patches/0037-v41-gfx1151-split-attention.patch
      ./patches/0038-v41-gfx1151-wmma-indexer.patch
      ./patches/0039-engram-graph-pad-to-bucket.patch
      ./patches/0040-v41-gfx1151-native-rmsnorm-add.patch
      ./patches/0041-v41-gfx1151-indexer-wqb-native-gemv.patch
      ./patches/0042-v41-gfx1151-mxfp4-decode.patch
      ./patches/0043-mooncake-restore-before-publish.patch
      ./patches/0044-mooncake-cache-namespace.patch
      ./patches/0045-engram-dspark-graphs.patch
      ./patches/0046-v41-gfx1151-mask-padded-rows.patch
      ./patches/0047-v41-gfx1151-row-generic-decode.patch
      ./patches/0048-v41-gfx1151-draft-exact-rows.patch
      ./patches/0049-exact-decode-graph-buckets.patch
      ./patches/0050-v41-gfx1151-indexer-runtime-lengths.patch
      ./patches/0051-v41-gfx1151-bf16-shadow-dense-prefill.patch
      ./patches/0052-dspark-adaptive-verify-width.patch
      ./patches/0053-v41-gfx1151-engram-m8-tile4.patch
      ./patches/0054-v41-gfx1151-m8-gemv-launches.patch
      ./patches/0057-v41-prefill-compressed-moe-allreduce.patch
      ./patches/0058-v41-prefill-mxfp4-gate-geometry.patch
      ./patches/0059-engram-async-hash-ids.patch
      ./patches/0060-mooncake-async-host-indices.patch
      ./patches/0061-engram-gpu-row-cache.patch
      ./patches/0062-native-rmsnorm-prefill.patch
      ./patches/0063-v41-fused-compressed-ring-prefill.patch
      ./patches/0064-native-mhc-stats.patch
      ./patches/0065-v41-parallel-indexer-score.patch
      ./patches/0069-v41-paired-indexer-k-reuse.patch
      ./patches/0070-engram-deferred-history-preflight.patch
      ./patches/0071-engram-native-preflight-before-model.patch
      ./patches/0072-engram-fatal-result-status.patch
      ./patches/0073-engram-mapped-cache-fatal-hook.patch
      ./patches/0074-v41-checkpoint-attention.patch
      ./patches/0077-v41-prefill-shared-routed-overlap.patch
      ./patches/0078-v41-prefill-mhc-stats-overlap.patch
      ./patches/0079-v41-gfx1151-prefill-fused-combine.patch
      ./patches/0067-dspark-draft-row-chunks.patch
      ./patches/0068-v41-wmma-decode-gemv.patch
      ./patches/0080-scheduler-timing-record.patch
      ./patches/0081-mooncake-namespace-swa-replay-flags.patch
      ./patches/0082-swa-replay-refuse-prompt-logprobs.patch
      ./patches/0083-prefill-tail-merge.patch
      ./patches/0084-timing-record-never-raises.patch
      ./patches/0085-v41-hip-block-attention.patch
      ./patches/0086-v41-mxfp4-prefill-tiles.patch
      ./patches/0087-v41-hip-block-indexer.patch
      ./patches/0088-v41-prefill-quant-combine-tuning.patch
      ./patches/0089-v41-prefill-moe-glue.patch
      ./patches/0090-v41-mscclpp-allreduce.patch
      ./patches/0100-v41-moe-grouped-gemm.patch
      ./patches/0101-v41-moe-grouped-decode.patch
      ./patches/0102-v41-prefill-split-overlap.patch
    ];
  };
  baseKernel = lib.findFirst (
    p: lib.hasPrefix "sglang-kernel-" (p.pname or "")
  ) (throw "DeepSeek V4.1 requires the native SGLang kernel package") base.dependencies;
  kernel = baseKernel.overridePythonAttrs (old: {
    pname = "sglang-kernel-v41-gfx1151";
    version = "0.4.7-${builtins.substring 0 8 revision}";
    inherit src;
    sourceRoot = "sglang-${revision}/python/sglang/kernels/aot";
    patches = (old.patches or [ ]) ++ [ ./kernel-patches/0001-topk-logical-ties-overflow.patch ];
  });
in
base.overridePythonAttrs (old: {
  pname = "sglang-v41-rocm-gfx1151";
  inherit version src;
  sourceRoot = "sglang-${revision}/python";
  format = "pyproject";
  postPatch = (old.postPatch or "") + ''
    # The extensions are built separately with the upstream Rust build hooks.
    substituteInPlace pyproject.toml \
      --replace-fail '"setuptools-rust>=1.11",' ""
  '';
  build-system = with pythonPackages; [
    setuptools
    setuptools-scm
    torch
    wheel
  ];
  dependencies = map (p: if p == baseKernel then kernel else p) old.dependencies;
  buildInputs = (old.buildInputs or [ ]) ++ [ rocmSdk ];
  env = (old.env or { }) // {
    SETUPTOOLS_SCM_PRETEND_VERSION = version;
    SGLANG_BUILD_RUST_EXTS = "none";
  };
  postInstall = old.postInstall + ''
    # Prebuild the opt-in mapped Engram gather; no compiler or JIT runs in a
    # serving request. The adjacent Python loader resolves this exact library.
    ${rocmSdk}/bin/therock-hip-clang++ -x hip --offload-arch=gfx1151 \
      -std=c++20 -O2 -Wall -Wextra -fPIC -shared -pthread \
      "$out/${pythonPackages.python.sitePackages}/sglang/srt/layers/engram_mapped_cache.hip.cpp" \
      -o "$out/${pythonPackages.python.sitePackages}/sglang/srt/layers/libengram_mapped_cache.so"
    # Prebuild the opt-in WMMA prefill attention (0085, SGLANG_DSV41_HIP_ATTENTION); the Python wrapper loads this exact library.
    ${rocmSdk}/bin/therock-hip-clang++ -x hip --offload-arch=gfx1151 \
      -std=c++20 -O3 -Wall -Wextra -fPIC -shared \
      "$out/${pythonPackages.python.sitePackages}/sglang/kernels/ops/attention/nsa_triton_decode/dsv41_attn_block.hip.cpp" \
      -o "$out/${pythonPackages.python.sitePackages}/sglang/kernels/ops/attention/nsa_triton_decode/libdsv41_attn_block.so"
    # Prebuild the opt-in WMMA FP4 indexer (0087, SGLANG_DSV41_HIP_INDEXER).
    ${rocmSdk}/bin/therock-hip-clang++ -x hip --offload-arch=gfx1151 \
      -std=c++20 -O3 -Wall -Wextra -fPIC -shared \
      "$out/${pythonPackages.python.sitePackages}/sglang/kernels/ops/attention/dsv4/dsv41_indexer_block.hip.cpp" \
      -o "$out/${pythonPackages.python.sitePackages}/sglang/kernels/ops/attention/dsv4/libdsv41_indexer_block.so"
    # Prebuild the opt-in grouped WMMA MXFP4 MoE GEMMs (0100/0101, SGLANG_DSV41_MOE_GROUPED, SGLANG_DSV41_MOE_GROUPED_DECODE).
    ${rocmSdk}/bin/therock-hip-clang++ -x hip --offload-arch=gfx1151 \
      -std=c++20 -O3 -Wall -Wextra -fPIC -shared \
      "$out/${pythonPackages.python.sitePackages}/sglang/kernels/ops/moe/dsv41_moe_grouped.hip.cpp" \
      -o "$out/${pythonPackages.python.sitePackages}/sglang/kernels/ops/moe/libdsv41_moe_grouped.so"
    # Prebuild the opt-in MSCCL++ small-message all-reduce (0090, SGLANG_DSV41_MSAR). libibverbs is dlopen'ed by MSCCL++; the path is pinned.
    ${rocmSdk}/bin/therock-hip-clang++ -x hip --offload-arch=gfx1151 \
      -std=c++17 -O2 -fPIC -shared -I${mscclpp-rocm}/include \
      -DMSAR_IBV_SO='"${rdma-core}/lib/libibverbs.so.1"' \
      "$out/${pythonPackages.python.sitePackages}/sglang/srt/distributed/device_communicators/dsv41_msar.hip.cpp" \
      -o "$out/${pythonPackages.python.sitePackages}/sglang/srt/distributed/device_communicators/libdsv41_msar.so" \
      -L${mscclpp-rocm}/lib -lmscclpp -L${rocmSdk}/lib -lamdhip64 \
      -Wl,-rpath,${mscclpp-rocm}/lib -Wl,-rpath,${rocmSdk}/lib -Wl,-rpath,${numactl}/lib -Wl,-rpath,${rdma-core}/lib
    extension_suffix="$(${pythonPackages.python.interpreter} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
    for module_stem in ${lib.escapeShellArgs nativeExtensions.moduleStems}; do
      ln -s "${nativeExtensions}/${pythonPackages.python.sitePackages}/$module_stem$extension_suffix" \
        "$out/${pythonPackages.python.sitePackages}/$module_stem$extension_suffix"
    done
  '';
  # Run after the inherited wrapper construction so the exact wheel and library
  # environment is available. The regression uses CPU tensors and device stubs.
  postFixup = (old.postFixup or "") + ''
    # The preflight/result-gate tests use CPU tensors and device stubs. The
    # native ABI test loads the prebuilt library without initializing a GPU.
    for test_script in ${lib.escapeShellArgs [
      ./tests/engram-deferred-history-cpu.py
      ./tests/engram-preflight-status-cpu.py
      ./tests/engram-native-preflight-cpu.py
      ./tests/engram-external-graph-rows-cpu.py
      ./tests/engram-result-gate-cpu.py
      ./tests/engram-mapped-preflight-cpu.py
      ./tests/engram-mapped-native-abi-cpu.py
]}; do
      HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
        ROCPROFILER_REGISTER_FORCE_LOAD=0 ROCP_TOOL_LIBRARIES="" \
        DS41_ENGRAM_SOURCE_ROOT="$out/${pythonPackages.python.sitePackages}" \
        "$out/bin/sglang-python" "$test_script"
    done
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/msar-dispatch-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/split-overlap-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/adaptive-dspark-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/draft-row-chunks-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/wmma-gemv-policy-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/mooncake-cache-namespace-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/mooncake-restore-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
      "$out/bin/sglang-python" ${./tests/routed-scaling.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
      "$out/bin/sglang-python" ${./tests/routed-policy-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
      "$out/bin/sglang-python" ${./tests/graph-buckets.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/exact-graph-buckets.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
      "$out/bin/sglang-python" ${./tests/engram-graph-prestage-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
      "$out/bin/sglang-python" ${./tests/prefill-dispatch.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/split-attention-dispatch.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/checkpoint-attention-dispatch.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
      "$out/bin/sglang-python" ${./tests/mxfp4-decode-dispatch.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/indexer-recompile-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/paired-indexer-dispatch-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/bf16-shadow-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/shadow-native-compose-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
      "$out/bin/sglang-python" ${./tests/native-dispatch-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
      "$out/bin/sglang-python" ${./tests/native-rows-policy-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/m8-gemv-launch-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
      "$out/bin/sglang-python" ${./tests/scheduler-owned-row.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/live-row-mask-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/draft-moe-dispatch-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/draft-projection-flags-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/sched-timing-record-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/swa-replay-guard-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/tail-merge-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/hip-attention-dispatch-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/hip-indexer-dispatch-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$out/bin/sglang-python" ${./tests/moe-grouped-dispatch-cpu.py} "$out"
    HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
      "$out/bin/sglang-python" ${./tests/openai-byte-tokenizer-cache.py}
  '';
  passthru = (old.passthru or { }) // {
    inherit nativeExtensions;
  };
  pythonImportsCheck = [
    "sglang"
    "sglang.srt.configs.deepseek_v41"
    "sglang.srt.server_args"
    "sglang.srt.mem_cache.rust_tree_core.mem_cache"
    "sglang.srt.rust_extensions._multimodal"
  ];
  meta = old.meta // {
    description = "DeepSeek V4.1 SGLang candidate with native gfx1151 kernels";
  };
})
