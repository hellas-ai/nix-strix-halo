{
  writeShellApplication,
  runCommand,
  bash,
  coreutils,
  gawk,
  gnugrep,
  hostname,
  python3,
  sglang-glm53-rocm,
  glm53-wmma-aot,
  glm53-aot-stage,
  therock-python-wheels,
  rocm-pm4-clr-split,
  rocm-pm4-rocr-split,
  rocm-pm4-bootstrap,
}:

# One GLM-5.3-Flash TP4 rank. GLM_* and SGLANG_* environment overrides remain
# available. Defaults match the deployed, numerically qualified fixed-BN16 MoE
# profile (docs/glm53-roofline-audit.md). GLM_PM4=0 disables PM4.
#
#   glm53-serve RANK [extra SGLang arguments]
let
  site = "${therock-python-wheels}/lib/python3.13/site-packages";
  # The PM4 bootstrap preloads its HIP/HSA runtimes into this wheel set's rocm_sdk: it must be the runtime's own.
  wheelsCheck = runCommand "glm53-serve-wheels-check" { } ''
    ${gnugrep}/bin/grep -q ${therock-python-wheels} ${sglang-glm53-rocm}/bin/sglang-python || {
      echo "sglang-glm53-rocm does not use ${therock-python-wheels}" >&2; exit 1; }
    touch $out
  '';
in
writeShellApplication {
  name = "glm53-serve";
  # The launcher and SGLang's ROCm capacity query must work with a minimal
  # systemd PATH, without relying on the host's interactive environment.
  runtimeInputs = [
    bash
    coreutils
    gawk
    gnugrep
    hostname
  ];
  derivationArgs.wheelsCheck = wheelsCheck;
  text = ''
    rank=''${1:?usage: glm53-serve RANK [extra SGLang arguments]}
    shift
    export GLM_SGLANG_BIN=${sglang-glm53-rocm}/bin/sglang
    export GLM_RUNTIME_ID=${builtins.baseNameOf (toString sglang-glm53-rocm)}
    # Only the validated fixed WMMA specializations are precompiled. Other
    # kernels still JIT; graphs remain process-local. An explicit cache or
    # GLM_AOT=0 keeps the normal writable runtime namespace as a fallback.
    if [[ ''${GLM_AOT:-1} == 1 && -z ''${TRITON_CACHE_DIR:-} ]]; then
      export TRITON_CACHE_DIR
      TRITON_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/glm53/$(hostname)/triton-aot/${builtins.baseNameOf (toString glm53-wmma-aot)}"
      ${glm53-aot-stage}/bin/glm53-aot-stage \
        --seed ${glm53-wmma-aot}/seed --identity ${glm53-wmma-aot}/identity.json \
        --runtime ${sglang-glm53-rocm} --dest "$TRITON_CACHE_DIR"
    fi
    export GLM_PYTHON=${python3}/bin/python3
    export GLM_MODEL_PATH=''${GLM_MODEL_PATH:-/models/GLM-5.3-Flash-FP8}
    export GLM_CONTEXT_LENGTH=''${GLM_CONTEXT_LENGTH:-131072}
    export GLM_MAX_TOTAL_TOKENS=''${GLM_MAX_TOTAL_TOKENS:-131072}
    export GLM_RADIX_CACHE=''${GLM_RADIX_CACHE:-1}
    # Admission and graph capture cover eight independent decoding requests.
    # Smaller batches replay the next captured shape; MTP remains opt-in.
    export GLM_MAX_REQUESTS=''${GLM_MAX_REQUESTS:-8}
    export GLM_CUDA_GRAPH=''${GLM_CUDA_GRAPH:-1}
    export SGLANG_ROCM_TRITON_RMSNORM=''${SGLANG_ROCM_TRITON_RMSNORM:-1}
    export SGLANG_GLM53_FUSED_QKVBFG=''${SGLANG_GLM53_FUSED_QKVBFG:-1}
    export SGLANG_DSV41_MSAR=''${SGLANG_DSV41_MSAR:-1}
    export SGLANG_DSV41_MSAR_PROXY_CPU=''${SGLANG_DSV41_MSAR_PROXY_CPU:-auto}
    export SGLANG_GLM53_WMMA_GEMV=''${SGLANG_GLM53_WMMA_GEMV:-1}
    export SGLANG_GLM53_MOE_GEMV_MAX_ROWS=''${SGLANG_GLM53_MOE_GEMV_MAX_ROWS:-16}
    export SGLANG_GLM53_MOE_WMMA=''${SGLANG_GLM53_MOE_WMMA:-1}
    export SGLANG_GLM53_MOE_WMMA_BLOCK_M=''${SGLANG_GLM53_MOE_WMMA_BLOCK_M:-auto}
    export SGLANG_GLM53_MOE_WMMA_GROUP_M=''${SGLANG_GLM53_MOE_WMMA_GROUP_M:-8}
    export SGLANG_GLM53_MOE_WMMA_STAGES=''${SGLANG_GLM53_MOE_WMMA_STAGES:-1}
    export SGLANG_GLM53_KDA_NOSYNC=''${SGLANG_GLM53_KDA_NOSYNC:-1}
    export SGLANG_GLM53_ROUTER_GEMV=''${SGLANG_GLM53_ROUTER_GEMV:-1}
    # Preserve ordinary-DECODE math when native MTP is explicitly selected.
    # The checks inside these paths restrict them to target verification.
    export SGLANG_GLM53_ROW_INVARIANT_FP8_GEMV=''${SGLANG_GLM53_ROW_INVARIANT_FP8_GEMV:-1}
    export SGLANG_GLM53_VERIFY_DECODE_MATH=''${SGLANG_GLM53_VERIFY_DECODE_MATH:-1}
    export SGLANG_GLM53_MTP_VERIFY_WIDTHS=''${SGLANG_GLM53_MTP_VERIFY_WIDTHS:-4,4,2,2}
    # Prefill chunks of 16+ rows run the dense FP8 projections as BF16 GEMMs (0015, ~1.4 GB per rank).
    export SGLANG_GLM53_BF16_SHADOW_MIN_M=''${SGLANG_GLM53_BF16_SHADOW_MIN_M:-16}
    if [[ ''${GLM_PM4:-1} == 1 ]]; then
      # Retained-PM4 graph replay (pkgs/rocm-pm4-split): the bootstrap sitecustomize loads these runtimes first.
      export PYTHONPATH=${rocm-pm4-bootstrap}''${PYTHONPATH:+:$PYTHONPATH}
      export DS41_PM4_CLR=${rocm-pm4-clr-split}/lib/libamdhip64.so.7
      export DS41_PM4_ROCR=${rocm-pm4-rocr-split}/lib/libhsa-runtime64.so.1
      export DS41_PM4_SITE=${site}
      export DEBUG_HIP_GRAPH_PM4=1 DEBUG_HIP_GRAPH_PM4_UNQUALIFIED=0 DEBUG_HIP_GRAPH_PM4_SPLIT=1
    fi
    exec bash ${../../../lib/bench/glm53-node.sh} "$rank" \
      --max-mamba-cache-size 40 --prefill-decode-interval 0 --cuda-graph-bs-decode 1 2 4 8 "$@"
  '';
  meta.description = "GLM-5.3-Flash TP4 rank on Strix Halo";
}
