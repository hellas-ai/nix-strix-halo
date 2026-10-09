{
  lib,
  writeShellApplication,
  runCommand,
  gnugrep,
  python3,
  sglang-glm53-rocm,
  therock-python-wheels,
  rocm-pm4-clr-split,
  rocm-pm4-rocr-split,
  rocm-pm4-bootstrap,
}:

# One GLM-5.3-Flash TP4 rank with the configuration qualified on 2026-10-09: every GLM opt-in on (Triton RMSNorm, fused
# linear-attention projections, MSCCL++ small all-reduce, WMMA FP8 GEMV for 2-8 rows, expert GEMV to 16 rows, sync-free
# KDA extend, router GEMV, BF16 dense projections in prefill), CUDA graphs with retained-PM4 replay, 128K context with
# the radix cache, up to four running requests. 22.5 tok/s decode at 12 and 14.5K tokens of context (60% of the 37.4 tok/s
# weight-read floor); a cold 14.5K-token prefill in ~44 s. Teacher-forced logprobs move no more than a server restart
# does, and natural-text NLL matches the stock runtime. The GLM_* knobs of lib/bench/glm53-node.sh still override;
# GLM_PM4=0 disables PM4.
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
  derivationArgs.wheelsCheck = wheelsCheck;
  text = ''
    rank=''${1:?usage: glm53-serve RANK [extra SGLang arguments]}
    shift
    export GLM_SGLANG_BIN=${sglang-glm53-rocm}/bin/sglang
    export GLM_PYTHON=${python3}/bin/python3
    export GLM_MODEL_PATH=''${GLM_MODEL_PATH:-/models/GLM-5.3-Flash-FP8}
    export GLM_CONTEXT_LENGTH=''${GLM_CONTEXT_LENGTH:-131072}
    export GLM_MAX_TOTAL_TOKENS=''${GLM_MAX_TOTAL_TOKENS:-131072}
    export GLM_RADIX_CACHE=''${GLM_RADIX_CACHE:-1}
    # Up to four requests decode together (graphs for batches 1-4); a lone request still replays the batch-1 graph.
    export GLM_MAX_REQUESTS=''${GLM_MAX_REQUESTS:-4}
    export GLM_CUDA_GRAPH=''${GLM_CUDA_GRAPH:-1}
    export SGLANG_ROCM_TRITON_RMSNORM=''${SGLANG_ROCM_TRITON_RMSNORM:-1}
    export SGLANG_GLM53_FUSED_QKVBFG=''${SGLANG_GLM53_FUSED_QKVBFG:-1}
    export SGLANG_DSV41_MSAR=''${SGLANG_DSV41_MSAR:-1}
    export SGLANG_DSV41_MSAR_PROXY_CPU=''${SGLANG_DSV41_MSAR_PROXY_CPU:-auto}
    export SGLANG_GLM53_WMMA_GEMV=''${SGLANG_GLM53_WMMA_GEMV:-1}
    export SGLANG_GLM53_MOE_GEMV_MAX_ROWS=''${SGLANG_GLM53_MOE_GEMV_MAX_ROWS:-16}
    export SGLANG_GLM53_KDA_NOSYNC=''${SGLANG_GLM53_KDA_NOSYNC:-1}
    export SGLANG_GLM53_ROUTER_GEMV=''${SGLANG_GLM53_ROUTER_GEMV:-1}
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
      --max-mamba-cache-size 128 --cuda-graph-max-bs-decode 4 "$@"
  '';
  meta.description = "GLM-5.3-Flash TP4 rank on Strix Halo with the qualified campaign configuration";
}
