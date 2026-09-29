#!/usr/bin/env bash
# One TP rank per Strix host. Put this flake's sglang-rocm on PATH or set
# GLM_SGLANG_BIN. This starts inference; staging alone never invokes it.
set -euo pipefail

rank=${1:?usage: glm53-node.sh RANK [additional SGLang arguments]}
shift
nodes=${GLM_NNODES:-4}
case "$nodes:$rank" in
  4:0|4:1|4:2|4:3|2:0|2:1) ;;
  *) echo "Use GLM_NNODES=2 or 4 and a zero-based RANK below that count" >&2; exit 2 ;;
esac

variant=${GLM_MODEL_VARIANT:-awq}
case "$variant" in
  awq) model_name=GLM-5.3-Flash-AWQ-W4A16 ;;
  fp8) model_name=GLM-5.3-Flash-FP8 ;;
  *) echo "GLM_MODEL_VARIANT must be awq or fp8" >&2; exit 2 ;;
esac
model=${GLM_MODEL_PATH:-/mnt/glm53-fabric/$model_name}
context_length=${GLM_CONTEXT_LENGTH:-32768}
"${GLM_PYTHON:-python3}" - "$model" "$variant" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
verified = json.loads((root / "_staging/verified.json").read_text())
revision, size, files = {
    "awq": ("abd7b07719111f137e1de8a0c1b7e01c11b74d1a", 190843146350, 24),
    "fp8": ("eb9eb208eb0d988989d07a6a12d0fdeb5f52574a", 328366173469, 73),
}[sys.argv[2]]
assert verified["revision"] == revision
assert verified["bytes"] == size
assert len(verified["files"]) == files
for entry in verified["files"]:
    assert (root / entry["path"]).stat().st_size == entry["size"], entry["path"]
PY

export HIP_VISIBLE_DEVICES=0
export SGLANG_USE_AITER=0
# The torch.topk(sorted=False) fallback changes expert order on identical
# scores. The fused router has stable ties/order and preserves the 2.5 scale.
export SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1
export SGLANG_ROCM_USE_MULTI_STREAM=0
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0
export NCCL_SOCKET_IFNAME=cx5fabric0
export GLOO_SOCKET_IFNAME=cx5fabric0
# RoCE must use the addressed fabric NIC and the switch's lossless DSCP 26
# queue (106 = DSCP 26 + ECT(0)). Require IB so a missing verbs library cannot
# silently turn a RoCE deployment into TCP. TCP remains an explicit control.
case ${GLM_TRANSPORT:-roce} in
  roce)
    export NCCL_IB_DISABLE=0 NCCL_NET=IB NCCL_IB_TC=106
    fabric_hcas=(/sys/class/net/cx5fabric0/device/infiniband/*)
    if [[ ${#fabric_hcas[@]} != 1 || ! -d ${fabric_hcas[0]} ]]; then
      echo "Expected one RDMA device for cx5fabric0" >&2
      exit 1
    fi
    export NCCL_IB_HCA="=${fabric_hcas[0]##*/}:1"
    ;;
  tcp) export NCCL_IB_DISABLE=1 NCCL_NET=Socket ;;
  *) echo "GLM_TRANSPORT must be roce or tcp" >&2; exit 2 ;;
esac
export AITER_JIT_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/glm53/$(hostname)/aiter/jit"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/glm53/$(hostname)/triton"
export OMP_NUM_THREADS=${GLM_CPU_THREADS:-8}
export SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=${GLM_LOAD_TIMEOUT:-1800}

options=(
  --model-path "$model" --served-model-name glm-5.3-flash
  --tp-size "$nodes" --nnodes "$nodes" --node-rank "$rank"
  --dist-init-addr "${GLM_DIST_ADDR:-192.168.25.101:50000}"
  --dist-timeout "${GLM_LOAD_TIMEOUT:-1800}"
  --weight-loader-disable-mmap
  --model-loader-extra-config '{"enable_multithread_load":false}'
  --dtype bfloat16 --kv-cache-dtype bfloat16
  --attention-backend dsa --dsa-prefill-backend triton --dsa-decode-backend triton
  --linear-attn-backend triton --moe-runner-backend triton
  --disable-shared-experts-fusion --disable-custom-all-reduce
  --context-length "$context_length"
  --chunked-prefill-size "${GLM_PREFILL_CHUNK:-1024}"
  --max-running-requests "${GLM_MAX_REQUESTS:-1}"
  --max-total-tokens "${GLM_MAX_TOTAL_TOKENS:-$context_length}"
  --mem-fraction-static "${GLM_MEMORY_FRACTION:-0.75}"
  --reasoning-parser glm45 --tool-call-parser glm47
  --default-chat-template-kwargs '{"clear_thinking":true,"reasoning_effort":"low"}'
  --enable-cache-report
  --host 127.0.0.1 --port "${GLM_PORT:-30000}"
)
if [[ ${GLM_RADIX_CACHE:-0} == 0 ]]; then options+=(--disable-radix-cache); fi
if [[ ${GLM_CUDA_GRAPH:-0} == 0 ]]; then options+=(--disable-cuda-graph); fi
if [[ $variant == fp8 ]]; then options+=(--fp8-gemm-backend triton); fi
exec "${GLM_SGLANG_BIN:-sglang}" serve "${options[@]}" "$@"
