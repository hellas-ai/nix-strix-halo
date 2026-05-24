{
  writeShellApplication,
  coreutils,
  vllm-env-lemonade,
  vllm-lemonade-gemma4-31b-q8-kernel-cache,
  packageSuffix,
}:

writeShellApplication {
  name = "vllm-lemonade-gemma4-31b-q8-${packageSuffix}";
  runtimeInputs = [ coreutils ];
  text = ''
    set -euo pipefail

    # vLLM/Inductor AOT artifacts embed absolute cache paths, including
    # inside binary blobs, so this prebuilt cache intentionally lives at
    # the same path used by the Nix build.
    cache_root="/tmp/vllm-lemonade-${packageSuffix}/gemma4-31b-q8"
    export VLLM_CACHE_ROOT="$cache_root"
    export TRITON_CACHE_DIR="''${TRITON_CACHE_DIR:-$VLLM_CACHE_ROOT/triton_cache}"
    export TORCHINDUCTOR_CACHE_DIR="''${TORCHINDUCTOR_CACHE_DIR:-$VLLM_CACHE_ROOT/inductor_cache}"
    export HF_HOME="''${HF_HOME:-/models/.cache/huggingface}"
    export VLLM_DISABLE_AITER="''${VLLM_DISABLE_AITER:-1}"
    export VLLM_CONFIG_ROOT="''${VLLM_CONFIG_ROOT:-/tmp/vllm-lemonade-${packageSuffix}/config}"
    export VLLM_XLA_CACHE_PATH="''${VLLM_XLA_CACHE_PATH:-/tmp/vllm-lemonade-${packageSuffix}/xla_cache}"
    unset VLLM_LEMONADE_CACHE_ROOT
    mkdir -p "$VLLM_CONFIG_ROOT" "$VLLM_XLA_CACHE_PATH"

    cache_store="${vllm-lemonade-gemma4-31b-q8-kernel-cache}"
    stamp="$VLLM_CACHE_ROOT/.prebuilt-gemma4-31b-q8-kernels"
    if [ ! -e "$stamp" ] || [ "$(cat "$stamp" 2>/dev/null || true)" != "$cache_store" ]; then
      tmp="$VLLM_CACHE_ROOT.tmp.$$"
      rm -rf "$tmp"
      mkdir -p "$tmp"
      cp -R "$cache_store/." "$tmp/"
      chmod -R u+w "$tmp"
      printf '%s\n' "$cache_store" > "$tmp/.prebuilt-gemma4-31b-q8-kernels"
      rm -rf "$VLLM_CACHE_ROOT"
      mv "$tmp" "$VLLM_CACHE_ROOT"
    fi

    exec "${vllm-env-lemonade}/bin/vllm" "$@"
  '';
}
