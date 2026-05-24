{
  writeShellApplication,
  coreutils,
  vllm-env-lemonade,
  vllm-lemonade-qwen36-27b-cache,
  packageSuffix,
}:

writeShellApplication {
  name = "vllm-lemonade-qwen36-27b-${packageSuffix}";
  runtimeInputs = [ coreutils ];
  text = ''
    set -euo pipefail

    cache_root="''${VLLM_LEMONADE_CACHE_ROOT:-''${XDG_CACHE_HOME:-''${HOME:-/tmp}/.cache}/vllm/lemonade-${packageSuffix}}"
    export VLLM_CACHE_ROOT="''${VLLM_CACHE_ROOT:-$cache_root}"
    export TRITON_CACHE_DIR="''${TRITON_CACHE_DIR:-$VLLM_CACHE_ROOT/triton_cache}"
    export TORCHINDUCTOR_CACHE_DIR="''${TORCHINDUCTOR_CACHE_DIR:-$VLLM_CACHE_ROOT/inductor_cache}"

    cache_store="${vllm-lemonade-qwen36-27b-cache}"
    stamp="$VLLM_CACHE_ROOT/.prebuilt-qwen36-27b"
    if [ ! -e "$stamp" ] || [ "$(cat "$stamp" 2>/dev/null || true)" != "$cache_store" ]; then
      tmp="$VLLM_CACHE_ROOT.tmp.$$"
      rm -rf "$tmp"
      mkdir -p "$tmp"
      cp -R "$cache_store/." "$tmp/"
      chmod -R u+w "$tmp"
      printf '%s\n' "$cache_store" > "$tmp/.prebuilt-qwen36-27b"
      rm -rf "$VLLM_CACHE_ROOT"
      mv "$tmp" "$VLLM_CACHE_ROOT"
    fi

    exec "${vllm-env-lemonade}/bin/vllm" "$@"
  '';
}
