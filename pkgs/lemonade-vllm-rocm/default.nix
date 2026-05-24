{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  makeWrapper,
  gnutar,
  gzip,
  coreutils,
  bash,
  bzip2,
  elfutils,
  expat,
  glibc,
  libdrm,
  libffi,
  libuuid,
  libxcrypt,
  libxcrypt-legacy,
  ncurses,
  numactl,
  ocl-icd,
  openssl,
  rdma-core,
  sqlite,
  xz,
  tbb,
  zlib,
  zstd,
  target ? "gfx1151",
  releaseTag ? "vllm0.21.0-rocm7.13.0-gfx1151",
}:

let
  baseUrl = "https://github.com/lemonade-sdk/vllm-rocm/releases/download/${releaseTag}";
  archiveBase = "${releaseTag}-x64";
  pythonShebang = "/opt/vllm/bin/python3";
in
stdenv.mkDerivation {
  pname = "vllm-rocm-lemonade-${target}";
  version = releaseTag;

  partcount = fetchurl {
    url = "${baseUrl}/${archiveBase}.partcount";
    hash = "sha256-U8I05ehHK2rFHBrhyrP+BvrQU7646/2Jd7AQZVv908M=";
  };

  part01 = fetchurl {
    url = "${baseUrl}/${archiveBase}.part01-of-02.tar.gz";
    hash = "sha256-0/Ng991GDRQ1Bx71UKwzalIJrj6EULXk9mgisOgP6lw=";
  };

  part02 = fetchurl {
    url = "${baseUrl}/${archiveBase}.part02-of-02.tar.gz";
    hash = "sha256-yYuewM801jwGdxFbvSMDiZoAnpumldUgC5V/lbkxZO0=";
  };

  nativeBuildInputs = [
    autoPatchelfHook
    coreutils
    gnutar
    gzip
    makeWrapper
  ];

  buildInputs = [
    bash
    bzip2
    elfutils
    expat
    glibc
    libdrm
    libffi
    libuuid
    libxcrypt
    libxcrypt-legacy
    ncurses
    numactl
    ocl-icd
    openssl
    rdma-core
    sqlite
    stdenv.cc.cc.lib
    tbb
    xz
    zlib
    zstd
  ];

  autoPatchelfIgnoreMissingDeps = [
    "libc10_cuda.so"
    "libjpeg.so.8"
    "libtorch_cuda.so"
  ];

  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  dontStrip = true;

  appendRunpaths = [
    "$ORIGIN/../lib"
    "$ORIGIN/../lib64"
    "$ORIGIN/../llvm/lib"
    "$ORIGIN/../lib/llvm/lib"
  ];

  installPhase = ''
        runHook preInstall

        test "$(cat "$partcount")" = "2"

        mkdir -p "$out"
        cat "$part01" "$part02" | tar -xzf - -C "$out"
        chmod -R u+w "$out"
        find "$out/bin" -type f -exec chmod u+x {} +

        while IFS= read -r script; do
          substituteInPlace "$script" \
            --replace-fail "${pythonShebang}" "$out/bin/python3"
        done < <(grep -rl '^#!${pythonShebang}' "$out/bin" || true)

        mkdir -p "$out/lib"
        "$CXX" -shared -fPIC -O2 "${./c10-hip-compat.cc}" \
          -o "$out/lib/libvllm-rocm-c10-hip-compat.so" \
          -L"$out/lib/python3.12/site-packages/torch/lib" \
          -lc10_hip -ltorch_hip -lc10 \
          -Wl,-rpath,"$out/lib/python3.12/site-packages/torch/lib"

        mkdir -p "$out/lib/python3.12/site-packages/amdsmi"
        cp "${./amdsmi-compat/__init__.py}" \
          "$out/lib/python3.12/site-packages/amdsmi/__init__.py"

        aiter_enum_h="$out/lib/python3.12/site-packages/csrc/include/aiter_enum.h"
        mkdir -p "$(dirname "$aiter_enum_h")"
        cat > "$aiter_enum_h" <<'EOF'
    typedef enum {
      AITER_DTYPE_fp8 = 0,
      AITER_DTYPE_fp8_e8m0 = 1,
      AITER_DTYPE_fp16 = 2,
      AITER_DTYPE_bf16 = 3,
      AITER_DTYPE_fp32 = 4,
      AITER_DTYPE_i4x2 = 5,
      AITER_DTYPE_fp4x2 = 6,
      AITER_DTYPE_u32 = 7,
      AITER_DTYPE_i32 = 8,
      AITER_DTYPE_i16 = 9,
      AITER_DTYPE_i8 = 10,
    } AiterDtype;
    EOF

        gguf_loader="$out/lib/python3.12/site-packages/vllm/model_executor/model_loader/gguf_loader.py"
        substituteInPlace "$gguf_loader" \
          --replace-fail '        if model_type == "gemma3_text":' '        if model_type in ("gemma3_text", "gemma4_text"):' \
          --replace-fail '            model_type = "gemma3"' '            model_type = model_type.removesuffix("_text")' \
          --replace-fail '        is_multimodal = hasattr(model_config.hf_config, "vision_config")' '        is_multimodal = (hasattr(model_config.hf_config, "vision_config") and model_config.hf_config.vision_config is not None)'

        config_py="$out/lib/python3.12/site-packages/vllm/transformers_utils/config.py"
        substituteInPlace "$config_py" \
          --replace-fail '    if check_gguf_file(model):
            kwargs["gguf_file"] = Path(model).name
            gguf_model_repo = Path(model).parent
        elif is_remote_gguf(model):' '    if check_gguf_file(model):
            return model, tokenizer, vllm_speculative_config
        elif is_remote_gguf(model):'

        import_utils_py="$out/lib/python3.12/site-packages/vllm/utils/import_utils.py"
        substituteInPlace "$import_utils_py" \
          --replace-fail 'def has_aiter() -> bool:
        """Whether the optional `aiter` package is available."""
        return _has_module("aiter")' 'def has_aiter() -> bool:
        """Whether the optional `aiter` package is available."""
        if os.environ.get("VLLM_DISABLE_AITER") == "1":
            return False
        return _has_module("aiter")'

        envs_py="$out/lib/python3.12/site-packages/vllm/envs.py"
        substituteInPlace "$envs_py" \
          --replace-fail '    "VLLM_CACHE_ROOT": lambda: os.path.expanduser(
            os.getenv(
                "VLLM_CACHE_ROOT",
                os.path.join(get_default_cache_root(), "vllm"),
            )
        ),' '    "VLLM_CACHE_ROOT": lambda: os.path.expanduser(
            os.getenv(
                "VLLM_CACHE_ROOT",
                os.path.join(get_default_cache_root(), "vllm"),
            )
        ),
        "VLLM_DISABLE_AITER": lambda: bool(
            int(os.getenv("VLLM_DISABLE_AITER", "0"))
        ),'

        quark_ocp_mx_py="$out/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py"
        substituteInPlace "$quark_ocp_mx_py" \
          --replace-fail 'from collections.abc import Callable' 'import os
    from collections.abc import Callable' \
          --replace-fail 'try:
        from aiter.ops.shuffle import shuffle_weight' 'try:
        if os.environ.get("VLLM_DISABLE_AITER") == "1":
            raise ImportError("aiter disabled by VLLM_DISABLE_AITER")
        from aiter.ops.shuffle import shuffle_weight'

        rocm_core="$out/lib/python3.12/site-packages/_rocm_sdk_core"
        mkdir -p "$rocm_core/bin" "$rocm_core/.info"

        printf '7.13.0\n' > "$rocm_core/.info/version"
        printf '7.13.99004\n' > "$rocm_core/.info/version-dev"

        if [ ! -e "$rocm_core/hip" ]; then
          ln -s . "$rocm_core/hip"
        fi

        cat > "$rocm_core/bin/hipconfig" <<EOF
    #!${bash}/bin/bash
    case "\''${1:-}" in
      --version)
        echo "7.13.99004"
        ;;
      --rocmpath|--path)
        echo "$rocm_core"
        ;;
      --hipclangpath)
        echo "$rocm_core/lib/llvm/bin"
        ;;
      --full)
        echo "HIP version  : 7.13.99004"
        echo "ROCm path    : $rocm_core"
        echo "HIP path     : $rocm_core"
        echo "HIP clang    : $rocm_core/lib/llvm/bin/clang++"
        ;;
      *)
        echo "7.13.99004"
        ;;
    esac
    EOF

        cat > "$rocm_core/bin/hipcc" <<EOF
    #!${bash}/bin/bash
    if [ "\''${1:-}" = "--version" ]; then
      echo "HIP version: 7.13.99004"
      exec "$rocm_core/lib/llvm/bin/clang++" --version
    fi
    exec "$rocm_core/lib/llvm/bin/clang++" --rocm-path="$rocm_core" --hip-path="$rocm_core" "\$@"
    EOF

        cat > "$rocm_core/bin/rocminfo" <<'EOF'
    #!/usr/bin/env bash
    for candidate in /run/current-system/sw/bin/rocminfo /usr/bin/rocminfo; do
      if [ -x "$candidate" ]; then
        exec "$candidate" "$@"
      fi
    done
    cat <<'ROCINFO'
    ROCk module is loaded

    ==========
    HSA Agents
    ==========
    *******
    Agent 1
    *******
      Name:                    gfx1151
      Marketing Name:          Radeon 8060S Graphics
      Vendor Name:             AMD
      Device Type:             GPU
      Compute Unit:            40
    ROCINFO
    EOF

        cat > "$rocm_core/bin/rocm_agent_enumerator" <<'EOF'
    #!/usr/bin/env bash
    if [ -x /run/current-system/sw/bin/rocm_agent_enumerator ]; then
      exec /run/current-system/sw/bin/rocm_agent_enumerator "$@"
    fi
    echo gfx1151
    EOF

        cat > "$rocm_core/bin/offload-arch" <<'EOF'
    #!/usr/bin/env bash
    echo gfx1151
    EOF

        chmod +x \
          "$rocm_core/bin/hipconfig" \
          "$rocm_core/bin/hipcc" \
          "$rocm_core/bin/rocminfo" \
          "$rocm_core/bin/rocm_agent_enumerator" \
          "$rocm_core/bin/offload-arch"

        for tool in hipconfig hipcc rocminfo rocm_agent_enumerator offload-arch; do
          rm -f "$out/bin/$tool"
          ln -s "$rocm_core/bin/$tool" "$out/bin/$tool"
        done

        mv "$out/bin/vllm" "$out/bin/vllm-unwrapped"
        makeWrapper "$out/bin/vllm-unwrapped" "$out/bin/vllm" \
          --set HSA_OVERRIDE_GFX_VERSION 11.5.1 \
          --set HSA_NO_SCRATCH_RECLAIM 1 \
          --set HSA_ENABLE_INTERRUPT 0 \
          --set HIP_PLATFORM amd \
          --set GPU_ARCHS gfx1151 \
          --set PYTORCH_ROCM_ARCH gfx1151 \
          --set FLASH_ATTENTION_TRITON_AMD_ENABLE TRUE \
          --set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL 1 \
          --set ROCM_HOME "$rocm_core" \
          --set ROCM_PATH "$rocm_core" \
          --set HIP_PATH "$rocm_core" \
          --set DEVICE_LIB_PATH "$rocm_core/lib/llvm/amdgcn/bitcode" \
          --set HIP_DEVICE_LIB_PATH "$rocm_core/lib/llvm/amdgcn/bitcode" \
          --set CC "${stdenv.cc}/bin/cc" \
          --set CXX "${stdenv.cc}/bin/c++" \
          --prefix PATH : "$out/bin" \
          --prefix PATH : "${lib.makeBinPath [ stdenv.cc ]}" \
          --run 'if [ -z "''${VLLM_CACHE_ROOT:-}" ]; then cache_root="''${XDG_CACHE_HOME:-''${HOME:-/tmp}/.cache}/vllm/lemonade-gfx1151"; mkdir -p "$cache_root" 2>/dev/null || cache_root="/tmp/vllm-''${USER:-$(id -u)}/lemonade-gfx1151"; export VLLM_CACHE_ROOT="$cache_root"; fi; mkdir -p "$VLLM_CACHE_ROOT" 2>/dev/null || true; if [ -z "''${TRITON_CACHE_DIR:-}" ]; then export TRITON_CACHE_DIR="$VLLM_CACHE_ROOT/triton_cache"; fi; mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true; if [ -z "''${TORCHINDUCTOR_CACHE_DIR:-}" ]; then export TORCHINDUCTOR_CACHE_DIR="$VLLM_CACHE_ROOT/inductor_cache"; fi; mkdir -p "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true' \
          --run 'if [ -z "''${AITER_JIT_DIR:-}" ]; then cache_root="''${XDG_CACHE_HOME:-''${HOME:-/tmp}/.cache}/aiter"; mkdir -p "$cache_root" 2>/dev/null || cache_root="/tmp/aiter-''${USER:-$(id -u)}"; cache_dir="$cache_root/lemonade-gfx1151"; stamp="$cache_dir/.prebuilt-vllm0.21.0-rocm7.13.0-gfx1151"; if [ ! -e "$stamp" ]; then rm -rf "$cache_dir.tmp"; mkdir -p "$cache_dir.tmp"; cp '"$out"'/lib/python3.12/site-packages/aiter/jit/module_*.so "$cache_dir.tmp/"; cp '"$out"'/lib/python3.12/site-packages/aiter/jit/optCompilerConfig.json "$cache_dir.tmp/" 2>/dev/null || true; mkdir -p "$cache_dir.tmp/build"; chmod -R u+w "$cache_dir.tmp"; touch "$cache_dir.tmp/.prebuilt-vllm0.21.0-rocm7.13.0-gfx1151"; rm -rf "$cache_dir"; mv "$cache_dir.tmp" "$cache_dir"; fi; export AITER_JIT_DIR="$cache_dir"; fi' \
          --run 'export LD_PRELOAD="'"$out"'/lib/libvllm-rocm-c10-hip-compat.so''${LD_PRELOAD:+ $LD_PRELOAD}"'

        runHook postInstall
  '';

  meta = with lib; {
    description = "Lemonade vLLM ROCm binary bundle for Strix Halo (${target})";
    homepage = "https://github.com/lemonade-sdk/vllm-rocm";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
