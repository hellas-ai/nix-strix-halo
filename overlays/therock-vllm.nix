# vLLM built against TheRock 7.13 wheels (rocm-sdk-core, rocm-sdk-devel)
# pinned to a single hardware target. Wires HSA env overrides + HIPCC
# link flags into the vllm CLI launcher and provides an aiter-jit
# kernel cache builder.
{
  lib,
  rocmTargets,
  target,
  hsaOverrideGfxVersion ? target.hsaOverride or "11.5.1",
}:
let
  s = target.packageSuffix;
in
final: prev:
let
  sdkBase = final."therock-rocm-${s}";
  sdk = sdkBase // {
    localGpuTargets = rocmTargets;
    gpuTargets = rocmTargets;
  };
  therockRocmPackages = {
    clr = sdk;
    hipcc = sdk;
    rocminfo = sdk;
    rocm-device-libs = sdk;
    llvm = sdk;
    rocthrust = sdk;
    rocprim = sdk;
    hipcub = sdk;
    hipblas = sdk;
    hipblas-common = sdk;
    hipblaslt = sdk;
    hipfft = sdk;
    hipsparse = sdk;
    hiprand = sdk;
    hipsolver = sdk;
    miopen-hip = sdk;
    miopen = sdk;
    rccl = sdk;
    rocblas = sdk;
    rocm-comgr = sdk;
    rocfft = sdk;
    rocrand = sdk;
    rocm-runtime = sdk;
    rocsolver = sdk;
    rocsparse = sdk;
    composable_kernel = sdk;
  };
  rocmRuntimePath = prev.lib.makeBinPath [
    sdk
    prev.gcc
    prev.ninja
    prev.openssl
  ];
  aiterRuntimePath = prev.lib.makeBinPath [
    sdk
    prev.gcc
    prev.ninja
    prev.openssl
  ];
  aiterComposableKernelSrc = "${prev.rocmPackages.composable_kernel.src}/projects/composablekernel";
  cxxIncludePath = prev.lib.concatStringsSep ":" [
    "${prev.stdenv.cc.cc}/include/c++/${prev.stdenv.cc.cc.version}"
    "${prev.stdenv.cc.cc}/include/c++/${prev.stdenv.cc.cc.version}/${prev.stdenv.hostPlatform.config}"
    "${prev.glibc.dev}/include"
  ];
  nativeLibraryPath = prev.lib.makeLibraryPath [
    sdk
    prev.stdenv.cc.cc.lib
    prev.glibc
  ];
  hipccLinkFlags = prev.lib.concatStringsSep " " [
    "-L${sdk}/lib"
    "-L${prev.stdenv.cc.cc.lib}/lib"
    "-L${prev.stdenv.cc.cc}/lib/gcc/${prev.stdenv.hostPlatform.config}/${prev.stdenv.cc.cc.version}"
    "-L${prev.glibc}/lib"
    "-B${prev.glibc}/lib"
  ];
  wrapWithHsa =
    name: env:
    prev.symlinkJoin {
      inherit name;
      paths = [ env ];
      buildInputs = [ prev.makeWrapper ];
      postBuild = ''
        for bin in "$out"/bin/*; do
          [[ -L "$bin" || -f "$bin" ]] || continue
          target=$(readlink -f "$bin")
          rm "$bin"
          makeWrapper "$target" "$bin" \
            --set HSA_NO_SCRATCH_RECLAIM 1 \
            --set HSA_ENABLE_INTERRUPT 0 \
            --set HSA_OVERRIDE_GFX_VERSION ${hsaOverrideGfxVersion} \
            --set ROCM_HOME ${sdk} \
            --set ROCM_PATH ${sdk} \
            --set HIP_PATH ${sdk} \
            --set HIP_PLATFORM amd \
            --set DEVICE_LIB_PATH ${sdk}/amdgcn/bitcode \
            --set HIP_DEVICE_LIB_PATH ${sdk}/amdgcn/bitcode \
            --prefix PATH : ${rocmRuntimePath} \
            --prefix CPATH : ${prev.glibc.dev}/include \
            --prefix CPLUS_INCLUDE_PATH : ${cxxIncludePath} \
            --prefix LIBRARY_PATH : ${nativeLibraryPath} \
            --run 'export HIPCC_LINK_FLAGS_APPEND="${hipccLinkFlags}''${HIPCC_LINK_FLAGS_APPEND:+ $HIPCC_LINK_FLAGS_APPEND}"' \
            --run 'if [ -z "''${AITER_JIT_DIR:-}" ]; then export AITER_JIT_DIR="''${XDG_CACHE_HOME:-''${HOME:-/tmp}/.cache}/aiter/raw-${s}"; fi' \
            --run 'export GPU_ARCHS="''${GPU_ARCHS:-${s}}"' \
            --run 'export PYTORCH_ROCM_ARCH="''${PYTORCH_ROCM_ARCH:-${s}}"' \
            --run 'export FLASH_ATTENTION_TRITON_AMD_ENABLE="''${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"' \
            --run 'export TORCH_BLAS_PREFER_HIPBLASLT="''${TORCH_BLAS_PREFER_HIPBLASLT:-1}"' \
            --run 'export PYTORCH_TUNABLEOP_ENABLED="''${PYTORCH_TUNABLEOP_ENABLED:-0}"' \
            --run 'export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="''${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"'
        done
      '';
      passthru = {
        inherit (env) python;
        rocm = sdk;
        unwrapped = env;
      };
    };
  mkAiterJitCache =
    {
      name,
      vllmEnv,
      modules ? [
        "module_aiter_enum"
        "module_rmsnorm"
      ],
    }:
    prev.stdenv.mkDerivation {
      pname = name;
      version = final.python312Packages.amd-aiter.version;
      dontUnpack = true;
      dontStrip = true;
      requiredSystemFeatures = [ "${s}" ];
      nativeBuildInputs = [
        prev.gcc
        prev.ninja
        sdk
      ];

      buildPhase = ''
        runHook preBuild

        export HOME="$TMPDIR/home"
        export AITER_JIT_DIR="$TMPDIR/aiter-jit"
        export GPU_ARCHS="${s}"
        export PYTORCH_ROCM_ARCH="${s}"
        export ROCM_HOME="${sdk}"
        export ROCM_PATH="$ROCM_HOME"
        export HIP_PATH="$ROCM_HOME"
        export HIP_PLATFORM="amd"
        export DEVICE_LIB_PATH="${sdk}/amdgcn/bitcode"
        export HIP_DEVICE_LIB_PATH="$DEVICE_LIB_PATH"
        export CPATH="${prev.glibc.dev}/include''${CPATH:+:$CPATH}"
        export CPLUS_INCLUDE_PATH="${cxxIncludePath}''${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
        export LIBRARY_PATH="${nativeLibraryPath}''${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export HIPCC_LINK_FLAGS_APPEND="${hipccLinkFlags}''${HIPCC_LINK_FLAGS_APPEND:+ $HIPCC_LINK_FLAGS_APPEND}"
        export MAX_JOBS="''${NIX_BUILD_CORES:-8}"
        export PATH="${aiterRuntimePath}:$PATH"

        mkdir -p "$HOME" "$AITER_JIT_DIR"
        ${vllmEnv}/bin/python - <<'PY'
        from aiter.jit.core import build_module, get_args_of_build

        modules = ${builtins.toJSON modules}
        for md_name in modules:
            args = get_args_of_build(md_name)
            build_module(
                md_name=md_name,
                srcs=args["srcs"],
                flags_extra_cc=args["flags_extra_cc"],
                flags_extra_hip=args["flags_extra_hip"],
                blob_gen_cmd=args["blob_gen_cmd"],
                extra_include=args["extra_include"],
                extra_ldflags=args["extra_ldflags"],
                verbose=args["verbose"],
                is_python_module=args["is_python_module"],
                is_standalone=args["is_standalone"],
                torch_exclude=args["torch_exclude"],
                hipify=args.get("hipify", False),
            )
        PY

        ${prev.lib.concatMapStringsSep "\n" (module: ''
          test -f "$AITER_JIT_DIR/${module}.so"
        '') modules}

        runHook postBuild
      '';

      installPhase = ''
        runHook preInstall

        mkdir -p "$out/build"
        cp "$AITER_JIT_DIR"/module_*.so "$out"/

        runHook postInstall
      '';

      meta = with prev.lib; {
        description = "Prebuilt AITER runtime-JIT modules for vLLM/TheRock on ${s}";
        platforms = platforms.linux;
      };
    };
  mkVllmAiterWrapper =
    {
      name,
      vllmEnv,
      aiterJitCache,
    }:
    prev.writeShellApplication {
      inherit name;
      runtimeInputs = [
        prev.coreutils
        prev.gcc
        prev.ninja
        sdk
      ];
      text = ''
        export GPU_ARCHS="''${GPU_ARCHS:-${s}}"
        export PYTORCH_ROCM_ARCH="''${PYTORCH_ROCM_ARCH:-${s}}"
        export ROCM_HOME="''${ROCM_HOME:-${sdk}}"
        export ROCM_PATH="''${ROCM_PATH:-$ROCM_HOME}"
        export HIP_PATH="''${HIP_PATH:-$ROCM_HOME}"
        export HIP_PLATFORM="''${HIP_PLATFORM:-amd}"
        export DEVICE_LIB_PATH="''${DEVICE_LIB_PATH:-${sdk}/amdgcn/bitcode}"
        export HIP_DEVICE_LIB_PATH="''${HIP_DEVICE_LIB_PATH:-$DEVICE_LIB_PATH}"
        export CPATH="${prev.glibc.dev}/include''${CPATH:+:$CPATH}"
        export CPLUS_INCLUDE_PATH="${cxxIncludePath}''${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
        export LIBRARY_PATH="${nativeLibraryPath}''${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export HIPCC_LINK_FLAGS_APPEND="${hipccLinkFlags}''${HIPCC_LINK_FLAGS_APPEND:+ $HIPCC_LINK_FLAGS_APPEND}"
        export HSA_NO_SCRATCH_RECLAIM="''${HSA_NO_SCRATCH_RECLAIM:-1}"
        export HSA_ENABLE_INTERRUPT="''${HSA_ENABLE_INTERRUPT:-0}"
        export HSA_OVERRIDE_GFX_VERSION="''${HSA_OVERRIDE_GFX_VERSION:-${hsaOverrideGfxVersion}}"
        export FLASH_ATTENTION_TRITON_AMD_ENABLE="''${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"
        export TORCH_BLAS_PREFER_HIPBLASLT="''${TORCH_BLAS_PREFER_HIPBLASLT:-1}"
        export PYTORCH_TUNABLEOP_ENABLED="''${PYTORCH_TUNABLEOP_ENABLED:-0}"
        export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="''${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
        export VLLM_ROCM_USE_AITER="''${VLLM_ROCM_USE_AITER:-1}"
        export VLLM_ROCM_USE_AITER_LINEAR="''${VLLM_ROCM_USE_AITER_LINEAR:-1}"
        export VLLM_ROCM_USE_AITER_MOE="''${VLLM_ROCM_USE_AITER_MOE:-0}"
        export VLLM_ROCM_USE_AITER_RMSNORM="''${VLLM_ROCM_USE_AITER_RMSNORM:-0}"
        export VLLM_ROCM_USE_AITER_MHA="''${VLLM_ROCM_USE_AITER_MHA:-1}"
        export VLLM_ROCM_USE_AITER_TRITON_GEMM="''${VLLM_ROCM_USE_AITER_TRITON_GEMM:-1}"
        export VLLM_ROCM_USE_AITER_TRITON_ROPE="''${VLLM_ROCM_USE_AITER_TRITON_ROPE:-1}"
        export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION="''${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:-1}"
        export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="''${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-0}"

        if [ -z "''${AITER_JIT_DIR:-}" ]; then
          cache_root="''${XDG_CACHE_HOME:-''${HOME:-/tmp}/.cache}/aiter"
          cache_dir="$cache_root/${aiterJitCache.name}"
          stamp="$cache_dir/.prebuilt-${aiterJitCache.name}"
          if [ ! -e "$stamp" ]; then
            rm -rf "$cache_dir.tmp"
            mkdir -p "$cache_dir.tmp"
            cp -R "${aiterJitCache}/." "$cache_dir.tmp/"
            chmod -R u+w "$cache_dir.tmp"
            touch "$cache_dir.tmp/.prebuilt-${aiterJitCache.name}"
            rm -rf "$cache_dir"
            mv "$cache_dir.tmp" "$cache_dir"
          fi
          export AITER_JIT_DIR="$cache_dir"
        fi

        exec "${vllmEnv}/bin/vllm" "$@"
      '';
    };
  dropVllmDependencyNames = [
    "bitsandbytes"
    "datasets"
    "diskcache"
    "lark"
    "mistral-common"
    "mistral_common"
    "mistralai"
    "opencv-python-headless"
    "outlines"
    "outlines-core"
    "peft"
    "pyarrow"
    "timm"
    "torchcodec"
    "torchvision"
    "xformers"
  ];
  dropVllmDeps =
    names: deps:
    prev.lib.filter (
      dep:
      let
        name = dep.pname or dep.name or "";
      in
      !(prev.lib.elem name names)
    ) deps;
  disablePythonChecks =
    pkg:
    pkg.overridePythonAttrs (_old: {
      doCheck = false;
      pythonImportsCheck = [ ];
    });
  vllmTherock =
    (final.python312Packages.vllm.override {
      rocmSupport = true;
      cudaSupport = false;
      gpuTargets = rocmTargets;
      rocmPackages = therockRocmPackages;
      amdsmi = final.python312Packages.amdsmi;
    }).overridePythonAttrs
      (old: {
        patches = (old.patches or [ ]) ++ [
          ../pkgs/vllm/patches/0001-hipify-copy-unchanged-cu-to-hip.patch
        ];
        postPatch = (old.postPatch or "") + ''
                        substituteInPlace csrc/quantization/gptq/compat.cuh \
                          --replace-fail 'namespace gptq {' 'namespace gptq {

          #if defined(USE_ROCM) && defined(TORCH_HIP_VERSION) && TORCH_HIP_VERSION >= 713
          #define VLLM_GPTQ_SKIP_LEGACY_HALF_ATOMIC_ADD 1
          #endif'
                        substituteInPlace csrc/quantization/gptq/compat.cuh \
                          --replace-fail '#if defined(__CUDA_ARCH__) || defined(USE_ROCM)' \
                            '#if !defined(VLLM_GPTQ_SKIP_LEGACY_HALF_ATOMIC_ADD) && (defined(__CUDA_ARCH__) || defined(USE_ROCM))'
        '';
        env = (old.env or { }) // {
          HIP_PATH = "${sdk}";
          HIP_PLATFORM = "amd";
          CMAKE_ARGS = prev.lib.concatStringsSep " " [
            ((old.env or { }).CMAKE_ARGS or "")
            "-DCMAKE_HIP_COMPILER=${sdk}/bin/therock-hip-clang++"
            "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${sdk}"
            "-DHIP_ROOT_DIR=${sdk}"
          ];
        };
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
          final.pkg-config
        ];
        pythonRemoveDeps = (old.pythonRemoveDeps or [ ]) ++ dropVllmDependencyNames;
        dependencies = dropVllmDeps dropVllmDependencyNames (old.dependencies or [ ]) ++ [
          final.python312Packages.amdsmi
          final.python312Packages.cloudpickle
        ];
        propagatedBuildInputs = dropVllmDeps dropVllmDependencyNames (old.propagatedBuildInputs or [ ]) ++ [
          final.python312Packages.amdsmi
          final.python312Packages.cloudpickle
        ];
      });
  vllmEnvTherock = wrapWithHsa "vllm-env-therock-${s}" (
    final.python312.withPackages (_ps: [
      vllmTherock
      final.python312Packages.diskcache
      final.python312Packages."mistral-common"
      final.python312Packages.pycountry
      final.python312Packages.ray
    ])
  );
  vllmEnvTherockAiter = wrapWithHsa "vllm-env-therock-aiter-${s}" (
    final.python312.withPackages (_ps: [
      vllmTherock
      final.python312Packages.amd-aiter
      final.python312Packages.diskcache
      final.python312Packages."mistral-common"
      final.python312Packages.pycountry
      final.python312Packages.ray
    ])
  );
  vllmAiterJitTherock = mkAiterJitCache {
    name = "vllm-aiter-jit-therock-${s}";
    vllmEnv = vllmEnvTherockAiter;
  };
in
{
  pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
    (
      pyfinal: pyprev:
      if lib.versions.majorMinor pyprev.python.version == "3.12" then
        {
          amd-aiter =
            (pyprev.amd-aiter.override {
              rocmPackages = therockRocmPackages;
            }).overridePythonAttrs
              (old: {
                env = (old.env or { }) // {
                  ROCM_PATH = "${sdk}";
                  ROCM_HOME = "${sdk}";
                  HIP_PATH = "${sdk}";
                };
                postPatch = (old.postPatch or "") + ''
                  rm -rf 3rdparty/composable_kernel
                  ln -s ${aiterComposableKernelSrc} 3rdparty/composable_kernel
                '';
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  sdk
                ];
                buildInputs = (old.buildInputs or [ ]) ++ [
                  sdk
                ];
              });
          "compressed-tensors" = pyprev."compressed-tensors".overridePythonAttrs (old: {
            dependencies = (old.dependencies or [ ]) ++ [
              pyfinal.psutil
            ];
            nativeCheckInputs = (old.nativeCheckInputs or [ ]) ++ [
              final.openssl
            ];
            propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [
              pyfinal.psutil
            ];
          });
          depyf = disablePythonChecks pyprev.depyf;
          llguidance = disablePythonChecks pyprev.llguidance;
          pydevd = disablePythonChecks pyprev.pydevd;
          "mistral-common" = disablePythonChecks pyprev."mistral-common";
        }
      else
        { }
    )
  ];
  "therock-rocm-packages-${s}" = therockRocmPackages;
  "therock-python-overlay-smoke-${s}" = final.python312.withPackages (ps: [
    ps.torch
    ps.triton
  ]);
  "vllm-rocm-therock-${s}" = vllmTherock;
  "vllm-env-therock-${s}" = vllmEnvTherock;
  "vllm-env-therock-aiter-${s}" = vllmEnvTherockAiter;
  "vllm-aiter-jit-therock-${s}" = vllmAiterJitTherock;
  "vllm-aiter-therock-${s}" = mkVllmAiterWrapper {
    name = "vllm-aiter-therock-${s}";
    vllmEnv = vllmEnvTherockAiter;
    aiterJitCache = vllmAiterJitTherock;
  };
}
