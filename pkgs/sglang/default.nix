{
  lib,
  stdenv,
  fetchurl,
  fetchFromGitHub,
  autoPatchelfHook,
  cargo,
  gnumake,
  makeWrapper,
  openssl,
  patchelf,
  pkg-config,
  rdma-core,
  rustPlatform,
  rustc,
  symlinkJoin,
  writableTmpDirAsHomeHook,
  pythonPackages,
  rocmSdk,
  packageSuffix ? "rocm",
  hsaOverrideGfxVersion ? null,
  mediaRuntime ? false,
  buildFromSource ? false,
  packageVariant ? null,
  packageVersion ? if mediaRuntime then "0.5.20" else "0.5.17.dev0",
  packageCacheTag ? "sglang-${packageVersion}",
  packageSrc ? null,
  packagePatches ? if mediaRuntime then [ ] else null,
  packageExtraDependencies ? [ ],
  packageAiter ? pythonPackages.amd-aiter,
  packagePostPatch ? "",
  packageDoCheck ? true,
  packageDoInstallCheck ? true,
  runtimePatches ? map (name: ./patches + "/${name}") (
    builtins.filter (lib.hasSuffix ".patch") (builtins.attrNames (builtins.readDir ./patches))
  ),
}:

let
  sglangDependencies = import ./dependencies.nix {
    inherit
      pythonPackages
      fetchurl
      autoPatchelfHook
      stdenv
      ;
  };
  pythonSitePackages = pythonPackages.python.sitePackages;
  pythonTag = builtins.replaceStrings [ "." ] [ "" ] pythonPackages.python.pythonVersion;
  rocmSitePackages = pythonPackages.torch.passthru.sitePackages or null;
  rocmRuntimeLibraryPath =
    (pythonPackages.torch.passthru.rocmRuntimeEnv or { }).LD_LIBRARY_PATH or "";
  gpuArch = if lib.hasPrefix "gfx" packageSuffix then packageSuffix else null;
  hasNativeKernels =
    if mediaRuntime then
      (
        !fromSource
        && lib.elem gpuArch [
          "gfx1151"
          "gfx942"
          "gfx950"
          "gfx1250"
        ]
      )
    else
      (lib.elem gpuArch [
        "gfx1151"
        "gfx942"
        "gfx950"
        "gfx1250"
      ]);
  rocmSdkForJit = symlinkJoin {
    name = "${rocmSdk.name or "rocm-sdk"}-sglang-jit";
    paths = [ rocmSdk ];
    postBuild = ''
      rm -f "$out/bin/hipcc"
      printf '%s\n' \
        '#!${stdenv.shell}' \
        'needs_xhip=0' \
        'for arg in "$@"; do' \
        '  case "$arg" in' \
        '    *.cu|*.hip|*.cpp|*.cc|*.cxx) needs_xhip=1 ;;' \
        '  esac' \
        'done' \
        'if [ "$needs_xhip" = 1 ]; then' \
        '  exec ${rocmSdk}/bin/therock-hip-clang++ -x hip "$@"' \
        'fi' \
        'exec ${rocmSdk}/bin/therock-hip-clang++ "$@"' \
        > "$out/bin/hipcc"
      chmod 755 "$out/bin/hipcc"
    '';
  };
  sglangKernel = import ./kernel.nix {
    inherit
      lib
      stdenv
      fetchurl
      autoPatchelfHook
      pythonPackages
      packageSuffix
      gpuArch
      ;
    rocmSdk = rocmSdkForJit;
  };
  outlinesCoreCargoLock = ./outlines-core-0_1_26-Cargo.lock;
  outlines-core_0_1_26 = pythonPackages.buildPythonPackage rec {
    pname = "outlines-core";
    version = "0.1.26";
    pyproject = true;

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/d3/f3/274d07f4702728b43581235a77e545ec602b25f9b0098b288a0f3052521d/outlines_core-${version}.tar.gz";
      hash = "sha256-SBxDATQed8yPGDLWFnhK201GG0/sZYeOfA0sunFjoYk=";
    };

    cargoDeps = rustPlatform.importCargoLock {
      lockFile = outlinesCoreCargoLock;
    };

    postPatch = ''
      cp --no-preserve=mode ${outlinesCoreCargoLock} Cargo.lock
    '';

    nativeBuildInputs = [
      cargo
      pkg-config
      rustPlatform.cargoSetupHook
      rustc
    ];

    buildInputs = [ openssl.dev ];

    # onig_sys 69.8.1 vendors C code that predates C23's strict function
    # prototypes. GCC 15 defaults to gnu23, so keep this one C dependency on
    # the language version it supports.
    env.CFLAGS = "-std=gnu17";

    build-system = with pythonPackages; [
      setuptools-rust
      setuptools-scm
    ];

    dependencies = with pythonPackages; [
      interegular
      jsonschema
    ];

    doCheck = false;
    pythonImportsCheck = [ "outlines_core" ];
  };
  outlines_0_1_11 = pythonPackages.buildPythonPackage rec {
    pname = "outlines";
    version = "0.1.11";
    format = "wheel";

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/13/b4/99ea4a122bef60e3fd6402d19665aff1f928e0daf8fac3044d0b73f72003/outlines-${version}-py3-none-any.whl";
      hash = "sha256-9aXyJC7ZgC06q3qSeJv0AI1zTFdr6SWMwKKX9pASRyc=";
    };

    dependencies = with pythonPackages; [
      airportsdata
      cloudpickle
      diskcache
      interegular
      jinja2
      jsonschema
      lark
      nest-asyncio
      numpy
      outlines-core_0_1_26
      pycountry
      pydantic
      referencing
      requests
      torch
      tqdm
      typing-extensions
    ];

    pythonImportsCheck = [ "outlines" ];
  };
  torchaoNoChecks =
    if mediaRuntime then
      (pythonPackages.torchao.overridePythonAttrs (old: {
        doCheck = false;
        # ROCm-enabled torchao 0.17 builds its swizzle extension against the
        # hipBLASLt C++ API. The Python wheel closure carries the runtime library,
        # but ROCm 10 no longer carries this header in its core wheel.
        buildInputs = (old.buildInputs or [ ]) ++ [ rocmSdk ];
        nativeCheckInputs = [ ];
        pythonImportsCheck = old.pythonImportsCheck or [ "torchao" ];
      }))
    else
      (
        if buildFromSource then
          (pythonPackages.torchao.overridePythonAttrs (old: {
            doCheck = false;
            nativeCheckInputs = [ ];
            pythonImportsCheck = old.pythonImportsCheck or [ "torchao" ];
          }))
        else
          (pythonPackages.torchao.overridePythonAttrs (old: {
            doCheck = false;
            # ROCm-enabled torchao 0.17 builds its swizzle extension against the
            # hipBLASLt C++ API. The Python wheel closure carries the runtime library,
            # but ROCm 10 no longer carries this header in its core wheel.
            buildInputs = (old.buildInputs or [ ]) ++ [ rocmSdk ];
            nativeCheckInputs = [ ];
            pythonImportsCheck = old.pythonImportsCheck or [ "torchao" ];
          }))
      );
  xgrammar_0_2_1 =
    if mediaRuntime then
      (import ../xgrammar-0_2.nix {
        inherit
          lib
          pythonPackages
          fetchurl
          autoPatchelfHook
          patchelf
          stdenv
          ;
      })
    else
      (
        if buildFromSource then
          (pythonPackages.buildPythonPackage rec {
            pname = "xgrammar";
            version = "0.2.1";
            format = "wheel";

            src = fetchurl {
              url = "https://files.pythonhosted.org/packages/32/75/25ddd211f073a9db8299bdfec4534874d5d5d5f69499bb0c2ce9bf75f483/xgrammar-${version}-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
              hash = "sha256-ugJAJKRU8x08uIZ5obBz66ATOGDxRSXx9YVvpG3sUig=";
            };

            nativeBuildInputs = [
              autoPatchelfHook
              patchelf
            ];

            buildInputs = [
              pythonPackages.apache-tvm-ffi
              stdenv.cc.cc.lib
            ];

            autoPatchelfIgnoreMissingDeps = [
              "libtvm_ffi.so"
            ];

            dependencies = with pythonPackages; [
              apache-tvm-ffi
              numpy
              pydantic
              torch
              transformers
              triton
              typing-extensions
            ];

            pythonImportsCheck = [ "xgrammar" ];

            postFixup = ''
              patchelf --add-rpath ${lib.escapeShellArg tvmFfiLibDir} \
                "$out/${pythonSitePackages}/xgrammar/libxgrammar_bindings.so"
            '';
          })
        else
          (import ../xgrammar-0_2.nix {
            inherit
              lib
              pythonPackages
              fetchurl
              autoPatchelfHook
              patchelf
              stdenv
              ;
          })
      );
  defaultSrc = fetchFromGitHub {
    owner = "sgl-project";
    repo = "sglang";
    # kimi-k3 branch: adds sglang/srt/models/kimi_k3.py
    rev = "15fa94d2c6724999c299996fe40a5b276a559812";
    hash = "sha256-R4DOQMWx/Zt5I6jBNTGLdocMG9lGoQQzV3SeF0h3aek=";
  };
  resolvedSrc = if packageSrc == null then defaultSrc else packageSrc;
  resolvedPatches =
    if packagePatches == null then
      [
        ./patches/legacy/0001-rocm-hisparse-optional-sgl-kernel.patch
        ./patches/legacy/0002-rocm-default-triton-and-xgrammar-fallback.patch
      ]
    else
      packagePatches;
  tvmFfiLibDir = "${pythonPackages.apache-tvm-ffi}/${pythonSitePackages}/tvm_ffi/lib";
  modelscopeWithCompatibleSetuptools = pythonPackages.modelscope.override {
    setuptools = pythonPackages.setuptools_80;
  };
  fromSource = packageSrc != null;
  version =
    if mediaRuntime then
      (packageVersion)
    else
      (if buildFromSource then (packageVersion) else ("0.5.20"));
in
assert lib.assertMsg (
  rocmSitePackages != null
) "sglang-rocm requires the TheRock torch wheel package with passthru.sitePackages";
pythonPackages.buildPythonApplication (
  rec {
    pname =
      if mediaRuntime then
        ("sglang-rocm${lib.optionalString (packageVariant != null) "-${packageVariant}"}-${packageSuffix}")
      else
        (
          if buildFromSource then
            ("sglang-rocm${lib.optionalString (packageVariant != null) "-${packageVariant}"}-${packageSuffix}")
          else
            ("sglang-rocm-${packageSuffix}")
        );
    version =
      if mediaRuntime then
        (packageVersion)
      else
        (if buildFromSource then (packageVersion) else ("0.5.20"));
    src =
      if mediaRuntime then
        (
          if fromSource then
            packageSrc
          else
            fetchurl {
              url = "https://files.pythonhosted.org/packages/ec/49/bd2e0f7eac9d826cf02b04e64f9285f5c559440dc8ffe31927fe42c26cf1/sglang-${version}-cp313-cp313-manylinux_2_34_x86_64.whl";
              hash = "sha256-ewa/flefwibjbVE5/K8qGnEQ9eE7kGdHYkFmC6AeVac=";
            }
        )
      else
        (
          if buildFromSource then
            (resolvedSrc)
          else
            (fetchurl {
              url = "https://files.pythonhosted.org/packages/ec/49/bd2e0f7eac9d826cf02b04e64f9285f5c559440dc8ffe31927fe42c26cf1/sglang-${version}-cp313-cp313-manylinux_2_34_x86_64.whl";
              hash = "sha256-ewa/flefwibjbVE5/K8qGnEQ9eE7kGdHYkFmC6AeVac=";
            })
        );
    nativeBuildInputs =
      if mediaRuntime then
        ([
          autoPatchelfHook
          makeWrapper
          writableTmpDirAsHomeHook
        ])
      else
        (
          if buildFromSource then
            ([
              makeWrapper
            ])
          else
            ([
              autoPatchelfHook
              makeWrapper
              writableTmpDirAsHomeHook
            ])
        );
    dependencies =
      if mediaRuntime then
        (
          with pythonPackages;
          [
            aiohttp
            anthropic
            apache-tvm-ffi
            blobfile
            build
            compressed-tensors
            datasets
            distro
            easydict
            einops
            fastapi
            gguf
            interegular
            ipython
            sglangDependencies.kernels
            llguidance
            pythonPackages."mistral-common"
            msgspec
            ninja
            numba
            numpy
            nvidia-ml-py
            openai
            openai-harmony
            orjson
            outlines_0_1_11
            packaging
            partial-json-parser
            pillow
            prometheus-client
            psutil
            pybase64
            pydantic
            python-multipart
            pyzmq
            requests
            scipy
            sentencepiece
            setproctitle
            sglangDependencies.smg-grpc-servicer
            soundfile
            tiktoken
            timm
            torch
            torch-memory-saver
            torchaoNoChecks
            torchaudio
            torchvision
            tqdm
            transformers
            uvicorn
            uvloop
            watchfiles
            xgrammar_0_2_1
            xxhash
            zstandard
          ]
          ++ lib.optional hasNativeKernels sglangKernel
          ++ lib.optional (packageAiter != null) packageAiter
          ++ packageExtraDependencies
        )
      else
        (
          if buildFromSource then
            (
              (with pythonPackages; [
                aiohttp
              ])
              ++ lib.optional (packageAiter != null) packageAiter
              ++ (with pythonPackages; [
                anthropic
                apache-tvm-ffi
                blobfile
                build
                compressed-tensors
                datasets
                distro
                easydict
                einops
                fastapi
                gguf
                interegular
                ipython
                kernels
                llguidance
                pythonPackages."mistral-common"
                modelscopeWithCompatibleSetuptools
                msgspec
                ninja
                numba
                numpy
                nvidia-ml-py
                openai
                openai-harmony
                orjson
                outlines_0_1_11
                packaging
                partial-json-parser
                pillow
                prometheus-client
                psutil
                pybase64
                pydantic
                python-multipart
                pyzmq
                requests
                scipy
                sentencepiece
                setproctitle
                smg-grpc-servicer
                soundfile
                tiktoken
                timm
                torch
                torch-memory-saver
                torchaoNoChecks
                torchaudio
                torchvision
                tqdm
                transformers
                uvicorn
                uvloop
                watchfiles
                xgrammar_0_2_1
                xxhash
                zstandard
              ])
              ++ packageExtraDependencies
            )
          else
            (
              with pythonPackages;
              [
                aiohttp
                amd-aiter
                anthropic
                apache-tvm-ffi
                blobfile
                build
                compressed-tensors
                datasets
                distro
                easydict
                einops
                fastapi
                gguf
                interegular
                ipython
                sglangDependencies.kernels
                llguidance
                pythonPackages."mistral-common"
                msgspec
                ninja
                numba
                numpy
                nvidia-ml-py
                openai
                openai-harmony
                orjson
                outlines_0_1_11
                packaging
                partial-json-parser
                pillow
                prometheus-client
                psutil
                pybase64
                pydantic
                python-multipart
                pyzmq
                requests
                scipy
                sentencepiece
                setproctitle
                sglangDependencies.smg-grpc-servicer
                soundfile
                tiktoken
                timm
                torch
                torch-memory-saver
                torchaoNoChecks
                torchaudio
                torchvision
                tqdm
                transformers
                uvicorn
                uvloop
                watchfiles
                xgrammar_0_2_1
                xxhash
                zstandard
              ]
              ++ lib.optional hasNativeKernels sglangKernel
            )
        );
    pythonRemoveDeps =
      if mediaRuntime then
        (
          [
            "cuda-python"
            "cuda-tile"
            "flash-attn-4"
            "flashinfer-cubin"
            "flashinfer-python"
            "flashinfer_cubin"
            "flashinfer_python"
            "helion"
            "humming-kernels"
            # Optional alternative model hub; nixpkgs marks it insecure. Local paths
            # and Hugging Face downloads do not import it.
            "modelscope"
            "nvshmem4py-cu13"
            "nvidia-cutlass-dsl"
            "nvidia-mathdx"
            "py-spy"
            "quack-kernels"
            "sgl-deep-ep"
            "sgl-deep-gemm"
            "tilelang"
            "torchcodec"
            "tokenspeed-mla"
            "tokenspeed_mla"
          ]
          ++ lib.optional (!hasNativeKernels) "sglang-kernel"
        )
      else
        (
          if buildFromSource then
            ([
              "cuda-python"
              "decord2"
              "flash-attn-4"
              "flashinfer-cubin"
              "flashinfer-python"
              "flashinfer_cubin"
              "flashinfer_python"
              "helion"
              "humming-kernels"
              "nvidia-cutlass-dsl"
              "nvidia-mathdx"
              "py-spy"
              "quack-kernels"
              "sgl-deep-gemm"
              "sglang-kernel"
              "tilelang"
              "torchcodec"
              "tokenspeed-mla"
              "tokenspeed_mla"
            ])
          else
            (
              [
                "cuda-python"
                "cuda-tile"
                "flash-attn-4"
                "flashinfer-cubin"
                "flashinfer-python"
                "flashinfer_cubin"
                "flashinfer_python"
                "humming-kernels"
                # Optional alternative model hub; nixpkgs marks it insecure. Local paths
                # and Hugging Face downloads do not import it.
                "modelscope"
                "nvshmem4py-cu13"
                "nvidia-cutlass-dsl"
                "nvidia-mathdx"
                "py-spy"
                "quack-kernels"
                "sgl-deep-ep"
                "sgl-deep-gemm"
                "tilelang"
                "torchcodec"
                "tokenspeed-mla"
                "tokenspeed_mla"
              ]
              ++ lib.optional (!hasNativeKernels) "sglang-kernel"
            )
        );
    pythonRelaxDeps =
      if mediaRuntime then
        ([
          "sglang-kernel"
          "apache-tvm-ffi"
          "blobfile"
          "llguidance"
          "numba"
          "openai"
          "openai-harmony"
          "outlines"
          "soundfile"
          "timm"
          "tokenizers"
          "torch"
          "torchao"
          "torchaudio"
          "torchcodec"
          "transformers"
          "xgrammar"
        ])
      else
        (
          if buildFromSource then
            ([
              "apache-tvm-ffi"
              "blobfile"
              "kernels"
              "llguidance"
              "numba"
              "openai"
              "openai-harmony"
              "outlines"
              "soundfile"
              "timm"
              "torch"
              "torchao"
              "torchaudio"
              "torchcodec"
              "transformers"
              "xgrammar"
            ])
          else
            ([
              "sglang-kernel"
              "apache-tvm-ffi"
              "blobfile"
              "llguidance"
              "numba"
              "openai"
              "openai-harmony"
              "outlines"
              "soundfile"
              "timm"
              "tokenizers"
              "torch"
              "torchaudio"
              "torchcodec"
              "transformers"
              "xgrammar"
            ])
        );
    postFixup =
      if mediaRuntime then
        (''
          wrapPythonPrograms

          rocm_site=${lib.escapeShellArg rocmSitePackages}
          rocm_lib_path="$(find "$rocm_site" -type d \( -name lib -o -name lib64 \) -print | paste -sd:)"
          # RCCL opens libibverbs dynamically. Without it, requesting RoCE silently
          # selects the socket transport even when the host has working RDMA devices.
          rocm_lib_path=${
            lib.escapeShellArg (lib.makeLibraryPath [ rdma-core ])
          }:${lib.escapeShellArg rocmRuntimeLibraryPath}:"$rocm_lib_path"

          wrap_args=(
            --set HIP_PLATFORM amd
            --set ROCM_HOME ${lib.escapeShellArg rocmSdkForJit}
            --set ROCM_PATH ${lib.escapeShellArg rocmSdkForJit}
            --set HIP_PATH ${lib.escapeShellArg rocmSdkForJit}
            --set CC ${lib.escapeShellArg "${stdenv.cc}/bin/cc"}
            --set CXX ${lib.escapeShellArg "${rocmSdkForJit}/bin/therock-hip-clang++"}
            --run 'if [ -z "''${AITER_JIT_DIR:-}" ]; then export AITER_JIT_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/aiter/${packageCacheTag}-${packageSuffix}-py${pythonTag}/jit"; fi'
            --run 'if [ -z "''${AITER_ROOT_DIR:-}" ]; then export AITER_ROOT_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/aiter/${packageCacheTag}-${packageSuffix}-py${pythonTag}/root"; fi'
            --prefix PATH : ${
              lib.escapeShellArg (
                lib.makeBinPath [
                  rocmSdkForJit
                  gnumake
                  pythonPackages.ninja
                ]
              )
            }
            --prefix LD_LIBRARY_PATH : "$rocm_lib_path"
          )
          ${lib.optionalString (gpuArch != null) ''
            wrap_args+=(
              --set GPU_ARCHS ${lib.escapeShellArg gpuArch}
              --set PYTORCH_ROCM_ARCH ${lib.escapeShellArg gpuArch}
            )
          ''}
          ${lib.optionalString (hsaOverrideGfxVersion != null) ''
            wrap_args+=(--set HSA_OVERRIDE_GFX_VERSION ${lib.escapeShellArg hsaOverrideGfxVersion})
          ''}

          for bin in "$out/bin/sglang" "$out/bin/killall_sglang" "$out/bin/sglang-python"; do
            [ -x "$bin" ] || continue
            wrapProgram "$bin" "''${wrap_args[@]}"
          done
        '')
      else
        (
          if buildFromSource then
            (''
              wrapPythonPrograms

              rocm_site=${lib.escapeShellArg rocmSitePackages}
              rocm_lib_path="$(find "$rocm_site" -type d \( -name lib -o -name lib64 \) -print | paste -sd:)"
              rocm_lib_path=${lib.escapeShellArg rocmRuntimeLibraryPath}:"$rocm_lib_path"

              wrap_args=(
                --set HIP_PLATFORM amd
                --set ROCM_HOME ${lib.escapeShellArg rocmSdkForJit}
                --set ROCM_PATH ${lib.escapeShellArg rocmSdkForJit}
                --set HIP_PATH ${lib.escapeShellArg rocmSdkForJit}
                --set CC ${lib.escapeShellArg "${stdenv.cc}/bin/cc"}
                --set CXX ${lib.escapeShellArg "${rocmSdkForJit}/bin/therock-hip-clang++"}
                --run 'if [ -z "''${AITER_JIT_DIR:-}" ]; then export AITER_JIT_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/aiter/${packageCacheTag}-${packageSuffix}-py${pythonTag}/jit"; fi'
                --run 'if [ -z "''${AITER_ROOT_DIR:-}" ]; then export AITER_ROOT_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/aiter/${packageCacheTag}-${packageSuffix}-py${pythonTag}/root"; fi'
                --prefix PATH : ${
                  lib.escapeShellArg (
                    lib.makeBinPath [
                      rocmSdkForJit
                      gnumake
                      pythonPackages.ninja
                    ]
                  )
                }
                --prefix LD_LIBRARY_PATH : "$rocm_lib_path"
              )
              ${lib.optionalString (gpuArch != null) ''
                wrap_args+=(
                  --set GPU_ARCHS ${lib.escapeShellArg gpuArch}
                  --set PYTORCH_ROCM_ARCH ${lib.escapeShellArg gpuArch}
                  --set TVM_FFI_ROCM_ARCH_LIST ${lib.escapeShellArg gpuArch}
                )
              ''}
              ${lib.optionalString (hsaOverrideGfxVersion != null) ''
                wrap_args+=(--set HSA_OVERRIDE_GFX_VERSION ${lib.escapeShellArg hsaOverrideGfxVersion})
              ''}

              for bin in "$out/bin/sglang" "$out/bin/killall_sglang"; do
                [ -x "$bin" ] || continue
                wrapProgram "$bin" "''${wrap_args[@]}"
              done
            '')
          else
            (''
              wrapPythonPrograms

              rocm_site=${lib.escapeShellArg rocmSitePackages}
              rocm_lib_path="$(find "$rocm_site" -type d \( -name lib -o -name lib64 \) -print | paste -sd:)"
              # RCCL opens libibverbs dynamically. Without it, requesting RoCE silently
              # selects the socket transport even when the host has working RDMA devices.
              rocm_lib_path=${
                lib.escapeShellArg (lib.makeLibraryPath [ rdma-core ])
              }:${lib.escapeShellArg rocmRuntimeLibraryPath}:"$rocm_lib_path"

              wrap_args=(
                # Persistent KV objects must not cross runtime/kernel build identities.
                --set SGLANG_CACHE_NUMERICS_ID "$out"
                # HIP initialization in a parent sets SDK variables inherited by spawned
                # workers. Register this wheel's Kineto tool so they record device events,
                # while preserving an explicit user-selected profiler tool.
                --set-default ROCP_TOOL_LIBRARIES ${lib.escapeShellArg "${rocmSitePackages}/torch/lib/libtorch_cpu.so"}
                --set HIP_PLATFORM amd
                --set ROCM_HOME ${lib.escapeShellArg rocmSdkForJit}
                --set ROCM_PATH ${lib.escapeShellArg rocmSdkForJit}
                --set HIP_PATH ${lib.escapeShellArg rocmSdkForJit}
                --set CC ${lib.escapeShellArg "${stdenv.cc}/bin/cc"}
                --set CXX ${lib.escapeShellArg "${rocmSdkForJit}/bin/therock-hip-clang++"}
                --run 'if [ -z "''${AITER_JIT_DIR:-}" ]; then export AITER_JIT_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/aiter/sglang-${version}-${packageSuffix}-py${pythonTag}/jit"; fi'
                --run 'if [ -z "''${AITER_ROOT_DIR:-}" ]; then export AITER_ROOT_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/aiter/sglang-${version}-${packageSuffix}-py${pythonTag}/root"; fi'
                --prefix PATH : ${
                  lib.escapeShellArg (
                    lib.makeBinPath [
                      rocmSdkForJit
                      gnumake
                      pythonPackages.ninja
                    ]
                  )
                }
                --prefix LD_LIBRARY_PATH : "$rocm_lib_path"
              )
              ${lib.optionalString (gpuArch != null) ''
                wrap_args+=(
                  --set GPU_ARCHS ${lib.escapeShellArg gpuArch}
                  --set PYTORCH_ROCM_ARCH ${lib.escapeShellArg gpuArch}
                )
              ''}
              ${lib.optionalString (hsaOverrideGfxVersion != null) ''
                wrap_args+=(--set HSA_OVERRIDE_GFX_VERSION ${lib.escapeShellArg hsaOverrideGfxVersion})
              ''}

              for bin in "$out/bin/sglang" "$out/bin/killall_sglang" "$out/bin/sglang-python"; do
                [ -x "$bin" ] || continue
                wrapProgram "$bin" "''${wrap_args[@]}"
              done
            '')
        );
    pythonImportsCheck =
      if mediaRuntime then
        (
          [
            "sglang"
          ]
          ++ lib.optionals (!fromSource) [
            "sglang.srt.configs.glm5_next"
            "sglang.srt.server_args"
            "sglang.srt.mem_cache.rust_tree_core.mem_cache"
          ]
        )
      else
        (
          if buildFromSource then
            ([
              "sglang"
            ])
          else
            ([
              "sglang"
              "sglang.srt.configs.glm5_next"
              "sglang.srt.server_args"
              "sglang.srt.mem_cache.rust_tree_core.mem_cache"
            ])
        );
    meta = {
      description = "SGLang serving framework packaged against the TheRock ROCm Python stack";
      homepage = "https://github.com/sgl-project/sglang";
      license = lib.licenses.asl20;
      mainProgram = "sglang";
      maintainers = with lib.maintainers; [ georgewhewell ];
      platforms = [ "x86_64-linux" ];
    };
  }
  // lib.optionalAttrs (!mediaRuntime && !buildFromSource) {
    format = "wheel";
    buildInputs = [
      stdenv.cc.cc.lib
    ];
    dontUseNinjaBuild = true;
    postInstall = ''
      ${lib.concatMapStringsSep "\n" (patch: ''
        patch --batch --fuzz=0 -p1 -d "$out/${pythonSitePackages}" < ${patch}
      '') runtimePatches}
      # Keep component checks in the same Python/ROCm environment as serving.
      cat > "$out/bin/sglang-python" <<'PY'
      #!${pythonPackages.python.interpreter}
      import os
      import runpy
      import sys

      if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
          print("Usage: sglang-python SCRIPT [ARGS...]")
          raise SystemExit(0 if len(sys.argv) > 1 else 2)
      sys.argv = sys.argv[1:]
      sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[0])))
      runpy.run_path(sys.argv[0], run_name="__main__")
      PY
      chmod +x "$out/bin/sglang-python"
    '';
    preFixup = ''
      # The Rust radix-tree extension links against libtorch from the wheel.
      addAutoPatchelfSearchPath ${lib.escapeShellArg "${rocmSitePackages}/torch/lib"}
    '';
  }
  // lib.optionalAttrs (!mediaRuntime && buildFromSource) {
    pyproject = true;
    sourceRoot = "${resolvedSrc.name}/python";
    env.SETUPTOOLS_SCM_PRETEND_VERSION = version;
    env.SGLANG_BUILD_RUST_EXTS = "none";
    patches = resolvedPatches;
    postPatch = packagePostPatch;
    build-system = with pythonPackages; [
      setuptools
      setuptools-scm
      # Required by pyproject's build-system, but SGLANG_BUILD_RUST_EXTS=none
      # means no crate is compiled; drop the hook that expects cargo vendor env.
      (setuptools-rust.overrideAttrs (_old: {
        setupHook = null;
      }))
      wheel
    ];
    doCheck = packageDoCheck;
    doInstallCheck = packageDoInstallCheck;
  }
  // lib.optionalAttrs (mediaRuntime) {
    format = if fromSource then "pyproject" else "wheel";
    sourceRoot = if fromSource then "${packageSrc.name}/python" else null;
    env = lib.optionalAttrs fromSource {
      SETUPTOOLS_SCM_PRETEND_VERSION = version;
      SGLANG_BUILD_RUST_EXTS = "none";
    };
    patches = packagePatches;
    postPatch = packagePostPatch;
    build-system = lib.optionals fromSource (
      with pythonPackages;
      [
        setuptools
        setuptools-scm
        (setuptools-rust.overrideAttrs (_: {
          setupHook = null;
        }))
        wheel
      ]
    );
    doCheck = packageDoCheck;
    doInstallCheck = packageDoInstallCheck;
    buildInputs = [
      stdenv.cc.cc.lib
    ];
    dontUseNinjaBuild = true;
    postInstall =
      lib.optionalString (!fromSource) ''
        for patch_file in ${./patches}/*.patch; do
          patch --batch --fuzz=0 -p1 -d "$out/${pythonSitePackages}" < "$patch_file"
        done
      ''
      + ''
        # Keep component checks in the same Python/ROCm environment as serving.
        cat > "$out/bin/sglang-python" <<'PY'
        #!${pythonPackages.python.interpreter}
        import os
        import runpy
        import sys

        if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
            print("Usage: sglang-python SCRIPT [ARGS...]")
            raise SystemExit(0 if len(sys.argv) > 1 else 2)
        sys.argv = sys.argv[1:]
        sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[0])))
        runpy.run_path(sys.argv[0], run_name="__main__")
        PY
        chmod +x "$out/bin/sglang-python"
      '';
    preFixup = ''
      # The Rust radix-tree extension links against libtorch from the wheel.
      addAutoPatchelfSearchPath ${lib.escapeShellArg "${rocmSitePackages}/torch/lib"}
    '';
  }
)
