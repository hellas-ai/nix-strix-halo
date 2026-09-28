{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  cargo,
  gnumake,
  makeWrapper,
  openssl,
  patchelf,
  pkg-config,
  rustPlatform,
  rustc,
  symlinkJoin,
  writableTmpDirAsHomeHook,
  pythonPackages,
  rocmSdk,
  packageSuffix ? "rocm",
  hsaOverrideGfxVersion ? null,
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
  hasNativeKernels = lib.elem gpuArch [
    "gfx1151"
    "gfx942"
    "gfx950"
    "gfx1250"
  ];
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
  tvmFfiLibDir = "${pythonPackages.apache-tvm-ffi}/${pythonSitePackages}/tvm_ffi/lib";
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
  torchaoNoChecks = pythonPackages.torchao.overridePythonAttrs (old: {
    doCheck = false;
    # ROCm-enabled torchao 0.17 builds its swizzle extension against the
    # hipBLASLt C++ API. The Python wheel closure carries the runtime library,
    # but ROCm 10 no longer carries this header in its core wheel.
    buildInputs = (old.buildInputs or [ ]) ++ [ rocmSdk ];
    nativeCheckInputs = [ ];
    pythonImportsCheck = old.pythonImportsCheck or [ "torchao" ];
  });
  xgrammar_0_2_1 = pythonPackages.buildPythonPackage rec {
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
  };
in
assert lib.assertMsg (
  rocmSitePackages != null
) "sglang-rocm requires the TheRock torch wheel package with passthru.sitePackages";
pythonPackages.buildPythonApplication rec {
  pname = "sglang-rocm-${packageSuffix}";
  version = "0.5.20";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/ec/49/bd2e0f7eac9d826cf02b04e64f9285f5c559440dc8ffe31927fe42c26cf1/sglang-${version}-cp313-cp313-manylinux_2_34_x86_64.whl";
    hash = "sha256-ewa/flefwibjbVE5/K8qGnEQ9eE7kGdHYkFmC6AeVac=";
  };

  nativeBuildInputs = [
    autoPatchelfHook
    makeWrapper
    writableTmpDirAsHomeHook
  ];

  buildInputs = [
    stdenv.cc.cc.lib
  ];

  dontUseNinjaBuild = true;

  dependencies =
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
    ++ lib.optional hasNativeKernels sglangKernel;

  pythonRemoveDeps = [
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
  ++ lib.optional (!hasNativeKernels) "sglang-kernel";

  pythonRelaxDeps = [
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
  ];

  postInstall = ''
    for patch_file in ${./patches}/*.patch; do
      patch --batch --fuzz=0 -p1 -d "$out/${pythonSitePackages}" < "$patch_file"
    done
  '';

  preFixup = ''
    # The Rust radix-tree extension links against libtorch from the wheel.
    addAutoPatchelfSearchPath ${lib.escapeShellArg "${rocmSitePackages}/torch/lib"}
  '';

  postFixup = ''
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

    for bin in "$out/bin/sglang" "$out/bin/killall_sglang"; do
      [ -x "$bin" ] || continue
      wrapProgram "$bin" "''${wrap_args[@]}"
    done
  '';

  pythonImportsCheck = [
    "sglang"
    "sglang.srt.configs.glm5_next"
    "sglang.srt.server_args"
    "sglang.srt.mem_cache.rust_tree_core.mem_cache"
  ];

  meta = {
    description = "SGLang serving framework packaged against the TheRock ROCm Python stack";
    homepage = "https://github.com/sgl-project/sglang";
    license = lib.licenses.asl20;
    mainProgram = "sglang";
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = [ "x86_64-linux" ];
  };
}
