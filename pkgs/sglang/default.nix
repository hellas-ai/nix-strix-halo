{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  gnumake,
  makeWrapper,
  patchelf,
  symlinkJoin,
  python312Packages,
  rocmSdk,
  packageSuffix ? "rocm",
  hsaOverrideGfxVersion ? null,
}:

let
  pythonSitePackages = python312Packages.python.sitePackages;
  rocmSitePackages = python312Packages.torch.passthru.sitePackages or null;
  gpuArch = if lib.hasPrefix "gfx" packageSuffix then packageSuffix else null;
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
  tvmFfiLibDir = "${python312Packages.apache-tvm-ffi}/${pythonSitePackages}/tvm_ffi/lib";
  outlines-core_0_1_26 = python312Packages.buildPythonPackage rec {
    pname = "outlines-core";
    version = "0.1.26";
    format = "wheel";

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/e2/1d/a36292b6198986bd9c3ff8c24355deb82ed5475403379ee40b5b5473e2e3/outlines_core-${version}-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
      hash = "sha256-6GobtGrcXL9t/Xp/5BBeDipMbgQXMqBTEmtBxSGh8iM=";
    };

    nativeBuildInputs = [
      autoPatchelfHook
    ];

    buildInputs = [
      stdenv.cc.cc.lib
    ];

    dependencies = with python312Packages; [
      interegular
      jsonschema
    ];

    pythonImportsCheck = [ "outlines_core" ];
  };
  outlines_0_1_11 = python312Packages.buildPythonPackage rec {
    pname = "outlines";
    version = "0.1.11";
    format = "wheel";

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/13/b4/99ea4a122bef60e3fd6402d19665aff1f928e0daf8fac3044d0b73f72003/outlines-${version}-py3-none-any.whl";
      hash = "sha256-9aXyJC7ZgC06q3qSeJv0AI1zTFdr6SWMwKKX9pASRyc=";
    };

    dependencies = with python312Packages; [
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
  torchaoNoChecks = python312Packages.torchao.overridePythonAttrs (old: {
    doCheck = false;
    nativeCheckInputs = [ ];
    pythonImportsCheck = old.pythonImportsCheck or [ "torchao" ];
  });
  xgrammar_0_2_1 = python312Packages.buildPythonPackage rec {
    pname = "xgrammar";
    version = "0.2.1";
    format = "wheel";

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/96/4b/327b3cf702b685a2be28d15490faa4beeac00c4fbcf9bb2d7db0fda32931/xgrammar-${version}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      hash = "sha256-y8YBTcHJL8MXsUUZEhyBY/41/ZNBeeWkXYP3gP8jGCY=";
    };

    nativeBuildInputs = [
      autoPatchelfHook
      patchelf
    ];

    buildInputs = [
      python312Packages.apache-tvm-ffi
      stdenv.cc.cc.lib
    ];

    autoPatchelfIgnoreMissingDeps = [
      "libtvm_ffi.so"
    ];

    dependencies = with python312Packages; [
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
python312Packages.buildPythonApplication rec {
  pname = "sglang-rocm-${packageSuffix}";
  version = "0.5.14";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/1a/f8/49727c6252937cd8b97aade22c7c0c6be86ea53b4a7f64eebb8cbf8accdd/sglang-${version}-cp312-cp312-manylinux_2_34_x86_64.whl";
    hash = "sha256-RXOqx3QEW6kmurL376qtexIOT4RpXM8nxZ0nb4Vt7Kg=";
  };

  nativeBuildInputs = [
    autoPatchelfHook
    makeWrapper
  ];

  buildInputs = [
    stdenv.cc.cc.lib
  ];

  dontUseNinjaBuild = true;

  dependencies = with python312Packages; [
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
    kernels
    llguidance
    python312Packages."mistral-common"
    modelscope
    msgspec
    ninja
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
  ];

  pythonRemoveDeps = [
    "cuda-python"
    "flash-attn-4"
    "flashinfer-cubin"
    "flashinfer-python"
    "flashinfer_cubin"
    "flashinfer_python"
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
  ];

  pythonRelaxDeps = [
    "apache-tvm-ffi"
    "blobfile"
    "kernels"
    "llguidance"
    "openai"
    "openai-harmony"
    "outlines"
    "timm"
    "torch"
    "torchaudio"
    "torchcodec"
    "transformers"
    "xgrammar"
  ];

  postInstall = ''
    for patch_file in ${./patches}/*.patch; do
      patch -p1 -d "$out/${pythonSitePackages}" < "$patch_file"
    done
  '';

  postFixup = ''
    wrapPythonPrograms

    rocm_site=${lib.escapeShellArg rocmSitePackages}
    rocm_lib_path="$(find "$rocm_site" -type d \( -name lib -o -name lib64 \) -print | paste -sd:)"

    wrap_args=(
      --set HIP_PLATFORM amd
      --set ROCM_HOME ${lib.escapeShellArg rocmSdkForJit}
      --set ROCM_PATH ${lib.escapeShellArg rocmSdkForJit}
      --set HIP_PATH ${lib.escapeShellArg rocmSdkForJit}
      --set CC ${lib.escapeShellArg "${stdenv.cc}/bin/cc"}
      --set CXX ${lib.escapeShellArg "${rocmSdkForJit}/bin/therock-hip-clang++"}
      --run 'if [ -z "''${AITER_JIT_DIR:-}" ]; then export AITER_JIT_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/aiter/sglang-${version}-${packageSuffix}-py312/jit"; fi'
      --run 'if [ -z "''${AITER_ROOT_DIR:-}" ]; then export AITER_ROOT_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/aiter/sglang-${version}-${packageSuffix}-py312/root"; fi'
      --prefix PATH : ${
        lib.escapeShellArg (
          lib.makeBinPath [
            rocmSdkForJit
            gnumake
            python312Packages.ninja
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
