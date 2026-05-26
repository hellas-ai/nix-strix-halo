{
  lib,
  target,
  vllmSrc,
  vllmVersion,
}:
let
  s = target.packageSuffix;
in
final: prev:
let
  sdkBase = final."therock-rocm-${s}";
  vllmGpuTargets = target.buildTargets;
  sdk = sdkBase // {
    localGpuTargets = vllmGpuTargets;
    gpuTargets = vllmGpuTargets;
  };
  tritonKernels = prev.fetchFromGitHub {
    owner = "triton-lang";
    repo = "triton";
    tag = "v3.6.0";
    hash = "sha256-JFSpQn+WsNnh7CAPlcpOcUp0nyKXNbJEANdXqmkt4Tc=";
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

  dropVllmDependencyNames = [
    "amd-quark"
    "bitsandbytes"
    "conch-triton-kernels"
    "datasets"
    "mistral-common"
    "mistral_common"
    "mistralai"
    "opencv-python-headless"
    "outlines"
    "peft"
    "pyarrow"
    "pytest-asyncio"
    "runai-model-streamer"
    "runai_model_streamer"
    "tensorizer"
    "tilelang"
    "timm"
    "torchcodec"
    "torchvision"
    "xformers"
  ];

  dropNamedDeps =
    names: deps:
    prev.lib.filter (
      dep:
      let
        name = dep.pname or dep.name or "";
      in
      !(prev.lib.elem name names)
    ) deps;

  vllmTherock =
    (final.python312Packages.vllm.override {
      rocmSupport = true;
      cudaSupport = false;
      gpuTargets = vllmGpuTargets;
      rocmPackages = therockRocmPackages;
      amdsmi = final.python312Packages.amdsmi;
    }).overridePythonAttrs
      (old: {
        version = vllmVersion;
        src = vllmSrc;

        patches =
          builtins.filter (patch: !(lib.hasSuffix "0006-drop-rocm-extra-reqs.patch" (toString patch))) (
            old.patches or [ ]
          )
          ++ [
            ../pkgs/vllm/patches/0001-hipify-copy-unchanged-cu-to-hip.patch
          ];

        postPatch = ''
          rm vllm/third_party/pynvml.py
          substituteInPlace tests/utils.py \
            --replace-fail \
              "from vllm.third_party.pynvml import" \
              "from pynvml import"
          substituteInPlace vllm/utils/import_utils.py \
            --replace-fail \
              "import vllm.third_party.pynvml as pynvml" \
              "import pynvml"

          substituteInPlace pyproject.toml \
            --replace-fail '"torch == 2.11.0"' '"torch"' \
            --replace-fail "setuptools>=77.0.3,<81.0.0" "setuptools"

          substituteInPlace CMakeLists.txt \
            --replace-fail \
              'set(PYTHON_SUPPORTED_VERSIONS' \
              'set(PYTHON_SUPPORTED_VERSIONS "${lib.versions.majorMinor final.python312.version}"'
        '';

        env = (old.env or { }) // {
          VLLM_VERSION_OVERRIDE = vllmVersion;
          HIP_PATH = "${sdk}";
          HIP_PLATFORM = "amd";
          TRITON_KERNELS_SRC_DIR = "${tritonKernels}/python/triton_kernels/triton_kernels";
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
        dependencies = dropNamedDeps dropVllmDependencyNames (old.dependencies or [ ]) ++ [
          final.python312Packages.amdsmi
          final.python312Packages.cloudpickle
          final.python312Packages.diskcache
          final.python312Packages.lark
          final.python312Packages.outlines-core
          final.python312Packages.pillow
          final.python312Packages.prometheus-client
          final.python312Packages.protobuf
          final.python312Packages.pyyaml
          final.python312Packages.regex
          final.python312Packages.requests
          final.python312Packages.six
          final.python312Packages.tqdm
          final.python312Packages.watchfiles
        ];
        propagatedBuildInputs =
          dropNamedDeps dropVllmDependencyNames (old.propagatedBuildInputs or [ ])
          ++ [
            final.python312Packages.amdsmi
            final.python312Packages.cloudpickle
            final.python312Packages.diskcache
            final.python312Packages.lark
            final.python312Packages.outlines-core
            final.python312Packages.pillow
            final.python312Packages.prometheus-client
            final.python312Packages.protobuf
            final.python312Packages.pyyaml
            final.python312Packages.regex
            final.python312Packages.requests
            final.python312Packages.six
            final.python312Packages.tqdm
            final.python312Packages.watchfiles
          ];
      });
in
{
  "vllm-rocm-therock-${s}" = vllmTherock;
}
