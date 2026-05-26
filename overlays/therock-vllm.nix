{
  lib,
  target,
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
        dependencies = dropNamedDeps dropVllmDependencyNames (old.dependencies or [ ]) ++ [
          final.python312Packages.amdsmi
          final.python312Packages.cloudpickle
        ];
        propagatedBuildInputs =
          dropNamedDeps dropVllmDependencyNames (old.propagatedBuildInputs or [ ])
          ++ [
            final.python312Packages.amdsmi
            final.python312Packages.cloudpickle
          ];
      });
in
{
  "vllm-rocm-therock-${s}" = vllmTherock;
}
