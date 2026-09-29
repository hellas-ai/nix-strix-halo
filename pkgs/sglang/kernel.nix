{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  pythonPackages,
  rocmSdk,
  packageSuffix,
  gpuArch,
}:

pythonPackages.buildPythonPackage {
  pname = "sglang-kernel-${packageSuffix}";
  version = "0.4.7";
  format = "setuptools";

  # Keep the native code on the same release as the Python runtime.
  src = fetchurl {
    name = "sglang-v0.5.20.tar.gz";
    url = "https://github.com/sgl-project/sglang/archive/refs/tags/v0.5.20.tar.gz";
    hash = "sha256-s/pR1lTVKWLF3rdUmZrhiu6k0G8T1Jf9xc8CRqHfrJs=";
  };
  sourceRoot = "sglang-0.5.20/python/sglang/kernels/aot";

  postPatch = ''
    cp pyproject_rocm.toml pyproject.toml
    ${lib.optionalString (gpuArch == "gfx1151") ''
      sh ../../../../docker/patches/sgl-kernel-gfx1151.sh setup_rocm.py
    ''}
    # TheRock's Nix-aware hipcc wrapper invokes Clang directly.
    substituteInPlace setup_rocm.py \
      --replace-fail '--amdgpu-target=' '--offload-arch='
    cp setup_rocm.py setup.py
  '';

  nativeBuildInputs = [
    autoPatchelfHook
    pythonPackages.ninja
  ];
  buildInputs = [
    rocmSdk
    stdenv.cc.cc.lib
  ];
  build-system = with pythonPackages; [
    setuptools
    torch
    wheel
  ];
  dependencies = with pythonPackages; [ torch ];

  dontUseNinjaBuild = true;
  env = {
    ROCM_HOME = "${rocmSdk}";
    HIP_PATH = "${rocmSdk}";
    HIP_PLATFORM = "amd";
    AMDGPU_TARGET = gpuArch;
    PYTORCH_ROCM_ARCH = gpuArch;
    CXX = "${rocmSdk}/bin/therock-hip-clang++";
    LD_LIBRARY_PATH = pythonPackages.torch.passthru.rocmRuntimeEnv.LD_LIBRARY_PATH;
  };
  preBuild = ''
    export MAX_JOBS="$NIX_BUILD_CORES"
  '';
  preFixup = ''
    addAutoPatchelfSearchPath "${pythonPackages.torch.passthru.sitePackages}/torch/lib"
  '';

  # Importing the extension registers the HIP ops without requiring a GPU.
  pythonImportsCheck = [ "sgl_kernel" ];
  meta = {
    description = "SGLang HIP kernels for ${packageSuffix}";
    homepage = "https://github.com/sgl-project/sglang";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
