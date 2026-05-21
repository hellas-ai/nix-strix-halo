{
  lib,
  buildPythonPackage,
  fetchPypi,
  python,
  numpy,
  fastrlock,
  rocmPackages,
  autoPatchelfHook,
}:

# CuPy ROCm-7.x variant. cupy-rocm-7-0 ships as a 70 MB manylinux wheel
# with the ROCm runtime calls already bundled — we just need to make
# sure the system ROCm libraries are findable at import time.
#
# vllm + lmcache import this on ROCm to manage CUDA-API-compatible GPU
# memory (cupy.cuda.MemoryPool etc.) when running on AMD.

let
  wheelInfo =
    if python.pythonVersion == "3.13" then
      {
        hash = "sha256-6+eLTMN+YZrkJVXgnWwY0GVSRyq80Sp9zEQB0ba9C7A=";
        pyTag = "cp313";
      }
    else
      throw "cupy-rocm-7-0: no wheel mapping for python ${python.pythonVersion}";
in
buildPythonPackage rec {
  pname = "cupy-rocm-7-0";
  version = "14.0.1";
  format = "wheel";

  src = fetchPypi {
    pname = "cupy_rocm_7_0";
    inherit version;
    format = "wheel";
    dist = wheelInfo.pyTag;
    python = wheelInfo.pyTag;
    abi = wheelInfo.pyTag;
    platform = "manylinux2014_x86_64";
    inherit (wheelInfo) hash;
  };

  nativeBuildInputs = [ autoPatchelfHook ];

  buildInputs = with rocmPackages; [
    clr
    rocm-runtime
    rocm-device-libs
    hipblas
    hipblas-common
    hipblaslt
    hipfft
    hiprand
    hipsolver
    hipsparse
    miopen-hip
    rocblas
    rocrand
    rocsolver
    rocsparse
  ];

  dependencies = [
    fastrlock
    numpy
  ];

  pythonRelaxDeps = true;
  pythonImportsCheck = [ "cupy" ];

  # autoPatchelfHook is strict about not finding rocrand etc; skip as
  # a fallback when the prebuilt wheel ships hard-coded versioned
  # paths (`librocrand.so.X`) we can't resolve. Run with strict=false
  # by default and rely on RUNPATH discovery at import time.
  autoPatchelfIgnoreMissingDeps = true;

  meta = {
    description = "CuPy: NumPy-compatible array library for ROCm 7.x";
    homepage = "https://cupy.dev/";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
  };
}
