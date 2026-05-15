{
  lib,
  stdenv,
  buildPythonPackage,
  fetchPypi,
  python,
  setuptools,
  setuptools-scm,
  ninja,
  packaging,
  wheel,
  cmake,
  rocmPackages,
  pybind11,

  # runtime
  aiofile,
  aiofiles,
  aiohttp,
  awscrt,
  blake3,
  cupy-rocm-7-0,
  fastapi,
  httptools,
  httpx,
  msgspec,
  numba,
  numpy,
  opentelemetry-api,
  opentelemetry-exporter-otlp,
  opentelemetry-exporter-prometheus,
  opentelemetry-sdk,
  prometheus-client,
  psutil,
  py-cpuinfo,
  pyyaml,
  pyzmq,
  redis,
  rixl ? null,
  safetensors,
  sortedcontainers,
  torch,
  transformers,
  uvicorn,
}:

# LMCache 0.4.4 with ROCm support. vLLM imports `lmcache` for KV-cache
# management when `--kv-transfer-config` is set; on ROCm we need the
# `BUILD_WITH_HIP=1` setup.py path which:
#   1. Hipifies csrc/*.cu  -> csrc_hip/*
#   2. Picks rocm_core.txt deps (cupy-rocm-7-0) over cuda_core.txt
#
# Some unconditional CUDA-only deps in `requirements/common.txt`
# (cufile-python, nvtx) are stripped via postPatch — both have lazy
# imports inside lmcache so removal is safe at runtime; ROCm has its
# own HipFile fallback and nvtx is wrapped in try/except already.

buildPythonPackage rec {
  pname = "lmcache";
  version = "0.4.4";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-7f0vthR6Jucwksj/006FLB4dlvMjZFfYh5xClEz/vTk=";
  };

  postPatch = ''
    # Drop CUDA-only runtime deps that lmcache pip-pulls unconditionally.
    # cufile-python: NVIDIA GPUDirect Storage; only used inside
    #   CuFileMemoryAllocator (lazy-imported in __init__).
    # nvtx: NVIDIA profiling annotations; lmcache/utils.py wraps the
    #   import in try/except and falls back to a no-op decorator.
    substituteInPlace requirements/common.txt \
      --replace-fail "cufile-python" "" \
      --replace-fail "nvtx" ""

    # pyproject.toml [build-system].requires pins torch==2.10.0 (lmcache's
    # release wheel-build target). nixpkgs has 2.11.0; pythonRelaxDeps
    # only affects runtime deps, not build-system deps. Loosen it.
    substituteInPlace pyproject.toml \
      --replace-fail '"torch==2.10.0"' '"torch"' \
      --replace-fail "setuptools>=77.0.3,<81.0.0" "setuptools"
  '';

  build-system = [
    setuptools
    setuptools-scm
    ninja
    packaging
    wheel
    pybind11
    torch
  ];

  # CMake is invoked by lmcache's own setup.py, not by stdenv's cmake
  # hook — keep it as a regular nativeBuildInput so it's on PATH and
  # disable the auto-configure phase the hook would otherwise run.
  dontUseCmakeConfigure = true;

  nativeBuildInputs = [
    cmake
    rocmPackages.hipcc
  ];

  buildInputs = with rocmPackages; [
    clr
    rocm-runtime
    rocm-device-libs
    rocthrust
    rocprim
    hipcub
    hipsparse
    hipblas
    hipblas-common
    hipblaslt
    hipfft
    hiprand
    hipsolver
    miopen-hip
    rocblas
    rocrand
    rocsolver
    rocsparse
  ];

  env = {
    BUILD_WITH_HIP = "1";
    ROCM_PATH = "${rocmPackages.clr}";
    PYTORCH_ROCM_ARCH = "gfx1151";
    # Torch's ATen/hip headers transitively include `hipsparse/hipsparse.h`,
    # `thrust/complex.h`, etc. lmcache's setup.py builds via torch's
    # `cpp_extension`, which spawns hipcc/clang with explicit -I args
    # *only* for `${ROCM_PATH}/include` — none of the per-package nixpkgs
    # ROCm derivations (rocthrust, hipsparse, etc.) get on the path.
    # CPATH is the canonical "extra include dirs" env var honoured by
    # both g++ and hipcc/clang, so this is the cheapest single-knob fix.
    CPATH = lib.concatStringsSep ":" (
      (map (p: "${lib.getInclude p}/include") (with rocmPackages; [
        rocthrust
        rocprim
        hipcub
        hipsparse
        hipblas
        hipblas-common
        hipblaslt
        hipfft
        hiprand
        hipsolver
        miopen-hip
        rocblas
        rocrand
        rocsolver
        rocsparse
      ]))
      # pybind11 headers — lmcache's csrc/cachegen_kernels_hip.cuh
      # includes <pybind11/pybind11.h>. The python pybind11 wheel
      # places them under .../site-packages/pybind11/include.
      ++ [ "${pybind11}/lib/${python.libPrefix}/site-packages/pybind11/include" ]
    );
  };

  dependencies = [
    aiofile
    aiofiles
    aiohttp
    awscrt
    blake3
    cupy-rocm-7-0
    fastapi
    httptools
    httpx
    msgspec
    numba
    numpy
    opentelemetry-api
    opentelemetry-exporter-otlp
    opentelemetry-exporter-prometheus
    opentelemetry-sdk
    prometheus-client
    psutil
    py-cpuinfo
    pyyaml
    pyzmq
    redis
    safetensors
    sortedcontainers
    torch
    transformers
    uvicorn
  ]
  ++ lib.optionals (rixl != null) [ rixl ];

  pythonRelaxDeps = true;
  pythonImportsCheck = [ "lmcache" ];

  # Tests need a live GPU + multi-process redis fixture
  doCheck = false;

  meta = {
    description = "Knowledge-cache transport for vLLM (ROCm build)";
    homepage = "https://github.com/LMCache/LMCache";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
