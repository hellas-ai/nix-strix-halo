# MSCCL++ v0.9.0 for gfx1151 with the IB transport, for the small-message all-reduce investigation (AA-next/action3).
#
# Two changes against upstream (patches/0001):
#   - gfx1151 as the HIP target;
#   - on ROCm every "GPU" allocation (GpuBuffer, semaphores, scratch) comes from hipHostMalloc. These nodes are UMA, the memory is directly
#     accessible by the GPU, and ibv_reg_mr accepts it, whereas ibv_reg_mr on hipMalloc memory fails ("Bad address": no peer-memory module,
#     and upstream has no dma-buf path on HIP).
{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  pkg-config,
  patchelf,
  numactl,
  rdma-core,
  nlohmann_json,
  rocmSdk,
}:
stdenv.mkDerivation {
  pname = "mscclpp-gfx1151";
  version = "0.9.0";
  src = fetchFromGitHub {
    owner = "microsoft";
    repo = "mscclpp";
    rev = "v0.9.0";
    hash = "sha256-2a0iSKPDLsl44LZYwCYy4X3tKZ1XicS7KMBNukfRFLk=";
  };
  patches = [ ./0001-rocm-host-memory-for-ib-gfx1151.patch ];
  nativeBuildInputs = [
    cmake
    pkg-config
    patchelf
  ];
  buildInputs = [
    rocmSdk
    numactl
    rdma-core
  ];
  cmakeFlags = [
    "-DMSCCLPP_BYPASS_GPU_CHECK=ON"
    "-DMSCCLPP_USE_ROCM=ON"
    "-DMSCCLPP_USE_IB=ON"
    "-DMSCCLPP_BUILD_TESTS=OFF"
    "-DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF"
    "-DCMAKE_CXX_COMPILER=${rocmSdk}/bin/therock-hip-clang++"
    "-DCMAKE_PREFIX_PATH=${rocmSdk}"
    "-DROCM_PATH=${rocmSdk}"
    "-DFETCHCONTENT_SOURCE_DIR_JSON=${nlohmann_json.src}"
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
  ];
  env.ROCM_PATH = rocmSdk;
  # The ROCm compiler is used directly (not through the nixpkgs wrapper), so the dependencies are not on the RUNPATH; the loader needs them for the
  # dlopen of libmscclpp from Python (no LD_LIBRARY_PATH in the serving unit).
  postFixup = ''
    for so in $out/lib/libmscclpp*.so*; do
      [ -L "$so" ] && continue
      patchelf --add-rpath ${lib.makeLibraryPath [ numactl rdma-core rocmSdk ]} "$so"
    done
  '';
  meta.description = "MSCCL++ with host-memory IB registration on Strix Halo (gfx1151)";
}
