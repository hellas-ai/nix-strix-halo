# ROCWMMA derivation builder - builds from source against ROCm 7
{ pkgs, rocwmma }:

{ rocm, target }:

pkgs.stdenv.mkDerivation rec {
  pname = "rocwmma";
  version = "main";
  
  src = rocwmma;
  
  nativeBuildInputs = with pkgs; [
    cmake
    ninja
    lld
    llvmPackages.bintools
    autoPatchelfHook
  ];
  
  buildInputs = with pkgs; [
    stdenv.cc.cc.lib
  ];
  
  # Change FP8 check from FATAL_ERROR to STATUS (as done in amd-strix-halo-toolboxes)
  postPatch = ''
    substituteInPlace CMakeLists.txt \
      --replace 'message(FATAL_ERROR "The detected ROCm does not support data type' \
                'message(STATUS "The detected ROCm does not support data type'
  '';
  
  # Build ROCWMMA from source using nixpkgs clang (like llama.cpp)
  # This avoids linking issues with ROCm's clang in Nix environment
  cmakeFlags = [
    "-G Ninja"
    "-DCMAKE_C_COMPILER=${pkgs.clang}/bin/clang"
    "-DCMAKE_CXX_COMPILER=${pkgs.clang}/bin/clang++"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}"
    "-DCMAKE_CROSSCOMPILING=ON"
    "-DROCWMMA_BUILD_TESTS=OFF"
    "-DROCWMMA_BUILD_SAMPLES=OFF"
    "-DGPU_TARGETS=${target}"
    # Point to ROCm for HIP headers and libs
    "-DHIP_ROOT_DIR=${rocm}"
    "-DHIP_PATH=${rocm}"
    # Tell CMake where to find pthreads
    "-DCMAKE_THREAD_LIBS_INIT=-pthread"
    "-DCMAKE_HAVE_THREADS_LIBRARY=1"
    "-DCMAKE_USE_PTHREADS_INIT=1"
    "-DCMAKE_USE_PTHREADS=ON"
    "-DTHREADS_PREFER_PTHREAD_FLAG=ON"
    # Tell CMake where to find OpenMP (using ROCm's libomp)
    "-DOpenMP_C_INCLUDE_DIR=${rocm}/llvm/include"
    "-DOpenMP_CXX_INCLUDE_DIR=${rocm}/llvm/include"
    "-DOpenMP_CXX_FLAGS=-fopenmp"
    "-DOpenMP_CXX_LIB_NAMES=omp"
    "-DOpenMP_omp_LIBRARY=${rocm}/llvm/lib/libomp.so"
  ];
  
  preConfigure = ''
    export HIP_PATH="${rocm}"
    export ROCM_PATH="${rocm}"
    export HIP_PLATFORM="amd"
    export PATH="${rocm}/bin:$PATH"
    export HIP_CLANG_PATH="${pkgs.clang}/bin"
  '';
}