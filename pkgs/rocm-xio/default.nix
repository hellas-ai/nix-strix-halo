{
  lib,
  stdenv,
  fetchurl,
  cmake,
  ninja,
  pkg-config,
  makeWrapper,
  writeText,
  rocmSdk,
  rdma-core,
  libdrm,
  numactl,
  cli11,
  curl,
  zlib,
  ncurses,
  ocl-icd,
  src,
  version ? "0.1.0",
  offloadArch ? "gfx1151",
}:

let
  hsakmtShim = writeText "rocm-xio-therock-hsakmt-shim.cmake" ''
    set(numa_FOUND TRUE)
    if(NOT TARGET numa::numa)
      add_library(numa::numa UNKNOWN IMPORTED)
      set_target_properties(numa::numa PROPERTIES
        IMPORTED_LOCATION "${lib.getLib numactl}/lib/libnuma.so"
        INTERFACE_INCLUDE_DIRECTORIES "${lib.getDev numactl}/include")
    endif()

    function(_rocm_xio_fix_therock_hsakmt)
      if(TARGET hsakmt::hsakmt)
        get_target_property(_libs hsakmt::hsakmt INTERFACE_LINK_LIBRARIES)
        if(_libs)
          list(FILTER _libs EXCLUDE REGEX "^/usr/lib64/libc\\.so$")
          set_target_properties(hsakmt::hsakmt PROPERTIES
            INTERFACE_LINK_LIBRARIES "''${_libs}")
        endif()
      endif()
    endfunction()

    cmake_language(DEFER CALL _rocm_xio_fix_therock_hsakmt)
  '';

  runtimeLibraryPath = lib.makeLibraryPath [
    rocmSdk
    rdma-core
    libdrm
    numactl
    stdenv.cc.cc.lib
    zlib
    ncurses
    ocl-icd
  ];

  nvmeHeader = fetchurl {
    url = "https://raw.githubusercontent.com/torvalds/linux/v6.18/include/linux/nvme.h";
    hash = "sha256-Ctc8aR4gxnD+yL8Yj5rk+S+ofFFimzGZ3DElXNrHDZc=";
  };

  nvmeIoctlHeader = fetchurl {
    url = "https://raw.githubusercontent.com/torvalds/linux/v6.18/include/uapi/linux/nvme_ioctl.h";
    hash = "sha256-0es90HcF+2QgvXGFDOlXcQNwYLqvqK1C5zBFYdaBwLk=";
  };
in
stdenv.mkDerivation {
  pname = "rocm-xio-${offloadArch}";
  inherit src version;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    makeWrapper
  ];

  buildInputs = [
    rocmSdk
    rdma-core
    libdrm
    numactl
    cli11
    curl
  ];

  postPatch = ''
    patchShebangs .
    cat > scripts/build/fetch-nvme-headers.sh <<'EOF'
    #!${stdenv.shell}
    set -euo pipefail

    output_dir="$1"
    mkdir -p "$output_dir"
    cp ${nvmeHeader} "$output_dir/linux-nvme.h"
    cp ${nvmeIoctlHeader} "$output_dir/linux-nvme_ioctl.h"
    EOF
    chmod 755 scripts/build/fetch-nvme-headers.sh
  '';

  ROCM_PATH = rocmSdk;
  ROCM_HOME = rocmSdk;
  HIP_PATH = rocmSdk;
  HIP_PLATFORM = "amd";

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_COMPILER=${rocmSdk}/bin/therock-hip-clang++"
    "-DCMAKE_HIP_COMPILER=${rocmSdk}/bin/therock-hip-clang++"
    "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${rocmSdk}"
    "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${hsakmtShim}"
    "-DCMAKE_PREFIX_PATH=${rocmSdk};${cli11};${rdma-core};${libdrm};${numactl}"
    "-DROCM_PATH=${rocmSdk}"
    "-DOFFLOAD_ARCH=${offloadArch}"
    "-DGDA_BNXT=OFF"
    "-DGDA_MLX5=OFF"
    "-DGDA_IONIC=OFF"
    "-DGDA_ERNIC=OFF"
    "-DGDA_USB4_RDMA=ON"
    "-DBUILD_CLIENTS=ON"
    "-DINSTALL_TESTER=ON"
    "-DBUILD_TESTING=OFF"
    "-DRDMA_CORE_LIB_DIR=${lib.getLib rdma-core}/lib"
    "-DCMAKE_HIP_FLAGS=-I${cli11}/include"
    "-DCMAKE_EXE_LINKER_FLAGS=-L${lib.getLib libdrm}/lib -Wl,-rpath,${lib.getLib libdrm}/lib"
  ];

  postInstall = ''
    wrapProgram "$out/bin/xio-tester" \
      --set ROCM_PATH "${rocmSdk}" \
      --set ROCM_HOME "${rocmSdk}" \
      --set HIP_PATH "${rocmSdk}" \
      --set HIP_PLATFORM amd \
      --set HSA_OVERRIDE_GFX_VERSION "11.5.1" \
      --prefix PATH : "${rocmSdk}/bin:${rocmSdk}/llvm/bin" \
      --prefix LD_LIBRARY_PATH : "${runtimeLibraryPath}:${rocmSdk}/lib64:${rocmSdk}/llvm/lib"
  '';

  meta = {
    description = "ROCm XIO GPU-initiated IO library and xio-tester for ${offloadArch}";
    homepage = "https://github.com/ROCm/rocm-xio";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
