{
  lib,
  stdenv,
  fetchFromGitHub,
  rocmUpdateScript,
  cmake,
  rocm-cmake,
  rocm-smi,
  rocm-core,
  pkg-config,
  clr,
  mscclpp,
  perl,
  hipify,
  python3,
  fmt,
  gtest,
  chrpath,
  roctracer,
  rocprofiler,
  rocprofiler-register,
  rdma-core,
  autoPatchelfHook,
  buildTests ? false,
  gpuTargets ? (clr.localGpuTargets or [ ]),
  # for passthru.tests
  rccl,
}:

let
  useAsan = buildTests;
  useUbsan = buildTests;
  san = lib.optionalString (useAsan || useUbsan) (
    "-fno-gpu-sanitize -fsanitize=undefined "
    + (lib.optionalString useAsan "-fsanitize=address -shared-libsan ")
  );
in
# Note: we can't properly test or make use of multi-node collective ops
# https://github.com/NixOS/nixpkgs/issues/366242 tracks kernel support
# kfd_peerdirect support which is on out-of-tree amdkfd in ROCm/ROCK-Kernel-Driver
# infiniband ib_peer_mem support isn't in the mainline kernel but is carried by some distros
stdenv.mkDerivation (finalAttrs: {
  pname = "rccl${clr.gpuArchSuffix}";
  version = "7.2.2";

  outputs = [
    "out"
  ]
  ++ lib.optionals buildTests [
    "test"
  ];

  patches = [
    ./rccl-test-missing-iomanip.diff
  ];

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "rccl";
    rev = "rocm-${finalAttrs.version}";
    hash = "sha256-A1IQYIDWqu3JLiPQ70G52s1/0ZweQxFlgMUH81qJWmU=";
  };

  requiredSystemFeatures = [ "big-parallel" ]; # Very resource intensive LTO

  nativeBuildInputs = [
    cmake
    rocm-cmake
    clr
    perl
    hipify
    python3
    pkg-config
    autoPatchelfHook # ASAN doesn't add rpath without this
  ];

  buildInputs = [
    rocm-smi
    fmt
    gtest
    roctracer
    rocprofiler
    rocprofiler-register
    mscclpp
    # libibverbs so RCCL builds the IB transport; without it, the
    # `ibv_*` symbol set is absent from librccl.so and NCCL falls back
    # to TCP sockets for all inter-rank traffic. Needed to compare
    # br0.lan vs usb4_rdma0 transports in 2-node FSDP runs.
    rdma-core
  ]
  ++ lib.optionals buildTests [
    chrpath
  ];

  cmakeFlags = [
    "-DHIP_CLANG_NUM_PARALLEL_JOBS=4"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DROCM_PATH=${clr}"
    "-DHIP_COMPILER=${clr}/bin/amdclang++"
    "-DCMAKE_CXX_COMPILER=${clr}/bin/amdclang++"
    "-DROCM_PATCH_VERSION=${rocm-core.ROCM_LIBPATCH_VERSION}"
    "-DROCM_VERSION=${rocm-core.ROCM_LIBPATCH_VERSION}"
    "-DBUILD_BFD=OFF" # Can't get it to detect bfd.h
    "-DENABLE_MSCCL_KERNEL=ON"
    # FIXME: this is still running a download because if(NOT mscclpp_nccl_FOUND) is commented out T_T
    "-DENABLE_MSCCLPP=OFF"
    #"-DMSCCLPP_ROOT=${mscclpp}"
    # Manually define CMAKE_INSTALL_<DIR>
    # See: https://github.com/NixOS/nixpkgs/pull/197838
    "-DCMAKE_INSTALL_BINDIR=bin"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_INSTALL_INCLUDEDIR=include"
  ]
  ++ lib.optionals (gpuTargets != [ ]) [
    # AMD can't make up their minds and keep changing which one is used in different projects.
    "-DAMDGPU_TARGETS=${lib.concatStringsSep ";" gpuTargets}"
    "-DGPU_TARGETS=${lib.concatStringsSep ";" gpuTargets}"
  ]
  ++ lib.optionals buildTests [
    "-DBUILD_TESTS=ON"
  ];

  # -O2 and -fno-strict-aliasing due to UB issues in RCCL :c
  # Reported upstream
  env.CFLAGS = "-I${clr}/include -I${roctracer}/include -O2 -fno-strict-aliasing ${san}-fno-omit-frame-pointer -momit-leaf-frame-pointer";
  env.CXXFLAGS = "-I${clr}/include -I${roctracer}/include -O2 -fno-strict-aliasing ${san}-fno-omit-frame-pointer -momit-leaf-frame-pointer";
  env.LDFLAGS = "${san}";
  # 7.13's CMakeLists.txt sets HOST_OS_ID via /etc/os-release but never
  # uses ${HOST_OS_ID} as a literal, so the old ubuntu substitution no
  # longer matches. The ROCM_SMI_INCLUDE_DIR -> _DIRS substitution is
  # also stale (the line is gone). Just patch shebangs.
  postPatch = ''
    patchShebangs src tools
  '';

  postInstall =
    # 7.13 / NCCL 2.28 introduced a device-side API exposed via
    # <nccl_device.h>. The upstream RCCL install target only ships rccl.h
    # and nccl_net.h, but pytorch 2.11+ #includes nccl_device.h from its
    # symm_mem c10d backend whenever USE_NCCL is on (it does that for
    # rocmSupport via useSystemNccl). Copy the device headers in so
    # downstream consumers can compile.
    ''
      # cwd at postInstall is the cmake build dir, not the source root,
      # so resolve the source headers via $NIX_BUILD_TOP/$sourceRoot.
      _src_inc="$NIX_BUILD_TOP/$sourceRoot/src/include"
      install -Dm644 "$_src_inc/nccl_device.h" $out/include/rccl/nccl_device.h
      install -Dm644 "$_src_inc/nccl_device.h" $out/include/nccl_device.h
      cp -a "$_src_inc/nccl_device" $out/include/
      cp -a "$_src_inc/nccl_device" $out/include/rccl/
      # nccl_device/impl/{comm,core}__types.h have
      # `#include "../{comm,core}_tmp.h"`, which upstream synthesizes by
      # `cp {comm,core}.h {comm,core}_tmp.h` in tools/topo_expl/Makefile.
      # Mirror those here so downstream consumers don't trip on the
      # dangling includes.
      install -Dm644 "$_src_inc/nccl_device/comm.h" $out/include/nccl_device/comm_tmp.h
      install -Dm644 "$_src_inc/nccl_device/comm.h" $out/include/rccl/nccl_device/comm_tmp.h
      install -Dm644 "$_src_inc/nccl_device/core.h" $out/include/nccl_device/core_tmp.h
      install -Dm644 "$_src_inc/nccl_device/core.h" $out/include/rccl/nccl_device/core_tmp.h
      # nccl_device/core.h has `#include <nccl.h>`, which upstream
      # configure_file's at build time to both rccl/rccl.h (installed)
      # and PROJECT_BINARY_DIR/include/nccl.h (not installed). Mirror
      # the latter so downstream code that follows the device-header
      # chain can resolve <nccl.h>.
      install -Dm644 $out/include/rccl/rccl.h $out/include/nccl.h
    ''
    + lib.optionalString useAsan ''
      patchelf --add-needed ${clr}/llvm/lib/linux/libclang_rt.asan-${stdenv.hostPlatform.parsed.cpu.name}.so $out/lib/librccl.so
    ''
    + lib.optionalString buildTests ''
      mkdir -p $test/bin
      mv $out/bin/* $test/bin
      rmdir $out/bin
    '';

  passthru.updateScript = rocmUpdateScript { inherit finalAttrs; };

  # This package with sanitizers + manual integration test binaries built
  # must be ran manually
  passthru.tests.rccl = rccl.override {
    buildTests = true;
  };

  meta = {
    description = "ROCm communication collectives library";
    homepage = "https://github.com/ROCm/rccl";
    license = with lib.licenses; [
      bsd2
      bsd3
    ];
    teams = [ lib.teams.rocm ];
    platforms = lib.platforms.linux;
  };
})
