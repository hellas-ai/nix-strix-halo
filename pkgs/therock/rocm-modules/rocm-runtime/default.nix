{
  lib,
  stdenv,
  fetchFromGitHub,
  rocmUpdateScript,
  pkg-config,
  cmake,
  python3,
  xxd,
  rocm-device-libs,
  rocprofiler-register,
  elfutils,
  libdrm,
  numactl,
  llvm,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "rocm-runtime";
  version = "7.2.2";

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "rocm-systems";
    rev = "rocm-${finalAttrs.version}";
    sparseCheckout = [
      "projects/rocr-runtime"
      "shared"
    ];
    hash = "sha256-hcyjOLMtoBX/p6r6R9Bl9635DuvI6rTn1KziHMeyYM0=";
  };
  sourceRoot = "${finalAttrs.src.name}/projects/rocr-runtime";

  cmakeBuildType = "RelWithDebInfo";
  separateDebugInfo = true;
  __structuredAttrs = true;
  strictDeps = true;

  nativeBuildInputs = [
    pkg-config
    cmake
    # 7.15 replaced create_trap_handler_header.sh with a Python generator
    # (trap_handler/CMakeLists.txt does find_package(Python3 REQUIRED)).
    python3
    xxd # used by create_hsaco_ascii_file.sh
    llvm.rocm-toolchain
  ];

  buildInputs = [
    llvm.clang-unwrapped
    llvm.llvm
    elfutils
    libdrm
    numactl
    rocprofiler-register
  ];

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_INSTALL_BINDIR=bin"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_INSTALL_INCLUDEDIR=include"
  ];

  patches = [
    # Circular-dep workaround: aqlprofile relies on hsa-runtime64 which is
    # part of rocm-runtime; rocprofiler loads aqlprofile directly instead.
    ./remove-hsa-aqlprofile-dep.patch
  ];

  postPatch = ''
    # 7.15 replaced the trap_handler/blit_shaders header generators with
    # Python scripts (invoked via Python3_EXECUTABLE); only the blit_src
    # generator still has a shell variant.
    patchShebangs --build \
      runtime/hsa-runtime/image/blit_src/create_hsaco_ascii_file.sh
    patchShebangs --host runtime

    substituteInPlace runtime/hsa-runtime/image/blit_src/CMakeLists.txt \
      --replace-fail 'COMMAND clang' "COMMAND ${llvm.rocm-toolchain}/bin/clang"

    export HIP_DEVICE_LIB_PATH="${rocm-device-libs}/amdgcn/bitcode"
  '';

  passthru.updateScript = rocmUpdateScript { inherit finalAttrs; };

  meta = {
    description = "Platform runtime for ROCm";
    homepage = "https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime";
    license = with lib.licenses; [ ncsa ];
    maintainers = with lib.maintainers; [ lovesegfault ];
    teams = [ lib.teams.rocm ];
    platforms = lib.platforms.linux;
  };
})
