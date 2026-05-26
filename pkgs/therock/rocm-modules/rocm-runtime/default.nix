{
  lib,
  stdenv,
  fetchFromGitHub,
  rocmUpdateScript,
  pkg-config,
  cmake,
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
    patchShebangs --build \
      runtime/hsa-runtime/core/runtime/trap_handler/create_trap_handler_header.sh \
      runtime/hsa-runtime/core/runtime/blit_shaders/create_blit_shader_header.sh \
      runtime/hsa-runtime/image/blit_src/create_hsaco_ascii_file.sh
    patchShebangs --host image core runtime

    substituteInPlace CMakeLists.txt \
      --replace 'hsa/include/hsa' 'include/hsa'

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
