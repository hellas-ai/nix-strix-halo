{
  lib,
  stdenv,
  cmake,
  ninja,
  zlib,
  zstd,
  llvm,
  python3,
}:

let
  llvmNativeTarget =
    if stdenv.hostPlatform.isx86_64 then
      "X86"
    else if stdenv.hostPlatform.isAarch64 then
      "AArch64"
    else
      throw "Unsupported ROCm LLVM platform";
in
stdenv.mkDerivation (finalAttrs: {
  pname = "rocm-device-libs";
  # In-tree with ROCm LLVM
  inherit (llvm.llvm) version;
  src = llvm.llvm.monorepoSrc;
  sourceRoot = "${finalAttrs.src.name}/amd/device-libs";
  strictDeps = true;
  __structuredAttrs = true;

  postPatch =
    # Use our sysrooted toolchain instead of direct clang target
    ''
      substituteInPlace cmake/OCL.cmake \
        --replace-fail '$<TARGET_FILE:clang>' "${llvm.rocm-toolchain}/bin/clang"
    ''
    # nixpkgs sets CMAKE_INSTALL_LIBDIR to the absolute $out/lib path, so
    # PACKAGE_PREFIX (= ${CMAKE_INSTALL_LIBDIR}/cmake/AMDDeviceLibs) is
    # also absolute. The upstream cmake then runs the build-tree
    # configure_file directly into $out at build time, populating
    # AMDDeviceLibsConfig.cmake with build-time absolute bitcode paths.
    # The install rule that should overwrite this with the install-tree
    # variant then sees an "up-to-date" target and skips, leaving the
    # bogus build paths in the installed config. Force the build-tree
    # variant into the build directory instead so the install-tree
    # variant actually wins.
    + ''
      substituteInPlace cmake/Packages.cmake \
        --replace-fail \
          '  ''${PACKAGE_PREFIX}/AMDDeviceLibsConfig.cmake' \
          '  ''${CMAKE_CURRENT_BINARY_DIR}/AMDDeviceLibsConfig.cmake.buildtree'
    '';

  patches = [ ];

  nativeBuildInputs = [
    cmake
    ninja
    python3
    llvm.rocm-toolchain
  ];

  buildInputs = [
    llvm.llvm
    llvm.clang-unwrapped
    zlib
    zstd
  ];

  cmakeFlags = [
    "-DLLVM_TARGETS_TO_BUILD=AMDGPU;${llvmNativeTarget}"
    # The install-tree config derives AMD_DEVICE_LIBS_PREFIX by popping N
    # parent dirs off CMAKE_CURRENT_LIST_FILE, where N is the number of
    # "/"-separated components in PACKAGE_PREFIX. nixpkgs's default
    # CMAKE_INSTALL_LIBDIR is the absolute $out/lib path, which yields N=7
    # (way too many — it pops past `/`). Force the relative form here so
    # PACKAGE_PREFIX becomes lib/cmake/AMDDeviceLibs (N=3, the correct
    # number of levels back to $out).
    "-DCMAKE_INSTALL_LIBDIR=lib"
  ];

  meta = {
    description = "Set of AMD-specific device-side language runtime libraries";
    homepage = "https://github.com/ROCm/ROCm-Device-Libs";
    license = lib.licenses.ncsa;
    maintainers = with lib.maintainers; [ lovesegfault ];
    teams = [ lib.teams.rocm ];
    platforms = lib.platforms.linux;
  };
})
