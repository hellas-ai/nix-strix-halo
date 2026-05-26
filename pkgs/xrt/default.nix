{
  lib,
  stdenv,
  callPackage,
  cmake,
  pkg-config,
  git,
  curl,
  boost,
  libdrm,
  systemd,
  ocl-icd,
  opencl-headers,
  libuuid,
  libxml2,
  ncurses,
  yaml-cpp,
  openssl,
  rapidjson,
  protobuf,
  libsystemtap,
  src,
  version,
  xdnaSrc,
  xdnaVersion ? version,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "xrt";
  inherit version src;

  __structuredAttrs = true;
  strictDeps = true;

  nativeBuildInputs = [
    cmake
    pkg-config
    git
  ];

  buildInputs = [
    boost
    curl
    libdrm
    systemd
    ocl-icd
    opencl-headers
    libuuid
    libxml2
    ncurses
    yaml-cpp
    openssl
    rapidjson
    protobuf
    libsystemtap
  ];

  cmakeDir = "../src";

  env.LDFLAGS = "-Wl,--copy-dt-needed-entries";

  postPatch = ''
    # aiebu's asm/dump tools statically link glibc, which nixpkgs does not
    # provide. The XRT/NPU host runtime does not need those binaries; FLM
    # only needs libxrt_coreutil/libxrt_driver_xdna at runtime.
    substituteInPlace src/runtime_src/core/common/aiebu/src/cpp/utils/CMakeLists.txt \
      --replace-fail "add_subdirectory(asm)" "" \
      --replace-fail "add_subdirectory(dump)" ""
    substituteInPlace src/runtime_src/core/common/aiebu/CMakeLists.txt \
      --replace-fail "add_subdirectory(test)" ""
  '';

  cmakeFlags = [
    (lib.cmakeBool "XRT_NATIVE_BUILD" true)
    (lib.cmakeBool "XRT_SKIP_SUBMODULE_UPDATE" true)
  ];

  preBuild = ''
    # XRT's CMake bakes /usr/src into kernel-module install paths even when
    # we skip the kernel module build; rewrite to $out so packaging succeeds.
    find . -name cmake_install.cmake -exec sed -i 's|/usr/src|'"$out"'/src|g' {} \; || true
  '';

  preInstall = ''
    find . -name cmake_install.cmake -exec sed -i \
      -e 's|/usr/src|'"$out"'/src|g' \
      -e 's|/usr/local/bin|'"$out"'/bin|g' \
      -e 's|/usr/local/lib|'"$out"'/lib|g' \
      -e 's|/usr/local|'"$out"'|g' \
      -e 's|/usr/lib|'"$out"'/lib|g' \
      -e 's|/usr/bin|'"$out"'/bin|g' \
      -e 's|/etc/OpenCL|'"$out"'/etc/OpenCL|g' \
      -e 's|/etc/|'"$out"'/etc/|g' \
      {} \;
  '';

  postInstall = ''
    # Generated .pc files end up with `//nix/...` after the sed pass above;
    # collapse the double slash.
    find $out -name "*.pc" -exec sed -i 's|//nix|/nix|g' {} \;

    # When CMake's destination is absolute (the path substitutions above),
    # `make install` honours DESTDIR by nesting $out under $out. Flatten it.
    if [ -d "$out$out" ]; then
      cp -rn "$out$out"/* "$out/" || true
      rm -rf "$out/nix"
    fi
  '';

  passthru = {
    xdna = callPackage ./xdna.nix {
      xrt = finalAttrs.finalPackage;
      src = xdnaSrc;
      version = xdnaVersion;
    };
  };

  meta = {
    description = "Xilinx Runtime for FPGA/ACAP devices (and Ryzen AI NPUs via xrt.xdna)";
    homepage = "https://github.com/Xilinx/XRT";
    license = lib.licenses.asl20;
    maintainers = [ ];
    platforms = lib.platforms.linux;
  };
})
