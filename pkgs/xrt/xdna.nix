{
  lib,
  stdenv,
  symlinkJoin,
  cmake,
  pkg-config,
  git,
  python3,
  gawk,
  xrt,
  boost,
  libdrm,
  libuuid,
  libelf,
  ocl-icd,
  opencl-headers,
  ncurses,
  libxml2,
  yaml-cpp,
  openssl,
  rapidjson,
  protobuf,
  systemd,
  libsystemtap,
  writeText,
  src,
  version,
}:

let
  xrt-amdxdna-plugin = stdenv.mkDerivation {
    pname = "xrt-amdxdna-plugin";
    inherit version src;

    nativeBuildInputs = [
      cmake
      pkg-config
      git
      python3
      gawk
    ];

    buildInputs = [
      xrt
      boost
      libdrm
      libuuid
      libelf
      ocl-icd
      opencl-headers
      ncurses
      libxml2
      yaml-cpp
      openssl
      rapidjson
      protobuf
      systemd
      libsystemtap
    ];

    env.LDFLAGS = "-Wl,--copy-dt-needed-entries";

    postPatch = ''
      # The bundled XRT must use the same hermetic distro metadata as the
      # standalone package. NIX_REDIRECTS does not redirect ordinary awk I/O.
      substituteInPlace xrt/src/CMake/nativeLnx.cmake xrt/src/CMake/cpackLin.cmake \
        --replace-fail /etc/os-release ${writeText "xrt-os-release" ''
          ID=nixos
          VERSION_ID="${lib.trivial.release}"
        ''}
    '';

    cmakeFlags = [
      (lib.cmakeBool "SKIP_KMOD" true)
    ];

    preInstall = ''
      find . -name cmake_install.cmake -exec sed -i \
        -e 's|/bins/|'"$out"'/bins/|g' \
        {} \;
    '';

    postInstall = ''
      if [ -d "$out/bins$out" ]; then
        cp -rn "$out/bins$out"/* "$out/" || true
        rm -rf "$out/bins"
      fi
      if [ -d "$out$out" ]; then
        cp -rn "$out$out"/* "$out/" || true
        rm -rf "$out/nix"
      fi
      if [ -d "$out/opt/xilinx/xrt/lib" ]; then
        cp -r $out/opt/xilinx/xrt/lib/* $out/lib/ || true
      fi
    '';

    meta = {
      description = "AMD XDNA driver userspace plugin for XRT";
      homepage = "https://github.com/amd/xdna-driver";
      license = lib.licenses.asl20;
      maintainers = with lib.maintainers; [ georgewhewell ];
      platforms = lib.platforms.linux;
    };
  };
in
# Plugin first so its NPU-aware libs shadow base xrt's stubs.
symlinkJoin {
  name = "xrt-amdxdna-${version}";
  paths = [
    xrt-amdxdna-plugin
    xrt
  ];

  passthru = {
    plugin = xrt-amdxdna-plugin;
    inherit (xrt) version;
  };

  meta = {
    description = "Xilinx Runtime with AMD XDNA NPU support for Ryzen AI";
    homepage = "https://github.com/amd/xdna-driver";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = lib.platforms.linux;
  };
}
