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

    preConfigure = ''
      # The build inspects /etc/os-release to gate behaviour; provide a
      # NixOS-flavoured stub via NIX_REDIRECTS so we don't depend on a
      # writable /etc.
      mkdir -p $TMPDIR/etc
      echo 'ID=nixos' > $TMPDIR/etc/os-release
      export NIX_REDIRECTS=/etc/os-release=$TMPDIR/etc/os-release
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
      maintainers = [ ];
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
    maintainers = [ ];
    platforms = lib.platforms.linux;
  };
}
