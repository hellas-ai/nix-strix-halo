# Prometheus textfile exporter for the AMD XDNA NPU (amdxdna driver).
#
# The exporter reads the NPU via the DRM GET_INFO/GET_ARRAY ioctls, so it needs
# the amdxdna uapi header. Per the "depend on kernel uapi" decision it takes a
# linuxHeaders (default: the pinned nixpkgs one) rather than vendoring a copy;
# the NixOS module feeds the host kernel's headers so the 7.x-only queries
# (power mode, resource info, hardware-context occupancy) compile in. Features
# absent from the supplied header are detected here and simply left out.
{
  stdenv,
  linuxHeaders,
}:

stdenv.mkDerivation {
  pname = "amd-npu-exporter";
  version = "0.1.0";

  dontUnpack = true;
  strictDeps = true;

  buildPhase = ''
    runHook preBuild

    header=${linuxHeaders}/include/drm/amdxdna_accel.h
    if [ ! -e "$header" ]; then
      echo "amd-npu-exporter: ${linuxHeaders} has no amdxdna uapi header" >&2
      exit 1
    fi

    # Turn each 7.x-only query the header actually declares into a -DHAVE_*
    # flag so main.c compiles against 6.14+ headers and grows metrics on 7.x.
    flags=""
    grep -q DRM_AMDXDNA_GET_POWER_MODE      "$header" && flags="$flags -DHAVE_NPU_POWER_MODE"
    grep -q DRM_AMDXDNA_QUERY_RESOURCE_INFO "$header" && flags="$flags -DHAVE_NPU_RESOURCE_INFO"
    grep -q DRM_AMDXDNA_GET_ARRAY           "$header" && flags="$flags -DHAVE_NPU_GET_ARRAY"
    echo "amd-npu-exporter: capability flags:$flags"

    $CC -std=c11 -O2 -Wall -Wextra -Wpedantic \
      -I${linuxHeaders}/include $flags \
      ${./main.c} -o amd-npu-exporter

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -D -m 0755 amd-npu-exporter "$out/bin/amd-npu-exporter"
    runHook postInstall
  '';

  meta = {
    description = "Prometheus textfile exporter for the AMD XDNA NPU";
    mainProgram = "amd-npu-exporter";
  };
}
