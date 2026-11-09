{
  lib,
  stdenv,
  kernel,
  amduprof,
}:

stdenv.mkDerivation {
  pname = "amduprof-power-profiler";
  version = "10.6.3-${kernel.modDirVersion}";

  inherit (amduprof) src sourceRoot;

  nativeBuildInputs = kernel.moduleBuildDependencies;

  dontConfigure = true;

  postUnpack = ''
    tar -xzf "$sourceRoot/bin/AMDPowerProfilerDriverSource.tar.gz" -C "$sourceRoot"
  '';

  preBuild = ''
    cd AMDPowerProfiler-10.6.3

    # The upstream feature probes only look below /usr/src. Probe the selected
    # Nix kernel instead, and let Kbuild provide the rest of the build flags.
    kernelSource="${kernel.dev}/lib/modules/${kernel.modDirVersion}/build/source"
    if grep -q "vm_flags_set" "$kernelSource/include/linux/mm.h"; then
      extraFlags="-DVM_FLAGS_SET"
    else
      extraFlags=""
    fi
    if grep -q "hrtimer_setup" "$kernelSource/include/linux/hrtimer.h"; then
      extraFlags="$extraFlags -DHRTIMER_SETUP"
    fi
    substituteInPlace Makefile \
      --replace-fail 'ccflags-y := -I$(PWD)/inc -mpopcnt -DKERNEL_MODULE' \
        "ccflags-y := -I\$(PWD)/inc -mpopcnt -DKERNEL_MODULE $extraFlags"
  '';

  buildPhase = ''
    runHook preBuild
    make -C "${kernel.dev}/lib/modules/${kernel.modDirVersion}/build" \
      M="$PWD" \
      modules
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -D -m 0644 AMDPowerProfiler.ko \
      "$out/lib/modules/${kernel.modDirVersion}/misc/AMDPowerProfiler.ko"
    runHook postInstall
  '';

  meta = {
    description = "AMD uProf power-profiling kernel module";
    homepage = "https://www.amd.com/en/developer/uprof.html";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    maintainers = with lib.maintainers; [ georgewhewell ];
  };
}
