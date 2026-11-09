{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  makeWrapper,
  binutils,
  coreutils,
  gawk,
  gnugrep,
  netcat,
  util-linux,
  dbus,
  elfutils,
  fontconfig,
  freetype,
  glib,
  libGL,
  libGLU,
  libarchive,
  libcap,
  libdrm,
  libglvnd,
  libpciaccess,
  libunwind,
  libxkbcommon,
  libice,
  libsm,
  libx11,
  libxau,
  libxdmcp,
  libxext,
  libxi,
  libxmu,
  libxrender,
  libxcb,
  numactl,
  ncurses5,
  xcbutil,
  xcbutilimage,
  xcbutilkeysyms,
  xcbutilrenderutil,
  xcbutilwm,
  zlib,
  zstd,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "amduprof";
  version = "5.3.521";

  src = fetchurl {
    url = "https://download.amd.com/developer/eula/uprof/uprof-5-3/AMDuProf_Linux_x64_${finalAttrs.version}.tar.bz2";
    hash = "sha256-E0ZZxtk5S+MjYyrUBhacG9U2I7J+WPXrcvcjXaqmwPk=";
    # AMD redirects requests without a referrer back to the product page.
    curlOptsList = [
      "--referer"
      "https://www.amd.com/en/developer/uprof/uprof-eula/uprof-5-3-eula.html?filename=AMDuProf_Linux_x64_${finalAttrs.version}.tar.bz2"
    ];
  };

  sourceRoot = "AMDuProf_Linux_x64_${finalAttrs.version}";

  nativeBuildInputs = [
    autoPatchelfHook
    makeWrapper
  ];

  buildInputs = [
    dbus
    elfutils
    fontconfig
    freetype
    glib
    libGL
    libGLU
    libarchive
    libcap
    libdrm
    libglvnd
    libpciaccess
    libunwind
    libxkbcommon
    numactl
    ncurses5
    stdenv.cc.cc.lib
    xcbutil
    xcbutilimage
    xcbutilkeysyms
    xcbutilrenderutil
    xcbutilwm
    libice
    libsm
    libx11
    libxau
    libxdmcp
    libxext
    libxi
    libxmu
    libxrender
    libxcb
    zlib
    zstd
  ];

  dontBuild = true;
  dontConfigure = true;
  dontStrip = true;

  # The legacy GPU agent is loaded only for supported ROCm/Instinct targets;
  # keep it available without making CPU profiling depend on a ROCm release.
  autoPatchelfIgnoreMissingDeps = [
    "libhsa-runtime64.so.1"
    "librocprofiler64.so.1"
    "libroctracer64.so.1"
  ];

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/opt/amduprof" "$out/bin" "$out/share/man/man1"
    cp -a . "$out/opt/amduprof/"

    for command in \
      AMDCpuTopology \
      AMDProfilerService \
      AMDReportGenerator \
      AMDuProf \
      AMDuProfCLI \
      AMDuProfPcm
    do
      ln -s "$out/opt/amduprof/bin/$command" "$out/bin/$command"
    done
    ln -s "$out/opt/amduprof/bin/AMDPerf/AMDuProfSys" "$out/bin/AMDuProfSys"
    ln -s "$out/opt/amduprof/bin/AMDSystemCheck/AMDSystemCheck" "$out/bin/AMDSystemCheck"
    ln -s "$out/opt/amduprof/bin/Help/man1/"*.1.gz "$out/share/man/man1/"

    runHook postInstall
  '';

  postFixup = ''
    patchShebangs "$out/opt/amduprof/bin"

    # Avoid printing the executable path on every invocation, and preserve
    # arguments without routing them through an extra shell evaluation.
    for command in AMDuProf AMDProfilerService AMDuProfCLI; do
      substituteInPlace "$out/opt/amduprof/bin/$command" \
        --replace-warn "grep '^/'" "grep -q '^/'" \
        --replace-warn 'eval "$AMDuProfBinaryExePath $@"' 'exec "$AMDuProfBinaryExePath" "$@"'
    done

    for command in AMDuProf AMDProfilerService AMDuProfCLI; do
      wrapProgram "$out/opt/amduprof/bin/$command" \
        --prefix PATH : ${
          lib.makeBinPath [
            binutils
            coreutils
            gawk
            gnugrep
            netcat
            util-linux
          ]
        }
    done
  '';

  meta = {
    description = "AMD performance analysis suite for Zen processors";
    homepage = "https://www.amd.com/en/developer/uprof.html";
    license = lib.licenses.unfree;
    platforms = [ "x86_64-linux" ];
    maintainers = with lib.maintainers; [ georgewhewell ];
    mainProgram = "AMDuProfCLI";
  };
})
