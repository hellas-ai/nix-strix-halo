{
  lib,
  stdenv,
  gnumake,
  makeWrapper,
  coreutils,
  gnugrep,
  gawk,
  iproute2,
  procps,
  rocmSdk,
  src,
  version ? "experimental",
  offloadArch ? "gfx1151",
}:

let
  runtimePath = lib.makeBinPath [
    coreutils
    gnugrep
    gawk
    iproute2
    procps
    rocmSdk
  ];

  runtimeLibraryPath = lib.makeLibraryPath [
    rocmSdk
    stdenv.cc.cc.lib
  ];

  rocmEnvironment = [
    "--set"
    "HIP_PLATFORM"
    "amd"
    "--set"
    "HSA_OVERRIDE_GFX_VERSION"
    "11.5.1"
    "--set"
    "ROCM_HOME"
    "${rocmSdk}"
    "--set"
    "ROCM_PATH"
    "${rocmSdk}"
    "--set"
    "HIP_PATH"
    "${rocmSdk}"
    "--set"
    "DEVICE_LIB_PATH"
    "${rocmSdk}/lib/llvm/amdgcn/bitcode"
    "--set"
    "HIP_DEVICE_LIB_PATH"
    "${rocmSdk}/lib/llvm/amdgcn/bitcode"
    "--prefix"
    "PATH"
    ":"
    runtimePath
    "--prefix"
    "LD_LIBRARY_PATH"
    ":"
    runtimeLibraryPath
  ];
in
stdenv.mkDerivation {
  pname = "ds4-rocm-${offloadArch}";
  inherit src version;

  nativeBuildInputs = [
    gnumake
    makeWrapper
  ];

  dontConfigure = true;

  buildPhase = ''
    runHook preBuild

    make rocm-upstream -j"$NIX_BUILD_CORES" \
      CC="$CC" \
      ROCM_PATH="${rocmSdk}" \
      ROCM_HIPCC="${rocmSdk}/bin/therock-hip-clang++" \
      ROCM_ARCH="${offloadArch}" \
      CFLAGS="-O3 -ffast-math -D_GNU_SOURCE -Wall -Wextra -std=c99" \
      ROCM_CFLAGS="-O3 -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -I. -Wno-unused-command-line-argument -x hip --offload-arch=${offloadArch}" \
      ROCM_LDLIBS="-lm -pthread -L${rocmSdk}/lib -Wl,-rpath,${rocmSdk}/lib -lhipblas -lhipblaslt -lamdhip64"

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    install -Dm755 ds4-rocm-upstream "$out/bin/ds4-rocm-upstream"
    install -Dm755 ds4-server-rocm-upstream "$out/bin/ds4-server-rocm-upstream"
    install -Dm755 ds4-bench-rocm-upstream "$out/bin/ds4-bench-rocm-upstream"

    ln -s ds4-rocm-upstream "$out/bin/ds4"
    ln -s ds4-server-rocm-upstream "$out/bin/ds4-server"
    ln -s ds4-bench-rocm-upstream "$out/bin/ds4-bench"

    mkdir -p "$out/share/ds4"
    cp -R scripts speed-bench README.md LICENSE "$out/share/ds4/"
    chmod -R u+w "$out/share/ds4/scripts"
    patchShebangs "$out/share/ds4/scripts"

    substituteInPlace "$out/share/ds4/scripts/run_ds4_bench_rocm_upstream.sh" \
      --replace-fail 'make ds4-bench-rocm-upstream -j"$(nproc)"' ':' \
      --replace-fail './ds4-bench-rocm-upstream' 'ds4-bench-rocm-upstream'

    substituteInPlace "$out/share/ds4/scripts/start_ds4_cli_rocm_upstream.sh" \
      --replace-fail 'make ds4-rocm-upstream -j"$(nproc)"' ':' \
      --replace-fail './ds4-rocm-upstream' 'ds4-rocm-upstream'

    substituteInPlace "$out/share/ds4/scripts/start_ds4_server.sh" \
      --replace-fail 'make ds4-server -j"$(nproc)"' ':' \
      --replace-fail '  ./ds4-server' '  ds4-server-rocm-upstream'

    makeWrapper "$out/share/ds4/scripts/run_ds4_bench_rocm_upstream.sh" "$out/bin/ds4-bench-fast-full" \
      --set DS4_SERVER_FAST_FULL 1 \
      ${lib.escapeShellArgs rocmEnvironment}

    makeWrapper "$out/share/ds4/scripts/start_ds4_cli_rocm_upstream.sh" "$out/bin/ds4-cli-fast-full" \
      ${lib.escapeShellArgs rocmEnvironment}

    makeWrapper "$out/share/ds4/scripts/start_ds4_server.sh" "$out/bin/ds4-server-fast-full" \
      --set DS4_SERVER_FAST_FULL 1 \
      ${lib.escapeShellArgs rocmEnvironment}

    for bin in ds4-rocm-upstream ds4-server-rocm-upstream ds4-bench-rocm-upstream; do
      wrapProgram "$out/bin/$bin" ${lib.escapeShellArgs rocmEnvironment}
    done

    runHook postInstall
  '';

  meta = {
    description = "Experimental ds4 ROCm/HIP build for AMD Strix Halo";
    homepage = "https://github.com/ejpir/ds4-hip/tree/rocm-upstream-shape-cyberneurova";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
