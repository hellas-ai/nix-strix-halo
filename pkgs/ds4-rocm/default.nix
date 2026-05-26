{
  lib,
  stdenv,
  gnumake,
  makeWrapper,
  coreutils,
  curl,
  gnugrep,
  gnused,
  gawk,
  iproute2,
  procps,
  rocmSdk,
  src,
  version ? "experimental",
  packageSuffix ? offloadArch,
  offloadArch ? "gfx1151",
  hsaOverrideGfxVersion ? null,
}:

let
  runtimePath = lib.makeBinPath [
    coreutils
    curl
    gnugrep
    gnused
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
  ]
  ++ lib.optionals (hsaOverrideGfxVersion != null) [
    "--set"
    "HSA_OVERRIDE_GFX_VERSION"
    hsaOverrideGfxVersion
  ]
  ++ [
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
  pname = "ds4-rocm-${packageSuffix}";
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

    for bin in \
      ds4-rocm-upstream \
      ds4-server-rocm-upstream \
      ds4-bench-rocm-upstream \
      ds4-eval-rocm-upstream \
      ds4-agent-rocm-upstream; do
      install -Dm755 "$bin" "$out/bin/$bin"
    done

    ln -s ds4-rocm-upstream "$out/bin/ds4"
    ln -s ds4-server-rocm-upstream "$out/bin/ds4-server"
    ln -s ds4-bench-rocm-upstream "$out/bin/ds4-bench"
    ln -s ds4-eval-rocm-upstream "$out/bin/ds4-eval"
    ln -s ds4-agent-rocm-upstream "$out/bin/ds4-agent"

    mkdir -p "$out/share/ds4"
    cp -R scripts speed-bench "$out/share/ds4/"
    for file in README.md LICENSE MODEL_CARD.md AGENT.md gfx1151.md; do
      if [ -e "$file" ]; then
        install -Dm644 "$file" "$out/share/ds4/$file"
      fi
    done
    patchShebangs "$out/share/ds4/scripts"

    substituteInPlace "$out/share/ds4/scripts/run_ds4_bench_rocm_upstream.sh" \
      --replace-fail 'make ds4-bench-rocm-upstream -j"$(nproc)"' ':' \
      --replace-fail './ds4-bench-rocm-upstream' 'ds4-bench-rocm-upstream'

    substituteInPlace "$out/share/ds4/scripts/run_ds4_eval_rocm_upstream.sh" \
      --replace-fail 'make ds4-eval-rocm-upstream -j"$(nproc)"' ':' \
      --replace-fail './ds4-eval-rocm-upstream' 'ds4-eval-rocm-upstream'

    substituteInPlace "$out/share/ds4/scripts/start_ds4_agent_rocm_upstream.sh" \
      --replace-fail 'make ds4-agent-rocm-upstream -j"$(nproc)"' ':' \
      --replace-fail './ds4-agent-rocm-upstream' 'ds4-agent-rocm-upstream'

    substituteInPlace "$out/share/ds4/scripts/start_ds4_cli_rocm_upstream.sh" \
      --replace-fail 'make ds4-rocm-upstream -j"$(nproc)"' ':' \
      --replace-fail './ds4-rocm-upstream' 'ds4-rocm-upstream'

    substituteInPlace "$out/share/ds4/scripts/start_ds4_server.sh" \
      --replace-fail 'make "$SERVER_MAKE_TARGET" -j"$(nproc)"' ':' \
      --replace-fail 'SERVER_BIN="./ds4-server-rocm-upstream"' 'SERVER_BIN="ds4-server-rocm-upstream"'

    makeWrapper "$out/share/ds4/scripts/run_ds4_bench_rocm_upstream.sh" "$out/bin/ds4-bench-fast-full" \
      --set DS4_SERVER_FAST_FULL 1 \
      --prefix PATH : "$out/bin" \
      ${lib.escapeShellArgs rocmEnvironment}

    makeWrapper "$out/share/ds4/scripts/run_ds4_eval_rocm_upstream.sh" "$out/bin/ds4-eval-fast-full" \
      --prefix PATH : "$out/bin" \
      ${lib.escapeShellArgs rocmEnvironment}

    makeWrapper "$out/share/ds4/scripts/start_ds4_agent_rocm_upstream.sh" "$out/bin/ds4-agent-fast-full" \
      --prefix PATH : "$out/bin" \
      ${lib.escapeShellArgs rocmEnvironment}

    makeWrapper "$out/share/ds4/scripts/start_ds4_cli_rocm_upstream.sh" "$out/bin/ds4-cli-fast-full" \
      --prefix PATH : "$out/bin" \
      ${lib.escapeShellArgs rocmEnvironment}

    makeWrapper "$out/share/ds4/scripts/start_ds4_server.sh" "$out/bin/ds4-server-fast-full" \
      --set DS4_SERVER_FAST_FULL 1 \
      --prefix PATH : "$out/bin" \
      ${lib.escapeShellArgs rocmEnvironment}

    install -Dm755 download_model.sh "$out/bin/ds4-download-model"
    substituteInPlace "$out/bin/ds4-download-model" \
      --replace-fail 'ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)' 'ROOT=''${DS4_ROOT:-$PWD}'
    patchShebangs "$out/bin/ds4-download-model"
    wrapProgram "$out/bin/ds4-download-model" \
      --prefix PATH : "${runtimePath}"

    for bin in \
      ds4-rocm-upstream \
      ds4-server-rocm-upstream \
      ds4-bench-rocm-upstream \
      ds4-eval-rocm-upstream \
      ds4-agent-rocm-upstream; do
      wrapProgram "$out/bin/$bin" ${lib.escapeShellArgs rocmEnvironment}
    done

    runHook postInstall
  '';

  meta = {
    description = "Experimental DwarfStar 4 ROCm/HIP build";
    homepage = "https://github.com/ejpir/ds4-hip/tree/rocm-upstream-shape-cyberneurova";
    license = lib.licenses.mit;
    mainProgram = "ds4";
    platforms = [ "x86_64-linux" ];
  };
}
