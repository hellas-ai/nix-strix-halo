{
  lib,
  stdenv,
  gnumake,
  makeWrapper,
  coreutils,
  curl,
  gnused,
  src,
  version ? "unstable",
  darwinHostSdkRoot ? "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk",
  darwinHostDeploymentTarget ? "26.0",
  darwinHostClang ? "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang",
}:

let
  metalSources = {
    DS4_METAL_FLASH_ATTN_SOURCE = "flash_attn.metal";
    DS4_METAL_DENSE_SOURCE = "dense.metal";
    DS4_METAL_MOE_SOURCE = "moe.metal";
    DS4_METAL_DSV4_HC_SOURCE = "dsv4_hc.metal";
    DS4_METAL_UNARY_SOURCE = "unary.metal";
    DS4_METAL_DSV4_KV_SOURCE = "dsv4_kv.metal";
    DS4_METAL_DSV4_ROPE_SOURCE = "dsv4_rope.metal";
    DS4_METAL_DSV4_MISC_SOURCE = "dsv4_misc.metal";
    DS4_METAL_ARGSORT_SOURCE = "argsort.metal";
    DS4_METAL_CPY_SOURCE = "cpy.metal";
    DS4_METAL_CONCAT_SOURCE = "concat.metal";
    DS4_METAL_GET_ROWS_SOURCE = "get_rows.metal";
    DS4_METAL_SUM_ROWS_SOURCE = "sum_rows.metal";
    DS4_METAL_SOFTMAX_SOURCE = "softmax.metal";
    DS4_METAL_REPEAT_SOURCE = "repeat.metal";
    DS4_METAL_GLU_SOURCE = "glu.metal";
    DS4_METAL_NORM_SOURCE = "norm.metal";
    DS4_METAL_BIN_SOURCE = "bin.metal";
    DS4_METAL_SET_ROWS_SOURCE = "set_rows.metal";
  };

  metalSourceWrapperArgs = lib.concatStringsSep " " (
    lib.mapAttrsToList (
      envName: fileName: "--set ${envName} \"$out/share/ds4/metal/${fileName}\""
    ) metalSources
  );

  darwinSdkFlags = [
    "-isysroot"
    darwinHostSdkRoot
    "-mmacosx-version-min=${darwinHostDeploymentTarget}"
  ];

  commonCFlags = [
    "-O3"
    "-ffast-math"
    "-g"
    "-Wall"
    "-Wextra"
  ]
  ++ lib.optionals stdenv.hostPlatform.isDarwin darwinSdkFlags;
in
stdenv.mkDerivation {
  pname = "ds4";
  inherit src version;

  __noChroot = stdenv.hostPlatform.isDarwin;

  nativeBuildInputs = [
    gnumake
    makeWrapper
  ];

  dontConfigure = true;

  preBuild = lib.optionalString stdenv.hostPlatform.isDarwin ''
    if [ ! -d "${darwinHostSdkRoot}" ]; then
      echo "ds4: missing macOS SDK: ${darwinHostSdkRoot}" >&2
      echo "ds4: install Xcode with the macOS 26 SDK, or override darwinHostSdkRoot" >&2
      exit 1
    fi
    if [ ! -x "${darwinHostClang}" ]; then
      echo "ds4: missing Xcode clang: ${darwinHostClang}" >&2
      echo "ds4: install Xcode, or override darwinHostClang" >&2
      exit 1
    fi
  '';

  buildPhase = ''
    runHook preBuild

    make -j"$NIX_BUILD_CORES" \
      CC="${if stdenv.hostPlatform.isDarwin then darwinHostClang else "$CC"}" \
      NATIVE_CPU_FLAG= \
      CFLAGS="${lib.escapeShellArgs (commonCFlags ++ [ "-std=c99" ])}" \
      OBJCFLAGS="${lib.escapeShellArgs (commonCFlags ++ [ "-fobjc-arc" ])}"

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    for bin in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
      install -Dm755 "$bin" "$out/bin/$bin-unwrapped"
      makeWrapper "$out/bin/$bin-unwrapped" "$out/bin/$bin" \
        ${metalSourceWrapperArgs}
    done

    mkdir -p "$out/share/ds4"
    cp -R metal speed-bench "$out/share/ds4/"
    install -Dm644 README.md "$out/share/ds4/README.md"
    install -Dm644 LICENSE "$out/share/ds4/LICENSE"
    install -Dm644 MODEL_CARD.md "$out/share/ds4/MODEL_CARD.md"

    install -Dm755 download_model.sh "$out/bin/ds4-download-model"
    substituteInPlace "$out/bin/ds4-download-model" \
      --replace-fail 'ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)' 'ROOT=''${DS4_ROOT:-$PWD}'
    patchShebangs "$out/bin/ds4-download-model"
    wrapProgram "$out/bin/ds4-download-model" \
      --prefix PATH : "${
        lib.makeBinPath [
          coreutils
          curl
          gnused
        ]
      }"

    runHook postInstall
  '';

  meta = with lib; {
    description = "DwarfStar 4 native inference engine for DeepSeek V4 Flash";
    homepage = "https://github.com/antirez/ds4";
    license = licenses.mit;
    platforms = platforms.darwin;
  };
}
