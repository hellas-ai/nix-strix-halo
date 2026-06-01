{
  lib,
  stdenv,
  gnumake,
  makeWrapper,
  coreutils,
  curl,
  gnused,
  apple-sdk_26,
  src,
  version ? "unstable",
  darwinSdk ? apple-sdk_26,
  darwinSdkRoot ? darwinSdk.sdkroot,
  darwinDeploymentTarget ? "26.0",
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

  optionalDarwinSdkRoot = lib.optionalString (darwinSdkRoot != null) ''
    sdk_root=${lib.escapeShellArg darwinSdkRoot}
  '';

  optionalDarwinDeploymentTarget = lib.optionalString (darwinDeploymentTarget != null) ''
    deployment_target=${lib.escapeShellArg darwinDeploymentTarget}
  '';
in
stdenv.mkDerivation {
  pname = "ds4";
  inherit src version;

  nativeBuildInputs = [
    gnumake
    makeWrapper
  ];

  buildInputs = lib.optionals stdenv.hostPlatform.isDarwin [
    darwinSdk
  ];

  dontConfigure = true;

  preBuild = lib.optionalString stdenv.hostPlatform.isDarwin ''
    ${optionalDarwinSdkRoot}
    ${optionalDarwinDeploymentTarget}

    if [ -z "''${sdk_root:-}" ]; then
      sdk_root="''${SDKROOT:-}"
    fi

    if [ ! -d "$sdk_root" ]; then
      echo "ds4: missing macOS SDK: $sdk_root" >&2
      echo "ds4: override darwinSdk or darwinSdkRoot with a newer macOS SDK" >&2
      exit 1
    fi

    echo "ds4: using macOS SDK: $sdk_root"

    export DS4_DARWIN_SDK_ROOT="$sdk_root"
    export SDKROOT="$sdk_root"
    if [ -n "''${deployment_target:-}" ]; then
      export MACOSX_DEPLOYMENT_TARGET="$deployment_target"
    fi
  '';

  buildPhase = ''
    runHook preBuild

    darwin_sdk_flags=()
    if [ -n "''${DS4_DARWIN_SDK_ROOT:-}" ]; then
      darwin_sdk_flags+=("-isysroot" "$DS4_DARWIN_SDK_ROOT")
    fi
    if [ -n "''${MACOSX_DEPLOYMENT_TARGET:-}" ]; then
      darwin_sdk_flags+=("-mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET")
    fi

    make -j"$NIX_BUILD_CORES" \
      CC="$CC" \
      NATIVE_CPU_FLAG= \
      CFLAGS="-O3 -ffast-math -g -Wall -Wextra ''${darwin_sdk_flags[*]} -std=c99" \
      OBJCFLAGS="-O3 -ffast-math -g -Wall -Wextra ''${darwin_sdk_flags[*]} -fobjc-arc"

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

  meta = {
    description = "DwarfStar 4 native inference engine for DeepSeek V4 Flash";
    homepage = "https://github.com/antirez/ds4";
    license = lib.licenses.mit;
    mainProgram = "ds4";
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = lib.platforms.darwin;
  };
}
