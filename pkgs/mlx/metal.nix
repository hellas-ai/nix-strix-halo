{
  lib,
  stdenv,
  mlx,
  fetchFromGitHub,
  fetchzip,
  replaceVars,
  nanobind,
  python,
  zsh,
  apple-sdk_26,
  mlx-src,
  buildStage ? 1,
  pname ? if buildStage == 2 then "mlx-metal" else "mlx",
  backendPackage ? null,
  darwinSdk ? apple-sdk_26,
  darwinSdkRoot ? darwinSdk.sdkroot,
  darwinSdkVersion ? darwinSdk.version,
  darwinDeploymentTarget ? "26.0",
}:

assert lib.assertMsg stdenv.hostPlatform.isDarwin "MLX Metal is Darwin-only";
assert lib.assertMsg (
  buildStage == 1 || buildStage == 2
) "MLX Metal buildStage must be 1 (frontend) or 2 (backend)";
assert lib.assertMsg (
  buildStage != 1 || backendPackage != null
) "MLX Metal stage 1 requires backendPackage";

let
  gguf-tools = fetchFromGitHub {
    owner = "antirez";
    repo = "gguf-tools";
    rev = "8fa6eb65236618e28fd7710a0fba565f7faa1848";
    hash = "sha256-15FvyPOFqTOr5vdWQoPnZz+mYH919++EtghjozDlnSA=";
  };

  metal-cpp = fetchzip {
    url = "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip";
    hash = "sha256-7n2eI2lw/S+Us6l7YPAATKwcIbRRpaQ8VmES7S8ZjY8=";
  };

  inherit (python) sitePackages;
  backendLibDir = "${backendPackage}/${sitePackages}/mlx/lib";
in
mlx.overrideAttrs (old: {
  inherit pname;
  version = "0.32.0";

  src = mlx-src;
  patches = [
    ./patches/use-system-nanobind.patch
    ./patches/use-system-json.patch
    (replaceVars ./patches/darwin-sdk-version.patch {
      sdkVersion = darwinSdkVersion;
    })
  ];

  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ zsh ];
  buildInputs = (old.buildInputs or [ ]) ++ [ darwinSdk ];
  __noChroot = true;
  propagatedBuildInputs = lib.optionals (buildStage == 1) (
    (old.propagatedBuildInputs or [ ]) ++ [ backendPackage ]
  );

  postPatch = (old.postPatch or "") + ''
    substituteInPlace mlx/backend/cpu/jit_compiler.cpp \
      --replace-fail "${stdenv.cc}/bin/c++" "/usr/bin/c++"
  '';

  env = {
    PYPI_RELEASE = "1";
    MLX_BUILD_STAGE = toString buildStage;
    CMAKE_ARGS = lib.concatStringsSep " " [
      (lib.cmakeBool "MLX_BUILD_METAL" true)
      (lib.cmakeBool "USE_SYSTEM_FMT" true)
      (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_GGUFLIB" "${gguf-tools}")
      (lib.cmakeOptionType "filepath" "FETCHCONTENT_SOURCE_DIR_METAL_CPP" "${metal-cpp}")
      (lib.cmakeFeature "nanobind_DIR" "${nanobind}/${sitePackages}/nanobind/cmake")
      (lib.cmakeFeature "CMAKE_OSX_SYSROOT" darwinSdkRoot)
      (lib.cmakeFeature "CMAKE_OSX_DEPLOYMENT_TARGET" darwinDeploymentTarget)
    ];
  };

  preBuild = (old.preBuild or "") + ''
    if [ ! -d "${darwinSdkRoot}" ]; then
      echo "mlx-metal: missing macOS SDK: ${darwinSdkRoot}" >&2
      exit 1
    fi
    export SDKROOT=${lib.escapeShellArg darwinSdkRoot}
    export MACOSX_DEPLOYMENT_TARGET=${lib.escapeShellArg darwinDeploymentTarget}

    metal_tool="$(/usr/bin/xcrun -sdk macosx -find metal 2>/dev/null || true)"
    if [ -z "$metal_tool" ] || ! "$metal_tool" --version >/dev/null 2>&1; then
      metal_tool="$(find /private/var/run/com.apple.security.cryptexd/mnt \
        -path '*/Metal.xctoolchain/usr/bin/metal' -type f | head -n 1 || true)"
    fi
    if [ -z "$metal_tool" ] || ! "$metal_tool" --version >/dev/null 2>&1; then
      echo "mlx-metal: missing usable Apple Metal Toolchain" >&2
      echo "mlx-metal: run xcodebuild -downloadComponent MetalToolchain on the builder" >&2
      exit 1
    fi

    metallib_tool="$(dirname "$metal_tool")/metallib"
    if [ ! -x "$metallib_tool" ]; then
      echo "mlx-metal: missing metallib next to $metal_tool" >&2
      exit 1
    fi

    xcrun_wrapper="$TMPDIR/mlx-xcrun-wrapper"
    mkdir -p "$xcrun_wrapper"
    cat > "$xcrun_wrapper/xcrun" <<'SH'
    #!/usr/bin/env bash
    set -euo pipefail

    args=("$@")
    tool=""
    i=0
    while [ "$i" -lt "''${#args[@]}" ]; do
      case "''${args[$i]}" in
        -sdk|--sdk)
          i=$((i + 2))
          ;;
        *)
          tool="''${args[$i]}"
          i=$((i + 1))
          break
          ;;
      esac
    done

    case "$tool" in
      metal)
        exec "$MLX_METAL_TOOL" "''${args[@]:$i}"
        ;;
      metallib)
        exec "$MLX_METALLIB_TOOL" "''${args[@]:$i}"
        ;;
      *)
        exec /usr/bin/xcrun "$@"
        ;;
    esac
    SH
    chmod +x "$xcrun_wrapper/xcrun"

    export MLX_METAL_TOOL="$metal_tool"
    export MLX_METALLIB_TOOL="$metallib_tool"
    export PATH="$xcrun_wrapper:$PATH:/usr/bin:/bin"
  '';

  postInstall =
    (old.postInstall or "")
    + lib.optionalString (buildStage == 1) ''
      mkdir -p "$out/${sitePackages}/mlx/lib"
      for path in ${backendLibDir}/*; do
        ln -sf "$path" "$out/${sitePackages}/mlx/lib/$(basename "$path")"
      done
    ''
    + lib.optionalString (buildStage == 2) ''
      find "$out/${sitePackages}/mlx" -maxdepth 1 -type f \
        \( -name '*.py' -o -name '*.pyc' -o -name 'py.typed' \) -delete
      rm -rf "$out/${sitePackages}/mlx/__pycache__"

      cmake_targets="$out/${sitePackages}/mlx/share/cmake/MLX/MLXTargets.cmake"
      if [ -f "$cmake_targets" ]; then
        substituteInPlace "$cmake_targets" \
          --replace-fail "${darwinSdkRoot}/System/Library/Frameworks/Metal.framework" "-framework;Metal" \
          --replace-fail "${darwinSdkRoot}/System/Library/Frameworks/Foundation.framework" "-framework;Foundation" \
          --replace-fail "${darwinSdkRoot}/System/Library/Frameworks/QuartzCore.framework" "-framework;QuartzCore" \
          --replace-fail "${darwinSdkRoot}/System/Library/Frameworks/Accelerate.framework" "-framework;Accelerate"
      fi
    '';

  nativeCheckInputs = [ ];
  doCheck = false;
  doInstallCheck = false;
  dontCheckRuntimeDeps = true;
  dontUsePytestCheck = true;
  dontStrip = true;
  pythonImportsCheck = lib.optionals (buildStage == 1) [ "mlx.core" ];

  meta = with lib; {
    description =
      if buildStage == 2 then
        "Apple MLX Metal backend built from source"
      else
        "Apple MLX with Metal backend built from source";
    homepage = "https://github.com/ml-explore/mlx";
    license = licenses.mit;
    maintainers = with maintainers; [ georgewhewell ];
    platforms = [ "aarch64-darwin" ];
  };
})
