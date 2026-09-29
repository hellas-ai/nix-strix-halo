{
  lib,
  callPackage,
  cctools,
  darwin,
  mlxPackage,
  mlxMetalPackage,
  python312,
  pyproject-build-systems,
  pyproject-nix,
  runCommand,
  uv2nix,
}:

let
  pname = "vllm-metal";
  version = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project.version;
  lock = builtins.fromTOML (builtins.readFile ./uv.lock);
  lockedVersion = name: (lib.findFirst (package: package.name == name) null lock.package).version;

  workspace = uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ./.;
  };

  # Keep uv2nix's dependency metadata while replacing the binary wheels with
  # the source-built MLX outputs.  pyproject-nix follows the `dependencies`
  # passthru attribute recursively when constructing the virtual environment,
  # so inserting an ordinary Nixpkgs Python package directly would discard the
  # lock-file dependency graph.
  useSourcePackage =
    uvPackage: sourcePackage:
    uvPackage.overrideAttrs (_: {
      phases = [ "installPhase" ];
      installPhase = ''
        mkdir -p "$out"
        cp -a ${sourcePackage}/. "$out/"
      '';
    });

  pythonSet = (callPackage pyproject-nix.build.packages { python = python312; }).overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.wheel
      # Keep the paired upstream Darwin wheels together to preserve their
      # tested Python/native ABI and avoid duplicating vLLM's native release
      # build.  MLX and MLX-Metal are still replaced below with JACCL-enabled
      # builds from pinned source.
      (workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      })
      (_: prev: {
        mlx = useSourcePackage prev.mlx mlxPackage;
        mlx-metal = useSourcePackage prev.mlx-metal mlxMetalPackage;

        vllm-metal = prev.vllm-metal.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            cctools
            darwin.sigtool
          ];
          postFixup = (old.postFixup or "") + ''
            for extension in "$out/${python312.sitePackages}/vllm_metal/metal/"_paged_ops*.so; do
              # The release wheel records its CI runner's install name and an
              # unqualified MLX dependency. Bind it to our exact source build.
              # A sibling symlink keeps load commands short enough for the
              # wheel's Mach-O header, which has no room for store paths.
              ln -s ${mlxMetalPackage}/${python312.sitePackages}/mlx/lib/libmlx.dylib \
                "$(dirname "$extension")/libmlx.dylib"
              install_name_tool -id "@rpath/$(basename "$extension")" \
                -change @rpath/libmlx.dylib @loader_path/libmlx.dylib \
                "$extension"
              codesign -f -s - "$extension"
            done
          '';
        });

        # vllm and mlx-vlm depend on the headless and GUI OpenCV wheels,
        # respectively. Keep the headless cv2 implementation and only retain
        # the GUI wheel's distribution metadata for dependency resolution.
        opencv-python = prev.opencv-python.overrideAttrs (old: {
          postInstall = (old.postInstall or "") + ''
            rm -r "$out/${python312.sitePackages}/cv2"
          '';
        });
      })
    ]
  );
in
assert lib.assertMsg
  (mlxPackage.version == lockedVersion "mlx" && mlxMetalPackage.version == lockedVersion "mlx-metal")
  "vllm-metal: update the MLX source and paired release wheels together; their native ABIs must match";
assert lib.assertMsg (
  lockedVersion "vllm-metal" == version && lib.removeSuffix "+cpu" (lockedVersion "vllm") == version
) "vllm-metal: the project version and both locked release wheels must match";
(pythonSet.mkVirtualEnv "${pname}-${version}" workspace.deps.default).overrideAttrs (
  final: old: {
    passthru = (old.passthru or { }) // {
      tests = (old.passthru.tests or { }) // {
        imports = runCommand "${pname}-imports" { nativeBuildInputs = [ final.finalPackage ]; } ''
          mkdir "$out"
          export HOME="$TMPDIR/home"
          mkdir -p "$HOME"
          export HF_HUB_OFFLINE=1
          python -c 'import mlx.core, mlx_lm, ray, vllm, vllm_metal' > "$out/imports"
          vllm --help > "$out/vllm-help"
        '';
        metal =
          runCommand "${pname}-metal-smoke"
            {
              nativeBuildInputs = [ final.finalPackage ];
              requiredSystemFeatures = [ "metal" ];
              __noChroot = true;
            }
            ''
              export HOME="$TMPDIR/home"
              mkdir -p "$HOME"
              export HF_HUB_OFFLINE=1
              export VLLM_METAL_BUILD_FROM_SOURCE=0
              python ${./smoke.py} > "$out"
            '';
        server =
          runCommand "${pname}-serve-smoke"
            {
              nativeBuildInputs = [ final.finalPackage ];
              requiredSystemFeatures = [ "metal" ];
              __noChroot = true;
            }
            ''
              export HOME="$TMPDIR/home"
              mkdir -p "$HOME"
              export HF_HUB_OFFLINE=1
              export VLLM_METAL_BUILD_FROM_SOURCE=0
              python ${./serve-smoke.py} > "$out"
            '';
      };
    };

    meta = (old.meta or { }) // {
      description = "vLLM hardware plugin for Apple Silicon";
      homepage = "https://github.com/vllm-project/vllm-metal";
      changelog = "https://github.com/vllm-project/vllm-metal/releases/tag/v${version}";
      license = lib.licenses.asl20;
      maintainers = with lib.maintainers; [ georgewhewell ];
      mainProgram = "vllm";
      platforms = [ "aarch64-darwin" ];
      sourceProvenance = with lib.sourceTypes; [
        fromSource
        binaryNativeCode
      ];
    };
  }
)
