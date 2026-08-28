{
  lib,
  callPackage,
  mlxPackage,
  mlxMetalPackage,
  mlx-lm-src,
  python312,
  pyproject-build-systems,
  pyproject-nix,
  runCommand,
  uv2nix,
}:

let
  pname = "vllm-metal";
  version = "0.3.0.dev20260826134128";

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

        mlx-lm = prev.mlx-lm.overrideAttrs (_: {
          # Deliberate 0.31.3 compatibility override; see the input comment.
          src = mlx-lm-src;
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
(pythonSet.mkVirtualEnv "${pname}-${version}" workspace.deps.default).overrideAttrs (
  final: old: {
    passthru = (old.passthru or { }) // {
      tests = (old.passthru.tests or { }) // {
        imports = runCommand "${pname}-imports" { nativeBuildInputs = [ final.finalPackage ]; } ''
          mkdir "$out"
          python -c 'import mlx.core, mlx_lm, ray, vllm, vllm_metal' > "$out/imports"
          vllm --help > "$out/vllm-help"
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
