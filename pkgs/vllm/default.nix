args@{
  lib,
  stdenv,
  symlinkJoin,
  vllm,
  vllmSources,
  vllmVersion,
  cudaPackages ? { },
  ...
}:

let
  upstreamArgs = builtins.removeAttrs args [
    "lib"
    "stdenv"
    "symlinkJoin"
    "vllm"
    "vllmSources"
    "vllmVersion"
  ];

  flashmla = stdenv.mkDerivation {
    pname = "flashmla";
    version = "1.0.0";
    src = vllmSources.flashmla;

    dontConfigure = true;

    buildPhase = ''
      rm -rf csrc/cutlass
      ln -sf ${vllmSources.flashmla-cutlass} csrc/cutlass
    '';

    installPhase = ''
      cp -rva . $out
    '';
  };

  vllm-flash-attn = stdenv.mkDerivation {
    pname = "vllm-flash-attn";
    version = "2.7.2.post1";
    src = vllmSources.flash-attn;

    dontConfigure = true;

    buildPhase = ''
      rm -rf csrc/cutlass
      ln -sf ${vllmSources.cutlass} csrc/cutlass
    '';

    installPhase = ''
      cp -rva . $out
    '';
  };

  replaceCmakeFeatures =
    names: flags:
    let
      isReplaced =
        flag: lib.any (name: lib.hasPrefix "-D${name}" flag || lib.hasPrefix "-D${name}:" flag) names;
    in
    lib.filter (flag: !isReplaced flag) flags;
in
(vllm.override (
  upstreamArgs
  // {
    inherit vllm-flash-attn;
  }
)).overridePythonAttrs
  (
    old:
    let
      oldCmakeFlags = old.cmakeFlags or [ ];
      oldCmakeArgs = old.CMAKE_ARGS or (old.env or { }).CMAKE_ARGS or "";
      hasCudaCmake = lib.any (flag: lib.hasPrefix "-DFETCHCONTENT_SOURCE_DIR_CUTLASS" flag) oldCmakeFlags;
      removedUpstreamPatchNames = [
        "0002-setup.py-nix-support-respect-cmakeFlags.patch"
        "0003-propagate-pythonpath.patch"
        "0005-drop-intel-reqs.patch"
        "0006-drop-rocm-extra-reqs.patch"
        "0007-drop-quack-reqs.patch"
      ];

      cmakeFlags =
        replaceCmakeFeatures [
          "FETCHCONTENT_SOURCE_DIR_CUTLASS"
          "FLASH_MLA_SRC_DIR"
          "VLLM_FLASH_ATTN_SRC_DIR"
          "CUDA_TOOLKIT_ROOT_DIR"
        ] oldCmakeFlags
        ++ lib.optionals hasCudaCmake [
          (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_CUTLASS" "${lib.getDev vllmSources.cutlass}")
          (lib.cmakeFeature "FLASH_MLA_SRC_DIR" "${lib.getDev flashmla}")
          (lib.cmakeFeature "VLLM_FLASH_ATTN_SRC_DIR" "${lib.getDev vllm-flash-attn}")
          (lib.cmakeFeature "CUDA_TOOLKIT_ROOT_DIR" "${symlinkJoin {
            name = "cuda-merged-${cudaPackages.cudaMajorMinorVersion}";
            paths =
              builtins.concatMap
                (pkg: [
                  (lib.getBin pkg)
                  (lib.getLib pkg)
                  (lib.getDev pkg)
                ])
                (
                  with cudaPackages;
                  [
                    cuda_cudart
                    cuda_cccl
                    libcurand
                    libcusparse
                    libcusolver
                    cuda_nvtx
                    cuda_nvrtc
                    libcublas
                  ]
                );
          }}")
        ];
    in
    {
      version = vllmVersion;
      src = vllmSources.vllm;

      patches = lib.filter (
        patch: !(lib.elem (builtins.baseNameOf (toString patch)) removedUpstreamPatchNames)
      ) (old.patches or [ ]);

      pythonRemoveDeps = (old.pythonRemoveDeps or [ ]) ++ [
        "conch-triton-kernels"
        "intel-openmp"
        "nvidia-cutlass-dsl"
        "pytest-asyncio"
        "quack-kernels"
        "runai-model-streamer"
        "setuptools"
        "setuptools-scm"
        "tensorizer"
      ];

      inherit cmakeFlags;

      env = (old.env or { }) // {
        CMAKE_BUILD_TYPE = "Release";
        CMAKE_ARGS = lib.concatStringsSep " " (
          lib.filter (arg: arg != "") ([ oldCmakeArgs ] ++ cmakeFlags)
        );
      };

      makeWrapperArgs = (old.makeWrapperArgs or [ ]) ++ [
        "--prefix"
        "PYTHONPATH"
        ":"
        "$program_PYTHONPATH"
      ];

      passthru = (old.passthru or { }) // {
        inherit vllm-flash-attn;
        upstream = vllm;
      };

      meta = (old.meta or { }) // {
        changelog = "https://github.com/vllm-project/vllm/releases/tag/v${vllmVersion}";
      };
    }
  )
