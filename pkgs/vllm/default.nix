args@{
  lib,
  stdenv,
  symlinkJoin,
  vllm,
  vllmSources,
  vllmVersion,
  cudaPackages ? { },
  amdsmi ? null,
  grpcio-reflection ? null,
  ijson ? null,
  rocmPackages ? { },
  gpuTargets ? [ ],
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
    "amdsmi"
    "grpcio-reflection"
    "ijson"
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
      oldEnv = old.env or { };
      oldCmakeFlags = old.cmakeFlags or [ ];
      oldCmakeArgs = if old ? CMAKE_ARGS then old.CMAKE_ARGS else oldEnv.CMAKE_ARGS or "";
      isRocmBuild = (oldEnv.VLLM_TARGET_DEVICE or "") == "rocm";
      rocmPath =
        if oldEnv ? ROCM_PATH then
          oldEnv.ROCM_PATH
        else if old ? ROCM_PATH then
          old.ROCM_PATH
        else if oldEnv ? ROCM_HOME then
          oldEnv.ROCM_HOME
        else if old ? ROCM_HOME then
          old.ROCM_HOME
        else
          null;
      rocmGpuTargetString =
        if gpuTargets != [ ] then
          lib.concatStringsSep ";" gpuTargets
        else if oldEnv ? PYTORCH_ROCM_ARCH then
          oldEnv.PYTORCH_ROCM_ARCH
        else if old ? PYTORCH_ROCM_ARCH then
          old.PYTORCH_ROCM_ARCH
        else
          null;
      rocmGpuTargetList =
        if rocmGpuTargetString == null then [ ] else lib.splitString ";" rocmGpuTargetString;
      isRocmGfx1151Build = isRocmBuild && lib.elem "gfx1151" rocmGpuTargetList;
      rocmExtraIncludeFlags = lib.concatMapStringsSep " " (pkg: "-I${lib.getInclude pkg}/include") [
        rocmPackages.rocthrust
        rocmPackages.rocprim
        rocmPackages.hipcub
      ];
      requiredRuntimeDeps = lib.filter (dep: dep != null) [
        amdsmi
        grpcio-reflection
        ijson
      ];
      appendMissingRuntimeDeps =
        deps:
        let
          depNames = map (dep: dep.pname or dep.name or "") deps;
        in
        deps ++ lib.filter (dep: !(lib.elem (dep.pname or dep.name or "") depNames)) requiredRuntimeDeps;
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
          "CMAKE_HIP_ARCHITECTURES"
          "GPU_TARGETS"
        ] oldCmakeFlags
        ++ lib.optionals (isRocmBuild && rocmGpuTargetString != null) [
          (lib.cmakeFeature "CMAKE_HIP_ARCHITECTURES" rocmGpuTargetString)
          (lib.cmakeFeature "GPU_TARGETS" rocmGpuTargetString)
        ]
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

      postPatch = ''
        rm vllm/third_party/pynvml.py
        substituteInPlace tests/utils.py \
          --replace-fail \
            "from vllm.third_party.pynvml import" \
            "from pynvml import"
        substituteInPlace vllm/utils/import_utils.py \
          --replace-fail \
            "import vllm.third_party.pynvml as pynvml" \
            "import pynvml"

        substituteInPlace pyproject.toml \
          --replace-fail '"torch == 2.11.0"' '"torch >= 2.9.0"' \
          --replace-fail "setuptools>=77.0.3,<81.0.0" "setuptools"
      ''
      + lib.optionalString isRocmGfx1151Build ''
        substituteInPlace csrc/sampler.cu \
          --replace-fail \
            "constexpr int kNumThreadsPerBlockMerge = 1024;" \
            "constexpr int kNumThreadsPerBlockMerge = 512;"
      '';

      pythonRelaxDeps =
        let
          oldPythonRelaxDeps = old.pythonRelaxDeps or [ ];
        in
        if lib.isBool oldPythonRelaxDeps then oldPythonRelaxDeps else oldPythonRelaxDeps ++ [ "torch" ];

      pythonRemoveDeps = (old.pythonRemoveDeps or [ ]) ++ [
        "amd-quark"
        "conch-triton-kernels"
        "intel-openmp"
        "mcp"
        "nvidia-cutlass-dsl"
        "opentelemetry-semantic-conventions-ai"
        "pytest-asyncio"
        "quack-kernels"
        "runai-model-streamer"
        "setuptools"
        "setuptools-scm"
        "tensorizer"
      ];

      dependencies = appendMissingRuntimeDeps (old.dependencies or [ ]);
      propagatedBuildInputs = appendMissingRuntimeDeps (old.propagatedBuildInputs or [ ]);

      inherit cmakeFlags;

      buildInputs =
        (old.buildInputs or [ ])
        ++ lib.optionals isRocmBuild (
          with rocmPackages;
          [
            rocm-comgr
            rocrand
            hiprand
            rocblas
            miopen-hip
            hipfft
            hipcub
            hipsolver
            rocsolver
            hipblaslt
            rocm-runtime
          ]
        );

      env =
        oldEnv
        // {
          CMAKE_BUILD_TYPE = "Release";
          CMAKE_ARGS = lib.concatStringsSep " " (
            lib.filter (arg: arg != "") ([ oldCmakeArgs ] ++ cmakeFlags)
          );
        }
        // lib.optionalAttrs (vllmSources ? triton-kernels) {
          TRITON_KERNELS_SRC_DIR = "${vllmSources.triton-kernels}/python/triton_kernels/triton_kernels";
        }
        // lib.optionalAttrs (rocmPath != null) {
          ROCM_PATH = rocmPath;
        }
        // lib.optionalAttrs (rocmGpuTargetString != null) {
          PYTORCH_ROCM_ARCH = rocmGpuTargetString;
        }
        // lib.optionalAttrs isRocmBuild {
          HIPFLAGS = lib.concatStringsSep " " (
            lib.filter (arg: arg != "") [
              (oldEnv.HIPFLAGS or "")
              rocmExtraIncludeFlags
            ]
          );
          CXXFLAGS = lib.concatStringsSep " " (
            lib.filter (arg: arg != "") [
              (oldEnv.CXXFLAGS or "")
              rocmExtraIncludeFlags
            ]
          );
        };

      preBuild = (old.preBuild or "") + ''
        export MAX_JOBS="$NIX_BUILD_CORES"
      '';

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
