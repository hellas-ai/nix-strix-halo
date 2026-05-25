{
  lib,
  target,
  rocmTargets,
  therockRocmSources,
  therockPythonWheelSources,
  therockRocmSourceSources,
  therockRocmThirdPartySources,
}:
final: prev:
let
  targetBySuffix = lib.listToAttrs (
    map (rocmTarget: {
      name = rocmTarget.packageSuffix;
      value = rocmTarget;
    }) rocmTargets
  );

  targetFor =
    suffix:
    targetBySuffix.${suffix} or {
      packageSuffix = suffix;
      runtimeArch = suffix;
      buildTargets = [ suffix ];
      hsaOverride = null;
    };

  hsaOverrideFor =
    rocmTarget: if (rocmTarget.hsaOverride or null) != null then rocmTarget.hsaOverride else "11.5.1";

  sourceEntries = lib.mapAttrsToList (
    key: source: source // { inherit key; }
  ) therockRocmSourceSources;

  sourceMatchesTarget = suffix: source: (source.target or null) == suffix;

  sourceIsCompilerStage =
    source: (source.stage or null) == "compiler-stage" || lib.hasSuffix "-compiler-stage" source.key;

  sourceFor =
    suffix: predicate:
    let
      matches = builtins.filter (
        source: sourceMatchesTarget suffix source && predicate source
      ) sourceEntries;
    in
    if matches == [ ] then throw "missing TheRock source pin for ${suffix}" else builtins.head matches;

  sourceFullFor = suffix: sourceFor suffix (source: !sourceIsCompilerStage source);
  sourceCompilerFor = suffix: sourceFor suffix sourceIsCompilerStage;
  defaultSourceFull = sourceFullFor target.packageSuffix;

  pythonWheelSourcesByTarget =
    therockPythonWheelSources.targets or (
      if therockPythonWheelSources ? target then
        {
          ${therockPythonWheelSources.target} = therockPythonWheelSources;
        }
      else
        therockPythonWheelSources
    );

  pythonWheelSourcesFor =
    rocmTarget:
    let
      bySuffix = pythonWheelSourcesByTarget.${rocmTarget.packageSuffix} or null;
    in
    if bySuffix != null then bySuffix else pythonWheelSourcesByTarget.${rocmTarget.runtimeArch} or null;

  hasPythonWheelSources = rocmTarget: pythonWheelSourcesFor rocmTarget != null;

  normalizeRocmVersion =
    version: if builtins.length (lib.splitVersion version) == 2 then "${version}.0" else version;

  mkTorchRocm =
    rocmTarget:
    (final.python3Packages.torch.override {
      rocmSupport = true;
      cudaSupport = false;
      rocmPackages = final.therockRocmPackages.${rocmTarget.runtimeArch};
    }).overrideAttrs
      (old: {
        postPatch = (old.postPatch or "") + ''
          # Current RCCL reports NCCL_VERSION_CODE 22803 (2.28.3) but does
          # not ship the NCCL 2.28+ device-side symmetric memory APIs. Keep
          # PyTorch's NCCL symmetric-memory gates closed on ROCm.
          substituteInPlace torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp \
            --replace-fail \
              '#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)' \
              '#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0) && !defined(USE_ROCM)' \
            --replace-fail \
              '#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)' \
              '#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0) && !defined(USE_ROCM)'

          # intra_node_comm.cpp calls rsmi_init() under `#ifdef USE_ROCM`,
          # but PyTorch's dependency list does not include rocm_smi64.
          substituteInPlace cmake/Dependencies.cmake \
            --replace-fail \
              'hip::amdhip64 MIOpen hiprtc::hiprtc) # libroctx will be linked in with MIOpen' \
              'hip::amdhip64 MIOpen hiprtc::hiprtc rocm_smi64) # libroctx will be linked in with MIOpen'
        '';
      });

  torchRocmTargets = builtins.filter hasPythonWheelSources rocmTargets;

  torchRocmPerArch = lib.listToAttrs (
    map (rocmTarget: {
      name = "torch-rocm-${rocmTarget.packageSuffix}";
      value = mkTorchRocm rocmTarget;
    }) torchRocmTargets
  );

  mkTherockRocmSdkAttrs =
    suffix: source:
    let
      rocmTarget = targetFor suffix;
      sdk = prev.callPackage ./rocm-sdk {
        target = suffix;
        inherit (source) version url hash;
      };
      env = prev.callPackage ./rocm-env {
        therockRocmSdk = sdk;
        rdmaCore = final.rdma-core;
        packageSuffix = suffix;
        hsaOverrideGfxVersion = hsaOverrideFor rocmTarget;
      };
      rocshmemEnv = prev.callPackage ./rocm-rocshmem-env {
        therockRocmSdk = sdk;
        therockRocmEnv = env;
        packageSuffix = suffix;
      };
    in
    {
      "therock-rocm-${suffix}" = sdk;
      "therock-rocm-${suffix}-env" = env;
      "therock-rocm-${suffix}-rocshmem-env" = rocshmemEnv;
    };

  therockRocmSdkPerArch = lib.foldl' lib.recursiveUpdate { } (
    lib.mapAttrsToList mkTherockRocmSdkAttrs therockRocmSources.linux
  );

  mkTherockPythonStack =
    rocmTarget:
    let
      suffix = rocmTarget.packageSuffix;
      wheelSources = pythonWheelSourcesFor rocmTarget;
    in
    {
      "therock-python-${suffix}" = prev.callPackage ./python-env {
        inherit wheelSources;
      };
      "therock-python-wheels-${suffix}" = prev.callPackage ./python-wheels {
        inherit wheelSources;
      };
      "therock-amdsmi-${suffix}" = prev.python312Packages.callPackage ./amdsmi {
        wheels = final."therock-python-wheels-${suffix}";
      };
    };

  therockPythonTargets = builtins.filter hasPythonWheelSources rocmTargets;

  therockPythonPerArch = lib.foldl' lib.recursiveUpdate { } (
    map mkTherockPythonStack therockPythonTargets
  );

  therockFromSourceThirdPartyFetches = {
    esmiIbLibrarySource = prev.fetchgit {
      inherit (therockRocmThirdPartySources.esmiIbLibrary) url rev hash;
    };
    ireeLibbacktraceSource = prev.fetchzip {
      inherit
        (therockRocmThirdPartySources.archives."libbacktrace-b9e40069c0b47a722286b94eb5231f7f05c08713.zip")
        url
        ;
      hash = "sha512-vWfrdxHfNcfoh/SWh6dOTLRtqXcW0godukXv5ad6DqzfbwC7TOIbyw0Oke40oLRqFCLfg1M2f7bqnIrLnISmKw==";
      stripRoot = true;
    };
    ireeAmdDeviceLibsArchive = prev.fetchurl {
      url = "https://github.com/shark-infra/amdgpu-device-libs/releases/download/v20231101/amdgpu-device-libs-llvm-6086c272a3a59eb0b6b79dcbe00486bf4461856a.tgz";
      hash = "sha256-M2NiQWxo/di7gDKPZcp+uqDBGeoZyV323zDIMqTfObk=";
    };
    rocprofilerOtf2Archive = prev.fetchurl {
      url = "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/otf2-3.0.3.tar.gz";
      hash = "sha256-GKOQX3kXNAOH4+3I5XZvMasa9B9OzFZl2mx2nKIcTug=";
    };
    rocprofilerSysBinutilsArchive = prev.fetchurl {
      url = "https://ftpmirror.gnu.org/gnu/binutils/binutils-2.46.0.tar.gz";
      hash = "sha256-hgj+RKt95kX2rQqJgxO3UziEJJDWCa24XJ+ygnw3avI=";
    };
    spirvHeadersSource = prev.fetchzip {
      inherit (therockRocmThirdPartySources.spirvHeaders) url hash;
      stripRoot = true;
    };
    tracySource = prev.fetchFromGitHub {
      owner = "wolfpld";
      repo = "tracy";
      rev = "5479a42ef9346b64e6d1b860ae58aa8abdb0c7f6";
      hash = "sha256-4J8b+72k+xpeT6KsrkioF1xfWEBsGg2eLRg9iONxP/I=";
    };
  };

  mkTherockFromSourceStack =
    rocmTarget:
    let
      suffix = rocmTarget.packageSuffix;
      sourceFull = sourceFullFor suffix;
      sourceCompiler = sourceCompilerFor suffix;
      cmakeConfig = {
        target = suffix;
        amdgpuTargets = rocmTarget.buildTargets;
        distBundleName = suffix;
      };
      projectTargetUnexcludes = lib.optionalAttrs (builtins.elem suffix rocmTarget.buildTargets) {
        rocprofiler-compute = [ suffix ];
      };

      mkSource =
        nameSuffix: source:
        prev.callPackage ./rocm-source {
          name = "therock-rocm-source-${suffix}-${nameSuffix}";
          inherit (source)
            url
            ref
            rev
            hash
            fetchArgs
            ;
          deepNestedSubmodules = source.deepNestedSubmodules or [ ];
        };

      sourceTree = mkSource "full" sourceFull;
      compilerTree = mkSource "compiler-stage" sourceCompiler;

      amdLlvm = prev.callPackage ./rocm-from-source {
        stdenv = prev.llvmPackages_21.stdenv;
        inherit (cmakeConfig) target amdgpuTargets distBundleName;
        inherit (sourceCompiler) version;
        profile = "compiler";
        therockSource = compilerTree;
        thirdPartySources = { };
        buildTargets = [ "artifact-amd-llvm" ];
        installMode = "prebuilt-stages";
        inherit (therockFromSourceThirdPartyFetches) spirvHeadersSource;
      };

      fromSource = prev.callPackage ./rocm-from-source (
        {
          stdenv = prev.llvmPackages_21.stdenv;
          inherit (cmakeConfig) target amdgpuTargets distBundleName;
          inherit (sourceFull) version;
          profile = "full";
          therockSource = sourceTree;
          prebuiltStageTree = amdLlvm;
          thirdPartySources = therockRocmThirdPartySources.archives;
          inherit projectTargetUnexcludes;
        }
        // therockFromSourceThirdPartyFetches
      );
    in
    {
      "therock-rocm-source-${suffix}" = sourceTree;
      "therock-rocm-source-${suffix}-compiler-stage" = compilerTree;
      "therock-amd-llvm-${suffix}" = amdLlvm;
      "therock-rocm-from-source-${suffix}" = fromSource;
      "therock-rocm-from-source-${suffix}-configure" = fromSource.override {
        nameSuffix = "configure";
        prebuiltStageTree = null;
        installMode = "configure-only";
      };
      "therock-rocm-third-party-mirror-${suffix}" = fromSource.thirdPartyMirror;
      "therock-rocm-${suffix}-core" = final.therockRocmPackages.${suffix}.rocm-core;
      "therock-rocm-${suffix}-cmake" = final.therockRocmPackages.${suffix}.rocm-cmake;
    };

  therockFromSourceTargets = builtins.filter (
    rocmTarget: builtins.any (source: sourceMatchesTarget rocmTarget.packageSuffix source) sourceEntries
  ) rocmTargets;

  therockFromSourcePerArch = lib.foldl' lib.recursiveUpdate { } (
    map mkTherockFromSourceStack therockFromSourceTargets
  );
in
{
  therockRocmPackages = lib.recurseIntoAttrs (
    prev.callPackage ./rocm-modules {
      therockSource = final."therock-rocm-source-${target.packageSuffix}";
      therockVersion = defaultSourceFull.rocmVersion or (normalizeRocmVersion defaultSourceFull.version);
    }
  );
}
// torchRocmPerArch
// therockRocmSdkPerArch
// therockPythonPerArch
// therockFromSourcePerArch
