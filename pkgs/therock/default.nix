{
  lib,
  target,
  rocmTargets,
  therockRocmSources,
  therockPythonWheelSources,
  therockRocmSourcePins,
  therockRocmSourceTrees,
  therockRocmThirdPartySources,
  therockPythonConfig ? import ./python-config.nix { inherit lib; },
}:
final: prev:
let
  therockPython = prev.${therockPythonConfig.packageAttr};
  therockPythonPackages = prev.${therockPythonConfig.packagesAttr};

  targetBySuffix = lib.listToAttrs (
    map (rocmTarget: {
      name = rocmTarget.packageSuffix;
      value = rocmTarget;
    }) rocmTargets
  );

  targetFor =
    suffix:
    if builtins.hasAttr suffix targetBySuffix then
      targetBySuffix.${suffix}
    else
      throw "missing TheRock target config for ${suffix}";

  hsaOverrideFor = rocmTarget: rocmTarget.hsaOverride or null;

  sourcePinsByTarget = therockRocmSourcePins.targets;

  sourceFor =
    suffix:
    if builtins.hasAttr suffix sourcePinsByTarget then
      sourcePinsByTarget.${suffix}
    else
      throw "missing TheRock source pin for ${suffix}";

  defaultSource = sourceFor target.packageSuffix;

  lockedSourceTreeFor =
    suffix:
    let
      sourceTree = therockRocmSourceTrees.${suffix};
    in
    if !(builtins.hasAttr suffix therockRocmSourceTrees) then
      throw "missing locked TheRock source tree inputs for ${suffix}"
    else if !(builtins.isAttrs sourceTree && sourceTree ? root && sourceTree ? submodules) then
      throw "invalid locked TheRock source tree inputs for ${suffix}"
    else
      sourceTree;

  pythonWheelSourcesByTarget = therockPythonWheelSources.targets;

  pythonWheelSourcesFor =
    rocmTarget:
    let
      bySuffix = pythonWheelSourcesByTarget.${rocmTarget.packageSuffix} or null;
    in
    if bySuffix != null then bySuffix else pythonWheelSourcesByTarget.${rocmTarget.runtimeArch} or null;

  hasPythonWheelSources = rocmTarget: pythonWheelSourcesFor rocmTarget != null;

  normalizeRocmVersion =
    version: if builtins.length (lib.splitVersion version) == 2 then "${version}.0" else version;

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
      wheels = prev.callPackage ./python-wheels {
        inherit therockPython therockPythonPackages wheelSources;
      };
    in
    assert lib.assertMsg (wheelSources.pythonTag == therockPythonConfig.pythonTag)
      "TheRock wheel pin for ${suffix} uses ${wheelSources.pythonTag}, but Python config expects ${therockPythonConfig.pythonTag}";
    {
      "therock-python-${suffix}" = prev.callPackage ./python-env {
        inherit
          therockPython
          wheels
          wheelSources
          ;
      };
      "therock-python-wheels-${suffix}" = wheels;
      "therock-amdsmi-${suffix}" = therockPythonPackages.callPackage ./amdsmi {
        wheels = final."therock-python-wheels-${suffix}";
      };
      "torch-rocm-${suffix}" = final."therock-python-wheels-${suffix}";
    };

  therockPythonTargets = builtins.filter hasPythonWheelSources rocmTargets;

  therockPythonPerArch = lib.foldl' lib.recursiveUpdate { } (
    map mkTherockPythonStack therockPythonTargets
  );

  therockFromSourceThirdPartyFetches = {
    esmiIbLibrarySource = prev.fetchgit {
      inherit (therockRocmThirdPartySources.esmiIbLibrary) url rev hash;
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
  };

  mkTherockFromSourceStack =
    rocmTarget:
    let
      suffix = rocmTarget.packageSuffix;
      source = sourceFor suffix;
      cmakeConfig = {
        target = suffix;
        amdgpuTargets = rocmTarget.buildTargets;
        distBundleName = suffix;
      };
      projectTargetUnexcludes = lib.optionalAttrs (builtins.elem suffix rocmTarget.buildTargets) {
        rocprofiler-compute = [ suffix ];
      };

      mkLockedSource =
        source:
        let
          sourceTreeInputs = lockedSourceTreeFor suffix;
          installSubmodules = lib.concatMapStringsSep "\n" (submodule: ''
            install_source ${lib.escapeShellArg submodule.path} ${lib.escapeShellArg (toString submodule.source)}
          '') sourceTreeInputs.submodules;
        in
        prev.runCommandLocal "therock-rocm-source-${suffix}-full-${builtins.substring 0 12 source.rev}"
          {
            nativeBuildInputs = [ prev.patch ];
          }
          ''
            install_source() {
              destination="$1"
              source_path="$2"

              mkdir -p "$out/$(dirname "$destination")"
              rm -rf "$out/$destination"
              mkdir -p "$out/$destination"
              cp -a --reflink=auto "$source_path"/. "$out/$destination"/
              chmod -R u+w "$out/$destination"
            }

            apply_project_patches() {
              patch_project="$1"
              source_path="$2"
              patch_dir="$out/patches/amd-mainline/$patch_project"

              if [ ! -d "$patch_dir" ]; then
                return 0
              fi

              for patch_file in "$patch_dir"/*.patch; do
                if [ -e "$patch_file" ]; then
                  patch -d "$out/$source_path" -p1 --forward --batch < "$patch_file"
                fi
              done
            }

            mkdir -p "$out"
            cp -a --reflink=auto ${sourceTreeInputs.root}/. "$out/"
            chmod -R u+w "$out"

            ${installSubmodules}

            apply_project_patches rocm-systems rocm-systems
            apply_project_patches llvm-project compiler/amd-llvm
            apply_project_patches rocm-libraries rocm-libraries
          '';

      sourceTree = mkLockedSource source;

      amdLlvm = prev.callPackage ./rocm-from-source {
        stdenv = prev.llvmPackages_21.stdenv;
        llvmPackages = prev.llvmPackages_21;
        inherit (cmakeConfig) target amdgpuTargets distBundleName;
        inherit (source) version;
        profile = "compiler";
        therockSource = sourceTree;
        thirdPartySources = { };
        buildTargets = [ "artifact-amd-llvm" ];
        installMode = "prebuilt-stages";
        inherit (therockFromSourceThirdPartyFetches) spirvHeadersSource;
      };

      fromSource = prev.callPackage ./rocm-from-source (
        {
          stdenv = prev.llvmPackages_21.stdenv;
          llvmPackages = prev.llvmPackages_21;
          inherit (cmakeConfig) target amdgpuTargets distBundleName;
          inherit (source) version;
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
    rocmTarget: builtins.hasAttr rocmTarget.packageSuffix sourcePinsByTarget
  ) rocmTargets;

  therockFromSourcePerArch = lib.foldl' lib.recursiveUpdate { } (
    map mkTherockFromSourceStack therockFromSourceTargets
  );
in
{
  therockRocmPackages = lib.recurseIntoAttrs (
    prev.callPackage ./rocm-modules {
      therockSource = final."therock-rocm-source-${target.packageSuffix}";
      therockVersion = defaultSource.rocmVersion or (normalizeRocmVersion defaultSource.version);
    }
  );
}
// therockRocmSdkPerArch
// therockPythonPerArch
// therockFromSourcePerArch
