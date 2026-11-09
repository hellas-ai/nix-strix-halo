{
  lib,
  mediaRuntime ? false,
  target,
  rocmTargets,
  therockRocmSources,
  therockPythonWheelSources,
  therockRocmSourcePins,
  therockRocmSourceTrees,
  therockPythonConfig ? import ./python-config.nix { inherit lib; },
}:
final: prev:
let
  therockPython = prev.${therockPythonConfig.packageAttr};
  therockPythonPackages = prev.${therockPythonConfig.packagesAttr};

  sourcePinsByTarget = therockRocmSourcePins.targets;

  sourceFor =
    rocmTarget:
    if builtins.hasAttr rocmTarget.therockSourceTarget sourcePinsByTarget then
      sourcePinsByTarget.${rocmTarget.therockSourceTarget}
    else
      throw "missing TheRock source pin for ${rocmTarget.packageSuffix}";

  defaultSource = sourceFor target;

  lockedSourceTreeFor =
    rocmTarget:
    let
      sourceTree = therockRocmSourceTrees.${rocmTarget.therockSourceTarget};
    in
    if !(builtins.hasAttr rocmTarget.therockSourceTarget therockRocmSourceTrees) then
      throw "missing locked TheRock source tree inputs for ${rocmTarget.packageSuffix}"
    else if !(builtins.isAttrs sourceTree && sourceTree ? root && sourceTree ? submodules) then
      throw "invalid locked TheRock source tree inputs for ${rocmTarget.packageSuffix}"
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
      sdk = prev.callPackage ./rocm-sdk {
        target = suffix;
        inherit (source) version url hash;
      };
    in
    {
      "therock-rocm-${suffix}" = sdk;
    };

  therockRocmSdkPerArch = lib.foldl' lib.recursiveUpdate { } (
    lib.mapAttrsToList mkTherockRocmSdkAttrs therockRocmSources.linux
  );

  mkTherockPythonStack =
    rocmTarget:
    let
      suffix = rocmTarget.packageSuffix;
      baseWheelSources = pythonWheelSourcesFor rocmTarget;
      wheelSources =
        baseWheelSources
        // lib.optionalAttrs mediaRuntime {
          packages = baseWheelSources.packages // {
            rocm-bootstrap = therockPythonWheelSources.mediaBootstrap;
          };
        };
      wheels = prev.callPackage ./python-wheels {
        inherit
          therockPython
          therockPythonPackages
          wheelSources
          mediaRuntime
          ;
      };
    in
    assert lib.assertMsg (wheelSources.pythonTag == therockPythonConfig.pythonTag)
      "TheRock wheel pin for ${suffix} uses ${wheelSources.pythonTag}, but Python config expects ${therockPythonConfig.pythonTag}";
    {
      "therock-python-wheels-${suffix}" = wheels;
      "torch-rocm-${suffix}" = final."therock-python-wheels-${suffix}";
    };

  therockPythonTargets = builtins.filter hasPythonWheelSources rocmTargets;

  therockPythonPerArch = lib.foldl' lib.recursiveUpdate { } (
    map mkTherockPythonStack therockPythonTargets
  );

  mkTherockSourceStack =
    rocmTarget:
    let
      suffix = rocmTarget.packageSuffix;
      source = sourceFor rocmTarget;
      sourceTreeInputs = lockedSourceTreeFor rocmTarget;

      mkLockedSource =
        source: lockedInputs:
        let
          installSubmodules = lib.concatMapStringsSep "\n" (submodule: ''
            install_source ${lib.escapeShellArg submodule.path} ${lib.escapeShellArg (toString submodule.source)}
          '') lockedInputs.submodules;
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
            cp -a --reflink=auto ${lockedInputs.root}/. "$out/"
            chmod -R u+w "$out"

            ${installSubmodules}

            apply_project_patches rocm-systems rocm-systems
            apply_project_patches llvm-project compiler/amd-llvm
            apply_project_patches rocm-libraries rocm-libraries
          '';

      sourceTree = mkLockedSource source sourceTreeInputs;
    in
    {
      "therock-rocm-source-${suffix}" = sourceTree;
      "therock-rocm-${suffix}-core" = final.therockRocmPackages.${suffix}.rocm-core;
      "therock-rocm-${suffix}-cmake" = final.therockRocmPackages.${suffix}.rocm-cmake;
    };

  therockSourceTargets = builtins.filter (
    rocmTarget: builtins.hasAttr rocmTarget.therockSourceTarget sourcePinsByTarget
  ) rocmTargets;

  therockSourcePerArch = lib.foldl' lib.recursiveUpdate { } (
    map mkTherockSourceStack therockSourceTargets
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
// therockSourcePerArch
