{
  inputs,
  lib,
  inputVersion,
  rocmTargets,
  defaultRocmTarget,
  defaultTherockSources,
}:

# DS4 (DwarfStar 4) ROCm packaging. Builds one variant per rocmTarget
# that has matching TheRock binary sources; exposes the default-target
# build as the unsuffixed `ds4-rocm` alias.

final: prev:

let
  ds4RocmTargets = builtins.filter (
    rocmTarget: builtins.hasAttr rocmTarget.packageSuffix defaultTherockSources.rocm.linux
  ) rocmTargets;

  ds4RocmTargetPackages = lib.listToAttrs (
    map (
      rocmTarget:
      let
        s = rocmTarget.packageSuffix;
      in
      {
        name = "ds4-rocm-${s}";
        value = prev.callPackage ../pkgs/ds4-rocm {
          src = inputs.ds4-hip;
          rocmSdk = final."therock-rocm-${s}";
          version = inputVersion "experimental" inputs.ds4-hip;
          packageSuffix = s;
          offloadArch = builtins.head rocmTarget.buildTargets;
          hsaOverrideGfxVersion = rocmTarget.hsaOverride or null;
        };
      }
    ) ds4RocmTargets
  );
in
ds4RocmTargetPackages
// {
  ds4-rocm = ds4RocmTargetPackages."ds4-rocm-${defaultRocmTarget.packageSuffix}";
}
