{
  lib,
  mediaRuntime ? false,
  provider ? "therock-bin",
  rocmTarget,
  rocmTargets,
  sources,
  therockPythonConfig ? import ../pkgs/therock/python-config.nix { inherit lib; },
}:

# Parameterised ROCm overlay. Picks one of:
#
#   - "therock-bin"     binary TheRock SDK tarball pinned per target;
#                        delegates to pkgs/therock/default.nix.
#   - "nixpkgs"         nixpkgs.rocmPackages.${target}, which nixpkgs
#                        already publishes as the per-arch narrowed
#                        scope. Doesn't pull in any TheRock attrs; the
#                        local package overlay only exposes packages
#                        that can build without them.

let
  providers = import ../lib/providers.nix { inherit lib; };
  assertionOk = providers.assertRocmProvider provider;

  therockBinOverlay = import ../pkgs/therock {
    inherit lib rocmTargets mediaRuntime;
    target = rocmTarget;
    therockRocmSources = sources.rocm;
    therockPythonWheelSources = sources.pythonWheels;
    therockRocmSourcePins = sources.rocmSourcePins;
    therockRocmSourceTrees = sources.rocmSourceTrees;
    inherit therockPythonConfig;
  };

  suffix = rocmTarget.packageSuffix;
in

assert assertionOk;

final: prev:
if !prev.stdenv.hostPlatform.isLinux then
  { }
else if provider == "therock-bin" then
  therockBinOverlay final prev
else if provider == "nixpkgs" then
  let
    narrowed =
      prev.rocmPackages.${suffix} or (throw "nixpkgs.rocmPackages has no narrowed scope for ${suffix}");
  in
  {
    rocmPackages = narrowed;
  }
else
  throw "unreachable: assertRocmProvider should have rejected ${provider}"
