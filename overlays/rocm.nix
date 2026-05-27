{
  lib,
  provider ? "therock-bin",
  rocmTarget,
  rocmTargets,
  sources,
}:

# Parameterised ROCm overlay. Picks one of:
#
#   - "therock-bin"     binary TheRock SDK tarball pinned per target;
#                        delegates to pkgs/therock/default.nix.
#   - "nixpkgs"         nixpkgs.rocmPackages narrowed to the target's
#                        gpuTargets (magma, rccl, hipblas family, CK,
#                        aotriton). Stub — see TODO.
#   - "therock-source"  TheRock built from source via
#                        pkgs/therock/rocm-from-source. Stub — wire up
#                        when the source build leaves the experimental
#                        track in pkgs/therock/.
#
# The same overlay also baseline-narrows the target's gpuTargets where
# the provider doesn't already, so cross-provider consumers see a
# uniform `pkgs.rocmPackages` shape.

let
  providers = import ../lib/providers.nix { inherit lib; };
in

assert providers.assertRocmProvider provider;

if provider == "therock-bin" then
  import ../pkgs/therock {
    inherit lib rocmTargets;
    target = rocmTarget;
    therockRocmSources = sources.rocm;
    therockPythonWheelSources = sources.pythonWheels;
    therockRocmSourcePins = sources.rocmSourcePins;
    therockRocmSourceTrees = sources.rocmSourceTrees;
    therockRocmThirdPartySources = sources.rocmThirdParty;
  }
else if provider == "nixpkgs" then
  throw ''
    rocm provider "nixpkgs" is not implemented yet. The provider tag is
    reserved; an overlay would narrow nixpkgs rocmPackages (magma, rccl,
    hipblaslt, hipfft, miopen, rocblas, rocfft, composable_kernel,
    aotriton) to rocmTarget.gpuTargets. See lib/providers.nix.
  ''
else if provider == "therock-source" then
  throw ''
    rocm provider "therock-source" is not implemented yet. Source-built
    TheRock lives in pkgs/therock/rocm-from-source but is not wired into
    a final overlay. See lib/providers.nix.
  ''
else
  throw "unreachable: assertRocmProvider should have rejected ${provider}"
