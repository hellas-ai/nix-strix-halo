{
  lib,
  provider ? "therock-wheels",
  rocmTarget,
}:

# Parameterised python overlay. Swaps `python312Packages.torch`,
# `triton`, `amdsmi`, and the `rocm-sdk-*` shims depending on provider.
#
#   - "therock-wheels"  TheRock-published binary wheels. Default.
#                        Delegates to overlays/therock-python.nix.
#   - "nixpkgs"         nixpkgs torch with rocmSupport=true and the
#                        target's gpuTargets. Stub — see TODO.
#   - "therock-source"  PyTorch + Triton built from source against the
#                        chosen rocm provider. Stub.

final: prev:
if !prev.stdenv.isLinux then { } else
(
  let
    providers = import ../lib/providers.nix { inherit lib; };
  in

  assert providers.assertPythonProvider provider;

  if provider == "therock-wheels" then
    import ./therock-python.nix {
      inherit lib;
      target = rocmTarget;
    } final prev
  else if provider == "nixpkgs" then
    throw ''
      python provider "nixpkgs" is not implemented yet. Would override
      pythonPackagesExtensions to swap torch -> nixpkgs torch with
      rocmSupport=true and gpuTargets = rocmTarget.gpuTargets, plus
      triton -> triton-no-cuda. See lib/providers.nix.
    ''
  else if provider == "therock-source" then
    throw ''
      python provider "therock-source" is not implemented yet. Would
      build PyTorch and Triton from source against the active rocm
      provider's SDK. See lib/providers.nix.
    ''
  else
    throw "unreachable: assertPythonProvider should have rejected ${provider}"
)
