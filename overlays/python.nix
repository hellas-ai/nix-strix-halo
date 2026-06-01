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
# Reserved future providers are listed in lib.providers.pythonProviderStubs.

final: prev:
if !prev.stdenv.isLinux then
  { }
else
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
    else
      throw "unreachable: assertPythonProvider should have rejected ${provider}"
  )
