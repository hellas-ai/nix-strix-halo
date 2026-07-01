{
  lib,
  provider ? "nixpkgs",
  rocmTarget ? null,
  therockPythonConfig ? import ../pkgs/therock/python-config.nix { inherit lib; },
}:

# Parameterised Python overlay. Always applies shared Python package
# compatibility fixes, and optionally swaps the configured TheRock Python
# package set's `torch`, `triton`, `amdsmi`, and `rocm-sdk-*` shims.
#
#   - "nixpkgs"         Stock nixpkgs Python packages plus the shared fixes.
#   - "therock-wheels"  TheRock-published binary wheels, used by the
#                        flake's default package set.
# Reserved future providers are listed in lib.providers.pythonProviderStubs.

final: prev:
let
  providers = import ../lib/providers.nix { inherit lib; };
  pythonFixes =
    _pyfinal: pyprev:
    lib.optionalAttrs (prev.stdenv.hostPlatform.isDarwin && pyprev ? sentence-transformers) {
      # sentence-transformers' runtime closure is usable on Darwin, but its
      # nixpkgs test extras pull `phonemizer -> dlinfo`, and dlinfo is marked
      # broken on Darwin. Downstream eval tooling only needs runtime imports.
      sentence-transformers = pyprev.sentence-transformers.overridePythonAttrs (_old: {
        nativeCheckInputs = [ ];
        doCheck = false;
        pythonImportsCheck = [ ];
      });
    };

  providerAttrs =
    assert providers.assertPythonProvider provider;

    # TheRock wheels are Linux-only; non-Linux systems stay on stock
    # nixpkgs Python packages and still receive pythonFixes below.
    if provider == "nixpkgs" || !prev.stdenv.hostPlatform.isLinux then
      { }
    else if provider == "therock-wheels" then
      assert lib.assertMsg (rocmTarget != null) "therock-wheels python provider requires rocmTarget";
      import ./therock-python.nix {
        inherit lib;
        target = rocmTarget;
        pythonConfig = therockPythonConfig;
      } final prev
    else
      throw "unreachable: assertPythonProvider should have rejected ${provider}";
in
providerAttrs
// {
  pythonPackagesExtensions =
    (providerAttrs.pythonPackagesExtensions or (prev.pythonPackagesExtensions or [ ]))
    ++ [ pythonFixes ];
}
