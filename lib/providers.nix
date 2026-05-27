{ lib }:

# Provider tag registry for the parameterised overlays. The flake's
# package set is built from one rocm provider × one python provider ×
# the rocm target; everything else (which packages live in the flake,
# CPU tuning) is orthogonal.
#
# Adding a new provider:
#   1. Implement the overlay branch in overlays/{rocm,python}.nix.
#   2. Add the tag to the corresponding list below.
#
# Consumers that pick a provider by string tag get an assertion failure
# with the list of valid tags if they typo.

let
  rocmProviders = [
    # Binary TheRock SDK tarball + per-target rocm-libs + python wheel
    # narrowing. Default. Implemented in overlays/rocm.nix.
    "therock-bin"

    # nixpkgs `rocmPackages.*`, narrowed to the target's gpuTargets.
    # Not implemented yet — overlays/rocm.nix throws.
    "nixpkgs"

    # TheRock built from source via pkgs/therock/rocm-from-source.
    # Not implemented yet — overlays/rocm.nix throws.
    "therock-source"
  ];

  pythonProviders = [
    # TheRock-published cu/rocm wheels. Default. Implemented in
    # overlays/python.nix.
    "therock-wheels"

    # nixpkgs python3Packages.torch with rocmSupport = true. Stub.
    "nixpkgs"

    # PyTorch + Triton built from source against the chosen rocm
    # provider. Stub.
    "therock-source"
  ];

  assertProvider =
    kind: providers: tag:
    lib.assertMsg (lib.elem tag providers)
      "unknown ${kind} provider \"${tag}\"; valid: ${lib.concatStringsSep ", " providers}";
in
{
  inherit rocmProviders pythonProviders;

  assertRocmProvider = tag: assertProvider "rocm" rocmProviders tag;
  assertPythonProvider = tag: assertProvider "python" pythonProviders tag;
}
