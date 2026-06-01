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
#   - "therock-source"  TheRock built from source via
#                        pkgs/therock/rocm-from-source. Reuses the
#                        therock-bin attrs and swaps the active
#                        target's `therock-rocm-${suffix}` alias to
#                        point at the source build.
#   - "nixpkgs"         nixpkgs.rocmPackages.${target}, which nixpkgs
#                        already publishes as the per-arch narrowed
#                        scope. Doesn't pull in any TheRock attrs, so
#                        downstream packages that reference
#                        therock-rocm-* (vllm, ds4, mlx) won't resolve.

let
  providers = import ../lib/providers.nix { inherit lib; };
  assertionOk = providers.assertRocmProvider provider;

  therockBinOverlay = import ../pkgs/therock {
    inherit lib rocmTargets;
    target = rocmTarget;
    therockRocmSources = sources.rocm;
    therockPythonWheelSources = sources.pythonWheels;
    therockRocmSourcePins = sources.rocmSourcePins;
    therockRocmSourceTrees = sources.rocmSourceTrees;
    therockRocmThirdPartySources = sources.rocmThirdParty;
  };

  suffix = rocmTarget.packageSuffix;
in

assert assertionOk;

final: prev:
if !prev.stdenv.isLinux then
  { }
else if provider == "therock-bin" then
  therockBinOverlay final prev
else if provider == "therock-source" then
  let
    binAttrs = therockBinOverlay final prev;
  in
  binAttrs
  // {
    # Swap the active target's `therock-rocm-${suffix}` alias to point at
    # the from-source build. Downstream consumers (vllm overlay, the
    # `therock-rocm` alias in overlays/pkgs.nix) read this attr, so the
    # source-built SDK flows through transparently. Sibling attrs
    # (`-env`, `-rocshmem-env`, `-core`, `-cmake`) still reference the
    # binary tarball — they are mostly tooling wrappers and don't change
    # behaviour for the source build.
    "therock-rocm-${suffix}" = binAttrs."therock-rocm-from-source-${suffix}";
  }
else if provider == "nixpkgs" then
  let
    narrowed =
      prev.rocmPackages.${suffix}
        or (throw "nixpkgs.rocmPackages has no narrowed scope for ${suffix}");
  in
  {
    rocmPackages = narrowed;
  }
else
  throw "unreachable: assertRocmProvider should have rejected ${provider}"
