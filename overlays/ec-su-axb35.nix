{ inputs }:

_: prev:

let
  ecPackages = prev.callPackage ../pkgs/ec-su-axb35.nix {
    ec-su-axb35-src = inputs.ec-su-axb35;
  };
in
{
  ec-su-axb35 = ecPackages.kernelModule;
  ec-su-axb35-monitor = ecPackages.monitor;
}
