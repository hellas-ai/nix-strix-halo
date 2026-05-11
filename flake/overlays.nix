{
  inputs,
  rocmTargets,
  vllmLib,
}:

let
  # Nix checks every overlays attr as an overlay. This attr is intended to be
  # called with a CPU string; the fallback only satisfies bare overlay checks.
  mkTunedOverlay =
    final: if builtins.isString final then vllmLib.mkCpuTuningOverlay { cpu = final; } else _: { };
in
{
  default =
    final: prev:
    let
      ecPackages = prev.callPackage ../pkgs/ec-su-axb35.nix {
        ec-su-axb35-src = inputs.ec-su-axb35;
      };
    in
    {
      ec-su-axb35 = ecPackages.kernelModule;
      ec-su-axb35-monitor = ecPackages.monitor;

      rocmPackages = prev.rocmPackages.overrideScope (
        _: rocmPrev: {
          clr = rocmPrev.clr.override {
            localGpuTargets = rocmTargets;
          };
        }
      );

      llama-cpp-rocm = prev.llama-cpp.override {
        rocmSupport = true;
        rpcSupport = true;
        inherit (final) rocmPackages;
        rocmGpuTargets = rocmTargets;
      };
    };

  inherit mkTunedOverlay;

  tuned = vllmLib.mkCpuTuningOverlay {
    cpu = "znver5";
  };
}
