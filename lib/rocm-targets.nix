{ lib }:

rec {
  mkRocmTarget =
    {
      packageSuffix,
      runtimeArch ? packageSuffix,
      buildTargets ? [ runtimeArch ],
      rocmGpuTargets ? buildTargets,
      systemFeature ? runtimeArch,
      hsaOverride ? null,
      therockTarget ? packageSuffix,
      description ? packageSuffix,
    }:
    {
      inherit
        packageSuffix
        runtimeArch
        buildTargets
        rocmGpuTargets
        systemFeature
        hsaOverride
        therockTarget
        description
        ;
    };

  mkRocmNarrowOverlay =
    {
      rocmGpuTargets,
      packageAttrs ? [ "clr" ],
      extraScope ? (_rocmFinal: _rocmPrev: { }),
    }:
    _final: prev:
    lib.optionalAttrs prev.stdenv.isLinux {
      rocmPackages = prev.rocmPackages.overrideScope (
        rocmFinal: rocmPrev:
        (lib.genAttrs packageAttrs (
          name:
          rocmPrev.${name}.override {
            localGpuTargets = rocmGpuTargets;
          }
        ))
        // extraScope rocmFinal rocmPrev
      );
    };
}
