{
  mkRocmTarget =
    {
      packageSuffix,
      runtimeArch ? packageSuffix,
      buildTargets ? [ runtimeArch ],
      rocmGpuTargets ? buildTargets,
      systemFeature ? runtimeArch,
      hsaOverride ? null,
      therockTarget ? packageSuffix,
      therockSourceTarget ? packageSuffix,
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
        therockSourceTarget
        description
        ;
    };
}
