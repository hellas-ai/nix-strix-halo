{ mkRocmTarget }:

let
  defaultRocmTarget = mkRocmTarget {
    packageSuffix = "gfx1151";
    hsaOverride = "11.5.1";
    description = "gfx1151 ROCm target";

    # gfx90a is a composable_kernel build workaround; top-level TheRock target
    # packages are still keyed by the explicit target records below.
    rocmGpuTargets = [
      "gfx1151"
      "gfx90a"
    ];
  };
in
{
  inherit defaultRocmTarget;

  defaultRocmGpuTargets = defaultRocmTarget.rocmGpuTargets;
  rocmTargets = [ defaultRocmTarget ];
}
