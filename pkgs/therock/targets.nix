{ mkRocmTarget }:

let
  rocmTargetsBySuffix = {
    gfx1010 = mkRocmTarget {
      packageSuffix = "gfx1010";
      runtimeArch = "gfx1010";
      therockTarget = "gfx101X-dgpu";
      description = "gfx1010 ROCm target";
    };

    gfx1030 = mkRocmTarget {
      packageSuffix = "gfx1030";
      therockTarget = "gfx103X-all";
      # The source checkout is architecture-independent; share its lock graph.
      therockSourceTarget = "gfx1151";
      description = "Radeon Pro V620 (gfx1030) ROCm target";
    };

    gfx1036 = mkRocmTarget {
      packageSuffix = "gfx1036";
      runtimeArch = "gfx1036";
      buildTargets = [ "gfx1030" ];
      hsaOverride = "10.3.0";
      therockTarget = "gfx103X-all";
      description = "gfx1036 ROCm target built as gfx1030";
    };

    gfx1103 = mkRocmTarget {
      packageSuffix = "gfx1103";
      runtimeArch = "gfx1103";
      buildTargets = [ "gfx1102" ];
      hsaOverride = "11.0.2";
      therockTarget = "gfx110X-all";
      description = "gfx1103 ROCm target built as gfx1102";
    };

    gfx1151 = mkRocmTarget {
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
  };
  defaultRocmTarget = rocmTargetsBySuffix.gfx1151;
in
{
  inherit rocmTargetsBySuffix;

  inherit defaultRocmTarget;

  defaultRocmGpuTargets = defaultRocmTarget.rocmGpuTargets;
  rocmTargets = [
    rocmTargetsBySuffix.gfx1010
    rocmTargetsBySuffix.gfx1030
    rocmTargetsBySuffix.gfx1036
    rocmTargetsBySuffix.gfx1103
    rocmTargetsBySuffix.gfx1151
  ];
}
