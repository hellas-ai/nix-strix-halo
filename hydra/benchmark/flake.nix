{
  description = "Hydra benchmark job wrapper";

  inputs.src.url = "path:../..";
  inputs.nixpkgs.follows = "src/nixpkgs";

  outputs =
    { nixpkgs, src, ... }:
    let
      inherit (nixpkgs) lib;

      benchmarkSystems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      supportedFeatures = {
        x86_64-linux = [
          "benchmark"
          "gfx1151"
        ];
        aarch64-darwin = [ "benchmark" ];
      };

      addBenchmarkFeature =
        drv:
        drv.overrideAttrs (old: {
          requiredSystemFeatures = lib.unique ((old.requiredSystemFeatures or [ ]) ++ [ "benchmark" ]);
        });

      supportedOn =
        system: drv:
        lib.all (feature: lib.elem feature supportedFeatures.${system}) (drv.requiredSystemFeatures or [ ]);
    in
    {
      hydraJobs = lib.genAttrs benchmarkSystems (
        system:
        lib.filterAttrs (_: supportedOn system) (
          lib.mapAttrs (_: addBenchmarkFeature) src.benchmarks.${system}
        )
      );
    };
}
