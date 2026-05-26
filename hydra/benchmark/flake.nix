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

      addBenchmarkFeature =
        drv:
        drv.overrideAttrs (old: {
          requiredSystemFeatures = lib.unique ((old.requiredSystemFeatures or [ ]) ++ [ "benchmark" ]);
        });
    in
    {
      hydraJobs = lib.genAttrs benchmarkSystems (system:
        lib.mapAttrs (_: addBenchmarkFeature) src.benchmarks.${system});
    };
}
