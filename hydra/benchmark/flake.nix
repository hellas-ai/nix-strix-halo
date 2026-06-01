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
    in
    {
      hydraJobs = lib.genAttrs benchmarkSystems (
        system:
        import "${src}/hydra.nix" {
          self = src;
          inherit system;
          jobset = "benchmarks";
        }
      );
    };
}
