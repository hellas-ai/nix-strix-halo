{
  description = "Hydra CI job wrapper";

  inputs.src.url = "path:../..";
  inputs.nixpkgs.follows = "src/nixpkgs";

  outputs =
    { nixpkgs, src, ... }:
    let
      inherit (nixpkgs) lib;

      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];
    in
    {
      hydraJobs = lib.genAttrs systems (
        system:
        import "${src}/hydra.nix" {
          self = src;
          inherit system;
          jobset = "default";
        }
      );
    };
}
