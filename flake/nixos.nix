{
  inputs,
  nixpkgs,
  overlays,
  self,
}:

{
  modules = {
    default = _: {
      nixpkgs.overlays = [ overlays.default ];
    };

    rpc-server = import ../modules/rpc-server.nix;
    benchmark-runner = import ../modules/benchmark-runner.nix;
    ec-su-axb35 = import ../modules/ec-su-axb35.nix;
    ryzenadj = import ../modules/ryzenadj.nix;
    disko-raid0 = import ../modules/disko-raid0.nix;
    tuning = import ../modules/tuning.nix;
  };

  configurations = {
    fevm-faex9 = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      specialArgs = {
        inherit inputs self;
      };
      modules = [
        inputs.disko.nixosModules.disko
        ../examples/fevm-faex9/configuration.nix
      ];
    };
  };
}
