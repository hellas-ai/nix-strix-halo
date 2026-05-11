inputs@{
  self,
  nixpkgs,
  nixpkgs-vllm,
  ...
}:

let
  inherit (nixpkgs) lib;

  systems = [ "x86_64-linux" ];
  forAllSystems = lib.genAttrs systems;

  vllmLib = import ./vllm.nix {
    inherit inputs lib nixpkgs-vllm;
  };

  rocmTargets = [
    "gfx1151"
    # gfx90a works around a composable_kernel sharding bug in nixpkgs.
    "gfx90a"
  ];

  overlays = import ./overlays.nix {
    inherit
      inputs
      rocmTargets
      vllmLib
      ;
  };

  nixos = import ./nixos.nix {
    inherit
      inputs
      nixpkgs
      overlays
      self
      ;
  };

  defaultPackagesFor =
    system:
    import nixpkgs {
      inherit system;
      overlays = [ overlays.default ];
    };

  zen5PackagesFor =
    system:
    import nixpkgs {
      inherit system;
      overlays = [
        overlays.default
        overlays.tuned
      ];
    };

  perSystem =
    f:
    forAllSystems (
      system:
      let
        pkgs = defaultPackagesFor system;
        zen5Packages = zen5PackagesFor system;
      in
      f {
        inherit
          pkgs
          system
          zen5Packages
          ;
      }
    );
in
{
  inherit overlays;

  lib = vllmLib;

  nixosModules = nixos.modules;
  nixosConfigurations = nixos.configurations;

  legacyPackages = perSystem (
    { pkgs, zen5Packages, ... }:
    {
      defaultPackages = pkgs;
      inherit zen5Packages;
    }
  );

  packages = perSystem (
    {
      system,
      pkgs,
      ...
    }:
    import ./packages.nix {
      inherit
        lib
        pkgs
        system
        vllmLib
        ;
    }
  );

  apps = perSystem (
    { pkgs, ... }:
    import ./apps.nix {
      inherit pkgs;
    }
  );

  devShells = perSystem (
    { pkgs, ... }:
    import ./dev-shells.nix {
      inherit pkgs;
    }
  );

  formatter = perSystem ({ pkgs, ... }: pkgs.nixfmt-tree);

  checks = perSystem (
    { pkgs, ... }:
    import ./checks.nix {
      inherit lib pkgs;
      src = ../.;
    }
  );
}
