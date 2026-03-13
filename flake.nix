{
  description = "Strix Halo NixOS configuration and tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    ec-su-axb35 = {
      url = "github:cmetz/ec-su_axb35-linux";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    disko,
    ...
  } @ inputs: let
    # gfx90a added as workaround for composable_kernel sharding bug
    # (nixpkgs CK build fails without at least one gfx9 target)
    targets = ["gfx1151" "gfx90a"];
  in
    {
      overlays.default = final: prev: let
        ecPackages = prev.callPackage ./pkgs/ec-su-axb35.nix {
          ec-su-axb35-src = inputs.ec-su-axb35;
        };
      in {
        # EC-SU_AXB35 packages
        ec-su-axb35 = ecPackages.kernelModule;
        ec-su-axb35-monitor = ecPackages.monitor;

        # Build our gpu targets for nixos rocm
        rocmPackages = prev.rocmPackages.overrideScope (
          rocmFinal: rocmPrev: {
            clr = rocmPrev.clr.override {
              localGpuTargets = targets;
            };
          }
        );

        # Llama.cpp with ROCm support using upstream nixpkgs
        llamacpp-rocm = prev.llama-cpp.override {
          rocmSupport = true;
          rpcSupport = true;
          rocmPackages = final.rocmPackages;
          rocmGpuTargets = targets;
        };
      };

      # NixOS modules
      nixosModules = {
        default = {
          config,
          pkgs,
          ...
        }: {
          nixpkgs.overlays = [
            self.overlays.default
          ];
        };
        rpc-server = import ./modules/rpc-server.nix;
        benchmark-runner = import ./modules/benchmark-runner.nix;
        ec-su-axb35 = import ./modules/ec-su-axb35.nix;
        ryzenadj = import ./modules/ryzenadj.nix;
        disko-raid0 = import ./modules/disko-raid0.nix;
        tuning = import ./modules/tuning.nix;
      };

      # NixOS configurations
      nixosConfigurations = {
        fevm-faex9 = nixpkgs.lib.nixosSystem {
          system = "x86_64-linux";
          specialArgs = {
            inherit inputs;
            inherit self;
          };
          modules = [
            inputs.disko.nixosModules.disko
            ./examples/fevm-faex9/configuration.nix
          ];
        };
      };
    }
    //
    # Per-system outputs (ROCm only works on Linux)
    flake-utils.lib.eachSystem ["x86_64-linux"] (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [self.overlays.default];
        };

        benchmarks = import ./bench/default.nix {
          inherit pkgs;
          packages = {
            inherit (pkgs) llamacpp-rocm llama-cpp-vulkan;
          };
        };
      in {
        packages =
          {
            default = pkgs.llamacpp-rocm;
            llamacpp-rocm = pkgs.llamacpp-rocm;
            llama-cpp-vulkan = pkgs.llama-cpp-vulkan;
            ec-su-axb35-monitor = pkgs.ec-su-axb35-monitor;
          }
          // (pkgs.lib.concatMapAttrs (
              model: benchs:
                pkgs.lib.mapAttrs' (
                  name: drv: {
                    name = "bench-${model}-${name}";
                    value = drv;
                  }
                )
                benchs
            )
            benchmarks);

        apps = {
          llama-cli = {
            type = "app";
            program = toString (pkgs.writeShellScript "llama-cli" ''
              export HSA_OVERRIDE_GFX_VERSION=11.5.1
              ${pkgs.llamacpp-rocm}/bin/llama-cli "$@"
            '');
          };

          llama-server = {
            type = "app";
            program = toString (pkgs.writeShellScript "llama-server" ''
              export HSA_OVERRIDE_GFX_VERSION=11.5.1
              ${pkgs.llamacpp-rocm}/bin/llama-server "$@"
            '');
          };
        };
      }
    )
    // flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [self.overlays.default];
        };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (python3.withPackages (ps:
              with ps; [
                pandas
                plotly
                numpy
              ]))
            nix-fast-build
          ];
        };
      }
    );
}
