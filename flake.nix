{
  description = "Strix Halo NixOS configuration and tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    ec-su-axb35 = {
      url = "github:cmetz/ec-su_axb35-linux";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }@inputs:
    let
      inherit (nixpkgs) lib;
      systems = [ "x86_64-linux" ];
      forAllSystems = lib.genAttrs systems;

      # gfx90a added as workaround for composable_kernel sharding bug
      # (nixpkgs CK build fails without at least one gfx9 target)
      targets = [
        "gfx1151"
        "gfx90a"
      ];

      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ self.overlays.default ];
        };

      perSystem = f: forAllSystems (system: f (pkgsFor system));

      mkFevmFaex9Configuration =
        {
          system ? "x86_64-linux",
          extraModules ? [ ],
          specialArgs ? { },
        }:
        nixpkgs.lib.nixosSystem {
          inherit system;
          specialArgs = {
            inherit inputs self;
          }
          // specialArgs;
          modules = [
            ./examples/fevm-faex9/configuration.nix
          ]
          ++ extraModules;
        };
    in
    {
      lib = {
        inherit mkFevmFaex9Configuration;
      };

      overlays.default =
        final: prev:
        let
          ecPackages = prev.callPackage ./pkgs/ec-su-axb35.nix {
            ec-su-axb35-src = inputs.ec-su-axb35;
          };
        in
        {
          # EC-SU_AXB35 packages
          ec-su-axb35 = ecPackages.kernelModule;
          ec-su-axb35-monitor = ecPackages.monitor;

          # Build our gpu targets for nixos rocm
          rocmPackages = prev.rocmPackages.overrideScope (
            _: rocmPrev: {
              clr = rocmPrev.clr.override {
                localGpuTargets = targets;
              };
            }
          );

          # Llama.cpp with ROCm support using upstream nixpkgs
          llama-cpp-rocm = prev.llama-cpp.override {
            rocmSupport = true;
            rpcSupport = true;
            inherit (final) rocmPackages;
            rocmGpuTargets = targets;
          };
        };

      # NixOS modules
      nixosModules = {
        default = _: {
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

      nixosConfigurations = {
        fevm-faex9 = mkFevmFaex9Configuration { };
      };
    }
    // {
      packages = perSystem (
        pkgs:
        let
          benchmarks = import ./bench/default.nix {
            inherit pkgs;
            packages = {
              inherit (pkgs) llama-cpp-rocm llama-cpp-vulkan;
            };
          };
        in
        {
          default = pkgs.llama-cpp-rocm;
          inherit (pkgs)
            ec-su-axb35-monitor
            llama-cpp-rocm
            llama-cpp-vulkan
            ;
        }
        // (pkgs.lib.concatMapAttrs (
          model: benchs:
          pkgs.lib.mapAttrs' (name: drv: {
            name = "bench-${model}-${name}";
            value = drv;
          }) benchs
        ) benchmarks)
      );

      apps = perSystem (pkgs: {
        llama-cli = {
          type = "app";
          program = toString (
            pkgs.writeShellScript "llama-cli" ''
              export HSA_OVERRIDE_GFX_VERSION=11.5.1
              ${pkgs.llama-cpp-rocm}/bin/llama-cli "$@"
            ''
          );
          meta.description = "Run llama-cli with the Strix Halo ROCm environment";
        };

        llama-server = {
          type = "app";
          program = toString (
            pkgs.writeShellScript "llama-server" ''
              export HSA_OVERRIDE_GFX_VERSION=11.5.1
              ${pkgs.llama-cpp-rocm}/bin/llama-server "$@"
            ''
          );
          meta.description = "Run llama-server with the Strix Halo ROCm environment";
        };
      });

      devShells = perSystem (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            deadnix
            nix-fast-build
            nixfmt-tree
            statix
            (python3.withPackages (
              ps: with ps; [
                numpy
                pandas
                plotly
              ]
            ))
          ];
        };
      });

      formatter = perSystem (pkgs: pkgs.nixfmt-tree);

      checks = perSystem (
        pkgs:
        let
          mkSourceCheck =
            name: nativeBuildInputs: command:
            pkgs.runCommandLocal "ci-${name}"
              {
                inherit nativeBuildInputs;
                src = lib.cleanSource ./.;
              }
              ''
                export HOME="$TMPDIR"
                export XDG_CACHE_HOME="$TMPDIR/cache"
                cp -r --no-preserve=mode "$src" source
                cd source
                ${command}
                touch "$out"
              '';
        in
        {
          format = mkSourceCheck "format" [ pkgs.nixfmt-tree ] ''
            treefmt --tree-root . --walk filesystem --fail-on-change
          '';

          deadnix = mkSourceCheck "deadnix" [ pkgs.deadnix ] ''
            deadnix --fail .
          '';

          statix = mkSourceCheck "statix" [ pkgs.statix ] ''
            statix check .
          '';
        }
      );

      hydraJobs = perSystem (
        pkgs:
        let
          system = pkgs.stdenv.hostPlatform.system;

          mkAggregate =
            aggregateName: jobs:
            pkgs.linkFarm "nix-strix-halo-${aggregateName}" (
              lib.mapAttrsToList (name: path: {
                inherit name path;
              }) jobs
            );

          prQuickJobs = self.checks.${system};
          prQuick = mkAggregate "pr-quick" prQuickJobs;

          afterPrQuick =
            name: path:
            pkgs.linkFarm "nix-strix-halo-pr-full-${name}" [
              {
                name = "pr-quick";
                path = prQuick;
              }
              {
                inherit name path;
              }
            ];

          prFullJobs = {
            default = afterPrQuick "default" self.packages.${system}.default;
          }
          // lib.optionalAttrs (system == "x86_64-linux") {
            system = afterPrQuick "system" self.nixosConfigurations.fevm-faex9.config.system.build.toplevel;
          };
        in
        {
          "pr-quick" = prQuickJobs // {
            all = prQuick;
          };

          "pr-full" = prFullJobs // {
            all = mkAggregate "pr-full" prFullJobs;
          };
        }
      );
    };
}
