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
      linuxSystems = [ "x86_64-linux" ];
      darwinSystems = [
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      systems = linuxSystems ++ darwinSystems;
      forAllSystems = lib.genAttrs systems;
      forLinuxSystems = lib.genAttrs linuxSystems;

      rocmTargetLib = import ./lib/rocm-targets.nix { inherit lib; };
      therockTargetConfig = import ./pkgs/therock/targets.nix {
        inherit (rocmTargetLib) mkRocmTarget;
      };
      inherit (therockTargetConfig)
        defaultRocmGpuTargets
        defaultRocmTarget
        rocmTargets
        ;

      defaultTherockSources = {
        rocm = builtins.fromJSON (builtins.readFile ./pkgs/therock/sources/rocm.json);
        pythonWheels = builtins.fromJSON (builtins.readFile ./pkgs/therock/sources/python-wheels.json);
        rocmSource = builtins.fromJSON (builtins.readFile ./pkgs/therock/sources/rocm-source.json);
        rocmThirdParty = builtins.fromJSON (builtins.readFile ./pkgs/therock/sources/rocm-third-party.json);
      };

      mkTherockRocmOverlay =
        {
          rocmTargets,
          target ? builtins.head rocmTargets,
          sources ? defaultTherockSources,
        }:
        import ./overlays/therock-rocm.nix {
          inherit lib rocmTargets target;
          therockRocmSources = sources.rocm;
          therockPythonWheelSources = sources.pythonWheels;
          therockRocmSourceSources = sources.rocmSource;
          therockRocmThirdPartySources = sources.rocmThirdParty;
        };

      mkTherockPythonOverlay =
        { target }:
        import ./overlays/therock-python.nix {
          inherit lib target;
        };

      mkTherockOverlays =
        args:
        let
          target = args.target or (builtins.head args.rocmTargets);
          rocmOverlay = mkTherockRocmOverlay args;
          pythonOverlay = mkTherockPythonOverlay { inherit target; };
        in
        {
          rocm = rocmOverlay;
          python = pythonOverlay;
        };

      therockOverlays = mkTherockOverlays {
        inherit rocmTargets;
        target = defaultRocmTarget;
      };

      therockRocmOverlay = therockOverlays.rocm;
      therockPythonOverlay = therockOverlays.python;

      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ self.overlays.default ];
        };

      perSystem = f: forAllSystems (system: f (pkgsFor system));
      perLinuxSystem = f: forLinuxSystems (system: f (pkgsFor system));

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
        inherit
          defaultTherockSources
          mkFevmFaex9Configuration
          mkTherockOverlays
          mkTherockPythonOverlay
          mkTherockRocmOverlay
          ;
        inherit (rocmTargetLib)
          mkRocmNarrowOverlay
          mkRocmTarget
          ;
        therockTargets = therockTargetConfig;
      };

      overlays = {
        default =
          final: prev:
          lib.optionalAttrs prev.stdenv.isLinux (
            let
              ecPackages = prev.callPackage ./pkgs/ec-su-axb35.nix {
                ec-su-axb35-src = inputs.ec-su-axb35;
              };
              llamaCppTargetPackages = lib.listToAttrs (
                map (rocmTarget: {
                  name = "llama-cpp-rocm-${rocmTarget.packageSuffix}";
                  value = prev.llama-cpp.override {
                    rocmSupport = true;
                    rpcSupport = true;
                    inherit (final) rocmPackages;
                    inherit (rocmTarget) rocmGpuTargets;
                  };
                }) rocmTargets
              );
            in
            {
              # EC-SU_AXB35 packages
              ec-su-axb35 = ecPackages.kernelModule;
              ec-su-axb35-monitor = ecPackages.monitor;
              strix-halo-mes-firmware = prev.callPackage ./pkgs/strix-halo-mes-firmware.nix { };

              # Generic ROCm llama.cpp build; target narrowing is explicit below.
              llama-cpp-rocm = prev.llama-cpp.override {
                rocmSupport = true;
                rpcSupport = true;
                inherit (final) rocmPackages;
              };
            }
            // llamaCppTargetPackages
            // (therockRocmOverlay final prev)
          );

        rocm-narrow-gfx1151 = rocmTargetLib.mkRocmNarrowOverlay {
          rocmGpuTargets = defaultRocmGpuTargets;
        };

        therock-rocm = final: prev: lib.optionalAttrs prev.stdenv.isLinux (therockRocmOverlay final prev);
        therock-python =
          final: prev: lib.optionalAttrs prev.stdenv.isLinux (therockPythonOverlay final prev);
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
          genericPackages = {
            default = pkgs.llama-cpp;
            inherit (pkgs) llama-cpp;
          };

          linuxPackages =
            let
              therockPackageNamesFor =
                rocmTarget:
                let
                  s = rocmTarget.packageSuffix;
                in
                [
                  "therock-amd-llvm-${s}"
                  "therock-rocm-${s}"
                  "therock-rocm-${s}-env"
                  "therock-rocm-${s}-rocshmem-env"
                  "therock-rocm-${s}-core"
                  "therock-rocm-${s}-cmake"
                  "therock-rocm-from-source-${s}"
                  "therock-rocm-from-source-${s}-configure"
                  "therock-rocm-source-${s}"
                  "therock-rocm-source-${s}-compiler-stage"
                  "therock-rocm-third-party-mirror-${s}"
                  "therock-python-${s}"
                  "therock-python-wheels-${s}"
                  "therock-amdsmi-${s}"
                  "torch-rocm-${s}"
                ];
              therockPackageNames = lib.concatMap therockPackageNamesFor rocmTargets;
              requiredTherockPackageNames = therockPackageNamesFor defaultRocmTarget;
              missingRequiredTherockPackages = builtins.filter (
                name: !(builtins.hasAttr name pkgs)
              ) requiredTherockPackageNames;

              llamaCppTargetPackageNames = map (
                rocmTarget: "llama-cpp-rocm-${rocmTarget.packageSuffix}"
              ) rocmTargets;
              llamaCppTargetPackages = lib.genAttrs llamaCppTargetPackageNames (name: pkgs.${name});

              therockPackages =
                assert lib.assertMsg (missingRequiredTherockPackages == [ ])
                  "missing expected default TheRock package(s): ${lib.concatStringsSep ", " missingRequiredTherockPackages}";
                lib.genAttrs (builtins.filter (name: builtins.hasAttr name pkgs) therockPackageNames) (
                  name: pkgs.${name}
                );

              benchmarks = import ./bench/default.nix {
                inherit pkgs;
                packages = {
                  inherit (pkgs) llama-cpp-vulkan;
                  llama-cpp-rocm = pkgs.llama-cpp-rocm-gfx1151;
                };
              };
            in
            {
              inherit (pkgs)
                ec-su-axb35-monitor
                llama-cpp-rocm
                llama-cpp-vulkan
                strix-halo-mes-firmware
                ;
            }
            // llamaCppTargetPackages
            // therockPackages
            // (pkgs.lib.concatMapAttrs (
              model: benchs:
              pkgs.lib.mapAttrs' (name: drv: {
                name = "bench-${model}-${name}";
                value = drv;
              }) benchs
            ) benchmarks);
        in
        genericPackages // lib.optionalAttrs pkgs.stdenv.isLinux linuxPackages
      );

      apps = perSystem (
        pkgs:
        let
          s = defaultRocmTarget.packageSuffix;
          mkApp =
            {
              name,
              packageName ? name,
              binary ? name,
              description,
            }:
            {
              type = "app";
              program = "${builtins.getAttr packageName pkgs}/bin/${binary}";
              meta.description = description;
            };
          mkDirectApp =
            {
              package,
              binary,
              description,
            }:
            {
              type = "app";
              program = "${package}/bin/${binary}";
              meta.description = description;
            };
          mkTargetLlamaApp =
            {
              name,
              package,
              hsaOverride ? null,
              binary,
              description,
            }:
            {
              type = "app";
              program = toString (
                pkgs.writeShellScript name ''
                  ${lib.optionalString (
                    hsaOverride != null
                  ) "export HSA_OVERRIDE_GFX_VERSION=${lib.escapeShellArg hsaOverride}"}
                  ${package}/bin/${binary} "$@"
                ''
              );
              meta.description = description;
            };

          mkTargetLlamaApps =
            rocmTarget:
            let
              suffix = rocmTarget.packageSuffix;
              package = builtins.getAttr "llama-cpp-rocm-${suffix}" pkgs;
              hsaOverride = rocmTarget.hsaOverride or null;
            in
            {
              "llama-cli-${suffix}" = mkTargetLlamaApp {
                name = "llama-cli-${suffix}";
                inherit package hsaOverride;
                binary = "llama-cli";
                description = "Run llama-cli with ROCm narrowed for ${rocmTarget.description}";
              };

              "llama-server-${suffix}" = mkTargetLlamaApp {
                name = "llama-server-${suffix}";
                inherit package hsaOverride;
                binary = "llama-server";
                description = "Run llama-server with ROCm narrowed for ${rocmTarget.description}";
              };
            };

          genericApps = {
            llama-cli = mkDirectApp {
              package = pkgs.llama-cpp;
              binary = "llama-cli";
              description = "Run llama-cli";
            };

            llama-server = mkDirectApp {
              package = pkgs.llama-cpp;
              binary = "llama-server";
              description = "Run llama-server";
            };
          };

          linuxApps = {
            llama-cli-rocm = mkDirectApp {
              package = pkgs.llama-cpp-rocm;
              binary = "llama-cli";
              description = "Run llama-cli with generic ROCm support";
            };

            llama-server-rocm = mkDirectApp {
              package = pkgs.llama-cpp-rocm;
              binary = "llama-server";
              description = "Run llama-server with generic ROCm support";
            };
          }
          // (lib.foldl' lib.recursiveUpdate { } (map mkTargetLlamaApps rocmTargets))
          // lib.optionalAttrs (builtins.hasAttr "therock-rocm-${s}-env" pkgs) {
            "therock-rocm-${s}-env" = mkApp {
              name = "therock-rocm-${s}-env";
              description = "Run a command in the pinned TheRock ROCm ${s} environment";
            };
          }
          // lib.optionalAttrs (builtins.hasAttr "therock-rocm-${s}-rocshmem-env" pkgs) {
            "therock-rocm-${s}-rocshmem-env" = mkApp {
              name = "therock-rocm-${s}-rocshmem-env";
              description = "Run a command in the pinned TheRock ROCm ${s} rocSHMEM environment";
            };
          }
          // lib.optionalAttrs (builtins.hasAttr "therock-python-${s}" pkgs) {
            "therock-python-${s}" = mkApp {
              name = "therock-python-${s}";
              binary = "therock-python";
              description = "Run Python in the pinned TheRock ROCm/PyTorch wheel environment";
            };
            "therock-python-${s}-env" = mkApp {
              name = "therock-python-${s}-env";
              packageName = "therock-python-${s}";
              binary = "therock-python-env";
              description = "Run a command in the pinned TheRock ROCm/PyTorch wheel environment";
            };
          };
        in
        genericApps // lib.optionalAttrs pkgs.stdenv.isLinux linuxApps
      );

      devShells = perSystem (pkgs: {
        default = pkgs.mkShell {
          packages =
            (with pkgs; [
              deadnix
              nix-fast-build
              nixfmt-tree
              statix
            ])
            ++ lib.optionals pkgs.stdenv.isLinux [
              (pkgs.python3.withPackages (
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

      checks = perLinuxSystem (
        pkgs:
        let
          s = defaultRocmTarget.packageSuffix;
          therockPytorchPackage = "torch-rocm-${s}";

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

          "therock-pytorch-${s}" = pkgs.${therockPytorchPackage};
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
