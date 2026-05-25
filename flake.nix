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
      flattenBenchmarks =
        benchmarks:
        lib.concatMapAttrs (
          model: benchs:
          lib.mapAttrs' (name: drv: {
            name = "bench-${model}-${name}";
            value = drv;
          }) benchs
        ) benchmarks;

      mkFevmFaex9Configuration =
        {
          diskoModule,
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
            diskoModule
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
        benchmarks = import ./bench/lib.nix { inherit lib; };
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
    }
    // {
      packages = perSystem (
        pkgs:
        let
          genericBenchmarks = import ./bench/default.nix {
            inherit pkgs;
            tools = [
              {
                name = "cpu";
                package = pkgs.llama-cpp;
                executable = "llama-bench";
                backend = "cpu";
                metadata = {
                  accelerator = "cpu";
                };
              }
            ];
          };

          genericPackages = {
            default = pkgs.llama-cpp;
            inherit (pkgs) llama-cpp;
          }
          // flattenBenchmarks genericBenchmarks;

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

              defaultTargetMetadata = {
                inherit (defaultRocmTarget)
                  packageSuffix
                  rocmGpuTargets
                  runtimeArch
                  systemFeature
                  ;
              }
              // lib.optionalAttrs (defaultRocmTarget.hsaOverride != null) {
                inherit (defaultRocmTarget) hsaOverride;
              };

              acceleratedBenchmarks = import ./bench/default.nix {
                inherit pkgs;
                tools = [
                  {
                    name = "rocm";
                    package = pkgs.llama-cpp-rocm-gfx1151;
                    executable = "llama-bench";
                    backend = "rocm";
                    env = lib.optionalAttrs (defaultRocmTarget.hsaOverride != null) {
                      HSA_OVERRIDE_GFX_VERSION = defaultRocmTarget.hsaOverride;
                    };
                    requirements = {
                      systemFeatures = [ defaultRocmTarget.systemFeature ];
                      hostProfiles = [ "linux-amd-kfd" ];
                    };
                    metadata = {
                      accelerator = "rocm";
                      target = defaultTargetMetadata;
                    };
                  }
                  {
                    name = "vulkan";
                    package = pkgs.llama-cpp-vulkan;
                    executable = "llama-bench";
                    backend = "vulkan";
                    requirements = {
                      systemFeatures = [ defaultRocmTarget.systemFeature ];
                      hostProfiles = [ "linux-drm-render" ];
                    };
                    metadata = {
                      accelerator = "vulkan";
                      target = defaultTargetMetadata;
                    };
                  }
                ];
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
            // flattenBenchmarks acceleratedBenchmarks;
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

      checks = perSystem (
        pkgs:
        let
          system = pkgs.stdenv.hostPlatform.system;
          s = defaultRocmTarget.packageSuffix;
          therockPytorchPackage = "torch-rocm-${s}";
          packageSet = self.packages.${system};
          cpuBenchmark = packageSet.bench-llama2-7b-llama-cpp-cpu-b512-fa1;
          cpuBenchmarkMetadata = builtins.toJSON cpuBenchmark.passthru.benchmark;

          benchmarkRunnerProfileConfig =
            (nixpkgs.lib.nixosSystem {
              system = "x86_64-linux";
              modules = [
                self.nixosModules.benchmark-runner
                (_: {
                  services.benchmark-runner = {
                    enable = true;
                    systemFeatures = [ defaultRocmTarget.systemFeature ];
                    enabledProfiles = [
                      "linux-amd-kfd"
                      "caller-profile"
                    ];
                    profiles.caller-profile = {
                      systemFeatures = [ "caller-feature" ];
                      sandboxPaths = [ "/caller/device" ];
                      udevRules = [
                        ''KERNEL=="caller-device", GROUP="nixbld", MODE="0660"''
                      ];
                    };
                  };
                })
              ];
            }).config;
          benchmarkRunnerProfileMetadata = builtins.toJSON {
            features = benchmarkRunnerProfileConfig.nix.settings.system-features;
            sandboxPaths = benchmarkRunnerProfileConfig.nix.settings.extra-sandbox-paths;
            udevRules = benchmarkRunnerProfileConfig.services.udev.extraRules;
          };

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
                cp -R "$src" source
                chmod -R u+w source
                cd source
                ${command}
                touch "$out"
              '';
        in
        {
          format = mkSourceCheck "format" [ pkgs.nixfmt-tree ] ''
            treefmt --ci --tree-root . --walk filesystem .
          '';

          deadnix = mkSourceCheck "deadnix" [ pkgs.deadnix ] ''
            deadnix --fail .
          '';

          statix = mkSourceCheck "statix" [ pkgs.statix ] ''
            statix check .
          '';

          benchmark-metadata =
            pkgs.runCommandLocal "ci-benchmark-metadata"
              {
                nativeBuildInputs = [ pkgs.jq ];
              }
              ''
                cat > metadata.json <<'JSON'
                ${cpuBenchmarkMetadata}
                JSON

                jq -e '
                  .kind == "llama-cpp"
                  and .accelerator == "cpu"
                  and .requirements.systemFeatures == []
                  and .requirements.hostProfiles == []
                  and .requirements.sandboxPaths == []
                  and .model.path == "/models/llama-2-7b/llama-2-7b.Q4_K_M.gguf"
                  and .packages == [ "llama-cpp" ]
                  and .params.batch == 512
                  and .params.fa == 1
                  and (.command | length) >= 6
                ' metadata.json

                touch "$out"
              '';

          package-llama-cpp = pkgs.llama-cpp;
        }
        // lib.optionalAttrs pkgs.stdenv.isLinux {
          benchmark-runner-profiles =
            pkgs.runCommandLocal "ci-benchmark-runner-profiles"
              {
                nativeBuildInputs = [ pkgs.jq ];
              }
              ''
                cat > profile.json <<'JSON'
                ${benchmarkRunnerProfileMetadata}
                JSON

                jq -e '
                  (.features | index("${defaultRocmTarget.systemFeature}") != null)
                  and (.features | index("caller-feature") != null)
                  and (.sandboxPaths | index("/dev/dri") != null)
                  and (.sandboxPaths | index("/dev/kfd") != null)
                  and (.sandboxPaths | index("/caller/device") != null)
                  and (.udevRules | contains("caller-device"))
                ' profile.json

                touch "$out"
              '';

          "package-llama-cpp-rocm-${s}" = pkgs."llama-cpp-rocm-${s}";
          package-strix-halo-mes-firmware = pkgs.strix-halo-mes-firmware;
          "therock-pytorch-${s}" = pkgs.${therockPytorchPackage};
        }
      );
    };
}
