{
  description = "Strix Halo NixOS configuration and tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    ec-su-axb35 = {
      url = "github:cmetz/ec-su_axb35-linux";
      flake = false;
    };

    llama-cpp-master = {
      url = "github:ggml-org/llama.cpp";
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

      rocmTargetLib = import ./lib/rocm-targets.nix;
      therockTargetConfig = import ./pkgs/therock/targets.nix {
        inherit (rocmTargetLib) mkRocmTarget;
      };
      inherit (therockTargetConfig)
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

      cudaPackagesFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ self.overlays.default ];
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaCapabilities = [ "8.9" ];
          };
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

      mkBenchmarkSuites =
        pkgs:
        let
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

          acceleratedBenchmarks = import ./bench/default.nix {
            inherit pkgs;
            tools = [
              {
                name = "rocm";
                package = pkgs.${"llama-cpp-rocm-${defaultRocmTarget.packageSuffix}"};
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
        flattenBenchmarks genericBenchmarks
        // lib.optionalAttrs pkgs.stdenv.isLinux (flattenBenchmarks acceleratedBenchmarks);

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
        benchmarks = import ./bench/lib.nix { inherit lib; };
        inherit (rocmTargetLib) mkRocmTarget;
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
              applyMasterSrc =
                attrName: pkg:
                pkg.overrideAttrs (old: {
                  pname = attrName;
                  version =
                    "master-" + (inputs.llama-cpp-master.shortRev or inputs.llama-cpp-master.rev or "unknown");
                  src = inputs.llama-cpp-master;
                  npmDeps = null;
                  nativeBuildInputs = prev.lib.filter (x: x.pname or "" != "npm-config-hook") (
                    old.nativeBuildInputs or [ ]
                  );
                  preConfigure = ''
                    prependToVar cmakeFlags "-DLLAMA_BUILD_COMMIT:STRING=${
                      inputs.llama-cpp-master.shortRev or inputs.llama-cpp-master.rev or "master"
                    }"
                  '';
                  cmakeFlags = (old.cmakeFlags or [ ]) ++ [
                    (prev.lib.cmakeBool "LLAMA_BUILD_UI" false)
                    (prev.lib.cmakeFeature "LLAMA_BUILD_NUMBER" "0")
                  ];
                });
              mkTargetedPackage =
                pname: pkg:
                pkg.overrideAttrs (_old: {
                  inherit pname;
                });
              mkLlamaCppRocmBase =
                rocmTarget:
                prev.llama-cpp.override {
                  rocmSupport = true;
                  rpcSupport = true;
                  inherit (final) rocmPackages;
                  inherit (rocmTarget) rocmGpuTargets;
                };
              llamaCppRocmTargetPackages = lib.listToAttrs (
                map (rocmTarget: {
                  name = "llama-cpp-rocm-${rocmTarget.packageSuffix}";
                  value = mkTargetedPackage "llama-cpp-rocm-${rocmTarget.packageSuffix}" (
                    mkLlamaCppRocmBase rocmTarget
                  );
                }) rocmTargets
              );
              llamaCppMasterRocmTargetPackages = lib.listToAttrs (
                map (rocmTarget: {
                  name = "llama-cpp-master-rocm-${rocmTarget.packageSuffix}";
                  value = applyMasterSrc "llama-cpp-master-rocm-${rocmTarget.packageSuffix}" (
                    mkLlamaCppRocmBase rocmTarget
                  );
                }) rocmTargets
              );
              llamaCppRocmBase = mkLlamaCppRocmBase defaultRocmTarget;
              llamaCppVulkanBase = prev.llama-cpp.override {
                vulkanSupport = true;
                rpcSupport = true;
              };
              llamaCppCudaBase = prev.llama-cpp.override {
                cudaSupport = true;
                rpcSupport = true;
              };
            in
            {
              # EC-SU_AXB35 packages
              ec-su-axb35 = ecPackages.kernelModule;
              ec-su-axb35-monitor = ecPackages.monitor;
              strix-halo-mes-firmware = prev.callPackage ./pkgs/strix-halo-mes-firmware.nix { };

              # Generic ROCm llama.cpp build; target narrowing is explicit below.
              llama-cpp-rocm = llamaCppRocmBase;
              llama-cpp-vulkan = llamaCppVulkanBase;
              llama-cpp-cuda = llamaCppCudaBase;
              llama-cpp-master-rocm = applyMasterSrc "llama-cpp-master-rocm" llamaCppRocmBase;
              llama-cpp-master-vulkan = applyMasterSrc "llama-cpp-master-vulkan" llamaCppVulkanBase;
              llama-cpp-master-cuda = applyMasterSrc "llama-cpp-master-cuda" llamaCppCudaBase;
            }
            // llamaCppRocmTargetPackages
            // llamaCppMasterRocmTargetPackages
            // (therockRocmOverlay final prev)
          );

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
        benchmark-executor = import ./modules/benchmark-executor.nix;
        benchmark-runner = import ./modules/benchmark-runner.nix;
        ec-su-axb35 = import ./modules/ec-su-axb35.nix;
        ryzenadj = import ./modules/ryzenadj.nix;
        disko-raid0 = import ./modules/disko-raid0.nix;
        tuning = import ./modules/tuning.nix;
      };

      nixosConfigurations = {
        fevm-faex9 = mkFevmFaex9Configuration { };
      };

      darwinModules = {
        benchmark-executor = import ./modules/benchmark-executor.nix;
      };
    }
    // {
      benchmarks = perSystem mkBenchmarkSuites;

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
              llamaCppMasterTargetPackageNames = map (
                rocmTarget: "llama-cpp-master-rocm-${rocmTarget.packageSuffix}"
              ) rocmTargets;
              llamaCppTargetPackages = lib.genAttrs (
                llamaCppTargetPackageNames ++ llamaCppMasterTargetPackageNames
              ) (name: pkgs.${name});
              cudaPkgs = cudaPackagesFor pkgs.stdenv.hostPlatform.system;

              therockPackages =
                assert lib.assertMsg (missingRequiredTherockPackages == [ ])
                  "missing expected default TheRock package(s): ${lib.concatStringsSep ", " missingRequiredTherockPackages}";
                lib.genAttrs (builtins.filter (name: builtins.hasAttr name pkgs) therockPackageNames) (
                  name: pkgs.${name}
                );

            in
            {
              inherit (pkgs)
                ec-su-axb35-monitor
                llama-cpp-rocm
                llama-cpp-vulkan
                llama-cpp-master-rocm
                llama-cpp-master-vulkan
                strix-halo-mes-firmware
                ;
              inherit (cudaPkgs) llama-cpp-cuda llama-cpp-master-cuda;
            }
            // llamaCppTargetPackages
            // therockPackages;
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
          benchmarkSet = self.benchmarks.${system};
          cpuBenchmark = benchmarkSet.bench-llama2-7b-llama-cpp-cpu-b512-fa1;
          cpuBenchmarkMetadata = builtins.toJSON cpuBenchmark.passthru.benchmark;

          benchmarkRunnerProfileConfig =
            (nixpkgs.lib.nixosSystem {
              system = "x86_64-linux";
              modules = [
                self.nixosModules.benchmark-runner
                (_: {
                  boot.kernelParams = [ "iommu=off" ];
                  benchmark.runners.ci = {
                    gpus = [
                      {
                        type = "amd";
                        arch = defaultRocmTarget.systemFeature;
                      }
                    ];
                    systemFeatures = [
                      defaultRocmTarget.systemFeature
                      "caller-feature"
                    ];
                    extraSandboxPaths = [ "/caller/device" ];
                  };
                })
              ];
            }).config;
          benchmarkRunnerProfileMetadata = builtins.toJSON {
            features = benchmarkRunnerProfileConfig.nix.settings.system-features;
            sandboxPaths = benchmarkRunnerProfileConfig.nix.settings.extra-sandbox-paths;
            udevRules = benchmarkRunnerProfileConfig.services.udev.extraRules;
          };

          benchmarkExecutorConfig =
            (nixpkgs.lib.nixosSystem {
              system = "x86_64-linux";
              modules = [
                self.nixosModules.benchmark-executor
                (_: {
                  benchmark.executor = {
                    enable = true;
                    builders.gpu-builder = {
                      hostName = "gpu-builder.example";
                      sshUser = "builder";
                      systemFeatures = [ "rocm" ];
                      gpus = [
                        {
                          type = "amd";
                          arch = "1151";
                        }
                      ];
                      publicKey = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBenchmarkExecutorCheck";
                    };
                  };
                })
              ];
            }).config;
          benchmarkExecutorMetadata = builtins.toJSON {
            buildMachines = benchmarkExecutorConfig.nix.buildMachines;
            knownHosts = benchmarkExecutorConfig.programs.ssh.knownHosts;
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
            treefmt --tree-root . --walk filesystem --fail-on-change
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

          benchmark-executor =
            pkgs.runCommandLocal "ci-benchmark-executor"
              {
                nativeBuildInputs = [ pkgs.jq ];
              }
              ''
                cat > executor.json <<'JSON'
                ${benchmarkExecutorMetadata}
                JSON

                jq -e '
                  (.buildMachines | length) == 1
                  and .buildMachines[0].hostName == "gpu-builder.example"
                  and .buildMachines[0].sshUser == "builder"
                  and (.buildMachines[0].supportedFeatures | index("rocm") != null)
                  and (.buildMachines[0].supportedFeatures | index("gfx1151") != null)
                  and .knownHosts."gpu-builder.example".publicKey == "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBenchmarkExecutorCheck"
                ' executor.json

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
                  and (.sandboxPaths | index("/dev/accel") == null)
                  and (.sandboxPaths | index("/caller/device") != null)
                  and (.udevRules | contains("KERNEL==\"kfd\""))
                ' profile.json

                touch "$out"
              '';

          "package-llama-cpp-rocm-${s}" = pkgs."llama-cpp-rocm-${s}";
          package-strix-halo-mes-firmware = pkgs.strix-halo-mes-firmware;
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
