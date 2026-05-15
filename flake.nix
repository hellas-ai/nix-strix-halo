{
  description = "Strix Halo NixOS configuration and tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    vllm = {
      url = "github:vllm-project/vllm/releases/v0.20.2";
      flake = false;
    };

    vllm-cutlass-src = {
      url = "github:NVIDIA/cutlass/v4.4.2";
      flake = false;
    };

    vllm-flash-attn-src = {
      url = "github:vllm-project/flash-attention/f5bc33cfc02c744d24a2e9d50e6db656de40611c";
      flake = false;
    };

    vllm-flashmla-src = {
      url = "github:vllm-project/FlashMLA/a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b";
      flake = false;
    };

    vllm-flashmla-cutlass-src = {
      url = "github:NVIDIA/cutlass/147f5673d0c1c3dcf66f78d677fd647e4a020219";
      flake = false;
    };

    vllm-triton-kernels-src = {
      url = "github:triton-lang/triton/v3.6.0";
      flake = false;
    };

    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    ec-su-axb35 = {
      url = "github:cmetz/ec-su_axb35-linux";
      flake = false;
    };

    # libibverbs-usb4 is the lower-layer RDMA stack (kernel patches,
    # rdma-core-usb4, rixl, lmcache) plus the canonical vllm wrapper
    # exposed as `overlays.vllm` / `overlays.vllm-rdma`. We compose its
    # vllm-rdma overlay into our own overlays.default below so strix
    # machines pick up an RDMA-enabled vllm by default.
    usb4-rdma = {
      url = "git+ssh://trex/home/grw/src/linux-libibverbs-usb4";
      inputs.nixpkgs.follows = "nixpkgs";
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

      versionFromReleaseRef = ref: lib.removePrefix "v" (lib.last (lib.splitString "/" ref));

      vllmLib = import ./lib/vllm.nix {
        inherit lib nixpkgs;
        vllmVersion = versionFromReleaseRef (inputs.vllm.sourceInfo.ref or "releases/v0.20.2");
        vllmSources = {
          vllm = inputs.vllm;
          cutlass = inputs.vllm-cutlass-src;
          flash-attn = inputs.vllm-flash-attn-src;
          flashmla = inputs.vllm-flashmla-src;
          flashmla-cutlass = inputs.vllm-flashmla-cutlass-src;
          triton-kernels = inputs.vllm-triton-kernels-src;
        };
      };

      rocmTargets = [
        "gfx1151"
      ];

      mkTunedOverlay =
        final: if builtins.isString final then vllmLib.mkCpuTuningOverlay { cpu = final; } else _: { };

      strixAdditionsOverlay =
        final: prev:
        let
          ecPackages = prev.callPackage ./pkgs/ec-su-axb35.nix {
            ec-su-axb35-src = inputs.ec-su-axb35;
          };
        in
        {
          ec-su-axb35 = ecPackages.kernelModule;
          ec-su-axb35-monitor = ecPackages.monitor;

          llama-cpp-rocm = prev.llama-cpp.override {
            rocmSupport = true;
            rpcSupport = true;
            rocmGpuTargets = rocmTargets;
          };
        };

      overlays = {
        # Composed default: pulls libibverbs's RDMA-enabled vllm overlay
        # (wraps stock python3Packages.vllm + injects rixl + lmcache) and
        # layers our own strix additions (ec-su, llama-cpp-rocm) on top.
        # Consumers (nixos-config strix machines) get a vllm-rdma-ready
        # pkgs set with one overlay import. Doesn't include rocm-narrow —
        # that's still opt-in via overlays.rocm-narrow below since it
        # invalidates downstream caches.
        default = nixpkgs.lib.composeManyExtensions [
          inputs.usb4-rdma.overlays.vllm-rdma
          strixAdditionsOverlay
        ];

        # Narrows rocmPackages.clr to `rocmTargets` (currently gfx1151) for
        # faster builds on strix-halo. Applying this overlay invalidates
        # downstream caches and removes support for other GPUs from the system
        # OpenCL ICD — only enable on hosts that exclusively run gfx1151.
        rocm-narrow = _: prev: {
          rocmPackages = prev.rocmPackages.overrideScope (
            _: rocmPrev: {
              clr = rocmPrev.clr.override {
                localGpuTargets = rocmTargets;
              };
            }
          );
        };

        inherit mkTunedOverlay;

        tuned = vllmLib.mkCpuTuningOverlay {
          cpu = "znver5";
        };
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

      vllmTargets = [
        {
          name = "cpu";
          hardware = vllmLib.hardwareProfiles.none;
        }
        {
          name = "rocm-gfx1151";
          hardware = vllmLib.hardwareProfiles.gfx1151;
        }
        {
          name = "cuda-rtx4090";
          hardware = vllmLib.hardwareProfiles.rtx4090;
        }
      ];

      mkVllmAliases =
        {
          system,
          cpu ? null,
          suffix ? "",
        }:
        let
          tunePackage = cpu != null;
          mkTargetAliases =
            {
              name,
              hardware,
            }:
            let
              packageSet = vllmLib.mkPackageSet (
                {
                  inherit system hardware;
                }
                // lib.optionalAttrs (cpu != null) {
                  inherit cpu;
                }
              );
              packageName = "vllm-${name}${suffix}";
              envName = "vllm-env-${name}${suffix}";
            in
            {
              "${packageName}" = vllmLib.mkVllmPackage {
                pkgs = packageSet;
                inherit hardware tunePackage;
              };

              "${envName}" = vllmLib.mkVllmEnv {
                pkgs = packageSet;
                inherit hardware tunePackage;
                name = envName;
              };
            };
        in
        lib.foldl' (aliases: target: aliases // mkTargetAliases target) { } vllmTargets;

      mkBenchmarkPackages =
        pkgs:
        let
          benchmarks = import ./bench/default.nix {
            inherit pkgs;
            packages = {
              inherit (pkgs) llama-cpp-rocm llama-cpp-vulkan;
            };
          };
        in
        pkgs.lib.concatMapAttrs (
          model: benchs:
          pkgs.lib.mapAttrs' (name: drv: {
            name = "bench-${model}-${name}";
            value = drv;
          }) benchs
        ) benchmarks;

      mkSourceCheck =
        pkgs: name: nativeBuildInputs: command:
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
      inherit overlays;

      lib = vllmLib;

      nixosModules = {
        default = _: {
          nixpkgs.overlays = [
            self.overlays.default
          ];
        };
        rocm-narrow = _: {
          nixpkgs.overlays = [
            self.overlays.rocm-narrow
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
        fevm-faex9 = nixpkgs.lib.nixosSystem {
          system = "x86_64-linux";
          specialArgs = {
            inherit inputs self;
          };
          modules = [
            inputs.disko.nixosModules.disko
            ./examples/fevm-faex9/configuration.nix
          ];
        };
      };

      legacyPackages = perSystem (
        {
          system,
          pkgs,
          zen5Packages,
        }:
        {
          defaultPackages = pkgs // mkVllmAliases { inherit system; };
          zen5Packages =
            zen5Packages
            // mkVllmAliases {
              inherit system;
              cpu = "znver5";
            };
        }
      );

      packages = perSystem (
        {
          system,
          pkgs,
          zen5Packages,
        }:
        {
          default = pkgs.llama-cpp-rocm;
          inherit (pkgs)
            ec-su-axb35-monitor
            llama-cpp-rocm
            llama-cpp-vulkan
            ;

          llama-cpp-rocm-zen5 = zen5Packages.llama-cpp-rocm;
        }
        // mkVllmAliases { inherit system; }
        // mkVllmAliases {
          inherit system;
          cpu = "znver5";
          suffix = "-zen5";
        }
        // mkBenchmarkPackages pkgs
      );

      apps = perSystem (
        { pkgs, ... }:
        {
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
        }
      );

      devShells = perSystem (
        { pkgs, ... }:
        {
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
        }
      );

      formatter = perSystem ({ pkgs, ... }: pkgs.nixfmt-tree);

      checks = perSystem (
        { pkgs, ... }:
        {
          format = mkSourceCheck pkgs "format" [ pkgs.nixfmt-tree ] ''
            treefmt --fail-on-change
          '';

          deadnix = mkSourceCheck pkgs "deadnix" [ pkgs.deadnix ] ''
            deadnix --fail .
          '';

          statix = mkSourceCheck pkgs "statix" [ pkgs.statix ] ''
            statix check .
          '';
        }
      );
    };
}
