{
  description = "Strix Halo NixOS configuration and tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-vllm.url = "github:CertainLach/nixpkgs/push-lklxouywkrnv";

    vllm-src = {
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

    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    ec-su-axb35 = {
      url = "github:cmetz/ec-su_axb35-linux";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-vllm,
      ...
    }@inputs:
    let
      inherit (nixpkgs) lib;
      systems = [ "x86_64-linux" ];
      forAllSystems = lib.genAttrs systems;
      versionFromReleaseRef = ref: lib.removePrefix "v" (lib.last (lib.splitString "/" ref));
      # GitHub flake locks retain the resolved commit for branch inputs, but not
      # always the original branch name. Keep this fallback aligned with
      # inputs.vllm-src.url.
      vllmLib = import ./lib/vllm.nix {
        inherit lib nixpkgs-vllm;
        vllmVersion = versionFromReleaseRef (inputs.vllm-src.sourceInfo.ref or "releases/v0.20.2");
        vllmSources = {
          vllm = inputs.vllm-src;
          cutlass = inputs.vllm-cutlass-src;
          flash-attn = inputs.vllm-flash-attn-src;
          flashmla = inputs.vllm-flashmla-src;
          flashmla-cutlass = inputs.vllm-flashmla-cutlass-src;
        };
      };

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

      perSystem = f: forAllSystems (system: f system (pkgsFor system));
    in
    {
      overlays = {
        default =
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

        vllm-cpu = vllmLib.mkVllmOverlay {
          hardware = vllmLib.hardwareProfiles.none;
        };

        vllm-rocm-gfx1151 = lib.composeManyExtensions [
          (vllmLib.mkRocmOverlay {
            hardware = vllmLib.hardwareProfiles.gfx1151;
          })
          (vllmLib.mkVllmOverlay {
            hardware = vllmLib.hardwareProfiles.gfx1151;
          })
        ];

        vllm-cuda-rtx4090 = lib.composeManyExtensions [
          (vllmLib.mkCudaOverlay {
            hardware = vllmLib.hardwareProfiles.rtx4090;
          })
          (vllmLib.mkVllmOverlay {
            hardware = vllmLib.hardwareProfiles.rtx4090;
          })
        ];

        rocm-gfx1151 = vllmLib.mkRocmOverlay {
          hardware = vllmLib.hardwareProfiles.gfx1151;
        };

        cuda-rtx4090 = vllmLib.mkCudaOverlay {
          hardware = vllmLib.hardwareProfiles.rtx4090;
        };

        cpu-tuned-zen5 = vllmLib.mkCpuTuningOverlay { };
      };

      lib = vllmLib;

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
    // {
      packages = perSystem (
        system: pkgs:
        let
          benchmarks = import ./bench/default.nix {
            inherit pkgs;
            packages = {
              inherit (pkgs) llama-cpp-rocm llama-cpp-vulkan;
            };
          };
          vllmCpuPkgs = vllmLib.mkPackageSet {
            inherit system;
            hardware = vllmLib.hardwareProfiles.none;
          };
          vllmCpuZen5Pkgs = vllmLib.mkPackageSet {
            inherit system;
            hardware = vllmLib.hardwareProfiles.none;
            cpu = "znver5";
          };
          vllmRocmPkgs = vllmLib.mkPackageSet {
            inherit system;
            hardware = vllmLib.hardwareProfiles.gfx1151;
          };
          vllmRocmZen5Pkgs = vllmLib.mkPackageSet {
            inherit system;
            hardware = vllmLib.hardwareProfiles.gfx1151;
            cpu = "znver5";
          };
          vllmCudaPkgs = vllmLib.mkPackageSet {
            inherit system;
            hardware = vllmLib.hardwareProfiles.rtx4090;
          };
          vllmCudaZen5Pkgs = vllmLib.mkPackageSet {
            inherit system;
            hardware = vllmLib.hardwareProfiles.rtx4090;
            cpu = "znver5";
          };
        in
        {
          default = pkgs.llama-cpp-rocm;
          inherit (pkgs)
            ec-su-axb35-monitor
            llama-cpp-rocm
            llama-cpp-vulkan
            ;

          vllm-cpu = vllmLib.mkVllmPackage {
            pkgs = vllmCpuPkgs;
            hardware = vllmLib.hardwareProfiles.none;
          };
          vllm-env-cpu = vllmLib.mkVllmEnv {
            pkgs = vllmCpuPkgs;
            hardware = vllmLib.hardwareProfiles.none;
          };
          vllm-cpu-zen5 = vllmLib.mkVllmPackage {
            pkgs = vllmCpuZen5Pkgs;
            hardware = vllmLib.hardwareProfiles.none;
            tunePackage = true;
          };
          vllm-env-cpu-zen5 = vllmLib.mkVllmEnv {
            pkgs = vllmCpuZen5Pkgs;
            hardware = vllmLib.hardwareProfiles.none;
            tunePackage = true;
          };

          vllm-rocm-gfx1151 = vllmLib.mkVllmPackage {
            pkgs = vllmRocmPkgs;
            hardware = vllmLib.hardwareProfiles.gfx1151;
          };
          vllm-env-rocm-gfx1151 = vllmLib.mkVllmEnv {
            pkgs = vllmRocmPkgs;
            hardware = vllmLib.hardwareProfiles.gfx1151;
            name = "vllm-env-rocm-gfx1151";
          };
          vllm-rocm-gfx1151-zen5 = vllmLib.mkVllmPackage {
            pkgs = vllmRocmZen5Pkgs;
            hardware = vllmLib.hardwareProfiles.gfx1151;
            tunePackage = true;
          };
          vllm-env-rocm-gfx1151-zen5 = vllmLib.mkVllmEnv {
            pkgs = vllmRocmZen5Pkgs;
            hardware = vllmLib.hardwareProfiles.gfx1151;
            tunePackage = true;
            name = "vllm-env-rocm-gfx1151-zen5";
          };

          vllm-cuda-rtx4090 = vllmLib.mkVllmPackage {
            pkgs = vllmCudaPkgs;
            hardware = vllmLib.hardwareProfiles.rtx4090;
          };
          vllm-env-cuda-rtx4090 = vllmLib.mkVllmEnv {
            pkgs = vllmCudaPkgs;
            hardware = vllmLib.hardwareProfiles.rtx4090;
          };
          vllm-cuda-rtx4090-zen5 = vllmLib.mkVllmPackage {
            pkgs = vllmCudaZen5Pkgs;
            hardware = vllmLib.hardwareProfiles.rtx4090;
            tunePackage = true;
          };
          vllm-env-cuda-rtx4090-zen5 = vllmLib.mkVllmEnv {
            pkgs = vllmCudaZen5Pkgs;
            hardware = vllmLib.hardwareProfiles.rtx4090;
            tunePackage = true;
          };
        }
        // (pkgs.lib.concatMapAttrs (
          model: benchs:
          pkgs.lib.mapAttrs' (name: drv: {
            name = "bench-${model}-${name}";
            value = drv;
          }) benchs
        ) benchmarks)
      );

      apps = perSystem (
        _: pkgs: {
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
        _: pkgs: {
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

      formatter = perSystem (_: pkgs: pkgs.nixfmt-tree);

      checks = perSystem (
        _: pkgs:
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
            treefmt --fail-on-change
          '';

          deadnix = mkSourceCheck "deadnix" [ pkgs.deadnix ] ''
            deadnix --fail .
          '';

          statix = mkSourceCheck "statix" [ pkgs.statix ] ''
            statix check .
          '';
        }
      );
    };
}
