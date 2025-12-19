{
  description = "Llama.cpp with pre-built ROCm binaries from TheRock";

  inputs = {
    # nixpkgs.url = "github:LunNova/nixpkgs/lunnova/rocm-6.4.x";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    chaotic.url = "github:chaotic-cx/nyx/nyxpkgs-unstable";

    flake-utils.url = "github:numtide/flake-utils";
    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    llama-cpp = {
      url = "github:ggerganov/llama.cpp";
      flake = false;
    };
    rocwmma = {
      url = "github:ROCm/rocWMMA";
      flake = false;
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
    llama-cpp,
    rocwmma,
    ...
  } @ inputs: let
    rocmSources = builtins.fromJSON (builtins.readFile ./rocm-sources.json);

    # Define ROCm targets we support
    targets = ["gfx1151"];
  in
    # Overlay output
    {
      overlays.default = final: prev: let
        # Helper to build ROCm for a target
        mkRocm = target:
          prev.callPackage ./pkgs/rocm7-bin.nix {
            inherit rocmSources target;
          };

        # Helper to build clang wrapper for a rocm package
        mkClangWrapper = rocm:
          prev.callPackage ./pkgs/rocm-clang-wrapper.nix {
            inherit rocm;
          };

        # Helper to build rocwmma for a target
        mkRocwmma = target: rocm: rocmClangWrapper:
          final.callPackage ./pkgs/rocwmma.nix {
            inherit rocwmma rocm rocmClangWrapper;
            targets = [target];
          };

        # Helper to build llama-cpp variants from source
        mkLlamaCpp = args:
          prev.callPackage ./pkgs/llamacpp-rocm.nix ({
              inherit llama-cpp;
            }
            // args);

        # Build packages for each target
        targetPackages = builtins.listToAttrs (
          builtins.concatMap (target: let
            rocm = mkRocm target;
            clangWrapper = mkClangWrapper rocm;
            rocwmmaPkg = mkRocwmma target rocm clangWrapper;
          in [
            {
              name = "rocm7-bin-${target}";
              value = rocm;
            }
            {
              name = "rocm-clang-wrapper-${target}";
              value = clangWrapper;
            }
            {
              name = "rocwmma-${target}";
              value = rocwmmaPkg;
            }
          ])
          targets
        );

        # Build llama-cpp packages
        llamacppPackages = builtins.listToAttrs (
          builtins.concatMap (target: let
            rocm = targetPackages."rocm7-bin-${target}";
            clangWrapper = targetPackages."rocm-clang-wrapper-${target}";
            rocwmmaPkg = targetPackages."rocwmma-${target}";
          in [
            {
              name = target;
              value = mkLlamaCpp {
                inherit target rocm;
                rocmClangWrapper = clangWrapper;
              };
            }
            {
              name = "${target}-rocwmma";
              value = mkLlamaCpp {
                inherit target rocm;
                rocmClangWrapper = clangWrapper;
                rocwmma = rocwmmaPkg;
                enableRocwmma = true;
                enableHipBlasLt = true;
              };
            }
          ])
          targets
        );

        # EC-SU_AXB35 packages
        ecPackages = prev.callPackage ./pkgs/ec-su-axb35.nix {
          ec-su-axb35-src = inputs.ec-su-axb35;
        };
      in
        targetPackages
        // {
          # EC-SU_AXB35 packages
          ec-su-axb35 = ecPackages.kernelModule;
          ec-su-axb35-monitor = ecPackages.monitor;

          # Monitoring tools
          strixtop = prev.callPackage ./pkgs/strixtop.nix {
            ec-su-axb35-monitor = ecPackages.monitor;
          };

          # Updated shaderc for Vulkan support
          shaderc = prev.callPackage ./pkgs/shaderc.nix {};

          # Build our gpu targets for nixos rocm
          rocmPackages = prev.rocmPackages.overrideScope (
            rocmFinal: rocmPrev: {
              clr = rocmPrev.clr.override {
                localGpuTargets = targets;
              };
            }
          );

          # Llama.cpp ROCm packages under llamacpp-rocm namespace
          llamacpp-rocm = llamacppPackages;

          # Binary packages for each target
          llamacpp-rocm-bin-gfx1151 = prev.callPackage ./pkgs/llamacpp-rocm-bin.nix {
            gfxTarget = "gfx1151";
          };

          # Ryzenadj curve optimizer test script
          ryzenadj-co-test = prev.callPackage ./pkgs/ryzenadj-co-test {};
        };

      # NixOS modules
      nixosModules = {
        default = {
          config,
          pkgs,
          ...
        }: {
          imports = [
            inputs.chaotic.nixosModules.default
            inputs.disko.nixosModules.disko
          ];
          nixpkgs.overlays = [
            self.overlays.default
            inputs.chaotic.overlays.default
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
            ./examples/fevm-faex9/configuration.nix
          ];
        };
      };
    }
    //
    # Per-system outputs (ROCm only works on Linux)
    flake-utils.lib.eachSystem ["x86_64-linux"] (
      system: let
        # Apply our overlay to get custom packages
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            self.overlays.default
            inputs.chaotic.overlays.default
          ];
        };

        # Helper functions using overlay packages
        mkVulkanWrapper = {
          driver,
          driverName,
        }: let
          vulkanBase =
            (pkgs.llama-cpp.override {
              vulkanSupport = true;
              rpcSupport = true;
            }).overrideAttrs (old: {
              pname = "llama-cpp-vulkan";
            });
        in
          pkgs.runCommand "llama-cpp-vulkan-${driverName}" {
            buildInputs = [pkgs.makeWrapper];
            passthru.pname = "llama-cpp-vulkan-${driverName}";
          } ''
            mkdir -p $out/bin
            makeWrapper ${vulkanBase}/bin/llama-bench $out/bin/llama-bench \
              --set VK_ICD_FILENAMES "${driver}"
            makeWrapper ${vulkanBase}/bin/llama-cli $out/bin/llama-cli \
              --set VK_ICD_FILENAMES "${driver}"
            makeWrapper ${vulkanBase}/bin/llama-server $out/bin/llama-server \
              --set VK_ICD_FILENAMES "${driver}"
          '';
      in {
        packages = let
          # Build matrix for each target - reuse overlay packages
          buildForTarget = target: {
            # ROCm toolchain (from overlay)
            "rocm-${target}" = pkgs."rocm7-bin-${target}";

            # ROCm clang wrapper (from overlay)
            "rocm-clang-wrapper-${target}" = pkgs."rocm-clang-wrapper-${target}";

            # Standard build from source (from overlay)
            "llamacpp-rocm-${target}" = pkgs.llamacpp-rocm.${target};

            # ROCWMMA-optimized build from source (from overlay)
            "llamacpp-rocm-${target}-rocwmma" = pkgs.llamacpp-rocm."${target}-rocwmma";

            # Binary package
            "llamacpp-rocm-bin-${target}" = pkgs."llamacpp-rocm-bin-${target}";
          };

          # Merge all target builds into one attrset
          allPackages =
            pkgs.lib.foldl' (
              acc: target:
                acc // buildForTarget target
            ) {}
            targets;

          # Vulkan builds
          vulkanPackages = {
            "llama-cpp-vulkan" =
              (pkgs.llama-cpp.override {
                vulkanSupport = true;
                rpcSupport = true;
              }).overrideAttrs (old: {
                pname = "llama-cpp-vulkan";
              });
            "llama-cpp-vulkan-radv" = mkVulkanWrapper {
              driver = "${pkgs.mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json";
              driverName = "radv";
            };
            "llama-cpp-vulkan-amdvlk" = mkVulkanWrapper {
              driver = "${pkgs.amdvlk}/share/vulkan/icd.d/amd_icd64.json";
              driverName = "amdvlk";
            };
          };
        in
          allPackages
          // vulkanPackages
          // {
            # Default package
            default = allPackages.llamacpp-rocm-gfx1151;

            # EC-SU_AXB35 monitor tool (from overlay)
            ec-su-axb35-monitor = pkgs.ec-su-axb35-monitor;
          };

        # Benchmarks as separate output
        benchmarks = import ./bench/default.nix {
          inherit pkgs;
          packages = self.packages.${system};
        };

        # Apps for easy running
        apps = {
          llama-cli = {
            type = "app";
            program = toString (pkgs.writeShellScript "llama-cli" ''
              export HSA_OVERRIDE_GFX_VERSION=11.5.1
              ${self.packages.${system}.llamacpp-rocm-gfx1151-rocwmma}/bin/llama-cli "$@"
            '');
          };

          llama-server = {
            type = "app";
            program = toString (pkgs.writeShellScript "llama-server" ''
              export HSA_OVERRIDE_GFX_VERSION=11.5.1
              ${self.packages.${system}.llamacpp-rocm-gfx1151-rocwmma}/bin/llama-server "$@"
            '');
          };
        };
      }
    )
    // flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            self.overlays.default
            inputs.chaotic.overlays.default
          ];
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
