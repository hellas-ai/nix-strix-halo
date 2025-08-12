{
  description = "Llama.cpp with pre-built ROCm binaries from TheRock";

  inputs = {
    # nixpkgs.url = "github:LunNova/nixpkgs/lunnova/rocm-6.4.x";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
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
    llama-cpp,
    rocwmma,
    ec-su-axb35,
  }: let
    rocmSources = builtins.fromJSON (builtins.readFile ./rocm-sources.json);

    # Import package builders
    mkRocmDerivation = pkgs: import ./pkgs/rocm7-bin.nix {inherit pkgs rocmSources;};
    mkRocmClangWrapper = pkgs: import ./pkgs/rocm-clang-wrapper.nix {inherit pkgs rocmSources;};
    mkRocwmmaDerivation = pkgs: import ./pkgs/rocwmma.nix {inherit pkgs rocwmma;};
    mkLlamaCppDerivation = pkgs:
      import ./pkgs/llama-cpp.nix {
        inherit pkgs llama-cpp;
        mkRocwmmaDerivation = mkRocwmmaDerivation pkgs;
        mkRocmClangWrapper = mkRocmClangWrapper pkgs;
      };
    mkEcSuAxb35 = pkgs: import ./pkgs/ec-su-axb35.nix {inherit pkgs ec-su-axb35;};

    # Import benchmark runner
    mkBenchmark = pkgs: import ./bench/default.nix;
  in
    # Overlay output
    {
      overlays.default = final: prev: let
        # EC-SU_AXB35 packages
        ecPackages = mkEcSuAxb35 prev;
      in {
        # EC-SU_AXB35 packages
        ec-su-axb35 = ecPackages.ec-su-axb35;
        ec-su-axb35-monitor = ecPackages.ec-su-axb35-monitor;

        # Monitoring tools
        strixtop = prev.callPackage ./pkgs/strixtop.nix {};

        # Llama.cpp ROCm packages
        llamacpp-rocm = let
          mkRocm = mkRocmDerivation prev;
          mkLlamaCpp = mkLlamaCppDerivation prev;
          targets = ["gfx110X" "gfx1151" "gfx120X"];

          # Build overlay packages for each target
          buildOverlay = target: {
            # ROCm toolchain
            "rocm-${target}" = mkRocm {inherit target;};

            # Standard build (short name for convenience)
            "${target}" = mkLlamaCpp {
              inherit target;
              rocm = mkRocm {inherit target;};
            };

            # ROCWMMA-optimized build
            "${target}-rocwmma" = mkLlamaCpp {
              inherit target;
              rocm = mkRocm {inherit target;};
              enableRocwmma = true;
              enableHipBlasLt = true;
            };
          };
        in
          # Merge all overlays
          prev.lib.foldl' (
            acc: target:
              acc // buildOverlay target
          ) {}
          targets;
      };

      # NixOS modules
      nixosModules = {
        default = {
          nixpkgs.overlays = [self.overlays.default];
        };
        rpc-server = import ./modules/rpc-server.nix;
        benchmark-runner = import ./modules/benchmark-runner.nix;
        ec-su-axb35 = import ./modules/ec-su-axb35.nix;
      };
    }
    //
    # Per-system outputs (ROCm only works on Linux)
    flake-utils.lib.eachSystem ["x86_64-linux"] (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        mkRocm = mkRocmDerivation pkgs;
        mkClangWrapper = mkRocmClangWrapper pkgs;
        mkLlamaCpp = mkLlamaCppDerivation pkgs;
      in {
        packages = let
          # GPU targets we support
          targets = ["gfx110X" "gfx1151" "gfx120X"];

          # Build matrix for each target
          buildForTarget = target: {
            # ROCm toolchain
            "rocm-${target}" = mkRocm {inherit target;};

            # ROCm clang wrapper
            "rocm-clang-wrapper-${target}" = mkClangWrapper {inherit target;};

            # Standard build
            "llama-cpp-${target}" = mkLlamaCpp {
              inherit target;
              rocm = mkRocm {inherit target;};
            };

            # ROCWMMA-optimized build
            "llama-cpp-${target}-rocwmma" = mkLlamaCpp {
              inherit target;
              rocm = mkRocm {inherit target;};
              enableRocwmma = true;
              enableHipBlasLt = true;
            };

            # Standalone rocwmma package
            "rocwmma-${target}" = mkRocwmmaDerivation pkgs {
              rocm = mkRocm {inherit target;};
              rocmClangWrapper = mkClangWrapper {inherit target;};
              inherit target;
            };
          };

          # Merge all target builds into one attrset
          allPackages =
            pkgs.lib.foldl' (
              acc: target:
                acc // buildForTarget target
            ) {}
            targets;

          # EC-SU_AXB35 packages
          ecPackages = mkEcSuAxb35 pkgs;
        in
          allPackages
          // {
            # Default package
            default = allPackages.llama-cpp-gfx1151;

            # EC-SU_AXB35 monitor tool
            ec-su-axb35-monitor = ecPackages.ec-su-axb35-monitor;
          };

        # Benchmarks as separate output
        benchmarks = import ./bench/default.nix {
          inherit pkgs;
          packages = self.packages.${system};
        };

        # Apps for easy running
        apps = {
          # Interactive llama.cpp CLI with best ROCm build
          llama-cli = {
            type = "app";
            program = toString (pkgs.writeShellScript "llama-cli" ''
              export HSA_OVERRIDE_GFX_VERSION=11.5.1
              ${self.packages.${system}.llama-cpp-gfx1151-rocwmma}/bin/llama-cli "$@"
            '');
          };

          # Start llama.cpp server with best ROCm build
          llama-server = {
            type = "app";
            program = toString (pkgs.writeShellScript "llama-server" ''
              export HSA_OVERRIDE_GFX_VERSION=11.5.1
              ${self.packages.${system}.llama-cpp-gfx1151-rocwmma}/bin/llama-server "$@"
            '');
          };
        };
      }
    )
    // flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
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
    )
    // {
      # Top-level shortcut to x86_64-linux benchmarks
      benchmarks = let
        pkgs = nixpkgs.legacyPackages.x86_64-linux;
        packages = self.packages.x86_64-linux;
      in
        import ./bench/default.nix {
          inherit pkgs packages;
        };
    };
}
