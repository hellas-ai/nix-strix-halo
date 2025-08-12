{
  description = "Llama.cpp with pre-built ROCm binaries from TheRock";

  inputs = {
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
      url = "github:cmetz/ec-su_axb35-linux/main";
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
    mkRocmDerivation = pkgs: import ./pkgs/rocm.nix {inherit pkgs rocmSources;};
    mkRocwmmaDerivation = pkgs: import ./pkgs/rocwmma.nix {inherit pkgs rocwmma;};
    mkLlamaCppDerivation = pkgs:
      import ./pkgs/llama-cpp.nix {
        inherit pkgs llama-cpp;
        mkRocwmmaDerivation = mkRocwmmaDerivation pkgs;
      };
    mkEcSuAxb35 = pkgs: import ./pkgs/ec-su-axb35.nix {inherit pkgs ec-su-axb35;};

    # Import benchmark runner
    mkBenchmark = pkgs: import ./bench/default.nix;
  in
    # Overlay output
    {
      overlays.default = final: prev: {
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
        rpc-server = import ./modules/rpc-server.nix;
        benchmark-runner = import ./modules/benchmark-runner.nix;
        ec-su-axb35 = { config, lib, pkgs, ... }: {
          config.lib.inputs.ec-su-axb35 = ec-su-axb35;
          imports = [ ./modules/ec-su-axb35.nix ];
        };
        default = import ./modules/rpc-server.nix;
      };
    }
    //
    # Per-system outputs (ROCm only works on Linux)
    flake-utils.lib.eachSystem ["x86_64-linux"] (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        mkRocm = mkRocmDerivation pkgs;
        mkLlamaCpp = mkLlamaCppDerivation pkgs;
      in {
        packages = let
          # GPU targets we support
          targets = ["gfx110X" "gfx1151" "gfx120X"];

          # Build matrix for each target
          buildForTarget = target: {
            # ROCm toolchain
            "rocm-${target}" = mkRocm {inherit target;};

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
          update-rocm = {
            type = "app";
            program = toString (pkgs.writeShellScript "update-rocm" ''
              ${pkgs.python3}/bin/python3 ${./update-rocm.py} "$@"
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
            (python3.withPackages (ps: with ps; [
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
