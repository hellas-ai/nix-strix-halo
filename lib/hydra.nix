{
  lib,
  self,
  pkgsFor,
  defaultRocmTarget,
}:

let
  benchmarkHostFeatures = {
    x86_64-linux = [
      "benchmark"
      "gfx1151"
      "xdna2"
      "rtx4090"
    ];
    aarch64-darwin = [ "benchmark" ];
  };

  mkBenchmarkJobs =
    pkgs:
    let
      system = pkgs.stdenv.hostPlatform.system;
      addBenchmarkFeature =
        drv:
        drv.overrideAttrs (old: {
          requiredSystemFeatures = lib.unique ((old.requiredSystemFeatures or [ ]) ++ [ "benchmark" ]);
        });

      supportedOnBenchmarkHost =
        _name: drv:
        lib.all (feature: lib.elem feature (benchmarkHostFeatures.${system} or [ ])) (
          drv.requiredSystemFeatures or [ ]
        );
    in
    lib.filterAttrs supportedOnBenchmarkHost (
      lib.mapAttrs (_: addBenchmarkFeature) self.benchmarks.${system}
    );

  mkJobs =
    pkgs:
    let
      system = pkgs.stdenv.hostPlatform.system;
      benchmarks = self.benchmarks.${system};

      maintainerMeta = {
        maintainers = with lib.maintainers; [ georgewhewell ];
      };

      mkLinkFarm =
        name: entries:
        pkgs.runCommandLocal name
          {
            meta = maintainerMeta;
          }
          ''
            mkdir -p "$out"
            ${lib.concatMapStringsSep "\n" (entry: ''
              ln -s ${entry.path} "$out"/${lib.escapeShellArg entry.name}
            '') entries}
          '';

      mkAggregate =
        aggregateName: jobs:
        mkLinkFarm "nix-strix-halo-${aggregateName}" (
          lib.mapAttrsToList (name: path: {
            inherit name path;
          }) jobs
        );

      isSourceCheckSystem = system == "x86_64-linux";

      prQuickJobs = lib.optionalAttrs isSourceCheckSystem self.checks.${system};
      prQuick = mkAggregate "pr-quick" prQuickJobs;

      afterPrQuick =
        name: path:
        mkLinkFarm "nix-strix-halo-pr-full-${name}" (
          lib.optionals isSourceCheckSystem [
            {
              name = "pr-quick";
              path = prQuick;
            }
          ]
          ++ [
            {
              inherit name path;
            }
          ]
        );

      mkPkgsForProvider =
        {
          rocmProvider,
          pythonProvider ? "therock-wheels",
        }:
        pkgsFor {
          inherit system rocmProvider pythonProvider;
        };

      fromSourcePkgs = lib.optionalAttrs (system == "x86_64-linux") (mkPkgsForProvider {
        rocmProvider = "therock-source";
      });

      nixpkgsRocmPkgs = lib.optionalAttrs (system == "x86_64-linux") (mkPkgsForProvider {
        rocmProvider = "nixpkgs";
      });

      prFullJobs = {
        default = afterPrQuick "default" self.packages.${system}.default;
        jaccl = afterPrQuick "jaccl" self.packages.${system}.jaccl;
      }
      // lib.optionalAttrs (system == "aarch64-darwin") {
        ds4 = afterPrQuick "ds4" self.packages.${system}.ds4;
        mlx = afterPrQuick "mlx" self.packages.${system}.mlx;
        mlx-metal = afterPrQuick "mlx-metal" self.packages.${system}.mlx-metal;
        mlx-metal-gemm-smoke = afterPrQuick "mlx-metal-gemm-smoke" benchmarks.bench-mlx-metal-gemm-smoke;
      }
      // lib.optionalAttrs (system == "x86_64-linux") {
        inherit (self.packages.${system})
          ds4-rocm
          ec-su-axb35-monitor
          fastflowlm
          llama-cpp-rocm
          llama-cpp-vulkan
          llama-cpp-master
          strix-halo-mes-firmware
          therock-rocm
          tokenizers-cpp
          vllm-rocm
          xrt-amdxdna
          ;
        mlx-rocm = afterPrQuick "mlx-rocm" self.packages.${system}.mlx-rocm;
        mlx-rocm-gemm-smoke =
          afterPrQuick "mlx-rocm-gemm-smoke"
            benchmarks."bench-mlx-rocm-${defaultRocmTarget.packageSuffix}-gemm-smoke";
        live-iso = afterPrQuick "live-iso" self.packages.${system}.live-iso;
        vllm-throughput-smoke =
          afterPrQuick "vllm-throughput-smoke"
            benchmarks."bench-qwen3-0-6b-vllm-rocm-${defaultRocmTarget.packageSuffix}-throughput-smoke";
        ds4-rocm-smoke = afterPrQuick "ds4-rocm-smoke" benchmarks.bench-deepseek-v4-flash-ds4-rocm-gfx1151-smoke;
        fastflowlm-npu-smoke = afterPrQuick "fastflowlm-npu-smoke" benchmarks.bench-llama3-2-1b-fastflowlm-medium;
        cuda-rtx4090-device-smoke = afterPrQuick "cuda-rtx4090-device-smoke" benchmarks.bench-cuda-rtx4090-llama-cpp-master-device-smoke;

        vllm-rocm-from-source = fromSourcePkgs.vllm-rocm;
        therock-rocm-from-source = fromSourcePkgs.therock-rocm;
        llama-cpp-rocm-from-source = fromSourcePkgs.llama-cpp-rocm;
        llama-cpp-rocm-nixpkgs = nixpkgsRocmPkgs.llama-cpp-rocm;
      };
    in
    lib.optionalAttrs isSourceCheckSystem {
      pr-quick = prQuickJobs // {
        all = prQuick;
      };
    }
    // {
      pr-full = prFullJobs // {
        all = mkAggregate "pr-full" prFullJobs;
      };
    }
    // lib.optionalAttrs (system == "x86_64-linux") {
      live-iso = self.packages.${system}.live-iso;
    };
in
{
  inherit
    benchmarkHostFeatures
    mkBenchmarkJobs
    mkJobs
    ;
}
