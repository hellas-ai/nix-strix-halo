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
    aarch64-darwin = [
      "benchmark"
      "metal"
    ];
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
      isGateSystem = system == "x86_64-linux";
      x86Packages = self.packages.x86_64-linux;
      x86Benchmarks = self.benchmarks.x86_64-linux;
      darwinPackages = self.packages.aarch64-darwin;
      darwinBenchmarks = self.benchmarks.aarch64-darwin;

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

      mkPkgsForProvider =
        {
          rocmProvider,
          pythonProvider ? "therock-wheels",
        }:
        pkgsFor {
          inherit system rocmProvider pythonProvider;
        };

      fromSourcePkgs = lib.optionalAttrs isGateSystem (mkPkgsForProvider {
        rocmProvider = "therock-source";
      });

      nixpkgsRocmPkgs = lib.optionalAttrs isGateSystem (mkPkgsForProvider {
        rocmProvider = "nixpkgs";
      });

      checkJobs = lib.optionalAttrs isGateSystem self.checks.x86_64-linux;

      buildJobs = lib.optionalAttrs isGateSystem {
        inherit (x86Packages)
          default
          jaccl
          ds4-rocm
          ec-su-axb35-monitor
          fastflowlm
          llama-cpp-rocm
          llama-cpp-vulkan
          llama-cpp-master
          mlx-rocm
          strix-halo-mes-firmware
          therock-rocm
          tokenizers-cpp
          vllm-rocm
          xrt-amdxdna
          live-iso
          ;

        darwin-default = darwinPackages.default;
        darwin-jaccl = darwinPackages.jaccl;
        ds4-metal = darwinPackages.ds4;
        inherit (darwinPackages)
          mlx
          mlx-metal
          ;

        vllm-rocm-from-source = fromSourcePkgs.vllm-rocm;
        therock-rocm-from-source = fromSourcePkgs.therock-rocm;
        llama-cpp-rocm-from-source = fromSourcePkgs.llama-cpp-rocm;
        llama-cpp-rocm-nixpkgs = nixpkgsRocmPkgs.llama-cpp-rocm;
      };

      smokeJobs = lib.optionalAttrs isGateSystem {
        ds4-metal = darwinBenchmarks.bench-deepseek-v4-flash-ds4-metal-smoke;
        ds4-rocm = x86Benchmarks.bench-deepseek-v4-flash-ds4-rocm-gfx1151-smoke;
        mlx-metal = darwinBenchmarks.bench-mlx-metal-gemm-smoke;
        mlx-rocm = x86Benchmarks."bench-mlx-rocm-${defaultRocmTarget.packageSuffix}-gemm-smoke";
        fastflowlm-npu = x86Benchmarks.bench-llama3-2-1b-fastflowlm-medium;
        vllm-rocm =
          x86Benchmarks."bench-qwen3-0-6b-vllm-rocm-${defaultRocmTarget.packageSuffix}-throughput-smoke";
        cuda-rtx4090 = x86Benchmarks.bench-cuda-rtx4090-llama-cpp-master-device-smoke;
      };
    in
    lib.optionalAttrs isGateSystem {
      ci = {
        checks = mkAggregate "ci-checks" checkJobs;
        build = mkAggregate "ci-build" buildJobs;
        smoke = mkAggregate "ci-smoke" smokeJobs;
      };
    };
in
{
  inherit
    benchmarkHostFeatures
    mkBenchmarkJobs
    mkJobs
    ;
}
