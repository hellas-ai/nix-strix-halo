{
  lib,
  self,
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
      gfx1030Packages = self.legacyPackages.x86_64-linux.gfx1030;
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

      checkJobs = lib.optionalAttrs isGateSystem (
        self.checks.x86_64-linux // { vllm-metal-module = self.checks.aarch64-darwin.vllm-metal-module; }
      );

      buildJobs = lib.optionalAttrs isGateSystem ({
        inherit (x86Packages)
          default
          glm53-serve
          sglang-glm53-rocm
          sglang-rocm-h3
          ec-su-axb35-monitor
          fastflowlm
          kerf-multikernel
          llama-cpp-rocm
          llama-cpp-vulkan
          linux-multikernel
          multikernel-demo-initrd
          therock-rocm
          tokenizers-cpp
          xrt-amdxdna
          live-iso
          ;

        darwin-default = darwinPackages.default;
        darwin-jaccl = darwinPackages.jaccl;
        inherit (darwinPackages)
          mlx
          mlx-metal
          vllm-metal
          ;

        # V620 builders consume these binary gfx1030 artifacts. Keep runtime
        # smoke tests on gfx1151 until the Hydra fleet has V620 hardware.
        gfx1030-llama-cpp-rocm = gfx1030Packages.llama-cpp-rocm;
        gfx1030-sglang-rocm = gfx1030Packages.sglang-rocm;
        gfx1030-therock-python-wheels = gfx1030Packages.therock-python-wheels;
        gfx1030-therock-rocm = gfx1030Packages.therock-rocm;
      });

      smokeJobs = lib.optionalAttrs isGateSystem {
        mlx-metal = darwinBenchmarks.bench-mlx-metal-gemm-smoke;
        vllm-metal = darwinPackages.vllm-metal.tests.metal;
        vllm-metal-imports = darwinPackages.vllm-metal.tests.imports;
        vllm-metal-server = darwinPackages.vllm-metal.tests.server;
        fastflowlm-npu = x86Benchmarks.bench-llama3-2-1b-fastflowlm-short;

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
