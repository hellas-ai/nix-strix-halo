# Hydra build matrix for nix-strix-halo.
#
# This file composes the specific benchmarks + aggregates Hydra builds.
# It is NOT a flake output — Hydra (and curious humans) consume it
# directly:
#
#   nix-build hydra.nix --argstr system x86_64-linux -A pr-quick.all
#   nix-build hydra.nix --argstr system x86_64-linux -A pr-full.vllm-throughput-smoke
#
# Sits on top of the flake's high-level interface:
#   - flake.lib.bench               benchmark helper library
#   - flake.lib.therockTargets      target catalogue
#   - flake.packages.<system>       default-target packages
#   - flake.legacyPackages.<system>.<targetSuffix>  per-target package sets

{
  self ? builtins.getFlake (toString ./.),
  system ? builtins.currentSystem,
}:

let
  inherit (self.inputs) nixpkgs;
  inherit (nixpkgs) lib;

  flakeLib = self.lib;
  benchLib = flakeLib.bench;
  inherit (flakeLib.therockTargets) defaultRocmTarget;

  pkgs = self.legacyPackages.${system}.${defaultRocmTarget.packageSuffix};
  # CUDA-configured nixpkgs for the RTX 4090 device smoke bench.
  cudaPkgs = import nixpkgs {
    inherit system;
    config = {
      allowUnfree = true;
      cudaSupport = true;
      cudaCapabilities = [ "8.9" ];
    };
    overlays = [ self.overlays.default ];
  };

  modelsRoot = benchLib.defaultModelsRoot pkgs;

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

  flattenBenchmarks =
    benchmarks:
    lib.concatMapAttrs (
      model: benchs:
      lib.mapAttrs' (name: drv: {
        name = "bench-${model}-${name}";
        value = drv;
      }) benchs
    ) benchmarks;

  genericBenchmarks = import ./bench/default.nix {
    inherit pkgs modelsRoot;
    tools = [
      {
        name = "cpu";
        package = pkgs.llama-cpp;
        executable = "llama-bench";
        backend = "cpu";
        metadata.accelerator = "cpu";
      }
    ];
  };

  acceleratedBenchmarks = import ./bench/default.nix {
    inherit pkgs modelsRoot;
    tools = [
      {
        name = "rocm";
        package = pkgs.llama-cpp-rocm;
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
        env = {
          VK_DRIVER_FILES = "${pkgs.mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json";
        };
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

  linuxBenchmarks = lib.optionalAttrs pkgs.stdenv.isLinux (
    let
      fastflowlmBenchmarks =
        (import ./bench/suites/fastflowlm.nix {
          inherit pkgs modelsRoot;
          package = pkgs.fastflowlm;
        }).benchmarks;
      ds4Benchmarks =
        (import ./bench/suites/ds4.nix {
          inherit pkgs modelsRoot;
          package = pkgs.ds4-rocm;
          target = defaultTargetMetadata;
        }).benchmarks;
      vllmBenchmarks =
        (import ./bench/suites/vllm.nix {
          inherit pkgs modelsRoot;
          package = pkgs.vllm-rocm;
          target = defaultTargetMetadata;
        }).benchmarks;
      mlxRocmBenchmarks =
        (import ./bench/suites/mlx.nix {
          inherit pkgs;
          package = pkgs.mlx-rocm;
          target = defaultTargetMetadata;
        }).benchmarks;
    in
    flattenBenchmarks acceleratedBenchmarks
    // flattenBenchmarks fastflowlmBenchmarks
    // flattenBenchmarks ds4Benchmarks
    // flattenBenchmarks vllmBenchmarks
    // flattenBenchmarks mlxRocmBenchmarks
  );

  darwinBenchmarks = lib.optionalAttrs pkgs.stdenv.isDarwin (
    let
      ds4MetalBenchmarks =
        (import ./bench/suites/ds4.nix {
          inherit pkgs modelsRoot;
          package = self.packages.${system}.ds4;
          accelerator = "metal";
        }).benchmarks;
      mlxMetalBenchmarks =
        (import ./bench/suites/mlx.nix {
          inherit pkgs;
          package = self.packages.${system}.mlx;
          accelerator = "metal";
        }).benchmarks;
    in
    flattenBenchmarks ds4MetalBenchmarks // flattenBenchmarks mlxMetalBenchmarks
  );

  cudaRtx4090Benchmarks = lib.optionalAttrs pkgs.stdenv.isLinux (
    let
      nvidiaSmi = cudaPkgs.linuxPackages.nvidia_x11.bin;
      nvidiaDriver = cudaPkgs.linuxPackages.nvidia_x11;
    in
    {
      bench-cuda-rtx4090-llama-cpp-master-device-smoke = benchLib.mkBenchmark {
        pkgs = cudaPkgs;
        name = "cuda-rtx4090-llama-cpp-master-device-smoke";
        package = cudaPkgs.llama-cpp-master-cuda;
        packages = [ nvidiaSmi ];
        command = [
          (cudaPkgs.writeShellScript "cuda-rtx4090-llama-cpp-master-device-smoke" ''
            set -euo pipefail
            ${nvidiaSmi}/bin/nvidia-smi -L
            ${cudaPkgs.llama-cpp-master-cuda}/bin/llama-bench --list-devices
          '')
        ];
        env = {
          LD_LIBRARY_PATH = "${nvidiaDriver}/lib";
        };
        requirements = {
          systemFeatures = [
            "rtx4090"
            "benchmark"
          ];
          hostProfiles = [ "linux-nvidia-cuda" ];
          sandboxPaths = [
            "/dev/nvidia0"
            "/dev/nvidiactl"
            "/dev/nvidia-uvm"
            "/dev/nvidia-uvm-tools"
            "/dev/nvidia-caps"
            "/proc/driver/nvidia"
          ];
        };
        metadata = {
          kind = "cuda-device-smoke";
          suite = "cuda";
          accelerator = "cuda";
          scenario = "device-smoke";
          target = {
            vendor = "nvidia";
            arch = "sm_89";
            deviceClass = "rtx4090";
            systemFeature = "rtx4090";
          };
          tool = {
            backend = "cuda";
            executable = "llama-bench --list-devices";
            packageRole = "llama-cpp-master-cuda";
          };
        };
        meta.platforms = [ "x86_64-linux" ];
        description = "Run CUDA llama.cpp device smoke test";
      };
    }
  );

  benchmarks =
    flattenBenchmarks genericBenchmarks // linuxBenchmarks // darwinBenchmarks // cudaRtx4090Benchmarks;

  # Hydra aggregates
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

  # pr-quick currently has no jobs — flake.checks lost its metadata
  # assertions in the cleanup. Bring them back here later if needed.
  prQuickJobs = { };
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

  # Re-compose the default-target package set with a non-default rocm
  # provider, so hydra can build the parameterised abstraction
  # end-to-end. The composition mirrors flake.nix's internal `pkgsFor`
  # (intentionally not exposed via flake.lib).
  mkPkgsForProvider =
    {
      rocmProvider,
      pythonProvider ? "therock-wheels",
    }:
    import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
      };
      overlays = [
        self.inputs.thunderbolt-ibverbs.overlays.default
        (
          _: prev:
          let
            tb = self.inputs.thunderbolt-ibverbs.packages.${prev.stdenv.hostPlatform.system};
          in
          {
            thunderbolt-ibverbs-bench-tools = tb.bench-tools;
            thunderbolt-ibverbs-perftest = tb.perftest;
          }
          // lib.optionalAttrs prev.stdenv.isLinux {
            inherit (tb)
              linux-thunderbolt
              linux-thunderbolt-dev
              linux-thunderbolt-modules
              rdma-core-usb4
              thunderbolt-ibverbs
              thunderbolt-ibverbs-linux-thunderbolt
              ;
            rdma-core = tb.rdma-core-usb4;
          }
        )
        (flakeLib.mkRocmOverlay {
          provider = rocmProvider;
          rocmTarget = defaultRocmTarget;
        })
        (flakeLib.mkPythonOverlay {
          provider = pythonProvider;
          rocmTarget = defaultRocmTarget;
        })
        (import ./overlays/therock-vllm.nix {
          inherit lib;
          target = defaultRocmTarget;
          vllmSrc = self.inputs.vllm-src;
          vllmVersion = "0.21.0";
        })
        (flakeLib.mkPkgsOverlay { rocmTarget = defaultRocmTarget; })
      ];
    };

  fromSourcePkgs = lib.optionalAttrs (system == "x86_64-linux") (mkPkgsForProvider {
    rocmProvider = "therock-source";
  });

  # nixpkgs.rocmPackages.${suffix} narrowed scope. vllm/ds4/mlx hard-ref
  # TheRock attrs and don't resolve here, but llama-cpp-rocm exercises
  # the dispatcher cleanly.
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
    # Default-target binary outputs covering the locally-defined package surface.
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
    cuda-rtx4090-device-smoke = afterPrQuick "cuda-rtx4090-device-smoke" benchmarks.bench-cuda-rtx4090-llama-cpp-master-device-smoke;

    # Provider-variant builds. Exercise the rocm.nix dispatcher
    # end-to-end against the from-source TheRock build. Long builds.
    vllm-rocm-from-source = fromSourcePkgs.vllm-rocm;
    therock-rocm-from-source = fromSourcePkgs.therock-rocm;
    llama-cpp-rocm-from-source = fromSourcePkgs.llama-cpp-rocm;

    # nixpkgs provider variant — same dispatcher, different rocm
    # source. Only llama-cpp-rocm builds against it (vllm/ds4/mlx
    # are TheRock-shaped).
    llama-cpp-rocm-nixpkgs = nixpkgsRocmPkgs.llama-cpp-rocm;
  };
in
{
  inherit benchmarks;

  pr-quick = prQuickJobs // {
    all = prQuick;
  };

  pr-full = prFullJobs // {
    all = mkAggregate "pr-full" prFullJobs;
  };
}
// lib.optionalAttrs (system == "x86_64-linux") {
  # Re-export the raw ISO so Hydra's "download iso" UI can find the
  # underlying build product. The linkFarm-wrapped `pr-full.live-iso`
  # loses the `nix-support/hydra-build-products` file.
  live-iso = self.packages.${system}.live-iso;
}
