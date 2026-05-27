{ lib }:

# Top-level benchmark composition. `mkBenchmarkSuites` is what the flake
# wires into `benchmarks.<system>`. It glues together generic, accelerated,
# Linux-only, Darwin-only, and CUDA-smoke suites into one flat
# `bench-<model>-<name>` attrset.

let
  flattenBenchmarks =
    benchmarks:
    lib.concatMapAttrs (
      model: benchs:
      lib.mapAttrs' (name: drv: {
        name = "bench-${model}-${name}";
        value = drv;
      }) benchs
    ) benchmarks;
in
{
  inherit flattenBenchmarks;

  mkBenchmarkSuites =
    {
      pkgs,
      self,
      defaultRocmTarget,
      cudaPkgs ? null,
    }:
    let
      system = pkgs.stdenv.hostPlatform.system;
      benchLib = import ./bench.nix { inherit lib; };
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

      genericBenchmarks = import ../bench/default.nix {
        inherit pkgs modelsRoot;
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

      acceleratedBenchmarks = import ../bench/default.nix {
        inherit pkgs modelsRoot;
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
            env = {
              # Pin the AMD Vulkan ICD for benchmark reproducibility.
              # Outside benchmarks the package keeps the stock nixpkgs
              # behaviour of resolving an ICD via /run/opengl-driver.
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
            (import ../bench/suites/fastflowlm.nix {
              inherit pkgs modelsRoot;
              package = pkgs.fastflowlm;
            }).benchmarks;
          ds4Benchmarks =
            (import ../bench/suites/ds4.nix {
              inherit pkgs modelsRoot;
              package = pkgs.${"ds4-rocm-${defaultRocmTarget.packageSuffix}"};
              target = defaultTargetMetadata;
            }).benchmarks;
          vllmBenchmarks =
            (import ../bench/suites/vllm.nix {
              inherit pkgs modelsRoot;
              package = pkgs.${"vllm-rocm-therock-${defaultRocmTarget.packageSuffix}"};
              target = defaultTargetMetadata;
            }).benchmarks;
          mlxRocmBenchmarks =
            (import ../bench/suites/mlx.nix {
              inherit pkgs;
              package = self.packages.${system}."mlx-rocm-${defaultRocmTarget.packageSuffix}";
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
            (import ../bench/suites/ds4.nix {
              inherit pkgs modelsRoot;
              package = self.packages.${system}.ds4;
              accelerator = "metal";
            }).benchmarks;
          mlxMetalBenchmarks =
            (import ../bench/suites/mlx.nix {
              inherit pkgs;
              package = self.packages.${system}.mlx;
              accelerator = "metal";
            }).benchmarks;
        in
        flattenBenchmarks ds4MetalBenchmarks // flattenBenchmarks mlxMetalBenchmarks
      );

      cudaSmokeBenchmarks = lib.optionalAttrs (pkgs.stdenv.isLinux && cudaPkgs != null) (
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
              # Pin the NVIDIA userspace driver for benchmark
              # reproducibility. The runner host's kernel module must
              # match this nvidia_x11 version.
              LD_LIBRARY_PATH = "${nvidiaDriver}/lib";
            };
            requirements = {
              systemFeatures = [ "cuda-smoke" ];
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
              kind = "cuda-smoke";
              suite = "cuda";
              accelerator = "cuda";
              scenario = "device-smoke";
              target = {
                vendor = "nvidia";
                arch = "sm_89";
                deviceClass = "rtx4090";
                systemFeature = "cuda-smoke";
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
    in
    flattenBenchmarks genericBenchmarks // linuxBenchmarks // darwinBenchmarks // cudaSmokeBenchmarks;
}
