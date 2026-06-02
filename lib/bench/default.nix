{
  lib,
  self,
  defaultRocmTarget,
  cudaPkgsFor,
  benchLib,
}:

pkgs:
let
  system = pkgs.stdenv.hostPlatform.system;
  cudaPkgs = cudaPkgsFor system;
  modelsRoot = benchLib.defaultModelsRoot pkgs;

  flattenBenchmarks =
    benchmarks:
    lib.concatMapAttrs (
      model: benchs:
      lib.mapAttrs' (name: drv: {
        name = "bench-${model}-${name}";
        value = drv;
      }) benchs
    ) benchmarks;

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

  genericBenchmarks = import ./llamacpp.nix {
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

  acceleratedBenchmarks = import ./llamacpp.nix {
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
        (import ./fastflowlm.nix {
          inherit pkgs modelsRoot;
          package = pkgs.fastflowlm;
        }).benchmarks;
      ds4Benchmarks =
        (import ./ds4.nix {
          inherit pkgs modelsRoot;
          package = pkgs.ds4-rocm;
          target = defaultTargetMetadata;
        }).benchmarks;
      vllmBenchmarks =
        (import ./vllm.nix {
          inherit pkgs modelsRoot;
          package = pkgs.vllm-rocm;
          target = defaultTargetMetadata;
        }).benchmarks;
      mlxRocmBenchmarks =
        (import ./mlx.nix {
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
        (import ./ds4.nix {
          inherit pkgs modelsRoot;
          package = self.packages.${system}.ds4;
          accelerator = "metal";
        }).benchmarks;
      ds4PiBenchmarks =
        (import ./ds4-pi.nix {
          inherit pkgs modelsRoot;
          ds4Package = self.packages.${system}.ds4;
          piWrapPackage = self.packages.${system}.pi-wrap;
        }).benchmarks;
      mlxMetalBenchmarks =
        (import ./mlx.nix {
          inherit pkgs;
          package = self.packages.${system}.mlx;
          accelerator = "metal";
        }).benchmarks;
    in
    flattenBenchmarks ds4MetalBenchmarks
    // flattenBenchmarks ds4PiBenchmarks
    // flattenBenchmarks mlxMetalBenchmarks
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
in
flattenBenchmarks genericBenchmarks // linuxBenchmarks // darwinBenchmarks // cudaRtx4090Benchmarks
