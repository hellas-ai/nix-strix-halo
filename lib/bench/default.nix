{
  lib,
  self,
  defaultRocmTarget,
  benchLib,
}:

pkgs:
let
  system = pkgs.stdenv.hostPlatform.system;
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

  linuxBenchmarks = lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux (
    let
      fastflowlmBenchmarks =
        (import ./fastflowlm.nix {
          inherit pkgs modelsRoot;
          package = pkgs.fastflowlm;
        }).benchmarks;
    in
    flattenBenchmarks acceleratedBenchmarks // flattenBenchmarks fastflowlmBenchmarks
  );

  darwinBenchmarks = lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin (
    let
      mlxMetalBenchmarks =
        (import ./mlx.nix {
          inherit pkgs;
          package = self.packages.${system}.mlx;
          accelerator = "metal";
        }).benchmarks;
    in
    flattenBenchmarks mlxMetalBenchmarks
  );

in
flattenBenchmarks genericBenchmarks // linuxBenchmarks // darwinBenchmarks
