{
  lib,
  pkgs,
  system,
  vllmLib,
}:

let
  inherit (vllmLib) hardwareProfiles;

  benchmarks = import ../bench/default.nix {
    inherit pkgs;
    packages = {
      inherit (pkgs) llama-cpp-rocm llama-cpp-vulkan;
    };
  };

  benchmarkPackages = pkgs.lib.concatMapAttrs (
    model: benchs:
    pkgs.lib.mapAttrs' (name: drv: {
      name = "bench-${model}-${name}";
      value = drv;
    }) benchs
  ) benchmarks;

  vllmTargets = [
    {
      name = "cpu";
      hardware = hardwareProfiles.none;
    }
    {
      name = "rocm-gfx1151";
      hardware = hardwareProfiles.gfx1151;
      envName = "vllm-env-rocm-gfx1151";
      tunedEnvName = "vllm-env-rocm-gfx1151-zen5";
    }
    {
      name = "cuda-rtx4090";
      hardware = hardwareProfiles.rtx4090;
    }
  ];

  mkVllmTargetPackages =
    {
      name,
      hardware,
      envName ? null,
      tunedEnvName ? null,
    }:
    let
      pkgsFor = vllmLib.mkPackageSet {
        inherit system hardware;
      };
      tunedPkgsFor = vllmLib.mkPackageSet {
        inherit system hardware;
        cpu = "znver5";
      };
      package =
        packageSet: tunePackage:
        vllmLib.mkVllmPackage {
          pkgs = packageSet;
          inherit hardware tunePackage;
        };
      env =
        packageSet: tunePackage: explicitName:
        vllmLib.mkVllmEnv (
          {
            pkgs = packageSet;
            inherit hardware tunePackage;
          }
          // lib.optionalAttrs (explicitName != null) {
            name = explicitName;
          }
        );
    in
    {
      "vllm-${name}" = package pkgsFor false;
      "vllm-env-${name}" = env pkgsFor false envName;
      "vllm-${name}-zen5" = package tunedPkgsFor true;
      "vllm-env-${name}-zen5" = env tunedPkgsFor true tunedEnvName;
    };

  vllmPackages = lib.foldl' (
    packages: target: packages // mkVllmTargetPackages target
  ) { } vllmTargets;
in
{
  default = pkgs.llama-cpp-rocm;
  inherit (pkgs)
    ec-su-axb35-monitor
    llama-cpp-rocm
    llama-cpp-vulkan
    ;
}
// vllmPackages
// benchmarkPackages
