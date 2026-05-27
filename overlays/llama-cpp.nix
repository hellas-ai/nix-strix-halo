{
  inputs,
  lib,
  rocmTargets,
  defaultRocmTarget,
}:

# All llama.cpp variants this flake exposes:
#
#   - llama-cpp-rocm           (default rocm target, narrowed)
#   - llama-cpp-vulkan
#   - llama-cpp-cuda
#   - llama-cpp-rocm-<suffix>  (one per rocmTarget)
#   - llama-cpp-master-*       (same matrix, but built from llama-cpp-master input)
#
# Pulls the `master` variants by swapping `src` on the nixpkgs llama-cpp
# derivation and stripping the nixpkgs npm hook so the source-only
# rebuild does not require npmDeps.

final: prev:

let
  applyMasterSrc =
    attrName: pkg:
    pkg.overrideAttrs (old: {
      pname = attrName;
      version =
        "master-" + (inputs.llama-cpp-master.shortRev or inputs.llama-cpp-master.rev or "unknown");
      src = inputs.llama-cpp-master;
      npmDeps = null;
      nativeBuildInputs = prev.lib.filter (x: x.pname or "" != "npm-config-hook") (
        old.nativeBuildInputs or [ ]
      );
      preConfigure = ''
        prependToVar cmakeFlags "-DLLAMA_BUILD_COMMIT:STRING=${
          inputs.llama-cpp-master.shortRev or inputs.llama-cpp-master.rev or "master"
        }"
      '';
      cmakeFlags = (old.cmakeFlags or [ ]) ++ [
        (prev.lib.cmakeBool "LLAMA_BUILD_UI" false)
        (prev.lib.cmakeFeature "LLAMA_BUILD_NUMBER" "0")
      ];
    });

  mkTargetedPackage =
    pname: pkg:
    pkg.overrideAttrs (_old: {
      inherit pname;
    });

  mkLlamaCppRocmBase =
    rocmTarget:
    prev.llama-cpp.override {
      rocmSupport = true;
      rpcSupport = true;
      inherit (final) rocmPackages;
      inherit (rocmTarget) rocmGpuTargets;
    };

  llamaCppRocmBase = mkLlamaCppRocmBase defaultRocmTarget;
  llamaCppVulkanBase = prev.llama-cpp.override {
    vulkanSupport = true;
    rpcSupport = true;
  };
  llamaCppCudaBase = prev.llama-cpp.override {
    cudaSupport = true;
    rpcSupport = true;
  };

  llamaCppRocmTargetPackages = lib.listToAttrs (
    map (rocmTarget: {
      name = "llama-cpp-rocm-${rocmTarget.packageSuffix}";
      value = mkTargetedPackage "llama-cpp-rocm-${rocmTarget.packageSuffix}" (
        mkLlamaCppRocmBase rocmTarget
      );
    }) rocmTargets
  );

  llamaCppMasterRocmTargetPackages = lib.listToAttrs (
    map (rocmTarget: {
      name = "llama-cpp-master-rocm-${rocmTarget.packageSuffix}";
      value = applyMasterSrc "llama-cpp-master-rocm-${rocmTarget.packageSuffix}" (
        mkLlamaCppRocmBase rocmTarget
      );
    }) rocmTargets
  );
in
{
  llama-cpp-rocm = llamaCppRocmBase;
  llama-cpp-vulkan = llamaCppVulkanBase;
  llama-cpp-cuda = llamaCppCudaBase;
  llama-cpp-master-rocm = applyMasterSrc "llama-cpp-master-rocm" llamaCppRocmBase;
  llama-cpp-master-vulkan = applyMasterSrc "llama-cpp-master-vulkan" llamaCppVulkanBase;
  llama-cpp-master-cuda = applyMasterSrc "llama-cpp-master-cuda" llamaCppCudaBase;
}
// llamaCppRocmTargetPackages
// llamaCppMasterRocmTargetPackages
