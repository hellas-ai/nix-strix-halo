{
  inputs,
  lib,
  inputVersion,
  rocmTarget,
}:

# Locally-defined packages. One overlay, one source of truth for "what
# this flake adds on top of nixpkgs". rocm/python deps come from
# `final.rocmPackages` / `final.python3Packages`, so the consumer
# controls those via the rocm.nix / python.nix overlays applied earlier.
#
# Single outputs per logical package — `llama-cpp-rocm`, `ds4-rocm`,
# etc. — narrowed to `rocmTarget`. Building for a different target
# means composing this overlay with a different `rocmTarget`; the
# per-target suffix outputs that the old flake produced are now
# emergent from that composition rather than baked into the overlay.

final: prev:

lib.optionalAttrs prev.stdenv.isLinux (
  let
    ecPackages = prev.callPackage ../pkgs/ec-su-axb35.nix {
      ec-su-axb35-src = inputs.ec-su-axb35;
    };

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

    llamaCppRocm = prev.llama-cpp.override {
      rocmSupport = true;
      rpcSupport = true;
      inherit (final) rocmPackages;
      inherit (rocmTarget) rocmGpuTargets;
    };
    llamaCppVulkan = prev.llama-cpp.override {
      vulkanSupport = true;
      rpcSupport = true;
    };
    llamaCppCuda = prev.llama-cpp.override {
      cudaSupport = true;
      rpcSupport = true;
    };
  in
  {
    ec-su-axb35 = ecPackages.kernelModule;
    ec-su-axb35-monitor = ecPackages.monitor;
    strix-halo-mes-firmware = prev.callPackage ../pkgs/strix-halo-mes-firmware.nix { };
    tokenizers-cpp = prev.callPackage ../pkgs/tokenizers-cpp { };

    xrt = prev.callPackage ../pkgs/xrt {
      src = inputs.xrt-src;
      version = inputVersion "2.21" inputs.xrt-src;
      xdnaSrc = inputs.xdna-driver-src;
      xdnaVersion = inputVersion "1.7" inputs.xdna-driver-src;
    };
    xrt-amdxdna = final.xrt.xdna;

    fastflowlm = prev.callPackage ../pkgs/fastflowlm {
      inherit (final) tokenizers-cpp xrt;
      src = inputs.fastflowlm;
    };

    llama-cpp-rocm = llamaCppRocm;
    llama-cpp-vulkan = llamaCppVulkan;
    llama-cpp-cuda = llamaCppCuda;
    llama-cpp-master-rocm = applyMasterSrc "llama-cpp-master-rocm" llamaCppRocm;
    llama-cpp-master-vulkan = applyMasterSrc "llama-cpp-master-vulkan" llamaCppVulkan;
    llama-cpp-master-cuda = applyMasterSrc "llama-cpp-master-cuda" llamaCppCuda;

    ds4-rocm = prev.callPackage ../pkgs/ds4-rocm {
      src = inputs.ds4-hip;
      rocmSdk = final."therock-rocm-${rocmTarget.packageSuffix}";
      version = inputVersion "experimental" inputs.ds4-hip;
      inherit (rocmTarget) packageSuffix;
      offloadArch = builtins.head rocmTarget.buildTargets;
      hsaOverrideGfxVersion = rocmTarget.hsaOverride or null;
    };
  }
)
