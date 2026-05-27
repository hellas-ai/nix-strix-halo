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

    rocmOverride = {
      rocmSupport = true;
      rpcSupport = true;
      inherit (final) rocmPackages;
      inherit (rocmTarget) rocmGpuTargets;
    };
    vulkanOverride = {
      vulkanSupport = true;
      rpcSupport = true;
    };
    cudaOverride = {
      cudaSupport = true;
      rpcSupport = true;
    };

    masterRev = inputs.llama-cpp-master.shortRev or inputs.llama-cpp-master.rev or "unknown";
    # llama-cpp-master is the upstream HEAD variant. Override is via
    # `.overrideAttrs` (the package doesn't expose `src` through .override),
    # but `.override` composes through it — `llamaCppMaster.override
    # { rocmSupport = true; ... }` re-runs nixpkgs' llama-cpp with new
    # args and re-applies these attrs on top, so the rocm/vulkan/cuda
    # variants of master fall out of one master-src definition.
    llamaCppMaster = prev.llama-cpp.overrideAttrs (old: {
      pname = "llama-cpp-master";
      version = "master-${masterRev}";
      src = inputs.llama-cpp-master;
      npmDeps = null;
      nativeBuildInputs = lib.filter (x: x.pname or "" != "npm-config-hook") (
        old.nativeBuildInputs or [ ]
      );
      preConfigure = ''
        prependToVar cmakeFlags "-DLLAMA_BUILD_COMMIT:STRING=${masterRev}"
      '';
      cmakeFlags = (old.cmakeFlags or [ ]) ++ [
        (lib.cmakeBool "LLAMA_BUILD_UI" false)
        (lib.cmakeFeature "LLAMA_BUILD_NUMBER" "0")
      ];
    });
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

    llama-cpp-rocm = prev.llama-cpp.override rocmOverride;
    llama-cpp-vulkan = prev.llama-cpp.override vulkanOverride;
    llama-cpp-cuda = prev.llama-cpp.override cudaOverride;
    llama-cpp-master = llamaCppMaster;
    llama-cpp-master-rocm = llamaCppMaster.override rocmOverride;
    llama-cpp-master-vulkan = llamaCppMaster.override vulkanOverride;
    llama-cpp-master-cuda = llamaCppMaster.override cudaOverride;

    ds4-rocm = prev.callPackage ../pkgs/ds4-rocm {
      src = inputs.ds4-hip;
      rocmSdk = final."therock-rocm-${rocmTarget.packageSuffix}";
      version = inputVersion "experimental" inputs.ds4-hip;
      inherit (rocmTarget) packageSuffix;
      offloadArch = builtins.head rocmTarget.buildTargets;
      hsaOverrideGfxVersion = rocmTarget.hsaOverride or null;
    };

    mlx-rocm = final.python3Packages.callPackage ../pkgs/mlx/rocm.nix {
      pname = "mlx-rocm";
      rdma-core = final.rdma-core-usb4;
      rocmPackages = final.therockRocmPackages.${builtins.head rocmTarget.buildTargets};
      gfx = builtins.head rocmTarget.buildTargets;
    };

    # Unsuffixed aliases for the active rocmTarget. Lets consumers write
    # `pkgs.therock-rocm` instead of `pkgs.therock-rocm-gfx1151` once the
    # package set's target is fixed (which it is per pkgsFor invocation).
    therock-rocm = final."therock-rocm-${rocmTarget.packageSuffix}";
    therock-rocm-env = final."therock-rocm-${rocmTarget.packageSuffix}-env";
    therock-python = final."therock-python-${rocmTarget.packageSuffix}";
    therock-python-wheels = final."therock-python-wheels-${rocmTarget.packageSuffix}";
    therock-amdsmi = final."therock-amdsmi-${rocmTarget.packageSuffix}";
    torch-rocm = final."torch-rocm-${rocmTarget.packageSuffix}";
    vllm-rocm = final."vllm-rocm-therock-${rocmTarget.packageSuffix}";
  }
)
