{
  inputs,
  lib,
  inputVersion,
  rocmTarget,
  rocmProvider ? "therock-bin",
  pythonProvider ? "therock-wheels",
  sources ? null,
}:

# Locally-defined packages. One overlay, one source of truth for "what
# this flake adds on top of nixpkgs". rocm/python deps come from
# `final.rocmPackages` / `final.python3Packages`, so the consumer
# controls those via the rocm.nix / python.nix overlays applied earlier.
#
# Single output per logical package — `llama-cpp-rocm`, `ds4-rocm`,
# etc. — narrowed to `rocmTarget`. Building for a different target is a
# matter of composing this overlay with a different `rocmTarget`;
# per-target suffix attrs fall out of `legacyPackages.<system>.<suffix>`
# at the flake level rather than being baked into the overlay.

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
    suffix = rocmTarget.packageSuffix;
    firstBuildTarget = builtins.head rocmTarget.buildTargets;

    sourceHas = path: sources != null && lib.hasAttrByPath path sources;
    sourceHasTherockRocm = sourceHas [
      "rocm"
      "linux"
      suffix
    ];
    sourceHasTherockPython = sourceHas [
      "pythonWheels"
      "targets"
      suffix
    ];
    finalHas = name: builtins.hasAttr name final;

    rocmProviderHasTherockAttrs = rocmProvider == "therock-bin" || rocmProvider == "therock-source";
    pythonProviderHasTherockAttrs = pythonProvider == "therock-wheels";
    supportsTherockRocm =
      rocmProviderHasTherockAttrs
      && (if sources != null then sourceHasTherockRocm else finalHas "therock-rocm-${suffix}");
    supportsTherockPython =
      pythonProviderHasTherockAttrs
      && (if sources != null then sourceHasTherockPython else finalHas "therock-python-${suffix}");

    georgewhewellMaintained =
      drv:
      drv.overrideAttrs (old: {
        meta = (old.meta or { }) // {
          maintainers = with lib.maintainers; [ georgewhewell ];
        };
      });

    # Compose Strix llama.cpp variants with the thunderbolt-ibverbs
    # rdma-core overlay so ggml's RPC backend builds its RDMA transport.
    # Keep this in one helper so ROCm/Vulkan and master/current variants
    # do not drift.
    withRdmaRpc =
      drv:
      drv.overrideAttrs (old: {
        buildInputs = (old.buildInputs or [ ]) ++ [ final.rdma-core-usb4 ];
        cmakeFlags = (old.cmakeFlags or [ ]) ++ [
          (lib.cmakeBool "GGML_RPC_RDMA" true)
        ];
      });

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
      meta = (old.meta or { }) // {
        maintainers = with lib.maintainers; [ georgewhewell ];
      };
    });

    # ROCm packages compile hundreds-to-thousands of HIP kernels via amdclang;
    # tag them so Hydra schedules them on a `big-parallel`-advertising builder.
    bigParallel =
      drv:
      drv.overrideAttrs (old: {
        requiredSystemFeatures = (old.requiredSystemFeatures or [ ]) ++ [ "big-parallel" ];
      });

    setPname =
      pname: drv:
      drv.overrideAttrs (_: {
        inherit pname;
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

    llama-cpp-rocm = bigParallel (
      setPname "llama-cpp-rocm-${suffix}" (
        withRdmaRpc (georgewhewellMaintained (prev.llama-cpp.override rocmOverride))
      )
    );
    llama-cpp-vulkan = withRdmaRpc (georgewhewellMaintained (prev.llama-cpp.override vulkanOverride));
    llama-cpp-cuda = georgewhewellMaintained (prev.llama-cpp.override cudaOverride);
    llama-cpp-master = llamaCppMaster;
    llama-cpp-master-rocm = bigParallel (
      setPname "llama-cpp-master-rocm-${suffix}" (
        withRdmaRpc (georgewhewellMaintained (llamaCppMaster.override rocmOverride))
      )
    );
    llama-cpp-master-vulkan = withRdmaRpc (
      georgewhewellMaintained (llamaCppMaster.override vulkanOverride)
    );
    llama-cpp-master-cuda = georgewhewellMaintained (llamaCppMaster.override cudaOverride);
  }
  // lib.optionalAttrs supportsTherockRocm {
    ds4-rocm = bigParallel (
      prev.callPackage ../pkgs/ds4-rocm {
        src = inputs.ds4-hip;
        rocmSdk = final."therock-rocm-${suffix}";
        version = inputVersion "experimental" inputs.ds4-hip;
        inherit (rocmTarget) packageSuffix;
        offloadArch = firstBuildTarget;
        hsaOverrideGfxVersion = rocmTarget.hsaOverride or null;
      }
    );

    mlx-rocm = bigParallel (
      final.python3Packages.callPackage ../pkgs/mlx/rocm.nix {
        pname = "mlx-rocm-${suffix}";
        rdma-core = final.rdma-core-usb4;
        rocmPackages = final.therockRocmPackages.${firstBuildTarget};
        gfx = firstBuildTarget;
      }
    );

    # Unsuffixed aliases for the active rocmTarget. Lets consumers write
    # `pkgs.therock-rocm` instead of `pkgs.therock-rocm-gfx1151` once the
    # package set's target is fixed (which it is per pkgsFor invocation).
    therock-rocm = final."therock-rocm-${suffix}";
    therock-rocm-env = final."therock-rocm-${suffix}-env";
  }
  // lib.optionalAttrs supportsTherockPython {
    therock-python = final."therock-python-${suffix}";
    therock-python-wheels = final."therock-python-wheels-${suffix}";
    therock-amdsmi = final."therock-amdsmi-${suffix}";
    torch-rocm = final."torch-rocm-${suffix}";
  }
  // lib.optionalAttrs (supportsTherockRocm && supportsTherockPython) {
    vllm-rocm = final."vllm-rocm-therock-${suffix}";
  }
)
