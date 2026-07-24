{
  inputs,
  lib,
  inputVersion,
  rocmTarget,
  rocmProvider ? "therock-bin",
  pythonProvider ? "therock-wheels",
  therockPythonConfig ? import ../pkgs/therock/python-config.nix { inherit lib; },
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

let
  suffix = rocmTarget.packageSuffix;
  firstBuildTarget = builtins.head rocmTarget.buildTargets;

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
  mlxNanobind = final.python3Packages.callPackage ../pkgs/mlx/nanobind-2_13.nix { };
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

  fixDarwinLlamaRpcAlias =
    drv:
    drv.overrideAttrs (old: {
      postFixup =
        (old.postFixup or "")
        + lib.optionalString prev.stdenv.hostPlatform.isDarwin ''
          for bin in "$out"/bin/rpc-server "$out"/bin/llama-rpc-server "$out"/bin/ggml-rpc-server; do
            if [ -x "$bin" ] && ! otool -l "$bin" | grep -Fq "$out/lib"; then
              install_name_tool -add_rpath "$out/lib" "$bin"
            fi
          done
        '';
    });

  withoutLlamaUi =
    drv:
    drv.overrideAttrs (old: {
      npmDeps = null;
      nativeBuildInputs = lib.filter (x: x.pname or "" != "npm-config-hook") (
        old.nativeBuildInputs or [ ]
      );
      preConfigure = ''
        prependToVar cmakeFlags "-DLLAMA_BUILD_COMMIT:STRING=$(cat COMMIT)"
      '';
      cmakeFlags = (old.cmakeFlags or [ ]) ++ [
        (lib.cmakeBool "LLAMA_BUILD_UI" false)
      ];
    });

  normalizeLlamaRpcAlias =
    drv:
    drv.overrideAttrs (old: {
      postInstall =
        lib.replaceStrings
          [ "cp bin/rpc-server $out/bin/llama-rpc-server" ]
          [
            ''
              rpc_server_bin=
              for bin in bin/rpc-server bin/llama-rpc-server bin/ggml-rpc-server; do
                if [ -x "$bin" ]; then
                  rpc_server_bin="$bin"
                  break
                fi
              done
              if [ -z "$rpc_server_bin" ]; then
                echo "could not find llama.cpp RPC server binary" >&2
                exit 1
              fi
              cp "$rpc_server_bin" "$out/bin/llama-rpc-server"
            ''
          ]
          (old.postInstall or "");
    });

  localLlamaCpp =
    args:
    georgewhewellMaintained (
      fixDarwinLlamaRpcAlias (normalizeLlamaRpcAlias (withoutLlamaUi (prev.llama-cpp.override args)))
    );

  llamaCpp = localLlamaCpp { rpcSupport = true; };

  masterRev = inputs.llama-cpp-master.shortRev or inputs.llama-cpp-master.rev or "unknown";
  # llama-cpp-master is the upstream HEAD variant. Override is via
  # `.overrideAttrs` (the package doesn't expose `src` through .override),
  # but `.override` composes through it — `llamaCppMaster.override
  # { rocmSupport = true; ... }` re-runs nixpkgs' llama-cpp with new
  # args and re-applies these attrs on top, so the rocm/vulkan/cuda
  # variants of master fall out of one master-src definition.
  llamaCppMaster = llamaCpp.overrideAttrs (old: {
    pname = "llama-cpp-master";
    version = "master-${masterRev}";
    src = inputs.llama-cpp-master;
    preConfigure = ''
      prependToVar cmakeFlags "-DLLAMA_BUILD_COMMIT:STRING=${masterRev}"
    '';
    cmakeFlags = (old.cmakeFlags or [ ]) ++ [
      (lib.cmakeFeature "LLAMA_BUILD_NUMBER" "0")
    ];
    patches = (old.patches or [ ]) ++ [
      ../pkgs/llama-cpp/patches/0001-rpc-rdma-configurable-chunk-size.patch
      ../pkgs/llama-cpp/patches/0002-rpc-rdma-darwin-librdma.patch
      ../pkgs/llama-cpp/patches/0003-rpc-rdma-log-probe-and-avoid-darwin-fallback-hang.patch
      ../pkgs/llama-cpp/patches/0004-rpc-rdma-selectable-uc-qp.patch
      ../pkgs/llama-cpp/patches/0005-rpc-rdma-remote-lid-env.patch
      ../pkgs/llama-cpp/patches/0006-rpc-rdma-mtu-and-uc-init-shape.patch
      ../pkgs/llama-cpp/patches/0007-rpc-rdma-psn-env.patch
      ../pkgs/llama-cpp/patches/0008-rpc-rdma-trace-io.patch
      ../pkgs/llama-cpp/patches/0009-rpc-rdma-optional-shared-cq.patch
      ../pkgs/llama-cpp/patches/0010-rpc-rdma-fixed-frame-stream.patch
      ../pkgs/llama-cpp/patches/0011-rpc-keep-registry-socket-alive.patch
      ../pkgs/llama-cpp/patches/0012-rpc-rdma-configurable-rx-depth.patch
      ../pkgs/llama-cpp/patches/0013-rpc-server-one-shot-env.patch
      ../pkgs/llama-cpp/patches/0014-rpc-rdma-detect-tcp-close-while-polling.patch
      ../pkgs/llama-cpp/patches/0015-rpc-rdma-fixed-frame-tx-ring.patch
      ../pkgs/llama-cpp/patches/0016-rpc-rdma-configurable-tx-depth.patch
      ../pkgs/llama-cpp/patches/0017-rpc-rdma-fixed-frame-ack.patch
      ../pkgs/llama-cpp/patches/0018-rpc-rdma-fixed-frame-cookie.patch
      ../pkgs/llama-cpp/patches/0019-rpc-rdma-peer-device-map.patch
      ../pkgs/llama-cpp/patches/0020-rpc-rdma-fixed-frame-ack-window.patch
      ../pkgs/llama-cpp/patches/0021-rpc-cache-atomic-set-tensor.patch
    ];
  });

  # Compose Strix llama.cpp variants with the thunderbolt-ibverbs
  # rdma-core overlay so ggml's RPC backend builds its RDMA transport.
  withRdmaRpc =
    drv:
    drv.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ [ final.rdma-core-usb4 ];
      cmakeFlags = (old.cmakeFlags or [ ]) ++ [
        (lib.cmakeBool "GGML_RPC_RDMA" true)
      ];
    });

  # llama.cpp otherwise prefers a host /opt/rocm when one is visible. Apart
  # from making the build impure, that can mix a host HIP runtime with the
  # target-narrowed nixpkgs ROCm libraries selected above.
  withHermeticRocm =
    drv:
    drv.overrideAttrs (_: {
      env.ROCM_PATH = final.rocmPackages.clr;
    });

  # macOS 26 exposes Apple Thunderbolt RDMA through the SDK's infiniband
  # headers and umbrella librdma.dylib. This is only for Darwin builds; Linux
  # continues to use rdma-core-usb4 above.
  withDarwinRdmaRpc =
    drv:
    drv.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ [ final.apple-sdk_26 ];
      cmakeFlags = (old.cmakeFlags or [ ]) ++ [
        (lib.cmakeBool "GGML_RPC_RDMA" true)
      ];
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

  commonPackages = {
    llama-cpp = llamaCpp;
    llama-cpp-master = llamaCppMaster;
  };

  darwinPackages = lib.optionalAttrs prev.stdenv.isDarwin {
    llama-cpp-master-rdma = setPname "llama-cpp-master-rdma" (withDarwinRdmaRpc llamaCppMaster);
  };

  linuxPackages = lib.optionalAttrs prev.stdenv.isLinux (
    let
      ecPackages = prev.callPackage ../pkgs/ec-su-axb35.nix {
        ec-su-axb35-src = inputs.ec-su-axb35;
      };
    in
    {
      amdgpu-smu-exporter = prev.callPackage ../pkgs/amdgpu-smu-exporter { };
      llm-inference-bench = final.callPackage ../pkgs/llm-inference-bench { };

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

      mlirAiePackages = prev.callPackage ../pkgs/mlir-aie {
        inherit (final) xrt;
      };
      llvm-aie = final.mlirAiePackages.llvm-aie;
      mlir-aie = final.mlirAiePackages.mlir-aie;
      mlir-aie-env = final.mlirAiePackages.mlir-aie-env;

      fastflowlm = prev.callPackage ../pkgs/fastflowlm {
        inherit (final) tokenizers-cpp xrt;
        src = inputs.fastflowlm;
      };

      llama-cpp-rocm = bigParallel (
        setPname "llama-cpp-rocm-${suffix}" (withRdmaRpc (withHermeticRocm (localLlamaCpp rocmOverride)))
      );
      llama-cpp-vulkan = withRdmaRpc (localLlamaCpp vulkanOverride);
      llama-cpp-cuda = localLlamaCpp cudaOverride;
      llama-cpp-master-rocm = bigParallel (
        setPname "llama-cpp-master-rocm-${suffix}" (
          withRdmaRpc (withHermeticRocm (llamaCppMaster.override rocmOverride))
        )
      );
      llama-cpp-master-vulkan = withRdmaRpc (llamaCppMaster.override vulkanOverride);
      llama-cpp-master-cuda = llamaCppMaster.override cudaOverride;
    }
    // lib.optionalAttrs supportsTherockRocm {
      mlx-rocm = bigParallel (
        final.python3Packages.callPackage ../pkgs/mlx/rocm.nix {
          mlx = final.python3Packages.mlx.override {
            nanobind = mlxNanobind;
          };
          inherit (inputs) mlx-src;
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
    // lib.optionalAttrs (supportsTherockRocm && rocmTarget.supportsRocWmma) {
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
    }
    // lib.optionalAttrs supportsTherockPython {
      therock-python = final."therock-python-${suffix}";
      therock-python-wheels = final."therock-python-wheels-${suffix}";
      therock-amdsmi = final."therock-amdsmi-${suffix}";
      torch-rocm = final."torch-rocm-${suffix}";
    }
    // lib.optionalAttrs (supportsTherockRocm && supportsTherockPython) {
      sglang-rocm = prev.callPackage ../pkgs/sglang {
        pythonPackages = final.${therockPythonConfig.packagesAttr};
        rocmSdk = final."therock-rocm-${suffix}";
        inherit (rocmTarget) packageSuffix;
        hsaOverrideGfxVersion = rocmTarget.hsaOverride or null;
      };
      vllm-rocm = final."vllm-rocm-therock-${suffix}";
    }
  );
in
commonPackages // darwinPackages // linuxPackages
