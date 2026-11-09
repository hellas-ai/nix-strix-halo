{
  inputs,
  lib,
  mediaRuntime ? false,
  inputVersion,
  rocmTarget,
  rocmProvider ? "therock-bin",
  pythonProvider ? "therock-wheels",
  therockPythonConfig ? import ../pkgs/therock/python-config.nix { inherit lib; },
  sources ? null,
  legacyRuntime ? false,
}:

# Locally-defined packages. One overlay, one source of truth for "what
# this flake adds on top of nixpkgs". rocm/python deps come from
# `final.rocmPackages` / `final.python3Packages`, so the consumer
# controls those via the rocm.nix / python.nix overlays applied earlier.
#
# Single output per logical package — `llama-cpp-rocm`, `sglang-rocm`,
# etc. — narrowed to `rocmTarget`. Building for a different target is a
# matter of composing this overlay with a different `rocmTarget`;
# per-target suffix attrs fall out of `legacyPackages.<system>.<suffix>`
# at the flake level rather than being baked into the overlay.

final: prev:

let
  suffix = rocmTarget.packageSuffix;
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

  rocmProviderHasTherockAttrs = rocmProvider == "therock-bin";
  pythonProviderHasTherockAttrs = pythonProvider == "therock-wheels";
  supportsTherockRocm =
    rocmProviderHasTherockAttrs
    && (if sources != null then sourceHasTherockRocm else finalHas "therock-rocm-${suffix}");
  supportsTherockPython =
    pythonProviderHasTherockAttrs
    && (if sources != null then sourceHasTherockPython else finalHas "therock-python-wheels-${suffix}");

  linuxPackages = lib.optionalAttrs prev.stdenv.hostPlatform.isLinux (
    let
      ecPackages = prev.callPackage ../pkgs/ec-su-axb35.nix {
        ec-su-axb35-src = inputs.ec-su-axb35;
      };
    in
    {
      amduprof = prev.callPackage ../pkgs/amduprof { };
      amduprof-power-profiler =
        { kernel }:
        prev.callPackage ../pkgs/amduprof/power-profiler.nix {
          inherit kernel;
          inherit (final) amduprof;
        };
      amdgpu-smu-exporter = prev.callPackage ../pkgs/amdgpu-smu-exporter { };
      amd-npu-exporter = prev.callPackage ../pkgs/amd-npu-exporter { };

      ec-su-axb35 = ecPackages.kernelModule;
      ec-su-axb35-monitor = ecPackages.monitor;
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

    }
    // lib.optionalAttrs (legacyRuntime && supportsTherockRocm) {
      libfabric = prev.libfabric.overrideAttrs (old: {
        buildInputs = (old.buildInputs or [ ]) ++ [ final.rdma-core ];
        configureFlags = (old.configureFlags or [ ]) ++ [ "--enable-verbs=yes" ];
      });

      # nixpkgs configures aws-ofi-nccl for CUDA and the AWS/EFA platform,
      # which makes the plugin try to initialize libcuda even when RCCL loads
      # it. Build against the exact TheRock SDK used by torch/RCCL instead;
      # mixing another ROCm's libamdhip64 here causes runtime symbol failures.
      # The SDK also bundles RCCL's network-plugin headers under include/rccl.
      aws-ofi-nccl =
        let
          rocmSdk = final."therock-rocm-${suffix}";
        in
        prev.aws-ofi-nccl.overrideAttrs (
          old:
          let
            oldEnv = old.env or { };
          in
          {
            configureFlags =
              lib.filter (flag: flag != "--enable-platform-aws" && !(lib.hasPrefix "--with-cuda=" flag)) (
                old.configureFlags or [ ]
              )
              ++ [ "--with-rocm=${rocmSdk}" ];

            # Keep every non-CUDA dependency from the nixpkgs derivation, but
            # replace cuda_cudart and CUDA NCCL with TheRock's combined ROCm +
            # RCCL runtime/development SDK. libuuid is util-linux's dev output.
            buildInputs = [
              final.numactl
              final.libuuid
              final.rdma-core-usb4
              final.libfabric
              final.hwloc
              rocmSdk
            ];

            # The CUDA package also carries a hook that adds the host GPU
            # driver search path to every ELF. ROCm resolves its userspace
            # runtime from rocmSdk, so retain only the autotools hook here.
            nativeBuildInputs = [ final.autoreconfHook ];

            # TheRock intentionally has split lib/ and lib64/ trees. Upstream's
            # ROCm probe prefers lib64 when it exists, although libamdhip64 and
            # libhsa-runtime64 live in lib, so make their real location explicit.
            env = oldEnv // {
              CPPFLAGS = lib.concatStringsSep " " (
                lib.filter (value: value != "") [
                  (oldEnv.CPPFLAGS or "")
                  "-I${rocmSdk}/include/rccl"
                ]
              );
              LDFLAGS = lib.concatStringsSep " " (
                lib.filter (value: value != "") [
                  (oldEnv.LDFLAGS or "")
                  "-L${rocmSdk}/lib"
                ]
              );
            };
          }
        );

    }
    // lib.optionalAttrs supportsTherockRocm {
      # Unsuffixed aliases for the active rocmTarget. Lets consumers write
      # `pkgs.therock-rocm` instead of `pkgs.therock-rocm-gfx1151` once the
      # package set's target is fixed (which it is per pkgsFor invocation).
      therock-rocm = final."therock-rocm-${suffix}";
    }
    // lib.optionalAttrs supportsTherockPython {
      therock-python-wheels = final."therock-python-wheels-${suffix}";
      torch-rocm = final."torch-rocm-${suffix}";
    }
    // lib.optionalAttrs (supportsTherockRocm && supportsTherockPython) {
      sglang-rocm = prev.callPackage ../pkgs/sglang {
        inherit mediaRuntime;
        buildFromSource = legacyRuntime;
        pythonPackages = final.${therockPythonConfig.packagesAttr};
        rocmSdk = final."therock-rocm-${suffix}";
        inherit (rocmTarget) packageSuffix;
        hsaOverrideGfxVersion = rocmTarget.hsaOverride or null;
      };
      sglang-rocm-h3 = prev.callPackage ../pkgs/sglang-h3 {
        pythonPackages = final.${therockPythonConfig.packagesAttr};
        sglangRocm = final.sglang-rocm;
      };
    }
    // lib.optionalAttrs (supportsTherockRocm && supportsTherockPython && suffix == "gfx1151") {
      # GLM-5.3-Flash runtime with the retained compatibility patches and opt-in MoE prefill implementation.
      sglang-glm53-rocm = prev.callPackage ../pkgs/sglang/glm53 {
        pythonPackages = final.${therockPythonConfig.packagesAttr};
        rocmSdk = final."therock-rocm-${suffix}";
        inherit (final) sglang-rocm mscclpp-rocm;
      };
      glm53-wmma-aot = prev.callPackage ../pkgs/sglang/glm53/aot { inherit (final) sglang-glm53-rocm; };
      glm53-aot-stage = prev.callPackage ../pkgs/sglang/glm53/aot/stage.nix { };
      mscclpp-rocm = prev.callPackage ../pkgs/mscclpp {
        rocmSdk = final."therock-rocm-${suffix}";
      };
    }
    // lib.optionalAttrs (supportsTherockRocm && suffix == "gfx1151") (
      let
        # Experimental retained-PM4 HIP/HSA runtimes (see pkgs/rocm-pm4-split). Nothing deployed
        # references them; they only build when asked for by name.
        therockSourceTree = import ../pkgs/therock/sources/source-tree.nix { inherit inputs; };
        rocmSystems =
          lib.findFirst (m: m.path == "rocm-systems")
            (throw "rocm-pm4-split: no rocm-systems entry in pkgs/therock/sources/source-tree.nix")
            therockSourceTree.${suffix}.submodules;
        rocmPm4 = prev.callPackage ../pkgs/rocm-pm4-split {
          rocmSdk = final."therock-rocm-${suffix}";
          baseSource = rocmSystems.source;
        };
      in
      {
        rocm-pm4-bootstrap = rocmPm4.bootstrap;
        rocm-pm4-clr-split = rocmPm4.clrSplit;
        rocm-pm4-prepared-source = rocmPm4.prepared;
        rocm-pm4-rocr = rocmPm4.rocr;
        rocm-pm4-rocr-split = rocmPm4.rocrSplit;
        # Launcher for this checkout's GLM runtime, with optional PM4 replay.
        glm53-serve = prev.callPackage ../pkgs/sglang/glm53/serve.nix {
          inherit (final)
            sglang-glm53-rocm
            therock-python-wheels
            glm53-wmma-aot
            glm53-aot-stage
            ;
          rocm-pm4-clr-split = rocmPm4.clrSplit;
          rocm-pm4-rocr-split = rocmPm4.rocrSplit;
          rocm-pm4-bootstrap = rocmPm4.bootstrap;
        };
      }
    )
  );
in
linuxPackages
// lib.optionalAttrs mediaRuntime {
  # The video module's Vulkan runner retains ae37's RPC/no-UI package identity.
  llama-cpp = (prev.llama-cpp.override { rpcSupport = true; }).overrideAttrs (old: {
    npmDeps = null;
    nativeBuildInputs = lib.filter (x: x.pname or "" != "npm-config-hook") (
      old.nativeBuildInputs or [ ]
    );
    preConfigure = ''
      prependToVar cmakeFlags "-DLLAMA_BUILD_COMMIT:STRING=$(cat COMMIT)"
    '';
    cmakeFlags = (old.cmakeFlags or [ ]) ++ [ (lib.cmakeBool "LLAMA_BUILD_UI" false) ];
    postFixup = old.postFixup or "";
  });
}
