{
  lib,
  nixpkgs-vllm,
  vllmSources,
  vllmVersion,
}:

let
  appendCompileFlags =
    flags: pkg:
    pkg.overrideAttrs (old: {
      env = (old.env or { }) // {
        NIX_CFLAGS_COMPILE_AFTER = (old.env.NIX_CFLAGS_COMPILE_AFTER or "") + flags;
      };
    });

  appendPythonCompileFlags =
    flags: pkg:
    pkg.overridePythonAttrs (old: {
      env = (old.env or { }) // {
        NIX_CFLAGS_COMPILE_AFTER = (old.env.NIX_CFLAGS_COMPILE_AFTER or "") + flags;
      };
    });

  skipPythonChecks =
    pkg:
    pkg.overridePythonAttrs (_: {
      doCheck = false;
      doInstallCheck = false;
    });

  systemConfig = {
    x86_64-linux = "x86_64-unknown-linux-gnu";
  };
in
rec {
  mkRocmHardware =
    {
      name ? lib.concatStringsSep "-" gpuTargets,
      gpuTargets,
      localGpuTargets ? gpuTargets,
      composableKernelTargets ? localGpuTargets,
      aotritonTargets ? gpuTargets,
      runtimeEnv ? { },
    }:
    {
      inherit
        name
        gpuTargets
        localGpuTargets
        composableKernelTargets
        aotritonTargets
        runtimeEnv
        ;
      accelerator = "rocm";
    };

  mkCudaHardware =
    {
      name ? "cuda",
      cudaCapabilities,
      cudaPackagesAttr ? "cudaPackages_13",
      cudaForwardCompat ? false,
      runtimeEnv ? { },
    }:
    {
      inherit
        name
        cudaCapabilities
        cudaPackagesAttr
        cudaForwardCompat
        runtimeEnv
        ;
      accelerator = "cuda";
    };

  hardwareProfiles = {
    none = {
      name = "cpu";
      accelerator = "none";
      runtimeEnv = { };
    };

    gfx1151 = mkRocmHardware {
      name = "gfx1151";
      gpuTargets = [ "gfx1151" ];
      localGpuTargets = [
        "gfx1151"
        "gfx90a"
      ];
      composableKernelTargets = [
        "gfx90a"
        "gfx11-generic"
      ];
      runtimeEnv = {
        HSA_ENABLE_INTERRUPT = "0";
        HSA_NO_SCRATCH_RECLAIM = "1";
        HSA_OVERRIDE_GFX_VERSION = "11.5.1";
      };
    };

    rtx4090 = mkCudaHardware {
      name = "rtx4090";
      cudaCapabilities = [ "8.9" ];
    };
  };

  mkRocmOverlay =
    {
      hardware ? hardwareProfiles.gfx1151,
    }:
    final: prev:
    let
      inherit (hardware)
        gpuTargets
        localGpuTargets
        composableKernelTargets
        aotritonTargets
        ;
      narrow = drv: drv.override { inherit gpuTargets; };
      aotritonTargetString = lib.concatStringsSep ";" aotritonTargets;
    in
    {
      rocmPackages = prev.rocmPackages.overrideScope (
        _: rocmPrev: {
          clr = rocmPrev.clr.override {
            inherit localGpuTargets;
          };

          rccl = narrow rocmPrev.rccl;
          hipblaslt = narrow rocmPrev.hipblaslt;
          hipfft = narrow rocmPrev.hipfft;
          hiprand = narrow rocmPrev.hiprand;
          hipsparse = narrow rocmPrev.hipsparse;
          miopen = narrow rocmPrev.miopen;
          rocblas = narrow rocmPrev.rocblas;
          rocfft = narrow rocmPrev.rocfft;
          rocrand = narrow rocmPrev.rocrand;
          rocsolver = narrow rocmPrev.rocsolver;
          rocsparse = narrow rocmPrev.rocsparse;

          composable_kernel_base = rocmPrev.composable_kernel_base.override {
            gpuTargets = composableKernelTargets;
          };

          aotriton = rocmPrev.aotriton.overrideAttrs (old: {
            cmakeFlags =
              (final.lib.filter (flag: !final.lib.hasPrefix "-DAOTRITON_TARGET_ARCH" flag) (
                old.cmakeFlags or [ ]
              ))
              ++ [
                "-DAOTRITON_TARGET_ARCH:STRING=${aotritonTargetString}"
              ];
          });
        }
      );
    };

  mkCudaOverlay =
    {
      hardware ? hardwareProfiles.rtx4090,
    }:
    _: prev: {
      cudaPackages = prev.${hardware.cudaPackagesAttr};
    };

  mkVllmOverlay =
    {
      hardware ? hardwareProfiles.none,
    }:
    _: prev:
    let
      accelerator = hardware.accelerator or "none";
    in
    {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (
          pyFinal: pyPrev:
          {
            onnx-ir = pyPrev.onnx-ir.overridePythonAttrs (_: {
              doCheck = false;
              doInstallCheck = false;
            });

            vllm = pyFinal.callPackage ../pkgs/vllm {
              inherit vllmSources vllmVersion;
              inherit (pyPrev) vllm;
            };
          }
          // lib.optionalAttrs (accelerator == "rocm") {
            torch = pyPrev.torch.override {
              triton = pyFinal.triton-no-cuda;
              rocmSupport = true;
              cudaSupport = false;
              inherit (hardware) gpuTargets;
            };

            amd-aiter = pyFinal.callPackage ../pkgs/amd-aiter {
              inherit (pyPrev) amd-aiter;
            };

            conch-triton-kernels = pyFinal.callPackage ../pkgs/vllm/extras/conch-triton-kernels.nix { };
          }
          // lib.optionalAttrs (accelerator == "cuda") {
            torch = pyPrev.torch.override {
              cudaSupport = true;
              rocmSupport = false;
            };
          }
        )
      ];
    };

  mkCpuTuningOverlay =
    {
      preferVectorWidth512 ? true,
    }:
    _: prev:
    let
      vectorFlags = lib.optionalString preferVectorWidth512 " -mprefer-vector-width=512";
      tune = pkg: if vectorFlags == "" then pkg else appendCompileFlags vectorFlags pkg;
      tunePy = pkg: if vectorFlags == "" then pkg else appendPythonCompileFlags vectorFlags pkg;
    in
    {
      openblas = tune prev.openblas;

      libtpms = prev.libtpms.overrideAttrs (old: {
        env = (old.env or { }) // {
          NIX_CFLAGS_COMPILE = (old.env.NIX_CFLAGS_COMPILE or "") + " -Wno-error=stringop-overflow";
        };
      });

      meson = prev.meson.overrideAttrs (_: {
        doCheck = false;
        doInstallCheck = false;
      });

      valkey = prev.valkey.overrideAttrs (_: {
        doCheck = false;
        doInstallCheck = false;
      });

      rapidjson = prev.rapidjson.overrideAttrs (_: {
        doCheck = false;
        doInstallCheck = false;
      });

      protobuf = prev.protobuf.overrideAttrs (old: {
        env = (old.env or { }) // {
          NIX_CFLAGS_COMPILE_AFTER = (old.env.NIX_CFLAGS_COMPILE_AFTER or "") + " -O2";
        };
        doCheck = false;
        doInstallCheck = false;
        cmakeFlags = (old.cmakeFlags or [ ]) ++ [
          "-Dprotobuf_BUILD_TESTS=OFF"
        ];
      });

      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (_: pyPrev: {
          torch = tunePy pyPrev.torch;
          numpy = tunePy pyPrev.numpy;

          capturer = pyPrev.capturer.overridePythonAttrs (_: {
            doCheck = false;
          });
          numcodecs = skipPythonChecks pyPrev.numcodecs;
          scipy = skipPythonChecks pyPrev.scipy;
          astropy = skipPythonChecks pyPrev.astropy;
          torchcodec = skipPythonChecks pyPrev.torchcodec;
          torchaudio = skipPythonChecks pyPrev.torchaudio;

          grpcio = pyPrev.grpcio.overridePythonAttrs (old: {
            env = (old.env or { }) // {
              NIX_CFLAGS_COMPILE_AFTER = (old.env.NIX_CFLAGS_COMPILE_AFTER or "") + " -O2";
            };
          });
        })
      ];
    };

  mkPackageSet =
    {
      system ? "x86_64-linux",
      nixpkgsInput ? nixpkgs-vllm,
      hardware ? hardwareProfiles.none,
      cpu ? null,
      tuneCpuPackages ? cpu != null,
      extraOverlays ? [ ],
      config ? { },
    }:
    let
      accelerator = hardware.accelerator or "none";
      baseConfig = {
        allowUnfree = true;
      }
      // lib.optionalAttrs (accelerator == "cuda") {
        cudaSupport = true;
        inherit (hardware) cudaCapabilities;
        cudaForwardCompat = hardware.cudaForwardCompat or false;
      };
    in
    import nixpkgsInput (
      {
        config = lib.recursiveUpdate baseConfig config;
        overlays =
          lib.optionals (accelerator == "rocm") [
            (mkRocmOverlay { inherit hardware; })
          ]
          ++ lib.optionals (accelerator == "cuda") [
            (mkCudaOverlay { inherit hardware; })
          ]
          ++ [
            (mkVllmOverlay { inherit hardware; })
          ]
          ++ lib.optionals tuneCpuPackages [
            (mkCpuTuningOverlay { })
          ]
          ++ extraOverlays;
      }
      // (
        if cpu == null then
          { inherit system; }
        else
          {
            localSystem = {
              config = systemConfig.${system} or system;
              gcc = {
                arch = cpu;
                tune = cpu;
              };
            };
          }
      )
    );

  mkVllmPackage =
    {
      pkgs,
      hardware ? hardwareProfiles.none,
      tunePackage ? false,
    }:
    let
      accelerator = hardware.accelerator or "none";
      package = pkgs.python3Packages.vllm.override (
        {
          cudaSupport = false;
          rocmSupport = false;
        }
        // lib.optionalAttrs (accelerator == "rocm") {
          rocmSupport = true;
          inherit (hardware) gpuTargets;
        }
        // lib.optionalAttrs (accelerator == "cuda") {
          cudaSupport = true;
        }
      );
    in
    if tunePackage then appendPythonCompileFlags " -mprefer-vector-width=512" package else package;

  wrapRuntimeEnv =
    {
      pkgs,
      env,
      hardware ? hardwareProfiles.none,
      name ? "vllm-env-${hardware.name or "cpu"}",
    }:
    let
      runtimeEnv = hardware.runtimeEnv or { };
      wrapperArgs = lib.concatStringsSep " " (
        lib.mapAttrsToList (
          key: value: "--set ${lib.escapeShellArg key} ${lib.escapeShellArg (toString value)}"
        ) runtimeEnv
      );
    in
    if runtimeEnv == { } then
      env
    else
      pkgs.symlinkJoin {
        inherit name;
        paths = [ env ];
        nativeBuildInputs = [ pkgs.makeWrapper ];
        postBuild = ''
          for bin in "$out"/bin/*; do
            [[ -L "$bin" ]] || continue
            target=$(readlink -f "$bin")
            rm "$bin"
            makeWrapper "$target" "$bin" ${wrapperArgs}
          done
        '';
        passthru =
          (env.passthru or { })
          // lib.optionalAttrs (env ? python) {
            inherit (env) python;
          }
          // {
            unwrapped = env;
          };
      };

  mkVllmEnv =
    {
      pkgs,
      hardware ? hardwareProfiles.none,
      tunePackage ? false,
      withRay ? true,
      name ? "vllm-env-${hardware.name or "cpu"}",
    }:
    let
      vllm = mkVllmPackage {
        inherit pkgs hardware tunePackage;
      };
      env = pkgs.python3.withPackages (
        ps:
        [
          vllm
        ]
        ++ lib.optionals withRay [
          ps.ray
        ]
      );
    in
    wrapRuntimeEnv {
      inherit
        pkgs
        env
        hardware
        name
        ;
    };
}
