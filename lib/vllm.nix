{
  lib,
  nixpkgs,
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

  skipChecks =
    pkg:
    pkg.overrideAttrs (_: {
      doCheck = false;
      doInstallCheck = false;
    });

in
rec {
  mkRocmHardware =
    {
      name ? lib.concatStringsSep "-" gpuTargets,
      gpuTargets,
      localGpuTargets ? gpuTargets,
      composableKernelTargets ? localGpuTargets,
      composableKernelFallbackTarget ?
        if composableKernelTargets == [ ] then null else builtins.head composableKernelTargets,
      aotritonTargets ? gpuTargets,
      magmaTargets ? gpuTargets,
      enableRcclMscclKernel ? false,
      runtimeEnv ? { },
    }:
    {
      inherit
        name
        gpuTargets
        localGpuTargets
        composableKernelTargets
        composableKernelFallbackTarget
        aotritonTargets
        magmaTargets
        enableRcclMscclKernel
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
      localGpuTargets = [ "gfx1151" ];
      composableKernelTargets = [
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
        composableKernelFallbackTarget
        aotritonTargets
        magmaTargets
        enableRcclMscclKernel
        ;
      narrow = drv: drv.override { inherit gpuTargets; };
      aotritonTargetString = lib.concatStringsSep ";" aotritonTargets;
      gpuTargetString = lib.concatStringsSep ";" gpuTargets;
      gpuTargetSpace = lib.concatStringsSep " " gpuTargets;
      magmaTargetString = lib.concatStringsSep " " magmaTargets;
      magmaTargetNumbers = lib.concatMapStringsSep ";" (
        target: lib.removePrefix "gfx" target
      ) magmaTargets;
      magmaHipArchitectures = lib.concatStringsSep ";" magmaTargets;
      replaceCmakeFeatures =
        names: flags:
        let
          isReplaced =
            flag: lib.any (name: lib.hasPrefix "-D${name}" flag || lib.hasPrefix "-D${name}:" flag) names;
        in
        lib.filter (flag: !isReplaced flag) flags;
      overrideMagma =
        magma:
        (magma.override {
          rocmPackages = final.rocmPackages;
        }).overrideAttrs
          (old: {
            postPatch =
              (old.postPatch or "")
              + lib.optionalString (magmaTargetNumbers != "") ''
                if grep -Fq 'set(VALID_GFXS "700;701;702;703;704;705;801;802;803;805;810;900;902;904;906;908;909;90c;1010;1011;1012;1030;1031;1032;1033")' CMakeLists.txt; then
                  substituteInPlace CMakeLists.txt \
                    --replace-fail 'set(VALID_GFXS "700;701;702;703;704;705;801;802;803;805;810;900;902;904;906;908;909;90c;1010;1011;1012;1030;1031;1032;1033")' \
                                   'set(VALID_GFXS "700;701;702;703;704;705;801;802;803;805;810;900;902;904;906;908;909;90c;1010;1011;1012;1030;1031;1032;1033;${magmaTargetNumbers}")'
                fi

                if ! grep -Fq 'foreach(GPU_ARCH_FLAG ''${GPU_ARCH_FLAGS})' CMakeLists.txt \
                  && grep -Fq '#add_compile_options(' CMakeLists.txt; then
                  substituteInPlace CMakeLists.txt \
                    --replace-fail '#add_compile_options(' 'foreach(GPU_ARCH_FLAG ''${GPU_ARCH_FLAGS})
                    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:''${GPU_ARCH_FLAG}>)
                  endforeach()
                  #add_compile_options('
                fi
              '';
            cmakeFlags =
              replaceCmakeFeatures [
                "GPU_TARGET"
                "CMAKE_HIP_ARCHITECTURES"
              ] (old.cmakeFlags or [ ])
              ++ [
                (final.lib.cmakeFeature "GPU_TARGET" magmaTargetString)
                (final.lib.cmakeFeature "CMAKE_HIP_ARCHITECTURES" magmaHipArchitectures)
              ];
          });
      overrideComposableKernelBase =
        composableKernelBase:
        (composableKernelBase.override {
          gpuTargets = composableKernelTargets;
        }).overrideAttrs
          (old: {
            postPatch =
              (old.postPatch or "")
              + lib.optionalString (composableKernelFallbackTarget != null) ''
                if grep -Fq 'set(offload_targets "--offload-arch=gfx90a")' library/src/tensor_operation_instance/gpu/CMakeLists.txt; then
                  substituteInPlace library/src/tensor_operation_instance/gpu/CMakeLists.txt \
                    --replace-fail 'set(offload_targets "--offload-arch=gfx90a")' \
                                   'set(offload_targets "--offload-arch=${composableKernelFallbackTarget}")'
                fi
              '';
          });
      markUnbroken =
        drv:
        drv.overrideAttrs (old: {
          meta = (old.meta or { }) // {
            broken = false;
          };
        });
      overrideRccl =
        rccl:
        (narrow rccl).overrideAttrs (old: {
          cmakeFlags = replaceCmakeFeatures [ "ENABLE_MSCCL_KERNEL" ] (old.cmakeFlags or [ ]) ++ [
            (final.lib.cmakeBool "ENABLE_MSCCL_KERNEL" enableRcclMscclKernel)
          ];
        });
      overrideMscclpp =
        mscclpp:
        mscclpp.overrideAttrs (old: {
          postPatch =
            (old.postPatch or "")
            + lib.optionalString (gpuTargetSpace != "") ''
              if grep -Fq 'gfx908 gfx90a gfx942 gfx1030 gfx1100' CMakeLists.txt; then
                substituteInPlace CMakeLists.txt \
                  --replace-fail 'gfx908 gfx90a gfx942 gfx1030 gfx1100' '${gpuTargetSpace}'
              elif grep -Fq 'gfx90a gfx941 gfx942' CMakeLists.txt; then
                substituteInPlace CMakeLists.txt \
                  --replace-fail 'gfx90a gfx941 gfx942' '${gpuTargetSpace}'
              fi
            '';
          cmakeFlags =
            replaceCmakeFeatures [
              "GPU_TARGETS"
              "AMDGPU_TARGETS"
            ] (old.cmakeFlags or [ ])
            ++ [
              (final.lib.cmakeFeature "GPU_TARGETS" gpuTargetString)
              (final.lib.cmakeFeature "AMDGPU_TARGETS" gpuTargetString)
            ];
        });
    in
    {
      magma = overrideMagma prev.magma;
      magma-hip = overrideMagma prev.magma-hip;

      rocmPackages = prev.rocmPackages.overrideScope (
        rocmFinal: rocmPrev: {
          clr = rocmPrev.clr.override {
            inherit localGpuTargets;
          };

          rccl = overrideRccl rocmPrev.rccl;
          hipblaslt = narrow rocmPrev.hipblaslt;
          hipfft = narrow rocmPrev.hipfft;
          hiprand = narrow rocmPrev.hiprand;
          hipsparse = narrow rocmPrev.hipsparse;
          miopen = narrow rocmPrev.miopen;
          mscclpp = overrideMscclpp rocmPrev.mscclpp;
          rocblas = narrow rocmPrev.rocblas;
          rocfft = narrow rocmPrev.rocfft;
          rocrand = narrow rocmPrev.rocrand;
          rocsolver = narrow rocmPrev.rocsolver;
          rocsparse = narrow rocmPrev.rocsparse;

          composable_kernel_base = overrideComposableKernelBase rocmPrev.composable_kernel_base;
          composable_kernel = markUnbroken (
            rocmPrev.composable_kernel.override {
              composable_kernel_base = rocmFinal.composable_kernel_base;
            }
          );

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
      cpu ? null,
      preferVectorWidth512 ? cpu == "znver5",
    }:
    _: prev:
    let
      cpuFlags = lib.optionalString (cpu != null) " -march=${cpu} -mtune=${cpu}";
      vectorFlags = lib.optionalString preferVectorWidth512 " -mprefer-vector-width=512";
      compileFlags = cpuFlags + vectorFlags;
      tune = pkg: if compileFlags == "" then pkg else appendCompileFlags compileFlags pkg;
      tunePy = pkg: if compileFlags == "" then pkg else appendPythonCompileFlags compileFlags pkg;
      tuneFfmpeg =
        package:
        tune (
          package.override {
            withCuda = false;
            withCudaLLVM = false;
            withCudaNVCC = false;
            withCuvid = false;
            withNvcodec = false;
            withNvdec = false;
            withNvenc = false;
            withRav1e = false;
            withDocumentation = false;
            withHtmlDoc = false;
            withManPages = false;
            withPodDoc = false;
            withTxtDoc = false;
          }
        );
      acceleratorLlvmTargetCmakeFlags = [
        (lib.cmakeFeature "LLVM_TARGETS_TO_BUILD" "X86;AMDGPU;NVPTX")
      ];
      rustLlvmTargetCmakeFlags = [
        (lib.cmakeFeature "LLVM_TARGETS_TO_BUILD" "X86;WebAssembly;BPF")
      ];
      replaceCmakeFeatures =
        names: flags:
        let
          isReplaced =
            flag: lib.any (name: lib.hasPrefix "-D${name}" flag || lib.hasPrefix "-D${name}:" flag) names;
        in
        lib.filter (flag: !isReplaced flag) flags;
      limitLlvmTargets =
        flags: pkg:
        pkg.overrideAttrs (old: {
          cmakeFlags = replaceCmakeFeatures [ "LLVM_TARGETS_TO_BUILD" ] (old.cmakeFlags or [ ]) ++ flags;
        });
      limitLlvmPackageSet =
        flags: llvmPackages:
        llvmPackages.override {
          devExtraCmakeFlags = flags;
        };
      withoutInputsByName =
        names: inputs:
        lib.filter (
          input:
          let
            inputNames = [
              (input.pname or "")
              (input.name or "")
              (lib.getName input)
            ];
            matches =
              name:
              lib.any (
                inputName:
                inputName == name
                || lib.hasPrefix "${name}-" inputName
                || lib.hasSuffix "-${name}" inputName
                || lib.hasInfix "-${name}-" inputName
              ) inputNames;
          in
          !(lib.any matches names)
        ) inputs;
      openblasTarget =
        if cpu == null then
          null
        else
          {
            znver1 = "ZEN";
            znver2 = "ZEN";
            znver3 = "ZEN";
            znver4 = "ZEN";
            znver5 = "ZEN";
          }
          .${cpu} or null;
      tunedOpenblas =
        let
          package =
            if openblasTarget == null then
              prev.openblas
            else
              prev.openblas.override {
                target = openblasTarget;
                dynamicArch = false;
                enableAVX512 = preferVectorWidth512;
              };
        in
        skipChecks package;
      skipLlvmChecks =
        flags: llvmPackages:
        let
          apply =
            packageSet:
            packageSet.overrideScope (
              _: llvmPrev:
              {
                libllvm = skipChecks llvmPrev.libllvm;
                llvm = skipChecks llvmPrev.llvm;
              }
              // lib.optionalAttrs (llvmPrev ? clang) {
                clang = skipChecks llvmPrev.clang;
              }
              // lib.optionalAttrs (llvmPrev ? lld) {
                lld = skipChecks llvmPrev.lld;
              }
              // lib.optionalAttrs (llvmPrev ? compiler-rt) {
                compiler-rt = skipChecks llvmPrev.compiler-rt;
              }
            );
          scoped = apply (limitLlvmPackageSet flags llvmPackages);
        in
        scoped
        // {
          override = args: apply (limitLlvmPackageSet flags (llvmPackages.override args));
        };
    in
    {
      llama-cpp = tune prev.llama-cpp;
      ffmpeg = tuneFfmpeg prev.ffmpeg;
      ffmpeg-headless = tuneFfmpeg prev.ffmpeg-headless;
      openblas = tune tunedOpenblas;

      rdma-core = skipChecks (
        prev.rdma-core.overrideAttrs (old: {
          outputs = lib.filter (output: output != "man") (old.outputs or [ ]);
          nativeBuildInputs = withoutInputsByName [
            "docutils"
            "pandoc-cli"
          ] (old.nativeBuildInputs or [ ]);
          cmakeFlags = (old.cmakeFlags or [ ]) ++ [
            "-DNO_MAN_PAGES=1"
          ];
        })
      );

      libtiff = skipChecks (
        prev.libtiff.overrideAttrs (old: {
          outputs = lib.filter (
            output:
            !(lib.elem output [
              "doc"
              "man"
            ])
          ) (old.outputs or [ ]);
          nativeBuildInputs = withoutInputsByName [ "sphinx" ] (old.nativeBuildInputs or [ ]);
          cmakeFlags = (old.cmakeFlags or [ ]) ++ [
            "-Dtiff-docs=OFF"
          ];
        })
      );

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

      v4l-utils = prev.v4l-utils.overrideAttrs (old: {
        mesonFlags = replaceCmakeFeatures [ "bpf" ] (old.mesonFlags or [ ]) ++ [
          (lib.mesonOption "bpf" "disabled")
        ];
      });

      triton-llvm = skipChecks (
        limitLlvmTargets acceleratorLlvmTargetCmakeFlags (
          prev.triton-llvm.override {
            buildDocs = false;
            buildMan = false;
            buildTests = false;
          }
        )
      );

      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (_: pyPrev: {
          torch = tunePy pyPrev.torch;
          numpy = tunePy pyPrev.numpy;

          aiohappyeyeballs = (skipPythonChecks pyPrev.aiohappyeyeballs).overridePythonAttrs (old: {
            outputs = [ "out" ];
            "build-system" = withoutInputsByName [
              "furo"
              "myst-parser"
              "sphinx"
              "sphinx-hook"
            ] (old."build-system" or [ ]);
          });
          capturer = pyPrev.capturer.overridePythonAttrs (_: {
            doCheck = false;
          });
          curio = skipPythonChecks pyPrev.curio;
          frozenlist = skipPythonChecks pyPrev.frozenlist;
          deprecated = (skipPythonChecks pyPrev.deprecated).overridePythonAttrs (old: {
            outputs = [ "out" ];
            nativeBuildInputs = withoutInputsByName [
              "pytest-check-hook"
              "sphinx-hook"
            ] (old.nativeBuildInputs or [ ]);
          });
          django = skipPythonChecks pyPrev.django;
          einops = skipPythonChecks pyPrev.einops;
          httpcore = pyPrev.httpcore.overridePythonAttrs (old: {
            disabledTests = (old.disabledTests or [ ]) ++ [
              "test_connection_pool_timeout_during_response"
            ];
          });
          jsonschema = skipPythonChecks pyPrev.jsonschema;
          markdown-it-py = skipPythonChecks pyPrev.markdown-it-py;
          multidict = skipPythonChecks pyPrev.multidict;
          myst-parser = skipPythonChecks pyPrev.myst-parser;
          onnx-ir = skipPythonChecks pyPrev.onnx-ir;
          numcodecs = skipPythonChecks pyPrev.numcodecs;
          scipy = skipPythonChecks pyPrev.scipy;
          sniffio = skipPythonChecks pyPrev.sniffio;
          pyjwt = (skipPythonChecks pyPrev.pyjwt).overridePythonAttrs (old: {
            outputs = [ "out" ];
            nativeBuildInputs = withoutInputsByName [
              "pytest-check-hook"
              "sphinx-hook"
              "sphinx-rtd-theme"
            ] (old.nativeBuildInputs or [ ]);
          });
          pyopenssl = pyPrev.pyopenssl.overridePythonAttrs (old: {
            disabledTests = (old.disabledTests or [ ]) ++ [
              "test_moving_buffer_behavior"
            ];
          });
          requests-futures = skipPythonChecks pyPrev.requests-futures;
          astropy = skipPythonChecks pyPrev.astropy;
          tokenizers = skipPythonChecks pyPrev.tokenizers;
          torchcodec = skipPythonChecks pyPrev.torchcodec;
          torchaudio = skipPythonChecks pyPrev.torchaudio;
          wrapt = (skipPythonChecks pyPrev.wrapt).overridePythonAttrs (old: {
            outputs = [ "out" ];
            nativeBuildInputs = withoutInputsByName [
              "pytest-check-hook"
              "sphinx-hook"
              "sphinx-rtd-theme"
            ] (old.nativeBuildInputs or [ ]);
          });

          grpcio = pyPrev.grpcio.overridePythonAttrs (old: {
            env = (old.env or { }) // {
              NIX_CFLAGS_COMPILE_AFTER = (old.env.NIX_CFLAGS_COMPILE_AFTER or "") + " -O2";
            };
          });
          gguf = pyPrev.gguf.overridePythonAttrs (old: {
            propagatedBuildInputs = withoutInputsByName [ "pyside6" ] (old.propagatedBuildInputs or [ ]);
            dependencies = withoutInputsByName [ "pyside6" ] (old.dependencies or [ ]);
            postPatch = (old.postPatch or "") + ''
              substituteInPlace pyproject.toml \
                --replace-fail 'PySide6 = { version = "^6.9", python = ">=3.9,<3.14", optional = true }' "" \
                --replace-fail 'gui = ["PySide6"]' 'gui = []' \
                --replace-fail 'gguf-editor-gui = "gguf.scripts.gguf_editor_gui:main"' ""
            '';
          });
        })
      ];
    }
    // lib.optionalAttrs (prev ? rocmPackages) {
      rocmPackages = prev.rocmPackages.overrideScope (
        _: rocmPrev: {
          rocdbgapi = rocmPrev.rocdbgapi.override {
            buildDocs = false;
          };
        }
      );
    }
    // lib.optionalAttrs (prev ? llvm) {
      llvm = skipChecks (limitLlvmTargets acceleratorLlvmTargetCmakeFlags prev.llvm);
    }
    // lib.optionalAttrs (prev ? llvm_20) {
      llvm_20 = skipChecks (limitLlvmTargets acceleratorLlvmTargetCmakeFlags prev.llvm_20);
    }
    // lib.optionalAttrs (prev ? llvm_21) {
      llvm_21 = skipChecks (limitLlvmTargets rustLlvmTargetCmakeFlags prev.llvm_21);
    }
    // lib.optionalAttrs (prev ? llvm_22) {
      llvm_22 = skipChecks (limitLlvmTargets acceleratorLlvmTargetCmakeFlags prev.llvm_22);
    }
    // lib.optionalAttrs (prev ? llvmPackages) {
      llvmPackages = skipLlvmChecks acceleratorLlvmTargetCmakeFlags prev.llvmPackages;
    }
    // lib.optionalAttrs (prev ? llvmPackages_20) {
      llvmPackages_20 = skipLlvmChecks acceleratorLlvmTargetCmakeFlags prev.llvmPackages_20;
    }
    // lib.optionalAttrs (prev ? llvmPackages_21) {
      llvmPackages_21 = skipLlvmChecks rustLlvmTargetCmakeFlags prev.llvmPackages_21;
    }
    // lib.optionalAttrs (prev ? llvmPackages_22) {
      llvmPackages_22 = skipLlvmChecks acceleratorLlvmTargetCmakeFlags prev.llvmPackages_22;
    }
    // lib.optionalAttrs (prev ? llvmPackages_latest) {
      llvmPackages_latest = skipLlvmChecks acceleratorLlvmTargetCmakeFlags prev.llvmPackages_latest;
    }
    // lib.optionalAttrs (prev ? llama-cpp-rocm) {
      llama-cpp-rocm = tune prev.llama-cpp-rocm;
    }
    // lib.optionalAttrs (prev ? llama-cpp-vulkan) {
      llama-cpp-vulkan = tune prev.llama-cpp-vulkan;
    };

  mkPackageSet =
    {
      system ? "x86_64-linux",
      nixpkgsInput ? nixpkgs,
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
            (mkCpuTuningOverlay { inherit cpu; })
          ]
          ++ extraOverlays;
      }
      // {
        inherit system;
      }
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
      dropDependencies =
        names: pkg:
        pkg.overridePythonAttrs (old: {
          pythonRemoveDeps = (old.pythonRemoveDeps or [ ]) ++ names;
          dependencies = lib.filter (dep: !(lib.elem (dep.pname or dep.name or "") names)) (
            old.dependencies or [ ]
          );
          propagatedBuildInputs = lib.filter (dep: !(lib.elem (dep.pname or dep.name or "") names)) (
            old.propagatedBuildInputs or [ ]
          );
        });
      addDependencies =
        deps: pkg:
        pkg.overridePythonAttrs (old: {
          dependencies = (old.dependencies or [ ]) ++ deps;
          propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ deps;
        });
      tunedPackage =
        if tunePackage then
          addDependencies [ pkgs.python3Packages.cloudpickle ] (
            dropDependencies [
              "datasets"
              "diskcache"
              "lark"
              "mistral-common"
              "mistral_common"
              "mistralai"
              "opencv-python-headless"
              "outlines"
              "outlines-core"
              "peft"
              "pyarrow"
              "ray"
              "timm"
              "torchaudio"
              "torchcodec"
              "torchvision"
              "xformers"
            ] package
          )
        else
          package;
    in
    if tunePackage then
      appendPythonCompileFlags " -mprefer-vector-width=512" tunedPackage
    else
      tunedPackage;

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
      withRay ? false,
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
