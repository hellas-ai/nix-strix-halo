# Strix-halo additions: ec-su-axb35 modules, llama.cpp variants, TheRock
# ROCm SDK bindings, vLLM lemonade scaffolding, the per-arch torch-rocm-7_13
# matrix, and a number of small one-off tools (ds4, fastflowlm, xrt, etc.).
{
  inputs,
  lib,
  hardwareTargets,
  defaultRocmTarget,
  rocmTargets,
  therockRocmSources,
  therockPythonWheelSources,
  therockRocmSourceSources,
  therockRocmThirdPartySources,
}:
final: prev:
let
  ecPackages = prev.callPackage ../pkgs/ec-su-axb35.nix {
    ec-su-axb35-src = inputs.ec-su-axb35;
  };

  # Add libibverbs (rdma-core-usb4) to a llama-cpp variant so ggml's
  # RPC backend auto-detects it at cmake configure time and enables
  # GGML_RPC_RDMA. We also force the flag ON explicitly so a future
  # nixpkgs bump that flips the default doesn't silently disable it.
  applyRdmaSupport =
    pkg:
    pkg.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ [ final.rdma-core-usb4 ];
      cmakeFlags = (old.cmakeFlags or [ ]) ++ [
        (prev.lib.cmakeBool "GGML_RPC_RDMA" true)
      ];
    });

  # Override `src` to upstream ggml-org master, leaving the rest of
  # the nixpkgs derivation (cmake config, backend flags, deps) intact.
  # The attribute name is set as `pname` so derivation store paths
  # disambiguate from the nixpkgs-pinned variants.
  #
  # Master moved the embedded web UI from tools/server/webui/ to
  # tools/ui/ and made it optional via LLAMA_BUILD_UI. The nixpkgs
  # derivation's npm plumbing hard-codes the old path, so we strip
  # it out and set LLAMA_BUILD_UI=OFF — llama-server still builds
  # (linking a stub llama-ui static lib) and reports UI as disabled
  # at runtime. Restore the UI separately if you actually need it.
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
        # LLAMA_BUILD_NUMBER is substituted into common/build-info.cpp
        # as `int LLAMA_BUILD_NUMBER = @LLAMA_BUILD_NUMBER@;`, so it
        # must be a numeric literal. nixpkgs's cmakeFeature already
        # emitted `-DLLAMA_BUILD_NUMBER:STRING=master-6a257d4` (our
        # non-numeric version above); the duplicate here wins by
        # cmake's last-D-flag semantics. Commit ref still lives in
        # LLAMA_BUILD_COMMIT (set by preConfigure).
        (prev.lib.cmakeFeature "LLAMA_BUILD_NUMBER" "0")
      ];
    });

  rocmHardwareTargets = hardwareTargets.rocmTargets;

  # Backend "bases": before applyRdmaSupport / applyMasterSrc.
  mkLlamaCppRocmBaseFor =
    target:
    prev.llama-cpp.override {
      rocmSupport = true;
      rpcSupport = true;
      rocmGpuTargets = target.buildTargets;
    };

  mkTargetedPackage =
    pname: pkg:
    pkg.overrideAttrs (_old: {
      inherit pname;
    });

  mkTargetPackageAttrs =
    prefix: mkPackage:
    builtins.listToAttrs (
      map (target: {
        name = "${prefix}-${target.packageSuffix}";
        value = mkPackage target;
      }) rocmHardwareTargets
    );

  llamaCppRocmTargetPackages = mkTargetPackageAttrs "llama-cpp-rocm" (
    target:
    applyRdmaSupport (
      mkTargetedPackage "llama-cpp-rocm-${target.packageSuffix}" (mkLlamaCppRocmBaseFor target)
    )
  );

  llamaCppMasterRocmTargetPackages = mkTargetPackageAttrs "llama-cpp-master-rocm" (
    target:
    applyRdmaSupport (
      applyMasterSrc "llama-cpp-master-rocm-${target.packageSuffix}" (mkLlamaCppRocmBaseFor target)
    )
  );

  llamaCppRocmBase = mkLlamaCppRocmBaseFor defaultRocmTarget;

  llamaCppRocmTherockBase =
    let
      sdkBase = final."therock-rocm-${defaultRocmTarget.packageSuffix}";
      sdk = sdkBase // {
        localGpuTargets = rocmTargets;
        gpuTargets = rocmTargets;
        # llama-cpp's default cmakeFlags read
        # ${rocmPackages.clr.hipClangPath}/clang++. We override
        # CMAKE_HIP_COMPILER below, but the attribute must exist for
        # the override to evaluate.
        hipClangPath = "${sdkBase}/llvm/bin";
      };
      therockRocmPackages = {
        clr = sdk;
        hipblas = sdk;
        rocblas = sdk;
      };
    in
    (prev.llama-cpp.override {
      rocmSupport = true;
      rpcSupport = true;
      rocmGpuTargets = rocmTargets;
      rocmPackages = therockRocmPackages;
    }).overrideAttrs
      (old: {
        pname = "llama-cpp-rocm-therock";
        buildInputs = (old.buildInputs or [ ]) ++ [ sdkBase ];
        cmakeFlags = (old.cmakeFlags or [ ]) ++ [
          "-DCMAKE_HIP_COMPILER=${sdkBase}/bin/therock-hip-clang++"
          "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${sdkBase}"
          # Bake ${sdkBase}/lib into the RUNPATH of every linked
          # output (executables and shared libraries) so libggml-hip.so
          # can find libhipblas.so.3/libamdhip64.so.7 at runtime. The
          # HIP-language link is performed by therock-hip-clang++,
          # which bypasses cc-wrapper / NIX_LDFLAGS, so setting this
          # via CMake is the only way to reach it. Also covers the
          # postInstall shell-completion step which invokes
          # llama-server before the fixup-phase patchelf would run.
          "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath,${sdkBase}/lib"
          "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,${sdkBase}/lib"
        ];
        env = (old.env or { }) // {
          HIP_PATH = "${sdkBase}";
          HIP_PLATFORM = "amd";
          # -L lets the host-side link of rpc-server resolve symbols
          # in libamdhip64.so.7 (pulled in transitively via
          # libggml-hip.so's DT_NEEDED).
          NIX_LDFLAGS = "${(old.env or { }).NIX_LDFLAGS or ""} -L${sdkBase}/lib";
        };
      });

  llamaCppVulkanBase = prev.llama-cpp.override {
    vulkanSupport = true;
    rpcSupport = true;
  };

  # CUDA variant for NVIDIA hosts (e.g. fuckup, RTX 4090 / sm89).
  # Skips applyRdmaSupport: NVIDIA hosts don't ship the
  # usb4_rdma kernel module / rdma-core-usb4 userspace, so
  # linking libibverbs is just dead weight there. cudaSupport
  # / cudaCapabilities come from the consuming pkgs instance
  # (e.g. nixos-config's pkgsForCuda or our cudaPackagesFor
  # below).
  llamaCppCudaBase = prev.llama-cpp.override {
    cudaSupport = true;
    rpcSupport = true;
  };

  # PyTorch built against the 7.13 ROCm set scoped to a single
  # arch. nixpkgs's torch derivation auto-derives gpuTargets from
  # rocmPackages.clr.localGpuTargets, so passing the per-arch
  # sub-scope is enough.
  mkTorchRocm713 =
    target:
    (final.python3Packages.torch.override {
      rocmSupport = true;
      cudaSupport = false;
      rocmPackages = final.therockRocmPackages_7_13.${target.runtimeArch};
    }).overrideAttrs
      (old: {
        # GCC 15.2 randomly ICEs (segfault in
        # gt_ggc_mx_lang_tree_node) on pytorch's heavily-templated
        # CPU kernels under parallel compilation. Capping MAX_JOBS
        # lowers the probability of hitting the GC bug; 4 is
        # empirically reliable on builders with ~250G RAM.
        preBuild = lib.replaceStrings [ "MAX_JOBS=$NIX_BUILD_CORES" ] [ "MAX_JOBS=4" ] (old.preBuild or "");

        postPatch = (old.postPatch or "") + ''
          # RCCL 7.13 reports NCCL_VERSION_CODE 22803 (2.28.3)
          # but does not ship the NCCL 2.28+ device-side
          # symmetric memory APIs (ncclGetLsaPointer,
          # ncclDevCommRequirements::railGinBarrierCount, …).
          # pytorch's symm_mem backend keys off
          # NCCL_HAS_SYMMEM_DEVICE_SUPPORT, which is set purely
          # from NCCL_VERSION_CODE without an `is real NCCL`
          # check. Compound the version checks with
          # `!defined(USE_ROCM)` so the gates stay closed for
          # RCCL.
          substituteInPlace torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp \
            --replace-fail \
              '#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)' \
              '#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0) && !defined(USE_ROCM)' \
            --replace-fail \
              '#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)' \
              '#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0) && !defined(USE_ROCM)'

          # intra_node_comm.cpp calls rsmi_init() under
          # `#ifdef USE_ROCM`, but pytorch's
          # Caffe2_PUBLIC_HIP_DEPENDENCY_LIBS doesn't include
          # rocm_smi64. Without this libtorch_hip.so ends up
          # with an undefined rsmi_init symbol that breaks
          # `import torch`.
          substituteInPlace cmake/Dependencies.cmake \
            --replace-fail \
              'hip::amdhip64 MIOpen hiprtc::hiprtc) # libroctx will be linked in with MIOpen' \
              'hip::amdhip64 MIOpen hiprtc::hiprtc rocm_smi64) # libroctx will be linked in with MIOpen'
        '';
      });

  # Only build torch for arches that have a sub-scope in
  # therockRocmPackages_7_13. nixpkgs' rocmPackages exposes one
  # sub-scope per arch listed in clr.gpuTargets, plus the gfx9/10/11/12
  # bundle scopes. gfx1036 (Granite Ridge iGPU) isn't in any of those,
  # so it's excluded.
  torchRocm713Arches = [
    "gfx1010"
    "gfx1151"
  ];
  torch-rocm-7_13-perArch = lib.listToAttrs (
    map (target: {
      name = "torch-rocm-7_13-${target.packageSuffix}";
      value = mkTorchRocm713 target;
    }) (builtins.filter (t: builtins.elem t.runtimeArch torchRocm713Arches) hardwareTargets.rocmTargets)
  );

  # Per-arch TheRock SDK + dependents, generated for every arch with
  # binary release artifacts pinned in therock-rocm-sources.json.
  mkTherockRocmSdkAttrs =
    target: source:
    let
      sdk = prev.callPackage ../pkgs/therock-rocm-sdk {
        inherit target;
        inherit (source) version url hash;
      };
      env = prev.callPackage ../pkgs/therock-rocm-env {
        therockRocmSdk = sdk;
        rdmaCore = final.rdma-core-usb4;
        packageSuffix = target;
      };
      rocshmemEnv = prev.callPackage ../pkgs/therock-rocm-rocshmem-env {
        therockRocmSdk = sdk;
        therockRocmEnv = env;
        packageSuffix = target;
      };
      ds4 = prev.callPackage ../pkgs/ds4-rocm {
        src = inputs.ds4-hip;
        rocmSdk = sdk;
        version = "experimental-${inputs.ds4-hip.shortRev or inputs.ds4-hip.rev or "unknown"}";
        offloadArch = target;
      };
      rocmXio = prev.callPackage ../pkgs/rocm-xio {
        src = inputs.rocm-xio;
        rocmSdk = sdk;
        rdma-core = final.rdma-core-usb4;
        offloadArch = target;
        version = "0.1.0-${inputs.rocm-xio.shortRev or "local"}";
      };
    in
    {
      "therock-rocm-${target}" = sdk;
      "therock-rocm-${target}-env" = env;
      "therock-rocm-${target}-rocshmem-env" = rocshmemEnv;
      "ds4-rocm-${target}" = ds4;
      "rocm-xio-${target}" = rocmXio;
    };

  therockRocmSdkPerArch = lib.foldl' lib.recursiveUpdate { } (
    lib.mapAttrsToList mkTherockRocmSdkAttrs therockRocmSources.linux
  );

  # Per-arch Lemonade vLLM bundle and its derived caches/wrappers.
  # Generated for every target listed in `lemonadeTargets`; today
  # Lemonade only publishes for gfx1151 but adding a new arch is just a
  # matter of adding it to that list once upstream ships a release tag.
  mkLemonadeStack =
    target:
    let
      s = target.packageSuffix;
      hsaOverride = target.hsaOverride or "11.5.1";
      lemonade = prev.callPackage ../pkgs/lemonade-vllm-rocm {
        target = s;
        releaseTag = "vllm0.21.0-rocm7.13.0-${s}";
      };
      env = prev.callPackage ../pkgs/vllm-env-lemonade {
        vllm-rocm-lemonade = lemonade;
        packageSuffix = s;
        hsaOverrideGfxVersion = hsaOverride;
      };
      primeCache = prev.callPackage ../pkgs/vllm-lemonade-prime-cache {
        vllm-rocm-lemonade = lemonade;
        packageSuffix = s;
      };
      qwenCache = prev.callPackage ../pkgs/vllm-lemonade-qwen36-27b-cache {
        vllm-rocm-lemonade = lemonade;
        packageSuffix = s;
        hsaOverrideGfxVersion = hsaOverride;
      };
      qwen = prev.callPackage ../pkgs/vllm-lemonade-qwen36-27b {
        vllm-env-lemonade = env;
        vllm-lemonade-qwen36-27b-cache = qwenCache;
        packageSuffix = s;
      };
      gemmaKernelCache = prev.callPackage ../pkgs/vllm-lemonade-gemma4-31b-q8-kernel-cache {
        inherit (final) gemma4-31b-it-text-config;
        vllm-rocm-lemonade = lemonade;
        vllm-env-lemonade = env;
        packageSuffix = s;
        hsaOverrideGfxVersion = hsaOverride;
      };
      gemma = prev.callPackage ../pkgs/vllm-lemonade-gemma4-31b-q8 {
        vllm-env-lemonade = env;
        vllm-lemonade-gemma4-31b-q8-kernel-cache = gemmaKernelCache;
        packageSuffix = s;
      };
      gemmaPrime = prev.callPackage ../pkgs/vllm-lemonade-gemma4-31b-q8-prime {
        inherit (final) gemma4-31b-it-text-config;
        vllm-lemonade-gemma4-31b-q8 = gemma;
        packageSuffix = s;
      };
    in
    {
      "vllm-rocm-lemonade-${s}" = lemonade;
      "vllm-env-lemonade-${s}" = env;
      "vllm-lemonade-prime-cache-${s}" = primeCache;
      "vllm-lemonade-qwen36-27b-cache-${s}" = qwenCache;
      "vllm-lemonade-qwen36-27b-${s}" = qwen;
      "vllm-lemonade-gemma4-31b-q8-kernel-cache-${s}" = gemmaKernelCache;
      "vllm-lemonade-gemma4-31b-q8-${s}" = gemma;
      "vllm-lemonade-gemma4-31b-q8-prime-${s}" = gemmaPrime;
    };

  lemonadeTargets = [ defaultRocmTarget ];

  lemonadePerArch = lib.foldl' lib.recursiveUpdate { } (map mkLemonadeStack lemonadeTargets);

  # Per-arch TheRock Python wheel bundle: pinned wheel sources from
  # therock-python-wheel-sources.json. Only one arch present today.
  mkTherockPythonStack =
    target:
    let
      s = target.packageSuffix;
    in
    {
      "therock-python-${s}" = prev.callPackage ../pkgs/therock-python-env {
        wheelSources = therockPythonWheelSources;
      };
      "therock-python-wheels-${s}" = prev.callPackage ../pkgs/therock-python-wheels {
        wheelSources = therockPythonWheelSources;
      };
      "therock-amdsmi-${s}" = prev.python312Packages.callPackage ../pkgs/therock-amdsmi {
        wheels = final."therock-python-wheels-${s}";
      };
    };

  pythonTargets = [ defaultRocmTarget ];
  therockPythonPerArch = lib.foldl' lib.recursiveUpdate { } (map mkTherockPythonStack pythonTargets);

  # Per-arch TheRock from-source: monorepo snapshot, compiler-stage
  # snapshot, prebuilt amd-llvm, full build, build-cache mirror, and
  # the configure-only override used by `nix shell` to inspect cmake.
  # Wide closure of third-party fetches lives here so the per-target
  # call site stays compact.
  therockFromSourceThirdPartyFetches = {
    esmiIbLibrarySource = prev.fetchgit {
      inherit (therockRocmThirdPartySources.esmiIbLibrary) url rev hash;
    };
    ireeLibbacktraceSource = prev.fetchzip {
      inherit
        (therockRocmThirdPartySources.archives."libbacktrace-b9e40069c0b47a722286b94eb5231f7f05c08713.zip")
        url
        ;
      hash = "sha512-vWfrdxHfNcfoh/SWh6dOTLRtqXcW0godukXv5ad6DqzfbwC7TOIbyw0Oke40oLRqFCLfg1M2f7bqnIrLnISmKw==";
      stripRoot = true;
    };
    ireeAmdDeviceLibsArchive = prev.fetchurl {
      url = "https://github.com/shark-infra/amdgpu-device-libs/releases/download/v20231101/amdgpu-device-libs-llvm-6086c272a3a59eb0b6b79dcbe00486bf4461856a.tgz";
      hash = "sha256-M2NiQWxo/di7gDKPZcp+uqDBGeoZyV323zDIMqTfObk=";
    };
    rocprofilerOtf2Archive = prev.fetchurl {
      url = "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/otf2-3.0.3.tar.gz";
      hash = "sha256-GKOQX3kXNAOH4+3I5XZvMasa9B9OzFZl2mx2nKIcTug=";
    };
    rocprofilerSysBinutilsArchive = prev.fetchurl {
      url = "https://ftpmirror.gnu.org/gnu/binutils/binutils-2.46.0.tar.gz";
      hash = "sha256-hgj+RKt95kX2rQqJgxO3UziEJJDWCa24XJ+ygnw3avI=";
    };
    spirvHeadersSource = prev.fetchzip {
      inherit (therockRocmThirdPartySources.spirvHeaders) url hash;
      stripRoot = true;
    };
    tracySource = prev.fetchFromGitHub {
      owner = "wolfpld";
      repo = "tracy";
      rev = "5479a42ef9346b64e6d1b860ae58aa8abdb0c7f6";
      hash = "sha256-4J8b+72k+xpeT6KsrkioF1xfWEBsGg2eLRg9iONxP/I=";
    };
  };

  mkTherockFromSourceStack =
    target:
    let
      s = target.packageSuffix;
      sourceKey = suffix: "therock-7.13-${s}-full${suffix}";
      sourceFull = therockRocmSourceSources.${sourceKey ""};
      sourceCompiler = therockRocmSourceSources.${sourceKey "-compiler-stage"};
      cmakeConfig = {
        target = s;
        amdgpuTargets = target.buildTargets;
        distBundleName = s;
      };
      projectTargetUnexcludes = lib.optionalAttrs (builtins.elem s target.buildTargets) {
        rocprofiler-compute = [ s ];
      };

      mkSource =
        source:
        prev.callPackage ../pkgs/therock-rocm-source {
          name = "therock-rocm-source-${s}-full";
          inherit (source)
            url
            ref
            rev
            hash
            fetchArgs
            ;
          deepNestedSubmodules = source.deepNestedSubmodules or [ ];
        };
      sourceTree = mkSource sourceFull;
      compilerTree = mkSource sourceCompiler;

      amdLlvm = prev.callPackage ../pkgs/therock-rocm-from-source ({
        stdenv = prev.llvmPackages_21.stdenv;
        inherit (cmakeConfig) target amdgpuTargets distBundleName;
        inherit (sourceCompiler) version;
        profile = "compiler";
        therockSource = compilerTree;
        thirdPartySources = { };
        buildTargets = [ "artifact-amd-llvm" ];
        installMode = "prebuilt-stages";
        spirvHeadersSource = therockFromSourceThirdPartyFetches.spirvHeadersSource;
      });

      fromSource = prev.callPackage ../pkgs/therock-rocm-from-source (
        {
          stdenv = prev.llvmPackages_21.stdenv;
          inherit (cmakeConfig) target amdgpuTargets distBundleName;
          inherit (sourceFull) version;
          profile = "full";
          therockSource = sourceTree;
          prebuiltStageTree = amdLlvm;
          thirdPartySources = therockRocmThirdPartySources.archives;
          inherit projectTargetUnexcludes;
        }
        // therockFromSourceThirdPartyFetches
      );
    in
    {
      "therock-rocm-source-${s}" = sourceTree;
      "therock-rocm-source-${s}-compiler-stage" = compilerTree;
      "therock-amd-llvm-${s}" = amdLlvm;
      "therock-rocm-from-source-${s}" = fromSource;
      "therock-rocm-from-source-${s}-configure" = fromSource.override {
        nameSuffix = "configure";
        prebuiltStageTree = null;
        installMode = "configure-only";
      };
      "therock-rocm-third-party-mirror-${s}" = fromSource.thirdPartyMirror;
      "therock-rocm-7_13-${s}-core" = final.therockRocmPackages_7_13.${s}.rocm-core;
      "therock-rocm-7_13-${s}-cmake" = final.therockRocmPackages_7_13.${s}.rocm-cmake;
    };

  fromSourceTargets = [ defaultRocmTarget ];
  therockFromSourcePerArch = lib.foldl' lib.recursiveUpdate { } (
    map mkTherockFromSourceStack fromSourceTargets
  );
in
{
  ec-su-axb35 = ecPackages.kernelModule;
  ec-su-axb35-monitor = ecPackages.monitor;

  # Backend × source-pin matrix. RDMA support is universal for
  # strix-halo variants (every ROCm/Vulkan variant links
  # libibverbs via rdma-core-usb4 and enables GGML_RPC_RDMA).
  # CUDA variants stay RDMA-free — see llamaCppCudaBase above.
  # The `master` axis swaps in upstream ggml-org/master via
  # the llama-cpp-master flake input.
  llama-cpp-rocm = applyRdmaSupport llamaCppRocmBase;
  llama-cpp-rocm-therock = applyRdmaSupport llamaCppRocmTherockBase;
  llama-cpp-vulkan = applyRdmaSupport llamaCppVulkanBase;
  llama-cpp-cuda = llamaCppCudaBase;
  llama-cpp-master-rocm = applyRdmaSupport (applyMasterSrc "llama-cpp-master-rocm" llamaCppRocmBase);
  llama-cpp-master-rocm-therock = applyRdmaSupport (
    applyMasterSrc "llama-cpp-master-rocm-therock" llamaCppRocmTherockBase
  );
  llama-cpp-master-vulkan = applyRdmaSupport (
    applyMasterSrc "llama-cpp-master-vulkan" llamaCppVulkanBase
  );
  llama-cpp-master-cuda = applyMasterSrc "llama-cpp-master-cuda" llamaCppCudaBase;

  # The therock-rocm-<arch> SDK and its direct dependents (env wrappers,
  # rocm-xio, ds4-rocm). Generated for every arch listed in
  # therock-rocm-sources.json; today that's just gfx1151, but the
  # generator picks up new entries automatically.

  # FastFlowLM NPU stack. xrt+xdna are coupled (matching tags pinned
  # in the flake inputs); FastFlowLM tracks upstream main. The
  # `xrt-amdxdna` alias mirrors the combined symlinkJoin name in
  # PR #513841 for downstream consumers.
  tokenizers-cpp = prev.callPackage ../pkgs/tokenizers-cpp { };

  xrt = prev.callPackage ../pkgs/xrt {
    src = inputs.xrt-src;
    version = "2.21.75";
    xdnaSrc = inputs.xdna-driver-src;
  };

  xrt-amdxdna = final.xrt.xdna;

  fastflowlm = prev.callPackage ../pkgs/fastflowlm {
    inherit (final) tokenizers-cpp xrt;
    src = inputs.fastflowlm;
  };

  gemma4-31b-it-text-config = prev.runCommand "gemma4-31b-it-text-config" { } ''
    mkdir -p "$out"
    cp ${../pkgs/lemonade-vllm-rocm/gemma4-31b-it-text-config.json} "$out/config.json"
  '';

  # ROCm 7.13 package set. Source tree is arch-agnostic (same git
  # rev for every target); per-arch narrowing lives inside the scope
  # via `therockRocmPackages_7_13.<arch>` (nixpkgs convention).
  therockRocmPackages_7_13 = lib.recurseIntoAttrs (
    prev.callPackage ../pkgs/therock-rocm-7_13/rocm-modules {
      therockSource = final."therock-rocm-source-${defaultRocmTarget.packageSuffix}";
      therockVersion = "7.13.0";
    }
  );

}
// llamaCppRocmTargetPackages
// llamaCppMasterRocmTargetPackages
// torch-rocm-7_13-perArch
// therockRocmSdkPerArch
// therockPythonPerArch
// therockFromSourcePerArch
// lemonadePerArch
