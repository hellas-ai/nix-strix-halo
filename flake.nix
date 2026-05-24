{
  description = "Strix Halo NixOS configuration and tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # vLLM/ROCm packaging lives in this flake. The thunderbolt-ibverbs
    # input below is kept to the lower RDMA layer.

    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    ec-su-axb35 = {
      url = "github:cmetz/ec-su_axb35-linux";
      flake = false;
    };

    # thunderbolt-ibverbs is the lower-layer RDMA stack: kernel patches,
    # module packaging, and rdma-core-usb4.
    thunderbolt-ibverbs = {
      url = "git+file:///mnt/Home/src/thunderbolt-ibverbs";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # llama.cpp upstream master — used to build the `*-master-*` variants
    # below alongside the nixpkgs-pinned source variants.
    llama-cpp-master = {
      url = "github:ggml-org/llama.cpp";
      flake = false;
    };

    # FastFlowLM (NPU LLM runtime). Tracks upstream main; bump with
    # `nix flake update fastflowlm`. NPU kernel binaries shipped under
    # src/lib and src/xclbins are proprietary blobs (see LICENSE_BINARY.txt)
    # that we ship as-is — the host-side runtime above them is MIT.
    fastflowlm = {
      url = "github:FastFlowLM/FastFlowLM";
      flake = false;
    };

    ds4 = {
      url = "github:antirez/ds4";
      flake = false;
    };

    ds4-hip = {
      url = "github:ejpir/ds4-hip/rocm-upstream-shape-cyberneurova";
      flake = false;
    };

    rocm-xio = {
      url = "git+file:///mnt/Home/src/rocm-xio";
      flake = false;
    };

    # XRT and the xdna-driver plugin ship in lockstep through AMD's
    # lemonade PPA; treat their tags as a matched pair. Bump both at
    # once when moving versions. The `git+https://` URL with
    # submodules=1 is required because both repos pull in `aiebu`,
    # `gsl`, `nlohmann-json`, etc. as git submodules; `github:` URLs
    # don't recurse.
    xrt-src = {
      url = "git+https://github.com/Xilinx/XRT?ref=refs/tags/2.21.75&submodules=1";
      flake = false;
    };

    xdna-driver-src = {
      url = "git+https://github.com/amd/xdna-driver?ref=refs/tags/2.21.75&submodules=1";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }@inputs:
    let
      inherit (nixpkgs) lib;

      systems = [ "x86_64-linux" ];
      darwinSystems = [ "aarch64-darwin" ];
      forAllSystems = lib.genAttrs systems;
      forAllDarwinSystems = lib.genAttrs darwinSystems;

      hardwareTargets = import ./lib/hardware.nix { inherit lib; };
      inherit (hardwareTargets) defaultRocmTarget;
      rocmTargets = defaultRocmTarget.buildTargets;
      therockRocmSources = builtins.fromJSON (builtins.readFile ./therock-rocm-sources.json);
      therockPythonWheelSources = builtins.fromJSON (
        builtins.readFile ./therock-python-wheel-sources.json
      );
      therockRocmSourceSources = builtins.fromJSON (builtins.readFile ./therock-rocm-source-sources.json);
      therockRocmThirdPartySources = builtins.fromJSON (
        builtins.readFile ./therock-rocm-third-party-sources.json
      );

      # ---------------------------------------------------------------
      # Overlays
      # ---------------------------------------------------------------

      strixAdditionsOverlay = import ./overlays/strix-additions.nix {
        inherit
          inputs
          lib
          hardwareTargets
          defaultRocmTarget
          rocmTargets
          therockRocmSources
          therockPythonWheelSources
          therockRocmSourceSources
          therockRocmThirdPartySources
          ;
      };

      vllmRdmaExtrasOverlay = import ./overlays/vllm-rdma-extras.nix;
      therockPythonOverlay = import ./overlays/therock-python.nix {
        inherit lib;
        target = defaultRocmTarget;
      };

      therockVllmOverlay = import ./overlays/therock-vllm.nix {
        inherit lib rocmTargets;
        target = defaultRocmTarget;
      };

      mkRocmNarrowOverlay = import ./overlays/rocm-narrow.nix;

      rocmNarrowOverlays = lib.mapAttrs' (_: target: {
        name = "rocm-narrow-${target.packageSuffix}";
        value = mkRocmNarrowOverlay target;
      }) hardwareTargets.rocm;

      # ---------------------------------------------------------------
      # Flake outputs
      # ---------------------------------------------------------------

      overlays = {
        # Composed default for strix-halo NixOS machines. Order matters:
        # rdma-core-usb4 first, RDMA Python extras next, then Python/ROCm
        # runtime overlays and strix additions. Doesn't include rocm-narrow;
        # that's opt-in via overlays.rocm-narrow because it invalidates
        # downstream caches.
        default = nixpkgs.lib.composeManyExtensions [
          inputs.thunderbolt-ibverbs.overlays.rdma-core-usb4
          vllmRdmaExtrasOverlay
          therockPythonOverlay
          strixAdditionsOverlay
          therockVllmOverlay
        ];

        # Just the RDMA-enabled vllm composition without strix additions.
        # Useful for non-strix consumers that want vllm+rixl+lmcache but
        # not ec-su/llama-cpp-rocm.
        vllm-rdma = nixpkgs.lib.composeManyExtensions [
          inputs.thunderbolt-ibverbs.overlays.rdma-core-usb4
          vllmRdmaExtrasOverlay
        ];

        # vLLM with Python dependencies resolved against TheRock's pinned
        # cp312 Torch/Triton/ROCm wheels and the TheRock gfx1151 SDK.
        vllm-therock = nixpkgs.lib.composeManyExtensions [
          inputs.thunderbolt-ibverbs.overlays.rdma-core-usb4
          vllmRdmaExtrasOverlay
          therockPythonOverlay
          strixAdditionsOverlay
          therockVllmOverlay
        ];

        # Narrows rocmPackages.clr to `rocmTargets` (currently gfx1151) for
        # faster builds on strix-halo. Applying this overlay invalidates
        # downstream caches and removes support for other GPUs from the system
        # OpenCL ICD — only enable on hosts that exclusively run gfx1151.
        #
        # This is the *shallow* narrowing: only the ICD targets change. The
        # individual rocm leaves (rccl, hipblaslt, miopen, …) still build for
        # every nixpkgs-default arch. Pair with `rocm-narrow-deep` below to
        # also narrow those leaves.
        rocm-narrow = mkRocmNarrowOverlay defaultRocmTarget;

        # Deep ROCm narrowing for strix-halo: in addition to clr, also pins
        # the gpuTargets of every leaf rocmPackage (rccl, hipblaslt, hipfft,
        # hiprand, hipsparse, miopen, rocblas, rocfft, rocrand, rocsolver,
        # rocsparse, aotriton) to gfx1151. composable_kernel_base's broken
        # check requires at least one gfx9 mfma target, so it's pinned to
        # gfx90a+gfx11-generic — CK still only emits gfx1151 kernels, but
        # the meta.broken guard passes.
        #
        # Use this when you want every ROCm build in the closure to target
        # only gfx1151 (i.e. on strix-halo hosts where you don't need
        # multi-GPU portability). Invalidates downstream caches but cuts
        # build time dramatically: composable_kernel alone has ~20 per-arch
        # parts that otherwise would all rebuild.
        #
        # Same logic is also inline in linux-libibverbs-usb4/flake.nix as
        # `gfx1151Overlay`; consolidating that copy into this one is a
        # separate cleanup. For now both should produce equivalent results.
        rocm-narrow-deep = _final: prev: {
          rocmPackages = prev.rocmPackages.overrideScope (
            _: rocmPrev:
            let
              narrow = drv: drv.override { gpuTargets = rocmTargets; };
            in
            {
              clr = rocmPrev.clr.override { localGpuTargets = rocmTargets; };

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
                gpuTargets = [
                  "gfx90a"
                  "gfx11-generic"
                ];
              };

              aotriton = rocmPrev.aotriton.override {
                gpuTargets = rocmTargets;
              };
            }
          );
        };
      }
      // rocmNarrowOverlays;

      defaultPackagesFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ overlays.default ];
        };

      # CUDA-enabled pkgs for NVIDIA consumers of llama-cpp-*-cuda.
      # Capabilities pinned to 8.9 (RTX 4090 / Ada) which is what
      # fuckup uses; bump or expose as a param if other NVIDIA hosts
      # need different SMs.
      cudaPackagesFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ strixAdditionsOverlay ];
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaCapabilities = [ "8.9" ];
          };
        };

      darwinPackagesFor =
        system:
        import nixpkgs {
          inherit system;
        };

      ds4Version = "unstable-${inputs.ds4.shortRev or inputs.ds4.rev or "unknown"}";

      ds4For =
        pkgs:
        pkgs.callPackage ./pkgs/ds4 {
          src = inputs.ds4;
          version = ds4Version;
        };

      perSystem =
        f:
        forAllSystems (
          system:
          let
            pkgs = defaultPackagesFor system;
          in
          f {
            inherit
              pkgs
              system
              ;
          }
        );

      mkBenchmarkPackages =
        pkgs:
        let
          mkPackageSetForTarget =
            target:
            let
              isDefault = target.packageSuffix == defaultRocmTarget.packageSuffix;
            in
            {
              llama-cpp-rocm =
                if isDefault then pkgs.llama-cpp-rocm else pkgs."llama-cpp-rocm-${target.packageSuffix}";
              llama-cpp-master-rocm =
                if isDefault then
                  pkgs.llama-cpp-master-rocm
                else
                  pkgs."llama-cpp-master-rocm-${target.packageSuffix}";
              inherit (pkgs) llama-cpp-vulkan llama-cpp-master-vulkan;
            }
            // lib.optionalAttrs isDefault {
              inherit (pkgs) llama-cpp-rocm-therock llama-cpp-master-rocm-therock;
            };

          mkForTarget =
            target:
            let
              benchmarks = import ./bench/default.nix {
                inherit pkgs;
                packages = mkPackageSetForTarget target;
                gpuTarget = target.systemFeature;
                inherit (target) hsaOverride;
              };
              targetPrefix = lib.optionalString (
                target.packageSuffix != defaultRocmTarget.packageSuffix
              ) "${target.packageSuffix}-";
            in
            pkgs.lib.concatMapAttrs (
              model: benchs:
              pkgs.lib.mapAttrs' (name: drv: {
                name = "bench-${targetPrefix}${model}-${name}";
                value = drv;
              }) benchs
            ) benchmarks;
        in
        lib.foldl' (acc: target: acc // mkForTarget target) { } hardwareTargets.rocmTargets;

      mkFastflowlmBenchmarks =
        pkgs:
        (import ./bench/fastflowlm.nix {
          inherit pkgs;
          inherit (pkgs) fastflowlm;
        }).benchmarks;

      mkSourceCheck =
        pkgs: name: nativeBuildInputs: command:
        pkgs.runCommandLocal "ci-${name}"
          {
            inherit nativeBuildInputs;
            src = lib.cleanSource ./.;
          }
          ''
            export HOME="$TMPDIR"
            export XDG_CACHE_HOME="$TMPDIR/cache"
            cp -r --no-preserve=mode "$src" source
            cd source
            ${command}
            touch "$out"
          '';

      rocmNarrowNixosModules = lib.mapAttrs' (_: target: {
        name = "rocm-narrow-${target.packageSuffix}";
        value = _: {
          nixpkgs.overlays = [
            self.overlays."rocm-narrow-${target.packageSuffix}"
          ];
        };
      }) hardwareTargets.rocm;
    in
    {
      inherit overlays;

      lib = {
        hardware = hardwareTargets;
        inherit mkRocmNarrowOverlay;
      };

      nixosModules = {
        default = _: {
          nixpkgs.overlays = [
            self.overlays.default
          ];
        };
        rocm-narrow = _: {
          nixpkgs.overlays = [
            self.overlays.rocm-narrow
          ];
        };
        rocm-narrow-deep = _: {
          nixpkgs.overlays = [
            self.overlays.rocm-narrow-deep
          ];
        };
        rpc-server = import ./modules/rpc-server.nix;
        benchmark-runner = import ./modules/benchmark-runner.nix;
        fastflowlm-server = import ./modules/fastflowlm-server.nix;
        ec-su-axb35 = import ./modules/ec-su-axb35.nix;
        ryzenadj = import ./modules/ryzenadj.nix;
        disko-raid0 = import ./modules/disko-raid0.nix;
        tuning = import ./modules/tuning.nix;
      }
      // rocmNarrowNixosModules;

      nixosConfigurations = {
        fevm-faex9 = nixpkgs.lib.nixosSystem {
          system = "x86_64-linux";
          specialArgs = {
            inherit inputs self;
          };
          modules = [
            inputs.disko.nixosModules.disko
            ./examples/fevm-faex9/configuration.nix
          ];
        };
      };

      legacyPackages = perSystem (
        { pkgs, ... }:
        {
          defaultPackages = pkgs;
        }
      );

      packages =
        perSystem (
          { pkgs, system, ... }:
          let
            # Per-target package attr names to expose. `lib.optional`
            # filters out arches whose package isn't actually defined
            # (most lemonade/therock bundles only exist for gfx1151).
            perTargetPackageNames =
              target:
              let
                s = target.packageSuffix;
                wanted = [
                  "rocm-xio-${s}"
                  "therock-amd-llvm-${s}"
                  "therock-rocm-${s}"
                  "therock-rocm-${s}-env"
                  "therock-rocm-${s}-rocshmem-env"
                  "therock-rocm-7_13-${s}-core"
                  "therock-rocm-7_13-${s}-cmake"
                  "therock-rocm-from-source-${s}"
                  "therock-rocm-from-source-${s}-configure"
                  "therock-rocm-source-${s}"
                  "therock-rocm-third-party-mirror-${s}"
                  "therock-python-${s}"
                  "therock-python-overlay-smoke-${s}"
                  "therock-python-wheels-${s}"
                  "ds4-rocm-${s}"
                  "torch-rocm-7_13-${s}"
                  "llama-cpp-rocm-${s}"
                  "llama-cpp-master-rocm-${s}"
                  "vllm-rocm-lemonade-${s}"
                  "vllm-env-lemonade-${s}"
                  "vllm-lemonade-prime-cache-${s}"
                  "vllm-lemonade-qwen36-27b-cache-${s}"
                  "vllm-lemonade-qwen36-27b-${s}"
                  "vllm-lemonade-gemma4-31b-q8-kernel-cache-${s}"
                  "vllm-lemonade-gemma4-31b-q8-${s}"
                  "vllm-lemonade-gemma4-31b-q8-prime-${s}"
                  "vllm-rocm-therock-${s}"
                  "vllm-env-therock-${s}"
                  "vllm-env-therock-aiter-${s}"
                  "vllm-aiter-jit-therock-${s}"
                  "vllm-aiter-therock-${s}"
                ];
              in
              builtins.filter (n: pkgs ? ${n}) wanted;
            allPerTargetPackageNames = lib.concatMap perTargetPackageNames hardwareTargets.rocmTargets;
            # Separate pkgs instance with cudaSupport=true so the CUDA
            # variants below build against the cuda toolchain instead of
            # the default ROCm-flavoured pkgs.
            cudaPkgs = cudaPackagesFor system;
          in
          {
            default = pkgs.llama-cpp-rocm;
            inherit (cudaPkgs) llama-cpp-cuda llama-cpp-master-cuda;
            inherit (pkgs)
              ec-su-axb35-monitor
              fastflowlm
              tokenizers-cpp
              xrt
              xrt-amdxdna
              llama-cpp-rocm
              llama-cpp-rocm-therock
              llama-cpp-vulkan
              llama-cpp-master-rocm
              llama-cpp-master-rocm-therock
              llama-cpp-master-vulkan
              rdma-core-usb4
              gemma4-31b-it-text-config
              ;
          }
          // lib.genAttrs allPerTargetPackageNames (n: pkgs.${n})
          // mkBenchmarkPackages pkgs
          // mkFastflowlmBenchmarks pkgs
        )
        // forAllDarwinSystems (
          system:
          let
            pkgs = darwinPackagesFor system;
            ds4 = ds4For pkgs;
          in
          {
            default = ds4;
            inherit ds4;
          }
        );

      apps =
        perSystem (
          { pkgs, ... }:
          let
            # `pkg`'s default binary is named after the package; specify
            # `binary` to point at a different one.
            mkApp =
              {
                pkg,
                binary ? pkg.pname or pkg.name,
                description,
              }:
              {
                type = "app";
                program = "${pkg}/bin/${binary}";
                meta = { inherit description; };
              };

            mkHsaOverrideExport =
              target:
              lib.optionalString (
                target.hsaOverride != null
              ) ''export HSA_OVERRIDE_GFX_VERSION="${target.hsaOverride}"'';

            mkLlamaApp =
              {
                name,
                package,
                binary,
                target,
                description,
              }:
              {
                type = "app";
                program = toString (
                  pkgs.writeShellScript name ''
                    ${mkHsaOverrideExport target}
                    ${package}/bin/${binary} "$@"
                  ''
                );
                meta.description = description;
              };

            llamaAppAxes = {
              cli = {
                appPrefix = "llama-cli";
                packagePrefix = "llama-cpp-rocm";
                binary = "llama-cli";
                descriptionPrefix = "Run llama-cli with ROCm for";
              };
              server = {
                appPrefix = "llama-server";
                packagePrefix = "llama-cpp-rocm";
                binary = "llama-server";
                descriptionPrefix = "Run llama-server with ROCm for";
              };
              cliMaster = {
                appPrefix = "llama-cli-master";
                packagePrefix = "llama-cpp-master-rocm";
                binary = "llama-cli";
                descriptionPrefix = "Run llama-cli from ggml-org master with ROCm for";
              };
              serverMaster = {
                appPrefix = "llama-server-master";
                packagePrefix = "llama-cpp-master-rocm";
                binary = "llama-server";
                descriptionPrefix = "Run llama-server from ggml-org master with ROCm for";
              };
            };

            mkLlamaAppSet =
              {
                target,
                appSuffix ? "",
                packageSuffix ? "",
                packageNameFor ? null,
                descriptionFor ? null,
              }:
              let
                packageName =
                  spec:
                  if packageNameFor == null then "${spec.packagePrefix}${packageSuffix}" else packageNameFor spec;
                description =
                  spec:
                  if descriptionFor == null then
                    "${spec.descriptionPrefix} ${target.marketingName}"
                  else
                    descriptionFor spec;
              in
              lib.mapAttrs' (
                _: spec:
                let
                  name = "${spec.appPrefix}${appSuffix}";
                in
                {
                  inherit name;
                  value = mkLlamaApp {
                    inherit name target;
                    package = pkgs.${packageName spec};
                    inherit (spec) binary;
                    description = description spec;
                  };
                }
              ) llamaAppAxes;

            targetLlamaApps = lib.foldl' (
              acc: target:
              acc
              // mkLlamaAppSet {
                inherit target;
                appSuffix = "-${target.packageSuffix}";
                packageSuffix = "-${target.packageSuffix}";
              }
            ) { } hardwareTargets.rocmTargets;

            defaultLlamaApps = mkLlamaAppSet { target = defaultRocmTarget; };
            defaultTherockLlamaApps = mkLlamaAppSet {
              target = defaultRocmTarget;
              appSuffix = "-therock";
              packageNameFor = spec: "${spec.packagePrefix}-therock";
              descriptionFor =
                spec: "${spec.descriptionPrefix} ${defaultRocmTarget.marketingName} with TheRock ROCm";
            };

            # Per-target apps for the strix-halo SDK / lemonade / therock
            # bundles. Adding a new arch auto-generates these apps as
            # soon as the matching packages exist in `pkgs`; entries
            # whose backing package isn't present are silently dropped.
            mkPerTargetApps =
              target:
              let
                s = target.packageSuffix;
                m = target.marketingName;
                # appName, package attr name, binary, description.
                specs = [
                  [
                    "therock-rocm-${s}-env"
                    "therock-rocm-${s}-env"
                    null
                    "Run a command in the opt-in TheRock ROCm 7.13 ${s} environment"
                  ]
                  [
                    "therock-rocm-${s}-rocshmem-env"
                    "therock-rocm-${s}-rocshmem-env"
                    null
                    "Run a command in the TheRock ROCm 7.13 ${s} rocSHMEM test environment"
                  ]
                  [
                    "ds4-rocm-${s}"
                    "ds4-rocm-${s}"
                    "ds4"
                    "Run experimental ds4 ROCm/HIP on ${m}"
                  ]
                  [
                    "ds4-bench-rocm-${s}"
                    "ds4-rocm-${s}"
                    "ds4-bench"
                    "Run experimental ds4 ROCm/HIP benchmark on ${m}"
                  ]
                  [
                    "ds4-bench-fast-full-${s}"
                    "ds4-rocm-${s}"
                    "ds4-bench-fast-full"
                    "Run experimental ds4 ROCm/HIP fast-full benchmark preset on ${m}"
                  ]
                  [
                    "therock-python-${s}"
                    "therock-python-${s}"
                    "therock-python"
                    "Run Python in the pinned TheRock ROCm/PyTorch wheel environment"
                  ]
                  [
                    "therock-python-${s}-env"
                    "therock-python-${s}"
                    "therock-python-env"
                    "Run a command in the pinned TheRock ROCm/PyTorch wheel environment"
                  ]
                  [
                    "vllm-therock-${s}"
                    "vllm-env-therock-${s}"
                    "vllm"
                    "Run vLLM built against the TheRock ROCm/PyTorch/Triton overlay"
                  ]
                  [
                    "vllm-therock-aiter-${s}"
                    "vllm-aiter-therock-${s}"
                    "vllm-aiter-therock-${s}"
                    "Run vLLM with TheRock ROCm and a preseeded AITER JIT cache"
                  ]
                  [
                    "vllm-lemonade-${s}"
                    "vllm-rocm-lemonade-${s}"
                    "vllm"
                    "Run Lemonade's vLLM ROCm binary bundle for ${m}"
                  ]
                  [
                    "vllm-env-lemonade-${s}"
                    "vllm-env-lemonade-${s}"
                    "vllm"
                    "Run Lemonade's vLLM ROCm bundle with Ray available for distributed serving"
                  ]
                  [
                    "vllm-lemonade-prime-cache-${s}"
                    "vllm-lemonade-prime-cache-${s}"
                    null
                    "Prime Lemonade vLLM ROCm Triton/autotune caches on a ${s} host"
                  ]
                  [
                    "vllm-lemonade-qwen36-27b-${s}"
                    "vllm-lemonade-qwen36-27b-${s}"
                    null
                    "Run Lemonade vLLM with the prebuilt Qwen3.6-27B Triton cache"
                  ]
                  [
                    "vllm-lemonade-gemma4-31b-q8-${s}"
                    "vllm-lemonade-gemma4-31b-q8-${s}"
                    null
                    "Run Lemonade vLLM with the prebuilt Gemma4-31B Q8 kernel cache"
                  ]
                  [
                    "vllm-lemonade-gemma4-31b-q8-prime-${s}"
                    "vllm-lemonade-gemma4-31b-q8-prime-${s}"
                    null
                    "Prime Gemma4-31B Q8 vLLM caches before measured benchmark runs"
                  ]
                ];
                toAttr =
                  spec:
                  let
                    appName = builtins.elemAt spec 0;
                    pkgName = builtins.elemAt spec 1;
                    binary = builtins.elemAt spec 2;
                    description = builtins.elemAt spec 3;
                  in
                  lib.optionalAttrs (pkgs ? ${pkgName}) {
                    ${appName} = mkApp (
                      {
                        pkg = pkgs.${pkgName};
                        inherit description;
                      }
                      // lib.optionalAttrs (binary != null) { inherit binary; }
                    );
                  };
              in
              lib.foldl' (a: b: a // b) { } (map toAttr specs);

            perTargetApps = lib.foldl' (
              acc: target: acc // mkPerTargetApps target
            ) { } hardwareTargets.rocmTargets;
          in
          {
            flm = mkApp {
              pkg = pkgs.fastflowlm;
              binary = "flm";
              description = "FastFlowLM CLI on the AMD Ryzen AI NPU (Strix Halo XDNA2)";
            };
          }
          // perTargetApps
          // defaultLlamaApps
          // defaultTherockLlamaApps
          // targetLlamaApps
        )
        // forAllDarwinSystems (
          system:
          let
            inherit (self.packages.${system}) ds4;
            mkDs4App = binary: description: {
              type = "app";
              program = "${ds4}/bin/${binary}";
              meta.description = description;
            };
          in
          {
            default = mkDs4App "ds4" "Run DwarfStar 4 with Metal";
            ds4 = mkDs4App "ds4" "Run DwarfStar 4 with Metal";
            ds4-server = mkDs4App "ds4-server" "Run the DwarfStar 4 HTTP server with Metal";
            ds4-bench = mkDs4App "ds4-bench" "Run the DwarfStar 4 benchmark with Metal";
            ds4-download-model = mkDs4App "ds4-download-model" "Download DS4 DeepSeek V4 Flash GGUF weights";
          }
        );

      devShells = perSystem (
        { pkgs, ... }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              deadnix
              nix-fast-build
              nixfmt-tree
              statix
              (python3.withPackages (
                ps: with ps; [
                  numpy
                  pandas
                  plotly
                ]
              ))
            ];
          };
        }
      );

      formatter = perSystem ({ pkgs, ... }: pkgs.nixfmt-tree);

      checks = perSystem (
        { pkgs, ... }:
        {
          format = mkSourceCheck pkgs "format" [ pkgs.nixfmt-tree ] ''
            treefmt --fail-on-change
          '';

          deadnix = mkSourceCheck pkgs "deadnix" [ pkgs.deadnix ] ''
            deadnix --fail .
          '';

          statix = mkSourceCheck pkgs "statix" [ pkgs.statix ] ''
            statix check .
          '';
        }
      );
    };
}
