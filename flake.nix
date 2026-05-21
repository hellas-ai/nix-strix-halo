{
  description = "Strix Halo NixOS configuration and tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # vllm + flash-attn + cutlass + triton sources used to live here as
    # direct inputs. Now provided transitively via inputs.usb4-rdma,
    # which owns the canonical vllm wrapper (see overlays.vllm-rdma).

    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    ec-su-axb35 = {
      url = "github:cmetz/ec-su_axb35-linux";
      flake = false;
    };

    # libibverbs-usb4 is the lower-layer RDMA stack (kernel patches,
    # rdma-core-usb4, rixl, lmcache) plus the canonical vllm wrapper
    # exposed as `overlays.vllm` / `overlays.vllm-rdma`. We compose its
    # vllm-rdma overlay into our own overlays.default below so strix
    # machines pick up an RDMA-enabled vllm by default.
    usb4-rdma = {
      url = "git+ssh://trex/home/grw/src/linux-libibverbs-usb4";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # FastFlowLM (NPU LLM runtime). Tracks upstream main; bump with
    # `nix flake update fastflowlm`. NPU kernel binaries shipped under
    # src/lib and src/xclbins are proprietary blobs (see LICENSE_BINARY.txt)
    # that we ship as-is — the host-side runtime above them is MIT.
    fastflowlm = {
      url = "github:FastFlowLM/FastFlowLM";
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
      forAllSystems = lib.genAttrs systems;

      rocmTargets = [
        "gfx1151"
      ];

      strixAdditionsOverlay =
        final: prev:
        let
          ecPackages = prev.callPackage ./pkgs/ec-su-axb35.nix {
            ec-su-axb35-src = inputs.ec-su-axb35;
          };
        in
        {
          ec-su-axb35 = ecPackages.kernelModule;
          ec-su-axb35-monitor = ecPackages.monitor;

          llama-cpp-rocm = prev.llama-cpp.override {
            rocmSupport = true;
            rpcSupport = true;
            rocmGpuTargets = rocmTargets;
          };

          # FastFlowLM NPU stack. xrt+xdna are coupled (matching tags pinned
          # in the flake inputs); FastFlowLM tracks upstream main. The
          # `xrt-amdxdna` alias mirrors the combined symlinkJoin name in
          # nixpkgs PR #513841 for downstream consumers.
          tokenizers-cpp = prev.callPackage ./pkgs/tokenizers-cpp { };

          xrt = prev.callPackage ./pkgs/xrt {
            src = inputs.xrt-src;
            version = "2.21.75";
            xdnaSrc = inputs.xdna-driver-src;
          };

          xrt-amdxdna = final.xrt.xdna;

          fastflowlm = prev.callPackage ./pkgs/fastflowlm {
            inherit (final) tokenizers-cpp xrt;
            src = inputs.fastflowlm;
          };
        };

      # Python-side extras for RDMA-enabled vllm: rixl (ROCm NIXL port),
      # lmcache (KV-cache layer on top of rixl), cupy-rocm-7-0 (CuPy ROCm
      # build). These layer on top of libibverbs.overlays.vllm (bare
      # wrapper) — kept here because they're ROCm/strix-specific, while
      # libibverbs owns only the kernel + rdma-core layer.
      vllmRdmaExtrasOverlay = _final: prev: {
        pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
          (pyfinal: _pyprev: {
            rixl = pyfinal.callPackage ./pkgs/rixl { };
            cupy-rocm-7-0 = pyfinal.callPackage ./pkgs/cupy-rocm { };
            lmcache = pyfinal.callPackage ./pkgs/lmcache {
              rixl = pyfinal.rixl;
            };
          })
        ];
      };

      overlays = {
        # Composed default for strix-halo NixOS machines. Order matters:
        # libibverbs's bare vllm overlay first (defines vllm via the
        # wrapper); rdma extras (rixl/lmcache/cupy-rocm) on top so vllm
        # picks them up at runtime via try/except imports; then strix
        # additions (ec-su, llama-cpp-rocm). Doesn't include rocm-narrow —
        # that's opt-in via overlays.rocm-narrow because it invalidates
        # downstream caches.
        default = nixpkgs.lib.composeManyExtensions [
          inputs.usb4-rdma.overlays.rdma-core-usb4
          vllmRdmaExtrasOverlay
          strixAdditionsOverlay
        ];

        # Just the RDMA-enabled vllm composition without strix additions.
        # Useful for non-strix consumers that want vllm+rixl+lmcache but
        # not ec-su/llama-cpp-rocm.
        vllm-rdma = nixpkgs.lib.composeManyExtensions [
          inputs.usb4-rdma.overlays.rdma-core-usb4
          vllmRdmaExtrasOverlay
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
        rocm-narrow = _: prev: {
          rocmPackages = prev.rocmPackages.overrideScope (
            _: rocmPrev: {
              clr = rocmPrev.clr.override {
                localGpuTargets = rocmTargets;
              };
            }
          );
        };

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
        rocm-narrow-deep = final: prev: {
          rocmPackages = prev.rocmPackages.overrideScope (
            _: rocmPrev:
            let
              narrow = drv: drv.override { gpuTargets = rocmTargets; };
            in
            {
              clr = rocmPrev.clr.override { localGpuTargets = rocmTargets; };

              rccl       = narrow rocmPrev.rccl;
              hipblaslt  = narrow rocmPrev.hipblaslt;
              hipfft     = narrow rocmPrev.hipfft;
              hiprand    = narrow rocmPrev.hiprand;
              hipsparse  = narrow rocmPrev.hipsparse;
              miopen     = narrow rocmPrev.miopen;
              rocblas    = narrow rocmPrev.rocblas;
              rocfft     = narrow rocmPrev.rocfft;
              rocrand    = narrow rocmPrev.rocrand;
              rocsolver  = narrow rocmPrev.rocsolver;
              rocsparse  = narrow rocmPrev.rocsparse;

              composable_kernel_base = rocmPrev.composable_kernel_base.override {
                gpuTargets = [ "gfx90a" "gfx11-generic" ];
              };

              aotriton = rocmPrev.aotriton.overrideAttrs (old: {
                cmakeFlags = (final.lib.filter
                  (f: !final.lib.hasPrefix "-DAOTRITON_TARGET_ARCH" f)
                  (old.cmakeFlags or [ ])) ++ [
                  "-DAOTRITON_TARGET_ARCH:STRING=gfx1151"
                ];
              });
            }
          );
        };
      };

      defaultPackagesFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ overlays.default ];
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
          benchmarks = import ./bench/default.nix {
            inherit pkgs;
            packages = {
              inherit (pkgs) llama-cpp-rocm llama-cpp-vulkan;
            };
          };
        in
        pkgs.lib.concatMapAttrs (
          model: benchs:
          pkgs.lib.mapAttrs' (name: drv: {
            name = "bench-${model}-${name}";
            value = drv;
          }) benchs
        ) benchmarks;

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
    in
    {
      inherit overlays;

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
      };

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

      packages = perSystem (
        { pkgs, ... }:
        {
          default = pkgs.llama-cpp-rocm;
          inherit (pkgs)
            ec-su-axb35-monitor
            fastflowlm
            tokenizers-cpp
            xrt
            xrt-amdxdna
            llama-cpp-rocm
            llama-cpp-vulkan
            ;
        }
        // mkBenchmarkPackages pkgs
        // mkFastflowlmBenchmarks pkgs
      );

      apps = perSystem (
        { pkgs, ... }:
        {
          llama-cli = {
            type = "app";
            program = toString (
              pkgs.writeShellScript "llama-cli" ''
                export HSA_OVERRIDE_GFX_VERSION=11.5.1
                ${pkgs.llama-cpp-rocm}/bin/llama-cli "$@"
              ''
            );
            meta.description = "Run llama-cli with the Strix Halo ROCm environment";
          };

          llama-server = {
            type = "app";
            program = toString (
              pkgs.writeShellScript "llama-server" ''
                export HSA_OVERRIDE_GFX_VERSION=11.5.1
                ${pkgs.llama-cpp-rocm}/bin/llama-server "$@"
              ''
            );
            meta.description = "Run llama-server with the Strix Halo ROCm environment";
          };

          flm = {
            type = "app";
            program = "${pkgs.fastflowlm}/bin/flm";
            meta.description = "FastFlowLM CLI on the AMD Ryzen AI NPU (Strix Halo XDNA2)";
          };
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
