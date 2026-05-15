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
        };

      overlays = {
        # Composed default: pulls libibverbs's RDMA-enabled vllm overlay
        # (wraps stock python3Packages.vllm + injects rixl + lmcache) and
        # layers our own strix additions (ec-su, llama-cpp-rocm) on top.
        # Consumers (nixos-config strix machines) get a vllm-rdma-ready
        # pkgs set with one overlay import. Doesn't include rocm-narrow —
        # that's still opt-in via overlays.rocm-narrow below since it
        # invalidates downstream caches.
        default = nixpkgs.lib.composeManyExtensions [
          inputs.usb4-rdma.overlays.vllm-rdma
          strixAdditionsOverlay
        ];

        # Narrows rocmPackages.clr to `rocmTargets` (currently gfx1151) for
        # faster builds on strix-halo. Applying this overlay invalidates
        # downstream caches and removes support for other GPUs from the system
        # OpenCL ICD — only enable on hosts that exclusively run gfx1151.
        rocm-narrow = _: prev: {
          rocmPackages = prev.rocmPackages.overrideScope (
            _: rocmPrev: {
              clr = rocmPrev.clr.override {
                localGpuTargets = rocmTargets;
              };
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
        rpc-server = import ./modules/rpc-server.nix;
        benchmark-runner = import ./modules/benchmark-runner.nix;
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
            llama-cpp-rocm
            llama-cpp-vulkan
            ;
        }
        // mkBenchmarkPackages pkgs
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
