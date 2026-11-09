{
  description = "Strix Halo NixOS configuration and tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # H3 retains the qualified July Python/ROCm dependency graph until the coordinated bump.
    nixpkgs-legacy.url = "github:NixOS/nixpkgs/61b7c44c4073f0b827768aff0049561b5110ea5a";

    nix-ai-tools = {
      url = "github:numtide/nix-ai-tools";
      inputs.nixpkgs.follows = "nixpkgs-legacy";
    };

    nix-darwin = {
      url = "github:nix-darwin/nix-darwin";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        pyproject-nix.follows = "pyproject-nix";
      };
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
      };
    };

    ec-su-axb35 = {
      url = "github:cmetz/ec-su_axb35-linux";
      flake = false;
    };

    llama-cpp-spacemit = {
      url = "github:spacemit-com/llama.cpp/v0.1.5";
      flake = false;
    };

    fastflowlm = {
      url = "github:FastFlowLM/FastFlowLM";
      flake = false;
    };

    mlx-metal-src = {
      url = "github:ml-explore/mlx/v0.32.1";
      flake = false;
    };

    thunderbolt-ibverbs = {
      url = "github:hellas-ai/thunderbolt-ibverbs";
      inputs.nixpkgs.follows = "nixpkgs-legacy";
    };

    thunderbolt-ibverbs-glm = {
      url = "github:hellas-ai/thunderbolt-ibverbs";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    xrt-src = {
      url = "git+https://github.com/Xilinx/XRT?ref=XRT-2.21&submodules=1";
      flake = false;
    };

    xdna-driver-src = {
      url = "git+https://github.com/amd/xdna-driver?ref=1.7&submodules=1";
      flake = false;
    };

    # BEGIN generated TheRock source inputs
    # Complete pinned trees are part of the qualified source-assembler identities.
    "therock-src-10-0-gfx1151-root" = {
      url = "github:ROCm/TheRock/16adc4d875fd4f65ea23c7c84e1c66706fde3047";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-libraries-36c771f3" = {
      url = "github:ROCm/rocm-libraries/8d1ae90eff7d022f26019ec55b2ec6a7674b3112";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-9c0ec3e9" = {
      url = "github:ROCm/rocm-systems/6b0e43f341195e203754e08f850e437ff2fc09f9";
      flake = false;
    };

    "therock-src-10-0-gfx1151-base-half-011b45bd" = {
      url = "github:ROCm/half/207ee58595a64b5c4a70df221f1e6e704b807811";
      flake = false;
    };

    "therock-src-10-0-gfx1151-base-rocm-cmake-ec512d84" = {
      url = "github:ROCm/rocm-cmake/10155d7272ea1bf79f6b5a9dbc339657af1aa372";
      flake = false;
    };

    "therock-src-10-0-gfx1151-compiler-amd-llvm-1e702164" = {
      url = "github:ROCm/llvm-project/8f497e0992fb7513f7f78a6f6b6f1056c375e961";
      flake = false;
    };

    "therock-src-10-0-gfx1151-compiler-hipify-ea979a85" = {
      url = "github:ROCm/HIPIFY/1c0fdf11502f06c4b4e978e35a1135240ea38e6f";
      flake = false;
    };

    "therock-src-10-0-gfx1151-compiler-spirv-llvm-translator-b9250448" = {
      url = "github:ROCm/SPIRV-LLVM-Translator/ddeda6468d45cf1e888a0e81e50bc170756a335b";
      flake = false;
    };

    "therock-src-10-0-gfx1151-math-libs-libhipcxx-10b48ff4" = {
      url = "github:ROCm/libhipcxx/8e336919711e1eb7fece0678a3d9e92bccfe6673";
      flake = false;
    };

    "therock-src-10-0-gfx1151-debug-tools-rocgdb-source-e3322990" = {
      url = "github:ROCm/rocgdb/f37d75b6ce6163d0173cd38233b8011ee4fec50c";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-abseil-cpp-471579c4" = {
      url = "github:abseil/abseil-cpp/255c84dadd029fd8ad25c5efb5933e47beaa00c7";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-cereal-7f708e93" = {
      url = "github:jrmadsen/cereal/40a30defcb8874270a04836d316502b08904b2bc";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-elfio-cf38c296" = {
      url = "github:serge1/ELFIO/575bfdb12cd90fa8913660295103549f151d116a";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-filesystem-807165f4" = {
      url = "github:gulrak/filesystem/8a2edd6d92ed820521d42c94d179462bf06b5ed3";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-gotcha-10fa2c98" = {
      url = "github:ROCm/GOTCHA/b944da10ff9b3364ef2e4b12e02cb2464e05dd48";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-json-a3f85fd2" = {
      url = "github:nlohmann/json/e41905fcb0938a7502d25ea5242c6b41396e21b0";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-perfetto-9d309359" = {
      url = "github:google/perfetto/eb5ef24c58d13cec289d733d03f0f3f0ed321b12";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-ptl-db721fa0" = {
      url = "github:jrmadsen/PTL/48df41625430d27ce43cf197fd467a8dda87cb45";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-pybind11-6da66437" = {
      url = "github:pybind/pybind11/914c06fb252b6cc3727d0eedab6736e88a3fcb01";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-dyninst-5f477976" = {
      url = "github:ROCm/dyninst/81cd27fc9ae4984df307bc41d5cb8c1ce9bf761e";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-json-ef3d2b7e" = {
      url = "github:nlohmann/json/55f93686c01528224f448c19128836e7df245f72";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-onetbb-9159eae3" = {
      url = "github:uxlfoundation/oneTBB/f1862f38f83568d96e814e469ab61f88336cc595";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-papi-ae8ec169" = {
      url = "github:icl-utk-edu/papi/c27003e1642f1c151c9b18ef9e7610072b3c852d";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-perfetto-c27c4344" = {
      url = "github:google/perfetto/35b3d9845c2f4017865c4dc93fafcf6d202f1651";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-pybind11-92d9411d" = {
      url = "github:jrmadsen/pybind11/1a917f1852eb7819b671fc3fa862840f4c491a07";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-fb6426bf" = {
      url = "github:ROCm/timemory/54ae9d214a63132da22c7cab21c5fb4b7b5049f8";
      flake = false;
    };

    "therock-src-10-0-gfx1151-third-party-sysdeps-linux-amd-mesa-mesa-fork-a692c85c" = {
      url = "github:ROCm/mesa-fork/f5302d0b394b20729b14da829389cb4203006676";
      flake = false;
    };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-compute-src-vendored-pyyaml-b64e5974" =
      {
        url = "github:yaml/pyyaml/49790e73684bebad1df05ef8d828fa12f685bffb";
        flake = false;
      };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-compute-src-lib-external-fmt-db1fc275" =
      {
        url = "github:fmtlib/fmt/407c905e45ad75fc29bf0f9bb7c5c2fd3475976f";
        flake = false;
      };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-compute-src-lib-external-json-c887b002" =
      {
        url = "github:nlohmann/json/98386eb08b609c172f9f4bf3e74c8d18c5ad583c";
        flake = false;
      };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-exte-811a2531" =
      {
        url = "github:ROCm/GOTCHA/6ef1232a0acc99f3340c2ca4c5d23890e9b8a459";
        flake = false;
      };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-exte-742cff1f" =
      {
        url = "github:libunwind/libunwind/24947191d61dda869e039e0414fe97e9f594acd5";
        flake = false;
      };

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-exte-d76a29fa".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-pybind11-92d9411d";

    "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-exte-1cc6846c" =
      {
        url = "github:jrmadsen/yaml-cpp/1b50109f7bea60bd382d8ea7befce3d2bd67da5f";
        flake = false;
      };

    # Complete pinned trees are part of the qualified source-assembler identities.
    "therock-src-7-15-gfx1151-root" = {
      url = "github:ROCm/TheRock/1de3171d00f6de55e9ed517dc6ca6e825d1e4b55";
      flake = false;
    };

    "therock-src-7-15-gfx1151-rocm-libraries-36c771f3" = {
      url = "github:ROCm/rocm-libraries/fa9cdd186b222f6449a6be1710038bdee4faa328";
      flake = false;
    };

    "therock-src-7-15-gfx1151-rocm-systems-9c0ec3e9" = {
      url = "github:ROCm/rocm-systems/8ceab8e581fbd0e191b1a0a56f9e59d8d1ca4e00";
      flake = false;
    };

    "therock-src-7-15-gfx1151-base-half-011b45bd".follows =
      "therock-src-10-0-gfx1151-base-half-011b45bd";

    "therock-src-7-15-gfx1151-base-rocm-cmake-ec512d84".follows =
      "therock-src-10-0-gfx1151-base-rocm-cmake-ec512d84";

    "therock-src-7-15-gfx1151-compiler-amd-llvm-1e702164" = {
      url = "github:ROCm/llvm-project/92fe31231dd83aedaedd822daa2079ca8a445547";
      flake = false;
    };

    "therock-src-7-15-gfx1151-compiler-hipify-ea979a85" = {
      url = "github:ROCm/HIPIFY/c917282dbc1fe85d3814807a2c77d4288cff112d";
      flake = false;
    };

    "therock-src-7-15-gfx1151-compiler-spirv-llvm-translator-b9250448".follows =
      "therock-src-10-0-gfx1151-compiler-spirv-llvm-translator-b9250448";

    "therock-src-7-15-gfx1151-math-libs-libhipcxx-10b48ff4" = {
      url = "github:ROCm/libhipcxx/3ea4c70920e0cdd1bab79d1897043090a93372b6";
      flake = false;
    };

    "therock-src-7-15-gfx1151-debug-tools-rocgdb-source-e3322990" = {
      url = "github:ROCm/rocgdb/1b5ded2e8e4239e9dc94b7c77c1b753cc7f68e26";
      flake = false;
    };

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-abseil-cpp-471579c4".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-abseil-cpp-471579c4";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-cereal-7f708e93".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-cereal-7f708e93";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-elfio-cf38c296".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-elfio-cf38c296";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-filesystem-807165f4".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-filesystem-807165f4";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-gotcha-10fa2c98".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-gotcha-10fa2c98";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-json-a3f85fd2".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-json-a3f85fd2";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-perfetto-9d309359".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-perfetto-9d309359";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-ptl-db721fa0".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-ptl-db721fa0";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-pybind11-6da66437".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-sdk-external-pybind11-6da66437";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-systems-external-dyninst-5f477976" = {
      url = "github:ROCm/dyninst/5dff47bf6fb53e51cdc5e98c8ef6656080931ede";
      flake = false;
    };

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-systems-external-json-ef3d2b7e".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-json-ef3d2b7e";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-systems-external-onetbb-9159eae3".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-onetbb-9159eae3";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-systems-external-papi-ae8ec169".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-papi-ae8ec169";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-systems-external-perfetto-c27c4344".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-perfetto-c27c4344";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-systems-external-pybind11-92d9411d".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-pybind11-92d9411d";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-fb6426bf".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-fb6426bf";

    "therock-src-7-15-gfx1151-third-party-sysdeps-linux-amd-mesa-mesa-fork-a692c85c" = {
      url = "github:ROCm/mesa-fork/22abfc06d7cf4c115ca2e368505d75ba4af174b4";
      flake = false;
    };

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-compute-src-vendored-pyyaml-b64e5974".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-compute-src-vendored-pyyaml-b64e5974";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-compute-src-lib-external-fmt-db1fc275".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-compute-src-lib-external-fmt-db1fc275";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-compute-src-lib-external-json-c887b002".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-compute-src-lib-external-json-c887b002";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-exte-811a2531".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-exte-811a2531";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-exte-742cff1f".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-exte-742cff1f";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-exte-d76a29fa".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-pybind11-92d9411d";

    "therock-src-7-15-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-exte-1cc6846c".follows =
      "therock-src-10-0-gfx1151-rocm-systems-projects-rocprofiler-systems-external-timemory-exte-1cc6846c";

    # END generated TheRock source inputs
  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }@inputs:
    let
      inherit (nixpkgs) lib;
      linuxSystems = [ "x86_64-linux" ];
      darwinSystems = [ "aarch64-darwin" ];
      systems = linuxSystems ++ darwinSystems;
      forAllSystems = lib.genAttrs systems;
      inputShortRev =
        input: input.shortRev or (if input ? rev then builtins.substring 0 7 input.rev else "unknown");
      inputVersion = prefix: input: "${prefix}-unstable-${inputShortRev input}";

      rocmTargetLib = import ./lib/rocm-targets.nix;
      therockTargetConfig = import ./pkgs/therock/targets.nix {
        inherit (rocmTargetLib) mkRocmTarget;
      };
      inherit (therockTargetConfig)
        defaultRocmTarget
        rocmTargets
        ;

      providers = import ./lib/providers.nix { inherit lib; };
      benchLib = import ./lib/bench/lib.nix { inherit lib; };
      therockPythonConfig = import ./pkgs/therock/python-config.nix { inherit lib; };

      defaultTherockSources = {
        rocm = builtins.fromJSON (builtins.readFile ./pkgs/therock/sources/rocm.json);
        pythonWheels = builtins.fromJSON (builtins.readFile ./pkgs/therock/sources/python-wheels.json);
        rocmSourcePins = builtins.fromJSON (builtins.readFile ./pkgs/therock/sources/rocm-source.json);
        rocmSourceTrees = import ./pkgs/therock/sources/source-tree.nix { inherit inputs; };
      };

      legacyTherockSources = {
        rocm = builtins.fromJSON (builtins.readFile ./pkgs/therock/sources/rocm-7-15.json);
        pythonWheels = builtins.fromJSON (builtins.readFile ./pkgs/therock/sources/python-wheels-7-15.json);
        rocmSourcePins = builtins.fromJSON (builtins.readFile ./pkgs/therock/sources/rocm-source-7-15.json);
        rocmSourceTrees = import ./pkgs/therock/sources/source-tree-7-15.nix { inherit inputs; };
      };

      thunderboltFor =
        legacyRuntime: if legacyRuntime then inputs.thunderbolt-ibverbs else inputs.thunderbolt-ibverbs-glm;
      multikernelOverlay = import ./overlays/multikernel.nix;
      spacemitK3Overlay = import ./overlays/spacemit-k3.nix {
        inherit inputs inputVersion;
      };

      # Re-export thunderbolt-ibverbs flake packages under the names
      # downstream callers expect. The upstream package set uses
      # unprefixed names (`bench-tools`, `perftest`); we prefix them so
      # the global package namespace doesn't collide. Also alias
      # `rdma-core` to the USB4-enabled build on Linux.
      thunderboltRenamesOverlay =
        legacyRuntime: _: prev:
        let
          tbPkgs = (thunderboltFor legacyRuntime).packages.${prev.stdenv.hostPlatform.system};
        in
        {
          thunderbolt-ibverbs-bench-tools = tbPkgs.bench-tools;
          thunderbolt-ibverbs-perftest = tbPkgs.perftest;
        }
        // lib.optionalAttrs prev.stdenv.hostPlatform.isLinux {
          inherit (tbPkgs)
            linux-thunderbolt
            linux-thunderbolt-dev
            linux-thunderbolt-modules
            rdma-core-usb4
            thunderbolt-ibverbs
            thunderbolt-ibverbs-linux-thunderbolt
            ;
          rdma-core = tbPkgs.rdma-core-usb4;
        };

      # Compose the overlay list for a given (rocmTarget, providers, cpu).
      # The active rocm/python providers feed the locally-defined-packages
      # overlay through `final.rocmPackages` / `final.python3Packages`.
      mkOverlays =
        {
          rocmTarget,
          rocmProvider ? "therock-bin",
          pythonProvider ? "therock-wheels",
          cpu ? null,
          legacyRuntime ? false,
          mediaRuntime ? false,
        }:
        let
          sources = if legacyRuntime then legacyTherockSources else defaultTherockSources;
        in
        [
          (thunderboltFor (legacyRuntime || mediaRuntime)).overlays.default
          (thunderboltRenamesOverlay (legacyRuntime || mediaRuntime))
          multikernelOverlay
          (import ./overlays/rocm.nix {
            inherit lib therockPythonConfig mediaRuntime;
            provider = rocmProvider;
            inherit rocmTarget rocmTargets;
            inherit sources;
          })
          (import ./overlays/python.nix {
            inherit
              lib
              therockPythonConfig
              legacyRuntime
              mediaRuntime
              ;
            provider = pythonProvider;
            inherit rocmTarget;
          })
          (import ./overlays/pkgs.nix {
            inherit
              inputs
              lib
              inputVersion
              rocmTarget
              rocmProvider
              pythonProvider
              legacyRuntime
              mediaRuntime
              ;
            inherit sources;
          })
        ]
        ++ lib.optional (cpu != null) (import ./overlays/mtune.nix { inherit lib cpu; });

      # The parameterised package set. Different `rocmTarget` /
      # `rocmProvider` / `pythonProvider` produce different package sets;
      # the flake's default outputs commit to one combination, the
      # legacyPackages tree exposes the rest per rocmTarget.
      pkgsFor =
        {
          system,
          rocmTarget ? defaultRocmTarget,
          rocmProvider ? "therock-bin",
          pythonProvider ? "therock-wheels",
          cpu ? null,
          legacyRuntime ? false,
          mediaRuntime ? false,
        }:
        import (if legacyRuntime then inputs.nixpkgs-legacy else nixpkgs) {
          inherit system;
          config = {
            allowUnfree = true;
          };
          overlays = mkOverlays {
            inherit
              rocmTarget
              rocmProvider
              pythonProvider
              cpu
              legacyRuntime
              mediaRuntime
              ;
          };
        };

      defaultPkgsFor = system: pkgsFor { inherit system; };
      perSystem = f: forAllSystems (system: f (defaultPkgsFor system));

      mkBenchmarkSuites = import ./lib/bench {
        inherit
          lib
          self
          defaultRocmTarget
          benchLib
          ;
      };

      hydraLib = import ./lib/hydra.nix {
        inherit
          lib
          self
          ;
      };

      mkLiveIsoConfiguration =
        {
          system ? "x86_64-linux",
          extraModules ? [ ],
          specialArgs ? { },
        }:
        nixpkgs.lib.nixosSystem {
          inherit system;
          specialArgs = {
            inherit inputs self;
          }
          // specialArgs;
          modules = [
            ./examples/configuration.nix
          ]
          ++ extraModules;
        };
    in
    {
      lib = {
        inherit
          defaultTherockSources
          mkLiveIsoConfiguration
          ;
        inherit (rocmTargetLib) mkRocmTarget;
        inherit (providers) rocmProviders pythonProviders pythonProviderStubs;
        therockTargets = therockTargetConfig;
        bench = benchLib;

        # Parameterised overlay builders. Power users compose these to
        # build their own package sets; see lib.mkRocmOverlay etc. The
        # flake's `overlays.default` is the composition of all of these
        # against the default rocmTarget + providers.
        mkRocmOverlay =
          args:
          import ./overlays/rocm.nix (
            {
              inherit lib rocmTargets;
              sources = defaultTherockSources;
            }
            // args
          );
        mkPythonOverlay = args: import ./overlays/python.nix ({ inherit lib; } // args);
        mkPkgsOverlay =
          args:
          import ./overlays/pkgs.nix (
            {
              inherit lib inputs inputVersion;
              sources = defaultTherockSources;
            }
            // args
          );
        mkMtuneOverlay = args: import ./overlays/mtune.nix ({ inherit lib; } // args);
      };

      overlays = {
        default = lib.composeManyExtensions (mkOverlays {
          rocmTarget = defaultRocmTarget;
          legacyRuntime = true;
        });
        python = self.lib.mkPythonOverlay {
          provider = "nixpkgs";
        };
        multikernel = multikernelOverlay;
        spacemitK3 = spacemitK3Overlay;
      };

      # NixOS modules
      nixosModules = {
        default = _: {
          imports = [
            inputs.thunderbolt-ibverbs.nixosModules.default
          ];
          nixpkgs.overlays = [
            self.overlays.default
          ];
        };
        rpc-server = import ./modules/rpc-server.nix;
        benchmark-executor = import ./modules/benchmark-executor.nix;
        benchmark-runner = import ./modules/benchmark-runner.nix;
        ec-su-axb35 = import ./modules/ec-su-axb35.nix;
        fastflowlm = import ./modules/fastflowlm.nix;
        amduprof = import ./modules/amduprof.nix;
        grafana-dashboards = import ./modules/grafana-dashboards.nix;
        smu-exporter = import ./modules/smu-exporter.nix;
        npu-exporter = import ./modules/npu-exporter.nix;
        multikernel = import ./modules/multikernel.nix;
        ryzenadj = import ./modules/ryzenadj.nix;
        thunderbolt-ibverbs = inputs.thunderbolt-ibverbs.nixosModules.default;
      };

      nixosConfigurations = {
        live-iso = mkLiveIsoConfiguration { };
      };

      darwinModules = rec {
        benchmark-executor = import ./modules/benchmark-executor.nix;
        vllm-metal =
          { lib, ... }:
          {
            imports = [ ./modules/vllm-metal.nix ];
            services.vllm-metal.package = lib.mkDefault self.packages.aarch64-darwin.vllm-metal;
          };
        default = vllm-metal;
      };
    }
    // {
      # Per-target pkgs sets. Use to escape-hatch to a non-default rocm
      # target: `nix build .#legacyPackages.x86_64-linux.gfx1103.llama-cpp-rocm`.
      mediaPackages = forAllSystems (
        system:
        lib.listToAttrs (
          map (rocmTarget: {
            name = rocmTarget.packageSuffix;
            value = pkgsFor {
              inherit system rocmTarget;
              mediaRuntime = true;
            };
          }) rocmTargets
        )
      );

      legacyPackages = forAllSystems (
        system:
        lib.listToAttrs (
          map (rocmTarget: {
            name = rocmTarget.packageSuffix;
            value = pkgsFor {
              inherit system rocmTarget;
              legacyRuntime = true;
            };
          }) rocmTargets
        )
      );

      benchmarks = perSystem mkBenchmarkSuites;

      packages = perSystem (
        pkgs:
        let
          system = pkgs.stdenv.hostPlatform.system;
          aiTools = inputs.nix-ai-tools.packages.${system};
          packageRuntimeEnv = import ./lib/runtime-env.nix { inherit lib; };
          inherit (packageRuntimeEnv) wrapRuntimeEnv;
          jacclPackage = pkgs.callPackage ./pkgs/jaccl (
            {
              mlx-src = inputs.mlx-metal-src;
            }
            // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              rdma-core = pkgs.rdma-core-usb4;
            }
            // lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin {
              darwinDeploymentTarget = "26.2";
            }
          );
          mkMlxLm =
            {
              mlxPackage,
              pythonPackages ? pkgs.python3Packages,
            }:
            let
              mlxLm = pythonPackages.mlx-lm.overridePythonAttrs (
                oldAttrs:
                {
                  dependencies = lib.unique (
                    lib.filter (dep: dep.pname or "" != "mlx") (
                      oldAttrs.dependencies or oldAttrs.propagatedBuildInputs or [ ]
                    )
                    ++ [
                      mlxPackage
                    ]
                    ++ lib.optional pkgs.stdenv.hostPlatform.isLinux pythonPackages.sentencepiece
                  );
                  meta = (oldAttrs.meta or { }) // {
                    mainProgram = "mlx_lm";
                  };
                }
                // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
                  doCheck = false;
                  doInstallCheck = false;
                  dontCheckRuntimeDeps = true;
                  pythonImportsCheck = [ "mlx_lm" ];
                }
              );
            in
            if mlxPackage ? passthru && mlxPackage.passthru ? rocmRuntimeEnv then
              wrapRuntimeEnv {
                inherit pkgs;
                package = mlxLm;
                name = "mlx-lm-rocm-runtime";
                env = mlxPackage.passthru.rocmRuntimeEnv;
              }
            else
              mlxLm;

          availableDerivationNames =
            names: lib.filter (name: builtins.hasAttr name pkgs && lib.isDerivation pkgs.${name}) names;

          linuxLocalPackageNames = [
            "amd-npu-exporter"
            "amdgpu-smu-exporter"
            "ec-su-axb35"
            "ec-su-axb35-monitor"
            "fastflowlm"
            "kerf-init"
            "kerf-multikernel"
            "llama-cpp-rocm"
            "llama-cpp-vulkan"
            "glm53-wmma-aot"
            "glm53-aot-stage"
            "glm53-serve"
            "llvm-aie"
            "mlir-aie"
            "mlir-aie-env"
            "linux-multikernel"
            "mscclpp-rocm"
            "multikernel-demo-initrd"
            "rocm-pm4-bootstrap"
            "rocm-pm4-clr-split"
            "rocm-pm4-prepared-source"
            "rocm-pm4-rocr"
            "rocm-pm4-rocr-split"
            "sglang-glm53-rocm"
            "sglang-rocm"
            "therock-python-wheels"
            "therock-rocm"
            "tokenizers-cpp"
            "torch-rocm"
            "xrt"
            "xrt-amdxdna"
          ];

          genericPackages = {
            default = pkgs.llama-cpp;
            pi-wrap =
              (pkgsFor {
                inherit system;
                legacyRuntime = true;
              }).callPackage
                ./pkgs/pi-wrap
                {
                  inherit (aiTools) pi;
                };
            pi = aiTools.pi;
            inherit (pkgs)
              llama-cpp
              thunderbolt-ibverbs-bench-tools
              thunderbolt-ibverbs-perftest
              ;
          }
          // lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin { jaccl = jacclPackage; };

          spacemitK3Packages = lib.optionalAttrs (system == "x86_64-linux") (
            let
              spacemitPkgs = import nixpkgs {
                inherit system;
                config.allowUnfree = true;
                overlays = [ spacemitK3Overlay ];
              };
            in
            {
              inherit (spacemitPkgs)
                llama-cpp-spacemit
                spacemit-k3-runtime
                spacemit-k3-toolchain
                ;
            }
          );

          linuxPackages =
            lib.genAttrs (availableDerivationNames linuxLocalPackageNames) (name: pkgs.${name})
            // {
              inherit (pkgs)
                linux-thunderbolt
                linux-thunderbolt-dev
                linux-thunderbolt-modules
                rdma-core-usb4
                thunderbolt-ibverbs
                thunderbolt-ibverbs-linux-thunderbolt
                ;
              live-iso = self.nixosConfigurations.live-iso.config.system.build.isoImage;
              npu-toolchain = pkgs.mlir-aie-env;

            };

          darwinPackages =
            let
              mkMlxMetalStack =
                pythonPackages:
                let
                  mlxNanobind = pythonPackages.callPackage ./pkgs/mlx/nanobind-2_13.nix { };
                  mlxMetalBackend = pythonPackages.callPackage ./pkgs/mlx/metal.nix {
                    mlx-src = inputs.mlx-metal-src;
                    jaccl = jacclPackage;
                    nanobind = mlxNanobind;
                    pname = "mlx-metal";
                    buildStage = 2;
                    darwinDeploymentTarget = "26.2";
                  };
                  mlxMetal = pythonPackages.callPackage ./pkgs/mlx/metal.nix {
                    mlx-src = inputs.mlx-metal-src;
                    jaccl = jacclPackage;
                    nanobind = mlxNanobind;
                    pname = "mlx";
                    buildStage = 1;
                    backendPackage = mlxMetalBackend;
                    darwinDeploymentTarget = "26.2";
                  };
                in
                {
                  inherit mlxMetal mlxMetalBackend;
                };

              # Python 3.14 currently aborts in libffi while building MLX's
              # macOS wheel. Keep the complete Metal stack on Python 3.13.
              mlxPythonPackages = pkgs.python313Packages;
              inherit (mkMlxMetalStack mlxPythonPackages) mlxMetal mlxMetalBackend;
              mlxLm = mkMlxLm {
                mlxPackage = mlxMetal;
                pythonPackages = mlxPythonPackages;
              };
              vllmMlxStack = mkMlxMetalStack pkgs.python312Packages;
              vllmMetal = pkgs.callPackage ./pkgs/vllm-metal {
                inherit (inputs)
                  pyproject-build-systems
                  pyproject-nix
                  uv2nix
                  ;
                mlxPackage = vllmMlxStack.mlxMetal;
                mlxMetalPackage = vllmMlxStack.mlxMetalBackend;
              };
            in
            {
              mlx = mlxMetal;
              mlx-lm = mlxLm;
              mlx-metal = mlxMetalBackend;
              vllm-metal = vllmMetal;
            };
        in
        genericPackages
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          sglang-rocm-h3 = self.legacyPackages.${system}.gfx1151.sglang-rocm-h3;
        }
        // spacemitK3Packages
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux linuxPackages
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin darwinPackages
      );

      apps = perSystem (
        pkgs:
        let
          system = pkgs.stdenv.hostPlatform.system;
          runtimeEnv = import ./lib/runtime-env.nix { inherit lib; };
          inherit (runtimeEnv) mkApp;

          # Positional helper. `mkApp` is the structured form; this is
          # the inline version for entries where the structure adds no
          # information.
          ap =
            package: binary: description:
            mkApp { inherit package binary description; };

          genericApps = {
            pi-wrap =
              ap self.packages.${system}.pi-wrap "pi-wrap"
                "Run pi with provider settings from environment";
          };

          linuxApps = {
            flm = ap pkgs.fastflowlm "flm" "Run the FastFlowLM CLI on AMD Ryzen AI NPUs";
            sglang-rocm = ap pkgs.sglang-rocm "sglang" "Run SGLang with ROCm";

            live-iso-vm =
              let
                iso = self.packages.${system}.live-iso;
              in
              {
                type = "app";
                program = toString (
                  pkgs.writeShellScript "live-iso-vm" ''
                    set -euo pipefail
                    iso_file=$(echo ${iso}/iso/*.iso)
                    if [ ! -f "$iso_file" ]; then
                      echo "live ISO not found under ${iso}/iso" >&2
                      exit 1
                    fi
                    kvm_args=()
                    if [ -w /dev/kvm ]; then
                      kvm_args+=(-enable-kvm -cpu host)
                    fi
                    exec ${pkgs.qemu}/bin/qemu-system-x86_64 \
                      "''${kvm_args[@]}" \
                      -m ''${LIVE_ISO_MEM:-4G} \
                      -smp ''${LIVE_ISO_CPUS:-4} \
                      -cdrom "$iso_file" \
                      -boot d \
                      "$@"
                  ''
                );
                meta.description = "Boot the strix-halo live ISO in QEMU (override LIVE_ISO_MEM, LIVE_ISO_CPUS)";
              };
          };

          darwinApps =
            let
              mlxLm = self.packages.${system}.mlx-lm;
              vllmMetal = self.packages.${system}.vllm-metal;
            in
            {
              mlx-lm = ap mlxLm "mlx_lm" "Run the MLX LM CLI";
              mlx-lm-generate = ap mlxLm "mlx_lm.generate" "Generate text with MLX LM";
              mlx-lm-chat = ap mlxLm "mlx_lm.chat" "Run the MLX LM chat CLI";
              mlx-lm-server = ap mlxLm "mlx_lm.server" "Run the MLX LM HTTP server";
              vllm-metal = ap vllmMetal "vllm" "Run vLLM with the Apple Silicon Metal plugin";
            };
        in
        genericApps
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux linuxApps
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin darwinApps
      );

      checks = perSystem (
        pkgs:
        lib.optionalAttrs (pkgs.stdenv.hostPlatform.system == "x86_64-linux") (
          let
            system = pkgs.stdenv.hostPlatform.system;
            src = self;
            runSourceCheck =
              name: nativeBuildInputs: command:
              pkgs.runCommandLocal "ci-${name}"
                {
                  inherit nativeBuildInputs;
                  meta.maintainers = with lib.maintainers; [ georgewhewell ];
                }
                ''
                  export HOME="$TMPDIR/home"
                  export XDG_CACHE_HOME="$TMPDIR/cache"
                  mkdir -p "$HOME" "$XDG_CACHE_HOME"
                  cp -R --no-preserve=mode,ownership ${src} "$TMPDIR/src"
                  chmod -R u+rwX "$TMPDIR/src"
                  cd "$TMPDIR/src"
                  ${command}
                  touch "$out"
                '';

            multikernelEval = lib.nixosSystem {
              inherit system;
              modules = [
                self.nixosModules.multikernel
                {
                  nixpkgs.pkgs = pkgs;
                  system.stateVersion = "26.05";
                  boot.loader.grub.enable = false;
                  fileSystems."/" = {
                    device = "/dev/null";
                    fsType = "ext4";
                  };
                  boot.multikernel = {
                    enable = true;
                    pool = {
                      cpus = "12-15,28-31";
                      memory = "8GB";
                    };
                    instances.blue = {
                      id = 1;
                      cpus = "12-13,28-29";
                      memory = "2GB";
                    };
                  };
                }
              ];
            };

            multikernelModule =
              let
                plain = builtins.unsafeDiscardStringContext;
                poolScript = plain multikernelEval.config.systemd.services.multikernel-pool.script;
                instanceScript = plain multikernelEval.config.systemd.services.multikernel-instance-blue.script;
                controller = lib.findFirst (
                  package: lib.getName package == "multikernelctl"
                ) null multikernelEval.config.environment.systemPackages;
              in
              pkgs.runCommandLocal "ci-multikernel-module"
                {
                  nativeBuildInputs = [ pkgs.gnugrep ];
                  meta.maintainers = with lib.maintainers; [ georgewhewell ];
                }
                ''
                  cat > pool.sh <<'EOF'
                  ${poolScript}
                  EOF
                  cat > instance.sh <<'EOF'
                  ${instanceScript}
                  EOF
                  grep -F -- "--cpus=12-15,28-31" pool.sh
                  grep -F -- "--memory=8GB" pool.sh
                  grep -F -- "kerf create blue" instance.sh
                  grep -F -- "--id=1" instance.sh
                  grep -F -- "console=mktty0" instance.sh
                  grep -F -- "multikernel.demo.name=blue" instance.sh
                  grep -F -- 'systemctl stop "multikernel-instance-$instance.service"' ${controller}/bin/multikernelctl
                  grep -F -- 'systemctl stop multikernel-pool.service' ${controller}/bin/multikernelctl
                  touch "$out"
                '';

            multikernelKernelConfig =
              pkgs.runCommandLocal "ci-multikernel-kernel-config"
                {
                  nativeBuildInputs = [ pkgs.gnugrep ];
                  meta.maintainers = with lib.maintainers; [ georgewhewell ];
                }
                ''
                  config=${pkgs.linux-multikernel.configfile}
                  for symbol in \
                    KEXEC_FILE KEXEC_HANDOVER KEXEC_HANDOVER_ENABLE_DEFAULT \
                    MULTIKERNEL MKTTY SMP HOTPLUG_CPU MEMORY_HOTPLUG \
                    MEMORY_HOTREMOVE DAXFS VSOCKETS; do
                    grep -qx "CONFIG_$symbol=y" "$config"
                  done
                  grep -qx '# CONFIG_DEFERRED_STRUCT_PAGE_INIT is not set' "$config"
                  grep -qx '# CONFIG_MULTIKERNEL_VSOCKETS is not set' "$config"
                  cp "$config" "$out"
                '';

            multikernelInitrd =
              pkgs.runCommandLocal "ci-multikernel-demo-initrd"
                {
                  nativeBuildInputs = [
                    pkgs.cpio
                    pkgs.gnugrep
                    pkgs.zstd
                  ];
                  meta.maintainers = with lib.maintainers; [ georgewhewell ];
                }
                ''
                  mkdir unpack
                  cd unpack
                  zstd -dc ${pkgs.multikernel-demo-initrd}/initrd | cpio -id
                  find . -printf '%P\n' | sort > "$out"
                  grep -qx init "$out"
                  grep -qx bin/sh "$out"
                  grep -qx bin/busybox "$out"
                  grep -qx bin/multikernel-demo "$out"
                  demo=$(find nix/store -maxdepth 1 -type f -name '*-multikernel-demo')
                  grep -Fq '/sys/devices/system/cpu/online' "$demo"
                  grep -Fq 'physical APIC' "$demo"
                '';

            multikernelKerf =
              pkgs.runCommandLocal "ci-multikernel-kerf"
                {
                  nativeBuildInputs = [ pkgs.kerf-multikernel ];
                  meta.maintainers = with lib.maintainers; [ georgewhewell ];
                }
                ''
                  kerf --help > "$out"
                  grep -q 'Multikernel Management System' "$out"
                  kerf create --help | grep -q 'Use physical APIC'
                '';
          in
          {
            deadnix = runSourceCheck "deadnix" [ pkgs.deadnix ] "deadnix --fail .";
            statix = runSourceCheck "statix" [ pkgs.statix ] "statix check .";
            vllm-metal-lock = runSourceCheck "vllm-metal-lock" [ pkgs.uv ] ''
              uv lock --project pkgs/vllm-metal --check --offline --no-python-downloads \
                --python ${pkgs.python312}/bin/python3.12
            '';
            nixfmt = runSourceCheck "nixfmt" [
              pkgs.nixfmt-tree
            ] "treefmt --tree-root . --walk filesystem --fail-on-change .";
            multikernel-demo-initrd = multikernelInitrd;
            multikernel-kerf = multikernelKerf;
            multikernel-kernel-config = multikernelKernelConfig;
            multikernel-module = multikernelModule;

          }
        )
        // lib.optionalAttrs (pkgs.stdenv.hostPlatform.system == "aarch64-darwin") {
          vllm-metal-module = pkgs.callPackage ./pkgs/vllm-metal/module-check.nix {
            inherit (inputs) nix-darwin;
            module = self.darwinModules.vllm-metal;
          };
        }
      );

      devShells = perSystem (pkgs: {
        default = pkgs.mkShell {
          packages =
            (with pkgs; [
              deadnix
              nix-fast-build
              nix-prefetch-git
              nixfmt-tree
              statix
            ])
            ++ (with inputs.nix-ai-tools.packages.${pkgs.stdenv.hostPlatform.system}; [
              pi
            ])
            ++ [
              self.packages.${pkgs.stdenv.hostPlatform.system}.pi-wrap
            ]
            ++ lib.optionals pkgs.stdenv.hostPlatform.isLinux [
              (pkgs.python3.withPackages (
                ps: with ps; [
                  boto3
                  numpy
                  pandas
                  plotly
                ]
              ))
            ];
        };
      });

      formatter = perSystem (pkgs: pkgs.nixfmt-tree);

      hydraBenchmarkJobs = perSystem hydraLib.mkBenchmarkJobs;

      hydraJobs = perSystem hydraLib.mkJobs;

    };
}
