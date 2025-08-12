# Benchmark matrix generator - Clean declarative structure
{
  pkgs,
  packages,
}: let
  runner = import ./runner.nix;

  # === PACKAGE REGISTRY ===
  # Central definition of all available packages for benchmarking
  availablePackages = {
    # Upstream nixpkgs version
    nixpkgs = pkgs.llama-cpp.override {
      rocmSupport = true;
      rpcSupport = true;
    };

    # Vulkan build (cross-platform)
    vulkan = packages.llama-cpp-vulkan;

    # Our custom builds by target
    gfx1151 = {
      standard = packages.llama-cpp-gfx1151;
      rocwmma = packages.llama-cpp-gfx1151-rocwmma;
    };

    gfx110X = {
      standard = packages.llama-cpp-gfx110X;
      rocwmma = packages.llama-cpp-gfx110X-rocwmma;
    };

    gfx120X = {
      standard = packages.llama-cpp-gfx120X;
      rocwmma = packages.llama-cpp-gfx120X-rocwmma;
    };
  };

  # === TEST PROFILES ===
  # Reusable parameter sets for different test scenarios
  testProfiles = {
    # Basic full GPU offload test
    baseline = {
      params = {
        fa = 1;
        ngl = 999;
      };
    };

    # Compare with/without flash attention
    flashAttention = {
      params = [
        {
          fa = 0;
          ngl = 999;
        }
        {
          fa = 1;
          ngl = 999;
        }
      ];
    };

    # Test different batch sizes
    batchSweep = sizes: {
      params =
        map (b: {
          batch = b;
          fa = 1;
        })
        sizes;
    };

    # Test partial GPU offloading
    memoryOffload = ngls: {
      params =
        map (ngl: {
          fa = 1;
          inherit ngl;
        })
        ngls;
    };

    # Standard comparison suite
    comparison = {
      packages = ["nixpkgs" "standard" "rocwmma"];
      params = {
        fa = 1;
        ngl = 999;
      };
    };
  };

  # === MODEL CONFIGURATIONS ===
  # Define models and their benchmark suites
  modelConfigs = {
    llama2-7b = {
      path = "/models/llama-2-7b/llama-2-7b.Q4_K_M.gguf";
      target = "gfx1151";
      benchmarks = {
        # Compare nixpkgs vs custom vs rocwmma
        comparison = {
          packages = [
            availablePackages.nixpkgs
            availablePackages.gfx1151.standard
            availablePackages.gfx1151.rocwmma
            availablePackages.vulkan
          ];
          params = [
            {
              fa = 1;
              ngl = 999;
            }
          ];
        };

        # Test different Vulkan drivers
        vulkanDrivers = {
          packages = [availablePackages.vulkan];
          params = [
            {
              fa = 1;
              ngl = 999;
              # Default (auto-detect)
            }
            {
              fa = 1;
              ngl = 999;
              vkIcdFilenames = "${pkgs.mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json";
            }
            {
              fa = 1;
              ngl = 999;
              vkIcdFilenames = "${pkgs.amdvlk}/share/vulkan/icd.d/amd_icd64.json";
            }
          ];
        };

        # Batch size effects (rocwmma only)
        batchSizes = {
          packages = [availablePackages.gfx1151.rocwmma availablePackages.vulkan];
          params = map (b: {
            batch = b;
            fa = 1;
          }) [1 2 4 8 32 64 128 256 512];
        };

        # Flash attention comparison
        flashAttention = {
          packages = [
            availablePackages.vulkan
            availablePackages.gfx1151.rocwmma
          ];
          params = [
            {
              fa = 0;
              ngl = 999;
            }
            {
              fa = 1;
              ngl = 999;
            }
          ];
        };
      };
    };

    qwen25-32b = {
      path = "/models/qwen2.5-32b-instruct/qwen2.5-32b-instruct-q8_0-00001-of-00009.gguf";
      target = "gfx1151";
      benchmarks = {
        # Standard comparison
        comparison = {
          packages = [
            availablePackages.nixpkgs
            availablePackages.gfx1151.standard
            availablePackages.gfx1151.rocwmma
            availablePackages.vulkan
          ];
          params = [
            {
              fa = 1;
              ngl = 999;
            }
          ];
        };

        # Memory offloading tests
        memoryTests = {
          packages = [availablePackages.gfx1151.rocwmma];
          params = [
            {
              fa = 1;
              ngl = 50;
            }
            {
              fa = 1;
              ngl = 99;
            }
          ];
        };
      };
    };

    qwen3-coder-30b = {
      path = "/models/qwen3-coder-30b-a3b/BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf";
      target = "gfx1151";
      benchmarks = {
        # BF16 optimized testing
        comparison = {
          packages = [
            availablePackages.nixpkgs
            availablePackages.gfx1151.standard
            availablePackages.gfx1151.rocwmma
            availablePackages.vulkan
          ];
          params = [
            {
              fa = 1;
              ngl = 999;
            }
          ];
        };

        flashAttention = {
          packages = [availablePackages.gfx1151.rocwmma availablePackages.vulkan];
          params = [
            {
              fa = 0;
              ngl = 999;
            }
            {
              fa = 1;
              ngl = 999;
            }
          ];
        };

        promptProcessing = {
          packages = [availablePackages.gfx1151.rocwmma availablePackages.vulkan];
          params = [
            {
              batch = 128;
              fa = 1;
              ngl = 999;
            }
            {
              batch = 512;
              fa = 1;
              ngl = 999;
            }
          ];
        };
      };
    };
  };

  # === GENERATION FUNCTIONS ===
  # Generate benchmark name from config
  mkName = config: let
    driverSuffix = 
      if config ? vkIcdFilenames then
        if builtins.match ".*radeon.*" config.vkIcdFilenames != null then "radv"
        else if builtins.match ".*amd_icd.*" config.vkIcdFilenames != null then "amdvlk"
        else "custom"
      else "";
  in
    builtins.concatStringsSep "-" (
      [config.package.pname]
      ++ (pkgs.lib.optional (config ? batch) "b${toString config.batch}")
      ++ (pkgs.lib.optional (config ? fa) "fa${toString config.fa}")
      ++ (pkgs.lib.optional (config ? ngl) "ngl${toString config.ngl}")
      ++ (pkgs.lib.optional (driverSuffix != "") driverSuffix)
    );

  # Generate benchmark runner invocation
  mkBenchmark = {
    model,
    package,
    batch ? null,
    fa ? null,
    ngl ? null,
    rpc ? null,
    vkIcdFilenames ? null,
    ...
  } @ args:
    runner {
      inherit pkgs batch fa ngl rpc vkIcdFilenames;
      llamaCppPackage = package;
      modelPath = model;
      extraArgs = args.extraArgs or "";
    };

  # Generate all benchmarks for a model
  generateModelBenchmarks = modelName: modelConfig: let
    # Expand benchmark definitions into concrete configs
    expandBenchmark = benchName: benchDef:
      pkgs.lib.flatten (
        map (
          pkg:
            map (params:
              {
                package = pkg;
              }
              // params)
            benchDef.params
        )
        benchDef.packages
      );

    # Get all configs for this model
    allConfigs = pkgs.lib.flatten (
      pkgs.lib.mapAttrsToList expandBenchmark modelConfig.benchmarks
    );

    # Convert to benchmark attribute set
    benchmarks = builtins.listToAttrs (
      map (config: {
        name = mkName config;
        value = mkBenchmark (config // {model = modelConfig.path;});
      })
      allConfigs
    );
  in
    benchmarks;
in {
  # Generate benchmarks for each model
  llama2-7b = generateModelBenchmarks "llama2-7b" modelConfigs.llama2-7b;
  qwen25-32b = generateModelBenchmarks "qwen25-32b" modelConfigs.qwen25-32b;
  qwen3-coder-30b = generateModelBenchmarks "qwen3-coder-30b" modelConfigs.qwen3-coder-30b;
}
