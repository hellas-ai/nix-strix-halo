# Benchmark matrix generator
{ pkgs, packages }:

let
  runner = import ./runner.nix;

  # Standard package set for all benchmarks
  standardPackages = [
    (pkgs.llama-cpp.override {
      rocmSupport = true;
      rpcSupport = true;
    })
    packages.llama-cpp-gfx1151
    packages.llama-cpp-gfx1151-rocwmma
    packages.llama-cpp-vulkan-amdvlk
    packages.llama-cpp-vulkan-radv
  ];

  # Model configurations with their benchmark parameters
  modelConfigs = {
    llama2-7b = {
      path = "/models/llama-2-7b/llama-2-7b.Q4_K_M.gguf";
      benchmarks = {
        # Batch size effects
        batchSizes = {
          params = map (b: { batch = b; fa = 1; }) 
            [1 2 4 8 32 64 128 256 512];
        };
        
        # Flash attention comparison
        flashAttention = {
          params = [
            { fa = 0; ngl = 999; }
            { fa = 1; ngl = 999; }
          ];
        };
      };
    };

    qwen25-32b = {
      path = "/models/qwen2.5-32b-instruct/qwen2.5-32b-instruct-q8_0-00001-of-00009.gguf";
      benchmarks = {
        # Memory offloading tests
        memoryTests = {
          params = [
            { fa = 1; ngl = 50; }
            { fa = 1; ngl = 99; }
          ];
        };
      };
    };

    qwen3-coder-30b = {
      path = "/models/qwen3-coder-30b-a3b/BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf";
      benchmarks = {
        flashAttention = {
          params = [
            { fa = 0; ngl = 999; }
            { fa = 1; ngl = 999; }
          ];
        };
        
        promptProcessing = {
          params = [
            { batch = 128; fa = 1; ngl = 999; }
            { batch = 512; fa = 1; ngl = 999; }
          ];
        };
      };
    };
  };

  # Generate benchmark name from config
  mkName = config:
    builtins.concatStringsSep "-" (
      [config.package.pname]
      ++ (pkgs.lib.optional (config ? batch) "b${toString config.batch}")
      ++ (pkgs.lib.optional (config ? fa) "fa${toString config.fa}")
      ++ (pkgs.lib.optional (config ? ngl) "ngl${toString config.ngl}")
    );

  # Generate benchmark runner invocation
  mkBenchmark = { model, package, batch ? null, fa ? null, ngl ? null, rpc ? null, ... } @ args:
    runner {
      inherit pkgs batch fa ngl rpc;
      llamaCppPackage = package;
      modelPath = model;
      extraArgs = args.extraArgs or "";
    };

  # Generate all benchmarks for a model
  generateModelBenchmarks = modelName: modelConfig: let
    # Expand benchmark definitions into concrete configs
    allConfigs = pkgs.lib.flatten (
      pkgs.lib.mapAttrsToList (benchName: benchDef:
        pkgs.lib.flatten (
          map (pkg:
            map (params: { package = pkg; } // params) benchDef.params
          ) standardPackages
        )
      ) modelConfig.benchmarks
    );
  in
    builtins.listToAttrs (
      map (config: {
        name = mkName config;
        value = mkBenchmark (config // { model = modelConfig.path; });
      }) allConfigs
    );
in {
  # Generate benchmarks for each model
  llama2-7b = generateModelBenchmarks "llama2-7b" modelConfigs.llama2-7b;
  qwen25-32b = generateModelBenchmarks "qwen25-32b" modelConfigs.qwen25-32b;
  qwen3-coder-30b = generateModelBenchmarks "qwen3-coder-30b" modelConfigs.qwen3-coder-30b;
}