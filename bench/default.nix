{
  pkgs,
  tools,
  modelRoot ? "/models",
  matrixMetadata ? { },
}:
let
  inherit (pkgs) lib;
  benchLib = import ./lib.nix { inherit lib; };

  benchmarkTools = map benchLib.normalizeTool tools;

  modelConfigs = {
    llama2-7b =
      benchLib.mkModel {
        name = "llama2-7b";
        path = "${modelRoot}/llama-2-7b/llama-2-7b.Q4_K_M.gguf";
      }
      // {
        benchmarks = {
          batchSizes = {
            params =
              map
                (batch: {
                  inherit batch;
                  fa = 1;
                })
                [
                  1
                  2
                  4
                  8
                  32
                  64
                  128
                  256
                  512
                ];
          };

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
        };
      };

    qwen25-32b =
      benchLib.mkModel {
        name = "qwen25-32b";
        path = "${modelRoot}/qwen2.5-32b-instruct/qwen2.5-32b-instruct-q8_0-00001-of-00009.gguf";
      }
      // {
        benchmarks = {
          memoryTests = {
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

    qwen3-coder-30b =
      benchLib.mkModel {
        name = "qwen3-coder-30b";
        path = "${modelRoot}/qwen3-coder-30b-a3b/BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf";
      }
      // {
        benchmarks = {
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

          promptProcessing = {
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

  mkName =
    {
      tool,
      params,
      ...
    }:
    builtins.concatStringsSep "-" (
      [
        (tool.package.pname or tool.package.name)
        tool.name
      ]
      ++ (lib.optional (params ? batch) "b${toString params.batch}")
      ++ (lib.optional (params ? fa) "fa${toString params.fa}")
      ++ (lib.optional (params ? ngl) "ngl${toString params.ngl}")
    );

  mkBenchmark =
    model:
    {
      tool,
      params,
      scenario,
    }@config:
    let
      name = mkName config;
    in
    benchLib.mkLlamaCppBenchmark {
      inherit pkgs name params;
      inherit (tool)
        env
        package
        packages
        requirements
        ;
      executable = "${tool.package}/bin/${tool.executable}";
      model = removeAttrs model [ "benchmarks" ];
      extraArgs = tool.args;
      metadata = lib.recursiveUpdate {
        suite = "local-gguf";
        inherit scenario;
        tool = {
          inherit (tool) backend;
          packageRole = tool.name;
        };
      } (lib.recursiveUpdate matrixMetadata tool.metadata);
    };

  generateModelBenchmarks =
    model:
    let
      expanded = lib.flatten (
        lib.mapAttrsToList (
          scenario: benchDef:
          lib.flatten (
            map (
              tool:
              map (params: {
                inherit
                  scenario
                  tool
                  params
                  ;
              }) benchDef.params
            ) benchmarkTools
          )
        ) model.benchmarks
      );
    in
    builtins.listToAttrs (
      map (config: {
        name = mkName config;
        value = mkBenchmark model config;
      }) expanded
    );
in
assert lib.assertMsg (
  benchmarkTools != [ ]
) "bench/default.nix requires at least one tool descriptor";
{
  llama2-7b = generateModelBenchmarks modelConfigs.llama2-7b;
  qwen25-32b = generateModelBenchmarks modelConfigs.qwen25-32b;
  qwen3-coder-30b = generateModelBenchmarks modelConfigs.qwen3-coder-30b;
}
