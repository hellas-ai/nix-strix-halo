{
  pkgs,
  package,
  modelsRoot ? null,
  modelRoot ? null,
  npuSystemFeature ? "xdna2",
  hostProfile ? "linux-amd-xdna",
  matrixMetadata ? { },
}:

let
  inherit (pkgs) lib;
  benchLib = import ./lib.nix { inherit lib; };

  resolvedModelsRoot = if modelsRoot != null then modelsRoot else benchLib.defaultModelsRoot pkgs;
  resolvedModelRoot =
    if modelRoot != null then modelRoot else benchLib.modelPath resolvedModelsRoot [ "flm" ];

  prompts = {
    short = "Explain in one sentence what an NPU is.";

    medium = ''
      In about 200 words, contrast a tile-structured NPU with a SIMD CPU and
      a SIMT GPU for transformer inference. Cover dataflow execution, on-chip
      memory hierarchy, and where each architecture wins for the attention path
      of an LLM.
    '';

    long = ''
      Write a technical explanation of how a tile-structured kernel
      architecture on the AMD XDNA2 NPU enables efficient transformer
      inference. Cover why dataflow execution helps matmul-heavy attention, how
      the on-chip memory tile hierarchy reduces DRAM pressure, what unified
      memory changes on Strix Halo, and where the NPU is limited versus a
      discrete GPU.
    '';
  };

  clean = lib.replaceStrings [ ":" "." "/" "_" ] [ "-" "-" "-" "-" ];

  mkFlmArgs =
    {
      modelTag,
      ctxLen ? null,
      prefillChunkLen ? null,
      pmode ? null,
    }:
    lib.optionals (ctxLen != null) [
      "-c"
      ctxLen
    ]
    ++ lib.optionals (prefillChunkLen != null) [
      "--prefill-chunk-len"
      prefillChunkLen
    ]
    ++ lib.optionals (pmode != null) [
      "--pmode"
      pmode
    ]
    ++ [ modelTag ];

  modelConfigs = {
    llama3-2-1b = {
      tag = "llama3.2:1b";
      description = "Llama 3.2 1B FastFlowLM NPU model";
      cases = [
        { promptKey = "short"; }
        { promptKey = "medium"; }
        {
          promptKey = "long";
          ctxLen = 8192;
          prefillChunkLen = 512;
        }
      ];
    };

    gemma4-it-e4b = {
      tag = "gemma4-it:e4b";
      description = "Gemma 4 IT E4B FastFlowLM NPU model";
      cases = [
        { promptKey = "medium"; }
        {
          promptKey = "long";
          ctxLen = 8192;
          prefillChunkLen = 512;
        }
      ];
    };

    gpt-oss-20b = {
      tag = "gpt-oss:20b";
      description = "GPT OSS 20B FastFlowLM NPU model";
      cases = [
        { promptKey = "long"; }
        {
          promptKey = "long";
          ctxLen = 8192;
          prefillChunkLen = 512;
        }
      ];
    };
  };

  mkCaseName =
    {
      promptKey,
      ctxLen ? null,
      prefillChunkLen ? null,
      pmode ? null,
    }:
    "fastflowlm-${clean promptKey}"
    + lib.optionalString (ctxLen != null) "-ctx${toString ctxLen}"
    + lib.optionalString (prefillChunkLen != null) "-chunk${toString prefillChunkLen}"
    + lib.optionalString (pmode != null) "-${clean pmode}";

  mkBenchmark =
    modelName: model:
    {
      promptKey,
      ctxLen ? null,
      prefillChunkLen ? null,
      pmode ? null,
    }@case:
    let
      name = mkCaseName case;
      params = lib.filterAttrs (_: value: value != null) {
        inherit
          ctxLen
          prefillChunkLen
          pmode
          ;
      };
      flmArgs = mkFlmArgs {
        inherit
          ctxLen
          prefillChunkLen
          pmode
          ;
        modelTag = model.tag;
      };
      runner = pkgs.writeShellScript "${modelName}-${name}-runner" ''
        set -euo pipefail

        export HOME="$TMPDIR/home"
        mkdir -p "$HOME"

        {
          echo "/verbose"
          printf '%s\n' ${lib.escapeShellArg prompts.${promptKey}}
        } | ${package}/bin/flm run ${benchLib.shellArgs flmArgs}
      '';
    in
    benchLib.mkBenchmark {
      inherit
        pkgs
        name
        package
        ;
      command = [ runner ];
      env = {
        FLM_MODEL_PATH = resolvedModelRoot;
        FLM_DISABLE_UPDATE_CHECK = "1";
      };
      requirements = {
        systemFeatures = [ npuSystemFeature ];
        hostProfiles = [ hostProfile ];
        sandboxPaths = [
          "/dev/accel"
          "/sys/class/accel"
          resolvedModelRoot
        ];
      };
      metadata = lib.recursiveUpdate {
        kind = "fastflowlm";
        suite = "fastflowlm";
        accelerator = "npu";
        prompt = promptKey;
        inherit params;
        model = {
          name = modelName;
          inherit (model) tag description;
          path = resolvedModelRoot;
        };
        tool = {
          backend = "npu";
          packageRole = "fastflowlm";
        };
      } matrixMetadata;
      meta.platforms = [ "x86_64-linux" ];
      description = "Run FastFlowLM NPU benchmark ${modelName}/${name}";
    };

  generateModelBenchmarks =
    modelName: model:
    builtins.listToAttrs (
      map (case: {
        name = mkCaseName case;
        value = mkBenchmark modelName model case;
      }) model.cases
    );

  benchmarks = lib.mapAttrs generateModelBenchmarks modelConfigs;
in
{
  inherit
    benchmarks
    mkBenchmark
    prompts
    ;
}
