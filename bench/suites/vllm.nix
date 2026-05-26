{
  pkgs,
  package,
  modelsRoot ? null,
  modelRoot ? null,
  hfCacheHome ? null,
  target,
  hostProfile ? "linux-amd-kfd",
  matrixMetadata ? { },
}:

let
  inherit (pkgs) lib;
  benchLib = import ../lib.nix { inherit lib; };

  resolvedModelsRoot = if modelsRoot != null then modelsRoot else benchLib.defaultModelsRoot pkgs;
  resolvedModelRoot = if modelRoot != null then modelRoot else resolvedModelsRoot;
  resolvedHfCacheHome =
    if hfCacheHome != null then
      hfCacheHome
    else
      benchLib.modelPath resolvedModelRoot [
        ".cache"
        "huggingface"
      ];

  clean = lib.replaceStrings [ ":" "." "/" "_" ] [ "-" "-" "-" "-" ];

  optionalValueArg =
    flag: value:
    lib.optionals (value != null) [
      flag
      value
    ];

  optionalFlag = flag: enabled: lib.optional enabled flag;

  normalizeArgs =
    args:
    if builtins.isList args then
      args
    else if args == "" then
      [ ]
    else
      lib.splitString " " args;

  modelConfigs = {
    qwen3-0-6b = {
      id = "Qwen/Qwen3-0.6B";
      description = "Qwen3 0.6B Hugging Face model for vLLM smoke benchmarks";
      cases = [
        {
          mode = "throughput";
          scenario = "smoke";
          inputLen = 128;
          outputLen = 32;
          numPrompts = 8;
          maxModelLen = 512;
        }
        {
          mode = "latency";
          scenario = "smoke";
          inputLen = 128;
          outputLen = 32;
          batchSize = 1;
          numItersWarmup = 1;
          numIters = 3;
          maxModelLen = 512;
        }
      ];
    };
  };

  mkCaseName =
    {
      mode,
      scenario,
      ...
    }:
    "vllm-rocm-${target.packageSuffix}-${clean mode}-${clean scenario}";

  mkCommonArgs =
    model:
    {
      maxModelLen ? null,
      tensorParallelSize ? null,
      gpuMemoryUtilization ? null,
      trustRemoteCode ? false,
      dtype ? null,
      extraArgs ? [ ],
      ...
    }:
    [
      "--model"
      model.id
    ]
    ++ optionalValueArg "--max-model-len" maxModelLen
    ++ optionalValueArg "--tensor-parallel-size" tensorParallelSize
    ++ optionalValueArg "--gpu-memory-utilization" gpuMemoryUtilization
    ++ optionalValueArg "--dtype" dtype
    ++ optionalFlag "--trust-remote-code" trustRemoteCode
    ++ normalizeArgs extraArgs;

  mkThroughputArgs =
    model:
    {
      inputLen,
      outputLen,
      numPrompts,
      backend ? "vllm",
      datasetName ? "random",
      disableDetokenize ? false,
      asyncEngine ? false,
      ...
    }@case:
    [
      "bench"
      "throughput"
      "--backend"
      backend
      "--dataset-name"
      datasetName
    ]
    ++ (
      if datasetName == "random" then
        [
          "--random-input-len"
          inputLen
          "--random-output-len"
          outputLen
        ]
      else
        [
          "--input-len"
          inputLen
          "--output-len"
          outputLen
        ]
    )
    ++ [
      "--num-prompts"
      numPrompts
    ]
    ++ optionalFlag "--disable-detokenize" disableDetokenize
    ++ optionalFlag "--async-engine" asyncEngine
    ++ mkCommonArgs model case;

  mkLatencyArgs =
    model:
    {
      inputLen,
      outputLen,
      batchSize,
      numItersWarmup,
      numIters,
      n ? null,
      disableDetokenize ? false,
      useBeamSearch ? false,
      ...
    }@case:
    [
      "bench"
      "latency"
      "--input-len"
      inputLen
      "--output-len"
      outputLen
      "--batch-size"
      batchSize
      "--num-iters-warmup"
      numItersWarmup
      "--num-iters"
      numIters
    ]
    ++ optionalValueArg "--n" n
    ++ optionalFlag "--disable-detokenize" disableDetokenize
    ++ optionalFlag "--use-beam-search" useBeamSearch
    ++ mkCommonArgs model case;

  mkArgs =
    model:
    {
      mode,
      ...
    }@case:
    if mode == "throughput" then
      mkThroughputArgs model case
    else if mode == "latency" then
      mkLatencyArgs model case
    else
      throw "unsupported vLLM benchmark mode: ${mode}";

  mkRunner =
    model: case:
    let
      vllmArgs = mkArgs model case;
      hfCacheRepo = "models--${lib.replaceStrings [ "/" ] [ "--" ] model.id}";
    in
    pkgs.writeShellScript "vllm-${target.packageSuffix}-${case.mode}-${case.scenario}-runner" ''
      set -euo pipefail

      export HOME="$TMPDIR/home"
      export XDG_CACHE_HOME="$TMPDIR/cache"
      export VLLM_CACHE_ROOT="$TMPDIR/vllm-cache"
      export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor-cache"
      export TRITON_CACHE_DIR="$TMPDIR/triton-cache"
      mkdir -p \
        "$HOME" \
        "$XDG_CACHE_HOME" \
        "$VLLM_CACHE_ROOT" \
        "$TORCHINDUCTOR_CACHE_DIR" \
        "$TRITON_CACHE_DIR"

      model_cache="$HF_HOME/hub/${hfCacheRepo}"
      if [ ! -d "$HF_HOME" ]; then
        echo "HF_HOME does not exist: $HF_HOME" >&2
        exit 2
      fi
      if [ ! -d "$model_cache" ]; then
        echo "Hugging Face model cache is missing: $model_cache" >&2
        if [ -d "$HF_HOME/hub" ]; then
          echo "available cached models:" >&2
          find "$HF_HOME/hub" -maxdepth 1 -type d -name 'models--*' -printf '  %f\n' | sort >&2
        fi
        exit 2
      fi
      if [ -z "$(find "$model_cache/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]; then
        echo "Hugging Face model cache has no snapshots: $model_cache" >&2
        exit 2
      fi

      ${package}/bin/vllm ${benchLib.shellArgs vllmArgs} --output-json "$out/results.json"
      cat "$out/results.json"
    '';

  caseParams =
    case:
    lib.filterAttrs (_: value: value != null) (
      removeAttrs case [
        "backend"
        "datasetName"
        "disableDetokenize"
        "extraArgs"
        "mode"
        "scenario"
        "trustRemoteCode"
      ]
    );

  mkBenchmark =
    modelName: model:
    {
      mode,
      scenario,
      ...
    }@case:
    let
      name = mkCaseName case;
      params = caseParams case;
    in
    benchLib.mkBenchmark {
      inherit
        pkgs
        name
        package
        ;
      command = [ (mkRunner model case) ];
      env = {
        HF_HOME = resolvedHfCacheHome;
        HF_HUB_OFFLINE = "1";
        TRANSFORMERS_OFFLINE = "1";
        VLLM_NO_USAGE_STATS = "1";
        VLLM_DO_NOT_TRACK = "1";
        RAY_USAGE_STATS_ENABLED = "0";
      }
      // lib.optionalAttrs ((target.hsaOverride or null) != null) {
        HSA_OVERRIDE_GFX_VERSION = target.hsaOverride;
      };
      requirements = {
        systemFeatures = [ target.systemFeature ];
        hostProfiles = [ hostProfile ];
        sandboxPaths = [
          "/dev/dri"
          "/dev/kfd"
          "/sys/bus/pci/devices"
          "/sys/class/drm"
          "/sys/class/hwmon"
          "/sys/class/kfd"
          "/sys/class/net"
          "/sys/class/scsi_host"
          resolvedModelRoot
        ];
      };
      metadata = lib.recursiveUpdate {
        kind = "vllm";
        suite = "vllm";
        accelerator = "rocm";
        backend = "vllm";
        inherit
          mode
          params
          scenario
          ;
        target = {
          inherit (target)
            packageSuffix
            runtimeArch
            systemFeature
            ;
        };
        model = {
          name = modelName;
          cacheRoot = resolvedHfCacheHome;
          inherit (model)
            description
            id
            ;
        };
        tool = {
          backend = "rocm";
          executable = "vllm bench ${mode}";
          packageRole = "vllm-rocm-therock";
        };
      } matrixMetadata;
      meta.platforms = [ "x86_64-linux" ];
      description = "Run vLLM ${mode} benchmark ${modelName}/${scenario}";
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
    ;
}
