{
  pkgs,
  package,
  modelsRoot ? null,
  modelRoot ? null,
  modelPath ? null,
  target ? null,
  accelerator ? if pkgs.stdenv.isDarwin then "metal" else "rocm",
  hostProfile ? if accelerator == "metal" then "darwin-metal" else "linux-amd-kfd",
  matrixMetadata ? { },
  extraSystemFeatures ? [ ],
  extraSandboxPaths ? [ ],
  cases ? null,
}:

assert pkgs.lib.assertMsg (
  accelerator == "metal" || accelerator == "rocm"
) "unsupported DS4 benchmark accelerator: ${accelerator}";
assert pkgs.lib.assertMsg (
  accelerator == "metal" || target != null
) "DS4 ROCm benchmarks require a target record";

let
  inherit (pkgs) lib;
  benchLib = import ../lib.nix { inherit lib; };

  clean = lib.replaceStrings [ ":" "." "/" "_" ] [ "-" "-" "-" "-" ];

  resolvedModelsRoot = if modelsRoot != null then modelsRoot else benchLib.defaultModelsRoot pkgs;
  resolvedModelRoot =
    if modelRoot != null then modelRoot else benchLib.modelPath resolvedModelsRoot [ "ds4" ];
  resolvedModelPath =
    if modelPath != null then modelPath else benchLib.modelPath resolvedModelRoot [ "ds4flash.gguf" ];

  isMetal = accelerator == "metal";
  backend = if isMetal then "metal" else "hip";
  packageRole = if isMetal then "ds4" else "ds4-rocm";
  backendFlag = if isMetal then "--metal" else "--cuda";
  namePrefix = if isMetal then "ds4-metal" else "ds4-rocm-${target.packageSuffix}";
  promptBackend = if isMetal then "Metal" else "ROCm";
  metaPlatforms = if isMetal then [ "aarch64-darwin" ] else [ "x86_64-linux" ];
  systemFeatures =
    (
      if isMetal then
        [
          "metal"
          "benchmark"
        ]
      else
        [ target.systemFeature ]
    )
    ++ extraSystemFeatures;
  sandboxPaths =
    (
      if isMetal then
        [ resolvedModelRoot ]
      else
        [
          "/dev/dri"
          "/dev/kfd"
          "/sys/class/drm"
          "/sys/class/kfd"
          resolvedModelRoot
        ]
    )
    ++ extraSandboxPaths;
  targetMetadata =
    if isMetal then
      {
        packageSuffix = "metal";
        runtimeArch = "metal";
        systemFeature = "metal";
      }
    else
      {
        inherit (target)
          packageSuffix
          runtimeArch
          systemFeature
          ;
      };

  smokeCase = {
    scenario = "smoke";
    ctxStart = 128;
    ctxMax = 256;
    stepIncr = 128;
    genTokens = 8;
    promptRepeats = 1024;
    fastFull = false;
  };

  fastFullCase = {
    scenario = "fast-full";
    ctxStart = 2048;
    ctxMax = 16384;
    stepIncr = 2048;
    genTokens = 32;
    promptRepeats = 32768;
    fastFull = true;
  };

  modelConfigs = {
    deepseek-v4-flash = {
      path = resolvedModelPath;
      repo = "antirez/deepseek-v4-gguf";
      description = "DeepSeek V4 Flash GGUF for DwarfStar 4";
      cases = if cases == null then [ smokeCase ] ++ lib.optionals (!isMetal) [ fastFullCase ] else cases;
    };
  };

  mkCaseName =
    {
      scenario,
      ...
    }:
    "${namePrefix}-${clean scenario}";

  mkRunner =
    model:
    {
      ctxStart,
      ctxMax,
      stepIncr,
      genTokens,
      promptRepeats,
      fastFull ? false,
      ...
    }:
    let
      executable = if fastFull then "ds4-bench-fast-full" else "ds4-bench";
    in
    pkgs.writeShellScript "${namePrefix}-benchmark-runner" ''
      set -euo pipefail

      prompt="$TMPDIR/ds4-prompt.txt"
      for i in $(seq 1 ${toString promptRepeats}); do
        printf 'DwarfStar 4 ${promptBackend} benchmark prompt paragraph %05d. This text is intentionally repetitive and deterministic so the fixed token sequence is long enough for context frontier measurement. It discusses GPU memory bandwidth, attention prefill, routed experts, and decode throughput. ' "$i"
      done > "$prompt"

      ${package}/bin/${executable} \
        -m ${lib.escapeShellArg model.path} \
        ${backendFlag} \
        --prompt-file "$prompt" \
        --ctx-start ${toString ctxStart} \
        --ctx-max ${toString ctxMax} \
        --step-incr ${toString stepIncr} \
        --gen-tokens ${toString genTokens} \
        --csv "$out/results.csv"

      cat "$out/results.csv"
    '';

  mkBenchmark =
    modelName: model:
    {
      scenario,
      ctxStart,
      ctxMax,
      stepIncr,
      genTokens,
      fastFull ? false,
      ...
    }@case:
    let
      name = mkCaseName case;
      params = {
        inherit
          ctxStart
          ctxMax
          stepIncr
          genTokens
          fastFull
          ;
      };
    in
    benchLib.mkBenchmark {
      inherit
        pkgs
        name
        package
        ;
      command = [ (mkRunner model case) ];
      env = {
        DS4_MODEL = model.path;
        DS4_SERVER_PERFLEVEL = "skip";
        DS4_SERVER_FAST_FULL = if fastFull then "1" else null;
        DS4_METAL_NO_RESIDENCY = if isMetal then "1" else null;
        DS4_METAL_NO_MODEL_WARMUP = if isMetal then "1" else null;
      };
      requirements = {
        inherit
          systemFeatures
          sandboxPaths
          ;
        hostProfiles = [ hostProfile ];
      };
      metadata = lib.recursiveUpdate {
        kind = "ds4";
        suite = "ds4";
        inherit
          accelerator
          backend
          ;
        inherit
          params
          scenario
          ;
        target = targetMetadata;
        model = {
          name = modelName;
          inherit (model)
            description
            path
            repo
            ;
        };
        tool = {
          inherit
            backend
            packageRole
            ;
        };
      } matrixMetadata;
      meta.platforms = metaPlatforms;
      description = "Run DS4 ${promptBackend} benchmark ${modelName}/${name}";
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
