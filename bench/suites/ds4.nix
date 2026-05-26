{
  pkgs,
  package,
  modelRoot ? "/models/ds4",
  target,
  hostProfile ? "linux-amd-kfd",
  matrixMetadata ? { },
}:

let
  inherit (pkgs) lib;
  benchLib = import ../lib.nix { inherit lib; };

  clean = lib.replaceStrings [ ":" "." "/" "_" ] [ "-" "-" "-" "-" ];

  modelConfigs = {
    deepseek-v4-flash = {
      path = "${modelRoot}/ds4flash.gguf";
      repo = "antirez/deepseek-v4-gguf";
      description = "DeepSeek V4 Flash GGUF for DwarfStar 4";
      cases = [
        {
          scenario = "smoke";
          ctxStart = 128;
          ctxMax = 256;
          stepIncr = 128;
          genTokens = 8;
          promptRepeats = 1024;
          fastFull = false;
        }
        {
          scenario = "fast-full";
          ctxStart = 2048;
          ctxMax = 16384;
          stepIncr = 2048;
          genTokens = 32;
          promptRepeats = 32768;
          fastFull = true;
        }
      ];
    };
  };

  mkCaseName =
    {
      scenario,
      ...
    }:
    "ds4-rocm-${target.packageSuffix}-${clean scenario}";

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
    pkgs.writeShellScript "ds4-rocm-${target.packageSuffix}-benchmark-runner" ''
      set -euo pipefail

      prompt="$TMPDIR/ds4-prompt.txt"
      for i in $(seq 1 ${toString promptRepeats}); do
        printf 'DwarfStar 4 ROCm benchmark prompt paragraph %05d. This text is intentionally repetitive and deterministic so the fixed token sequence is long enough for context frontier measurement. It discusses GPU memory bandwidth, attention prefill, routed experts, and decode throughput on AMD Strix Halo. ' "$i"
      done > "$prompt"

      ${package}/bin/${executable} \
        -m ${lib.escapeShellArg model.path} \
        --cuda \
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
      };
      requirements = {
        systemFeatures = [ target.systemFeature ];
        hostProfiles = [ hostProfile ];
        sandboxPaths = [
          "/dev/dri"
          "/dev/kfd"
          "/sys/class/drm"
          "/sys/class/kfd"
          modelRoot
        ];
      };
      metadata = lib.recursiveUpdate {
        kind = "ds4";
        suite = "ds4";
        accelerator = "rocm";
        backend = "hip";
        inherit
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
          inherit (model)
            description
            path
            repo
            ;
        };
        tool = {
          backend = "rocm";
          packageRole = "ds4-rocm";
        };
      } matrixMetadata;
      meta.platforms = [ "x86_64-linux" ];
      description = "Run DS4 ROCm benchmark ${modelName}/${name}";
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
