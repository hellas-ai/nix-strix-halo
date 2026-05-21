# FastFlowLM NPU bench derivations.
#
# Each entry is a `runCommand` derivation that runs `flm run <model>`
# under /verbose mode and captures the perf summary (TTFT, prefill
# tok/s, decode tok/s, total tokens) along with the raw transcript.
#
# Bench hosts must:
#   1. Provide the `npu-strix` system feature (see
#      modules/benchmark-runner.nix).
#   2. Pre-pull each model with `FLM_MODEL_PATH=/models/flm flm pull <tag>`
#      so the sandbox can read it without network access.
{
  pkgs,
  fastflowlm,
  modelsPath ? "/models/flm",
  npuTarget ? "npu-strix",
}:

let
  inherit (pkgs) lib;

  # `flm` accepts model tags like `llama3.2:1b`; sanitize for derivation
  # names (no `:` or `.`).
  sanitize = lib.replaceStrings [ ":" "." "/" ] [ "-" "-" "-" ];

  prompts = {
    short = "Explain in one sentence what an NPU is.";

    medium = ''
      In about 200 words, contrast a tile-structured NPU (e.g. AMD XDNA2)
      with a SIMD-based CPU and a SIMT GPU for transformer inference.
      Cover dataflow vs lockstep execution, on-chip memory hierarchy, and
      where each architecture wins for the attention path of an LLM.
    '';

    long = ''
      Write a ~400 word technical explanation of how a tile-structured
      kernel architecture on the AMD XDNA2 NPU enables efficient
      transformer inference. Cover: (1) why dataflow execution on AI
      Engine cores beats SIMD on a CPU for matmul-heavy attention,
      (2) how the on-chip memory tile hierarchy reduces DRAM bandwidth
      pressure compared to a GPU, (3) what role unified memory plays
      on Strix Halo specifically, and (4) where the NPU is fundamentally
      limited versus a discrete GPU like an MI300. Technical but readable.
    '';
  };

  mkBench =
    {
      model,
      promptKey ? "medium",
      ctxLen ? null,
      prefillChunkLen ? null,
      pmode ? null,
    }:
    let
      flmArgs = lib.concatStringsSep " " (
        lib.optional (ctxLen != null) "-c ${toString ctxLen}"
        ++ lib.optional (prefillChunkLen != null) "--prefill-chunk-len ${toString prefillChunkLen}"
        ++ lib.optional (pmode != null) "--pmode ${pmode}"
      );
      name =
        "bench-flm-${sanitize model}-${promptKey}"
        + lib.optionalString (ctxLen != null) "-ctx${toString ctxLen}"
        + lib.optionalString (prefillChunkLen != null) "-chunk${toString prefillChunkLen}"
        + lib.optionalString (pmode != null) "-${pmode}";
    in
    pkgs.runCommand name {
      buildInputs = [ fastflowlm ];
      requiredSystemFeatures = [ npuTarget ];
      promptText = prompts.${promptKey};
      meta = {
        description = "FastFlowLM NPU bench: ${model} (${promptKey} prompt)";
        platforms = [ "x86_64-linux" ];
      };
    } ''
      set -euo pipefail

      # flm reads/writes the model cache under $HOME/.config/flm by
      # default; redirect to the pre-staged /models/flm tree via
      # FLM_MODEL_PATH so the build doesn't need network or write access.
      export HOME="$TMPDIR/home"
      export FLM_MODEL_PATH="${modelsPath}"
      export FLM_DISABLE_UPDATE_CHECK=1
      mkdir -p "$HOME" "$out"

      printf '%s\n' "Model:            ${model}"           > "$out/config.txt"
      printf '%s\n' "Prompt:           ${promptKey}"      >> "$out/config.txt"
      printf '%s\n' "ctx-len:          ${toString ctxLen}"           >> "$out/config.txt"
      printf '%s\n' "prefill-chunk:    ${toString prefillChunkLen}" >> "$out/config.txt"
      printf '%s\n' "pmode:            ${toString pmode}"           >> "$out/config.txt"
      printf '%s\n' "FLM_MODEL_PATH:   $FLM_MODEL_PATH"             >> "$out/config.txt"

      # Two-line stdin: enable verbose metrics, then the prompt. flm
      # exits once the input stream ends.
      {
        echo "/verbose"
        printf '%s\n' "$promptText"
      } | ${fastflowlm}/bin/flm run ${flmArgs} ${model} \
        > "$out/transcript.txt" 2>&1

      # Pull the four numeric perf lines out of /verbose's footer into
      # a stable CSV.
      {
        echo "metric,value"
        grep -E '^\s*(Total tokens|TTFT|Prefill speed|Decoding speed):' \
          "$out/transcript.txt" \
          | sed -E 's/^[[:space:]]*//; s/:[[:space:]]+/,/'
      } > "$out/summary.csv"

      # Surface the summary on stdout for `nix log`.
      cat "$out/summary.csv"
    '';

  # Default sweep: each model gets a baseline run plus a deeper-context
  # variant. Add entries here to grow the matrix.
  cases = [
    { model = "llama3.2:1b"; promptKey = "medium"; }
    { model = "llama3.2:1b"; promptKey = "long"; ctxLen = 8192; prefillChunkLen = 512; }
    { model = "gemma4-it:e4b"; promptKey = "medium"; }
    { model = "gemma4-it:e4b"; promptKey = "long"; ctxLen = 8192; prefillChunkLen = 512; }
    { model = "gpt-oss:20b"; promptKey = "long"; }
    { model = "gpt-oss:20b"; promptKey = "long"; ctxLen = 8192; prefillChunkLen = 512; }
  ];

  benchmarks = lib.listToAttrs (
    map (cfg: rec {
      value = mkBench cfg;
      name = value.name;
    }) cases
  );
in
{
  inherit
    mkBench
    prompts
    benchmarks
    ;
}
