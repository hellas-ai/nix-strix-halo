# Simple benchmark runner
{
  pkgs,
  llamaPackage,
  modelPath ? null,
  mmap ? null,
  fa ? null,
  ngl ? null,
  threads ? null,
  batch ? null,
  ubatch ? null,
  rpc ? null,
  extraArgs ? "",
  gpuTarget ? "gfx1151",
  hsaOverride ? null,
}:
pkgs.runCommand "benchmark-${llamaPackage.pname}"
  {
    buildInputs = [ llamaPackage ];
    requiredSystemFeatures = [ gpuTarget ];
  }
  ''
    echo "Running benchmark with the following parameters:"
    echo "Model Path: ${modelPath}"

    ${pkgs.lib.optionalString (
      hsaOverride != null
    ) ''export HSA_OVERRIDE_GFX_VERSION="${hsaOverride}"''}
    ${llamaPackage}/bin/llama-bench \
      ${pkgs.lib.optionalString (modelPath != null) "-m ${modelPath}"} \
      ${pkgs.lib.optionalString (mmap != null) "--mmap ${toString mmap}"} \
      ${pkgs.lib.optionalString (fa != null) "-fa ${toString fa}"} \
      ${pkgs.lib.optionalString (ngl != null) "-ngl ${toString ngl}"} \
      ${pkgs.lib.optionalString (threads != null) "-t ${toString threads}"} \
      ${pkgs.lib.optionalString (batch != null) "-b ${toString batch}"} \
      ${pkgs.lib.optionalString (ubatch != null) "-ub ${toString ubatch}"} \
      ${pkgs.lib.optionalString (rpc != null) "--rpc ${rpc}"} \
      ${extraArgs} \
      > $out
  ''
