# Simple benchmark runner
{
  pkgs,
  llamaCppPackage,
  modelPath ? null,
  mmap ? null,
  fa ? null,
  ngl ? null,
  threads ? null,
  batch ? null,
  ubatch ? null,
  rpc ? null,
  vkIcdFilenames ? null,
  extraArgs ? "",
}:
pkgs.runCommand "benchmark-${llamaCppPackage.pname}" {
  buildInputs = [llamaCppPackage];
  requiredSystemFeatures = ["rocm"];
  __noChroot = true;
} ''
  export HSA_OVERRIDE_GFX_VERSION=11.5.1
  ${pkgs.lib.optionalString (vkIcdFilenames != null) "export VK_ICD_FILENAMES=${vkIcdFilenames}"}
  ${llamaCppPackage}/bin/llama-bench \
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
