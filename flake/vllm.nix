{
  inputs,
  lib,
  nixpkgs-vllm,
}:

let
  versionFromReleaseRef = ref: lib.removePrefix "v" (lib.last (lib.splitString "/" ref));
in
import ../lib/vllm.nix {
  inherit lib nixpkgs-vllm;

  # GitHub flake locks retain the resolved commit for branch inputs, but not
  # always the original branch name. Keep this fallback aligned with
  # inputs.vllm-src.url.
  vllmVersion = versionFromReleaseRef (inputs.vllm-src.sourceInfo.ref or "releases/v0.20.2");

  vllmSources = {
    vllm = inputs.vllm-src;
    cutlass = inputs.vllm-cutlass-src;
    flash-attn = inputs.vllm-flash-attn-src;
    flashmla = inputs.vllm-flashmla-src;
    flashmla-cutlass = inputs.vllm-flashmla-cutlass-src;
  };
}
