{ inputs }:

final: prev: {
  fastflowlm = prev.callPackage ../pkgs/fastflowlm {
    inherit (final) tokenizers-cpp xrt;
    src = inputs.fastflowlm;
  };
}
