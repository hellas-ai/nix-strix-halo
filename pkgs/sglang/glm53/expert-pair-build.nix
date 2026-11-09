{ pkgs }:

# Explicit build entry point using the root tree's existing observer package.
# The returned launcher uses the same experimental runtime for flag0 and flag1.
let
  runtime = pkgs.callPackage ./expert-pair.nix {
    pythonPackages = pkgs.python313Packages;
  };
  serve = pkgs.glm53-serve.override {
    sglang-glm53-rocm = runtime;
  };
in
{
  inherit runtime serve;
}
