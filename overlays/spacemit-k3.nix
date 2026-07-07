{
  inputs,
  inputVersion,
}:

final: _prev:

let
  buildPkgs = final.pkgsBuildBuild;
  llamaVersion = inputVersion "0.1.5" inputs.llama-cpp-spacemit;
  llamaCommit = inputs.llama-cpp-spacemit.shortRev or inputs.llama-cpp-spacemit.rev or "unknown";
in
{
  spacemit-k3-toolchain = buildPkgs.callPackage ../pkgs/spacemit/toolchain { };

  spacemit-k3-runtime = buildPkgs.callPackage ../pkgs/spacemit/runtime {
    spacemitK3Toolchain = final.spacemit-k3-toolchain;
  };

  spacemit-k3-spine-mlir = buildPkgs.callPackage ../pkgs/spacemit/spine-mlir {
    spacemitK3Toolchain = final.spacemit-k3-toolchain;
  };

  spine-triton-source = buildPkgs.callPackage ../pkgs/spacemit/spine-triton {
    src = inputs.spine-triton;
    version = inputVersion "0.5.5" inputs.spine-triton;
    spineMlir = final.spacemit-k3-spine-mlir;
  };

  llama-cpp-spacemit = buildPkgs.callPackage ../pkgs/llama-cpp-spacemit {
    src = inputs.llama-cpp-spacemit;
    version = llamaVersion;
    commit = llamaCommit;
    spacemitK3Toolchain = final.spacemit-k3-toolchain;
    spacemitK3Runtime = final.spacemit-k3-runtime;
  };
}
