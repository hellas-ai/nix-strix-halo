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
  extraArgs ? [ ],
  env ? { },
  requirements ? { },
  requiredSystemFeatures ? [ ],
  name ? "benchmark-${llamaPackage.pname or llamaPackage.name}",
  metadata ? { },
  meta ? { },
}:
let
  benchLib = import ./lib.nix { inherit (pkgs) lib; };
  normalizedRequirements = benchLib.normalizeRequirements requirements;
in
benchLib.mkLlamaCppBenchmark {
  inherit
    pkgs
    name
    extraArgs
    env
    metadata
    meta
    ;
  requirements = normalizedRequirements // {
    systemFeatures = normalizedRequirements.systemFeatures ++ requiredSystemFeatures;
  };
  package = llamaPackage;
  model = modelPath;
  params = {
    inherit
      mmap
      fa
      ngl
      threads
      batch
      ubatch
      rpc
      ;
  };
}
