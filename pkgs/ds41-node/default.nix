{
  lib,
  stdenvNoCC,
  bash,
  binutils,
  coreutils,
  gawk,
  gnugrep,
  util-linux,
  makeWrapper,
  sglang-v41-rocm,
}:

stdenvNoCC.mkDerivation {
  name = "ds41-node";
  dontUnpack = true;
  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall
    install -Dm644 ${../../lib/bench/ds41-node.sh} "$out/libexec/ds41-node.sh"
    install -Dm644 ${../../lib/bench/ds41-model-check.py} "$out/libexec/ds41-model-check.py"
    makeWrapper ${bash}/bin/bash "$out/bin/ds41-node" \
      --add-flags "$out/libexec/ds41-node.sh" \
      --set-default DS41_BINARY ${lib.getExe sglang-v41-rocm} \
      --prefix PATH : ${
        lib.makeBinPath [
          bash
          binutils
          coreutils
          gawk
          gnugrep
          util-linux
        ]
      }
    runHook postInstall
  '';

  meta = {
    description = "Run a native DeepSeek V4.1 Flash TP4 rank on Strix Halo";
    mainProgram = "ds41-node";
    platforms = [ "x86_64-linux" ];
    maintainers = [ lib.maintainers.georgewhewell ];
  };
}
