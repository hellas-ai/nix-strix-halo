{ stdenv }:

stdenv.mkDerivation {
  pname = "amdgpu-smu-exporter";
  version = "0.1.0";

  dontUnpack = true;
  strictDeps = true;

  buildPhase = ''
    runHook preBuild
    $CC -std=c11 -O2 -Wall -Wextra -Wpedantic \
      ${./main.c} -o amdgpu-smu-exporter
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -D -m 0755 amdgpu-smu-exporter \
      "$out/bin/amdgpu-smu-exporter"
    runHook postInstall
  '';
}
