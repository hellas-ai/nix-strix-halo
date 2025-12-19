{ pkgs }:

pkgs.stdenv.mkDerivation {
  pname = "ryzenadj-co-test";
  version = "0.1.0";

  src = ./ryzenadj-co-test.sh;
  dontUnpack = true;

  nativeBuildInputs = [ pkgs.makeWrapper ];

  installPhase = ''
    mkdir -p $out/bin
    cp $src $out/bin/ryzenadj-co-test
    chmod +x $out/bin/ryzenadj-co-test
    wrapProgram $out/bin/ryzenadj-co-test \
      --prefix PATH : ${pkgs.lib.makeBinPath [
        pkgs.ryzenadj
        pkgs.gawk
        pkgs.gnugrep
        pkgs.coreutils
      ]}
  '';

  meta = {
    description = "Test script to find optimal Curve Optimizer value for AMD Ryzen CPUs";
    mainProgram = "ryzenadj-co-test";
  };
}
