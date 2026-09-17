{
  lib,
  stdenv,
  fetchFromGitHub,
}:

stdenv.mkDerivation {
  pname = "kerf-init";
  version = "0.2.0";

  src = fetchFromGitHub {
    owner = "multikernel";
    repo = "kerf";
    rev = "8b72b3e9b266f8d32e707e2c1743ad7afc50b1ec";
    hash = "sha256-feP1fO7A6ARdth05Eo6PltzWkAC3UKdzZ1vtM0bg7hY=";
  };

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin"
    $CC -Wall -Wextra -Werror -O2 -static \
      -o "$out/bin/kerf-init" src/init/init.c
    runHook postInstall
  '';

  meta = {
    description = "Minimal static PID 1 for multikernel spawn initramfs images";
    homepage = "https://github.com/multikernel/kerf";
    license = lib.licenses.asl20;
    mainProgram = "kerf-init";
    platforms = [ "x86_64-linux" ];
    maintainers = with lib.maintainers; [ georgewhewell ];
  };
}
