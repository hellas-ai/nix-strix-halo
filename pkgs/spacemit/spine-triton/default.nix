{
  lib,
  stdenvNoCC,
  src,
  version,
  spineMlir,
}:

stdenvNoCC.mkDerivation {
  pname = "spine-triton-source";
  inherit src version;

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/share/spine-triton"
    cp -R . "$out/share/spine-triton/source"

    runHook postInstall
  '';

  passthru = {
    inherit spineMlir;
  };

  meta = {
    description = "Pinned SpacemiT Spine-Triton source tree";
    homepage = "https://github.com/spacemit-com/spine-triton";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = lib.platforms.linux;
  };
}
