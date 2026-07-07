{
  lib,
  stdenvNoCC,
  spacemitK3Toolchain,
}:

stdenvNoCC.mkDerivation {
  pname = "spacemit-k3-runtime";
  inherit (spacemitK3Toolchain) version;

  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  dontFixup = true;

  installPhase = ''
    runHook preInstall

    copyRuntimeLibs() {
      local src="$1"
      local dst="$2"

      [ -d "$src" ] || return
      mkdir -p "$dst"

      find "$src" -maxdepth 1 \( -type f -o -type l \) \
        \( -name "*.so" -o -name "*.so.*" -o -name "ld-linux-*.so.*" \) \
        -exec cp -P --target-directory="$dst" {} +
    }

    copyRuntimeLibs ${spacemitK3Toolchain}/sysroot/lib "$out/sysroot/lib"
    copyRuntimeLibs ${spacemitK3Toolchain}/riscv64-unknown-linux-gnu/lib "$out/riscv64-unknown-linux-gnu/lib"

    runHook postInstall
  '';

  meta = {
    description = "Runtime libraries from the SpacemiT K3 RISC-V toolchain";
    homepage = "https://github.com/spacemit-com/toolchain";
    license = lib.licenses.gpl3Plus;
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = [ "x86_64-linux" ];
  };
}
