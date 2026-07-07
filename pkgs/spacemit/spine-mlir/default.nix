{
  lib,
  stdenvNoCC,
  fetchurl,
  patchelf,
  spacemitK3Toolchain,
}:

let
  version = "0.5.4";
  runtimeRpath = lib.concatStringsSep ":" [
    "${spacemitK3Toolchain}/sysroot/lib"
    "${spacemitK3Toolchain}/riscv64-unknown-linux-gnu/lib"
  ];
in
stdenvNoCC.mkDerivation {
  pname = "spacemit-k3-spine-mlir";
  inherit version;

  src = fetchurl {
    url = "https://github.com/spacemit-com/spine-mlir/releases/download/${version}/spine-mlir-riscv64-${version}.tar.gz";
    hash = "sha256-ATaPcRWWddlzy3hFQBp6+7LPv9ftoGY2m8D5DKP4a0w=";
  };

  nativeBuildInputs = [ patchelf ];

  dontConfigure = true;
  dontBuild = true;
  dontPatchELF = true;
  dontStrip = true;

  installPhase = ''
    runHook preInstall

    release_dir=ci/workspace/AI/spine-mlir-build/build/riscv64/speir/Release/spine-mlir-riscv64-${version}
    mkdir -p "$out"
    cp -R "$release_dir"/. "$out/"
    chmod -R u+w "$out"

    while IFS= read -r -d "" elf; do
      if old_rpath=$(patchelf --print-rpath "$elf" 2>/dev/null); then
        patchelf --set-rpath "${runtimeRpath}''${old_rpath:+:$old_rpath}" "$elf" 2>/dev/null || true
      fi
      if patchelf --print-interpreter "$elf" >/dev/null 2>&1; then
        patchelf --set-interpreter "${spacemitK3Toolchain}/sysroot/lib/ld-linux-riscv64-lp64d.so.1" "$elf" 2>/dev/null || true
      fi
    done < <(find "$out" -type f \( -perm -0100 -o -name "*.so" -o -name "*.so.*" \) -print0)

    runHook postInstall
  '';

  meta = {
    description = "SpacemiT Spine MLIR runtime tools for RISC-V";
    homepage = "https://github.com/spacemit-com/spine-mlir";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = [ "x86_64-linux" ];
  };
}
