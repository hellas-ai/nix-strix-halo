{
  lib,
  stdenvNoCC,
  stdenv,
  fetchurl,
  file,
  libffi,
  libxml2,
  ncurses,
  patchelf,
  zlib,
  zstd,
}:

stdenvNoCC.mkDerivation {
  pname = "spacemit-k3-toolchain";
  version = "1.2.4";

  src = fetchurl {
    url = "https://github.com/spacemit-com/toolchain/releases/download/v1.2.4/spacemit-toolchain-linux-glibc-x86_64-v1.2.4.tar.xz";
    hash = "sha256-+TtKFt6MfNahR83c+0kGeOiN81Vw4XizFkE4vij6BrY=";
  };

  nativeBuildInputs = [
    file
    patchelf
  ];

  dontConfigure = true;
  dontBuild = true;
  dontPatchELF = true;
  dontMoveLib64 = true;
  dontPatchShebangs = true;
  dontCheckForBrokenSymlinks = true;
  dontStrip = true;

  installPhase = ''
    runHook preInstall

    mkdir -p "$out"
    cp -R ./* "$out/"
    chmod -R u+w "$out"

    for link in "$out"/bin/gp-*; do
      [ -L "$link" ] || continue
      target=$(readlink "$link")
      prefixed="$out/bin/riscv64-unknown-linux-gnu-$target"
      if [ ! -e "$out/bin/$target" ] && [ -e "$prefixed" ]; then
        ln -sfn "riscv64-unknown-linux-gnu-$target" "$link"
      fi
    done

    host_rpath="$out/lib:$out/lib64:${
      lib.makeLibraryPath [
        stdenv.cc.cc.lib
        libffi
        libxml2
        ncurses
        zlib
        zstd
      ]
    }"
    host_interpreter=${lib.escapeShellArg stdenv.cc.bintools.dynamicLinker}

    patch_host_elf() {
      local elf="$1"

      if ! file "$elf" | grep -Fq "x86-64"; then
        return
      fi

      if old_rpath=$(patchelf --print-rpath "$elf" 2>/dev/null); then
        patchelf --set-rpath "''${old_rpath:+$old_rpath:}$host_rpath" "$elf" 2>/dev/null || true
      fi

      if patchelf --print-interpreter "$elf" >/dev/null 2>&1; then
        patchelf --set-interpreter "$host_interpreter" "$elf" 2>/dev/null || true
      fi
    }

    for dir in "$out/bin" "$out/lib" "$out/lib64" "$out/libexec" "$out/riscv64-unknown-linux-gnu/bin"; do
      [ -d "$dir" ] || continue
      while IFS= read -r -d "" elf; do
        patch_host_elf "$elf"
      done < <(find "$dir" -type f \( -perm -0100 -o -name "*.so" -o -name "*.so.*" \) -print0)
    done

    runHook postInstall
  '';

  meta = {
    description = "SpacemiT K3 RISC-V cross toolchain with IME instruction support";
    homepage = "https://github.com/spacemit-com/toolchain";
    license = lib.licenses.gpl3Plus;
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = [ "x86_64-linux" ];
  };
}
