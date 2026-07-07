{
  lib,
  stdenvNoCC,
  cmake,
  gcc,
  ninja,
  patchelf,
  src,
  version,
  commit,
  spacemitK3Toolchain,
  spacemitK3Runtime,
}:

let
  runtimeRpath = lib.concatStringsSep ":" [
    "$out/lib"
    "${spacemitK3Runtime}/sysroot/lib"
    "${spacemitK3Runtime}/riscv64-unknown-linux-gnu/lib"
  ];
in
stdenvNoCC.mkDerivation {
  pname = "llama-cpp-spacemit";
  inherit src version;

  nativeBuildInputs = [
    cmake
    ninja
    patchelf
  ];

  dontPatchELF = true;
  dontStrip = true;

  env.RISCV_ROOT_PATH = spacemitK3Toolchain;

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_AR" "${spacemitK3Toolchain}/bin/riscv64-unknown-linux-gnu-ar")
    (lib.cmakeFeature "CMAKE_RANLIB" "${spacemitK3Toolchain}/bin/riscv64-unknown-linux-gnu-ranlib")
    (lib.cmakeFeature "CMAKE_STRIP" "${spacemitK3Toolchain}/bin/riscv64-unknown-linux-gnu-strip")
    (lib.cmakeFeature "CMAKE_TOOLCHAIN_FILE" "${src}/cmake/riscv64-spacemit-linux-gnu-gcc.cmake")
    (lib.cmakeFeature "CMAKE_INSTALL_PREFIX" "${placeholder "out"}")
    (lib.cmakeFeature "HOST_CXX_COMPILER" "${gcc}/bin/g++")
    (lib.cmakeFeature "LLAMA_BUILD_COMMIT" commit)
    (lib.cmakeFeature "LLAMA_BUILD_NUMBER" "0")
    (lib.cmakeBool "BUILD_SHARED_LIBS" true)
    (lib.cmakeBool "GGML_BLAS" false)
    (lib.cmakeBool "GGML_CLBLAST" false)
    (lib.cmakeBool "GGML_CPU_REPACK" false)
    (lib.cmakeBool "GGML_CPU_RISCV64_SPACEMIT" true)
    (lib.cmakeBool "GGML_CUDA" false)
    (lib.cmakeBool "GGML_HIP" false)
    (lib.cmakeBool "GGML_METAL" false)
    (lib.cmakeBool "GGML_OPENMP" false)
    (lib.cmakeBool "GGML_RPC" true)
    (lib.cmakeBool "GGML_RVV" true)
    (lib.cmakeBool "GGML_RV_ZBA" true)
    (lib.cmakeBool "GGML_RV_ZFH" true)
    (lib.cmakeBool "GGML_RV_ZICBOP" true)
    (lib.cmakeBool "GGML_RV_ZIHINTPAUSE" true)
    (lib.cmakeBool "GGML_RV_ZVFH" true)
    (lib.cmakeBool "GGML_VULKAN" false)
    (lib.cmakeBool "LLAMA_BUILD_EXAMPLES" false)
    (lib.cmakeBool "LLAMA_BUILD_SERVER" true)
    (lib.cmakeBool "LLAMA_BUILD_TESTS" false)
    (lib.cmakeBool "LLAMA_BUILD_UI" false)
    (lib.cmakeBool "LLAMA_CURL" false)
    (lib.cmakeBool "LLAMA_OPENSSL" false)
  ];

  postInstall = ''
    ln -sf "$out/bin/llama-cli" "$out/bin/llama"

    if [ -e "$out/bin/rpc-server" ] && [ ! -e "$out/bin/llama-rpc-server" ]; then
      ln -s rpc-server "$out/bin/llama-rpc-server"
    fi

    mkdir -p "$out/include"
    cp "$src/include/llama.h" "$out/include/"
  '';

  postFixup = ''
    while IFS= read -r -d "" elf; do
      if old_rpath=$(patchelf --print-rpath "$elf" 2>/dev/null); then
        patchelf --set-rpath "${runtimeRpath}''${old_rpath:+:$old_rpath}" "$elf" 2>/dev/null || true
      fi
      if patchelf --print-interpreter "$elf" >/dev/null 2>&1; then
        patchelf --set-interpreter "${spacemitK3Runtime}/sysroot/lib/ld-linux-riscv64-lp64d.so.1" "$elf" 2>/dev/null || true
      fi
    done < <(find "$out" -type f \( -perm -0100 -o -name "*.so" -o -name "*.so.*" \) -print0)
  '';

  passthru = {
    targetSystem = "riscv64-linux";
    inherit spacemitK3Runtime spacemitK3Toolchain;
  };

  meta = {
    description = "llama.cpp from SpacemiT's fork, built for K3/A100 IME2";
    homepage = "https://github.com/spacemit-com/llama.cpp";
    license = lib.licenses.mit;
    mainProgram = "llama";
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = [ "x86_64-linux" ];
  };
}
