{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  patchelf,
  boost,
  curl,
  cargo,
  ffmpeg,
  openxr-loader,
  libuuid,
  libdrm,
  fftw,
  fftwFloat,
  fftwLongDouble,
  tokenizers-cpp,
  xrt,
  src,
  # NPU firmware version baked into the binary's `flm validate` output.
  # FastFlowLM hardcodes this in src/CMakePresets.json; we mirror that
  # value here so the CMake build doesn't have to read JSON at eval time.
  npuVersion ? "32.0.203.304",
  version ? null,
}:

let
  presets = builtins.fromJSON (builtins.readFile "${src}/src/CMakePresets.json");
  upstreamVersion =
    (lib.findFirst (p: p.name or "" == "common-default") {
      cacheVariables.FLM_VERSION = "0";
    } presets.configurePresets).cacheVariables.FLM_VERSION;
  resolvedVersion =
    if version != null then
      version
    else if (src.shortRev or null) != null then
      "${upstreamVersion}-unstable-${src.shortRev}"
    else
      upstreamVersion;
in
stdenv.mkDerivation (finalAttrs: {
  pname = "fastflowlm";
  version = resolvedVersion;
  inherit src;

  sourceRoot = "${src.name or "source"}/src";

  postPatch = ''
        # Replace the bundled tokenizers-cpp subproject (which would try to
        # build vendored sentencepiece + msgpack) with our pre-built static
        # lib. We comment out the whole `add_subdirectory(... third_party/
        # tokenizers-cpp ... )` block — spans 3 lines in current upstream —
        # then prepend an IMPORTED target pointing at the prebuilt .a.
        sed -i '/add_subdirectory(''${CMAKE_SOURCE_DIR}\/\.\.\/third_party\/tokenizers-cpp/,/)$/{s/^/# /}' CMakeLists.txt
        sed -i '/^# add_subdirectory(''${CMAKE_SOURCE_DIR}\/\.\.\/third_party\/tokenizers-cpp/i\
    add_library(tokenizers_cpp STATIC IMPORTED)\
    set_target_properties(tokenizers_cpp PROPERTIES IMPORTED_LOCATION ''${TOKENIZERS_CPP_LIB_PATH} INTERFACE_INCLUDE_DIRECTORIES ''${TOKENIZERS_CPP_INCLUDE_PATH})' CMakeLists.txt

        # The upstream install rule symlinks the binary into /usr/local/bin
        # whenever the install prefix isn't there; that fails in the Nix
        # sandbox even though we point CMAKE_INSTALL_PREFIX at $out.
        sed -i 's/if(NOT WIN32 AND NOT CMAKE_INSTALL_PREFIX/if(FALSE AND NOT CMAKE_INSTALL_PREFIX/' CMakeLists.txt
  '';

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    patchelf
    boost
    curl
    cargo
  ];

  buildInputs = [
    tokenizers-cpp
    tokenizers-cpp.tokenizers-c
    xrt.xdna
    stdenv.cc.cc.lib
    ffmpeg
    openxr-loader
    libuuid
    libdrm
    fftw
    fftwFloat
    fftwLongDouble
  ];

  cmakeFlags = [
    (lib.cmakeFeature "TOKENIZERS_CPP_LIB_PATH" "${tokenizers-cpp}/lib/libtokenizers_cpp.a")
    (lib.cmakeFeature "TOKENIZERS_CPP_INCLUDE_PATH" "${tokenizers-cpp}/include")
    (lib.cmakeFeature "TOKENIZERS_C_LIB_PATH" "${tokenizers-cpp.tokenizers-c}/lib/libtokenizers_c.a")
    (lib.cmakeFeature "FLM_VERSION" finalAttrs.version)
    (lib.cmakeFeature "NPU_VERSION" npuVersion)
    (lib.cmakeFeature "XRT_INCLUDE_DIR" "${xrt.xdna}/include")
    (lib.cmakeFeature "XRT_LIB_DIR" "${xrt.xdna}/lib")
  ];

  env.NIX_LDFLAGS = "-L${tokenizers-cpp.tokenizers-c}/lib -ltokenizers_c";

  postFixup = ''
    # The bundled NPU runtime shared libs ($out/lib/flm/*.so) are
    # prebuilt blobs that bypass cc-wrapper, so cc-wrapper's RPATH
    # doesn't reach libgomp or the XRT plugin libs.
    for lib in $out/lib/flm/*.so; do
      patchelf --add-rpath ${stdenv.cc.cc.lib}/lib:${xrt.xdna}/lib "$lib" || true
    done

    # CMake bakes INSTALL_RPATH using "$ORIGIN/../''${CMAKE_INSTALL_LIBDIR}/flm",
    # and Nix's CMake sets CMAKE_INSTALL_LIBDIR to an absolute path,
    # producing a broken entry like "$ORIGIN/..//nix/store/.../lib/flm".
    # Strip the bogus entry (keeping cc-wrapper's ffmpeg/glibc/etc.
    # entries) and prepend the correct relative path plus XRT.
    current_rpath=$(patchelf --print-rpath $out/bin/flm)
    fixed_rpath=$(echo "$current_rpath" | tr ':' '\n' | grep -v '..//nix/store' | paste -sd:)
    patchelf --set-rpath "\$ORIGIN/../lib/flm:${xrt.xdna}/lib:$fixed_rpath" $out/bin/flm

    # flm resolves xclbins relative to its own bin/ directory.
    ln -sf ../share/flm/xclbins $out/bin/xclbins
  '';

  meta = {
    description = "High-performance LLM inference engine for AMD Ryzen AI NPUs";
    homepage = "https://github.com/FastFlowLM/FastFlowLM";
    license = lib.licenses.mit;
    maintainers = [ ];
    platforms = [ "x86_64-linux" ];
    mainProgram = "flm";
  };
})
