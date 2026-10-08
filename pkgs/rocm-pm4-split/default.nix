# Retained-PM4 graph replay for the TheRock 10 SDK: HIP runtime (CLR) and HSA runtime (ROCR).
#
# EXPERIMENTAL. Nothing in the serving closure references this package; it only builds when asked
# for (`nix build .#rocm-pm4-clr .#rocm-pm4-rocr-tmpring`) and is loaded into a worker through
# the sitecustomize in ./bootstrap (DS41_PM4_CLR / DS41_PM4_ROCR / DS41_PM4_SITE).
#
# Layers, in build order:
#
#   prepared      Codex's PM4 SDK port: five pinned rocm-systems feature deltas applied onto the
#                 exact TheRock 10 source (rocm-systems 6b0e43f) by ./backport/prepare_port.py.
#   rocr          HSA runtime from `prepared`. CLR links against this one.
#   clr           HIP runtime from `prepared` (kpack enabled). Reproduces Codex's
#                 clr-sdk10-pm4-7.15.26333-pm4-7dda3ac-kpack.
#   rocrTmpring   rocr + ./tmpring/scratchless-preserve-tmpring.patch (scratchless retained dispatches
#                 must not replace queue-owned TMPRING state) and a CPU compile test of the patched
#                 encoder. Reproduces Codex's rocr-sdk10-pm4-scratchless-tmpring-diagnostic-hypothesis-1.
#
# Equivalence with Codex's derivations (clr-sdk10-pm4-7.15.26333-pm4-7dda3ac-kpack, out 9g40fnkc;
# rocr-sdk10-pm4-scratchless-tmpring-diagnostic-hypothesis-1, out k69q8hw3): the prepared source is
# byte-for-byte the same store path (rocm-10-sdk-pm4-prepared-source splxq772), and so are the SDK
# path, flags and phases. `clr`'s libamdhip64.so.7 (.text, .data, .data.rel.ro) and `rocrTmpring`'s
# libhsa-runtime64.so.1 (.text, .data) are bit-identical to theirs. The store paths differ because
# Codex passed the SDK as builtins.storePath and this package builds it as a derivation.
{
  lib,
  stdenv,
  runCommand,
  symlinkJoin,
  writeShellScript,
  cmake,
  pkg-config,
  python3,
  python3Packages,
  perl,
  xxd,
  git,
  patchelf,
  elfutils,
  libdrm,
  numactl,
  zlib,
  libffi,
  zstd,
  libGL,
  libxml2,
  libx11,
  khronos-ocl-icd-loader,
  # TheRock 10 SDK (therock-rocm-<target>) and the rocm-systems checkout it was built from.
  rocmSdk,
  baseSource,
}:

let
  # The SDK the PM4 runtimes were built and qualified against: therock-rocm plus the hipcc
  # wrapper that pkgs/sglang/default.nix (rocmSdkForJit) installs. Kept identical so the outputs
  # below match Codex's store paths.
  sdk = symlinkJoin {
    name = "${rocmSdk.name or "rocm-sdk"}-sglang-jit";
    paths = [ rocmSdk ];
    postBuild = ''
      rm -f "$out/bin/hipcc"
      printf '%s\n' \
        '#!${stdenv.shell}' \
        'needs_xhip=0' \
        'for arg in "$@"; do' \
        '  case "$arg" in' \
        '    *.cu|*.hip|*.cpp|*.cc|*.cxx) needs_xhip=1 ;;' \
        '  esac' \
        'done' \
        'if [ "$needs_xhip" = 1 ]; then' \
        '  exec ${rocmSdk}/bin/therock-hip-clang++ -x hip "$@"' \
        'fi' \
        'exec ${rocmSdk}/bin/therock-hip-clang++ "$@"' \
        > "$out/bin/hipcc"
      chmod 755 "$out/bin/hipcc"
    '';
  };

  # Codex's feature deltas and the script that applies them. Only these files enter the bundle.
  bundle = builtins.path {
    path = ./backport;
    name = "pm4-sdk-backport-bundle";
    filter =
      path: type:
      if type == "directory" then
        builtins.baseNameOf path != "port-check" && builtins.baseNameOf path != "__pycache__"
      else
        lib.hasSuffix ".patch" path
        || builtins.elem (builtins.baseNameOf path) [
          "prepare_port.py"
          "configure_sdk_build.py"
        ];
  };

  prepared =
    runCommand "rocm-10-sdk-pm4-prepared-source"
      {
        nativeBuildInputs = [
          python3
          git
        ];
      }
      ''
        python ${bundle}/prepare_port.py --base ${baseSource} --destination "$out"
      '';

  # Single files keep their own store paths, as Codex's derivations reference them.
  file = name: path: builtins.path { inherit name path; };
  tmpringPatch = file "scratchless-preserve-tmpring.patch" ./tmpring/scratchless-preserve-tmpring.patch;
  prepareTest = file "prepare_test.py" ./tmpring/prepare_test.py;
  testEncoder = file "test_encoder.cpp" ./tmpring/test_encoder.cpp;

  rocr = stdenv.mkDerivation {
    pname = "rocr-sdk10-pm4";
    version = "6b0e43f-pm4-7dda3ac";
    src = prepared;
    sourceRoot = "rocm-10-sdk-pm4-prepared-source/projects/rocr-runtime";
    nativeBuildInputs = [
      cmake
      pkg-config
      python3
      xxd
      sdk
    ];
    buildInputs = [
      elfutils
      libdrm
      numactl
      zlib
      sdk
    ];
    cmakeBuildType = "RelWithDebInfo";
    cmakeFlags = [
      "-DBUILD_SHARED_LIBS=ON"
      "-DBUILD_ROCRTST=OFF"
      "-DENABLE_LDCONFIG=OFF"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      "-DCMAKE_INSTALL_INCLUDEDIR=include"
      "-DCMAKE_PREFIX_PATH=${sdk}"
    ];
    postPatch = ''
      python ${bundle}/configure_sdk_build.py . ${sdk}
      patchShebangs --build runtime
      substituteInPlace runtime/hsa-runtime/image/blit_src/CMakeLists.txt --replace-fail 'COMMAND clang' 'COMMAND ${sdk}/lib/llvm/bin/clang'
      export HIP_DEVICE_LIB_PATH=${sdk}/amdgcn/bitcode
    '';
    postInstall = ''
      mkdir -p $out/share/pm4-provenance
      cp ${prepared}/port-manifest.json $out/share/pm4-provenance/
      cp ${prepared}/projects/rocr-runtime/LICENSE* $out/share/pm4-provenance/ || true
      cp ${prepared}/projects/rocr-runtime/runtime/hsa-runtime/core/inc/amd_graph_command_encoder.h $out/share/pm4-provenance/
    '';
    meta.license = [
      lib.licenses.ncsa
      lib.licenses.asl20
    ];
  };

  llvmMc = runCommand "exact-sdk-llvm-mc-nix-loader" { nativeBuildInputs = [ patchelf ]; } ''
    mkdir -p $out/bin
    cp ${sdk}/lib/llvm/bin/llvm-mc $out/bin/llvm-mc
    chmod +w $out/bin/llvm-mc
    patchelf --set-interpreter ${stdenv.cc.bintools.dynamicLinker} \
      --add-rpath ${sdk}/lib/llvm/lib:${sdk}/lib:${lib.makeLibraryPath [ stdenv.cc.cc.lib ]} \
      $out/bin/llvm-mc
  '';

  pchClang = writeShellScript "pm4-sdk-pch-clang" ''
    if [ "$1" = -cc1 ]; then exec ${sdk}/lib/llvm/bin/clang "$@"; fi
    exec ${sdk}/bin/therock-hip-clang++ "$@"
  '';

  clr = stdenv.mkDerivation {
    pname = "clr-sdk10-pm4";
    version = "7.15.26333-pm4-7dda3ac-kpack";
    src = prepared;
    sourceRoot = "rocm-10-sdk-pm4-prepared-source/projects/clr";
    nativeBuildInputs = [
      cmake
      pkg-config
      python3
      python3Packages.cppheaderparser
      perl
      sdk
    ];
    buildInputs = [
      rocr
      sdk
      numactl
      libffi
      zstd
      zlib
      libGL
      libxml2
      libx11
      khronos-ocl-icd-loader
    ];
    cmakeBuildType = "RelWithDebInfo";
    cmakeFlags = [
      "-DCMAKE_POLICY_DEFAULT_CMP0072=NEW"
      "-DROCM_KPACK_ENABLED=ON"
      "-Drocm-kpack_DIR=${sdk}/lib/cmake/rocm-kpack"
      "-DCLR_BUILD_HIP=ON"
      "-DCLR_BUILD_OCL=OFF"
      "-DHIP_PLATFORM=amd"
      "-DHIP_COMMON_DIR=${prepared}/projects/hip"
      "-DHIPCC_BIN_DIR=${sdk}/bin"
      "-DROCM_PATH=${sdk}"
      "-DLLVM_DIR=${sdk}/lib/llvm"
      "-Dllvm-mc=${llvmMc}/bin/llvm-mc"
      "-Dhsa-runtime64_DIR=${rocr}/lib/cmake/hsa-runtime64"
      "-DCMAKE_PREFIX_PATH=${rocr};${sdk}"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      "-DCMAKE_INSTALL_INCLUDEDIR=include"
    ];
    postPatch = ''
      patchShebangs hipamd
      # Playback's HIP transitive dependency otherwise resolves to stock HSA
      # through the SDK toolchain rpath-link directories during executable link.
      substituteInPlace hipamd/src/hrr/playback/CMakeLists.txt \
        --replace-fail 'PRIVATE hrr_reader amdhip64 Threads::Threads' 'PRIVATE hrr_reader amdhip64 ${rocr}/lib/libhsa-runtime64.so Threads::Threads'
      # Header parser cannot parse glibc fortified inline bodies; this affects
      # generation only, not compilation hardening of the runtime.
      substituteInPlace hipamd/src/CMakeLists.txt \
        --replace-fail 'COMMAND ''${CMAKE_C_COMPILER}' 'COMMAND ''${CMAKE_COMMAND} -E env NIX_HARDENING_ENABLE= ''${CMAKE_C_COMPILER}'
      substituteInPlace hipamd/src/hip_embed_pch.sh \
        --replace-fail '$LLVM_DIR/bin/clang' '${pchClang}' \
        --replace-fail '$LLVM_DIR/bin/llvm-mc' '${llvmMc}/bin/llvm-mc'
      substituteInPlace hipamd/CMakeLists.txt \
        --replace-fail 'install(PROGRAMS ''${HIPCC_BIN_DIR}/hipcc.bat DESTINATION bin)' "" \
        --replace-fail 'install(PROGRAMS ''${HIPCC_BIN_DIR}/hipconfig.bat DESTINATION bin)' ""
    '';
    postInstall = ''
      mkdir -p $out/share/pm4-provenance
      cp ${prepared}/port-manifest.json $out/share/pm4-provenance/
      cp ${prepared}/projects/clr/LICENSE* $out/share/pm4-provenance/ || true
    '';
    postFixup = ''
      for f in $out/lib/*.so*; do
        if [ -f "$f" ] && [ ! -L "$f" ]; then patchelf --add-rpath "$out/lib:${rocr}/lib:${sdk}/lib" "$f"; fi
      done
    '';
    meta.license = lib.licenses.mit;
  };

  rocrTmpring = rocr.overrideAttrs (old: {
    pname = "rocr-sdk10-pm4-scratchless-tmpring-diagnostic";
    version = "hypothesis-1";
    __intentionallyOverridingVersion = true;
    patches = [ tmpringPatch ];
    postPatch = old.postPatch + ''
      python ${prepareTest} runtime/hsa-runtime/core/runtime/hsa_ven_amd_graph.cpp
      $CXX -std=c++17 -O2 -I. -Iruntime/hsa-runtime ${testEncoder} -o cpu-encoder-test
      ./cpu-encoder-test
    '';
    postInstall = ''
      mkdir -p $out/share/pm4-provenance
      cp ${prepared}/port-manifest.json $out/share/pm4-provenance/
      cp ${tmpringPatch} $out/share/pm4-provenance/
      cp ../runtime/hsa-runtime/core/inc/amd_graph_command_encoder.h $out/share/pm4-provenance/
      cp ../runtime/hsa-runtime/core/runtime/hsa_ven_amd_graph.cpp $out/share/pm4-provenance/
      cp ../cpu-encoder-test $out/share/pm4-provenance/
    '';
  });

  # sitecustomize that preloads the PM4 HSA and HIP runtimes into rocm_sdk before torch imports.
  # Put $out on PYTHONPATH and set DS41_PM4_CLR, DS41_PM4_ROCR and DS41_PM4_SITE (see the file).
  bootstrap = runCommand "pm4-bootstrap" { } ''
    install -Dm444 ${./bootstrap/sitecustomize.py} $out/sitecustomize.py
  '';
in
{
  inherit
    sdk
    prepared
    rocr
    clr
    rocrTmpring
    bootstrap
    ;
}
