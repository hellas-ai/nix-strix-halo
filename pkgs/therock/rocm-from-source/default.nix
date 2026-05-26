{
  lib,
  stdenv,
  runCommand,
  fetchurl,
  cmake,
  ninja,
  pkg-config,
  gcc,
  gfortran,
  git,
  autoconf,
  automake,
  libtool,
  bison,
  flex,
  texinfo,
  file,
  xxd,
  patchelf,
  curl,
  perl,
  patch,
  gnumake,
  unzip,
  which,
  openssl,
  libGL,
  glog,
  fmt,
  llvmPackages,
  libffi,
  libxml2,
  zlib,
  ncurses,
  gmp,
  mpfr,
  zstd,
  xz,
  libva,
  ffmpeg,
  sqlite,
  expat,
  numactl,
  libdrm,
  libcap,
  pciutils,
  libpciaccess,
  elfutils,
  libx11,
  xorgproto,
  python3,
  therockSource,
  thirdPartySources ? { },
  spirvHeadersSource ? null,
  esmiIbLibrarySource ? null,
  ireeLibbacktraceSource ? null,
  ireeAmdDeviceLibsArchive ? null,
  rocprofilerOtf2Archive ? null,
  rocprofilerSysBinutilsArchive ? null,
  tracySource ? null,
  buildJobs ? null,
  buildTargets ? [ ],
  installMode ? "rocm-install",
  stageSubprojects ? [
    "amd-llvm"
    "amd-comgr"
    "hipcc"
  ],
  nameSuffix ? "",
  prebuiltStageTree ? null,
  target,
  amdgpuTargets ? [ target ],
  distAmdgpuTargets ? amdgpuTargets,
  testAmdgpuTargets ? amdgpuTargets,
  amdgpuFamilies ? [ ],
  distAmdgpuFamilies ? [ ],
  testAmdgpuFamilies ? [ ],
  distBundleName ? target,
  projectTargetUnexcludes ? { },
  version ? "unstable",
  profile ? "full",
}:

let
  toList = value: if builtins.isList value then value else [ value ];

  cmakeList = values: lib.concatStringsSep ";" (toList values);

  cmakeListFlag = name: values: lib.optional ((toList values) != [ ]) "-D${name}=${cmakeList values}";

  cmakeQuotedArgs = values: lib.concatStringsSep " " (map (value: "\"${value}\"") (toList values));

  cmakeIdentifier = value: lib.replaceStrings [ "-" "." "+" ":" ] [ "_" "_" "_" "_" ] value;

  projectTargetUnexcludeCmake = lib.concatStringsSep "\n" (
    lib.mapAttrsToList (
      project: targets:
      let
        var = "_therock_unexclude_${cmakeIdentifier project}";
      in
      ''
        get_property(${var} GLOBAL PROPERTY "THEROCK_AMDGPU_PROJECT_TARGET_EXCLUDES_${project}")
        if(${var})
          list(REMOVE_ITEM ${var} ${cmakeQuotedArgs targets})
          set_property(GLOBAL PROPERTY "THEROCK_AMDGPU_PROJECT_TARGET_EXCLUDES_${project}" "${"$"}{${var}}")
        endif()
      ''
    ) projectTargetUnexcludes
  );

  thirdPartyFetches = lib.mapAttrs (
    _name: source:
    fetchurl {
      inherit (source) url hash;
    }
  ) thirdPartySources;

  thirdPartyMirror = runCommand "therock-rocm-third-party-mirror" { } (
    ''
      mkdir -p "$out"
    ''
    + lib.concatStringsSep "\n" (
      lib.mapAttrsToList (
        _name: source: "ln -s ${thirdPartyFetches.${_name}} \"$out/${baseNameOf source.url}\""
      ) thirdPartySources
    )
  );

  stageSubprojectShellWords = lib.concatStringsSep " " stageSubprojects;

  nixSysroot = runCommand "therock-rocm-nix-sysroot" { } (
    ''
      mkdir -p "$out/include" "$out/lib" "$out/usr"

      for path in ${stdenv.cc.libc.dev}/include/*; do
        ln -s "$path" "$out/include/$(basename "$path")" 2>/dev/null || true
      done

      for includeRoot in ${libGL.dev}/include ${libx11.dev}/include ${xorgproto}/include; do
        [ -d "$includeRoot" ] || continue
        (cd "$includeRoot"
         find . -type d -exec mkdir -p "$out/include/{}" \;
         find . \( -type f -o -type l \) -exec sh -c \
           'for path do ln -s "$PWD/$path" "$1/$path" 2>/dev/null || true; done' \
           sh "$out/include" {} +)
      done

      mkdir -p "$out/include/c++"
      for cxxRoot in ${stdenv.cc.cc} ${gcc.cc}; do
        [ -d "$cxxRoot/include/c++" ] || continue
        for versionDir in "$cxxRoot"/include/c++/*; do
          [ -d "$versionDir" ] || continue
          version="$(basename "$versionDir")"
          mkdir -p "$out/include/c++/$version"
          for path in "$versionDir"/*; do
            ln -s "$path" "$out/include/c++/$version/$(basename "$path")" 2>/dev/null || true
          done

          for targetDir in "$versionDir"/*-linux-gnu; do
            [ -d "$targetDir" ] || continue
            for triple in x86_64-pc-linux-gnu x86_64-linux-gnu ${stdenv.hostPlatform.config}; do
              ln -s "$targetDir" "$out/include/c++/$version/$triple" 2>/dev/null || true
            done
          done
        done
      done

      ln -s ../include "$out/usr/include"

      for libdir in ${stdenv.cc.libc}/lib ${stdenv.cc.cc.lib}/lib ${gcc.cc.lib}/lib ${stdenv.cc.cc}/lib/gcc/*/* ${gcc.cc}/lib/gcc/*/* ${libGL}/lib ${libx11}/lib; do
        [ -d "$libdir" ] || continue
        for path in "$libdir"/*; do
          ln -s "$path" "$out/lib/$(basename "$path")" 2>/dev/null || true
        done
      done

      # GNU libc exposes libc.so as a linker script with absolute /nix/store
      # GROUP() members. lld resolves those absolute paths relative to
      # --sysroot, so mirror the direct runtime stores inside the sysroot too.
      mkdir -p "$out/nix/store"
      for storePath in \
        ${stdenv.cc.libc} \
        ${stdenv.cc.libc.dev} \
        ${stdenv.cc.cc} \
        ${stdenv.cc.cc.lib} \
        ${gcc.cc} \
        ${gcc.cc.lib}; do
        ln -s "$storePath" "$out/nix/store/$(basename "$storePath")" 2>/dev/null || true
      done

      # The freshly-built AMD clang is not Nix-wrapped. Its GCC detector expects
      # a conventional sysroot-relative GCC install layout for crtbegin/libgcc.
      for ccRoot in ${stdenv.cc.cc} ${gcc.cc}; do
        for gccLibDir in "$ccRoot"/lib/gcc/*/*; do
          [ -d "$gccLibDir" ] || continue
          gccVersion="$(basename "$gccLibDir")"
          for triple in x86_64-pc-linux-gnu x86_64-linux-gnu ${stdenv.hostPlatform.config}; do
            for root in "$out/lib/gcc" "$out/usr/lib/gcc"; do
              mkdir -p "$root/$triple/$gccVersion"
              for path in "$gccLibDir"/* ${gcc.cc.lib}/lib/libgcc_s* ${gcc.cc.lib}/lib/libstdc++*; do
                [ -e "$path" ] || continue
                ln -s "$path" "$root/$triple/$gccVersion/$(basename "$path")" 2>/dev/null || true
              done
            done
          done
        done
      done

      ln -s ../lib "$out/usr/lib"
      ln -s lib "$out/lib64"
      ln -s ../lib "$out/usr/lib64"
    ''
    + lib.optionalString (profile != "compiler") ''
      materializeIncludeDir() {
        local includeDir="$1"
        [ -L "$includeDir" ] || return 0
        [ -d "$includeDir" ] || return 0

        local includeTarget
        includeTarget="$(readlink -f "$includeDir")"
        rm "$includeDir"
        mkdir -p "$includeDir"
        for path in "$includeTarget"/*; do
          [ -e "$path" ] || continue
          ln -s "$path" "$includeDir/$(basename "$path")" 2>/dev/null || true
        done
      }

      for includeRoot in ${libcap.dev}/include; do
        [ -d "$includeRoot" ] || continue
        (cd "$includeRoot"
         while IFS= read -r -d ''' path; do
           mkdir -p "$out/include/$path"
           materializeIncludeDir "$out/include/$path"
         done < <(find . -type d -print0)
         while IFS= read -r -d ''' path; do
           ln -s "$PWD/$path" "$out/include/$path" 2>/dev/null || true
         done < <(find . \( -type f -o -type l \) -print0))
      done

      for libdir in ${libcap.lib}/lib; do
        [ -d "$libdir" ] || continue
        for path in "$libdir"/*; do
          ln -s "$path" "$out/lib/$(basename "$path")" 2>/dev/null || true
        done
      done

      mkdir -p "$out/nix/store"
      for storePath in ${libcap.dev} ${libcap.lib}; do
        ln -s "$storePath" "$out/nix/store/$(basename "$storePath")" 2>/dev/null || true
      done
    ''
  );

  libelfCmakePackage = runCommand "therock-libelf-cmake-package" { } ''
    mkdir -p "$out/lib/cmake/LibElf"
    cat > "$out/lib/cmake/LibElf/LibElfConfig.cmake" <<'EOF'
    set(LibElf_FOUND TRUE)
    set(LIBELF_FOUND TRUE)
    set(LIBELF_INCLUDE_DIR "${elfutils.dev}/include")
    set(LIBELF_INCLUDE_DIRS "${elfutils.dev}/include")
    set(LIBELF_LIBRARY "${elfutils.out}/lib/libelf.so")
    set(LIBELF_LIBRARIES "${elfutils.out}/lib/libelf.so")

    if(NOT TARGET elf::elf)
      add_library(elf::elf SHARED IMPORTED)
      set_target_properties(elf::elf PROPERTIES
        IMPORTED_LOCATION "${elfutils.out}/lib/libelf.so"
        INTERFACE_INCLUDE_DIRECTORIES "${elfutils.dev}/include")
    endif()

    if(NOT TARGET elf)
      add_library(elf SHARED IMPORTED)
      set_target_properties(elf PROPERTIES
        IMPORTED_LOCATION "${elfutils.out}/lib/libelf.so"
        INTERFACE_INCLUDE_DIRECTORIES "${elfutils.dev}/include")
    endif()
    EOF
  '';

  numaCmakePackage = runCommand "therock-numa-cmake-package" { } ''
    mkdir -p "$out/lib/cmake/NUMA"
    cat > "$out/lib/cmake/NUMA/NUMAConfig.cmake" <<'EOF'
    set(NUMA_FOUND TRUE)
    set(NUMA_INCLUDE_DIR "${numactl.dev}/include")
    set(NUMA_INCLUDE_DIRS "${numactl.dev}/include")
    set(NUMA_LIBRARY "${numactl.out}/lib/libnuma.so")
    set(NUMA_LIBRARIES "${numactl.out}/lib/libnuma.so")

    if(NOT TARGET NUMA::NUMA)
      add_library(NUMA::NUMA SHARED IMPORTED)
      set_target_properties(NUMA::NUMA PROPERTIES
        IMPORTED_LOCATION "${numactl.out}/lib/libnuma.so"
        INTERFACE_INCLUDE_DIRECTORIES "${numactl.dev}/include")
    endif()
    EOF
  '';

  pythonEnv = python3.withPackages (
    ps: with ps; [
      boto3
      build
      cppheaderparser
      jinja2
      joblib
      lit
      mako
      meson
      msgpack
      packaging
      psutil
      pyelftools
      python-magic
      pyyaml
      pyzstd
      setuptools
      tomli
      zstandard
    ]
  );

  explicitParallelCmakeFlags = lib.optionals (buildJobs != null) [
    "-DTHEROCK_BACKGROUND_BUILD_JOBS=${toString buildJobs}"
    "-DFLANG_PARALLEL_COMPILE_JOBS=${toString buildJobs}"
    "-DLLVM_PARALLEL_COMPILE_JOBS=${toString buildJobs}"
    "-DLLVM_PARALLEL_LINK_JOBS=${toString buildJobs}"
  ];

  nixToolchainDriverFlags = "--sysroot=${nixSysroot} --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib";

  nixToolchainRuntimeLinkerFlags = lib.concatStringsSep " " [
    "-L${nixSysroot}/lib"
    "-Wl,-rpath,${nixSysroot}/lib"
    "-Wl,-rpath,${stdenv.cc.libc}/lib"
    "-Wl,-rpath,${stdenv.cc.cc.lib}/lib"
    "-Wl,-rpath,${gcc.cc.lib}/lib"
  ];

  nixToolchainExeLinkerFlags =
    nixToolchainRuntimeLinkerFlags + " -Wl,--dynamic-linker=${stdenv.cc.bintools.dynamicLinker}";

  rocgdbNixDependencyLinkerFlags = lib.concatStringsSep " " [
    "-L${gmp}/lib"
    "-L${mpfr}/lib"
    "-L${zlib.out}/lib"
    "-L${zstd.out}/lib"
    "-L${xz.out}/lib"
    "-L${ncurses.out}/lib"
    "-Wl,-rpath,${gmp}/lib"
    "-Wl,-rpath,${mpfr}/lib"
    "-Wl,-rpath,${zlib.out}/lib"
    "-Wl,-rpath,${zstd.out}/lib"
    "-Wl,-rpath,${xz.out}/lib"
    "-Wl,-rpath,${ncurses.out}/lib"
  ];

  rocgdbNixCppFlags = lib.concatStringsSep " " [
    "-I${zlib.dev}/include"
    "-I${zstd.dev}/include"
    "-I${xz.dev}/include"
    "-I${ncurses.dev}/include"
    "-I${ncurses.dev}/include/ncursesw"
  ];

  rocgdbNixPkgConfigPath = lib.concatStringsSep ":" [
    "${zstd.dev}/lib/pkgconfig"
    "${xz.dev}/lib/pkgconfig"
    "${ncurses.dev}/lib/pkgconfig"
    "${zlib.dev}/share/pkgconfig"
  ];

  ireeVendoredSourceCmakeArgs = ''
            -DFETCHCONTENT_FULLY_DISCONNECTED=ON
    ${
      lib.optionalString (ireeLibbacktraceSource != null) ''
        -DFETCHCONTENT_SOURCE_DIR_LIBBACKTRACE_SRC=''${CMAKE_CURRENT_SOURCE_DIR}/iree/third_party/libbacktrace-src
      ''
    }${
      lib.optionalString (ireeAmdDeviceLibsArchive != null) ''
        -DIREE_TARGET_BACKEND_ROCM_DEVICE_BC_PATH=''${CMAKE_CURRENT_SOURCE_DIR}/iree/third_party/amdgpu-device-libs
      ''
    }'';

  ireeCommonCmakeArgs = ''
          -DHAVE_STD_REGEX=ON
          -DIREE_ENABLE_RUNTIME_TRACING=ON
          -DIREE_ENABLE_COMPILER_TRACING=ON
          -DIREE_LINK_COMPILER_SHARED_LIBRARY=OFF
    ${ireeVendoredSourceCmakeArgs}'';

  fetchContentToolchainLine = lib.optionalString (profile != "compiler") ''
    string(APPEND _toolchain_contents "set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL \"Do not allow FetchContent network access\" FORCE)\n")
  '';

  mediaLibCmakeArgs = lib.optionalString (profile == "full") ''
    -DLIBVA_INCLUDE_DIR=${libva.dev}/include
    -DLIBVA_LIBRARY=${libva.out}/lib/libva.so
    -DLIBVA_DRM_LIBRARY=${libva.out}/lib/libva-drm.so
    -DLIBDRM_AMDGPU_INCLUDE_DIR=${libdrm.dev}/include
    -DLIBDRM_AMDGPU_LIBRARY=${libdrm.out}/lib/libdrm_amdgpu.so
    -DAVCODEC_INCLUDE_DIR=${ffmpeg.dev}/include
    -DAVCODEC_LIBRARY=${ffmpeg.lib}/lib/libavcodec.so
    -DAVFORMAT_INCLUDE_DIR=${ffmpeg.dev}/include
    -DAVFORMAT_LIBRARY=${ffmpeg.lib}/lib/libavformat.so
    -DAVUTIL_INCLUDE_DIR=${ffmpeg.dev}/include
    -DAVUTIL_LIBRARY=${ffmpeg.lib}/lib/libavutil.so
    -DPKG_CONFIG_EXECUTABLE=${pkg-config}/bin/pkg-config
  '';

  # TheRock sets FETCHCONTENT_FULLY_DISCONNECTED=ON globally so FetchContent
  # cannot extract the OTF2 URL archive on its own. OTF2's ExternalProject_Add
  # also uses BUILD_IN_SOURCE=1, so the source must live in a writable location.
  # Extract the tarball into the source tree (writable after unpack) and point
  # FetchContent at it via FETCHCONTENT_SOURCE_DIR_OTF2-SOURCE.
  rocprofilerOtf2SourcePatch = lib.optionalString (rocprofilerOtf2Archive != null) ''
    mkdir -p rocm-systems/projects/rocprofiler-sdk/external/otf2/otf2-source-prebuilt
    tar -xzf ${rocprofilerOtf2Archive} \
      -C rocm-systems/projects/rocprofiler-sdk/external/otf2/otf2-source-prebuilt \
      --strip-components=1
    chmod -R u+w rocm-systems/projects/rocprofiler-sdk/external/otf2/otf2-source-prebuilt

    substituteInPlace rocm-systems/projects/rocprofiler-sdk/external/otf2/CMakeLists.txt \
      --replace-fail \
        'set(FETCHCONTENT_BASE_DIR ''${ROCPROFILER_BINARY_DIR}/external/packages)' \
        'set(FETCHCONTENT_BASE_DIR ''${ROCPROFILER_BINARY_DIR}/external/packages)
    set(FETCHCONTENT_SOURCE_DIR_OTF2-SOURCE "''${CMAKE_CURRENT_SOURCE_DIR}/otf2-source-prebuilt" CACHE PATH "Pre-extracted OTF2 source tree" FORCE)'
  '';

  # rocFFT's sqlite.cmake uses FetchContent to fetch the SQLite amalgamation
  # zip. Same reason as OTF2: FETCHCONTENT_FULLY_DISCONNECTED=ON globally
  # prevents URL extraction, so we pre-extract the archive into the source
  # tree and override FETCHCONTENT_SOURCE_DIR_SQLITE_LOCAL to point at it.
  rocfftSourcePatch =
    lib.optionalString (profile != "compiler" && thirdPartySources ? "sqlite-amalgamation-3510300.zip")
      ''
        mkdir -p rocm-libraries/projects/rocfft/sqlite-source-prebuilt.tmp
        unzip -q ${thirdPartyFetches."sqlite-amalgamation-3510300.zip"} \
          -d rocm-libraries/projects/rocfft/sqlite-source-prebuilt.tmp
        mv rocm-libraries/projects/rocfft/sqlite-source-prebuilt.tmp/sqlite-amalgamation-3510300 \
           rocm-libraries/projects/rocfft/sqlite-source-prebuilt
        rmdir rocm-libraries/projects/rocfft/sqlite-source-prebuilt.tmp
        chmod -R u+w rocm-libraries/projects/rocfft/sqlite-source-prebuilt

        substituteInPlace rocm-libraries/projects/rocfft/cmake/sqlite.cmake \
          --replace-fail \
            'FetchContent_Declare(sqlite_local' \
            'set(FETCHCONTENT_SOURCE_DIR_SQLITE_LOCAL "''${CMAKE_CURRENT_LIST_DIR}/../sqlite-source-prebuilt" CACHE PATH "Pre-extracted SQLite amalgamation source tree" FORCE)
        FetchContent_Declare(sqlite_local'
      '';

  ireeTracingAndSourcePatch = lib.optionalString (profile == "full") ''
        substituteInPlace iree-libs/CMakeLists.txt \
          --replace-fail \
            '      -DIREE_USE_SYSTEM_DEPS=ON
          # In this targeted build, several submodules' \
            '      -DIREE_USE_SYSTEM_DEPS=ON
    ${ireeCommonCmakeArgs}      # In this targeted build, several submodules' \
          --replace-fail \
            '      -DIREE_USE_SYSTEM_DEPS=ON
      )' \
            '      -DIREE_USE_SYSTEM_DEPS=ON
    ${ireeCommonCmakeArgs}  )' \
          --replace-fail \
            '      -DIREE_USE_SYSTEM_DEPS=ON
          -DHIP_PLATFORM=amd
        COMPILER_TOOLCHAIN' \
            '      -DIREE_USE_SYSTEM_DEPS=ON
    ${ireeCommonCmakeArgs}      -DHIP_PLATFORM=amd
        COMPILER_TOOLCHAIN'
  '';

  hipLanguageFlagsToolchainLine = lib.optionalString (profile != "compiler") ''
    string(APPEND _toolchain_contents "string(APPEND CMAKE_C_FLAGS_INIT \" -isystem ${numactl.dev}/include\")\n")
    string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_INIT \" -isystem ${numactl.dev}/include\")\n")
    string(APPEND _toolchain_contents "string(APPEND CMAKE_HIP_FLAGS_INIT \" ${nixToolchainDriverFlags} --hip-path=@_hip_dist_dir@ --hip-device-lib-path=@_amd_llvm_device_lib_path@\")\n")
    string(APPEND _toolchain_contents "set(CMAKE_CXX_LINK_FLAGS \"${nixToolchainDriverFlags} ${nixToolchainExeLinkerFlags}\" CACHE STRING \"Nix CXX linker flags for HIP link helpers\" FORCE)\n")
    string(APPEND _toolchain_contents "set(HIP_HIPCC_FLAGS \"--sysroot=${nixSysroot};--gcc-toolchain=${nixSysroot}/usr;-B${nixSysroot}/lib\" CACHE STRING \"Nix HIP compiler flags\" FORCE)\n")
    string(APPEND _toolchain_contents "set(HIP_CLANG_FLAGS \"--sysroot=${nixSysroot};--gcc-toolchain=${nixSysroot}/usr;-B${nixSysroot}/lib\" CACHE STRING \"Nix HIP clang flags\" FORCE)\n")
  '';

  nixLiveLinkerFlagsSourcePatch = lib.optionalString (profile != "compiler") ''
      substituteInPlace cmake/therock_subproject.cmake \
        --replace-fail \
          '  string(APPEND _init_contents "set(THEROCK_PKG_CONFIG_DIRS \"@_private_pkg_config_dirs@\")\n")' \
          '  string(APPEND _init_contents "string(APPEND CMAKE_EXE_LINKER_FLAGS \" ${nixToolchainExeLinkerFlags}\")\n")
    string(APPEND _init_contents "string(APPEND CMAKE_SHARED_LINKER_FLAGS \" ${nixToolchainRuntimeLinkerFlags}\")\n")
    string(APPEND _init_contents "string(APPEND CMAKE_MODULE_LINKER_FLAGS \" ${nixToolchainRuntimeLinkerFlags}\")\n")
    string(APPEND _init_contents "set(THEROCK_PKG_CONFIG_DIRS \"@_private_pkg_config_dirs@\")\n")'
  '';

  hipRuntimeSourcePatch = lib.optionalString (profile != "compiler") ''
    substituteInPlace rocm-systems/projects/clr/hipamd/src/hip_embed_pch.sh \
      --replace-fail \
        'LLVM_DIR="$4"' \
        'LLVM_DIR="$4"
    NIX_CLANG_DRIVER_FLAGS=(--sysroot=${nixSysroot} --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib)' \
      --replace-fail \
        '$LLVM_DIR/bin/clang -O3' \
        '$LLVM_DIR/bin/clang "''${NIX_CLANG_DRIVER_FLAGS[@]}" -O3' \
      --replace-fail \
        '$LLVM_DIR/bin/clang $tmp/hiprtc_header.o' \
        '$LLVM_DIR/bin/clang "''${NIX_CLANG_DRIVER_FLAGS[@]}" $tmp/hiprtc_header.o'

    substituteInPlace rocm-systems/projects/clr/hipamd/src/CMakeLists.txt \
      --replace-fail \
        '    COMMAND ''${CMAKE_C_COMPILER}' \
        '    COMMAND ''${CMAKE_C_COMPILER}
        "--sysroot=${nixSysroot}"
        "--gcc-toolchain=${nixSysroot}/usr"
        "-B${nixSysroot}/lib"'

    substituteInPlace rocm-systems/projects/hip/cmake/FindHIP.cmake \
      --replace-fail \
        'set(CMAKE_HIP_CREATE_SHARED_LIBRARY "''${HIP_HIPCC_CMAKE_LINKER_HELPER} ''${HIP_CLANG_PATH} ''${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")' \
        'set(CMAKE_HIP_CREATE_SHARED_LIBRARY "''${HIP_HIPCC_CMAKE_LINKER_HELPER} ''${HIP_CLANG_PATH} ''${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} ${nixToolchainDriverFlags} ${nixToolchainRuntimeLinkerFlags} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")' \
      --replace-fail \
        'set(CMAKE_HIP_CREATE_SHARED_MODULE "''${HIP_HIPCC_CMAKE_LINKER_HELPER} ''${HIP_CLANG_PATH} ''${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <LINK_LIBRARIES> -shared" )' \
        'set(CMAKE_HIP_CREATE_SHARED_MODULE "''${HIP_HIPCC_CMAKE_LINKER_HELPER} ''${HIP_CLANG_PATH} ''${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} ${nixToolchainDriverFlags} ${nixToolchainRuntimeLinkerFlags} <LINK_FLAGS> <OBJECTS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <LINK_LIBRARIES> -shared" )' \
      --replace-fail \
        'set(CMAKE_HIP_LINK_EXECUTABLE "''${HIP_HIPCC_CMAKE_LINKER_HELPER} ''${HIP_CLANG_PATH} ''${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")' \
        'set(CMAKE_HIP_LINK_EXECUTABLE "''${HIP_HIPCC_CMAKE_LINKER_HELPER} ''${HIP_CLANG_PATH} ''${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} ${nixToolchainDriverFlags} ${nixToolchainExeLinkerFlags} <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")' \
      --replace-fail \
        'set(CMAKE_HIP_LINK_EXECUTABLE "''${HIP_HIPCC_CMAKE_LINKER_HELPER} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")' \
        'set(CMAKE_HIP_LINK_EXECUTABLE "''${HIP_HIPCC_CMAKE_LINKER_HELPER} ${nixToolchainDriverFlags} ${nixToolchainExeLinkerFlags} <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")'
  '';

  rocprofilerSourcePatch = lib.optionalString (profile != "compiler") ''
    substituteInPlace profiler/CMakeLists.txt \
      --replace-fail \
        '      -DROCPROFILER_BUILD_DEVELOPER=OFF' \
        '      -DSQLite3_INCLUDE_DIR=${sqlite.dev}/include
      -DSQLite3_LIBRARY=${sqlite.out}/lib/libsqlite3.so
      -DLibElf_DIR=${libelfCmakePackage}/lib/cmake/LibElf
      -DROCPROFILER_BUILD_DEVELOPER=OFF'

    substituteInPlace profiler/CMakeLists.txt \
      --replace-fail \
        '      -DROCPROFSYS_BUILD_DYNINST=ON' \
        '      -DROCPROFSYS_BUILD_DYNINST=ON
      "-DOpenMP_C_FLAGS=-fopenmp=libomp"
      "-DOpenMP_CXX_FLAGS=-fopenmp=libomp"
      -DOpenMP_C_INCLUDE_DIR=${llvmPackages.openmp.dev}/include
      -DOpenMP_CXX_INCLUDE_DIR=${llvmPackages.openmp.dev}/include
      -DOpenMP_C_LIB_NAMES=omp
      -DOpenMP_CXX_LIB_NAMES=omp
      -DOpenMP_omp_LIBRARY=${llvmPackages.openmp}/lib/libomp.so
      "-DCMAKE_Fortran_FLAGS_INIT=${nixToolchainDriverFlags}"
      "-DCMAKE_Fortran_FLAGS=${nixToolchainDriverFlags}"'

    ${rocprofilerOtf2SourcePatch}
  '';

  roctracerSourcePatch = lib.optionalString (profile != "compiler") ''
    substituteInPlace \
      rocm-systems/projects/roctracer/src/roctracer/loader.h \
      rocm-systems/projects/roctracer/src/tracer_tool/tracer_tool.cpp \
      rocm-systems/projects/roctracer/src/hip_stats/hip_stats.cpp \
      rocm-systems/projects/roctracer/plugin/file/file.cpp \
      --replace-fail '#include <experimental/filesystem>' '#include <filesystem>' \
      --replace-fail 'std::experimental::filesystem' 'std::filesystem'

    substituteInPlace rocm-systems/projects/roctracer/src/CMakeLists.txt \
      --replace-fail \
        '  COMMAND ''${CMAKE_C_COMPILER} "$<$<BOOL:''${HIP_INCLUDE_DIRECTORIES}>:-I$<JOIN:''${HIP_INCLUDE_DIRECTORIES},$<SEMICOLON>-I>>"' \
        '  COMMAND ''${CMAKE_C_COMPILER}
          "--sysroot=${nixSysroot}"
          "--gcc-toolchain=${nixSysroot}/usr"
          "-B${nixSysroot}/lib"
          "$<$<BOOL:''${HIP_INCLUDE_DIRECTORIES}>:-I$<JOIN:''${HIP_INCLUDE_DIRECTORIES},$<SEMICOLON>-I>>"' \
      --replace-fail \
        'target_link_libraries(roctracer PRIVATE util hsa-runtime64::hsa-runtime64 stdc++fs Threads::Threads dl)' \
        'target_link_libraries(roctracer PRIVATE util hsa-runtime64::hsa-runtime64 Threads::Threads dl)' \
      --replace-fail \
        'target_link_libraries(roctracer_tool util roctracer hsa-runtime64::hsa-runtime64 stdc++fs Threads::Threads atomic dl)' \
        'target_link_libraries(roctracer_tool util roctracer hsa-runtime64::hsa-runtime64 Threads::Threads atomic dl)' \
      --replace-fail \
        'target_link_libraries(hip_stats roctracer stdc++fs)' \
        'target_link_libraries(hip_stats roctracer)'

    substituteInPlace rocm-systems/projects/roctracer/plugin/file/CMakeLists.txt \
      --replace-fail \
        'target_link_libraries(file_plugin PRIVATE util roctracer  amd_comgr hsa-runtime64::hsa-runtime64 stdc++fs amd_comgr)' \
        'target_link_libraries(file_plugin PRIVATE util roctracer amd_comgr hsa-runtime64::hsa-runtime64 amd_comgr)'

    ${rocprofilerSourcePatch}
  '';

  rocshmemSourcePatch = lib.optionalString (profile != "compiler") ''
    NIX_ROCSHMEM_BITCODE_FLAGS=${lib.escapeShellArg nixToolchainDriverFlags} perl -0pi -e '
      my $flags = $ENV{NIX_ROCSHMEM_BITCODE_FLAGS};
      $flags =~ s/ /\n    /g;
      s/(set\(BITCODE_COMPILE_FLAGS_BASE\n)/$1    $flags\n/
        or die "failed to add Nix driver flags to rocSHMEM device bitcode compile\n";
    ' rocm-systems/projects/rocshmem/cmake/DeviceBitcode.cmake

    substituteInPlace rocm-systems/projects/rocshmem/src/gda/numa_wrapper.cpp \
      --replace-fail \
        'dlopen("libnuma.so", RTLD_NOW)' \
        'dlopen("${numactl.out}/lib/libnuma.so", RTLD_NOW)'
  '';

  compilerBuiltinsSourcePatch = ''
    substituteInPlace compiler/CMakeLists.txt \
      --replace-fail \
        "  set(_extra_llvm_cmake_args)" \
        "  set(_extra_llvm_cmake_args)
      list(APPEND _extra_llvm_cmake_args \"-DBUILTINS_CMAKE_ARGS=-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON\")"

    NIX_SYSROOT=${lib.escapeShellArg nixSysroot} perl -0pi -e '
      my $cmake_sysroot = chr(36) . "{CMAKE_SYSROOT}";
      my $needle = quotemeta("  if(CMAKE_SYSROOT)\n    set(sysroot_arg -DCMAKE_SYSROOT=$cmake_sysroot)\n  endif()");
      my $replacement = "  if(CMAKE_SYSROOT)\n    set(sysroot_arg -DCMAKE_SYSROOT=$cmake_sysroot)\n  else()\n    set(sysroot_arg -DCMAKE_SYSROOT=$ENV{NIX_SYSROOT})\n  endif()";
      s/$needle/$replacement/ or die "failed to patch LLVM external project sysroot propagation\n";
    ' compiler/amd-llvm/llvm/cmake/modules/LLVMExternalProjectUtils.cmake
  '';

  explicitParallelSourcePatch = lib.optionalString (buildJobs != null) ''
    substituteInPlace compiler/CMakeLists.txt \
      --replace-fail \
        "  set(_extra_llvm_cmake_args)" \
        "  set(_extra_llvm_cmake_args)
      message(STATUS \"LLVM parallel compile jobs forced to ${toString buildJobs}\")
      list(APPEND _extra_llvm_cmake_args \"-DLLVM_ENABLE_PCH=OFF\")
      list(APPEND _extra_llvm_cmake_args \"-DLLVM_PARALLEL_COMPILE_JOBS=${toString buildJobs}\")
      list(APPEND _extra_llvm_cmake_args \"-DCMAKE_C_FLAGS_RELEASE=-O2 -DNDEBUG\")
      list(APPEND _extra_llvm_cmake_args \"-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG\")
      list(APPEND _extra_llvm_cmake_args \"-DLIBUNWIND_ENABLE_SHARED=OFF\")
      list(APPEND _extra_llvm_cmake_args \"-DLIBUNWIND_ENABLE_STATIC=ON\")"
  '';

  commonCmakeFlags = [
    "-DCMAKE_C_COMPILER=${stdenv.cc}/bin/cc"
    "-DCMAKE_CXX_COMPILER=${stdenv.cc}/bin/c++"
    "-DCMAKE_Fortran_COMPILER=${gfortran}/bin/gfortran"
    "-DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=${distBundleName}"
    "-DLLVM_ENABLE_PCH=OFF"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_TESTING=OFF"
    "-DTHEROCK_BUILD_TESTING=OFF"
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    "-DTHEROCK_BUNDLE_SYSDEPS=OFF"
  ]
  ++ cmakeListFlag "THEROCK_AMDGPU_TARGETS" amdgpuTargets
  ++ cmakeListFlag "THEROCK_DIST_AMDGPU_TARGETS" distAmdgpuTargets
  ++ cmakeListFlag "THEROCK_TEST_AMDGPU_TARGETS" testAmdgpuTargets
  ++ cmakeListFlag "THEROCK_AMDGPU_FAMILIES" amdgpuFamilies
  ++ cmakeListFlag "THEROCK_DIST_AMDGPU_FAMILIES" distAmdgpuFamilies
  ++ cmakeListFlag "THEROCK_TEST_AMDGPU_FAMILIES" testAmdgpuFamilies
  ++ explicitParallelCmakeFlags
  ++ lib.optional (
    spirvHeadersSource != null
  ) "-DLLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR=${spirvHeadersSource}";

  profileCmakeFlags =
    if profile == "compiler" then
      [
        "-DTHEROCK_ENABLE_ALL=OFF"
        "-DTHEROCK_ENABLE_COMPILER=ON"
        "-DTHEROCK_ENABLE_HIPIFY=OFF"
        "-DTHEROCK_ENABLE_CORE=OFF"
        "-DTHEROCK_ENABLE_COMM_LIBS=OFF"
        "-DTHEROCK_ENABLE_MATH_LIBS=OFF"
        "-DTHEROCK_ENABLE_ML_LIBS=OFF"
        "-DTHEROCK_ENABLE_DEBUG_TOOLS=OFF"
        "-DTHEROCK_ENABLE_DC_TOOLS=OFF"
        "-DTHEROCK_ENABLE_IREE_LIBS=OFF"
        "-DTHEROCK_ENABLE_MEDIA_LIBS=OFF"
        "-DTHEROCK_ENABLE_PROFILER=OFF"
        "-DTHEROCK_ENABLE_OCL_ICD=OFF"
        "-DTHEROCK_ENABLE_OCL_RUNTIME=OFF"
        "-DTHEROCK_ENABLE_CORE_HIPTESTS=OFF"
        "-DTHEROCK_ENABLE_CORE_RUNTIME_TESTS=OFF"
      ]
    else if profile == "full" then
      [
        "-DROCGDB_EXTRA_LDFLAGS=${nixToolchainExeLinkerFlags} ${rocgdbNixDependencyLinkerFlags}"
        "-DROCGDB_EXTRA_CPPFLAGS=${rocgdbNixCppFlags}"
        "-DROCGDB_EXTRA_PKG_CONFIG_PATH=${rocgdbNixPkgConfigPath}"
        "-DROCGDB_FILE_CMD=${file}/bin/file"
        "-DROCGDB_GMP_INCLUDE_DIR=${gmp.dev}/include"
        "-DROCGDB_GMP_LIBRARY_DIR=${gmp}/lib"
        "-DROCGDB_MPFR_INCLUDE_DIR=${mpfr.dev}/include"
        "-DROCGDB_MPFR_LIBRARY_DIR=${mpfr}/lib"
      ]
    else
      throw "unknown TheRock ROCm profile: ${profile}";

  cmakeFlags = commonCmakeFlags ++ profileCmakeFlags;
in
stdenv.mkDerivation {
  pname = "therock-rocm-from-source-${target}-${profile}${
    lib.optionalString (nameSuffix != "") "-${nameSuffix}"
  }";
  inherit version;

  src = therockSource;

  patches = [
    ./patches/hipify-use-amd-llvm-toolchain.patch
    ./patches/rocshmem-tolerate-virtual-verbs-devices.patch
  ]
  ++ lib.optionals (profile != "compiler") [
    ./patches/rocgdb-forward-cmake-flags-to-autoconf.patch
    ./patches/opencl-packaging-tolerate-missing-os-release.patch
    ./patches/rdc-packaging-avoid-host-os-release.patch
    ./patches/rocprofiler-sdk-declare-fmt-build-dep.patch
    ./patches/rccl-device-linker-forward-cxx-driver-flags.patch
    ./patches/rocprofiler-systems-libiberty-allow-single-url.patch
    ./patches/fusilli-provider-honor-skip-tests.patch
    ./patches/kpack-ccob-parser-accept-zlib.patch
  ];

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    gfortran
    llvmPackages.lld
    git
  ]
  ++ lib.optionals (profile != "compiler") [
    autoconf
    llvmPackages.llvm.out
    unzip
  ]
  ++ [
    automake
    libtool
    bison
    flex
    texinfo
    xxd
    patchelf
    curl
    perl
    patch
    gnumake
    which
    pythonEnv
  ]
  ++ lib.optionals (profile == "full") [
    file
  ];

  buildInputs = [
    openssl
    libGL
    libGL.dev
    glog
    fmt
    libffi
    libxml2
    zlib
    ncurses
    expat
    numactl.out
    numactl.dev
    libdrm
    pciutils
    libpciaccess
    elfutils.out
    elfutils.dev
    libx11
    libx11.dev
    xorgproto
  ]
  ++ lib.optionals (profile == "full") [
    gmp
    gmp.dev
    mpfr
    mpfr.dev
    zstd.out
    zstd.dev
    xz.out
    xz.dev
    sqlite.out
    sqlite.dev
    libcap.dev
    libcap.lib
    libva
    libva.dev
    llvmPackages.openmp
    llvmPackages.openmp.dev
    ffmpeg.lib
    ffmpeg.dev
    libdrm.dev
    ncurses.dev
    zlib.dev
  ];

  configurePhase = ''
    runHook preConfigure

    ${lib.optionalString (prebuiltStageTree != null) ''
      mkdir -p build
      cp -a ${prebuiltStageTree}/. build/
      chmod -R u+w build
    ''}

    cmake -S . -B build -G Ninja -DCMAKE_INSTALL_PREFIX="$out" ${lib.escapeShellArgs cmakeFlags}
    runHook postConfigure
  '';

  postPatch = ''
                find third-party -name CMakeLists.txt -type f -print0 \
                  | xargs -0 perl -0pi -e 's|https://rocm-third-party-deps\.s3\.us-east-2\.amazonaws\.com/|file://${thirdPartyMirror}/|g'

                ${lib.optionalString (projectTargetUnexcludes != { }) ''
                                cat > cmake/therock_custom_amdgpu_targets.cmake <<'EOF'
                  ${projectTargetUnexcludeCmake}
                  EOF
                ''}

                substituteInPlace core/pre_hook_ROCR-Runtime.cmake \
                  --replace-fail \
                    'find_package(LibElf CONFIG REQUIRED)' \
                    'set(LibElf_DIR "${libelfCmakePackage}/lib/cmake/LibElf")
            find_package(LibElf CONFIG REQUIRED)
            set(NUMA_DIR "${numaCmakePackage}/lib/cmake/NUMA")
            find_package(NUMA CONFIG REQUIRED)'

                substituteInPlace rocm-systems/projects/rocr-runtime/libhsakmt/CMakeLists.txt \
                  --replace-fail \
                    '    message(STATUS "NUMA: " ''${NUMA})' \
                    '    include_directories(''${NUMA_INCLUDE_DIRS})
                message(STATUS "NUMA: " ''${NUMA})'

                substituteInPlace cmake/therock_subproject.cmake \
                  --replace-fail \
                    '    string(APPEND _toolchain_contents "set(CMAKE_LINKER \"@AMD_LLVM_LINKER@\")\n")' \
                    '    string(APPEND _toolchain_contents "set(CMAKE_LINKER \"@AMD_LLVM_LINKER@\")\n")
            string(APPEND _toolchain_contents "set(CMAKE_SYSROOT \"${nixSysroot}\")\n")
    ${fetchContentToolchainLine}
            string(APPEND _toolchain_contents "string(APPEND CMAKE_C_FLAGS_INIT \" --sysroot=${nixSysroot} --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib\")\n")
            string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_INIT \" --sysroot=${nixSysroot} --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib\")\n")
            string(APPEND _toolchain_contents "string(APPEND CMAKE_ASM_FLAGS_INIT \" --sysroot=${nixSysroot} --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib\")\n")
            string(APPEND _toolchain_contents "string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT \" ${nixToolchainExeLinkerFlags}\")\n")
            string(APPEND _toolchain_contents "string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT \" ${nixToolchainRuntimeLinkerFlags}\")\n")
            string(APPEND _toolchain_contents "string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT \" ${nixToolchainRuntimeLinkerFlags}\")\n")'

                substituteInPlace cmake/therock_subproject.cmake \
                  --replace-fail \
                    '    string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_INIT \" --hip-device-lib-path=@_amd_llvm_device_lib_path@\")\n")' \
                    '    string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_INIT \" --hip-device-lib-path=@_amd_llvm_device_lib_path@\")\n")
            string(APPEND _toolchain_contents "set(CMAKE_HIP_COMPILER \"@AMD_LLVM_CXX_COMPILER@\")\n")
            string(APPEND _toolchain_contents "set(CMAKE_HIP_COMPILER_ROCM_ROOT \"@_hip_dist_dir@\")\n")
            string(APPEND _toolchain_contents "set(CMAKE_HIP_PLATFORM \"amd\")\n")
        ${hipLanguageFlagsToolchainLine}'

                ${hipRuntimeSourcePatch}${nixLiveLinkerFlagsSourcePatch}
                ${roctracerSourcePatch}${rocshmemSourcePatch}${rocfftSourcePatch}

                substituteInPlace media-libs/CMakeLists.txt \
                  --replace-fail \
                    '    CMAKE_ARGS
          -DROCM_PATH=' \
                    '    CMAKE_ARGS
          -DROCM_PATH=
    ${mediaLibCmakeArgs}'

                substituteInPlace compiler/pre_hook_amd-llvm.cmake \
                  --replace-fail \
                    'set(RUNTIMES_CMAKE_ARGS "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON")' \
                    'set(RUNTIMES_CMAKE_ARGS
                  "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON"
                  "-DCMAKE_SYSROOT=${nixSysroot}"
                  "-DCMAKE_C_FLAGS_INIT=--sysroot=${nixSysroot} --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib"
                  "-DCMAKE_CXX_FLAGS_INIT=--sysroot=${nixSysroot} --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib"
                  "-DCMAKE_ASM_FLAGS_INIT=--sysroot=${nixSysroot} --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib"
                  "-DCMAKE_EXE_LINKER_FLAGS_INIT=${nixToolchainExeLinkerFlags}"
                  "-DCMAKE_SHARED_LINKER_FLAGS_INIT=${nixToolchainRuntimeLinkerFlags}"
                  "-DCOMPILER_RT_BUILD_BUILTINS=ON"
                  "-DCOMPILER_RT_BUILD_SANITIZERS=OFF"
                  "-DCOMPILER_RT_BUILD_XRAY=OFF"
                  "-DCOMPILER_RT_BUILD_LIBFUZZER=OFF"
                  "-DCOMPILER_RT_BUILD_PROFILE=OFF"
                  "-DCOMPILER_RT_BUILD_CTX_PROFILE=OFF"
                  "-DCOMPILER_RT_BUILD_MEMPROF=OFF"
                  "-DCOMPILER_RT_BUILD_ORC=OFF"
                  "-DCOMPILER_RT_BUILD_GWP_ASAN=OFF"
                )'

                ${explicitParallelSourcePatch}
                ${compilerBuiltinsSourcePatch}
                ${ireeTracingAndSourcePatch}
  ''
  + lib.optionalString (profile != "compiler") ''
    substituteInPlace dctools/CMakeLists.txt \
      --replace-fail \
        '      -DGRPC_DESIRED_VERSION=''${THEROCK_GRPC_VERSION}' \
        '      -DGRPC_DESIRED_VERSION=''${THEROCK_GRPC_VERSION}
      -DLIB_CAP=${libcap.lib}/lib/libcap.so'
  ''
  + lib.optionalString (spirvHeadersSource != null) ''
    substituteInPlace compiler/CMakeLists.txt \
      --replace-fail \
        "  # If the compiler is not pristine" \
        "  list(APPEND _extra_llvm_cmake_args \"-DLLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR=${spirvHeadersSource}\")

    # If the compiler is not pristine"

    perl -0pi -e 's|(set\(LLVM_EXTERNAL_SPIRV_LLVM_TRANSLATOR_SOURCE_DIR [^\n]+\))|$1\nset(LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR "${spirvHeadersSource}" CACHE PATH "SPIR-V headers source directory")| or die "failed to patch SPIR-V headers source\n";' compiler/pre_hook_amd-llvm.cmake
  ''
  + lib.optionalString (esmiIbLibrarySource != null) ''
    rm -rf rocm-systems/projects/amdsmi/esmi_ib_library
    mkdir -p rocm-systems/projects/amdsmi/esmi_ib_library
    cp -a ${esmiIbLibrarySource}/. rocm-systems/projects/amdsmi/esmi_ib_library/
    chmod -R u+w rocm-systems/projects/amdsmi/esmi_ib_library

    perl -0pi -e 's|    if\(NOT EXISTS \''${PROJECT_SOURCE_DIR}/esmi_ib_library/src\).*?    endif\(\)\n\n    # Make sure|    if(NOT EXISTS \''${PROJECT_SOURCE_DIR}/esmi_ib_library/src)\n        message(FATAL_ERROR "vendored esmi_ib_library is missing")\n    else()\n        message(STATUS "Using vendored esmi_ib_library")\n    endif()\n\n    # Make sure|s or die "failed to patch amdsmi esmi_ib_library network clone\n";' \
      rocm-systems/projects/amdsmi/CMakeLists.txt
  ''
  + lib.optionalString (ireeLibbacktraceSource != null) ''
    rm -rf iree-libs/iree/third_party/libbacktrace-src
    mkdir -p iree-libs/iree/third_party/libbacktrace-src
    cp -a ${ireeLibbacktraceSource}/. iree-libs/iree/third_party/libbacktrace-src/
    chmod -R u+w iree-libs/iree/third_party/libbacktrace-src
  ''
  + lib.optionalString (ireeAmdDeviceLibsArchive != null) ''
    rm -rf iree-libs/iree/third_party/amdgpu-device-libs
    mkdir -p iree-libs/iree/third_party/amdgpu-device-libs
    tar -xzf ${ireeAmdDeviceLibsArchive} -C iree-libs/iree/third_party/amdgpu-device-libs
    chmod -R u+w iree-libs/iree/third_party/amdgpu-device-libs
  ''
  + lib.optionalString (tracySource != null) ''
    rm -rf iree-libs/iree/third_party/tracy
    mkdir -p iree-libs/iree/third_party/tracy
    cp -a ${tracySource}/. iree-libs/iree/third_party/tracy/
    chmod -R u+w iree-libs/iree/third_party/tracy
  ''
  + ''
          BOOST_SHELL=${lib.escapeShellArg stdenv.shell} perl -0pi -e '
            my $cmd = q{      bash "-c" "find . -type f -exec sed -i -e \"1s@^#!/usr/bin/env bash@#!__BOOST_SHELL__@\" -e \"1s@^#!/usr/bin/env sh@#!__BOOST_SHELL__@\" {} +"
        COMMAND
    };
            $cmd =~ s/__BOOST_SHELL__/$ENV{BOOST_SHELL}/g;
            s/(else\(\)\n  # The unix bootstrap script.*?set\(_bootstrap_commands\n    COMMAND\n)/$1$cmd/s
              or die "failed to patch boost bootstrap commands\n";
          ' third-party/boost/cmake_project/CMakeLists.txt

          ${lib.optionalString (profile == "full") ''
                      substituteInPlace third-party/boost/CMakeLists.txt \
                        --replace-fail \
                          'atomic,filesystem,multi_index,system' \
                          'atomic,chrono,date_time,filesystem,multi_index,system,thread,timer'

                      substituteInPlace third-party/boost/cmake_project/CMakeLists.txt \
                        --replace-fail \
                          'set(B2_NJOBS "1")
            if(PROCESSOR_COUNT GREATER 8)
              set(B2_NJOBS "8")
            endif()' \
                          'set(B2_NJOBS "''${PROCESSOR_COUNT}")'

                      substituteInPlace third-party/boost/CMakeLists.txt \
                        --replace-fail \
                          'therock_cmake_subproject_provide_package(therock-boost boost_atomic "lib/cmake/boost_atomic-''${_boost_version}")' \
                          'therock_cmake_subproject_provide_package(therock-boost boost_atomic "lib/cmake/boost_atomic-''${_boost_version}")
            therock_cmake_subproject_provide_package(therock-boost boost_chrono "lib/cmake/boost_chrono-''${_boost_version}")
            therock_cmake_subproject_provide_package(therock-boost boost_date_time "lib/cmake/boost_date_time-''${_boost_version}")'

                      substituteInPlace third-party/boost/CMakeLists.txt \
                        --replace-fail \
                          'therock_cmake_subproject_provide_package(therock-boost boost_system "lib/cmake/boost_system-''${_boost_version}")' \
                          'therock_cmake_subproject_provide_package(therock-boost boost_system "lib/cmake/boost_system-''${_boost_version}")
            therock_cmake_subproject_provide_package(therock-boost boost_thread "lib/cmake/boost_thread-''${_boost_version}")
            therock_cmake_subproject_provide_package(therock-boost boost_timer "lib/cmake/boost_timer-''${_boost_version}")'
          ''}

          perl -0pi -e 's/(therock_cmake_subproject_declare\(rocprofiler-register.*?BACKGROUND_BUILD\n)/$1    CMAKE_ARGS\n      -DROCPROFILER_REGISTER_BUILD_GLOG=OFF\n      -DROCPROFILER_REGISTER_BUILD_FMT=OFF\n/s' base/CMakeLists.txt

          substituteInPlace profiler/CMakeLists.txt \
            --replace-fail \
              "      -DROCPROFILER_BUILD_CI=ON" \
              "      -DROCPROFILER_BUILD_CI=ON
            -DROCPROFILER_BUILD_FMT=OFF"

          ${lib.optionalString (profile == "full") ''
            substituteInPlace profiler/CMakeLists.txt \
              --replace-fail \
                '      -DROCPROFSYS_BUILD_BOOST=ON' \
                '      -DROCPROFSYS_BUILD_BOOST=OFF'

            ${lib.optionalString (rocprofilerSysBinutilsArchive != null) ''
                    substituteInPlace profiler/CMakeLists.txt \
                      --replace-fail \
                        '      -DROCPROFSYS_BUILD_LIBIBERTY=ON' \
                        '      -DROCPROFSYS_BUILD_LIBIBERTY=ON
              -DDYNINST_BINUTILS_DOWNLOAD_URL=file://${rocprofilerSysBinutilsArchive}'
            ''}

            perl -0pi -e 's/(therock_cmake_subproject_declare\(rocprofiler-systems.*?BUILD_DEPS.*?therock-fmt\n)(\s+RUNTIME_DEPS)/$1      therock-boost\n$2/s or die "failed to add therock-boost build dep to rocprofiler-systems\n";' profiler/CMakeLists.txt
          ''}

          patchShebangs .
  '';

  buildPhase = ''
    runHook preBuild
    if [ -d build/third-party/boost/source ]; then
      patchShebangs build/third-party/boost/source
    fi
    ${lib.optionalString (buildJobs != null) ''
      export CMAKE_BUILD_PARALLEL_LEVEL=${toString buildJobs}
    ''}
    parallel_args=()
    if [ -n "''${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
      parallel_args=(--parallel "$CMAKE_BUILD_PARALLEL_LEVEL")
    fi
    ${
      if installMode == "configure-only" then
        ''
          echo "Skipping build for configure-only check"
        ''
      else if buildTargets == [ ] then
        ''
          cmake --build build "''${parallel_args[@]}"
        ''
      else
        ''
          for target in ${lib.escapeShellArgs buildTargets}; do
            cmake --build build --target "$target" "''${parallel_args[@]}"
          done
        ''
    }
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    ${
      if installMode == "rocm-install" then
        ''
          cmake --install build
        ''
      else if installMode == "prebuilt-stages" then
        ''
          for subproject in ${stageSubprojectShellWords}; do
            src="build/compiler/$subproject/stage"
            if [ ! -d "$src" ]; then
              echo "missing expected TheRock stage directory: $src" >&2
              exit 1
            fi
            mkdir -p "$out/compiler/$subproject"
            cp -a "$src" "$out/compiler/$subproject/stage"
            touch "$out/compiler/$subproject/stage.prebuilt"
          done
        ''
      else if installMode == "check-only" || installMode == "configure-only" then
        ''
          mkdir -p "$out"
          if [ -d build/logs ]; then
            cp -a build/logs "$out/logs"
          fi
          if [ -f build/CMakeCache.txt ]; then
            cp build/CMakeCache.txt "$out/CMakeCache.txt"
          fi
          touch "$out/succeeded"
        ''
      else
        throw "unknown TheRock ROCm installMode: ${installMode}"
    }
    runHook postInstall
  '';

  passthru = {
    inherit thirdPartyMirror;
  };

  meta = {
    description = "TheRock ROCm ${version} built from source for ${target}";
    homepage = "https://github.com/ROCm/TheRock";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
