{
  lib,
  stdenv,
  runCommand,
  fetchurl,
  cmake,
  ninja,
  pkg-config,
  gfortran,
  git,
  automake,
  libtool,
  bison,
  flex,
  texinfo,
  xxd,
  patchelf,
  curl,
  perl,
  patch,
  gnumake,
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
  expat,
  numactl,
  libdrm,
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
  buildJobs ? null,
  buildTargets ? [ ],
  installMode ? "rocm-install",
  prebuiltStageTree ? null,
  target ? "gfx1151",
  version ? "7.13",
  profile ? "vllm",
}:

let
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

  nixSysroot = runCommand "therock-rocm-nix-sysroot" { } ''
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
    for versionDir in ${stdenv.cc.cc}/include/c++/*; do
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

    ln -s ../include "$out/usr/include"

    for libdir in ${stdenv.cc.libc}/lib ${stdenv.cc.cc.lib}/lib ${stdenv.cc.cc}/lib/gcc/*/* ${libGL}/lib ${libx11}/lib; do
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
      ${stdenv.cc.cc.lib}; do
      ln -s "$storePath" "$out/nix/store/$(basename "$storePath")" 2>/dev/null || true
    done

    # The freshly-built AMD clang is not Nix-wrapped. Its GCC detector expects
    # a conventional sysroot-relative GCC install layout for crtbegin/libgcc.
    for gccLibDir in ${stdenv.cc.cc}/lib/gcc/*/*; do
      gccVersion="$(basename "$gccLibDir")"
      for triple in x86_64-pc-linux-gnu x86_64-linux-gnu ${stdenv.hostPlatform.config}; do
        for root in "$out/lib/gcc" "$out/usr/lib/gcc"; do
          mkdir -p "$root/$triple/$gccVersion"
          for path in "$gccLibDir"/*; do
            ln -s "$path" "$root/$triple/$gccVersion/$(basename "$path")" 2>/dev/null || true
          done
        done
      done
    done

    ln -s ../lib "$out/usr/lib"
    ln -s lib "$out/lib64"
    ln -s ../lib "$out/usr/lib64"
  '';

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

  hipLanguageFlagsToolchainLine = lib.optionalString (profile != "compiler") ''
    string(APPEND _toolchain_contents "string(APPEND CMAKE_HIP_FLAGS_INIT \" --sysroot=${nixSysroot} --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib --hip-path=@_hip_dist_dir@ --hip-device-lib-path=@_amd_llvm_device_lib_path@\")\n")
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
    "-DTHEROCK_AMDGPU_TARGETS=${target}"
    "-DTHEROCK_DIST_AMDGPU_TARGETS=${target}"
    "-DTHEROCK_TEST_AMDGPU_TARGETS=${target}"
    "-DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=${target}"
    "-DLLVM_ENABLE_PCH=OFF"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_TESTING=OFF"
    "-DTHEROCK_BUILD_TESTING=OFF"
    "-DTHEROCK_BUNDLE_SYSDEPS=OFF"
  ]
  ++ explicitParallelCmakeFlags
  ++ lib.optional (
    spirvHeadersSource != null
  ) "-DLLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR=${spirvHeadersSource}";

  profileCmakeFlags =
    if profile == "vllm" then
      [
        "-DTHEROCK_ENABLE_ALL=OFF"
        "-DTHEROCK_ENABLE_CORE=ON"
        "-DTHEROCK_ENABLE_COMM_LIBS=ON"
        "-DTHEROCK_ENABLE_MATH_LIBS=ON"
        "-DTHEROCK_ENABLE_ML_LIBS=OFF"
        "-DTHEROCK_ENABLE_DEBUG_TOOLS=OFF"
        "-DTHEROCK_ENABLE_DC_TOOLS=OFF"
        "-DTHEROCK_ENABLE_IREE_LIBS=OFF"
        "-DTHEROCK_ENABLE_MEDIA_LIBS=OFF"
        "-DTHEROCK_ENABLE_PROFILER=OFF"
        "-DTHEROCK_FLAG_INCLUDE_PROFILER=OFF"
        "-DTHEROCK_ENABLE_OCL_ICD=OFF"
        "-DTHEROCK_ENABLE_OCL_RUNTIME=OFF"
        "-DTHEROCK_ENABLE_CORE_HIPTESTS=OFF"
        "-DTHEROCK_ENABLE_CORE_RUNTIME_TESTS=OFF"
        "-DTHEROCK_ENABLE_ROCSHMEM=OFF"
        "-DTHEROCK_ENABLE_FFT=OFF"
        "-DTHEROCK_ENABLE_SPARSE=OFF"
        "-DTHEROCK_ENABLE_SOLVER=OFF"
        "-DTHEROCK_ENABLE_ROCWMMA=OFF"
        "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=OFF"
        "-DTHEROCK_ENABLE_MIOPEN=OFF"
        "-DTHEROCK_ENABLE_HIPDNN=OFF"
        "-DTHEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS=OFF"
        "-DTHEROCK_ENABLE_MIOPENPROVIDER=OFF"
        "-DTHEROCK_ENABLE_HIPBLASLTPROVIDER=OFF"
        "-DTHEROCK_ENABLE_HIPKERNELPROVIDER=OFF"
        "-DTHEROCK_ENABLE_HIPDNN_SAMPLES=OFF"
      ]
    else if profile == "compiler" then
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
      [ ]
    else
      throw "unknown TheRock ROCm profile: ${profile}";

  cmakeFlags = commonCmakeFlags ++ profileCmakeFlags;
in
stdenv.mkDerivation {
  pname = "therock-rocm-from-source-${target}-${profile}";
  inherit version;

  src = therockSource;

  patches = [
    ./patches/hipify-use-amd-llvm-toolchain.patch
  ];

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    gfortran
    llvmPackages.lld
    git
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
    string(APPEND _toolchain_contents "string(APPEND CMAKE_C_FLAGS_INIT \" --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib\")\n")
    string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_INIT \" --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib\")\n")
    string(APPEND _toolchain_contents "string(APPEND CMAKE_ASM_FLAGS_INIT \" --gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib\")\n")
    string(APPEND _toolchain_contents "string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT \" -L${nixSysroot}/lib\")\n")
    string(APPEND _toolchain_contents "string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT \" -L${nixSysroot}/lib\")\n")
    string(APPEND _toolchain_contents "string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT \" -L${nixSysroot}/lib\")\n")'

        substituteInPlace cmake/therock_subproject.cmake \
          --replace-fail \
            '    string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_INIT \" --hip-device-lib-path=@_amd_llvm_device_lib_path@\")\n")' \
            '    string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_INIT \" --hip-device-lib-path=@_amd_llvm_device_lib_path@\")\n")
    string(APPEND _toolchain_contents "set(CMAKE_HIP_COMPILER \"@AMD_LLVM_CXX_COMPILER@\")\n")
    string(APPEND _toolchain_contents "set(CMAKE_HIP_COMPILER_ROCM_ROOT \"@_hip_dist_dir@\")\n")
    string(APPEND _toolchain_contents "set(CMAKE_HIP_PLATFORM \"amd\")\n")
${hipLanguageFlagsToolchainLine}'

        ${hipRuntimeSourcePatch}

        substituteInPlace compiler/pre_hook_amd-llvm.cmake \
          --replace-fail \
            'set(LLVM_ENABLE_PROJECTS "clang;lld;clang-tools-extra" CACHE STRING "Enable LLVM projects" FORCE)' \
            'set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "Enable LLVM projects" FORCE)' \
          --replace-fail \
            'set(LLVM_ENABLE_PROJECTS "clang;lld;clang-tools-extra;flang" CACHE STRING "Enable LLVM projects" FORCE)' \
            'set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "Enable LLVM projects" FORCE)' \
          --replace-fail \
            'set(LLVM_ENABLE_RUNTIMES "compiler-rt;libunwind;libcxx;libcxxabi;openmp;offload" CACHE STRING "Enabled runtimes" FORCE)' \
            'set(LLVM_ENABLE_RUNTIMES "compiler-rt;libunwind;libcxx;libcxxabi" CACHE STRING "Enabled runtimes" FORCE)' \
          --replace-fail \
            'set(RUNTIMES_CMAKE_ARGS "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON")' \
            'set(RUNTIMES_CMAKE_ARGS
          "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON"
          "-DCMAKE_SYSROOT=${nixSysroot}"
          "-DCMAKE_C_FLAGS_INIT=--gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib"
          "-DCMAKE_CXX_FLAGS_INIT=--gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib"
          "-DCMAKE_ASM_FLAGS_INIT=--gcc-toolchain=${nixSysroot}/usr -B${nixSysroot}/lib"
          "-DCMAKE_EXE_LINKER_FLAGS_INIT=-L${nixSysroot}/lib"
          "-DCMAKE_SHARED_LINKER_FLAGS_INIT=-L${nixSysroot}/lib"
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

        NIX_SYSROOT=${lib.escapeShellArg nixSysroot} perl -0pi -e '
          my $args = q{  set(RUNTIMES_CMAKE_ARGS
          "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON"
          "-DCMAKE_SYSROOT=__NIX_SYSROOT__"
          "-DCMAKE_C_FLAGS_INIT=--gcc-toolchain=__NIX_SYSROOT__/usr -B__NIX_SYSROOT__/lib"
          "-DCMAKE_CXX_FLAGS_INIT=--gcc-toolchain=__NIX_SYSROOT__/usr -B__NIX_SYSROOT__/lib"
          "-DCMAKE_ASM_FLAGS_INIT=--gcc-toolchain=__NIX_SYSROOT__/usr -B__NIX_SYSROOT__/lib"
          "-DCMAKE_EXE_LINKER_FLAGS_INIT=-L__NIX_SYSROOT__/lib"
          "-DCMAKE_SHARED_LINKER_FLAGS_INIT=-L__NIX_SYSROOT__/lib"
          "-DCOMPILER_RT_BUILD_BUILTINS=ON"
          "-DCOMPILER_RT_BUILD_SANITIZERS=OFF"
          "-DCOMPILER_RT_BUILD_XRAY=OFF"
          "-DCOMPILER_RT_BUILD_LIBFUZZER=OFF"
          "-DCOMPILER_RT_BUILD_PROFILE=OFF"
          "-DCOMPILER_RT_BUILD_CTX_PROFILE=OFF"
          "-DCOMPILER_RT_BUILD_MEMPROF=OFF"
          "-DCOMPILER_RT_BUILD_ORC=OFF"
          "-DCOMPILER_RT_BUILD_GWP_ASAN=OFF"
        )
    };
          $args =~ s/__NIX_SYSROOT__/$ENV{NIX_SYSROOT}/g;
          s/(set\(LLVM_ENABLE_RUNTIMES "compiler-rt;libunwind;libcxx;libcxxabi" CACHE STRING "Enabled runtimes" FORCE\)\n)/$1$args/
            or die "failed to add unconditional runtimes CMake args\n";
        ' compiler/pre_hook_amd-llvm.cmake

        ${explicitParallelSourcePatch}
        ${compilerBuiltinsSourcePatch}
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
  + ''
          BOOST_SHELL=${lib.escapeShellArg stdenv.shell} perl -0pi -e '
            my $cmd = q{      bash "-c" "find . -type f -exec sed -i -e \"1s@^#!/usr/bin/env bash@#!__BOOST_SHELL__@\" -e \"1s@^#!/usr/bin/env sh@#!__BOOST_SHELL__@\" {} +"
        COMMAND
    };
            $cmd =~ s/__BOOST_SHELL__/$ENV{BOOST_SHELL}/g;
            s/(else\(\)\n  # The unix bootstrap script.*?set\(_bootstrap_commands\n    COMMAND\n)/$1$cmd/s
              or die "failed to patch boost bootstrap commands\n";
          ' third-party/boost/cmake_project/CMakeLists.txt

          perl -0pi -e 's/(therock_cmake_subproject_declare\(rocprofiler-register.*?BACKGROUND_BUILD\n)/$1    CMAKE_ARGS\n      -DROCPROFILER_REGISTER_BUILD_GLOG=OFF\n      -DROCPROFILER_REGISTER_BUILD_FMT=OFF\n/s' base/CMakeLists.txt

          ${lib.optionalString (profile == "vllm") ''
            perl -0pi -e 's/, "rocprofiler-sdk"//g; s/"rocprofiler-sdk", //g; s/"rocprofiler-sdk"//g' BUILD_TOPOLOGY.toml

            substituteInPlace comm-libs/CMakeLists.txt \
              --replace-fail \
                "      -DBUILD_TESTS=''${THEROCK_BUILD_TESTING}" \
                "      -DBUILD_TESTS=''${THEROCK_BUILD_TESTING}
                  -DROCTX=OFF
                  -DRCCL_ROCPROFILER_REGISTER=OFF"
          ''}

          substituteInPlace profiler/CMakeLists.txt \
            --replace-fail \
              "      -DROCPROFILER_BUILD_CI=ON" \
              "      -DROCPROFILER_BUILD_CI=ON
            -DROCPROFILER_BUILD_FMT=OFF"

          patchShebangs .
  '';

  buildPhase = ''
    runHook preBuild
    if [ -d build/third-party/boost/source ]; then
      patchShebangs build/third-party/boost/source
    fi
    ${
      if buildJobs == null then
        ''
          if [ -n "''${NIX_BUILD_CORES:-}" ] && [ "''${NIX_BUILD_CORES:-0}" -gt 0 ]; then
            export CMAKE_BUILD_PARALLEL_LEVEL="$NIX_BUILD_CORES"
          fi
        ''
      else
        ''
          export CMAKE_BUILD_PARALLEL_LEVEL=${toString buildJobs}
        ''
    }
    parallel_args=()
    if [ -n "''${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
      parallel_args=(--parallel "$CMAKE_BUILD_PARALLEL_LEVEL")
    fi
    ${
      if buildTargets == [ ] then
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
          for subproject in amd-llvm amd-comgr hipcc; do
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
      else
        throw "unknown TheRock ROCm installMode: ${installMode}"
    }
    runHook postInstall
  '';

  meta = {
    description = "TheRock ROCm ${version} built from source for ${target}";
    homepage = "https://github.com/ROCm/TheRock";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
