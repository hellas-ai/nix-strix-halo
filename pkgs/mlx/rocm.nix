{
  lib,
  stdenv,
  mlx,
  applyPatches,
  bash,
  mlx-src,
  makeWrapper,
  patchelf,
  python,
  rdma-core,
  rocmPackages,
  gfx ? "gfx1151",
  pname ? "mlx-rocm",
}:

let
  mlx-rocm-src = applyPatches {
    name = "mlx-rocm-src";
    src = mlx-src;
    patches = [ ./rocm-include-rocblas.patch ];
  };
  gccCxxInclude = "${stdenv.cc.cc}/include/c++/${stdenv.cc.cc.version}";
  rocmRuntimeEnv = {
    ROCM_HOME = "${rocmPackages.clr}";
    ROCM_PATH = "${rocmPackages.clr}";
    HIP_PATH = "${rocmPackages.clr}";
    HIP_PLATFORM = "amd";
    HIP_CLANG_PATH = "${rocmPackages.clr}/llvm/bin";
    HSA_PATH = "${rocmPackages.rocm-runtime}";
    DEVICE_LIB_PATH = "${rocmPackages.rocm-device-libs}/amdgcn/bitcode";
    HIP_DEVICE_LIB_PATH = "${rocmPackages.rocm-device-libs}/amdgcn/bitcode";
    CPATH = lib.concatStringsSep ":" [
      "${rocmPackages.clr}/include"
      "${stdenv.cc.libc.dev}/include"
    ];
    CPLUS_INCLUDE_PATH = lib.concatStringsSep ":" [
      gccCxxInclude
      "${gccCxxInclude}/${stdenv.hostPlatform.config}"
      "${gccCxxInclude}/backward"
      "${stdenv.cc.libc.dev}/include"
    ];
    LIBRARY_PATH = lib.makeLibraryPath [
      rocmPackages.clr
      stdenv.cc.cc.lib
      stdenv.cc.libc
    ];
  };
  rocmWrapperArgs = lib.concatStringsSep " " (
    lib.mapAttrsToList (
      key: value: "--set ${lib.escapeShellArg key} ${lib.escapeShellArg (toString value)}"
    ) rocmRuntimeEnv
  );
in
mlx.overrideAttrs (old: {
  inherit pname;
  version = "0.32.0";
  src = mlx-rocm-src;

  patches = [ ];

  postPatch = (old.postPatch or "") + ''
          substituteInPlace python/mlx/_distributed_utils/launch.py \
            --replace-fail \
              'executable="/bin/bash"' \
              'executable="${bash}/bin/bash"'

          substituteInPlace CMakeLists.txt \
            --replace-fail \
              '  FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.12.0
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL)
      FetchContent_MakeAvailable(nanobind)' \
              '  find_package(nanobind CONFIG REQUIRED)'

          substituteInPlace CMakeLists.txt \
            --replace-fail \
              'message(STATUS "Downloading json")
    FetchContent_Declare(
      json
      URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
    FetchContent_MakeAvailable(json)
    target_include_directories(
      mlx PRIVATE $<BUILD_INTERFACE:''${json_SOURCE_DIR}/single_include/nlohmann>)' \
              'find_package(nlohmann_json REQUIRED)
    target_link_libraries(mlx PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)'

          substituteInPlace CMakeLists.txt \
            --replace-fail \
              '# Add standalone JACCL library (RDMA over Thunderbolt distributed backend)
    if(MLX_BUILD_CPU
       AND ''${CMAKE_SYSTEM_NAME} MATCHES "Darwin"
       AND DEFINED MACOS_SDK_VERSION
       AND MACOS_SDK_VERSION GREATER_EQUAL 26.2)
      add_subdirectory(''${CMAKE_CURRENT_LIST_DIR}/mlx/distributed/jaccl/lib
                       ''${CMAKE_BINARY_DIR}/jaccl)
    endif()' \
              '# Add standalone JACCL library (RDMA over Thunderbolt distributed backend)
    if(MLX_BUILD_CPU AND NOT WIN32)
      add_subdirectory(''${CMAKE_CURRENT_LIST_DIR}/mlx/distributed/jaccl/lib
                       ''${CMAKE_BINARY_DIR}/jaccl)
    endif()'

          substituteInPlace mlx/distributed/jaccl/CMakeLists.txt \
            --replace-fail \
              'if(MLX_BUILD_CPU
       AND ''${CMAKE_SYSTEM_NAME} MATCHES "Darwin"
       AND MACOS_SDK_VERSION VERSION_GREATER_EQUAL 26.2
       AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_GREATER_EQUAL 26.2)' \
              'if(MLX_BUILD_CPU AND NOT WIN32)'
          substituteInPlace mlx/distributed/jaccl/CMakeLists.txt \
            --replace-fail \
              'if(MLX_BUILD_CPU AND NOT WIN32)
      target_sources(mlx PRIVATE ''${CMAKE_CURRENT_SOURCE_DIR}/jaccl.cpp)' \
              'if(MLX_BUILD_CPU AND NOT WIN32)
      target_include_directories(mlx PRIVATE ${rdma-core.dev}/include ''${CMAKE_CURRENT_SOURCE_DIR}/lib)
      target_sources(mlx PRIVATE ''${CMAKE_CURRENT_SOURCE_DIR}/jaccl.cpp)'

          substituteInPlace mlx/distributed/jaccl/lib/CMakeLists.txt \
            --replace-fail \
              'message(STATUS "Downloading json for JACCL")
    FetchContent_Declare(
      json
      URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
    FetchContent_MakeAvailable(json)' \
              'find_package(nlohmann_json REQUIRED)' \
            --replace-fail \
              '# Check platform and SDK version requirements
    if(NOT ''${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      message(STATUS "JACCL requires macOS (Darwin). Skipping JACCL build.")
      return()
    endif()

    # Try to determine MACOS_SDK_VERSION if not set
    if(NOT DEFINED MACOS_SDK_VERSION)
      execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-version
        OUTPUT_VARIABLE MACOS_SDK_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(MACOS_SDK_VERSION)
        message(STATUS "Detected macOS SDK version: ''${MACOS_SDK_VERSION}")
      endif()
    endif()

    if(DEFINED MACOS_SDK_VERSION AND MACOS_SDK_VERSION VERSION_LESS "26.2")
      message(STATUS "JACCL requires macOS SDK >= 26.2. Skipping JACCL build.")
      return()
    endif()' \
              '# Build JACCL on the selected MLX platform; Nix provides rdma-core on Linux.' \
            --replace-fail \
              'target_include_directories(
      jaccl PRIVATE $<BUILD_INTERFACE:''${json_SOURCE_DIR}/single_include/nlohmann>)' \
              'target_link_libraries(jaccl PRIVATE nlohmann_json::nlohmann_json)' \
            --replace-fail \
              'target_compile_features(jaccl PUBLIC cxx_std_20)' \
              'target_include_directories(jaccl PRIVATE ${rdma-core.dev}/include)

    target_compile_features(jaccl PUBLIC cxx_std_20)'

          substituteInPlace mlx/distributed/jaccl/lib/jaccl/jaccl.h \
            --replace-fail \
              '#include <memory>
    #include <vector>' \
              '#include <memory>
    #include <optional>
    #include <vector>'

          substituteInPlace mlx/distributed/jaccl/lib/jaccl/rdma.cpp \
            --replace-fail \
              '#include <dlfcn.h>
    #include <unistd.h>
    #include <iostream>
    #include <sstream>' \
              '#include <dlfcn.h>
    #include <unistd.h>
    #include <cstring>
    #include <iostream>
    #include <sstream>' \
            --replace-fail \
              '  librdma_handle_ = dlopen("librdma.dylib", RTLD_NOW | RTLD_GLOBAL);' \
              '  librdma_handle_ = dlopen("${rdma-core}/lib/libibverbs.so.1", RTLD_NOW | RTLD_GLOBAL);'

          substituteInPlace mlx/distributed/jaccl/lib/jaccl/rdma.h \
            --replace-fail \
              'constexpr int BUFFER_SIZES = 8;' \
              'constexpr int BUFFER_SIZES = 9;' \
            --replace-fail \
              '#include <infiniband/verbs.h>

    #include <span>' \
              '#include <infiniband/verbs.h>

    #include <algorithm>
    #include <cstdlib>
    #include <cstring>
    #include <span>'

          substituteInPlace mlx/distributed/jaccl/lib/jaccl/rdma.h \
            --replace-fail \
              'inline std::pair<int, int64_t> buffer_size_from_message(int64_t msg) {
      if (__builtin_available(macOS 26.3, iOS 26.3, tvOS 26.3, visionOS 26.3, *)) {
        for (int k = BUFFER_SIZES - 1; k > 0; k--) {
          if (msg >= FRAME_SIZE * (1 << k)) {
            return {k, FRAME_SIZE * (1 << k)};
          }
        }
      }
      return {0, FRAME_SIZE};
    }' \
              'inline std::pair<int, int64_t> buffer_size_from_message(int64_t msg) {
      if (const char* forced = std::getenv("JACCL_BUFFER_SIZE")) {
        char* end = nullptr;
        long value = std::strtol(forced, &end, 10);
        if (end != forced && *end == 0 && value > 0) {
          for (int k = 0; k < BUFFER_SIZES; k++) {
            if (value == FRAME_SIZE * (1 << k)) {
              return {k, value};
            }
          }
        }
      }
    #if defined(__APPLE__)
      if (__builtin_available(macOS 26.3, iOS 26.3, tvOS 26.3, visionOS 26.3, *)) {
    #endif
        for (int k = BUFFER_SIZES - 1; k > 0; k--) {
          if (msg >= FRAME_SIZE * (1 << k)) {
            return {k, FRAME_SIZE * (1 << k)};
          }
        }
    #if defined(__APPLE__)
      }
    #endif
      return {0, FRAME_SIZE};
    }' \
            --replace-fail \
              '  int packet_sequence_number;
      ibv_gid global_identifier;' \
              '  int packet_sequence_number;
      int source_gid_index;
      ibv_gid global_identifier;'

          substituteInPlace mlx/distributed/jaccl/lib/jaccl/rdma.cpp \
            --replace-fail \
              '  src.local_id = -1;
    }' \
              '  src.local_id = -1;
      src.source_gid_index = 0;
    }' \
            --replace-fail \
              '  ibv_gid gid;
      for (int i = 0; i < port_attr.gid_tbl_len; i++) {
        ibv_gid tmp;
        if (ibv().query_gid(ctx, 1, i, &tmp) == 0) {
          if (*(uint64_t*)&tmp.raw[0] == 0 && *(uint16_t*)&tmp.raw[8] == 0 &&
              *(uint16_t*)&tmp.raw[10] == 0xffff) {
            gid = tmp;
            break;
          }
        }
      }

      src.local_id = port_attr.lid;' \
              '  ibv_gid gid = {};
      int gid_index = -1;
      for (int i = 0; i < port_attr.gid_tbl_len; i++) {
        ibv_gid tmp;
        if (ibv().query_gid(ctx, 1, i, &tmp) == 0) {
          if (*(uint64_t*)&tmp.raw[0] == 0 && *(uint16_t*)&tmp.raw[8] == 0 &&
              *(uint16_t*)&tmp.raw[10] == 0xffff) {
            gid = tmp;
            gid_index = i;
            break;
          }
        }
      }
      if (gid_index < 0) {
        throw std::runtime_error("[jaccl] No IPv4-mapped GID found for RDMA device");
      }

      src.local_id = port_attr.lid;' \
            --replace-fail \
              '  src.packet_sequence_number = 7;
      src.global_identifier = gid;' \
              '  src.packet_sequence_number = 7;
      src.source_gid_index = gid_index;
      src.global_identifier = gid;' \
            --replace-fail \
              '    attr.ah_attr.grh.sgid_index = 1;' \
              '    attr.ah_attr.grh.sgid_index = src.source_gid_index;' \
            --replace-fail \
              '    // Search for the name and try to open the device
        for (int i = 0; i < num_devices; i++) {
          if (name == ibv().get_device_name(devices[i])) {
            auto ctx = ibv().open_device(devices[i]);
            if (ctx == nullptr) {
              std::ostringstream msg;
              msg << "[jaccl] Could not open device " << name;
              throw std::runtime_error(msg.str());
            }
            connections.emplace_back(ctx);
            break;
          }
        }' \
              '    // Search for the name and try to open the device.
        bool found = false;
        for (int i = 0; i < num_devices; i++) {
          if (name == ibv().get_device_name(devices[i])) {
            auto ctx = ibv().open_device(devices[i]);
            if (ctx == nullptr) {
              std::ostringstream msg;
              msg << "[jaccl] Could not open device " << name;
              throw std::runtime_error(msg.str());
            }
            connections.emplace_back(ctx);
            found = true;
            break;
          }
        }
        if (!found) {
          std::ostringstream msg;
          msg << "[jaccl] Could not find RDMA device " << name;
          throw std::runtime_error(msg.str());
        }'

          substituteInPlace mlx/backend/rocm/CMakeLists.txt \
            --replace-fail \
              'if(arch MATCHES "^gfx1[12]")' \
              'if(arch MATCHES "^gfx12" OR arch MATCHES "^gfx110[0-2]$" OR arch MATCHES "^gfx115[0-3]$")' \
            --replace-fail \
              'message(STATUS "HIP include flags: ''${HIP_INCLUDE_FLAGS}")' \
              'list(APPEND HIP_INCLUDE_FLAGS "-I${rocmPackages.rocwmma}/include")
    message(STATUS "HIP include flags: ''${HIP_INCLUDE_FLAGS}")'
          substituteInPlace mlx/backend/rocm/quantized/qmm.hip \
            --replace-fail \
              'defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
        defined(__gfx1150__)' \
              'defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
        defined(__gfx1150__)'
  '';

  buildInputs =
    (old.buildInputs or [ ])
    ++ [
      rdma-core
      rdma-core.dev
    ]
    ++ (with rocmPackages; [
      clr
      rocblas
      hipblas
      hipblas-common
      hipblaslt
      composable_kernel
      rocthrust
      rocprim
      hiprand
      rocrand
      rocwmma
    ]);

  nativeBuildInputs =
    (old.nativeBuildInputs or [ ])
    ++ [
      makeWrapper
    ]
    ++ (with rocmPackages; [
      hipcc
      clang
    ]);

  env = (old.env or { }) // {
    CMAKE_ARGS = lib.concatStringsSep " " [
      ((old.env or { }).CMAKE_ARGS or "")
      "-DMLX_BUILD_METAL=OFF"
      "-DMLX_BUILD_ROCM=ON"
      "-DCMAKE_HIP_ARCHITECTURES=${gfx}"
      "-DCMAKE_C_COMPILER=${rocmPackages.clang}/bin/clang"
      "-DCMAKE_CXX_COMPILER=${rocmPackages.clang}/bin/clang++"
      "-DMLX_BUILD_TESTS=OFF"
    ];
    NIX_CFLAGS_COMPILE = ((old.env or { }).NIX_CFLAGS_COMPILE or "") + " -I${rdma-core.dev}/include";
  };

  postFixup = (old.postFixup or "") + ''
    ${patchelf}/bin/patchelf --add-rpath '$ORIGIN/../lib' "$out/${python.sitePackages}/mlx/lib64/libmlx.so"

    for bin in "$out"/bin/*; do
      [ -e "$bin" ] || continue
      [ ! -d "$bin" ] || continue
      wrapProgram "$bin" ${rocmWrapperArgs}
    done
  '';

  doCheck = false;
  doInstallCheck = false;
  pythonImportsCheck = [ "mlx.core" ];
  nativeCheckInputs = [ ];

  meta = old.meta // {
    description = "MLX with ROCm backend and portable JACCL support";
    changelog = "https://github.com/NripeshN/mlx/tree/rocm-support";
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = [ "x86_64-linux" ];
    broken = false;
  };

  passthru = (old.passthru or { }) // {
    inherit rocmRuntimeEnv;
  };
})
