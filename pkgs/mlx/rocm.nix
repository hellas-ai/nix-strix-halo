{
  lib,
  mlx,
  applyPatches,
  fetchFromGitHub,
  rdma-core,
  rocmPackages,
  gfx ? "gfx1151",
  pname ? "mlx-rocm",
}:

let
  mlx-rocm-src = applyPatches {
    src = fetchFromGitHub {
      owner = "NripeshN";
      repo = "mlx";
      rev = "39fac95d901c72175fce4baf973e375d4a054ba7";
      hash = "sha256-LZB1gY0nZO0GGxHfRGaX5uk5BOJJ89g33vPBy3x45YA=";
    };
    patches = [ ./rocm-include-rocblas.patch ];
  };
in
mlx.overrideAttrs (old: {
  inherit pname;
  src = mlx-rocm-src;

  patches = [ ];

  postPatch = (old.postPatch or "") + ''
          substituteInPlace CMakeLists.txt \
            --replace-fail \
              'FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.10.2
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL)
      FetchContent_MakeAvailable(nanobind)' \
              'find_package(nanobind CONFIG REQUIRED)'

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

          substituteInPlace mlx/distributed/jaccl/CMakeLists.txt \
            --replace-fail \
              'if(MLX_BUILD_CPU
       AND ''${CMAKE_SYSTEM_NAME} MATCHES "Darwin"
       AND MACOS_SDK_VERSION GREATER_EQUAL 26.2)' \
              'if(MLX_BUILD_CPU AND NOT WIN32)'
          substituteInPlace mlx/distributed/jaccl/CMakeLists.txt \
            --replace-fail \
              'if(MLX_BUILD_CPU AND NOT WIN32)
      target_sources(' \
              'if(MLX_BUILD_CPU AND NOT WIN32)
      target_include_directories(mlx PRIVATE ${rdma-core.dev}/include)
      target_sources('

          substituteInPlace mlx/distributed/jaccl/utils.cpp \
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

          substituteInPlace mlx/distributed/jaccl/utils.h \
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

          substituteInPlace mlx/distributed/jaccl/utils.h \
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

          substituteInPlace mlx/distributed/jaccl/utils.cpp \
            --replace-fail \
              '  src.local_id = -1;
    }' \
              '  src.local_id = -1;
      src.source_gid_index = 0;
    }' \
            --replace-fail \
              '  ibv_gid gid;
      ibv().query_gid(ctx, 1, 1, &gid);

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
              '  src.packet_sequence_number = 7; // TODO: Change to sth random
      src.global_identifier = gid;' \
              '  src.packet_sequence_number = 7; // TODO: Change to sth random
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

          substituteInPlace mlx/distributed/jaccl/mesh.h \
            --replace-fail \
              '#include "mlx/distributed/distributed_impl.h"' \
              '#include <functional>

    #include "mlx/distributed/distributed_impl.h"' \
            --replace-fail \
              '      Stream stream,
          ReduceOp reduce_op);' \
              '      Stream stream,
          ReduceOp reduce_op,
          std::function<void()> recv_ready);'

          substituteInPlace mlx/distributed/jaccl/mesh.cpp \
            --replace-fail \
              '    all_reduce<T>(input, output, stream, detail::SumOp<T>{});' \
              '    all_reduce<T>(
            input, output, stream, detail::SumOp<T>{}, [&] {
              side_channel_.all_gather<int>(0);
            });' \
            --replace-fail \
              '    all_reduce<T>(input, output, stream, detail::MaxOp<T>{});' \
              '    all_reduce<T>(
            input, output, stream, detail::MaxOp<T>{}, [&] {
              side_channel_.all_gather<int>(0);
            });' \
            --replace-fail \
              '    all_reduce<T>(input, output, stream, detail::MinOp<T>{});' \
              '    all_reduce<T>(
            input, output, stream, detail::MinOp<T>{}, [&] {
              side_channel_.all_gather<int>(0);
            });' \
            --replace-fail \
              '    Stream stream,
        ReduceOp reduce_op) {' \
              '    Stream stream,
        ReduceOp reduce_op,
        std::function<void()> recv_ready) {' \
            --replace-fail \
              '  encoder.dispatch([in_ptr, out_ptr, size, this, reduce_op]() {' \
              '  encoder.dispatch([in_ptr, out_ptr, size, this, reduce_op, recv_ready]() {' \
            --replace-fail \
              '      mesh_.all_reduce(in_ptr, out_ptr, size, reduce_op);' \
              '      mesh_.all_reduce(in_ptr, out_ptr, size, reduce_op, recv_ready);'

          substituteInPlace mlx/distributed/jaccl/mesh_impl.h \
            --replace-fail \
              '#include <span>' \
              '#include <functional>
    #include <span>' \
            --replace-fail \
              '  all_reduce(const T* in_ptr, T* out_ptr, int64_t size, ReduceOp reduce_op) {' \
              '  all_reduce(
          const T* in_ptr,
          T* out_ptr,
          int64_t size,
          ReduceOp reduce_op,
          std::function<void()> recv_ready) {' \
            --replace-fail \
              '    // Prefill the pipeline
        int buff = 0;
        while (read_offset < total && buff < PIPELINE) {
          post_recv_all(sz, buff);
          std::copy(
              data + read_offset,
              data + std::min(read_offset + N, total),
              send_buffer(sz, buff).begin<T>());
          post_send_all(sz, buff);

          buff++;
          in_flight += 2 * num_peers;
          read_offset += N;
        }

        // Main loop' \
              '    // Prefill the pipeline
        int buff = 0;
        int preposted = 0;
        while (read_offset < total && buff < PIPELINE) {
          post_recv_all(sz, buff);
          std::copy(
              data + read_offset,
              data + std::min(read_offset + N, total),
              send_buffer(sz, buff).begin<T>());

          buff++;
          preposted++;
          in_flight += num_peers;
          read_offset += N;
        }

        recv_ready();

        for (int b = 0; b < preposted; b++) {
          post_send_all(sz, b);
          in_flight += num_peers;
        }

        // Main loop'

          for f in mlx/distributed/jaccl/mesh.cpp mlx/distributed/jaccl/ring.cpp; do
            substituteInPlace "$f" \
              --replace-fail \
                '  side_channel_.all_gather<int>(0);' \
                '  side_channel_.all_gather<int>(0);
      side_channel_.all_gather<int>(0);'
          done

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
        defined(__gfx1150__)' \
            --replace-fail \
              '      } else if (std::strstr(arch_name, "gfx11") != nullptr) {
            hw.tier = RocmQmvArchTier::Rdna3;
            hw.simds_per_cu = 2;
            hw.has_wmma = true;
          } else if (std::strstr(arch_name, "gfx10") != nullptr) {' \
              '      } else if (std::strstr(arch_name, "gfx1103") != nullptr) {
            hw.tier = RocmQmvArchTier::Rdna3;
            hw.simds_per_cu = 2;
            hw.has_wmma = false;
          } else if (std::strstr(arch_name, "gfx11") != nullptr) {
            hw.tier = RocmQmvArchTier::Rdna3;
            hw.simds_per_cu = 2;
            hw.has_wmma = true;
          } else if (std::strstr(arch_name, "gfx10") != nullptr) {'
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

  doCheck = false;
  doInstallCheck = false;
  pythonImportsCheck = [ ];
  nativeCheckInputs = [ ];

  meta = old.meta // {
    description = "MLX with ROCm backend and portable JACCL support";
    changelog = "https://github.com/NripeshN/mlx/commit/39fac95d901c72175fce4baf973e375d4a054ba7";
    maintainers = with lib.maintainers; [ georgewhewell ];
    platforms = [ "x86_64-linux" ];
    broken = false;
  };
})
