{
  lib,
  stdenv,
  runCommand,
  fetchzip,
  cmake,
  pkg-config,
  perl,
  rdma-core ? null,
  apple-sdk_26 ? null,
  mlx-src,
  darwinSdk ? apple-sdk_26,
  darwinSdkRoot ? if darwinSdk != null then darwinSdk.sdkroot else null,
  darwinSdkVersion ? "26.5",
  darwinDeploymentTarget ? "26.0",
}:

let
  json-src = fetchzip {
    url = "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz";
    hash = "sha256-cnGfiVhXzqfj5Fay823wntWcTnbh8r2SefDLslb1Dh0=";
  };

  linuxLibibverbsName =
    if stdenv.hostPlatform.isLinux then
      assert rdma-core != null;
      "${rdma-core}/lib/libibverbs.so.1"
    else
      "libibverbs.so.1";

  jaccl-src = runCommand "jaccl-src" { nativeBuildInputs = [ perl ]; } ''
        cp -r ${mlx-src}/mlx/distributed/jaccl/lib "$out"
        chmod -R u+w "$out"

        replace_if_present() {
          local file="$1"
          local from="$2"
          local to="$3"

          FROM="$from" TO="$to" perl -0pi -e '
            BEGIN {
              $from = $ENV{"FROM"};
              $to = $ENV{"TO"};
            }
            s/\Q$from\E/$to/g;
          ' "$file"
        }

        replace_if_present "$out/CMakeLists.txt" \
          'if(NOT ''${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      message(STATUS "JACCL requires macOS (Darwin). Skipping JACCL build.")
      return()
    endif()

    # Try to determine MACOS_SDK_VERSION if not set' \
          'if(''${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # Try to determine MACOS_SDK_VERSION if not set'

        replace_if_present "$out/CMakeLists.txt" \
          'if(DEFINED MACOS_SDK_VERSION AND MACOS_SDK_VERSION VERSION_LESS "26.2")
      message(STATUS "JACCL requires macOS SDK >= 26.2. Skipping JACCL build.")
      return()
    endif()

    add_library(' \
          'if(DEFINED MACOS_SDK_VERSION AND MACOS_SDK_VERSION VERSION_LESS "26.2")
      message(STATUS "JACCL requires macOS SDK >= 26.2. Skipping JACCL build.")
      return()
    endif()
    endif()

    add_library('

        replace_if_present "$out/jaccl/rdma.cpp" \
          'librdma_handle_ = dlopen("librdma.dylib", RTLD_NOW | RTLD_GLOBAL);' \
          '#if defined(__APPLE__)
      const char* librdma_name = "librdma.dylib";
    #else
      const char* librdma_name = "${linuxLibibverbsName}";
    #endif

      librdma_handle_ = dlopen(librdma_name, RTLD_NOW | RTLD_GLOBAL);'
        replace_if_present "$out/jaccl/rdma.cpp" \
          '  const char* librdma_name = "libibverbs.so.1";' \
          '  const char* librdma_name = "${linuxLibibverbsName}";'

        replace_if_present "$out/jaccl/jaccl.h" '#include <memory>' '#include <memory>
    #include <optional>'
        replace_if_present "$out/jaccl/mesh_impl.h" '#include <memory>' '#include <algorithm>
    #include <cstring>
    #include <memory>'
        replace_if_present "$out/jaccl/ring_impl.h" '#include <span>' '#include <algorithm>
    #include <cstring>
    #include <span>'

        if ! grep -Fq '#include <algorithm>' "$out/jaccl/rdma.h"; then
          perl -0pi -e 's@#include <infiniband/verbs.h>\n@#include <infiniband/verbs.h>\n\n#include <algorithm>\n@' "$out/jaccl/rdma.h"
        fi
        if ! grep -Fq '#include <cstdlib>' "$out/jaccl/rdma.h"; then
          perl -0pi -e 's@#include <algorithm>\n@#include <algorithm>\n#include <cstdlib>\n@' "$out/jaccl/rdma.h"
        fi

        replace_if_present "$out/jaccl/rdma.h" \
          '  if (__builtin_available(macOS 26.3, iOS 26.3, tvOS 26.3, visionOS 26.3, *)) {' \
          '#if defined(__APPLE__)
      if (__builtin_available(macOS 26.3, iOS 26.3, tvOS 26.3, visionOS 26.3, *)) {'
        replace_if_present "$out/jaccl/rdma.h" \
          '  }
      return {0, FRAME_SIZE};
    }' \
          '  }
    #else
      for (int k = BUFFER_SIZES - 1; k > 0; k--) {
        if (msg >= FRAME_SIZE * (1 << k)) {
          return {k, FRAME_SIZE * (1 << k)};
        }
      }
    #endif
      return {0, FRAME_SIZE};
    }'

        replace_if_present "$out/jaccl/rdma.h" \
          'constexpr int BUFFER_SIZES = 8;' \
          'constexpr int BUFFER_SIZES = 9;'
        replace_if_present "$out/jaccl/rdma.h" \
          'inline std::pair<int, int64_t> buffer_size_from_message(int64_t msg) {' \
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
      }'
        replace_if_present "$out/jaccl/rdma.h" \
          '  int packet_sequence_number;
      ibv_gid global_identifier;' \
          '  int packet_sequence_number;
      int source_gid_index;
      ibv_gid global_identifier;'

        replace_if_present "$out/jaccl/rdma.cpp" \
          '  src.local_id = -1;
    }' \
          '  src.local_id = -1;
      src.source_gid_index = 0;
    }'
        replace_if_present "$out/jaccl/rdma.cpp" \
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

      src.local_id = port_attr.lid;'
        replace_if_present "$out/jaccl/rdma.cpp" \
          '  src.packet_sequence_number = 7;
      src.global_identifier = gid;' \
          '  src.packet_sequence_number = 7;
      src.source_gid_index = gid_index;
      src.global_identifier = gid;'
        replace_if_present "$out/jaccl/rdma.cpp" \
          '    attr.ah_attr.grh.sgid_index = 1;' \
          '    attr.ah_attr.grh.sgid_index = src.source_gid_index;'
        replace_if_present "$out/jaccl/rdma.cpp" \
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

        replace_if_present "$out/jaccl/mesh.cpp" \
          '    mesh_.all_reduce(in_ptr, out_ptr, count, reduce_op);' \
          '    mesh_.all_reduce(
            in_ptr, out_ptr, count, reduce_op, [&] { side_channel_.barrier(); });'
        replace_if_present "$out/jaccl/mesh_impl.h" \
          '  template <typename T, typename ReduceOp>
      void all_reduce(const T* in, T* out, int64_t size, ReduceOp reduce_op) {
        // Fully connected all reduce with deterministic reduction order.' \
          '  template <typename T, typename ReduceOp>
      void all_reduce(const T* in, T* out, int64_t size, ReduceOp reduce_op) {
        all_reduce(in, out, size, reduce_op, [] {});
      }

      template <typename T, typename ReduceOp, typename RecvReady>
      void all_reduce(
          const T* in,
          T* out,
          int64_t size,
          ReduceOp reduce_op,
          RecvReady recv_ready) {
        // Fully connected all reduce with deterministic reduction order.'
    replace_if_present "$out/jaccl/mesh_impl.h" \
      '    int reduce_rank = 0;

    // Total number of chunks
    int64_t total_chunks = (total + N - 1) / N;

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < total && buff < PIPELINE) {' \
      '    int reduce_rank = 0;

    // Total number of chunks
    int64_t total_chunks = (total + N - 1) / N;

    // Prefill the pipeline
    int buff = 0;
    int preposted = 0;
    while (read_offset < total && buff < PIPELINE) {'
    replace_if_present "$out/jaccl/mesh_impl.h" \
      '      recv_end[rank_]++;
      post_send_all(sz, buff);

      buff++;
      in_flight += 2 * num_peers;
      read_offset += N;
    }

    // Main loop' \
      '      recv_end[rank_]++;
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
  '';
in
stdenv.mkDerivation {
  pname = "jaccl";
  version = "0.32.0-mlx";
  src = jaccl-src;

  nativeBuildInputs = [
    cmake
    pkg-config
  ];

  buildInputs =
    lib.optionals stdenv.hostPlatform.isLinux [
      rdma-core
      rdma-core.dev
    ]
    ++ lib.optionals stdenv.hostPlatform.isDarwin [ darwinSdk ];

  cmakeFlags = [
    "-DFETCHCONTENT_SOURCE_DIR_JSON=${json-src}"
  ]
  ++ lib.optionals stdenv.hostPlatform.isDarwin [
    "-DMACOS_SDK_VERSION=${darwinSdkVersion}"
  ];

  preConfigure = lib.optionalString stdenv.hostPlatform.isDarwin ''
    if [ ! -d "${darwinSdkRoot}" ]; then
      echo "jaccl: missing macOS SDK: ${darwinSdkRoot}" >&2
      exit 1
    fi
    export SDKROOT=${lib.escapeShellArg darwinSdkRoot}
    export MACOSX_DEPLOYMENT_TARGET=${lib.escapeShellArg darwinDeploymentTarget}
  '';

  meta = with lib; {
    description = "JACCL RDMA-over-Thunderbolt collective communication library from MLX";
    homepage = "https://github.com/ml-explore/mlx";
    license = licenses.mit;
    maintainers = with maintainers; [ georgewhewell ];
    platforms = platforms.linux ++ [ "aarch64-darwin" ];
  };
}
