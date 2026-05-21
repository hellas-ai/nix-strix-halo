{
  lib,
  stdenvNoCC,
  stdenv,
  fetchurl,
  patchelf,
  libdrm,
  numactl,
  rdma-core,
  target ? "gfx1151",
  version ? "7.13.0a20260515",
  url ? "https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-${target}-${version}.tar.gz",
  hash,
}:

stdenvNoCC.mkDerivation {
  pname = "therock-rocm-sdk-${target}";
  inherit version;

  src = fetchurl {
    inherit url hash;
  };

  nativeBuildInputs = [
    patchelf
  ];

  propagatedBuildInputs = [
    libdrm
    numactl
    rdma-core
  ];

  dontConfigure = true;
  dontBuild = true;
  dontPatchELF = true;
  dontStrip = true;

  unpackPhase = ''
    runHook preUnpack
    tar -xzf "$src"
    runHook postUnpack
  '';

  installPhase = ''
        runHook preInstall

        mkdir -p "$out"
        shopt -s dotglob nullglob

        if [ -d install ]; then
          cp -R install/* "$out/"
        else
          entries=(*/)
          if [ "''${#entries[@]}" -eq 1 ] && [ -d "''${entries[0]}install" ]; then
            cp -R "''${entries[0]}install/"* "$out/"
          else
            cp -R ./* "$out/"
          fi
        fi

        chmod -R u+w "$out"
        find "$out" -type f -name "*.so*" -exec chmod 755 {} \; 2>/dev/null || true
        find "$out/bin" "$out/llvm/bin" -type f -exec chmod 755 {} \; 2>/dev/null || true

        # TheRock binary SDKs are built as relocatable Linux tarballs, not Nix
        # derivations. Make the shipped ELF binaries/libraries usable in pure Nix
        # build sandboxes so CMake probes such as FindHIP can execute hipconfig.
        gcc_runtime=${lib.escapeShellArg "${stdenv.cc.cc.lib}/lib"}
        libdrm_runtime=${lib.escapeShellArg "${lib.getLib libdrm}/lib"}
        numa_runtime=${lib.escapeShellArg "${lib.getLib numactl}/lib"}
        rdma_runtime=${lib.escapeShellArg "${lib.getLib rdma-core}/lib"}
        dynamic_linker=${lib.escapeShellArg stdenv.cc.bintools.dynamicLinker}
        runtime_paths=(
          "$gcc_runtime"
          "$libdrm_runtime"
          "$numa_runtime"
          "$rdma_runtime"
        )

        patch_elf() {
          local elf="$1"
          if old_rpath=$(patchelf --print-rpath "$elf" 2>/dev/null); then
            new_rpath="$old_rpath"
            for runtime_path in "''${runtime_paths[@]}"; do
              case ":$new_rpath:" in
                *":$runtime_path:"*) ;;
                *) new_rpath="''${new_rpath:+$new_rpath:}$runtime_path" ;;
              esac
            done
            patchelf --set-rpath "$new_rpath" "$elf" 2>/dev/null || true
          fi

          if patchelf --print-interpreter "$elf" >/dev/null 2>&1; then
            patchelf --set-interpreter "$dynamic_linker" "$elf" 2>/dev/null || true
          fi
        }

        for dir in "$out/bin" "$out/llvm/bin" "$out/lib/llvm/bin"; do
          [ -d "$dir" ] || continue
          for tool in \
            hipcc hipconfig hipcc_cmake_linker_helper \
            rocminfo rocm_agent_enumerator rocm-smi amd-smi \
            amdclang amdclang++ clang clang++ clang-offload-bundler \
            lld ld.lld llvm-config llvm-ar llvm-ranlib llvm-link llvm-objcopy \
            rccl-UnitTests rcclras rocshmem_info rocshmem_functional_tests; do
            [ -f "$dir/$tool" ] || continue
            patch_elf "$dir/$tool"
          done
        done

        for dir in \
          "$out/lib" \
          "$out/lib/rocm_sysdeps/lib" \
          "$out/lib/host-math/lib" \
          "$out/llvm/lib" \
          "$out/lib/llvm/lib" \
          "$out/share/amd_smi/amdsmi"; do
          [ -d "$dir" ] || continue
          while IFS= read -r -d "" elf; do
            patch_elf "$elf"
          done < <(find "$dir" -maxdepth 1 -type f \( -name "*.so" -o -name "*.so.*" \) -print0)
        done

        cc_cflags_before="$(cat ${stdenv.cc}/nix-support/cc-cflags-before)"
        cc_cflags="$(cat ${stdenv.cc}/nix-support/cc-cflags)"
        libc_cflags="$(cat ${stdenv.cc}/nix-support/libc-cflags)"
        libc_crt1_cflags="$(cat ${stdenv.cc}/nix-support/libc-crt1-cflags)"
        libc_ldflags="$(cat ${stdenv.cc}/nix-support/libc-ldflags)"
        cc_ldflags="$(cat ${stdenv.cc}/nix-support/cc-ldflags)"
        libc="$(cat ${stdenv.cc}/nix-support/orig-libc)"
        dynamic_linker="$(cat ${stdenv.cc}/nix-support/dynamic-linker)"

        cat > "$out/bin/therock-hip-clang++" <<EOF
        #!/bin/sh
        exec "$out/lib/llvm/bin/clang++" \\
          --gcc-toolchain=${stdenv.cc.cc} \\
          --rocm-path="$out" \\
          $cc_cflags_before \\
          $cc_cflags \\
          $libc_cflags \\
          $libc_crt1_cflags \\
          "\$@" \\
          -L$libc/lib \\
          -Wl,--dynamic-linker=$dynamic_linker \\
          -Wl,-rpath,${stdenv.cc.cc.lib}/lib \\
          -Wl,-rpath,$libc/lib \\
          $libc_ldflags \\
          $cc_ldflags
    EOF
        sed -i 's/^    //' "$out/bin/therock-hip-clang++"
        chmod 755 "$out/bin/therock-hip-clang++"

        # TheRock's CMake HIP helper normally re-enters hipcc, which then invokes
        # the raw bundled clang++ during link steps. That loses Nix's libc/libstdc++
        # search paths and fails to link extension modules in pure builds. Keep the
        # CMake contract (first argv is HIP_CLANG_PATH), but route the actual link
        # through the Nix-aware compiler wrapper above.
        cat > "$out/bin/hipcc_cmake_linker_helper" <<EOF
        #!/bin/sh
        [ "\$#" -gt 0 ] && shift
        exec "$out/bin/therock-hip-clang++" "\$@"
    EOF
        sed -i 's/^    //' "$out/bin/hipcc_cmake_linker_helper"
        chmod 755 "$out/bin/hipcc_cmake_linker_helper"

        runHook postInstall
  '';

  passthru = {
    inherit target;
  };

  meta = {
    description = "TheRock ROCm SDK binary tarball for ${target}";
    homepage = "https://github.com/ROCm/TheRock";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
