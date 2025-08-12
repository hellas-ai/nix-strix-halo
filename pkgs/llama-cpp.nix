# Llama.cpp derivation with ROCm support
{ pkgs, llama-cpp, mkRocwmmaDerivation }:

{
  target,
  rocm,
  enableRocwmma ? false,
  enableHipBlasLt ? false,
}:

let
  amdgpuTargets =
    if target == "gfx110X"
    then "gfx1100"
    else if target == "gfx1151"
    then "gfx1151"
    else if target == "gfx120X"
    then "gfx1200;gfx1201"
    else target;

  # Create a wrapper lib directory with libgcc linker script
  gccLibWrapper = pkgs.runCommand "gcc-lib-wrapper" {} ''
    mkdir -p $out/lib
    # Create a linker script that redirects to libgcc_s
    cat > $out/lib/libgcc.so << EOF
    /* GNU ld script */
    GROUP ( ${pkgs.stdenv.cc.cc.libgcc}/lib/libgcc_s.so.1 )
    EOF
    # Copy actual libs
    cp -L ${pkgs.stdenv.cc.cc.libgcc}/lib/* $out/lib/ 2>/dev/null || true
  '';
  
  # Simple hipcc wrapper for compilation
  # For ROCWMMA builds, we need to use ROCm's clang++ for AMD GPU builtins
  hipccWrapper = if enableRocwmma then
    pkgs.writeScriptBin "hipcc" ''
      #!${pkgs.bash}/bin/bash
      has_source=false
      for arg in "$@"; do
        case "$arg" in
          *.c|*.cpp|*.cxx|*.cc|*.cu) has_source=true; break ;;
        esac
      done
      
      if [[ "$has_source" == "true" ]]; then
        exec ${rocm}/llvm/bin/clang++ \
          -x hip \
          -D__HIP_PLATFORM_AMD__ \
          --rocm-path=${rocm} \
          --rocm-device-lib-path=${rocm}/lib/llvm/amdgcn/bitcode \
          -isystem ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.stdenv.cc.cc.version} \
          -isystem ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.stdenv.cc.cc.version}/x86_64-unknown-linux-gnu \
          -isystem ${pkgs.glibc.dev}/include \
          -I${rocm}/include \
          -I${rocm}/include/hip \
          -I${mkRocwmmaDerivation { inherit rocm target; }}/include \
          -B${pkgs.glibc}/lib \
          -L${rocm}/lib \
          -L${pkgs.stdenv.cc.cc.lib}/lib \
          -L${gccLibWrapper}/lib \
          -L${pkgs.glibc}/lib \
          "$@"
      else
        exec ${rocm}/llvm/bin/clang++ \
          -isystem ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.stdenv.cc.cc.version} \
          -isystem ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.stdenv.cc.cc.version}/x86_64-unknown-linux-gnu \
          -isystem ${pkgs.glibc.dev}/include \
          -B${pkgs.glibc}/lib \
          -L${rocm}/lib \
          -L${pkgs.stdenv.cc.cc.lib}/lib \
          -L${gccLibWrapper}/lib \
          -L${pkgs.glibc}/lib "$@"
      fi
    ''
  else
    pkgs.writeScriptBin "hipcc" ''
      #!${pkgs.bash}/bin/bash
      has_source=false
      for arg in "$@"; do
        case "$arg" in
          *.c|*.cpp|*.cxx|*.cc|*.cu) has_source=true; break ;;
        esac
      done

      if [[ "$has_source" == "true" ]]; then
        exec ${pkgs.clang}/bin/clang++ \
          -x hip \
          -D__HIP_PLATFORM_AMD__ \
          --rocm-path=${rocm} \
          --rocm-device-lib-path=${rocm}/lib/llvm/amdgcn/bitcode \
          -I${rocm}/include \
          -I${rocm}/include/hip \
          -L${rocm}/lib \
          "$@"
      else
        exec ${pkgs.clang}/bin/clang++ -L${rocm}/lib "$@"
      fi
    '';
in
  pkgs.stdenv.mkDerivation {
    pname = "llama-cpp-rocm-${target}${pkgs.lib.optionalString enableRocwmma "-rocwmma"}";
    version = "git";
    hardeningDisable = ["all"];
    src = llama-cpp;

    patches = pkgs.lib.optionals enableRocwmma [
      ../patches/rocwmma-compatibility.patch
      ../patches/hip-version-fix.patch
    ];

    nativeBuildInputs = with pkgs; [
      cmake
      ninja
      pkg-config
      hipccWrapper
      lld
      llvmPackages.bintools
      autoPatchelfHook
    ];

    buildInputs = with pkgs; [
      curl
      rocm
      stdenv.cc.cc.lib
    ] ++ pkgs.lib.optionals enableRocwmma [
      (mkRocwmmaDerivation { inherit rocm target; })
    ];

    postPatch = ''
      substituteInPlace ggml/src/ggml-cuda/vendors/hip.h \
        --replace "HIP_VERSION >= 70000000" "HIP_VERSION >= 50600000"
    '' + pkgs.lib.optionalString enableRocwmma ''
      # Apply ROCWMMA compatibility fixes (following amd-strix-halo-toolboxes approach)
      # Replace hardcoded warp masks with GGML_HIP_WARP_MASK macro for ROCWMMA compatibility
      find ggml/src/ggml-cuda -name "*.cu" -o -name "*.cuh" | while read file; do
        sed -i 's/0xFFFFFFFF/GGML_HIP_WARP_MASK/g; s/0xffffffff/GGML_HIP_WARP_MASK/g' "$file" || true
      done
    '';

    cmakeFlags = [
      "-G Ninja"
      "-DCMAKE_C_COMPILER=${pkgs.clang}/bin/clang"
      "-DCMAKE_CXX_COMPILER=${hipccWrapper}/bin/hipcc"
      "-DCMAKE_BUILD_TYPE=Release"
      "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY"
      "-DAMDGPU_TARGETS=${amdgpuTargets}"
      "-DBUILD_SHARED_LIBS=ON"
      "-DLLAMA_BUILD_SERVER=ON"
      "-DLLAMA_BUILD_TESTS=OFF"
      "-DGGML_HIP=ON"
      "-DGGML_RPC=ON"
      "-DLLAMA_CURL=ON"
      "-DGGML_NATIVE=OFF"
    ] ++ pkgs.lib.optionals enableRocwmma [
      "-DGGML_HIP_ROCWMMA_FATTN=ON"
      "-DLLAMA_HIP_UMA=ON"
    ];

    preConfigure = ''
      export HIP_PATH="${rocm}"
      export ROCM_PATH="${rocm}"
      export HIP_PLATFORM="amd"
      export HIP_DEVICE_LIB_PATH="${rocm}/lib/llvm/amdgcn/bitcode"
      export PATH="${rocm}/bin:$PATH"
    '' + pkgs.lib.optionalString enableHipBlasLt ''
      export ROCBLAS_USE_HIPBLASLT=1
    '';

    postInstall = ''
      # Install rpc-server which isn't installed by default
      cp bin/rpc-server $out/bin/
      # Fix RPATH for rpc-server
      patchelf --set-rpath "$out/lib:${rocm}/lib:${pkgs.stdenv.cc.cc.lib}/lib" $out/bin/rpc-server
    '' + pkgs.lib.optionalString enableRocwmma ''
      # Wrap all binaries to set ROCBLAS_USE_HIPBLASLT=1 for ROCWMMA builds
      for bin in $out/bin/*; do
        if [ -f "$bin" ] && [ -x "$bin" ]; then
          mv "$bin" "$bin.unwrapped"
          cat > "$bin" << EOF
      #!/bin/bash
      # Set ROCBLAS_USE_HIPBLASLT=1 unless explicitly set to 0
      if [ "\''${ROCBLAS_USE_HIPBLASLT}" != "0" ]; then
        export ROCBLAS_USE_HIPBLASLT=1
      fi
      exec "$bin.unwrapped" "\$@"
      EOF
          chmod +x "$bin"
        fi
      done
    '';

    meta = with pkgs.lib; {
      description = "Llama.cpp with ROCm support for ${target}";
      homepage = "https://github.com/ggerganov/llama.cpp";
      license = licenses.mit;
      platforms = ["x86_64-linux"];
    };
  }