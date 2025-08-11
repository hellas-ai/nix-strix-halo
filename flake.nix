{
  description = "Llama.cpp with pre-built ROCm binaries from TheRock";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    llama-cpp = {
      url = "github:ggerganov/llama.cpp";
      flake = false;
    };
    rocwmma = {
      url = "github:ROCm/rocWMMA";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    llama-cpp,
    rocwmma,
  }: let
    rocmSources = builtins.fromJSON (builtins.readFile ./rocm-sources.json);

    # ROCWMMA derivation builder - builds from source against ROCm 7
    mkRocwmmaDerivation = pkgs: rocm: target: 
      pkgs.stdenv.mkDerivation rec {
        pname = "rocwmma";
        version = "main";
        
        src = rocwmma;
        
        nativeBuildInputs = with pkgs; [
          cmake
          ninja
          lld
          llvmPackages.bintools
          autoPatchelfHook
        ];
        
        buildInputs = with pkgs; [
          stdenv.cc.cc.lib
        ];
        
        # Change FP8 check from FATAL_ERROR to STATUS (as done in amd-strix-halo-toolboxes)
        postPatch = ''
          substituteInPlace CMakeLists.txt \
            --replace 'message(FATAL_ERROR "The detected ROCm does not support data type' \
                      'message(STATUS "The detected ROCm does not support data type'
        '';
        
        # Build ROCWMMA from source using nixpkgs clang (like llama.cpp)
        # This avoids linking issues with ROCm's clang in Nix environment
        cmakeFlags = [
          "-G Ninja"
          "-DCMAKE_C_COMPILER=${pkgs.clang}/bin/clang"
          "-DCMAKE_CXX_COMPILER=${pkgs.clang}/bin/clang++"
          "-DCMAKE_BUILD_TYPE=Release"
          "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}"
          "-DCMAKE_CROSSCOMPILING=ON"
          "-DROCWMMA_BUILD_TESTS=OFF"
          "-DROCWMMA_BUILD_SAMPLES=OFF"
          "-DGPU_TARGETS=${target}"
          # Point to ROCm for HIP headers and libs
          "-DHIP_ROOT_DIR=${rocm}"
          "-DHIP_PATH=${rocm}"
          # Tell CMake where to find pthreads
          "-DCMAKE_THREAD_LIBS_INIT=-pthread"
          "-DCMAKE_HAVE_THREADS_LIBRARY=1"
          "-DCMAKE_USE_PTHREADS_INIT=1"
          "-DCMAKE_USE_PTHREADS=ON"
          "-DTHREADS_PREFER_PTHREAD_FLAG=ON"
          # Tell CMake where to find OpenMP (using ROCm's libomp)
          "-DOpenMP_C_INCLUDE_DIR=${rocm}/llvm/include"
          "-DOpenMP_CXX_INCLUDE_DIR=${rocm}/llvm/include"
          "-DOpenMP_CXX_FLAGS=-fopenmp"
          "-DOpenMP_CXX_LIB_NAMES=omp"
          "-DOpenMP_omp_LIBRARY=${rocm}/llvm/lib/libomp.so"
        ];
        
        preConfigure = ''
          export HIP_PATH="${rocm}"
          export ROCM_PATH="${rocm}"
          export HIP_PLATFORM="amd"
          export PATH="${rocm}/bin:$PATH"
          export HIP_CLANG_PATH="${pkgs.clang}/bin"
        '';
      };

    # ROCm derivation builder
    mkRocmDerivation = pkgs: target: let
      source = rocmSources.linux.${target};
    in
      if source.url == ""
      then throw "ROCm sources not populated for ${target}. Run: python3 update-rocm.py"
      else
        pkgs.stdenv.mkDerivation {
              pname = "rocm-prebuilt-${target}";
              version = source.version;

              src = pkgs.fetchurl {
                url = source.url;
                sha256 = source.sha256;
              };

              dontBuild = true;
              dontConfigure = true;

              nativeBuildInputs = with pkgs; [
                gnutar
                gzip
                autoPatchelfHook
              ];

              buildInputs = with pkgs; [
                stdenv.cc.cc.lib
                gfortran.cc.lib
                zlib
                ncurses
              ];

              autoPatchelfIgnoreMissingDeps = [
                "libtest_linking_lib1.so"
                "libtest_linking_lib2.so"
              ];

              unpackPhase = "tar -xzf $src";

              installPhase = ''
                mkdir -p $out
                cp -r * $out/
                chmod -R u+w $out
                find $out -type f -name "*.so*" -exec chmod 755 {} \; 2>/dev/null || true
                find $out/bin -type f -exec chmod 755 {} \; 2>/dev/null || true
              '';

              meta = with pkgs.lib; {
                description = "Pre-built ROCm binaries from TheRock for ${target}";
                homepage = "https://github.com/ROCm/TheRock";
                license = licenses.mit;
                platforms = ["x86_64-linux"];
              };
        };

    # Llama.cpp derivation with ROCm support
    mkLlamaCppDerivation = pkgs: {
      target,
      rocm,
      enableRocwmma ? false,
      enableHipBlasLt ? false,
    }: let
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
                  -I${mkRocwmmaDerivation pkgs rocm target}/include \
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
              ./patches/rocwmma-compatibility.patch
              ./patches/hip-version-fix.patch
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
              (mkRocwmmaDerivation pkgs rocm target)
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
            '';

            meta = with pkgs.lib; {
              description = "Llama.cpp with ROCm support for ${target}";
              homepage = "https://github.com/ggerganov/llama.cpp";
              license = licenses.mit;
              platforms = ["x86_64-linux"];
            };
        };
  in
    # Overlay output
    {
      overlays.default = final: prev: {
        llamacpp-rocm = {
          # ROCm toolchain packages
          rocm-gfx110X = mkRocmDerivation prev "gfx110X";
          rocm-gfx1151 = mkRocmDerivation prev "gfx1151";
          rocm-gfx120X = mkRocmDerivation prev "gfx120X";
          
          # Standard Llama.cpp packages
          gfx110X = mkLlamaCppDerivation prev {
            target = "gfx110X";
            rocm = mkRocmDerivation prev "gfx110X";
          };
          gfx1151 = mkLlamaCppDerivation prev {
            target = "gfx1151";
            rocm = mkRocmDerivation prev "gfx1151";
          };
          gfx120X = mkLlamaCppDerivation prev {
            target = "gfx120X";
            rocm = mkRocmDerivation prev "gfx120X";
          };
          
          # ROCWMMA-enabled packages
          gfx110X-rocwmma = mkLlamaCppDerivation prev {
            target = "gfx110X";
            rocm = mkRocmDerivation prev "gfx110X";
            enableRocwmma = true;
            enableHipBlasLt = true;
          };
          gfx1151-rocwmma = mkLlamaCppDerivation prev {
            target = "gfx1151";
            rocm = mkRocmDerivation prev "gfx1151";
            enableRocwmma = true;
            enableHipBlasLt = true;
          };
          gfx120X-rocwmma = mkLlamaCppDerivation prev {
            target = "gfx120X";
            rocm = mkRocmDerivation prev "gfx120X";
            enableRocwmma = true;
            enableHipBlasLt = true;
          };
        };
      };

      # NixOS modules
      nixosModules = {
        rpc-server = import ./modules/rpc-server.nix;
        default = import ./modules/rpc-server.nix;
      };
    }
    // 
    # Per-system outputs
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        packages = {
          # ROCm packages
          rocm-gfx110X = mkRocmDerivation pkgs "gfx110X";
          rocm-gfx1151 = mkRocmDerivation pkgs "gfx1151";
          rocm-gfx120X = mkRocmDerivation pkgs "gfx120X";

          # Standard llama.cpp packages
          llama-cpp-gfx110X = mkLlamaCppDerivation pkgs {
            target = "gfx110X";
            rocm = self.packages.${system}.rocm-gfx110X;
          };

          llama-cpp-gfx1151 = mkLlamaCppDerivation pkgs {
            target = "gfx1151";
            rocm = self.packages.${system}.rocm-gfx1151;
          };

          llama-cpp-gfx120X = mkLlamaCppDerivation pkgs {
            target = "gfx120X";
            rocm = self.packages.${system}.rocm-gfx120X;
          };

          # ROCWMMA-enabled packages
          llama-cpp-gfx110X-rocwmma = mkLlamaCppDerivation pkgs {
            target = "gfx110X";
            rocm = self.packages.${system}.rocm-gfx110X;
            enableRocwmma = true;
            enableHipBlasLt = true;
          };

          llama-cpp-gfx1151-rocwmma = mkLlamaCppDerivation pkgs {
            target = "gfx1151";
            rocm = self.packages.${system}.rocm-gfx1151;
            enableRocwmma = true;
            enableHipBlasLt = true;
          };

          llama-cpp-gfx120X-rocwmma = mkLlamaCppDerivation pkgs {
            target = "gfx120X";
            rocm = self.packages.${system}.rocm-gfx120X;
            enableRocwmma = true;
            enableHipBlasLt = true;
          };

          default = self.packages.${system}.llama-cpp-gfx1151;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python3
            cmake
            ninja
            curl
            jq
          ];

          shellHook = ''
            echo "Nix Llama.cpp + ROCm development environment"
            echo ""
            echo "Available commands:"
            echo "  python3 update-rocm.py    - Update ROCm sources from S3"
            echo ""
            echo "Standard builds:"
            echo "  nix build .#llama-cpp-gfx110X  - Build for RDNA3 GPUs"
            echo "  nix build .#llama-cpp-gfx1151  - Build for STX Halo GPUs"
            echo "  nix build .#llama-cpp-gfx120X  - Build for RDNA4 GPUs"
            echo ""
            echo "ROCWMMA-optimized builds (15x faster flash attention):"
            echo "  nix build .#llama-cpp-gfx110X-rocwmma  - Build for RDNA3 with ROCWMMA"
            echo "  nix build .#llama-cpp-gfx1151-rocwmma  - Build for STX Halo with ROCWMMA"
            echo "  nix build .#llama-cpp-gfx120X-rocwmma  - Build for RDNA4 with ROCWMMA"
            echo ""
          '';
        };

        # Apps for easy running
        apps = {
          update-rocm = {
            type = "app";
            program = toString (pkgs.writeShellScript "update-rocm" ''
              ${pkgs.python3}/bin/python3 ${./update-rocm.py} "$@"
            '');
          };
        };
      }
    );
}
