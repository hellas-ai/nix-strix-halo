{
  description = "Llama.cpp with pre-built ROCm binaries from TheRock";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    llama-cpp = {
      url = "github:ggerganov/llama.cpp";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    llama-cpp,
  }: let
    rocmSources = builtins.fromJSON (builtins.readFile ./rocm-sources.json);

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
    }: let
          amdgpuTargets =
            if target == "gfx110X"
            then "gfx1100"
            else if target == "gfx1151"
            then "gfx1151"
            else if target == "gfx120X"
            then "gfx1200;gfx1201"
            else target;

          # Simple hipcc wrapper for compilation
          hipccWrapper = pkgs.writeScriptBin "hipcc" ''
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
            pname = "llama-cpp-rocm-${target}";
            version = "git";
            hardeningDisable = ["all"];
            src = llama-cpp;

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
            ];

            postPatch = ''
              substituteInPlace ggml/src/ggml-cuda/vendors/hip.h \
                --replace "HIP_VERSION >= 70000000" "HIP_VERSION >= 50600000"
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
            ];

            preConfigure = ''
              export HIP_PATH="${rocm}"
              export ROCM_PATH="${rocm}"
              export HIP_PLATFORM="amd"
              export HIP_DEVICE_LIB_PATH="${rocm}/lib/llvm/amdgcn/bitcode"
              export PATH="${rocm}/bin:$PATH"
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
          
          # Llama.cpp packages
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
          rocm-gfx110X = mkRocmDerivation pkgs "gfx110X";
          rocm-gfx1151 = mkRocmDerivation pkgs "gfx1151";
          rocm-gfx120X = mkRocmDerivation pkgs "gfx120X";

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
            echo "  nix build .#llama-cpp-gfx110X  - Build for RDNA3 GPUs"
            echo "  nix build .#llama-cpp-gfx1151  - Build for STX Halo GPUs"
            echo "  nix build .#llama-cpp-gfx120X  - Build for RDNA4 GPUs"
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
