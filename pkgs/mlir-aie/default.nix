{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  makeWrapper,
  symlinkJoin,
  python312Packages,
  zlib,
  zstd,
  libxml2,
  ncurses,
  libffi,
  libedit,
  libdrm,
  xrt,
}:

let
  python = python312Packages.python;

  llvm-aie = python312Packages.buildPythonPackage rec {
    pname = "llvm-aie";
    version = "21.0.0.2026070501+a9e2c148";
    format = "wheel";

    src = fetchurl {
      url = "https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-${version}-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      hash = "sha256-J2NNeWQfFlxn07NDHAte9Ka1kbstP6IhI958BAlVvZ0=";
    };

    nativeBuildInputs = [ autoPatchelfHook ];

    buildInputs = [
      stdenv.cc.cc.lib
      zlib
      zstd
      libxml2
      ncurses
      libffi
      libedit
    ];

    pythonImportsCheck = [ ];

    meta = {
      description = "Peano LLVM/Clang toolchain for AMD/Xilinx AI Engine processors";
      homepage = "https://github.com/Xilinx/llvm-aie";
      license = lib.licenses.asl20;
      maintainers = with lib.maintainers; [ georgewhewell ];
      platforms = [ "x86_64-linux" ];
    };
  };

  mlir-aie = python312Packages.buildPythonPackage rec {
    pname = "mlir-aie";
    version = "1.3.4";
    format = "wheel";

    src = fetchurl {
      url = "https://github.com/Xilinx/mlir-aie/releases/download/latest-wheels-4/mlir_aie-${version}-cp312-cp312-manylinux_2_35_x86_64.whl";
      hash = "sha256-wnO4mcFtz6p1DwZn6uj+/4vKizfdO8Z/xsWL2N3YmKo=";
    };

    nativeBuildInputs = [
      autoPatchelfHook
      makeWrapper
    ];

    buildInputs = [
      stdenv.cc.cc.lib
      zlib
      zstd
      libxml2
      ncurses
      libffi
      libedit
      libdrm
      xrt.xdna
    ];

    propagatedBuildInputs = with python312Packages; [
      aiofiles
      cloudpickle
      ml-dtypes
      numpy
      rich
    ];

    pythonImportsCheck = [
      "aie"
      "aie.iron"
    ];

    postInstall = ''
      site="$out/${python.sitePackages}"
      echo "$site/mlir_aie/python" > "$site/mlir_aie.pth"
      ln -s "$site/mlir_aie/python/aie" "$site/aie"
    '';

    postFixup = ''
      site="$out/${python.sitePackages}"
      peano="${llvm-aie}/${python.sitePackages}/llvm-aie"
      wrapProgram "$out/bin/aiecc.py" \
        --prefix PATH : "$peano/bin:${placeholder "out"}/bin:$site/mlir_aie/bin" \
        --set-default PEANO_INSTALL_DIR "$peano" \
        --set-default MLIR_AIE_INSTALL_DIR "$site/mlir_aie" \
        --set-default XILINX_XRT "${xrt.xdna}"

      for prog in aiecc aie-opt aie-translate aie-reset bootgen txn2mlir.py xchesscc_wrapper; do
        if [ -x "$out/bin/$prog" ]; then
          wrapProgram "$out/bin/$prog" \
            --prefix PATH : "$peano/bin" \
            --set-default PEANO_INSTALL_DIR "$peano" \
            --set-default MLIR_AIE_INSTALL_DIR "$site/mlir_aie" \
            --set-default XILINX_XRT "${xrt.xdna}"
        fi
      done
    '';

    meta = {
      description = "MLIR-AIE IRON/aiecc toolchain for AMD Ryzen AI NPUs";
      homepage = "https://github.com/Xilinx/mlir-aie";
      license = lib.licenses.asl20;
      maintainers = with lib.maintainers; [ georgewhewell ];
      platforms = [ "x86_64-linux" ];
    };
  };
in
{
  inherit llvm-aie mlir-aie;

  mlir-aie-env = symlinkJoin {
    name = "mlir-aie-env-${mlir-aie.version}";
    paths = [
      llvm-aie
      mlir-aie
    ];
    nativeBuildInputs = [ makeWrapper ];
    postBuild = ''
            mkdir -p "$out/nix-support"
            cat > "$out/nix-support/setup-hook" <<EOF
      export PEANO_INSTALL_DIR=${llvm-aie}/${python.sitePackages}/llvm-aie
      export MLIR_AIE_INSTALL_DIR=${mlir-aie}/${python.sitePackages}/mlir_aie
      export XILINX_XRT=${xrt.xdna}
      export PATH=${llvm-aie}/${python.sitePackages}/llvm-aie/bin:${mlir-aie}/bin:\$PATH
      EOF
    '';
    meta = {
      description = "Combined MLIR-AIE and Peano environment for AMD Ryzen AI NPU development";
      homepage = "https://github.com/Xilinx/mlir-aie";
      license = lib.licenses.asl20;
      maintainers = with lib.maintainers; [ georgewhewell ];
      platforms = [ "x86_64-linux" ];
    };
  };
}
