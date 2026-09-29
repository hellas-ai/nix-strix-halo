# Shared by SGLang and vLLM, whose parser APIs require xgrammar 0.2.x.
{
  lib,
  pythonPackages,
  fetchurl,
  autoPatchelfHook,
  patchelf,
  stdenv,
}:
let
  pythonSitePackages = pythonPackages.python.sitePackages;
  tvmFfiLibDir = "${pythonPackages.apache-tvm-ffi}/${pythonSitePackages}/tvm_ffi/lib";
in
pythonPackages.buildPythonPackage rec {
  pname = "xgrammar";
  version = "0.2.1";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/32/75/25ddd211f073a9db8299bdfec4534874d5d5d5f69499bb0c2ce9bf75f483/xgrammar-${version}-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-ugJAJKRU8x08uIZ5obBz66ATOGDxRSXx9YVvpG3sUig=";
  };

  nativeBuildInputs = [
    autoPatchelfHook
    patchelf
  ];

  buildInputs = [
    pythonPackages.apache-tvm-ffi
    stdenv.cc.cc.lib
  ];

  autoPatchelfIgnoreMissingDeps = [
    "libtvm_ffi.so"
  ];

  dependencies = with pythonPackages; [
    apache-tvm-ffi
    numpy
    pydantic
    torch
    transformers
    triton
    typing-extensions
  ];

  pythonImportsCheck = [ "xgrammar" ];

  postFixup = ''
    patchelf --add-rpath ${lib.escapeShellArg tvmFfiLibDir} \
      "$out/${pythonSitePackages}/xgrammar/libxgrammar_bindings.so"
  '';
}
