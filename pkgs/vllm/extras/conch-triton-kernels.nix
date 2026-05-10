{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  setuptools-scm,
  numpy,
  triton,
}:

buildPythonPackage rec {
  pname = "conch-triton-kernels";
  version = "1.2.1";
  pyproject = true;

  src = fetchPypi {
    pname = "conch_triton_kernels";
    inherit version;
    hash = "sha256-VlTG10fZdGrG2qvdjTHOgkwwBN/+3pH5ImRmcMzKk24=";
  };

  build-system = [
    setuptools
    setuptools-scm
  ];

  dependencies = [
    numpy
    triton
  ];

  pythonRelaxDeps = true;

  # PyPI dist is "conch-triton-kernels" but installs the top-level
  # `conch` module — matching what vllm's find_spec("conch") expects.
  pythonImportsCheck = [ "conch" ];

  # Tests rely on a working triton runtime + GPU.
  doCheck = false;

  meta = {
    description = "Third-party Triton kernel collection (Conch / StackAV)";
    homepage = "https://github.com/stackav-oss/conch";
    license = lib.licenses.asl20;
  };
}
