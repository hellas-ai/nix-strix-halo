{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
}:

buildPythonPackage rec {
  pname = "rdtsc";
  version = "0.2.1";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-mTqP6ArQDjzKl2/qoKT+mJeOleATq11miu7XiQveTos=";
  };

  postPatch = ''
    substituteInPlace src/rdtsc/__init__.py \
      --replace-fail "import pkg_resources" "from importlib import resources" \
      --replace-fail "pkg_resources.resource_filename('rdtsc', sofile)" 'str(resources.files("rdtsc").joinpath(sofile))'
  '';

  build-system = [ setuptools ];
  pythonImportsCheck = [ "rdtsc" ];

  meta = {
    description = "Small Python wrapper around the x86 RDTSC instruction";
    homepage = "https://github.com/Roguelazer/rdtsc";
    license = lib.licenses.isc;
    platforms = [ "x86_64-linux" ];
  };
}
