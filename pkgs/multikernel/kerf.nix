{
  lib,
  buildPythonApplication,
  fetchFromGitHub,
  poetry-core,
  click,
  libfdt,
  pyyaml,
  pyudev,
  rdtsc,
  zstandard,
}:

buildPythonApplication rec {
  pname = "kerf-multikernel";
  version = "0.2.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "multikernel";
    repo = "kerf";
    rev = "8b72b3e9b266f8d32e707e2c1743ad7afc50b1ec";
    hash = "sha256-feP1fO7A6ARdth05Eo6PltzWkAC3UKdzZ1vtM0bg7hY=";
  };

  build-system = [ poetry-core ];
  dependencies = [
    click
    libfdt
    pyyaml
    pyudev
    rdtsc
    zstandard
  ];

  # Hardware lifecycle tests require a multikernel host. Unit coverage is
  # exercised separately by the flake check with a writable fake sysfs tree.
  doCheck = false;
  pythonImportsCheck = [ "kerf" ];

  meta = {
    description = "Resource-safe lifecycle manager for Linux multikernel instances";
    homepage = "https://github.com/multikernel/kerf";
    license = lib.licenses.asl20;
    mainProgram = "kerf";
    platforms = [ "x86_64-linux" ];
    maintainers = with lib.maintainers; [ georgewhewell ];
  };
}
