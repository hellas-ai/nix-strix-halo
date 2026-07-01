# Opt-in TheRock Python overlay. Substitute only the interpreter selected
# by pkgs/therock/python-config.nix; other Python interpreters stay on
# nixpkgs defaults.
{
  lib,
  target,
  pythonConfig ? import ../pkgs/therock/python-config.nix { inherit lib; },
}:
let
  s = target.packageSuffix;
  disablePythonChecks =
    pkg:
    pkg.overridePythonAttrs (_old: {
      doCheck = false;
      pythonImportsCheck = [ ];
    });
in
final: prev: {
  pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
    (
      _pyfinal: pyprev:
      if lib.versions.majorMinor pyprev.python.version == pythonConfig.pythonVersion then
        let
          wheels = final."therock-python-wheels-${s}";
        in
        {
          amdsmi = final."therock-amdsmi-${s}";
          torch = wheels;
          torchvision = wheels;
          triton = wheels;
          triton-no-cuda = wheels;
          torchaudio = wheels;
          rocm = wheels;
          "rocm-sdk-core" = wheels;
          "rocm-sdk-devel" = wheels;
          "rocm-sdk-libraries-${s}" = wheels;

          "compressed-tensors" = pyprev."compressed-tensors".overridePythonAttrs (old: {
            doCheck = false;
            dependencies = (old.dependencies or [ ]) ++ [
              _pyfinal.psutil
            ];
            nativeCheckInputs = (old.nativeCheckInputs or [ ]) ++ [
              final.openssl
            ];
            propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [
              _pyfinal.psutil
            ];
          });
          depyf = disablePythonChecks pyprev.depyf;
          llguidance = disablePythonChecks pyprev.llguidance;
          pydevd = disablePythonChecks pyprev.pydevd;
          "mistral-common" = disablePythonChecks pyprev."mistral-common";
        }
      else
        { }
    )
  ];
}
