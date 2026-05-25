# Opt-in TheRock Python overlay. The TheRock wheels are currently cp312,
# so keep the substitution scoped to python312 package sets. Other Python
# interpreters stay on the nixpkgs/libibverbs defaults.
{ lib, target }:
let
  s = target.packageSuffix;
in
final: prev: {
  pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
    (
      _pyfinal: pyprev:
      if lib.versions.majorMinor pyprev.python.version == "3.12" then
        let
          wheels = final."therock-python-wheels-${s}";
        in
        {
          amdsmi = final."therock-amdsmi-${s}";
          torch = wheels;
          triton = wheels;
          triton-no-cuda = wheels;
          torchaudio = wheels;
          rocm = wheels;
          "rocm-sdk-core" = wheels;
          "rocm-sdk-devel" = wheels;
          "rocm-sdk-libraries-${s}" = wheels;
        }
      else
        { }
    )
  ];
}
