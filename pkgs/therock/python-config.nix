_:

let
  pythonVersion = "3.12";
  pythonDigits = builtins.replaceStrings [ "." ] [ "" ] pythonVersion;
in
{
  inherit pythonVersion;
  pythonTag = "cp${pythonDigits}";
  packageAttr = "python${pythonDigits}";
  packagesAttr = "python${pythonDigits}Packages";
}
