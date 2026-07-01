{
  nanobind,
  fetchFromGitHub,
}:

nanobind.overridePythonAttrs rec {
  version = "2.13.0";

  src = fetchFromGitHub {
    owner = "wjakob";
    repo = "nanobind";
    tag = "v${version}";
    fetchSubmodules = true;
    hash = "sha256-YAqjcVBkuNsXvrAaVmDRLQ1F38UBqdnIf8+OseNBzG4=";
  };
}
