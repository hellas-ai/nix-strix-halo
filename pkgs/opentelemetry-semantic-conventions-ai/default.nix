{
  lib,
  buildPythonPackage,
  fetchPypi,
  poetry-core,
}:

buildPythonPackage rec {
  pname = "opentelemetry-semantic-conventions-ai";
  version = "0.4.1";
  pyproject = true;

  src = fetchPypi {
    pname = "opentelemetry_semantic_conventions_ai";
    inherit version;
    hash = "sha256-qvWbLyTXRWkhcLlthtfFVg9CRD3PiM7UmunUVC2xkC8=";
  };

  build-system = [ poetry-core ];

  pythonImportsCheck = [ "opentelemetry.semconv_ai" ];

  meta = {
    description = "OpenTelemetry semantic conventions extension for generative AI";
    homepage = "https://github.com/traceloop/openllmetry/tree/main/packages/opentelemetry-semantic-conventions-ai";
    license = lib.licenses.asl20;
    maintainers = [ ];
  };
}
