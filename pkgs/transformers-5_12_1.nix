{
  lib,
  buildPythonPackage,
  fetchurl,
  filelock,
  huggingface-hub,
  numpy,
  packaging,
  pyyaml,
  regex,
  requests,
  safetensors,
  tokenizers,
  tqdm,
  typer,
}:

buildPythonPackage rec {
  pname = "transformers";
  version = "5.12.1";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/df/56/bbd60dd8668055803bf8ba55a81f9b8a8b31497f620109a9671d26a2076d/transformers-${version}-py3-none-any.whl";
    hash = "sha256-Kl4QnSAhJl33CY/7tzgpWsr1rSVvEsvFhtsupNy7Goo=";
  };

  dependencies = [
    filelock
    huggingface-hub
    numpy
    packaging
    pyyaml
    regex
    requests
    safetensors
    tokenizers
    tqdm
    typer
  ];

  pythonRelaxDeps = [
    "huggingface-hub"
    "regex"
    "tokenizers"
  ];

  doCheck = false;
  # Qwen4-Exp's config/model implementation is carried by the matching SGLang
  # support commit, not by the Transformers wheel.  This pin supplies the HF
  # APIs that implementation was developed and tested against.
  pythonImportsCheck = [
    "transformers"
  ];

  meta = {
    description = "Transformers API release used by SGLang's Qwen4-Exp support";
    homepage = "https://github.com/huggingface/transformers";
    license = lib.licenses.asl20;
    platforms = lib.platforms.unix;
  };
}
