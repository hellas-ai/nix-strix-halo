{
  lib,
  stdenv,
  autoPatchelfHook,
  buildPythonPackage,
  fetchurl,
  huggingface-hub,
}:

# The pinned Transformers 5.12.1 API accepts tokenizers <= 0.23.0.
# Keep this older experimental stack on the latest compatible stable wheel.
buildPythonPackage rec {
  pname = "tokenizers";
  version = "0.22.2";
  format = "wheel";
  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/2e/76/932be4b50ef6ccedf9d3c6639b056a967a86258c6d9200643f01269211ca/tokenizers-${version}-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
    hash = "sha256-NpzJ/IzBDLJBQ4c6DZVDi7juJXu4DHGYnj7ikOjXLGc=";
  };
  dependencies = [ huggingface-hub ];
  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [ stdenv.cc.cc.lib ];
  pythonImportsCheck = [ "tokenizers" ];
  meta = {
    description = "Tokenizers release paired with the Qwen4-Exp Transformers API";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
    sourceProvenance = [ lib.sourceTypes.binaryNativeCode ];
  };
}
