# SGLang 0.5.20 needs newer hub kernels and SMG APIs than this nixpkgs pin.
{
  pythonPackages,
  fetchurl,
  autoPatchelfHook,
  stdenv,
}:
let
  wheel =
    {
      pname,
      version,
      url,
      hash,
      dependencies ? [ ],
      native ? false,
    }:
    pythonPackages.buildPythonPackage {
      inherit pname version dependencies;
      format = "wheel";
      src = fetchurl { inherit url hash; };
      nativeBuildInputs = if native then [ autoPatchelfHook ] else [ ];
      buildInputs = if native then [ stdenv.cc.cc.lib ] else [ ];
      pythonImportsCheck = [ (builtins.replaceStrings [ "-" ] [ "_" ] pname) ];
    };
in
rec {
  kernels-data = wheel {
    pname = "kernels-data";
    version = "0.14.0";
    url = "https://files.pythonhosted.org/packages/b1/09/05be36b4a7396b765092e219f3f251671f690fdae04a853cf7b758487349/kernels_data-0.14.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
    hash = "sha256-xuPtfEYXyNFq21A3DJLEix46hIeXDSGK4FEyc1xPRfo=";
    native = true;
  };
  kernels = wheel {
    pname = "kernels";
    version = "0.14.1";
    url = "https://files.pythonhosted.org/packages/94/94/fb8cbf1427af4b520ad08a458b6696f1958732761bc284b55273fa3aa684/kernels-0.14.1-py3-none-any.whl";
    hash = "sha256-vhqREW4Uvh4BL8j0evtRFz7gUlhqVw+6PVlO8NGziSA=";
    dependencies = [
      pythonPackages.huggingface-hub
      pythonPackages.packaging
      pythonPackages.pyyaml
      pythonPackages.tomlkit
      kernels-data
    ];
  };
  smg-grpc-proto = wheel {
    pname = "smg-grpc-proto";
    version = "0.4.13";
    url = "https://files.pythonhosted.org/packages/bc/23/4a9b2dce0b0ad6e57d3bf7fd5ea7fc18c0578a078f5c19906329a89d2406/smg_grpc_proto-0.4.13-py3-none-any.whl";
    hash = "sha256-dBsZSp/LYWbfcIbCsAlnnnnR8/6fTXJMhMKKw8U9/To=";
    dependencies = [
      pythonPackages.grpcio
      pythonPackages.protobuf
    ];
  };
  smg-grpc-servicer = wheel {
    pname = "smg-grpc-servicer";
    version = "0.9.0";
    url = "https://files.pythonhosted.org/packages/d1/21/f0a93d744c011da188fdfb1fa6108b3af8f853910d6fc258932af31cc2af/smg_grpc_servicer-0.9.0-py3-none-any.whl";
    hash = "sha256-Nw04yB+V9Hs9dzdDghJOkA4ug5+6usOryXxTTEjZnJI=";
    dependencies = [
      pythonPackages.grpcio
      pythonPackages.grpcio-health-checking
      pythonPackages.grpcio-reflection
      smg-grpc-proto
    ];
  };
}
