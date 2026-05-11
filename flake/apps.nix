{ pkgs }:

{
  llama-cli = {
    type = "app";
    program = toString (
      pkgs.writeShellScript "llama-cli" ''
        export HSA_OVERRIDE_GFX_VERSION=11.5.1
        ${pkgs.llama-cpp-rocm}/bin/llama-cli "$@"
      ''
    );
    meta.description = "Run llama-cli with the Strix Halo ROCm environment";
  };

  llama-server = {
    type = "app";
    program = toString (
      pkgs.writeShellScript "llama-server" ''
        export HSA_OVERRIDE_GFX_VERSION=11.5.1
        ${pkgs.llama-cpp-rocm}/bin/llama-server "$@"
      ''
    );
    meta.description = "Run llama-server with the Strix Halo ROCm environment";
  };
}
