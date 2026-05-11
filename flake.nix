{
  description = "Strix Halo NixOS configuration and tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-vllm.url = "github:CertainLach/nixpkgs/push-lklxouywkrnv";

    vllm-src = {
      url = "github:vllm-project/vllm/releases/v0.20.2";
      flake = false;
    };

    vllm-cutlass-src = {
      url = "github:NVIDIA/cutlass/v4.4.2";
      flake = false;
    };

    vllm-flash-attn-src = {
      url = "github:vllm-project/flash-attention/f5bc33cfc02c744d24a2e9d50e6db656de40611c";
      flake = false;
    };

    vllm-flashmla-src = {
      url = "github:vllm-project/FlashMLA/a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b";
      flake = false;
    };

    vllm-flashmla-cutlass-src = {
      url = "github:NVIDIA/cutlass/147f5673d0c1c3dcf66f78d677fd647e4a020219";
      flake = false;
    };

    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    ec-su-axb35 = {
      url = "github:cmetz/ec-su_axb35-linux";
      flake = false;
    };
  };

  outputs = inputs: import ./flake inputs;
}
