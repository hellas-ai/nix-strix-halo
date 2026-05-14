# nix-strix-halo

Small Nix flake for Strix Halo / Sixunited AXB35 systems.

It provides:

- a nixpkgs overlay for `llama-cpp` with ROCm enabled for `gfx1151`
- latest vLLM packages for CPU, Strix Halo ROCm (`gfx1151`), and RTX 4090 CUDA (`sm_89`)
- Zen 5 tuned vLLM variants and reusable helpers for other CPU/GPU targets
- NixOS modules for llama.cpp RPC servers, EC fan/power controls, Ryzen power limits, RAID0 disk layout, system tuning, and benchmark hosts
- benchmark derivations for local `/models` GGUF files
- a minimal `fevm-faex9` NixOS example

## Outputs

```bash
nix flake show
```

Main package outputs:

- `packages.x86_64-linux.default`
- `packages.x86_64-linux.llama-cpp-rocm`
- `packages.x86_64-linux.llama-cpp-rocm-zen5`
- `packages.x86_64-linux.llama-cpp-vulkan`
- `packages.x86_64-linux.ec-su-axb35-monitor`
- `packages.x86_64-linux.vllm-cpu`
- `packages.x86_64-linux.vllm-cpu-zen5`
- `packages.x86_64-linux.vllm-rocm-gfx1151`
- `packages.x86_64-linux.vllm-rocm-gfx1151-zen5`
- `packages.x86_64-linux.vllm-cuda-rtx4090`
- `packages.x86_64-linux.vllm-cuda-rtx4090-zen5`
- `packages.x86_64-linux.vllm-env-*`
- `packages.x86_64-linux.bench-*`

Main app outputs:

- `apps.x86_64-linux.llama-cli`
- `apps.x86_64-linux.llama-server`

Package-set outputs:

- `legacyPackages.x86_64-linux.defaultPackages`
- `legacyPackages.x86_64-linux.zen5Packages`

Use package sets when CPU tuning should stay independent of accelerator choice:

- `legacyPackages.x86_64-linux.zen5Packages.llama-cpp`
- `legacyPackages.x86_64-linux.zen5Packages.llama-cpp-rocm`

## Use The Overlay

```nix
{
  inputs.nix-strix-halo.url = "github:hellas-ai/nix-strix-halo";

  outputs = { nixpkgs, nix-strix-halo, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        nix-strix-halo.nixosModules.default

        ({ pkgs, ... }: {
          environment.systemPackages = [
            pkgs.llama-cpp-rocm
          ];
        })
      ];
    };
  };
}
```

## vLLM

The vLLM package is a thin override of the upstream nixpkgs derivation. Its source tracks upstream `releases/v0.20.2` through the `vllm` flake input. Hardware defaults are intentionally narrow:

- `vllm-cpu`: CPU backend
- `vllm-rocm-gfx1151`: ROCm for Strix Halo / Radeon 8060S
- `vllm-cuda-rtx4090`: CUDA for RTX 4090 / `sm_89`
- `*-zen5`: same hardware target, imported with `localSystem.gcc.arch = "znver5"`
- `vllm-env-*`: Python environment with `vllm` and `ray`

The tuned package set exposes the same vLLM names without the `-zen5` suffix:

```bash
nix build .#zen5Packages.vllm-rocm-gfx1151
```

For custom targets, use the helper functions instead of adding more aliases:

```nix
let
  hardware = inputs.nix-strix-halo.lib.mkRocmHardware {
    name = "gfx1100";
    gpuTargets = [ "gfx1100" ];
  };

  vllmPkgs = inputs.nix-strix-halo.lib.mkPackageSet {
    system = "x86_64-linux";
    inherit hardware;
    cpu = "znver5";
  };
in
inputs.nix-strix-halo.lib.mkVllmPackage {
  pkgs = vllmPkgs;
  inherit hardware;
  tunePackage = true;
}
```

Available overlays:

- `overlays.default`
- `overlays.tuned`: Zen 5 tuning overlay
- `overlays.mkTunedOverlay "znver4"`: tuning overlay for another CPU target

```nix
overlays = [
  inputs.nix-strix-halo.overlays.default
  (inputs.nix-strix-halo.overlays.mkTunedOverlay "znver4")
];
```

## llama.cpp RPC Servers

```nix
{
  imports = [
    inputs.nix-strix-halo.nixosModules.default
    inputs.nix-strix-halo.nixosModules.rpc-server
  ];

  services.llama-cpp-rpc-servers.gpu = {
    enable = true;
    package = pkgs.llama-cpp-rocm;
    device = "0";
    host = "0.0.0.0";
    port = 50052;
    threads = 64;
    openFirewall = true;
  };
}
```

The module supports named instances under `services.llama-cpp-rpc-servers`.

## Strix Halo Modules

- `nixosModules.ec-su-axb35`: kernel module and optional monitor script for the Sixunited AXB35 EC
- `nixosModules.ryzenadj`: power and curve optimizer settings through `ryzenadj`
- `nixosModules.tuning`: high-performance kernel and TuneD defaults
- `nixosModules.disko-raid0`: dual-NVMe RAID0 Disko layout
- `nixosModules.benchmark-runner`: benchmark host setup, model downloads, and sandbox GPU access

`disko-raid0` targets `/dev/nvme0n1` and `/dev/nvme1n1`; treat it as machine-specific.

## Benchmarks

Benchmark packages expect model files under `/models` and require GPU access from the Nix build sandbox.

```bash
nix build .#bench-llama2-7b-llama-cpp-rocm-b512-fa1
cat result
```

Use `services.benchmark-runner` on benchmark hosts to add system features, GPU device access, and optional Hugging Face model download services.

## Development

```bash
nix develop
nix fmt -- --fail-on-change
nix flake check --no-build
```

## CI

Pull requests run a matrix derived from the flake's `checks` output on the shared self-hosted runner. Protect `master` by requiring the `required checks` job.

Pushes to `master` run a full build of non-benchmark package outputs. Benchmark derivations are evaluated by `nix flake check --no-build`, but are not built in CI because they require local models and hardware-specific system features.
