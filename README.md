# nix-strix-halo

Small Nix flake for Strix Halo / Sixunited AXB35 systems.

It provides:

- a nixpkgs overlay for `llama-cpp` with ROCm enabled for `gfx1151`
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
- `packages.x86_64-linux.llama-cpp-vulkan`
- `packages.x86_64-linux.ec-su-axb35-monitor`
- `packages.x86_64-linux.bench-*`

Main app outputs:

- `apps.x86_64-linux.llama-cli`
- `apps.x86_64-linux.llama-server`

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
