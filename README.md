# nix-strix-halo

Small Nix flake for Strix Halo / Sixunited AXB35 systems.

It provides:

- generic `llama-cpp` outputs for Linux and macOS
- Linux-only ROCm outputs, including generic ROCm builds and Strix Halo `gfx1151` narrowed builds
- TheRock ROCm/PyTorch packaging for configured Linux GPU targets
- NixOS modules for llama.cpp RPC servers, EC fan/power controls, Ryzen power limits, RAID0 disk layout, system tuning, and benchmark hosts
- benchmark derivations for local `/models` GGUF files
- a minimal `fevm-faex9` NixOS example builder

## Outputs

```bash
nix flake show
```

Main package outputs:

- `packages.aarch64-darwin.default`
- `packages.aarch64-darwin.llama-cpp`
- `packages.x86_64-darwin.default`
- `packages.x86_64-darwin.llama-cpp`
- `packages.x86_64-linux.default`
- `packages.x86_64-linux.llama-cpp`
- `packages.x86_64-linux.llama-cpp-rocm`
- `packages.x86_64-linux.llama-cpp-rocm-gfx1151`
- `packages.x86_64-linux.llama-cpp-vulkan`
- `packages.x86_64-linux.ec-su-axb35-monitor`
- `packages.x86_64-linux.strix-halo-mes-firmware`
- `packages.x86_64-linux.bench-*`

Main app outputs:

- `apps.aarch64-darwin.llama-cli`
- `apps.aarch64-darwin.llama-server`
- `apps.x86_64-darwin.llama-cli`
- `apps.x86_64-darwin.llama-server`
- `apps.x86_64-linux.llama-cli`
- `apps.x86_64-linux.llama-server`
- `apps.x86_64-linux.llama-cli-rocm`
- `apps.x86_64-linux.llama-server-rocm`
- `apps.x86_64-linux.llama-cli-gfx1151`
- `apps.x86_64-linux.llama-server-gfx1151`

Example configuration helper:

- `lib.mkFevmFaex9Configuration`

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
            pkgs.llama-cpp
            pkgs.llama-cpp-rocm-gfx1151
          ];
        })
      ];
    };
  };
}
```

On macOS, the overlay intentionally exposes only generic non-ROCm outputs such as `llama-cpp` and the CPU `llama-cli`/`llama-server` apps. ROCm, TheRock, EC, firmware, and benchmark outputs are Linux-only.

## TheRock ROCm Targets

Target records are generic build descriptors, not a hardware inventory. Hostnames, PCI IDs, and machine-specific policy belong in the caller's own configuration. This branch configures `gfx1151` by default:

- `packages.x86_64-linux.llama-cpp-rocm-gfx1151`
- `packages.x86_64-linux.therock-rocm-gfx1151`
- `packages.x86_64-linux.therock-rocm-gfx1151-env`
- `packages.x86_64-linux.therock-python-gfx1151`
- `packages.x86_64-linux.torch-rocm-gfx1151`

The lower-level ROCm module overlay also exposes reusable narrowed package scopes:

- `pkgs.therockRocmPackages.gfx1151`
- `pkgs.therockRocmPackages.gfx11`
- `pkgs.therockRocmPackages.gfx9`, `gfx10`, and `gfx12`

Consumers can build their own target lists with the exported helpers:

```nix
let
  myTarget = nix-strix-halo.lib.mkRocmTarget {
    packageSuffix = "gfx1100";
    hsaOverride = "11.0.0";
  };
in
{
  nixpkgs.overlays = [
    (nix-strix-halo.lib.mkRocmNarrowOverlay {
      rocmGpuTargets = myTarget.rocmGpuTargets;
    })
    (nix-strix-halo.lib.mkTherockRocmOverlay {
      rocmTargets = [ myTarget ];
      target = myTarget;
    })
  ];
}
```

To add another top-level target to this flake's default outputs, add a target record in `pkgs/therock/targets.nix`. The `llama-cpp-rocm-<target>` output appears from that record; TheRock binary/source/Python outputs appear as matching target-keyed JSON pins are added under `pkgs/therock/sources/`.

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
- `nixosModules.tuning`: high-performance kernel defaults, pinned Strix Halo MES firmware, and TuneD defaults
- `nixosModules.disko-raid0`: dual-NVMe RAID0 Disko layout
- `nixosModules.benchmark-runner`: benchmark host setup, model downloads, and sandbox GPU access

`disko-raid0` defines Disko options but does not import Disko itself. Callers must provide `disko.nixosModules.disko` before using it. It targets `/dev/nvme0n1` and `/dev/nvme1n1`; treat it as machine-specific.

## Example Host

The `fevm-faex9` example requires a caller-provided Disko module:

```bash
nix build --impure --file ./examples/fevm-faex9 \
  --arg diskoModule '(builtins.getFlake "github:nix-community/disko").nixosModules.disko' \
  config.system.build.toplevel
```

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
