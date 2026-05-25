# nix-strix-halo

Small Nix flake for Strix Halo / Sixunited AXB35 systems.

It provides:

- generic `llama-cpp` outputs for Linux and macOS
- Linux-only ROCm outputs, including generic ROCm builds and Strix Halo `gfx1151` narrowed builds
- TheRock ROCm/PyTorch packaging for configured Linux GPU targets
- NixOS modules for llama.cpp RPC servers, EC fan/power controls, Ryzen power limits, RAID0 disk layout, system tuning, and benchmark hosts
- cross-platform benchmark helpers and derivations for reproducible local model/tool runs
- a minimal `fevm-faex9` NixOS example builder

## Outputs

```bash
nix flake show
```

Main package outputs:

- `packages.aarch64-darwin.default`
- `packages.aarch64-darwin.bench-*`
- `packages.aarch64-darwin.llama-cpp`
- `packages.x86_64-darwin.default`
- `packages.x86_64-darwin.bench-*`
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

On macOS, the overlay intentionally exposes only generic non-ROCm outputs such as `llama-cpp`, CPU `llama-cli`/`llama-server` apps, and CPU benchmark derivations. ROCm, TheRock, EC, and firmware outputs are Linux-only.

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
- `nixosModules.benchmark-runner`: generic benchmark host setup, model downloads, and sandbox device access

`disko-raid0` defines Disko options but does not import Disko itself. Callers must provide `disko.nixosModules.disko` before using it. It targets `/dev/nvme0n1` and `/dev/nvme1n1`; treat it as machine-specific.

## Example Host

The `fevm-faex9` example requires a caller-provided Disko module:

```bash
nix build --impure --file ./examples/fevm-faex9 \
  --arg diskoModule '(builtins.getFlake "github:nix-community/disko").nixosModules.disko' \
  config.system.build.toplevel
```

## Benchmarks

Benchmark packages are generated from structured tool/model/scenario records. The default cross-platform matrix runs CPU `llama-bench` against local GGUF files under `/models`. Linux also exposes this flake's ROCm and Vulkan llama.cpp tool variants.

```bash
nix build .#bench-llama2-7b-llama-cpp-cpu-b512-fa1
cat result/stdout.txt
cat result/metadata.json
```

External flakes can reuse `lib.benchmarks` while injecting their own packages, model paths, required system features, environment variables, and host profile names. For example, with a caller-provided CUDA-enabled `myCudaLlamaCpp` package:

```nix
let
  bench = inputs.nix-strix-halo.lib.benchmarks;
in
bench.mkLlamaCppBenchmark {
  inherit pkgs;
  name = "llama-cpp-cuda-b512-fa1";
  package = myCudaLlamaCpp;
  model = "/models/llama-2-7b/llama-2-7b.Q4_K_M.gguf";
  params = {
    batch = 512;
    fa = 1;
  };
  requirements = {
    systemFeatures = [ "cuda-sm_89" ];
    hostProfiles = [ "linux-nvidia" ];
  };
  metadata = {
    accelerator = "cuda";
  };
}
```

Use `services.benchmark-runner` on NixOS benchmark hosts to add the Nix system features and sandbox device access required by benchmark derivations:

```nix
{
  services.benchmark-runner = {
    enable = true;
    systemFeatures = [
      "gfx1151"
      "aimax395"
    ];
    enabledProfiles = [ "linux-amd-kfd" ];
    modelsPath = "/models";
  };
}
```

The NixOS module is only a host adapter. It has built-in profiles for `linux-drm-render`, `linux-amd-kfd`, and `linux-nvidia`, and callers can define more profiles with their own sandbox paths and udev rules.

## Development

```bash
nix develop
nix fmt -- --fail-on-change
nix flake check --no-build
```

## CI

CI is driven by the flake's `checks` output. The buildbot should build `.#checks.<system>.*` for the systems it owns.

The checks include source formatting/linting, representative package builds, benchmark metadata validation, and Linux benchmark-runner module composition. Benchmark packages are evaluated by `nix flake check --no-build`, but the hardware/model-dependent benchmark derivations are not checks because they require local models and host-specific device access.
