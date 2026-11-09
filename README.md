# nix-strix-halo

[![Hydra CI](https://img.shields.io/endpoint?label=hydra%20ci&url=https%3A%2F%2Fhydra.hellas.ai%2Fjob%2Fhellas%2Fnix-strix-halo%2Fx86_64-linux.ci.smoke%2Fshield)](https://hydra.hellas.ai/job/hellas/nix-strix-halo/x86_64-linux.ci.smoke)

*NO SUPPORT / WARRANTY*

Until version reaches 1.0, the layout and functionality of this flake are subject to change and should not be relied upon, even when pinned to specific commits. I'll try not to rewrite history or introduce breaking changes, but until that point the primary purpose of this repo is for me to trigger CI and populate the `cache.hellas.ai` binary cache for development and testing workflows. Consider yourself warned.

*END WARNING*

Workspace flake for Hellas- libraries and applications from ML ecosystem, packaged composably with nix.

## What's in it

Default outputs target gfx1151 with AMD's stable ROCm 10 TheRock SDK and
TheRock-published Python wheels.

| Output (under `packages.x86_64-linux.*`) | What |
|---|---|
| `llama-cpp{,-rocm,-vulkan}` | nixpkgs package aliases |
| `sglang-rocm` | SGLang 0.5.20 against TheRock; [GLM-5.3-Flash bring-up plan](docs/glm53-flash-strix.md) |
| `fastflowlm` | XDNA2 NPU CLI (`flm`) |
| `therock-rocm`, `therock-python-wheels`, `torch-rocm` | TheRock binary SDK + wheels |
| `linux-multikernel`, `kerf-multikernel` | pinned Linux 7.0-mk2 host/spawn kernel and lifecycle manager |
| `multikernel-demo-initrd` | tiny interactive spawn initramfs for bare-metal isolation demos |
| `xrt`, `xrt-amdxdna`, `tokenizers-cpp`, `ec-su-axb35-monitor` | hardware support bits |
| `live-iso` | USB-flashable strix-halo live system |
| Darwin: `llama-cpp`, `mlx`, `mlx-metal`, `vllm-metal`, `jaccl` | cross-platform / Metal |

Apps include `sglang-rocm`, `flm`, `pi-wrap` and `live-iso-vm`.

## Use the overlay

```nix
{
  inputs.nix-strix-halo.url = "github:hellas-ai/nix-strix-halo";

  outputs = { nixpkgs, nix-strix-halo, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        nix-strix-halo.nixosModules.default
        ({ pkgs, ... }: {
          environment.systemPackages = [ pkgs.sglang-rocm ];
        })
      ];
    };
  };
}
```

## Run / build

```bash
nix run .#sglang-rocm -- --help
nix run .#flm -- list
nix build .#live-iso             # iso at ./result/iso/*.iso
nix run  .#live-iso-vm           # boot the iso in QEMU
```

### Darwin vLLM Metal and JACCL

`packages.aarch64-darwin.vllm-metal` packages the matching vLLM 0.31.0 and
vLLM-Metal 0.31.0 wheels with source-built MLX 0.32.1 and JACCL.

```bash
nix run .#vllm-metal -- serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --max-model-len 8192 --no-enable-prefix-caching
nix build .#packages.aarch64-darwin.vllm-metal.tests.metal
```

The `darwinModules.vllm-metal` service keeps its downloads, caches, and logs in
`/var/lib/vllm-metal`. See [the Metal guide](docs/vllm-metal.md) for host
requirements, service configuration, validation, coordinated dependency updates,
and the separate MLX-LM/JACCL tensor-parallel harness.

## Non-default targets and providers

The package set is parameterised over three axes:

| axis | tag/value source | default |
|---|---|---|
| rocm provider | `lib.rocmProviders` (`therock-bin`, `nixpkgs`) | `therock-bin` |
| python provider | `lib.pythonProviders` (`therock-wheels`; future stubs in `lib.pythonProviderStubs`) | `therock-wheels` |
| GPU target | `pkgs/therock/targets.nix` | `gfx1151` |

Per-target packages (default providers) are available under
`legacyPackages`:

```bash
nix build .#legacyPackages.x86_64-linux.gfx1103.llama-cpp-rocm
nix build .#legacyPackages.x86_64-linux.gfx1030.sglang-rocm # Radeon Pro V620
```

Only targets with matching TheRock binary/Python source pins expose
TheRock-shaped packages such as `therock-rocm` and `sglang-rocm`. At the moment those pins exist for `gfx1030` (Radeon Pro V620) and
`gfx1151`. Package availability may be narrower where an architecture lacks a
required feature.
`llama-cpp-rocm` is available for every target listed in
`pkgs/therock/targets.nix`.

For non-default providers compose your own overlays with `lib.mkRocmOverlay`,
`lib.mkPythonOverlay`, and `lib.mkPkgsOverlay`. Add a new target by
appending to `pkgs/therock/targets.nix`; add a new provider by
implementing the dispatch in `overlays/{rocm,python}.nix` and registering
the tag in `lib/providers.nix`.

## NixOS modules

- `nixosModules.default` — apply the overlay
- `nixosModules.rpc-server` — llama-cpp RPC server instances
- `nixosModules.fastflowlm` — FastFlowLM OpenAI-compatible server (XDNA2)
- `nixosModules.benchmark-runner` / `benchmark-executor` — local + remote bench infra
- `nixosModules.ec-su-axb35`, `ryzenadj` — Strix Halo hardware modules
- `nixosModules.multikernel` — declarative bare-metal kernel pools and spawn instances

The multikernel package builds the complete upstream `v7.0-mk2` tree, not a
patch stack replayed onto the current nixpkgs kernel. See
[docs/multikernel.md](docs/multikernel.md) for the module, local-source override,
hardware bring-up, rollback, and panic-isolation demonstration.

## Hydra / CI

Hydra reads the root flake's `hydraJobs` output. Required PR CI is split
into three gates:

- `ci.checks` runs source/meta checks such as formatting and Nix linting.
- `ci.build` builds the package surface, including cross-platform package
  outputs and provider variants.
- `ci.smoke` runs one small real-hardware smoke per accelerated engine.

The separate `hydraBenchmarkJobs` output is used by the background benchmark
jobset. Those benchmark sweeps are useful for regression data but are not
required for PR merge.

```bash
nix build .#hydraJobs.x86_64-linux.ci.checks
nix build .#hydraJobs.x86_64-linux.ci.build
nix build .#hydraJobs.x86_64-linux.ci.smoke
```

## Development

```bash
nix develop
nix fmt -- --fail-on-change
nix flake check --no-build
```
