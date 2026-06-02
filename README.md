# nix-strix-halo

[![Hydra CI](https://img.shields.io/endpoint?label=hydra%20ci&url=https%3A%2F%2Fhydra.hellas.ai%2Fjob%2Fhellas%2Fnix-strix-halo%2Fx86_64-linux.ci.smoke%2Fshield)](https://hydra.hellas.ai/job/hellas/nix-strix-halo/x86_64-linux.ci.smoke)

*NO SUPPORT / WARRANTY*

Until version reaches 1.0, the layout and functionality of this flake are subject to change and should not be relied upon, even when pinned to specific commits. I'll try not to rewrite history or introduce breaking changes, but until that point the primary purpose of this repo is for me to trigger CI and populate the `cache.hellas.ai` binary cache for development and testing workflows. Consider yourself warned.

*END WARNING*

Workspace flake for Hellas- libraries and applications from ML ecosystem, packaged composably with nix.

## What's in it

Default outputs target gfx1151 with the binary TheRock ROCm SDK and the
TheRock-published Python wheels.

| Output (under `packages.x86_64-linux.*`) | What |
|---|---|
| `llama-cpp{,-rocm,-vulkan,-cuda}` | nixpkgs llama-cpp + variant flags |
| `llama-cpp-master{,-rocm,-vulkan,-cuda}` | same matrix off llama.cpp HEAD |
| `vllm-rocm` | source-built vLLM 0.21 against TheRock |
| `mlx-rocm` | MLX with the ROCm backend |
| `ds4-rocm` | DwarfStar 4 HIP build |
| `fastflowlm` | XDNA2 NPU CLI (`flm`) |
| `therock-rocm`, `therock-python`, `torch-rocm` | TheRock binary SDK + wheels |
| `xrt`, `xrt-amdxdna`, `tokenizers-cpp`, `strix-halo-mes-firmware`, `ec-su-axb35-monitor` | hardware support bits |
| `live-iso` | USB-flashable strix-halo live system |
| Darwin: `llama-cpp`, `mlx`, `mlx-metal`, `ds4`, `jaccl` | cross-platform / Metal |

Apps mirror the package names — `apps.x86_64-linux.llama-cli-rocm`,
`flm`, `therock-python`, `live-iso-vm`, etc.

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
          environment.systemPackages = [ pkgs.vllm-rocm pkgs.llama-cpp-rocm ];
        })
      ];
    };
  };
}
```

## Run / build

```bash
nix run .#llama-cli-rocm -- -m model.gguf -p "Hello"
nix run .#flm -- list
nix build .#vllm-rocm
nix build .#live-iso             # iso at ./result/iso/*.iso
nix run  .#live-iso-vm           # boot the iso in QEMU
```

## Non-default targets and providers

The package set is parameterised over three axes:

| axis | tag/value source | default |
|---|---|---|
| rocm provider | `lib.rocmProviders` (`therock-bin`, `therock-source`, `nixpkgs`) | `therock-bin` |
| python provider | `lib.pythonProviders` (`therock-wheels`; future stubs in `lib.pythonProviderStubs`) | `therock-wheels` |
| GPU target | `pkgs/therock/targets.nix` | `gfx1151` |

Per-target packages (default providers) are available under
`legacyPackages`:

```bash
nix build .#legacyPackages.x86_64-linux.gfx1103.llama-cpp-rocm
```

Only targets with matching TheRock binary/Python source pins expose
TheRock-shaped packages such as `therock-rocm`, `ds4-rocm`, `mlx-rocm`, and
`vllm-rocm`. At the moment those pins exist for `gfx1151`; `llama-cpp-rocm`
is available for every target listed in `pkgs/therock/targets.nix`.

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
- `nixosModules.ec-su-axb35`, `ryzenadj`, `tuning` — Strix Halo hardware modules

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
nix build .#hydraBenchmarkJobs.x86_64-linux.bench-mlx-rocm-gfx1151-gemm-smoke
```

## Development

```bash
nix develop
nix fmt -- --fail-on-change
nix flake check --no-build
```
