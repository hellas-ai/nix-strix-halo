# nix-strix-halo

Small Nix flake for Strix Halo / Sixunited AXB35 systems.

It provides:

- a nixpkgs overlay for `llama-cpp` with ROCm enabled for `gfx1151`
- an opt-in TheRock ROCm preview SDK package for Strix Halo (`gfx1151`)
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
- `packages.x86_64-linux.therock-rocm-gfx1151`
- `packages.x86_64-linux.therock-rocm-gfx1151-env`
- `packages.x86_64-linux.vllm-env-therock-runtime-gfx1151`
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
- `apps.x86_64-linux.therock-rocm-gfx1151-env`
- `apps.x86_64-linux.vllm-therock-runtime-gfx1151`

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

## TheRock ROCm Preview

The default ROCm path stays on nixpkgs ROCm. TheRock is packaged separately so
we can test newer ROCm builds without changing system packages or invalidating
the stable vLLM/RCCL setup.

Current pin:

- `therock-rocm-gfx1151`: TheRock ROCm `7.13.0a20260515` for Strix Halo

Use the environment wrapper for ad-hoc tests of the prebuilt SDK pin:

```bash
nix run .#therock-rocm-gfx1151-env -- rocminfo
nix run .#therock-rocm-gfx1151-env -- hipcc --version
nix run .#therock-rocm-gfx1151-env -- bash
```

There is also a runtime-only vLLM compatibility wrapper:

```bash
nix run .#vllm-therock-runtime-gfx1151 -- --help
nix run .#vllm-therock-runtime-gfx1151 -- serve Qwen/Qwen2.5-7B-Instruct
```

That wrapper runs the existing `usb4-rdma` Strix vLLM environment with TheRock
first on `PATH` and `LD_LIBRARY_PATH`. It is useful for runtime compatibility
testing, but it is not the same as rebuilding PyTorch, vLLM, AITER, or Triton
against TheRock. Expect it to build the current nixpkgs ROCm vLLM closure.

Update the prebuilt SDK pin with:

```bash
python3 scripts/update-therock-rocm.py --target gfx1151 --series 7.13
```

This packages the SDK/runtime tarball only. The real "ROCm 7.13 vLLM" path is
the source-build flow below, followed by a `rocmPackages` overlay that rebuilds
the native Python stack against the source-built ROCm closure.

For a real source build, keep the two-stage pinning pattern:

```bash
python3 scripts/update-therock-rocm-source.py
source=$(nix build .#therock-rocm-source-gfx1151 --no-link --print-out-paths)
python3 scripts/update-therock-rocm-third-party.py --source "$source"
nix build .#therock-amd-llvm-gfx1151
nix build .#therock-rocm-from-source-gfx1151
```

The update scripts record the upstream source graph in JSON:

- `therock-rocm-source-sources.json`: TheRock URL, ref, resolved revision,
  target, source fetch policy, and recursive staged-source hash.
- `therock-rocm-third-party-sources.json`: third-party archive hashes, SPIR-V
  headers, and the `esmi_ib_library` revision needed by the TheRock build.

The flake reads those JSON files during evaluation. The ROCm build itself must
not discover new network dependencies or mutate the pins.

The source build is packaged in two stages for faster iteration:

- `therock-amd-llvm-gfx1151` builds TheRock's AMD LLVM, COMGR, and hipcc
  stages and exports them as a prebuilt TheRock build-tree slice.
- `therock-rocm-from-source-gfx1151` imports that slice before CMake configure,
  so TheRock's native `.prebuilt` mechanism skips the compiler rebuild while
  still building the runtime, math, and RCCL layers from source.

On a new TheRock revision, run the source build once with the reset hash, copy
Nix's reported recursive source hash back with:

```bash
python3 scripts/update-therock-rocm-source.py --hash sha256-...
```

Then rebuild `.#therock-rocm-source-gfx1151`, refresh the third-party JSON from
that staged source tree, review the JSON diff, and build
`.#therock-rocm-from-source-gfx1151`. The ROCm build package consumes those
pinned source snapshots without network access. The default `vllm` source fetch
disables debug, media, IREE, and ML-framework source trees, leaving the core
ROCm sources needed for compiler/runtime/math/RCCL:

```text
base, compilers, rocm-systems, rocm-libraries, math-libs
```

That is the ROCm side needed before exposing the result as a nixpkgs-compatible
`rocmPackages` set and rebuilding PyTorch, Triton, AITER, and vLLM against it.
That is intentionally separate from the runtime-only wrapper above.

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
