# nix-strix-halo

Small Nix flake for Strix Halo / Sixunited AXB35 systems.

It provides:

- generic `llama-cpp` outputs for Linux and macOS
- Linux-only ROCm outputs, including generic ROCm builds and Strix Halo `gfx1151` narrowed builds
- TheRock ROCm/PyTorch packaging for configured Linux GPU targets
- source-built vLLM ROCm packaging against the TheRock Python/ROCm stack
- FastFlowLM packaging for AMD XDNA2 NPU inference
- DS4-HIP packaging for DwarfStar 4 ROCm inference
- NixOS modules for llama.cpp RPC servers, EC fan/power controls, Ryzen power limits, RAID0 disk layout, system tuning, and benchmark hosts
- cross-platform benchmark helpers and derivations for reproducible local model/tool runs
- a minimal `fevm-faex9` NixOS example builder

## Outputs

```bash
nix flake show
```

Main package outputs:

- `packages.aarch64-darwin.default`
- `packages.aarch64-darwin.llama-cpp`
- `packages.x86_64-linux.default`
- `packages.x86_64-linux.ds4-rocm`
- `packages.x86_64-linux.ds4-rocm-gfx1151`
- `packages.x86_64-linux.fastflowlm`
- `packages.x86_64-linux.llama-cpp`
- `packages.x86_64-linux.llama-cpp-rocm`
- `packages.x86_64-linux.llama-cpp-rocm-gfx1151`
- `packages.x86_64-linux.llama-cpp-vulkan`
- `packages.x86_64-linux.ec-su-axb35-monitor`
- `packages.x86_64-linux.strix-halo-mes-firmware`
- `packages.x86_64-linux.vllm-rocm-therock-gfx1151`
- `packages.x86_64-linux.xrt-amdxdna`

Main app outputs:

- `apps.aarch64-darwin.llama-cli`
- `apps.aarch64-darwin.llama-server`
- `apps.x86_64-linux.llama-cli`
- `apps.x86_64-linux.llama-server`
- `apps.x86_64-linux.llama-cli-rocm`
- `apps.x86_64-linux.llama-server-rocm`
- `apps.x86_64-linux.llama-cli-gfx1151`
- `apps.x86_64-linux.llama-server-gfx1151`
- `apps.x86_64-linux.flm`

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
- `packages.x86_64-linux.vllm-rocm-therock-gfx1151`
- `packages.x86_64-linux.ds4-rocm-gfx1151`

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

The vLLM package exposes feature flags through normal derivation override arguments:

```nix
pkgs.vllm-rocm-therock-gfx1151.override {
  benchSupport = true; # default
  audioSupport = true; # default
}
```

`benchSupport` and `audioSupport` are enabled by default. Triton support is required and comes from the TheRock Python stack plus the pinned Triton kernels source. AITer, RIXL, and other unported upstream extras are exposed as disabled flags that fail with a clear unsupported-feature message if enabled before their packages are added.

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

## FastFlowLM

`packages.x86_64-linux.fastflowlm` provides the `flm` CLI, with `xrt-amdxdna` and `tokenizers-cpp` packaged in the same overlay. It expects FastFlowLM models to be pre-pulled outside the Nix build sandbox, typically under `/models/flm`.

```bash
nix run .#flm -- list
```

The NixOS server module requires IOMMU passthrough for the AMD XDNA device:

```nix
{
  imports = [
    inputs.nix-strix-halo.nixosModules.default
    inputs.nix-strix-halo.nixosModules.fastflowlm-server
  ];

  boot.kernelParams = [ "iommu=pt" ];

  services.fastflowlm-server = {
    enable = true;
    model = "llama3.2:1b";
    modelsPath = "/models/flm";
  };
}
```

## Strix Halo Modules

- `nixosModules.ec-su-axb35`: kernel module and optional monitor script for the Sixunited AXB35 EC
- `nixosModules.ryzenadj`: power and curve optimizer settings through `ryzenadj`
- `nixosModules.tuning`: high-performance kernel defaults, pinned Strix Halo MES firmware, and TuneD defaults
- `nixosModules.disko-raid0`: dual-NVMe RAID0 Disko layout
- `nixosModules.fastflowlm-server`: FastFlowLM OpenAI-compatible server for AMD XDNA NPUs
- `nixosModules.benchmark-runner`: local benchmark runner capabilities and sandbox device access
- `nixosModules.benchmark-executor` / `darwinModules.benchmark-executor`: remote builder setup for machines that submit benchmark builds

`disko-raid0` defines Disko options but does not import Disko itself. Callers must provide `disko.nixosModules.disko` before using it. It targets `/dev/nvme0n1` and `/dev/nvme1n1`; treat it as machine-specific.

## Example Host

The `fevm-faex9` example requires a caller-provided Disko module:

```bash
nix build --impure --file ./examples/fevm-faex9 \
  --arg diskoModule '(builtins.getFlake "github:nix-community/disko").nixosModules.disko' \
  config.system.build.toplevel
```

## Benchmarks

Benchmark derivations are generated from structured tool/model/scenario records and exposed under `benchmarks.<system>`. The default cross-platform matrix runs CPU `llama-bench` against local GGUF files under `/models`. Linux also exposes this flake's ROCm and Vulkan llama.cpp tool variants.

```bash
nix build .#benchmarks.x86_64-linux.bench-llama2-7b-llama-cpp-cpu-b512-fa1
cat result/stdout.txt
cat result/metadata.json
```

DS4-HIP benchmarks expect a caller-provided GGUF model at `/models/ds4/ds4flash.gguf`
and a benchmark runner advertising `gfx1151` with `/models` mounted into
the Nix sandbox:

```bash
nix build .#benchmarks.x86_64-linux.bench-deepseek-v4-flash-ds4-rocm-gfx1151-smoke
cat result/results.csv
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

Use `benchmark.runners` on NixOS benchmark hosts to add Nix system features and sandbox device access required by benchmark derivations:

```nix
boot.kernelParams = [
  "iommu=off"
];

benchmark.modelsPath = "/models";

benchmark.runners.my-strix-gpu = {
  gpus = [
    {
      type = "amd";
      arch = "1151";
    }
  ];
};
```

Benchmark runners create `benchmark.modelsPath` if needed and add it to Nix
build sandboxes. They do not manage the model files themselves; benchmark
derivations define the expected model paths under that root.

For FastFlowLM NPU benchmarks, advertise a separate runner with IOMMU passthrough rather than reusing an IOMMU-off GPU runner:

```nix
boot.kernelParams = [ "iommu=pt" ];

benchmark.runners.my-strix-npu = {
  requireIommuOff = false;
  npus = [
    {
      type = "amd";
      arch = "xdna2";
    }
  ];
};
```

By default, benchmark runners assert `iommu=off`; NPU hosts should use a separate runner profile with IOMMU passthrough.

Use `benchmark.executor` on machines that submit benchmark builds to register runner hosts as Nix remotes:

```nix
benchmark.executor = {
  enable = true;
  builders.my-gpu-runner = {
    hostName = "gpu-runner.example";
    sshUser = "grw";
    systemFeatures = [
      "rocm"
      "kvm"
      "big-parallel"
    ];
    gpus = [
      {
        type = "amd";
        arch = "1151";
      }
    ];
  };
};
```

## Development

```bash
nix develop
nix fmt -- --fail-on-change
nix flake check --no-build
```

## CI

CI is driven by the flake's `checks` output. The buildbot should build `.#checks.<system>.*` for the systems it owns.

The checks include source formatting/linting, representative package builds, benchmark metadata validation, executor builder translation, and Linux benchmark-runner module composition. Hardware/model-dependent benchmark derivations remain under `benchmarks` because they require local models and host-specific device access.
