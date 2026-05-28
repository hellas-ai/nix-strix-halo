# nix-strix-halo

[![Hydra quick][hydra-quick-badge]][hydra-quick]
[![Hydra full][hydra-full-badge]][hydra-full]

[hydra-quick]: https://hydra.hellas.ai/job/hellas/nix-strix-halo/x86_64-linux.pr-quick.all
[hydra-quick-badge]: https://img.shields.io/endpoint?label=hydra%20quick&url=https%3A%2F%2Fhydra.hellas.ai%2Fjob%2Fhellas%2Fnix-strix-halo%2Fx86_64-linux.pr-quick.all%2Fshield
[hydra-full]: https://hydra.hellas.ai/job/hellas/nix-strix-halo/x86_64-linux.pr-full.all
[hydra-full-badge]: https://img.shields.io/endpoint?label=hydra%20full&url=https%3A%2F%2Fhydra.hellas.ai%2Fjob%2Fhellas%2Fnix-strix-halo%2Fx86_64-linux.pr-full.all%2Fshield

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
- a USB-bootable Strix Halo live ISO with the overlay tooling pre-installed

## Outputs

```bash
nix flake show
```

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

On macOS, the overlay intentionally exposes only generic non-ROCm outputs such as `llama-cpp`, CPU `llama-cli`/`llama-server` apps, DS4 Metal outputs, MLX/JACCL library outputs, and CPU benchmark derivations. ROCm, TheRock, EC, and firmware outputs are Linux-only.

The DS4 Metal package builds against nixpkgs' `apple-sdk_26` by default because
it uses newer Metal APIs than the default Darwin SDK. Override `darwinSdk`,
`darwinSdkRoot`, or `darwinDeploymentTarget` for a different SDK policy.

## MLX And JACCL Libraries

The flake exposes library packages for shared MLX/JACCL development across Linux and macOS:

- `packages.x86_64-linux.jaccl`: standalone JACCL library from the pinned MLX source, built against `rdma-core`
- `packages.x86_64-linux.mlx` / `mlx-rocm`: compatibility aliases for the default MLX ROCm target
- `packages.x86_64-linux.mlx-rocm-gfx1151`: MLX with the ROCm backend and portable JACCL support, narrowed for Strix Halo
- `packages.aarch64-darwin.jaccl`: standalone JACCL library from the same pinned MLX source, built against Apple's RDMA library
- `packages.aarch64-darwin.mlx` / `mlx-metal`: MLX and its Metal backend built from the pinned upstream MLX source

On Linux, this flake composes the `thunderbolt-ibverbs` overlay and routes
`rdma-core` to `rdma-core-usb4`, so ROCm/RCCL, JACCL, and MLX ROCm builds see
the USB4 libibverbs provider.

## Thunderbolt RDMA

This flake composes the clean `thunderbolt-ibverbs` package set through a flake
input and exposes the patched kernel, kernel module, `rdma-core-usb4`, perftest,
and bench tools as package outputs. The live ISO boots the
patched kernel and installs the module/userspace tools, but leaves
`thunderbolt_ibverbs` unloaded at boot; use
`sudo thunderbolt-ibverbs-reload-system` on the live system when you want to
claim the Thunderbolt services for an RDMA run.

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
  otelSupport = true; # default, required by upstream vLLM
}
```

`benchSupport`, `audioSupport`, and `otelSupport` are enabled by default. Triton support is required and comes from the TheRock Python stack plus the pinned Triton kernels source. AITer, RIXL, and other unported upstream extras are exposed as disabled flags that fail with a clear unsupported-feature message if enabled before their packages are added.

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

## Live ISO

`examples/configuration.nix` builds a USB-flashable Strix Halo NixOS ISO on
top of nixpkgs' `installer/cd-dvd/installation-cd-base.nix`. It boots into a
NixOS live system with the Strix Halo tuning, MES firmware, AXB35 EC driver,
and the `llama-cpp-rocm-gfx1151`, `llama-cpp-vulkan`, `fastflowlm`, and
`ec-su-axb35-monitor` packages already installed.

```bash
nix build .#live-iso
sudo dd if=$(readlink -f result)/iso/*.iso of=/dev/sdX bs=4M conv=fsync status=progress
```

The latest successful Hydra build of this job is also served as a Hydra
build product. The redirector URL is stable, so it can be linked from docs
or scripts:

```
https://hydra.hellas.ai/job/hellas/nix-strix-halo/x86_64-linux.live-iso/latest-finished/download-by-type/file/iso
```

The squashfs inside the ISO is already zstd-compressed at level 19; the
outer `.iso` wrapper is left uncompressed so it can be flashed with `dd`
directly. Set `isoImage.compressImage = true;` for a `.iso.zst` if you want
to trade a few percent of size for an extra `zstd -d` step on the consumer
side.

The installer profile creates a passwordless `nixos` user (sudo via `wheel`)
and enables `sshd` and NetworkManager. Same as upstream NixOS installer
images: set a password with `passwd` or drop a key in
`~nixos/.ssh/authorized_keys` before exposing the box.

`lib.mkLiveIsoConfiguration` is the reusable helper if you want to add extra
modules or override defaults from a downstream flake.

To smoke-test the image without writing to a USB stick, boot it in QEMU. The
wrapper enables KVM when `/dev/kvm` is writable and accepts pass-through
`qemu-system-x86_64` arguments:

```bash
nix run .#live-iso-vm
# headless / VNC for remote hosts:
nix run .#live-iso-vm -- -display none -vnc :0
# tune resources:
LIVE_ISO_MEM=8G LIVE_ISO_CPUS=8 nix run .#live-iso-vm
```

## Benchmarks

Benchmark derivations are generated from structured tool/model/scenario records and exposed under `benchmarks.<system>`. The default cross-platform matrix runs CPU `llama-bench` against local GGUF files under the platform model root. Linux uses `/models`; Darwin uses `/Users/Shared/models`. Linux also exposes this flake's ROCm and Vulkan llama.cpp tool variants.

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

DS4 Metal benchmarks use the same benchmark shape on macOS. They expect the
GGUF model at `/Users/Shared/models/ds4/ds4flash.gguf` and a Darwin runner
advertising the `metal` and `benchmark` system features:

```bash
nix build .#benchmarks.aarch64-darwin.bench-deepseek-v4-flash-ds4-metal-smoke
cat result/results.csv
```

MLX GPU smoke benchmarks run a small float32 GEMM and verify the result on
the GPU backend. They do not require model files:

```bash
nix build .#benchmarks.x86_64-linux.bench-mlx-rocm-gfx1151-gemm-smoke
cat result/stdout.txt

nix build .#benchmarks.aarch64-darwin.bench-mlx-metal-gemm-smoke
cat result/stdout.txt
```

vLLM benchmarks use the upstream offline benchmark CLI in throughput and
latency modes. They expect the Hugging Face model cache under
`/models/.cache/huggingface` and run offline against `Qwen/Qwen3-0.6B`:

```bash
nix build .#benchmarks.x86_64-linux.bench-qwen3-0-6b-vllm-rocm-gfx1151-throughput-smoke
cat result/results.json

nix build .#benchmarks.x86_64-linux.bench-qwen3-0-6b-vllm-rocm-gfx1151-latency-smoke
cat result/results.json
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

The checks include source formatting/linting, representative package builds, benchmark metadata validation, executor builder translation, Linux benchmark-runner module composition, and model-free GPU smoke benchmarks for MLX. Model-dependent benchmark derivations remain under `benchmarks` because they require local models and host-specific device access.
