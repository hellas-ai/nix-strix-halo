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
| `llama-cpp{,-rocm,-vulkan,-cuda}` | nixpkgs llama.cpp with RPC enabled, embedded UI disabled + variant flags |
| `llama-cpp-master{,-rocm,-vulkan,-cuda}` | same matrix off llama.cpp HEAD |
| `vllm-rocm` | source-built vLLM 0.23 against TheRock |
| `mlx-rocm` | MLX with the ROCm backend |
| `ds4-rocm` | DwarfStar 4 HIP build |
| `fastflowlm` | XDNA2 NPU CLI (`flm`) |
| `strix-halo-vllm-pair-bench-gfx1151` | two-host vLLM transport-matrix bench driver |
| `therock-rocm`, `therock-python`, `torch-rocm` | TheRock binary SDK + wheels |
| `xrt`, `xrt-amdxdna`, `tokenizers-cpp`, `strix-halo-mes-firmware`, `ec-su-axb35-monitor` | hardware support bits |
| `live-iso` | USB-flashable strix-halo live system |
| Darwin: `llama-cpp`, `llama-cpp-master`, `llama-cpp-master-rdma`, `mlx`, `mlx-metal`, `ds4`, `jaccl` | cross-platform / Metal |

Apps mirror the package names — `apps.x86_64-linux.llama-cli-rocm`,
`llama-rpc-server`, `flm`, `therock-python`, `live-iso-vm`, etc.

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

### Darwin llama.cpp RPC RDMA lab path

`packages.aarch64-darwin.llama-cpp-master-rdma` builds llama.cpp HEAD with
Metal, RPC, and Apple Thunderbolt RDMA (`/usr/lib/librdma.dylib`) enabled.
The verified lab path is a Darwin/Metal client using Apple RDMA to a Linux
ROCm RPC server over Thunderbolt/USB4 verbs. The Apple provider rejects RC queue
pairs on this link, so force UC mode while testing.

```bash
# Linux ROCm RPC server
GGML_RDMA_DEV=usb4_rdma4 \
GGML_RDMA_GID=1 \
GGML_RDMA_QP_TYPE=UC \
GGML_RPC_RDMA_CHUNK_SIZE=4096 \
GGML_RPC_RDMA_RX_DEPTH=1020 \
GGML_RPC_RDMA_TX_DEPTH=32768 \
GGML_RDMA_PATH_MTU=1024 \
GGML_RDMA_REMOTE_LID=2 \
GGML_RPC_SERVER_ONE_SHOT=1 \
nix run .#llama-rpc-server-rocm -- --host 0.0.0.0 --port 50162 --threads 16 --cache

# Darwin/Metal client
GGML_RDMA_DEV=rdma_en3 \
GGML_RDMA_GID=1 \
GGML_RDMA_QP_TYPE=UC \
GGML_RPC_RDMA_CHUNK_SIZE=4096 \
GGML_RPC_RDMA_RX_DEPTH=1020 \
GGML_RPC_RDMA_TX_DEPTH=32768 \
GGML_RDMA_PATH_MTU=1024 \
GGML_RDMA_REMOTE_LID=1 \
nix run .#llama-cli-master-rdma -- --rpc <worker-ip>:50162 --list-devices

GGML_RDMA_DEV=rdma_en3 \
GGML_RDMA_GID=1 \
GGML_RDMA_QP_TYPE=UC \
GGML_RPC_RDMA_CHUNK_SIZE=4096 \
GGML_RPC_RDMA_RX_DEPTH=1020 \
GGML_RPC_RDMA_TX_DEPTH=32768 \
GGML_RDMA_PATH_MTU=1024 \
GGML_RDMA_REMOTE_LID=1 \
nix run .#llama-cli-master-rdma -- \
  -m ~/models/qwen3/qwen3-0.6b-q4_k_m.gguf \
  --rpc <worker-ip>:50162 \
  --device RPC0 \
  -ngl 99 \
  -p "Write one short sentence about RDMA." \
  -n 8 \
  --temp 0 \
  --single-turn \
  --no-display-prompt \
  --no-warmup
```

Before starting the client, `system_profiler SPThunderboltDataType` should show
a connected peer and `ibv_devinfo -d rdma_en3` should show the Apple port as
`PORT_ACTIVE`. The Linux worker needs a route back to the Apple RDMA address, for
example `ip route replace 10.0.5.3/32 dev ardma0`.
`GGML_RDMA_REMOTE_LID` is currently a lab override for the peer port's
`port_lid`; use the LID reported by `ibv_devinfo` for the peer endpoint.
Use 4 KiB fixed frames with a receive depth of 1020 on both endpoints for the
current Apple UC path; 16 KiB receive WQEs currently fail Darwin RTR on this
link. Fixed-frame ACK/retry stays on the RDMA data plane and is
enabled by default for UC/fixed-frame mode: each data frame carries a sequence,
receivers ACK the sequence after reposting the receive slot, senders wait for
the matching ACK, and a timed-out frame is retransmitted. Run the server one
client per process with `GGML_RPC_SERVER_ONE_SHOT=1`; pair it with systemd
`Restart=always` for repeated client runs. Set `GGML_RPC_RDMA_FRAME_ACK=0` only
for comparison runs, or override pieces explicitly with
`GGML_RPC_RDMA_SEND_FRAME_ACK`, `GGML_RPC_RDMA_WAIT_FRAME_ACK`,
`GGML_RPC_RDMA_ACK_TIMEOUT_US`, and `GGML_RPC_RDMA_ACK_RETRIES`.

### Two-host vLLM pair benchmark

`strix-halo-vllm-pair-bench-gfx1151` drives the multi-host vLLM transport
matrix for real lab runs. It copies the vLLM, GCC, RDMA, and benchmark-script
closures to both hosts, runs the matrix from the master node, then fetches the
result CSV. Scenarios: `qwen-peak`, `llama-tp2-win`, `qwen35-122b-awq-capacity`,
`qwen35-122b-awq-prime`, `minimax-m27-awq-strix-2h`.

```bash
nix run .#strix-halo-vllm-pair-bench-gfx1151 -- \
  --scenario qwen-peak \
  --master grw@strix-1.lan.satanic.link \
  --worker grw@strix-2.lan.satanic.link

# inspect the plan without touching the hosts
nix run .#strix-halo-vllm-pair-bench-gfx1151 -- --scenario qwen-peak --dry-run
```

Each scenario fixes the model, transports (`solo`, `lan_tcp`, `usb4_rdma`),
concurrencies, and vLLM args; results land in `./out-vllm-<scenario>-<ts>/`.

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
nix build .#legacyPackages.x86_64-linux.gfx1030.vllm-rocm # Radeon Pro V620
```

Only targets with matching TheRock binary/Python source pins expose
TheRock-shaped packages such as `therock-rocm`, `ds4-rocm`, `mlx-rocm`, and
`vllm-rocm`. At the moment those pins exist for `gfx1030` (Radeon Pro V620) and
`gfx1151`. Package availability may be narrower where an architecture lacks a
required feature: `ds4-rocm` requires rocWMMA and is not exposed for `gfx1030`.
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
- `nixosModules.ec-su-axb35`, `ryzenadj`, `tuning` — Strix Halo hardware modules

Example RDMA RPC server instance:

```nix
{
  services.llama-cpp-rpc-servers.rdma-worker = {
    enable = true;
    package = pkgs.llama-cpp-master-rocm;
    host = "0.0.0.0";
    port = 50162;
    threads = 16;
    openFirewall = true;
    restart = "always";
    environment = {
      GGML_RDMA_DEV = "usb4_rdma4";
      GGML_RDMA_GID = "1";
      GGML_RDMA_QP_TYPE = "UC";
      GGML_RPC_RDMA_CHUNK_SIZE = "4096";
      GGML_RPC_RDMA_RX_DEPTH = "1020";
      GGML_RPC_RDMA_TX_DEPTH = "32768";
      GGML_RDMA_PATH_MTU = "1024";
      GGML_RDMA_REMOTE_LID = "2";
      GGML_RPC_SERVER_ONE_SHOT = "1";
    };
  };
}
```

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
