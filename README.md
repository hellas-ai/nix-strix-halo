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
| `amdtop` | btop/nvitop-style TUI for AMD CPU, GPU and XDNA NPU telemetry |
| `xrt`, `xrt-amdxdna`, `tokenizers-cpp`, `strix-halo-mes-firmware`, `ec-su-axb35-monitor` | hardware support bits |
| `live-iso` | USB-flashable strix-halo live system |
| Darwin: `llama-cpp`, `llama-cpp-master`, `llama-cpp-master-rdma`, `mlx`, `mlx-metal`, `vllm-metal`, `ds4`, `jaccl` | cross-platform / Metal |

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

### Darwin vLLM Metal and JACCL

`packages.aarch64-darwin.vllm-metal` combines the upstream, version-matched
vLLM 0.28 and vLLM-Metal release wheels with MLX 0.32 and MLX-LM 0.31.3 built
from pinned source. The MLX build links this flake's JACCL package instead of
its bundled copy. Build and perform the model-free regression check with:

```bash
nix build -o result-vllm-metal .#packages.aarch64-darwin.vllm-metal
nix build -o result-vllm-metal-smoke \
  .#packages.aarch64-darwin.vllm-metal.tests.imports
./result-vllm-metal/bin/python \
  -c 'import mlx.core, mlx_lm, ray, vllm, vllm_metal'
```

The flake also exports `darwinModules.vllm-metal`. This example matches the
correctness-first Qwen3.8 deployment tested on a 128 GiB Mac: it binds only to
localhost, leaves speculative decoding off, admits one sequence, and reserves
85% of unified memory for the official BF16 weights and a 262,144-token context
window.

```nix
{
  imports = [ inputs.nix-strix-halo.darwinModules.vllm-metal ];

  services.vllm-metal = {
    enable = true;
    model = "Qwen/Qwen3.8-27B";
    revision = "706cebd746c4b6f2b1d1f892630867acfdfd3df8";
    servedModelName = "qwen38-dense";
    user = "grw";
    workingDirectory = "/Users/grw";
    environment = {
      HOME = "/Users/grw";
      HF_HOME = "/Users/grw/.cache/huggingface";
    };
    maxModelLen = 262144;
    maxNumSeqs = 1;
    gpuMemoryUtilization = 0.85;
    enablePrefixCaching = true;
    reasoningParser = "qwen3";
    enableAutoToolChoice = true;
    toolCallParser = "qwen3_coder";
  };
}
```

The model is an external runtime input rather than part of the Nix closure.
The example pins the exact Hugging Face snapshot used in the test; its BF16
weights occupy roughly 52 GiB on disk. vLLM downloads it into `HF_HOME` when
the snapshot is not already present.

The prompt and completion share `maxModelLen`; the server does not impose a
separate 8K completion ceiling. A client may advertise a 262,144-token maximum,
but a request still needs to leave room for its system prompt, tools, history,
and at least one generated token. Keep the default localhost bind and reach it
through an authenticated SSH tunnel unless deliberate LAN exposure is needed.

The Metal compiler is an Apple host component and cannot currently be placed
in the Nix sandbox. A builder needs macOS 26.2 or newer, the flake's macOS 26 SDK, and a
working `metal`/`metallib` toolchain. Install the latter once with
`xcodebuild -downloadComponent MetalToolchain`. The MLX derivation deliberately
uses `__noChroot`; do not claim bit-reproducibility across different installed
Apple Metal toolchains.

For two-host MLX tensor parallelism, assign IPv4 addresses to every direct
Thunderbolt interface used by JACCL. JACCL selects the resulting IPv4-mapped
GID dynamically and reports a clear error when none exists. A dual-link
hostfile has this shape (device names are host-specific); the same lab
configuration is available as `examples/jaccl-ring-dual.json`:

```json
{
  "backend": "jaccl-ring",
  "envs": ["MLX_METAL_FAST_SYNCH=1"],
  "hosts": [
    {"ssh": "10.55.0.1", "ips": ["10.55.0.1"],
     "rdma": [null, ["rdma_en1", "rdma_en2"]]},
    {"ssh": "10.55.0.2", "ips": [],
     "rdma": [["rdma_en3", "rdma_en5"], null]}
  ]
}
```

Before loading a model, validate the data path with the included 4 KiB exact
all-reduce. The Python executable and script must exist at the same paths on
both hosts.

```bash
./result-vllm-metal/bin/mlx.launch \
  --hostfile examples/jaccl-ring-dual.json -- \
  ./result-vllm-metal/bin/python examples/mlx-jaccl-allreduce.py
```

For a larger transport sanity check, append for example
`--elements 67108864 --warmup 3 --iterations 20`. The reported
`payload_gib_s` is application payload per rank, not physical link line rate.

`vllm-metal` does not currently expose cross-host tensor parallelism. To test
the same checkpoint with MLX-LM's tensor-parallel weight sharding over JACCL,
put the model at the same path on both hosts and run:

```bash
./result-vllm-metal/bin/mlx.launch \
  --hostfile examples/jaccl-ring-dual.json -- \
  ./result-vllm-metal/bin/python examples/mlx-lm-tp-generate.py \
  --model /absolute/path/on/both/hosts/model
```

Each rank constructs the model but `sharded_load` partitions supported layers
before evaluating their weights. This is a separate MLX-LM execution path, not
distributed vLLM serving, disaggregated prefill, or a Ray deployment. The
current MLX-LM Qwen3.5 loader discards checkpoint MTP tensors, so this probe
uses ordinary target-model decoding rather than MTP speculation.

Treat a successful process exit as transport evidence only: compare greedy
output against an identical single-host run before serving the model. In the
tested MLX 0.32.0 stack, TP=2 diverged from the single-host greedy output for
both the Qwen3.8 checkpoint and a Llama 3.2 1B control, despite exact JACCL
all-reduce. Cross-host MLX-LM TP is therefore not enabled for serving here.

Keep the two data cables in distinct `/30` subnets. Bridging both without STP
creates an L2 loop. For example, use `10.56.1.1/30` and `10.56.2.1/30` on the
first host, and `.2/30` on the corresponding interfaces of the second host:

```bash
sudo ifconfig en1 inet 10.56.1.1/30 up
sudo ifconfig en2 inet 10.56.2.1/30 up
# second host: en3 -> 10.56.1.2/30; en5 -> 10.56.2.2/30
```

The model configuration supplies its 262,144-token context window. Launch an
MLX-LM server with a 262,144-token default generation ceiling and a 4 GiB
retained prompt-cache budget per rank as follows:

```bash
./result/bin/mlx.launch --hostfile jaccl-ring-dual.json -- \
  ./result/bin/python -m mlx_lm server \
  --model Qwen/Qwen3.8-27B \
  --host 0.0.0.0 --port 8081 --max-tokens 262144 \
  --prompt-cache-size 8 --prompt-cache-bytes 4294967296
```

In the measured MBP/Goblin lab pair, either 80 Gbit/s cable sustained about 62.5
Gbit/s of JACCL collective traffic and both sustained 103.6 Gbit/s. The full
closure is 3.7 GiB. The official Qwen3.8-27B BF16 weights occupy roughly 52 GiB
on disk. TP=2 shards supported layers in memory, while the checkpoint must
still be readable by both ranks and active KV state and cache allocations
remain per-rank costs. A 131,017-token prefill on ordinary MLX-LM did not finish
in 1 hour 49 minutes, so 256K is an accepted window, not a demonstrated useful
throughput target for this backend.

| Capability | Verified result |
|---|---|
| MLX/JACCL TP=2 over two Thunderbolt links | Works |
| Exact-prefix reuse | Works; 44.913 s cold became 0.277 s warm |
| Ray import and single-node task | Works |
| Ray multi-node macOS control plane | Unsupported upstream; worker lost GCS after 60 s |
| MTP in `Qwen/Qwen3.8-27B` | Unavailable on this path; MLX-LM discards the checkpoint's MTP tensors |
| Distributed oMLX MTP/speculation | Unavailable; oMLX rejects it in distributed mode |

Do not expose this MLX-LM TP probe as an agent endpoint until its greedy output
matches the single-host reference. The single-host vLLM service above is the
validated OpenAI-compatible path. Its prompt and completion share the 262,144-
token context window; clients may advertise the same output ceiling, but each
request must leave room for the system prompt, tools, and conversation.

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
