# Darwin vLLM Metal and JACCL

`packages.aarch64-darwin.vllm-metal` combines the upstream, version-matched
vLLM 0.29.0 and vLLM-Metal 0.29.0 release wheels with MLX 0.32.1 and the
release-matched MLX-LM revision built from pinned source. The MLX build links this flake's JACCL package instead of
its bundled copy. Build and perform the regression checks without downloading model weights with:

```bash
nix build -o result-vllm-metal .#packages.aarch64-darwin.vllm-metal
nix build -o result-vllm-metal-smoke \
  .#packages.aarch64-darwin.vllm-metal.tests.imports
nix build -o result-vllm-metal-native \
  .#packages.aarch64-darwin.vllm-metal.tests.metal
nix build -o result-vllm-metal-server \
  .#packages.aarch64-darwin.vllm-metal.tests.server
nix build .#checks.aarch64-darwin.vllm-metal-module
./result-vllm-metal/bin/python \
  -c 'import mlx.core, mlx_lm, ray, vllm, vllm_metal'
```

The 0.29.0 package passed these checks on an M4 Mac running macOS 27.0 on
2026-09-20. The native test verified exact KV-cache scatter in float32,
float16, and bfloat16 with JACCL available. The HTTP test matched two greedy
completions against a PyTorch reference using a locally generated Llama model.
A separate real-model test of `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca` matched two simultaneous 16-token
completion requests against PyTorch float32, and streaming chat matched the
non-streaming response. These results cover single-host serving; the older
multi-host measurements below have not been repeated with this release.

The flake also exports `darwinModules.vllm-metal`. Its defaults are localhost,
an 8,192-token context, one sequence, and no prefix caching or speculation.
Activation creates the state directory for the configured service account;
model downloads, caches, and logs stay there. The host configuration owns the
account itself. This larger-context example requires a 128 GiB Mac and reserves
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
    user = "vllm"; # Existing unprivileged service account.
    stateDirectory = "/var/lib/vllm-metal";
    maxModelLen = 262144;
    maxNumSeqs = 1;
    gpuMemoryUtilization = 0.85;
    enablePrefixCaching = false;
    reasoningParser = "qwen3";
    enableAutoToolChoice = true;
    toolCallParser = "qwen3_coder";
  };
}
```

Prefix caching is opt-in: upstream still reports parity failures for some
hybrid models. The module passes an explicit disable flag when it is off.

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
working `metal`/`metallib` toolchain in a system Xcode or cryptex location.
The build does not use compiler downloads inside personal home directories.
Install the toolchain once with
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

This package is validated for single-host serving. vLLM-Metal 0.29 includes
pipeline and data parallel execution paths, but rejects tensor parallel sizes
greater than one; those multi-host serving paths are not validated here. The
included MLX-LM harness can compare a single-host greedy reference with
tensor-parallel weight sharding over JACCL. First capture the reference:

```bash
./result-vllm-metal/bin/python examples/mlx-lm-tp-generate.py \
  --reference --model /absolute/path/to/model
```

Then put the checkpoint at the same path on both hosts and run:

```bash
./result-vllm-metal/bin/mlx.launch \
  --hostfile examples/jaccl-ring-dual.json -- \
  ./result-vllm-metal/bin/python examples/mlx-lm-tp-generate.py \
  --model /absolute/path/on/both/hosts/model
```

Both commands emit JSON containing the generated token IDs. Each rank
constructs the model, but `sharded_load` partitions supported layers before
evaluating their weights. This is a correctness harness for MLX-LM, not a
distributed vLLM server, disaggregated prefill, or Ray deployment.

The following measurements are historical results from the original PR
(MLX 0.32.0 and vLLM 0.28), not validation of the updated 0.29.0 stack.
With MLX 0.32.0, TP=2 over JACCL matched the single-host greedy tokens exactly
for the unquantized `Qwen/Qwen3.5-0.8B` checkpoint. The unquantized
`Qwen/Qwen3.8-27B` run did not produce its first token within ten minutes on a
128 GiB/36 GiB host pair because the smaller host was swapping heavily. It is
therefore not presented as a serving configuration. The current MLX-LM loader
also discards this checkpoint's MTP tensors, so the harness uses ordinary
target-model decoding rather than MTP speculation.

Keep the two data cables in distinct `/30` subnets. Bridging both without STP
creates an L2 loop. For example, use `10.56.1.1/30` and `10.56.2.1/30` on the
first host, and `.2/30` on the corresponding interfaces of the second host:

```bash
sudo ifconfig en1 inet 10.56.1.1/30 up
sudo ifconfig en2 inet 10.56.2.1/30 up
# second host: en3 -> 10.56.1.2/30; en5 -> 10.56.2.2/30
```

In the measured MBP/Goblin lab pair, either 80 Gbit/s cable sustained about 62.5
Gbit/s of JACCL collective traffic and both sustained 103.6 Gbit/s. The tested
0.28 package closure was 3.7 GiB. The official Qwen3.8-27B BF16 weights occupy roughly 52 GiB
on disk. TP=2 shards supported layers in memory, while the checkpoint must
still be readable by both ranks and active KV state and cache allocations
remain per-rank costs. A 131,017-token prefill on ordinary MLX-LM did not finish
in 1 hour 49 minutes, so 256K is an accepted window, not a demonstrated useful
throughput target for this backend.

| Capability | Historical result (MLX 0.32.0 / vLLM 0.28) |
|---|---|
| JACCL exact all-reduce over two Thunderbolt links | Works |
| MLX-LM TP=2 greedy correctness | Exact match for unquantized Qwen3.5-0.8B |
| MLX-LM TP=2 Qwen3.8-27B serving | Not viable on the tested 36 GiB worker |
| Exact-prefix reuse | Works; 44.913 s cold became 0.277 s warm |
| Ray import and single-node task | Works |
| Ray multi-node macOS control plane | Worker lost GCS after 60 s in this experiment |
| MTP in `Qwen/Qwen3.8-27B` | Unavailable on this path; MLX-LM discards the checkpoint's MTP tensors |
| Distributed oMLX MTP/speculation | Unavailable; oMLX rejects it in distributed mode |

The 262,144-token example is an explicit capacity setting, not a long-context
throughput guarantee. Revalidate a production model after updating the stack.


## Updating the package

The two release wheels, MLX source, and Python lock form one compatibility
unit. The native plugin uses MLX private headers, so even an MLX patch release
requires a matching wheel. Package evaluation rejects a source/lock version
mismatch. MLX-LM uses the exact source revision declared by the plugin and
recorded in `uv.lock`; there is no second source override.

1. Select a stable vLLM-Metal release and its matching vLLM core wheel. Update
   both URLs and the project version in `pkgs/vllm-metal/pyproject.toml`.
2. Set `mlx-metal-src` in `flake.nix` to the exact MLX version required by that
   release. Run `nix flake lock`.
3. Run `nix shell --inputs-from . nixpkgs#uv --command uv lock --project pkgs/vllm-metal --python 3.12`.
   Review the resulting dependency changes and keep source hashes committed.
4. Build the package, import check, native Metal smoke, serving smoke, and
   module check using
   the commands above. The native smoke loads the wheel's compiled kernels and
   checks exact KV-cache scatter results for float32, float16, and bfloat16.
   The serving smoke generates a tiny local Llama checkpoint, starts the HTTP
   API, and compares two greedy completions against a PyTorch reference.
5. Re-run serving and any model-specific or multi-host validation needed by the
   deployment. Historical measurements above are not a substitute for this.

Dependabot covers the flake tooling inputs. It does not coordinate the direct
wheel URLs in this Python environment; updating those remains the procedure
above. A standalone MLX input update will fail the ABI guard until the paired
wheel and lock are updated.
