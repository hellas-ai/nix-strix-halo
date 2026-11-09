# Relocatable Triton warm-cache seeds

This optional CPU-only utility exports a quiescent Triton file cache, verifies a
manifest, and installs it into a new writable cache directory. It does not compile
kernels, capture graphs, prove kernel correctness, or guarantee that a workload
will hit the seed. Nix builds the utility and its CPU tests; generating the seed
requires a separate GPU warmup with the exact serving package and configuration.

Create an identity JSON with exactly these fields:

```json
{"runtime":"/nix/store/<exact-sglang-runtime>","target":"gfx1151","configuration":{"launcher":"/nix/store/<exact-launcher>","toolchain":"/nix/store/<exact-wheel-set>","profile":"plain-tp4-c8","environment":{}}}
```

Record actual store paths, the full AMD target including feature suffixes if
present, and relevant graph shapes, launch flags and compilation-affecting
environment values in `configuration`. Use the identical reviewed identity for
export and installation. The utility compares identities exactly; it cannot
independently discover or attest the live process configuration. Triton's own
keys still decide whether a compiled specialization can be reused.

```sh
glm53-triton-cache-seed export --identity identity.json --cache "$TRITON_CACHE_DIR" --seed ./seed
glm53-triton-cache-seed verify --identity identity.json --seed ./seed
glm53-triton-cache-seed install --identity identity.json --seed ./seed --dest ./fresh-cache
```

Both outputs must be absent, with an existing parent directory. Install publishes
one complete directory by an atomic rename. When no-replace rename is unsupported,
publishers use a persistent per-destination advisory lock; unrelated writers must
cooperate with that lock to prevent clobbering. Select the directory as
`TRITON_CACHE_DIR` before starting the process. Runtime misses can write there
normally. This intentionally does not merge into a live cache. The seed manifest
hashes the complete regular-file payload and rejects symlinks, foreign group
children, unexpected files, corruption and mismatched identity. Hashes detect
accidental corruption, not a malicious party able to replace the manifest too;
transport and pin the seed through a trusted Nix store path or separately pinned
content hash. Export only after all warmup writers stop. Runtime graph captures,
model weights, credentials and non-Triton caches are outside this format.

To qualify an actual seed, compare cold and seeded starts with the same runtime,
record cache hits/misses and graph capture separately, and test all required
C1/C2/C4/C8 and representative prefill shapes and contexts on the real model.
Repeat correctness and full-model numerical checks. Untested shapes may JIT.
