{
  description = "Strix Halo NixOS configuration and tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # vLLM/ROCm packaging lives in this flake. The thunderbolt-ibverbs
    # input below is kept to the lower RDMA layer.

    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    ec-su-axb35 = {
      url = "github:cmetz/ec-su_axb35-linux";
      flake = false;
    };

    # thunderbolt-ibverbs is the lower-layer RDMA stack: kernel patches,
    # module packaging, and rdma-core-usb4. Input is named `usb4-rdma`
    # for historical reasons; the underlying repo is now
    # /mnt/Home/src/thunderbolt-ibverbs.
    usb4-rdma = {
      url = "git+file:///mnt/Home/src/thunderbolt-ibverbs";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # llama.cpp upstream master — used to build the `*-master-*` variants
    # below alongside the nixpkgs-pinned source variants.
    llama-cpp-master = {
      url = "github:ggml-org/llama.cpp";
      flake = false;
    };

    # FastFlowLM (NPU LLM runtime). Tracks upstream main; bump with
    # `nix flake update fastflowlm`. NPU kernel binaries shipped under
    # src/lib and src/xclbins are proprietary blobs (see LICENSE_BINARY.txt)
    # that we ship as-is — the host-side runtime above them is MIT.
    fastflowlm = {
      url = "github:FastFlowLM/FastFlowLM";
      flake = false;
    };

    rocm-xio = {
      url = "git+file:///mnt/Home/src/rocm-xio";
      flake = false;
    };

    # XRT and the xdna-driver plugin ship in lockstep through AMD's
    # lemonade PPA; treat their tags as a matched pair. Bump both at
    # once when moving versions. The `git+https://` URL with
    # submodules=1 is required because both repos pull in `aiebu`,
    # `gsl`, `nlohmann-json`, etc. as git submodules; `github:` URLs
    # don't recurse.
    xrt-src = {
      url = "git+https://github.com/Xilinx/XRT?ref=refs/tags/2.21.75&submodules=1";
      flake = false;
    };

    xdna-driver-src = {
      url = "git+https://github.com/amd/xdna-driver?ref=refs/tags/2.21.75&submodules=1";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }@inputs:
    let
      inherit (nixpkgs) lib;

      systems = [ "x86_64-linux" ];
      forAllSystems = lib.genAttrs systems;

      hardwareTargets = import ./lib/hardware.nix { inherit lib; };
      defaultRocmTarget = hardwareTargets.defaultRocmTarget;
      rocmTargets = defaultRocmTarget.buildTargets;
      therockRocmSources = builtins.fromJSON (builtins.readFile ./therock-rocm-sources.json);
      therockPythonWheelSources = builtins.fromJSON (
        builtins.readFile ./therock-python-wheel-sources.json
      );
      therockRocmSourceSources = builtins.fromJSON (builtins.readFile ./therock-rocm-source-sources.json);
      therockRocmThirdPartySources = builtins.fromJSON (
        builtins.readFile ./therock-rocm-third-party-sources.json
      );

      strixAdditionsOverlay =
        final: prev:
        let
          ecPackages = prev.callPackage ./pkgs/ec-su-axb35.nix {
            ec-su-axb35-src = inputs.ec-su-axb35;
          };
          therockGfx1151 = therockRocmSources.linux.gfx1151;

          # Add libibverbs (rdma-core-usb4) to a llama-cpp variant so ggml's
          # RPC backend auto-detects it at cmake configure time and enables
          # GGML_RPC_RDMA. We also force the flag ON explicitly so a future
          # nixpkgs bump that flips the default doesn't silently disable it.
          applyRdmaSupport =
            pkg:
            pkg.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.rdma-core-usb4 ];
              cmakeFlags = (old.cmakeFlags or [ ]) ++ [
                (prev.lib.cmakeBool "GGML_RPC_RDMA" true)
              ];
            });

          # Override `src` to upstream ggml-org master, leaving the rest of
          # the nixpkgs derivation (cmake config, backend flags, deps) intact.
          # The attribute name is set as `pname` so derivation store paths
          # disambiguate from the nixpkgs-pinned variants.
          #
          # Master moved the embedded web UI from tools/server/webui/ to
          # tools/ui/ and made it optional via LLAMA_BUILD_UI. The nixpkgs
          # derivation's npm plumbing hard-codes the old path, so we strip
          # it out and set LLAMA_BUILD_UI=OFF — llama-server still builds
          # (linking a stub llama-ui static lib) and reports UI as disabled
          # at runtime. Restore the UI separately if you actually need it.
          applyMasterSrc =
            attrName: pkg:
            pkg.overrideAttrs (old: {
              pname = attrName;
              version =
                "master-" + (inputs.llama-cpp-master.shortRev or inputs.llama-cpp-master.rev or "unknown");
              src = inputs.llama-cpp-master;
              npmDeps = null;
              nativeBuildInputs = prev.lib.filter (x: x.pname or "" != "npm-config-hook") (
                old.nativeBuildInputs or [ ]
              );
              preConfigure = ''
                prependToVar cmakeFlags "-DLLAMA_BUILD_COMMIT:STRING=${
                  inputs.llama-cpp-master.shortRev or inputs.llama-cpp-master.rev or "master"
                }"
              '';
              cmakeFlags = (old.cmakeFlags or [ ]) ++ [
                (prev.lib.cmakeBool "LLAMA_BUILD_UI" false)
                # LLAMA_BUILD_NUMBER is substituted into common/build-info.cpp
                # as `int LLAMA_BUILD_NUMBER = @LLAMA_BUILD_NUMBER@;`, so it
                # must be a numeric literal. nixpkgs's cmakeFeature already
                # emitted `-DLLAMA_BUILD_NUMBER:STRING=master-6a257d4` (our
                # non-numeric version above); the duplicate here wins by
                # cmake's last-D-flag semantics. Commit ref still lives in
                # LLAMA_BUILD_COMMIT (set by preConfigure).
                (prev.lib.cmakeFeature "LLAMA_BUILD_NUMBER" "0")
              ];
            });

          rocmHardwareTargets = hardwareTargets.rocmTargets;

          # Backend "bases": before applyRdmaSupport / applyMasterSrc.
          mkLlamaCppRocmBaseFor =
            target:
            prev.llama-cpp.override {
              rocmSupport = true;
              rpcSupport = true;
              rocmGpuTargets = target.buildTargets;
            };

          mkTargetedPackage =
            pname: pkg:
            pkg.overrideAttrs (_old: {
              inherit pname;
            });

          mkTargetPackageAttrs =
            prefix: mkPackage:
            builtins.listToAttrs (
              map (target: {
                name = "${prefix}-${target.packageSuffix}";
                value = mkPackage target;
              }) rocmHardwareTargets
            );

          llamaCppRocmTargetPackages = mkTargetPackageAttrs "llama-cpp-rocm" (
            target:
            applyRdmaSupport (
              mkTargetedPackage "llama-cpp-rocm-${target.packageSuffix}" (mkLlamaCppRocmBaseFor target)
            )
          );

          llamaCppMasterRocmTargetPackages = mkTargetPackageAttrs "llama-cpp-master-rocm" (
            target:
            applyRdmaSupport (
              applyMasterSrc "llama-cpp-master-rocm-${target.packageSuffix}" (mkLlamaCppRocmBaseFor target)
            )
          );

          llamaCppRocmBase = mkLlamaCppRocmBaseFor defaultRocmTarget;

          llamaCppRocmTherockBase =
            let
              sdkBase = final.therock-rocm-gfx1151;
              sdk = sdkBase // {
                localGpuTargets = rocmTargets;
                gpuTargets = rocmTargets;
                # llama-cpp's default cmakeFlags read
                # ${rocmPackages.clr.hipClangPath}/clang++. We override
                # CMAKE_HIP_COMPILER below, but the attribute must exist for
                # the override to evaluate.
                hipClangPath = "${sdkBase}/llvm/bin";
              };
              therockRocmPackages = {
                clr = sdk;
                hipblas = sdk;
                rocblas = sdk;
              };
            in
            (prev.llama-cpp.override {
              rocmSupport = true;
              rpcSupport = true;
              rocmGpuTargets = rocmTargets;
              rocmPackages = therockRocmPackages;
            }).overrideAttrs
              (old: {
                pname = "llama-cpp-rocm-therock";
                buildInputs = (old.buildInputs or [ ]) ++ [ sdkBase ];
                cmakeFlags = (old.cmakeFlags or [ ]) ++ [
                  "-DCMAKE_HIP_COMPILER=${sdkBase}/bin/therock-hip-clang++"
                  "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${sdkBase}"
                  # Bake ${sdkBase}/lib into the RUNPATH of every linked
                  # output (executables and shared libraries) so libggml-hip.so
                  # can find libhipblas.so.3/libamdhip64.so.7 at runtime. The
                  # HIP-language link is performed by therock-hip-clang++,
                  # which bypasses cc-wrapper / NIX_LDFLAGS, so setting this
                  # via CMake is the only way to reach it. Also covers the
                  # postInstall shell-completion step which invokes
                  # llama-server before the fixup-phase patchelf would run.
                  "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath,${sdkBase}/lib"
                  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,${sdkBase}/lib"
                ];
                env = (old.env or { }) // {
                  HIP_PATH = "${sdkBase}";
                  HIP_PLATFORM = "amd";
                  # -L lets the host-side link of rpc-server resolve symbols
                  # in libamdhip64.so.7 (pulled in transitively via
                  # libggml-hip.so's DT_NEEDED).
                  NIX_LDFLAGS = "${(old.env or { }).NIX_LDFLAGS or ""} -L${sdkBase}/lib";
                };
              });

          llamaCppVulkanBase = prev.llama-cpp.override {
            vulkanSupport = true;
            rpcSupport = true;
          };
        in
        {
          ec-su-axb35 = ecPackages.kernelModule;
          ec-su-axb35-monitor = ecPackages.monitor;

          # Backend × source-pin matrix. RDMA support is universal (every
          # variant links libibverbs via rdma-core-usb4 and enables
          # GGML_RPC_RDMA). The `master` axis swaps in upstream ggml-org/master
          # via the llama-cpp-master flake input.
          llama-cpp-rocm = applyRdmaSupport llamaCppRocmBase;
          llama-cpp-rocm-therock = applyRdmaSupport llamaCppRocmTherockBase;
          llama-cpp-vulkan = applyRdmaSupport llamaCppVulkanBase;
          llama-cpp-master-rocm = applyRdmaSupport (applyMasterSrc "llama-cpp-master-rocm" llamaCppRocmBase);
          llama-cpp-master-rocm-therock = applyRdmaSupport (
            applyMasterSrc "llama-cpp-master-rocm-therock" llamaCppRocmTherockBase
          );
          llama-cpp-master-vulkan = applyRdmaSupport (
            applyMasterSrc "llama-cpp-master-vulkan" llamaCppVulkanBase
          );

          therock-rocm-gfx1151 = prev.callPackage ./pkgs/therock-rocm-sdk {
            target = "gfx1151";
            inherit (therockGfx1151) version url hash;
          };

          vllm-rocm-lemonade-gfx1151 = prev.callPackage ./pkgs/lemonade-vllm-rocm {
            target = "gfx1151";
            releaseTag = "vllm0.21.0-rocm7.13.0-gfx1151";
          };

          # FastFlowLM NPU stack. xrt+xdna are coupled (matching tags pinned
          # in the flake inputs); FastFlowLM tracks upstream main. The
          # `xrt-amdxdna` alias mirrors the combined symlinkJoin name in
          # PR #513841 for downstream consumers.
          tokenizers-cpp = prev.callPackage ./pkgs/tokenizers-cpp { };

          xrt = prev.callPackage ./pkgs/xrt {
            src = inputs.xrt-src;
            version = "2.21.75";
            xdnaSrc = inputs.xdna-driver-src;
          };

          xrt-amdxdna = final.xrt.xdna;

          fastflowlm = prev.callPackage ./pkgs/fastflowlm {
            inherit (final) tokenizers-cpp xrt;
            src = inputs.fastflowlm;
          };

          vllm-lemonade-prime-cache-gfx1151 = prev.writeShellApplication {
            name = "vllm-lemonade-prime-cache-gfx1151";
            runtimeInputs = [
              prev.coreutils
            ];
            text = ''
              set -euo pipefail

              model="''${1:-Qwen/Qwen3.6-27B}"
              if [ "$#" -gt 0 ]; then
                shift
              fi

              cache_root="''${VLLM_LEMONADE_CACHE_ROOT:-''${XDG_CACHE_HOME:-''${HOME:-/tmp}/.cache}/vllm/lemonade-gfx1151}"
              export VLLM_CACHE_ROOT="''${VLLM_CACHE_ROOT:-$cache_root}"
              export TRITON_CACHE_DIR="''${TRITON_CACHE_DIR:-$VLLM_CACHE_ROOT/triton_cache}"
              export TORCHINDUCTOR_CACHE_DIR="''${TORCHINDUCTOR_CACHE_DIR:-$VLLM_CACHE_ROOT/inductor_cache}"
              export HF_HOME="''${HF_HOME:-/models/.cache/huggingface}"
              export VLLM_LOGGING_LEVEL="''${VLLM_LOGGING_LEVEL:-INFO}"
              export TRITON_CACHE_AUTOTUNING="''${TRITON_CACHE_AUTOTUNING:-1}"

              mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

              echo "Priming Lemonade vLLM ROCm cache"
              echo "  model:             $model"
              echo "  VLLM_CACHE_ROOT:   $VLLM_CACHE_ROOT"
              echo "  TRITON_CACHE_DIR:  $TRITON_CACHE_DIR"
              echo "  HF_HOME:           $HF_HOME"

              exec "${final.vllm-rocm-lemonade-gfx1151}/bin/vllm" bench throughput \
                --model "$model" \
                --dataset-name random \
                --num-prompts "''${VLLM_PRIME_NUM_PROMPTS:-1}" \
                --random-input-len "''${VLLM_PRIME_INPUT_LEN:-512}" \
                --random-output-len "''${VLLM_PRIME_OUTPUT_LEN:-1}" \
                --max-model-len "''${VLLM_PRIME_MAX_MODEL_LEN:-4096}" \
                --gpu-memory-utilization "''${VLLM_PRIME_GPU_MEMORY_UTILIZATION:-0.85}" \
                --trust-remote-code \
                --enforce-eager \
                --limit-mm-per-prompt '{"image":0,"video":0}' \
                "$@"
            '';
          };

          vllm-lemonade-qwen36-27b-cache-gfx1151 =
            let
              vllm = final.vllm-rocm-lemonade-gfx1151;
              rocmCore = "${vllm}/lib/python3.12/site-packages/_rocm_sdk_core";
            in
            prev.stdenv.mkDerivation {
              pname = "vllm-lemonade-qwen36-27b-cache-gfx1151";
              version = vllm.version;

              dontUnpack = true;
              dontConfigure = true;
              dontStrip = true;

              requiredSystemFeatures = [ "gfx1151" ];
              preferLocalBuild = true;
              allowSubstitutes = false;

              buildPhase = ''
                runHook preBuild

                export HOME="$TMPDIR/home"
                export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
                export VLLM_CACHE_ROOT="$TMPDIR/vllm-cache"
                export TRITON_CACHE_DIR="$VLLM_CACHE_ROOT/triton_cache"
                export TORCHINDUCTOR_CACHE_DIR="$VLLM_CACHE_ROOT/inductor_cache"
                export VLLM_LOGGING_LEVEL="INFO"
                export TRITON_CACHE_AUTOTUNING="1"
                export HSA_OVERRIDE_GFX_VERSION="11.5.1"
                export HSA_NO_SCRATCH_RECLAIM="1"
                export HSA_ENABLE_INTERRUPT="0"
                export HIP_PLATFORM="amd"
                export GPU_ARCHS="gfx1151"
                export PYTORCH_ROCM_ARCH="gfx1151"
                export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
                export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"
                export ROCM_HOME="${rocmCore}"
                export ROCM_PATH="$ROCM_HOME"
                export HIP_PATH="$ROCM_HOME"
                export DEVICE_LIB_PATH="${rocmCore}/lib/llvm/amdgcn/bitcode"
                export HIP_DEVICE_LIB_PATH="$DEVICE_LIB_PATH"
                export CC="${prev.stdenv.cc}/bin/cc"
                export CXX="${prev.stdenv.cc}/bin/c++"
                export PATH="${vllm}/bin:${prev.stdenv.cc}/bin:$PATH"
                export LD_PRELOAD="${vllm}/lib/libvllm-rocm-c10-hip-compat.so''${LD_PRELOAD:+ $LD_PRELOAD}"

                mkdir -p \
                  "$HOME" \
                  "$XDG_CACHE_HOME" \
                  "$VLLM_CACHE_ROOT" \
                  "$TRITON_CACHE_DIR" \
                  "$TORCHINDUCTOR_CACHE_DIR"

                ${vllm}/bin/python3 -u - <<'PY'
                import math
                import torch

                from vllm.model_executor.layers.fla.ops import (
                    chunk_gated_delta_rule,
                    fused_post_conv_prep,
                )
                from vllm.model_executor.layers.fla.ops.fused_recurrent import (
                    fused_recurrent_gated_delta_rule_packed_decode,
                )
                from vllm.model_executor.layers.fla.ops.layernorm_guard import (
                    layer_norm_fwd,
                )
                from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
                from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
                    causal_conv1d_fn,
                    causal_conv1d_update,
                )
                from vllm.model_executor.layers.rotary_embedding.mrope import (
                    triton_mrope,
                )
                from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
                    chunked_prefill_paged_decode,
                )
                from vllm.v1.worker.block_table import BlockTable
                from vllm.v1.worker.utils import _zero_kv_blocks_kernel

                device = "cuda"
                dtype = torch.bfloat16
                seq_len = FLA_CHUNK_SIZE

                # Qwen3.6-27B GDN/linear-attention dimensions from config.json.
                num_k_heads = 16
                num_v_heads = 48
                head_k_dim = 128
                head_v_dim = 128
                qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

                print("priming qwen3.6 gdn prefill kernels", flush=True)
                a_log = torch.zeros(num_v_heads, device=device, dtype=torch.float32)
                dt_bias = torch.zeros(num_v_heads, device=device, dtype=torch.float32)
                mixed_qkv = torch.randn(seq_len, qkv_dim, device=device, dtype=dtype)
                a = torch.randn(seq_len, num_v_heads, device=device, dtype=dtype)
                b = torch.randn(seq_len, num_v_heads, device=device, dtype=dtype)

                q, k, v, g, beta = fused_post_conv_prep(
                    conv_output=mixed_qkv,
                    a=a,
                    b=b,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    num_k_heads=num_k_heads,
                    head_k_dim=head_k_dim,
                    head_v_dim=head_v_dim,
                    apply_l2norm=True,
                    output_g_exp=False,
                )

                state = torch.zeros(
                    1,
                    num_v_heads,
                    head_v_dim,
                    head_k_dim,
                    device=device,
                    dtype=torch.float32,
                )
                cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
                out, final_state = chunk_gated_delta_rule(
                    q=q.unsqueeze(0),
                    k=k.unsqueeze(0),
                    v=v.unsqueeze(0),
                    g=g.unsqueeze(0),
                    beta=beta.unsqueeze(0),
                    initial_state=state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=False,
                )
                torch.cuda.synchronize()
                print(
                    "primed qwen3.6 gdn kernels",
                    tuple(out.shape),
                    None if final_state is None else tuple(final_state.shape),
                    flush=True,
                )

                print("priming qwen3.6 recurrent decode kernel", flush=True)
                packed_out = torch.empty(
                    1, 1, num_v_heads, head_v_dim, device=device, dtype=dtype
                )
                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv[:1].contiguous(),
                    a[:1].contiguous(),
                    b[:1].contiguous(),
                    a_log,
                    dt_bias,
                    1.0 / math.sqrt(head_k_dim),
                    torch.zeros(
                        2,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        device=device,
                        dtype=torch.float32,
                    ),
                    packed_out,
                    torch.tensor([1], device=device, dtype=torch.int32),
                    use_qk_l2norm_in_kernel=True,
                )

                print("priming qwen3.6 causal-conv kernels", flush=True)
                conv_dim = qkv_dim
                conv_width = 4
                conv_weight = torch.randn(
                    conv_dim, conv_width, device=device, dtype=dtype
                )
                conv_bias = torch.zeros(conv_dim, device=device, dtype=dtype)
                conv_state = torch.zeros(
                    4,
                    conv_dim,
                    conv_width - 1,
                    device=device,
                    dtype=torch.float32,
                )
                cache_indices = torch.tensor([1], device=device, dtype=torch.int32)
                causal_conv1d_fn(
                    torch.randn(512, conv_dim, device=device, dtype=dtype).t(),
                    conv_weight,
                    conv_bias,
                    conv_state,
                    torch.tensor([0, 512], device=device, dtype=torch.int32),
                    cache_indices=cache_indices,
                    has_initial_state=torch.tensor(
                        [False], device=device, dtype=torch.bool
                    ),
                    activation="silu",
                )
                causal_conv1d_update(
                    torch.randn(1, conv_dim, device=device, dtype=dtype),
                    conv_state,
                    conv_weight,
                    conv_bias,
                    activation="silu",
                    conv_state_indices=cache_indices,
                )

                print("priming qwen3.6 attention prefill/decode kernels", flush=True)
                num_query_heads = 24
                num_kv_heads = 4
                attn_head_dim = 256
                physical_block = 784
                cache_x = 16
                key_cache = torch.randn(
                    2,
                    num_kv_heads,
                    attn_head_dim // cache_x,
                    physical_block,
                    cache_x,
                    device=device,
                    dtype=dtype,
                )
                value_cache = torch.randn(
                    2,
                    num_kv_heads,
                    attn_head_dim,
                    physical_block,
                    device=device,
                    dtype=dtype,
                )
                block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
                seq_lens = torch.tensor([512], device=device, dtype=torch.int32)
                k_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
                v_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
                for query_len in (512, 1):
                    query = torch.randn(
                        query_len,
                        num_query_heads,
                        attn_head_dim,
                        device=device,
                        dtype=dtype,
                    )
                    chunked_prefill_paged_decode(
                        query,
                        torch.randn(
                            query_len,
                            num_kv_heads,
                            attn_head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                        torch.randn(
                            query_len,
                            num_kv_heads,
                            attn_head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                        torch.empty_like(query),
                        "auto",
                        key_cache,
                        value_cache,
                        block_table,
                        torch.tensor([0, query_len], device=device, dtype=torch.int32),
                        seq_lens,
                        512,
                        query_len,
                        k_scale,
                        v_scale,
                        sm_scale=1.0 / math.sqrt(attn_head_dim),
                    )

                print("priming qwen3.6 worker helper kernels", flush=True)
                zero_buf = torch.empty(2048, device=device, dtype=torch.int32)
                _zero_kv_blocks_kernel[(1,)](
                    torch.tensor([zero_buf.data_ptr()], device=device, dtype=torch.uint64),
                    torch.tensor([0], device=device, dtype=torch.int64),
                    1,
                    N_SEGS=1,
                    PAGE_SIZE_EL=1024,
                    BLOCK_SIZE=1024,
                )
                block_table_helper = BlockTable(
                    16,
                    1,
                    64,
                    512,
                    False,
                    torch.device(device),
                    16,
                    1,
                )
                block_table_helper.add_row(list(range(32)), 0)
                block_table_helper.commit_block_table(1)
                block_table_helper.compute_slot_mapping(
                    1,
                    torch.tensor([0, 512], device=device, dtype=torch.int32),
                    torch.arange(512, device=device, dtype=torch.int64),
                )

                print("priming qwen3.6 norm and mrope kernels", flush=True)
                layer_norm_fwd(
                    torch.randn(1, 5120, device=device, dtype=dtype),
                    torch.ones(5120, device=device, dtype=dtype),
                    None,
                    1e-6,
                    z=torch.randn(1, 5120, device=device, dtype=dtype),
                    group_size=None,
                    norm_before_gate=True,
                    is_rms_norm=True,
                )
                layer_norm_fwd(
                    torch.randn(1, 128, device=device, dtype=dtype),
                    torch.ones(128, device=device, dtype=dtype),
                    None,
                    1e-6,
                    group_size=128,
                    is_rms_norm=True,
                )
                triton_mrope(
                    torch.randn(
                        512,
                        num_query_heads * attn_head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                    torch.randn(
                        512,
                        num_kv_heads * attn_head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                    torch.randn(3, 512, 32, device=device, dtype=dtype),
                    torch.randn(3, 512, 32, device=device, dtype=dtype),
                    [11, 11, 10],
                    attn_head_dim,
                    64,
                    True,
                )
                torch.cuda.synchronize()
                print("primed qwen3.6 decode/runtime kernels", flush=True)
                PY

                runHook postBuild
              '';

              installPhase = ''
                runHook preInstall

                mkdir -p "$out"
                cp -R "$VLLM_CACHE_ROOT"/. "$out"/
                find "$out" -type f | sort > "$out/manifest.txt"

                runHook postInstall
              '';

              meta = with prev.lib; {
                description = "Primed vLLM/Triton cache for Qwen3.6-27B on Lemonade ROCm gfx1151";
                platforms = platforms.linux;
              };
            };

          vllm-lemonade-qwen36-27b-gfx1151 = prev.writeShellApplication {
            name = "vllm-lemonade-qwen36-27b-gfx1151";
            runtimeInputs = [
              prev.coreutils
            ];
            text = ''
              set -euo pipefail

              cache_root="''${VLLM_LEMONADE_CACHE_ROOT:-''${XDG_CACHE_HOME:-''${HOME:-/tmp}/.cache}/vllm/lemonade-gfx1151}"
              export VLLM_CACHE_ROOT="''${VLLM_CACHE_ROOT:-$cache_root}"
              export TRITON_CACHE_DIR="''${TRITON_CACHE_DIR:-$VLLM_CACHE_ROOT/triton_cache}"
              export TORCHINDUCTOR_CACHE_DIR="''${TORCHINDUCTOR_CACHE_DIR:-$VLLM_CACHE_ROOT/inductor_cache}"

              cache_store="${final.vllm-lemonade-qwen36-27b-cache-gfx1151}"
              stamp="$VLLM_CACHE_ROOT/.prebuilt-qwen36-27b"
              if [ ! -e "$stamp" ] || [ "$(cat "$stamp" 2>/dev/null || true)" != "$cache_store" ]; then
                tmp="$VLLM_CACHE_ROOT.tmp.$$"
                rm -rf "$tmp"
                mkdir -p "$tmp"
                cp -R "$cache_store/." "$tmp/"
                chmod -R u+w "$tmp"
                printf '%s\n' "$cache_store" > "$tmp/.prebuilt-qwen36-27b"
                rm -rf "$VLLM_CACHE_ROOT"
                mv "$tmp" "$VLLM_CACHE_ROOT"
              fi

              exec "${final.vllm-rocm-lemonade-gfx1151}/bin/vllm" "$@"
            '';
          };

          therock-python-gfx1151 = prev.callPackage ./pkgs/therock-python-env {
            wheelSources = therockPythonWheelSources;
          };

          therock-python-wheels-gfx1151 = prev.callPackage ./pkgs/therock-python-wheels {
            wheelSources = therockPythonWheelSources;
          };
          therock-amdsmi-gfx1151 = prev.python312Packages.callPackage ./pkgs/therock-amdsmi {
            wheels = final.therock-python-wheels-gfx1151;
          };

          therock-rocm-source-gfx1151 =
            let
              source = therockRocmSourceSources."therock-7.13-gfx1151-vllm";
            in
            prev.callPackage ./pkgs/therock-rocm-source {
              name = "therock-rocm-source-gfx1151-vllm";
              inherit (source)
                url
                ref
                rev
                hash
                fetchArgs
                ;
            };

          therock-rocm-from-source-gfx1151 =
            let
              source = therockRocmSourceSources."therock-7.13-gfx1151-vllm";
              esmi = therockRocmThirdPartySources.esmiIbLibrary;
            in
            prev.callPackage ./pkgs/therock-rocm-from-source {
              stdenv = prev.gcc14Stdenv;
              target = "gfx1151";
              inherit (source) version;
              profile = "vllm";
              therockSource = final.therock-rocm-source-gfx1151;
              prebuiltStageTree = final.therock-amd-llvm-gfx1151;
              thirdPartySources = therockRocmThirdPartySources.archives;
              esmiIbLibrarySource = prev.fetchgit {
                inherit (esmi) url rev hash;
              };
              spirvHeadersSource = prev.fetchzip {
                inherit (therockRocmThirdPartySources.spirvHeaders) url hash;
                stripRoot = true;
              };
            };

          therock-amd-llvm-gfx1151 =
            let
              source = therockRocmSourceSources."therock-7.13-gfx1151-vllm";
            in
            prev.callPackage ./pkgs/therock-rocm-from-source {
              stdenv = prev.gcc14Stdenv;
              target = "gfx1151";
              inherit (source) version;
              profile = "compiler";
              therockSource = final.therock-rocm-source-gfx1151;
              thirdPartySources = therockRocmThirdPartySources.archives;
              buildTargets = [ "artifact-amd-llvm" ];
              installMode = "prebuilt-stages";
              spirvHeadersSource = prev.fetchzip {
                inherit (therockRocmThirdPartySources.spirvHeaders) url hash;
                stripRoot = true;
              };
            };

          therock-rocm-gfx1151-env = prev.writeShellApplication {
            name = "therock-rocm-gfx1151-env";
            text = ''
              rocm=${final.therock-rocm-gfx1151}

              export ROCM_HOME="$rocm"
              export ROCM_PATH="$rocm"
              export HIP_PATH="$rocm"
              export HIP_PLATFORM=amd
              export HSA_OVERRIDE_GFX_VERSION="11.5.1"

              export PATH="$rocm/bin:$rocm/llvm/bin''${PATH:+:$PATH}"

              lib_paths=(
                "$rocm/lib"
                "$rocm/lib64"
                "$rocm/lib/llvm/lib"
                "$rocm/llvm/lib"
                "${prev.stdenv.cc.cc.lib}/lib"
                "${prev.gfortran.cc.lib}/lib"
                "${prev.zlib}/lib"
                "${prev.ncurses}/lib"
                "${prev.ocl-icd}/lib"
                "${prev.numactl}/lib"
                "${final.rdma-core-usb4}/lib"
              )

              ld_path=
              for path in "''${lib_paths[@]}"; do
                if [ -d "$path" ]; then
                  if [ -n "$ld_path" ]; then
                    ld_path="$ld_path:$path"
                  else
                    ld_path="$path"
                  fi
                fi
              done
              export LD_LIBRARY_PATH="$ld_path''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

              for path in \
                "$rocm/lib/llvm/amdgcn/bitcode" \
                "$rocm/amdgcn/bitcode" \
                "$rocm/lib/llvm/lib/clang"/*/amdgcn/bitcode \
                "$rocm/llvm/lib/clang"/*/amdgcn/bitcode; do
                if [ -d "$path" ]; then
                  export DEVICE_LIB_PATH="$path"
                  export HIP_DEVICE_LIB_PATH="$path"
                  break
                fi
              done

              if [ "$#" -eq 0 ]; then
                exec ${prev.bashInteractive}/bin/bash
              fi

              exec "$@"
            '';
          };

          therock-rocm-gfx1151-rocshmem-env = prev.writeShellApplication {
            name = "therock-rocm-gfx1151-rocshmem-env";
            runtimeInputs = [ prev.coreutils ];
            text = ''
              setup_rocshmem_sysfs_filter() {
                local root fallback_device verbs name file

                root="$(mktemp -d "''${TMPDIR:-/tmp}/rocshmem-sysfs.XXXXXX")"
                mkdir -p "$root/class/infiniband" "$root/class/infiniband_verbs"

                shopt -s nullglob

                for verbs in /sys/class/infiniband_verbs/*; do
                  if [ -e "$verbs/device" ]; then
                    fallback_device="$verbs/device"
                    break
                  fi
                done

                for verbs in /sys/class/infiniband/*; do
                  ln -s "$verbs" "$root/class/infiniband/$(basename "$verbs")"
                done

                for verbs in /sys/class/infiniband_verbs/*; do
                  name="$(basename "$verbs")"
                  if [ -e "$verbs/device" ] || [ -z "''${fallback_device:-}" ]; then
                    ln -s "$verbs" "$root/class/infiniband_verbs/$name"
                    continue
                  fi

                  mkdir -p "$root/class/infiniband_verbs/$name"
                  for file in abi_version dev ibdev; do
                    if [ -e "$verbs/$file" ]; then
                      ln -s "$verbs/$file" "$root/class/infiniband_verbs/$name/$file"
                    fi
                  done
                  ln -s "$fallback_device" "$root/class/infiniband_verbs/$name/device"
                done

                printf '%s\n' "$root"
              }

              rocshmem_sysfs_filter=
              if [ -z "''${SYSFS_PATH:-}" ] && [ -d /sys/class/infiniband_verbs ]; then
                rocshmem_sysfs_filter="$(setup_rocshmem_sysfs_filter)"
                export SYSFS_PATH="$rocshmem_sysfs_filter"
              fi

              cleanup_rocshmem_sysfs_filter() {
                if [ -n "$rocshmem_sysfs_filter" ]; then
                  rm -rf "$rocshmem_sysfs_filter"
                fi
              }
              trap cleanup_rocshmem_sysfs_filter EXIT

              export PATH="${prev.openmpi}/bin:${prev.python3}/bin:${final.therock-rocm-gfx1151}/share/rocshmem:$PATH"
              export LD_LIBRARY_PATH="${prev.openmpi}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              export OMPI_MCA_pml="''${OMPI_MCA_pml:-^ucx}"
              export OMPI_MCA_osc="''${OMPI_MCA_osc:-^ucx}"

              ${final.therock-rocm-gfx1151-env}/bin/therock-rocm-gfx1151-env "$@"
            '';
          };

          rocm-xio-gfx1151 = prev.callPackage ./pkgs/rocm-xio {
            src = inputs.rocm-xio;
            rocmSdk = final.therock-rocm-gfx1151;
            rdma-core = final.rdma-core-usb4;
            offloadArch = "gfx1151";
            version = "0.1.0-${inputs.rocm-xio.shortRev or "local"}";
          };

        }
        // llamaCppRocmTargetPackages
        // llamaCppMasterRocmTargetPackages;

      # Python-side extras for RDMA-enabled vLLM: rixl (ROCm NIXL port),
      # lmcache (KV-cache layer on top of rixl), and cupy-rocm-7-0. These
      # stay here because they're ROCm/Strix-specific; thunderbolt-ibverbs
      # owns only the kernel + rdma-core layer.
      vllmRdmaExtrasOverlay = _final: prev: {
        pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
          (pyfinal: _pyprev: {
            rixl = pyfinal.callPackage ./pkgs/rixl { };
            cupy-rocm-7-0 = pyfinal.callPackage ./pkgs/cupy-rocm { };
            lmcache = pyfinal.callPackage ./pkgs/lmcache {
              inherit (pyfinal) rixl;
            };
          })
        ];
      };

      # Opt-in TheRock Python overlay. The TheRock wheels are currently cp312,
      # so keep the substitution scoped to python312 package sets. Other Python
      # interpreters stay on the nixpkgs/libibverbs defaults.
      therockPythonOverlay = final: prev: {
        pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
          (
            _pyfinal: pyprev:
            if lib.versions.majorMinor pyprev.python.version == "3.12" then
              let
                wheels = final.therock-python-wheels-gfx1151;
              in
              {
                amdsmi = final.therock-amdsmi-gfx1151;
                torch = wheels;
                triton = wheels;
                triton-no-cuda = wheels;
                torchaudio = wheels;
                rocm = wheels;
                "rocm-sdk-core" = wheels;
                "rocm-sdk-devel" = wheels;
                "rocm-sdk-libraries-gfx1151" = wheels;
              }
            else
              { }
          )
        ];
      };

      therockVllmOverlay =
        final: prev:
        let
          sdkBase = final.therock-rocm-gfx1151;
          sdk = sdkBase // {
            localGpuTargets = rocmTargets;
            gpuTargets = rocmTargets;
          };
          therockRocmPackages = {
            clr = sdk;
            hipcc = sdk;
            rocminfo = sdk;
            rocm-device-libs = sdk;
            llvm = sdk;
            rocthrust = sdk;
            rocprim = sdk;
            hipcub = sdk;
            hipblas = sdk;
            hipblas-common = sdk;
            hipblaslt = sdk;
            hipfft = sdk;
            hipsparse = sdk;
            hiprand = sdk;
            hipsolver = sdk;
            miopen-hip = sdk;
            miopen = sdk;
            rccl = sdk;
            rocblas = sdk;
            rocm-comgr = sdk;
            rocfft = sdk;
            rocrand = sdk;
            rocm-runtime = sdk;
            rocsolver = sdk;
            rocsparse = sdk;
            composable_kernel = sdk;
          };
          rocmRuntimePath = prev.lib.makeBinPath [
            sdk
            prev.gcc
            prev.ninja
            prev.openssl
          ];
          aiterRuntimePath = prev.lib.makeBinPath [
            sdk
            prev.gcc
            prev.ninja
            prev.openssl
          ];
          aiterComposableKernelSrc = "${prev.rocmPackages.composable_kernel.src}/projects/composablekernel";
          cxxIncludePath = prev.lib.concatStringsSep ":" [
            "${prev.stdenv.cc.cc}/include/c++/${prev.stdenv.cc.cc.version}"
            "${prev.stdenv.cc.cc}/include/c++/${prev.stdenv.cc.cc.version}/${prev.stdenv.hostPlatform.config}"
            "${prev.glibc.dev}/include"
          ];
          nativeLibraryPath = prev.lib.makeLibraryPath [
            sdk
            prev.stdenv.cc.cc.lib
            prev.glibc
          ];
          hipccLinkFlags = prev.lib.concatStringsSep " " [
            "-L${sdk}/lib"
            "-L${prev.stdenv.cc.cc.lib}/lib"
            "-L${prev.stdenv.cc.cc}/lib/gcc/${prev.stdenv.hostPlatform.config}/${prev.stdenv.cc.cc.version}"
            "-L${prev.glibc}/lib"
            "-B${prev.glibc}/lib"
          ];
          wrapWithHsa =
            name: env:
            prev.symlinkJoin {
              inherit name;
              paths = [ env ];
              buildInputs = [ prev.makeWrapper ];
              postBuild = ''
                for bin in "$out"/bin/*; do
                  [[ -L "$bin" || -f "$bin" ]] || continue
                  target=$(readlink -f "$bin")
                  rm "$bin"
                  makeWrapper "$target" "$bin" \
                    --set HSA_NO_SCRATCH_RECLAIM 1 \
                    --set HSA_ENABLE_INTERRUPT 0 \
                    --set HSA_OVERRIDE_GFX_VERSION 11.5.1 \
                    --set ROCM_HOME ${sdk} \
                    --set ROCM_PATH ${sdk} \
                    --set HIP_PATH ${sdk} \
                    --set HIP_PLATFORM amd \
                    --set DEVICE_LIB_PATH ${sdk}/amdgcn/bitcode \
                    --set HIP_DEVICE_LIB_PATH ${sdk}/amdgcn/bitcode \
                    --prefix PATH : ${rocmRuntimePath} \
                    --prefix CPATH : ${prev.glibc.dev}/include \
                    --prefix CPLUS_INCLUDE_PATH : ${cxxIncludePath} \
                    --prefix LIBRARY_PATH : ${nativeLibraryPath} \
                    --run 'export HIPCC_LINK_FLAGS_APPEND="${hipccLinkFlags}''${HIPCC_LINK_FLAGS_APPEND:+ $HIPCC_LINK_FLAGS_APPEND}"' \
                    --run 'if [ -z "''${AITER_JIT_DIR:-}" ]; then export AITER_JIT_DIR="''${XDG_CACHE_HOME:-''${HOME:-/tmp}/.cache}/aiter/raw-gfx1151"; fi' \
                    --run 'export GPU_ARCHS="''${GPU_ARCHS:-gfx1151}"' \
                    --run 'export PYTORCH_ROCM_ARCH="''${PYTORCH_ROCM_ARCH:-gfx1151}"' \
                    --run 'export FLASH_ATTENTION_TRITON_AMD_ENABLE="''${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"' \
                    --run 'export TORCH_BLAS_PREFER_HIPBLASLT="''${TORCH_BLAS_PREFER_HIPBLASLT:-1}"' \
                    --run 'export PYTORCH_TUNABLEOP_ENABLED="''${PYTORCH_TUNABLEOP_ENABLED:-0}"' \
                    --run 'export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="''${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"'
                done
              '';
              passthru = {
                inherit (env) python;
                rocm = sdk;
                unwrapped = env;
              };
            };
          mkAiterJitCache =
            {
              name,
              vllmEnv,
              modules ? [
                "module_aiter_enum"
                "module_rmsnorm"
              ],
            }:
            prev.stdenv.mkDerivation {
              pname = name;
              version = final.python312Packages.amd-aiter.version;
              dontUnpack = true;
              dontStrip = true;
              requiredSystemFeatures = [ "gfx1151" ];
              nativeBuildInputs = [
                prev.gcc
                prev.ninja
                sdk
              ];

              buildPhase = ''
                runHook preBuild

                export HOME="$TMPDIR/home"
                export AITER_JIT_DIR="$TMPDIR/aiter-jit"
                export GPU_ARCHS="gfx1151"
                export PYTORCH_ROCM_ARCH="gfx1151"
                export ROCM_HOME="${sdk}"
                export ROCM_PATH="$ROCM_HOME"
                export HIP_PATH="$ROCM_HOME"
                export HIP_PLATFORM="amd"
                export DEVICE_LIB_PATH="${sdk}/amdgcn/bitcode"
                export HIP_DEVICE_LIB_PATH="$DEVICE_LIB_PATH"
                export CPATH="${prev.glibc.dev}/include''${CPATH:+:$CPATH}"
                export CPLUS_INCLUDE_PATH="${cxxIncludePath}''${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
                export LIBRARY_PATH="${nativeLibraryPath}''${LIBRARY_PATH:+:$LIBRARY_PATH}"
                export HIPCC_LINK_FLAGS_APPEND="${hipccLinkFlags}''${HIPCC_LINK_FLAGS_APPEND:+ $HIPCC_LINK_FLAGS_APPEND}"
                export MAX_JOBS="''${NIX_BUILD_CORES:-8}"
                export PATH="${aiterRuntimePath}:$PATH"

                mkdir -p "$HOME" "$AITER_JIT_DIR"
                ${vllmEnv}/bin/python - <<'PY'
                from aiter.jit.core import build_module, get_args_of_build

                modules = ${builtins.toJSON modules}
                for md_name in modules:
                    args = get_args_of_build(md_name)
                    build_module(
                        md_name=md_name,
                        srcs=args["srcs"],
                        flags_extra_cc=args["flags_extra_cc"],
                        flags_extra_hip=args["flags_extra_hip"],
                        blob_gen_cmd=args["blob_gen_cmd"],
                        extra_include=args["extra_include"],
                        extra_ldflags=args["extra_ldflags"],
                        verbose=args["verbose"],
                        is_python_module=args["is_python_module"],
                        is_standalone=args["is_standalone"],
                        torch_exclude=args["torch_exclude"],
                        hipify=args.get("hipify", False),
                    )
                PY

                ${prev.lib.concatMapStringsSep "\n" (module: ''
                  test -f "$AITER_JIT_DIR/${module}.so"
                '') modules}

                runHook postBuild
              '';

              installPhase = ''
                runHook preInstall

                mkdir -p "$out/build"
                cp "$AITER_JIT_DIR"/module_*.so "$out"/

                runHook postInstall
              '';

              meta = with prev.lib; {
                description = "Prebuilt AITER runtime-JIT modules for vLLM/TheRock on gfx1151";
                platforms = platforms.linux;
              };
            };
          mkVllmAiterWrapper =
            {
              name,
              vllmEnv,
              aiterJitCache,
            }:
            prev.writeShellApplication {
              inherit name;
              runtimeInputs = [
                prev.coreutils
                prev.gcc
                prev.ninja
                sdk
              ];
              text = ''
                export GPU_ARCHS="''${GPU_ARCHS:-gfx1151}"
                export PYTORCH_ROCM_ARCH="''${PYTORCH_ROCM_ARCH:-gfx1151}"
                export ROCM_HOME="''${ROCM_HOME:-${sdk}}"
                export ROCM_PATH="''${ROCM_PATH:-$ROCM_HOME}"
                export HIP_PATH="''${HIP_PATH:-$ROCM_HOME}"
                export HIP_PLATFORM="''${HIP_PLATFORM:-amd}"
                export DEVICE_LIB_PATH="''${DEVICE_LIB_PATH:-${sdk}/amdgcn/bitcode}"
                export HIP_DEVICE_LIB_PATH="''${HIP_DEVICE_LIB_PATH:-$DEVICE_LIB_PATH}"
                export CPATH="${prev.glibc.dev}/include''${CPATH:+:$CPATH}"
                export CPLUS_INCLUDE_PATH="${cxxIncludePath}''${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
                export LIBRARY_PATH="${nativeLibraryPath}''${LIBRARY_PATH:+:$LIBRARY_PATH}"
                export HIPCC_LINK_FLAGS_APPEND="${hipccLinkFlags}''${HIPCC_LINK_FLAGS_APPEND:+ $HIPCC_LINK_FLAGS_APPEND}"
                export HSA_NO_SCRATCH_RECLAIM="''${HSA_NO_SCRATCH_RECLAIM:-1}"
                export HSA_ENABLE_INTERRUPT="''${HSA_ENABLE_INTERRUPT:-0}"
                export HSA_OVERRIDE_GFX_VERSION="''${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
                export FLASH_ATTENTION_TRITON_AMD_ENABLE="''${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"
                export TORCH_BLAS_PREFER_HIPBLASLT="''${TORCH_BLAS_PREFER_HIPBLASLT:-1}"
                export PYTORCH_TUNABLEOP_ENABLED="''${PYTORCH_TUNABLEOP_ENABLED:-0}"
                export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="''${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
                export VLLM_ROCM_USE_AITER="''${VLLM_ROCM_USE_AITER:-1}"
                export VLLM_ROCM_USE_AITER_LINEAR="''${VLLM_ROCM_USE_AITER_LINEAR:-1}"
                export VLLM_ROCM_USE_AITER_MOE="''${VLLM_ROCM_USE_AITER_MOE:-0}"
                export VLLM_ROCM_USE_AITER_RMSNORM="''${VLLM_ROCM_USE_AITER_RMSNORM:-0}"
                export VLLM_ROCM_USE_AITER_MHA="''${VLLM_ROCM_USE_AITER_MHA:-1}"
                export VLLM_ROCM_USE_AITER_TRITON_GEMM="''${VLLM_ROCM_USE_AITER_TRITON_GEMM:-1}"
                export VLLM_ROCM_USE_AITER_TRITON_ROPE="''${VLLM_ROCM_USE_AITER_TRITON_ROPE:-1}"
                export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION="''${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:-1}"
                export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="''${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-0}"

                if [ -z "''${AITER_JIT_DIR:-}" ]; then
                  cache_root="''${XDG_CACHE_HOME:-''${HOME:-/tmp}/.cache}/aiter"
                  cache_dir="$cache_root/${aiterJitCache.name}"
                  stamp="$cache_dir/.prebuilt-${aiterJitCache.name}"
                  if [ ! -e "$stamp" ]; then
                    rm -rf "$cache_dir.tmp"
                    mkdir -p "$cache_dir.tmp"
                    cp -R "${aiterJitCache}/." "$cache_dir.tmp/"
                    chmod -R u+w "$cache_dir.tmp"
                    touch "$cache_dir.tmp/.prebuilt-${aiterJitCache.name}"
                    rm -rf "$cache_dir"
                    mv "$cache_dir.tmp" "$cache_dir"
                  fi
                  export AITER_JIT_DIR="$cache_dir"
                fi

                exec "${vllmEnv}/bin/vllm" "$@"
              '';
            };
          dropVllmDependencyNames = [
            "bitsandbytes"
            "datasets"
            "diskcache"
            "lark"
            "mistral-common"
            "mistral_common"
            "mistralai"
            "opencv-python-headless"
            "outlines"
            "outlines-core"
            "peft"
            "pyarrow"
            "timm"
            "torchcodec"
            "torchvision"
            "xformers"
          ];
          dropVllmDeps =
            names: deps:
            prev.lib.filter (
              dep:
              let
                name = dep.pname or dep.name or "";
              in
              !(prev.lib.elem name names)
            ) deps;
          disablePythonChecks =
            pkg:
            pkg.overridePythonAttrs (_old: {
              doCheck = false;
              pythonImportsCheck = [ ];
            });
          vllmTherock =
            (final.python312Packages.vllm.override {
              rocmSupport = true;
              cudaSupport = false;
              gpuTargets = rocmTargets;
              rocmPackages = therockRocmPackages;
              amdsmi = final.python312Packages.amdsmi;
            }).overridePythonAttrs
              (old: {
                patches = (old.patches or [ ]) ++ [
                  ./pkgs/vllm/patches/0001-hipify-copy-unchanged-cu-to-hip.patch
                ];
                postPatch = (old.postPatch or "") + ''
                                substituteInPlace csrc/quantization/gptq/compat.cuh \
                                  --replace-fail 'namespace gptq {' 'namespace gptq {

                  #if defined(USE_ROCM) && defined(TORCH_HIP_VERSION) && TORCH_HIP_VERSION >= 713
                  #define VLLM_GPTQ_SKIP_LEGACY_HALF_ATOMIC_ADD 1
                  #endif'
                                substituteInPlace csrc/quantization/gptq/compat.cuh \
                                  --replace-fail '#if defined(__CUDA_ARCH__) || defined(USE_ROCM)' \
                                    '#if !defined(VLLM_GPTQ_SKIP_LEGACY_HALF_ATOMIC_ADD) && (defined(__CUDA_ARCH__) || defined(USE_ROCM))'
                '';
                env = (old.env or { }) // {
                  HIP_PATH = "${sdk}";
                  HIP_PLATFORM = "amd";
                  CMAKE_ARGS = prev.lib.concatStringsSep " " [
                    ((old.env or { }).CMAKE_ARGS or "")
                    "-DCMAKE_HIP_COMPILER=${sdk}/bin/therock-hip-clang++"
                    "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${sdk}"
                    "-DHIP_ROOT_DIR=${sdk}"
                  ];
                };
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  final.pkg-config
                ];
                pythonRemoveDeps = (old.pythonRemoveDeps or [ ]) ++ dropVllmDependencyNames;
                dependencies = dropVllmDeps dropVllmDependencyNames (old.dependencies or [ ]) ++ [
                  final.python312Packages.amdsmi
                  final.python312Packages.cloudpickle
                ];
                propagatedBuildInputs = dropVllmDeps dropVllmDependencyNames (old.propagatedBuildInputs or [ ]) ++ [
                  final.python312Packages.amdsmi
                  final.python312Packages.cloudpickle
                ];
              });
          vllmEnvTherock = wrapWithHsa "vllm-env-therock-gfx1151" (
            final.python312.withPackages (_ps: [
              vllmTherock
              final.python312Packages.ray
            ])
          );
          vllmEnvTherockAiter = wrapWithHsa "vllm-env-therock-aiter-gfx1151" (
            final.python312.withPackages (_ps: [
              vllmTherock
              final.python312Packages.amd-aiter
              final.python312Packages.ray
            ])
          );
          vllmAiterJitTherock = mkAiterJitCache {
            name = "vllm-aiter-jit-therock-gfx1151";
            vllmEnv = vllmEnvTherockAiter;
          };
        in
        {
          pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
            (
              pyfinal: pyprev:
              if lib.versions.majorMinor pyprev.python.version == "3.12" then
                {
                  amd-aiter =
                    (pyprev.amd-aiter.override {
                      rocmPackages = therockRocmPackages;
                    }).overridePythonAttrs
                      (old: {
                        env = (old.env or { }) // {
                          ROCM_PATH = "${sdk}";
                          ROCM_HOME = "${sdk}";
                          HIP_PATH = "${sdk}";
                        };
                        postPatch = (old.postPatch or "") + ''
                          rm -rf 3rdparty/composable_kernel
                          ln -s ${aiterComposableKernelSrc} 3rdparty/composable_kernel
                        '';
                        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                          sdk
                        ];
                        buildInputs = (old.buildInputs or [ ]) ++ [
                          sdk
                        ];
                      });
                  "compressed-tensors" = pyprev."compressed-tensors".overridePythonAttrs (old: {
                    dependencies = (old.dependencies or [ ]) ++ [
                      pyfinal.psutil
                    ];
                    nativeCheckInputs = (old.nativeCheckInputs or [ ]) ++ [
                      final.openssl
                    ];
                    propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [
                      pyfinal.psutil
                    ];
                  });
                  depyf = disablePythonChecks pyprev.depyf;
                  llguidance = disablePythonChecks pyprev.llguidance;
                  pydevd = disablePythonChecks pyprev.pydevd;
                  "mistral-common" = disablePythonChecks pyprev."mistral-common";
                }
              else
                { }
            )
          ];
          therock-rocm-packages-gfx1151 = therockRocmPackages;
          therock-python-overlay-smoke-gfx1151 = final.python312.withPackages (ps: [
            ps.torch
            ps.triton
          ]);
          vllm-rocm-therock-gfx1151 = vllmTherock;
          vllm-env-therock-gfx1151 = vllmEnvTherock;
          vllm-env-therock-aiter-gfx1151 = vllmEnvTherockAiter;
          vllm-aiter-jit-therock-gfx1151 = vllmAiterJitTherock;
          vllm-aiter-therock-gfx1151 = mkVllmAiterWrapper {
            name = "vllm-aiter-therock-gfx1151";
            vllmEnv = vllmEnvTherockAiter;
            aiterJitCache = vllmAiterJitTherock;
          };
        };

      mkRocmNarrowOverlay = target: _: prev: {
        rocmPackages = prev.rocmPackages.overrideScope (
          _: rocmPrev: {
            clr = rocmPrev.clr.override {
              localGpuTargets = target.buildTargets;
            };
          }
        );
      };

      rocmNarrowOverlays = lib.mapAttrs' (_: target: {
        name = "rocm-narrow-${target.packageSuffix}";
        value = mkRocmNarrowOverlay target;
      }) hardwareTargets.rocm;

      overlays = {
        # Composed default for strix-halo NixOS machines. Order matters:
        # rdma-core-usb4 first, RDMA Python extras next, then Python/ROCm
        # runtime overlays and strix additions. Doesn't include rocm-narrow;
        # that's opt-in via overlays.rocm-narrow because it invalidates
        # downstream caches.
        default = nixpkgs.lib.composeManyExtensions [
          inputs.usb4-rdma.overlays.rdma-core-usb4
          vllmRdmaExtrasOverlay
          therockPythonOverlay
          strixAdditionsOverlay
          therockVllmOverlay
        ];

        # Just the RDMA-enabled vllm composition without strix additions.
        # Useful for non-strix consumers that want vllm+rixl+lmcache but
        # not ec-su/llama-cpp-rocm.
        vllm-rdma = nixpkgs.lib.composeManyExtensions [
          inputs.usb4-rdma.overlays.rdma-core-usb4
          vllmRdmaExtrasOverlay
        ];

        # vLLM with Python dependencies resolved against TheRock's pinned
        # cp312 Torch/Triton/ROCm wheels and the TheRock gfx1151 SDK.
        vllm-therock = nixpkgs.lib.composeManyExtensions [
          inputs.usb4-rdma.overlays.rdma-core-usb4
          vllmRdmaExtrasOverlay
          therockPythonOverlay
          strixAdditionsOverlay
          therockVllmOverlay
        ];

        # Narrows rocmPackages.clr to `rocmTargets` (currently gfx1151) for
        # faster builds on strix-halo. Applying this overlay invalidates
        # downstream caches and removes support for other GPUs from the system
        # OpenCL ICD — only enable on hosts that exclusively run gfx1151.
        #
        # This is the *shallow* narrowing: only the ICD targets change. The
        # individual rocm leaves (rccl, hipblaslt, miopen, …) still build for
        # every nixpkgs-default arch. Pair with `rocm-narrow-deep` below to
        # also narrow those leaves.
        rocm-narrow = mkRocmNarrowOverlay defaultRocmTarget;

        # Deep ROCm narrowing for strix-halo: in addition to clr, also pins
        # the gpuTargets of every leaf rocmPackage (rccl, hipblaslt, hipfft,
        # hiprand, hipsparse, miopen, rocblas, rocfft, rocrand, rocsolver,
        # rocsparse, aotriton) to gfx1151. composable_kernel_base's broken
        # check requires at least one gfx9 mfma target, so it's pinned to
        # gfx90a+gfx11-generic — CK still only emits gfx1151 kernels, but
        # the meta.broken guard passes.
        #
        # Use this when you want every ROCm build in the closure to target
        # only gfx1151 (i.e. on strix-halo hosts where you don't need
        # multi-GPU portability). Invalidates downstream caches but cuts
        # build time dramatically: composable_kernel alone has ~20 per-arch
        # parts that otherwise would all rebuild.
        #
        # Same logic is also inline in linux-libibverbs-usb4/flake.nix as
        # `gfx1151Overlay`; consolidating that copy into this one is a
        # separate cleanup. For now both should produce equivalent results.
        rocm-narrow-deep = final: prev: {
          rocmPackages = prev.rocmPackages.overrideScope (
            _: rocmPrev:
            let
              narrow = drv: drv.override { gpuTargets = rocmTargets; };
            in
            {
              clr = rocmPrev.clr.override { localGpuTargets = rocmTargets; };

              rccl = narrow rocmPrev.rccl;
              hipblaslt = narrow rocmPrev.hipblaslt;
              hipfft = narrow rocmPrev.hipfft;
              hiprand = narrow rocmPrev.hiprand;
              hipsparse = narrow rocmPrev.hipsparse;
              miopen = narrow rocmPrev.miopen;
              rocblas = narrow rocmPrev.rocblas;
              rocfft = narrow rocmPrev.rocfft;
              rocrand = narrow rocmPrev.rocrand;
              rocsolver = narrow rocmPrev.rocsolver;
              rocsparse = narrow rocmPrev.rocsparse;

              composable_kernel_base = rocmPrev.composable_kernel_base.override {
                gpuTargets = [
                  "gfx90a"
                  "gfx11-generic"
                ];
              };

              aotriton = rocmPrev.aotriton.override {
                gpuTargets = rocmTargets;
              };
            }
          );
        };
      }
      // rocmNarrowOverlays;

      defaultPackagesFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ overlays.default ];
        };

      perSystem =
        f:
        forAllSystems (
          system:
          let
            pkgs = defaultPackagesFor system;
          in
          f {
            inherit
              pkgs
              system
              ;
          }
        );

      mkBenchmarkPackages =
        pkgs:
        let
          mkPackageSetForTarget =
            target:
            let
              isDefault = target.packageSuffix == defaultRocmTarget.packageSuffix;
            in
            {
              llama-cpp-rocm =
                if isDefault then pkgs.llama-cpp-rocm else pkgs."llama-cpp-rocm-${target.packageSuffix}";
              llama-cpp-vulkan = pkgs.llama-cpp-vulkan;
              llama-cpp-master-rocm =
                if isDefault then
                  pkgs.llama-cpp-master-rocm
                else
                  pkgs."llama-cpp-master-rocm-${target.packageSuffix}";
              llama-cpp-master-vulkan = pkgs.llama-cpp-master-vulkan;
            }
            // lib.optionalAttrs isDefault {
              llama-cpp-rocm-therock = pkgs.llama-cpp-rocm-therock;
              llama-cpp-master-rocm-therock = pkgs.llama-cpp-master-rocm-therock;
            };

          mkForTarget =
            target:
            let
              benchmarks = import ./bench/default.nix {
                inherit pkgs;
                packages = mkPackageSetForTarget target;
                gpuTarget = target.systemFeature;
                hsaOverride = target.hsaOverride;
              };
              targetPrefix = lib.optionalString (
                target.packageSuffix != defaultRocmTarget.packageSuffix
              ) "${target.packageSuffix}-";
            in
            pkgs.lib.concatMapAttrs (
              model: benchs:
              pkgs.lib.mapAttrs' (name: drv: {
                name = "bench-${targetPrefix}${model}-${name}";
                value = drv;
              }) benchs
            ) benchmarks;
        in
        lib.foldl' (acc: target: acc // mkForTarget target) { } hardwareTargets.rocmTargets;

      mkSourceCheck =
        pkgs: name: nativeBuildInputs: command:
        pkgs.runCommandLocal "ci-${name}"
          {
            inherit nativeBuildInputs;
            src = lib.cleanSource ./.;
          }
          ''
            export HOME="$TMPDIR"
            export XDG_CACHE_HOME="$TMPDIR/cache"
            cp -r --no-preserve=mode "$src" source
            cd source
            ${command}
            touch "$out"
          '';

      rocmNarrowNixosModules = lib.mapAttrs' (_: target: {
        name = "rocm-narrow-${target.packageSuffix}";
        value = _: {
          nixpkgs.overlays = [
            self.overlays."rocm-narrow-${target.packageSuffix}"
          ];
        };
      }) hardwareTargets.rocm;
    in
    {
      inherit overlays;

      lib = {
        hardware = hardwareTargets;
        inherit mkRocmNarrowOverlay;
      };

      nixosModules = {
        default = _: {
          nixpkgs.overlays = [
            self.overlays.default
          ];
        };
        rocm-narrow = _: {
          nixpkgs.overlays = [
            self.overlays.rocm-narrow
          ];
        };
        rocm-narrow-deep = _: {
          nixpkgs.overlays = [
            self.overlays.rocm-narrow-deep
          ];
        };
        rpc-server = import ./modules/rpc-server.nix;
        benchmark-runner = import ./modules/benchmark-runner.nix;
        ec-su-axb35 = import ./modules/ec-su-axb35.nix;
        ryzenadj = import ./modules/ryzenadj.nix;
        disko-raid0 = import ./modules/disko-raid0.nix;
        tuning = import ./modules/tuning.nix;
      }
      // rocmNarrowNixosModules;

      nixosConfigurations = {
        fevm-faex9 = nixpkgs.lib.nixosSystem {
          system = "x86_64-linux";
          specialArgs = {
            inherit inputs self;
          };
          modules = [
            inputs.disko.nixosModules.disko
            ./examples/fevm-faex9/configuration.nix
          ];
        };
      };

      legacyPackages = perSystem (
        { pkgs, ... }:
        {
          defaultPackages = pkgs;
        }
      );

      packages = perSystem (
        { pkgs, ... }:
        let
          rocmLlamaPackageNames = lib.concatMap (target: [
            "llama-cpp-rocm-${target.packageSuffix}"
            "llama-cpp-master-rocm-${target.packageSuffix}"
          ]) hardwareTargets.rocmTargets;
        in
        {
          default = pkgs.llama-cpp-rocm;
          inherit (pkgs)
            ec-su-axb35-monitor
            fastflowlm
            tokenizers-cpp
            xrt
            xrt-amdxdna
            llama-cpp-rocm
            llama-cpp-rocm-therock
            llama-cpp-vulkan
            llama-cpp-master-rocm
            llama-cpp-master-rocm-therock
            llama-cpp-master-vulkan
            rdma-core-usb4
            rocm-xio-gfx1151
            therock-python-gfx1151
            therock-python-wheels-gfx1151
            therock-python-overlay-smoke-gfx1151
            therock-amd-llvm-gfx1151
            therock-rocm-gfx1151
            therock-rocm-gfx1151-env
            therock-rocm-gfx1151-rocshmem-env
            therock-rocm-source-gfx1151
            therock-rocm-from-source-gfx1151
            vllm-rocm-lemonade-gfx1151
            vllm-lemonade-prime-cache-gfx1151
            vllm-lemonade-qwen36-27b-cache-gfx1151
            vllm-lemonade-qwen36-27b-gfx1151
            vllm-rocm-therock-gfx1151
            vllm-env-therock-gfx1151
            vllm-env-therock-aiter-gfx1151
            vllm-aiter-jit-therock-gfx1151
            vllm-aiter-therock-gfx1151
            ;
        }
        // lib.genAttrs rocmLlamaPackageNames (name: pkgs.${name})
        // mkBenchmarkPackages pkgs
      );

      apps = perSystem (
        { pkgs, ... }:
        let
          mkHsaOverrideExport =
            target:
            lib.optionalString (
              target.hsaOverride != null
            ) ''export HSA_OVERRIDE_GFX_VERSION="${target.hsaOverride}"'';

          mkLlamaApp =
            {
              name,
              package,
              binary,
              target,
              description,
            }:
            {
              type = "app";
              program = toString (
                pkgs.writeShellScript name ''
                  ${mkHsaOverrideExport target}
                  ${package}/bin/${binary} "$@"
                ''
              );
              meta.description = description;
            };

          llamaAppAxes = {
            cli = {
              appPrefix = "llama-cli";
              packagePrefix = "llama-cpp-rocm";
              binary = "llama-cli";
              descriptionPrefix = "Run llama-cli with ROCm for";
            };
            server = {
              appPrefix = "llama-server";
              packagePrefix = "llama-cpp-rocm";
              binary = "llama-server";
              descriptionPrefix = "Run llama-server with ROCm for";
            };
            cliMaster = {
              appPrefix = "llama-cli-master";
              packagePrefix = "llama-cpp-master-rocm";
              binary = "llama-cli";
              descriptionPrefix = "Run llama-cli from ggml-org master with ROCm for";
            };
            serverMaster = {
              appPrefix = "llama-server-master";
              packagePrefix = "llama-cpp-master-rocm";
              binary = "llama-server";
              descriptionPrefix = "Run llama-server from ggml-org master with ROCm for";
            };
          };

          mkLlamaAppSet =
            {
              target,
              appSuffix ? "",
              packageSuffix ? "",
              packageNameFor ? null,
              descriptionFor ? null,
            }:
            let
              packageName =
                spec:
                if packageNameFor == null then
                  "${spec.packagePrefix}${packageSuffix}"
                else
                  packageNameFor spec;
              description =
                spec:
                if descriptionFor == null then
                  "${spec.descriptionPrefix} ${target.marketingName}"
                else
                  descriptionFor spec;
            in
            lib.mapAttrs' (
              _: spec:
              let
                name = "${spec.appPrefix}${appSuffix}";
              in
              {
                inherit name;
                value = mkLlamaApp {
                  inherit name target;
                  package = pkgs.${packageName spec};
                  inherit (spec) binary;
                  description = description spec;
                };
              }
            ) llamaAppAxes;

          targetLlamaApps = lib.foldl' (
            acc: target:
            acc
            // mkLlamaAppSet {
              inherit target;
              appSuffix = "-${target.packageSuffix}";
              packageSuffix = "-${target.packageSuffix}";
            }
          ) { } hardwareTargets.rocmTargets;

          defaultLlamaApps = mkLlamaAppSet { target = defaultRocmTarget; };
          defaultTherockLlamaApps = mkLlamaAppSet {
            target = defaultRocmTarget;
            appSuffix = "-therock";
            packageNameFor = spec: "${spec.packagePrefix}-therock";
            descriptionFor = spec: "${spec.descriptionPrefix} ${defaultRocmTarget.marketingName} with TheRock ROCm";
          };
        in
        {
          therock-rocm-gfx1151-env = {
            type = "app";
            program = "${pkgs.therock-rocm-gfx1151-env}/bin/therock-rocm-gfx1151-env";
            meta.description = "Run a command in the opt-in TheRock ROCm 7.13 gfx1151 environment";
          };

          therock-rocm-gfx1151-rocshmem-env = {
            type = "app";
            program = "${pkgs.therock-rocm-gfx1151-rocshmem-env}/bin/therock-rocm-gfx1151-rocshmem-env";
            meta.description = "Run a command in the TheRock ROCm 7.13 gfx1151 rocSHMEM test environment";
          };

          therock-python-gfx1151 = {
            type = "app";
            program = "${pkgs.therock-python-gfx1151}/bin/therock-python";
            meta.description = "Run Python in the pinned TheRock ROCm/PyTorch wheel environment";
          };

          therock-python-gfx1151-env = {
            type = "app";
            program = "${pkgs.therock-python-gfx1151}/bin/therock-python-env";
            meta.description = "Run a command in the pinned TheRock ROCm/PyTorch wheel environment";
          };

          vllm-therock-gfx1151 = {
            type = "app";
            program = "${pkgs.vllm-env-therock-gfx1151}/bin/vllm";
            meta.description = "Run vLLM built against the TheRock ROCm/PyTorch/Triton overlay";
          };

          vllm-lemonade-gfx1151 = {
            type = "app";
            program = "${pkgs.vllm-rocm-lemonade-gfx1151}/bin/vllm";
            meta.description = "Run Lemonade's vLLM ROCm binary bundle for Strix Halo";
          };

          vllm-lemonade-prime-cache-gfx1151 = {
            type = "app";
            program = "${pkgs.vllm-lemonade-prime-cache-gfx1151}/bin/vllm-lemonade-prime-cache-gfx1151";
            meta.description = "Prime Lemonade vLLM ROCm Triton/autotune caches on a gfx1151 host";
          };

          vllm-lemonade-qwen36-27b-gfx1151 = {
            type = "app";
            program = "${pkgs.vllm-lemonade-qwen36-27b-gfx1151}/bin/vllm-lemonade-qwen36-27b-gfx1151";
            meta.description = "Run Lemonade vLLM with the prebuilt Qwen3.6-27B Triton cache";
          };

          vllm-therock-aiter-gfx1151 = {
            type = "app";
            program = "${pkgs.vllm-aiter-therock-gfx1151}/bin/vllm-aiter-therock-gfx1151";
            meta.description = "Run vLLM with TheRock ROCm and a preseeded AITER JIT cache";
          };

          flm = {
            type = "app";
            program = "${pkgs.fastflowlm}/bin/flm";
            meta.description = "FastFlowLM CLI on the AMD Ryzen AI NPU (Strix Halo XDNA2)";
          };
        }
        // defaultLlamaApps
        // defaultTherockLlamaApps
        // targetLlamaApps
      );

      devShells = perSystem (
        { pkgs, ... }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              deadnix
              nix-fast-build
              nixfmt-tree
              statix
              (python3.withPackages (
                ps: with ps; [
                  numpy
                  pandas
                  plotly
                ]
              ))
            ];
          };
        }
      );

      formatter = perSystem ({ pkgs, ... }: pkgs.nixfmt-tree);

      checks = perSystem (
        { pkgs, ... }:
        {
          format = mkSourceCheck pkgs "format" [ pkgs.nixfmt-tree ] ''
            treefmt --fail-on-change
          '';

          deadnix = mkSourceCheck pkgs "deadnix" [ pkgs.deadnix ] ''
            deadnix --fail .
          '';

          statix = mkSourceCheck pkgs "statix" [ pkgs.statix ] ''
            statix check .
          '';
        }
      );
    };
}
