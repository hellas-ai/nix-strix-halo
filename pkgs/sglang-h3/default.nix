{
  lib,
  fetchFromGitHub,
  pythonPackages,
  sglangRocm,
}:

let
  revision = "b6a09f38fcc5e96574324b4acc19d421c539cfc6";
  h3Src = fetchFromGitHub {
    owner = "sgl-project";
    repo = "sglang";
    rev = revision;
    hash = "sha256-6dOPfG+DMod67l3pC3+/Z/E/XrY+i6cdfkw++2SQ4H4=";
  };
  diffusionDependencies = with pythonPackages; [
    # The base package deliberately contains only the autoregressive server
    # closure. MiniMax-H3 enters through SGLang's multimodal diffusion server.
    # Keep its small media/config layer here and CUDA-only extras out.
    addict
    cloudpickle
    diffusers
    imageio
    imageio-ffmpeg
    pyyaml
  ];
  h3PostPatch = ''
    substituteInPlace sglang/multimodal_gen/runtime/layers/quantization/mxfp4.py \
      --replace-fail 'import logging' $'import logging\nimport os' \
      --replace-fail \
        $'_is_hip = is_hip()\n\nif _is_hip:' \
        $'_is_hip = is_hip()\n\nif _is_hip and os.environ.get("SGLANG_USE_AITER", "0") == "1":'

    substituteInPlace sglang/srt/layers/quantization/mxfp4.py \
      --replace-fail \
        $'_is_cpu_amx_available = cpu_has_amx_support()\n\nif _is_hip:' \
        $'_is_cpu_amx_available = cpu_has_amx_support()\n\nif _use_aiter:'

    # H3 needs only the base quantization type. Avoid importing every optional
    # CUDA/ROCm/NPU quantizer while config discovery selects an unquantized model.
    substituteInPlace sglang/multimodal_gen/configs/models/dits/base.py \
      --replace-fail \
        'from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig' \
        'from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import QuantizationConfig'

    # This import is used only for annotations, but importing a submodule still
    # executes the SRT quantization package and probes GPU 0. Keep it type-only
    # so `sglang serve --help` and H3 config discovery work without a live GPU.
    substituteInPlace sglang/multimodal_gen/runtime/loader/transformer_load_utils.py \
      --replace-fail \
        'from typing import Callable, Optional' \
        'from typing import TYPE_CHECKING, Any, Callable, Optional' \
      --replace-fail \
        'from sglang.srt.layers.quantization import QuantizationConfig' \
        $'if TYPE_CHECKING:\n    from sglang.srt.layers.quantization.base_config import QuantizationConfig\nelse:\n    QuantizationConfig = Any'

    # VLA is unrelated to H3 and pulls the full autoregressive manager stack
    # while the diffusion registry is merely being imported. Preserve its
    # public names lazily; registry discovery can then skip it independently.
    substituteInPlace sglang/multimodal_gen/runtime/pipelines_core/stages/__init__.py \
      --replace-fail \
        $'from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (\n    VLAActionDenoisingStage,\n    VLAActionPostprocessStage,\n    VLAObservationPreprocessStage,\n    VLAPrefixEncodingStage,\n)' \
        $'def __getattr__(name):\n    if name in {\n        "VLAActionDenoisingStage",\n        "VLAActionPostprocessStage",\n        "VLAObservationPreprocessStage",\n        "VLAPrefixEncodingStage",\n    }:\n        from sglang.multimodal_gen.runtime.pipelines_core.stages import vla\n        return getattr(vla, name)\n    raise AttributeError(name)'

    substituteInPlace sglang/srt/layers/quantization/__init__.py \
      --replace-fail 'import builtins' $'import builtins\nimport os' \
      --replace-fail \
        $'from sglang.srt.layers.quantization.quark.quark import QuarkConfig\nfrom sglang.srt.layers.quantization.quark_int4fp8_moe import QuarkInt4Fp8Config' \
        $'if os.environ.get("SGLANG_USE_AITER", "0") == "1":\n    from sglang.srt.layers.quantization.quark.quark import QuarkConfig\n    from sglang.srt.layers.quantization.quark_int4fp8_moe import QuarkInt4Fp8Config\nelse:\n    QuarkConfig = DummyConfig\n    QuarkInt4Fp8Config = DummyConfig'

    # Avoid a Python 3.13 recursive-import failure while the CLI resolves the
    # deeper diffusion serve module. Preserve the public API lazily.
    substituteInPlace sglang/multimodal_gen/__init__.py \
      --replace-fail \
        $'from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig\nfrom sglang.multimodal_gen.configs.sample import SamplingParams\nfrom sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator' \
        $'def __getattr__(name):\n    if name == "PipelineConfig":\n        from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig\n        return PipelineConfig\n    if name == "SamplingParams":\n        from sglang.multimodal_gen.configs.sample import SamplingParams\n        return SamplingParams\n    if name == "DiffGenerator":\n        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator\n        return DiffGenerator\n    raise AttributeError(name)'
  '';
in
(sglangRocm.override {
  packageVariant = "h3";
  packageVersion = "0.5.17";
  packageCacheTag = "sglang-0.5.17-h3";
  packageSrc = h3Src;
  packagePatches = [
    ./patches/0001-cache-dit-optional.patch
    ./patches/0002-diffusion-cli-spawn.patch
    ./patches/0003-bounded-shard-prefetch.patch
    ./patches/0004-stream-identity-weights.patch
    ./patches/0005-lazy-diffusion-quantization-imports.patch
    ./patches/0006-external-multinode-ranks.patch
    ./patches/0007-minimax-h3-latent-checkpoints.patch
    ./patches/0008-minimax-h3-rank-weight-cache.patch
    ./patches/0009-minimax-h3-rank0-vae.patch
    ./patches/0010-minimax-h3-unmanaged-runtime-device.patch
    ./patches/0011-minimax-h3-block-runtime-device.patch
    ./patches/0012-minimax-h3-rocm-torch-kernel-fallback.patch
    ./patches/0013-rocm-bounded-sdpa.patch
    ./patches/0014-minimax-h3-cpu-logits-gather.patch
    ./patches/0015-fsdp-mesh-follows-load-device.patch
    ./patches/0016-stream-rank-cache-restore.patch
    ./patches/0017-stream-ordinary-rank-cache-restore.patch
    ./patches/0018-serialize-local-rank-cache-io.patch
    ./patches/0019-minimax-h3-cpu-keyframe-encode.patch
    ./patches/0020-minimax-h3-half-resolution.patch
  ];
  # H3 selects native Torch SDPA. AITER is unsupported on gfx1030 and its
  # mere presence makes Diffusers import it eagerly before backend selection.
  packageAiter = null;
  packageExtraDependencies = diffusionDependencies;
  packagePostPatch = h3PostPatch;
  packageDoCheck = false;
  packageDoInstallCheck = false;
}).overrideAttrs (old: {
  passthru = (old.passthru or { }) // {
    inherit revision;
    modelFamily = "MiniMax-H3";
  };

  meta = (old.meta or { }) // {
    description = "SGLang MiniMax-H3 server packaged against the target-specific TheRock ROCm stack";
    homepage = "https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3";
    license = lib.licenses.asl20;
  };
})
