{
  lib,
  python3,
  runCommand,
  makeWrapper,
  src,
  torch,
  packageSuffix,
}:

let
  # Force every downstream python package to use our torch instead of
  # the nixpkgs one, so transformers / peft / trl / accelerate / datasets
  # all import the same torch ABI.
  pythonForFinetune = python3.override {
    self = pythonForFinetune;
    packageOverrides = _pyfinal: _pyprev: {
      inherit torch;
      # transformers builds a torch-cuda check at install time; ours is
      # ROCm so we skip the check via dontCheckRuntimeDeps.
    };
  };

  pyenv = pythonForFinetune.withPackages (ps: [
    ps.torch
    ps.transformers
    ps.peft
    ps.trl
    ps.accelerate
    ps.datasets
    # Tokenizers + chat templating deps that train.py imports
    ps.safetensors
    ps.sentencepiece
    ps.huggingface-hub
    ps.einops
    # Misc
    ps.tqdm
    ps.pyyaml
  ]);

in
runCommand "strix-halo-finetune-env-${packageSuffix}"
  {
    nativeBuildInputs = [ makeWrapper ];
    passthru = { inherit pyenv; };
  }
  ''
    mkdir -p "$out/bin" "$out/share/strix-halo-finetune"
    cp -r ${src}/workspace/. "$out/share/strix-halo-finetune/"
    chmod -R u+w "$out/share/strix-halo-finetune"

    # Switch FSDP state-dict to SHARDED + add --no-save flag for smoke
    # tests. Patch paths are a/workspace/train.py; we strip both
    # components since workspace/ was flattened into the share dir.
    (cd "$out/share/strix-halo-finetune" && \
      patch -p2 < ${./train-fsdp-state-dict.patch})
    (cd "$out/share/strix-halo-finetune" && \
      patch -p2 < ${./train-gemma3-token-type-ids.patch})

    makeWrapper "${pyenv}/bin/python3" "$out/bin/strix-halo-finetune" \
      --set FINETUNE_WORKSPACE "$out/share/strix-halo-finetune" \
      --set GPU_ARCHS "${packageSuffix}" \
      --set PYTORCH_ROCM_ARCH "${packageSuffix}" \
      --set HSA_OVERRIDE_GFX_VERSION 11.5.1 \
      --set HSA_NO_SCRATCH_RECLAIM 1 \
      --set HSA_ENABLE_INTERRUPT 0 \
      --add-flags "$out/share/strix-halo-finetune/train.py"

    makeWrapper "${pyenv}/bin/torchrun" "$out/bin/strix-halo-finetune-torchrun" \
      --set GPU_ARCHS "${packageSuffix}" \
      --set PYTORCH_ROCM_ARCH "${packageSuffix}" \
      --set HSA_OVERRIDE_GFX_VERSION 11.5.1 \
      --set HSA_NO_SCRATCH_RECLAIM 1 \
      --set HSA_ENABLE_INTERRUPT 0

    # Convenience: a shell entrypoint that drops you in the env with
    # the workspace already cwd'd.
    cat > "$out/bin/strix-halo-finetune-shell" <<EOF
#!${pythonForFinetune.stdenv.shell}
export GPU_ARCHS="${packageSuffix}"
export PYTORCH_ROCM_ARCH="${packageSuffix}"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_INTERRUPT=0
export PATH="${pyenv}/bin:\$PATH"
cd "$out/share/strix-halo-finetune"
exec "\$@"
EOF
    chmod +x "$out/bin/strix-halo-finetune-shell"

    runHook postInstall || true
  ''
