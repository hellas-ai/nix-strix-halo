{ runCommand, python3, sglang-glm53-rocm }:
runCommand "glm53-wmma-gfx1151-aot-seed" { nativeBuildInputs = [ python3 ]; } ''
  export XDG_CACHE_HOME="$TMPDIR/cache"
  mkdir -p "$XDG_CACHE_HOME"
  export TRITON_CACHE_DIR="$TMPDIR/triton"
  export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
  export SGLANG_CACHE_DIR="$TMPDIR/sglang"
  export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" ROCP_TOOL_LIBRARIES=""
  export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
  mkdir -p "$out"
  ${sglang-glm53-rocm}/bin/sglang-python ${./compile.py} --report "$out/compile-report.json" > "$TMPDIR/compile.log"
  cat > "$TMPDIR/identity.json" <<'EOF'
  {"runtime":"${sglang-glm53-rocm}","target":"gfx1151","configuration":{"scope":"glm53-wmma-tp4-dense-c2-c4-c8","compiler_environment":{},"pointer_alignment":16,"pointer_range_bits":32}}
  EOF
  python3 ${../cache/seed.py} export --identity "$TMPDIR/identity.json" --cache "$TRITON_CACHE_DIR" --seed "$out/seed"
  python3 ${../cache/seed.py} verify --identity "$TMPDIR/identity.json" --seed "$out/seed"
  cp "$TMPDIR/identity.json" "$out/identity.json"
''
