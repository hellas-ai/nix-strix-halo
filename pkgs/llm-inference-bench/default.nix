{
  fetchFromGitHub,
  lib,
  makeWrapper,
  numactl,
  python3,
  stdenv,
}:

let
  pythonEnv = python3.withPackages (ps: [
    ps.httpx
    ps.openai
    ps.psutil
    ps.requests
    ps.rich
  ]);
in
stdenv.mkDerivation {
  pname = "llm-inference-bench";
  version = "0.4.29";

  src = fetchFromGitHub {
    owner = "local-inference-lab";
    repo = "llm-inference-bench";
    rev = "86cf05c2f42f4d21b909b6e684424ca1aab89fd5";
    hash = "sha256-wjUDTL5QFtL/o5ogSjLJmgObiPyrpj6rYm/SHVyAgFE=";
  };

  strictDeps = true;

  nativeBuildInputs = [ makeWrapper ];
  buildInputs = [ numactl ];

  buildPhase = ''
    runHook preBuild

    g++ -O3 -std=c++17 tools/amd_fabric/llm_amd_fabric.cpp \
      -o llm_amd_fabric -lnuma -pthread

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    install -D -m 0755 llm_decode_bench.py \
      "$out/share/llm-inference-bench/llm_decode_bench.py"
    install -D -m 0755 llm_cjk_watchdog.py \
      "$out/share/llm-inference-bench/llm_cjk_watchdog.py"
    cp -r data "$out/share/llm-inference-bench/data"

    install -D -m 0755 llm_amd_fabric \
      "$out/share/llm-inference-bench/tools/amd_fabric/llm_amd_fabric"
    mkdir -p "$out/bin"
    ln -s ../share/llm-inference-bench/tools/amd_fabric/llm_amd_fabric \
      "$out/bin/llm-amd-fabric"

    makeWrapper ${pythonEnv}/bin/python "$out/bin/llm-decode-bench" \
      --add-flags "$out/share/llm-inference-bench/llm_decode_bench.py"
    makeWrapper ${pythonEnv}/bin/python "$out/bin/llm-cjk-watchdog" \
      --add-flags "$out/share/llm-inference-bench/llm_cjk_watchdog.py"

    runHook postInstall
  '';

  # tools/p2pmark is CUDA/NVIDIA-only and is intentionally not built for this
  # ROCm/AMD-focused flake.

  meta = {
    description = "LLM inference benchmark and AMD NUMA/xGMI fabric diagnostic";
    homepage = "https://github.com/local-inference-lab/llm-inference-bench";
    license = lib.licenses.mit;
    mainProgram = "llm-decode-bench";
    platforms = [ "x86_64-linux" ];
  };
}
