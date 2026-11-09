{ python3, runCommand, writeShellApplication }:
let
  checked = runCommand "glm53-aot-stage-check" { nativeBuildInputs = [ python3 ]; } ''
    mkdir -p aot cache
    cp ${./stage.py} aot/stage.py
    cp ${./test_stage.py} aot/test_stage.py
    cp ${../cache/seed.py} cache/seed.py
    python3 aot/test_stage.py
    touch "$out"
  '';
in
writeShellApplication {
  name = "glm53-aot-stage";
  derivationArgs.stageCheck = checked;
  runtimeInputs = [ python3 ];
  text = ''
    exec python3 ${./stage.py} --seed-tool ${../cache/seed.py} "$@"
  '';
  meta.description = "Stage a GLM Triton AOT seed into a fresh writable cache";
}
