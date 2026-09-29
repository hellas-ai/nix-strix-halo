{
  coreutils,
  lib,
  pi,
  writeShellApplication,
  writeText,
}:

let
  models = writeText "glm53-pi-models.json" (
    builtins.toJSON {
      providers.strix-glm = {
        baseUrl = "http://127.0.0.1:30053/v1";
        api = "openai-completions";
        apiKey = "local";
        compat = {
          supportsDeveloperRole = false;
          supportsReasoningEffort = true;
        };
        models = [
          {
            id = "glm-5.3-flash";
            name = "GLM-5.3-Flash on four Strix nodes";
            reasoning = true;
            input = [ "text" ];
            contextWindow = 131072;
            maxTokens = 8192;
            cost = {
              input = 0;
              output = 0;
              cacheRead = 0;
              cacheWrite = 0;
            };
          }
        ];
      };
    }
  );
in
writeShellApplication {
  name = "glm53-pi";
  runtimeInputs = [ coreutils ];
  text = ''
    agent_dir="''${GLM_PI_DIR:-''${XDG_STATE_HOME:-$HOME/.local/state}/glm53-pi}"
    mkdir -p "$agent_dir"
    if [ ! -e "$agent_dir/models.json" ]; then
      cp ${models} "$agent_dir/models.json"
      chmod u+w "$agent_dir/models.json"
    fi
    export PI_CODING_AGENT_DIR="$agent_dir"
    exec ${lib.getExe pi} --offline --provider strix-glm \
      --model glm-5.3-flash --thinking low "$@"
  '';
  meta.description = "Run Pi against the four-node GLM-5.3-Flash endpoint";
}
