{
  coreutils,
  jq,
  pi,
  writeShellApplication,
}:

writeShellApplication {
  name = "pi-wrap";
  runtimeInputs = [
    coreutils
    jq
  ];

  text = ''
    set -euo pipefail

    json_string() {
      jq -Rn --arg value "$1" '$value'
    }

    provider="''${PI_PROVIDER:-}"
    if [ -z "$provider" ]; then
      if [ -n "''${ANTHROPIC_BASE_URL:-}" ] && [ -z "''${OPENAI_BASE_URL:-}" ]; then
        provider="anthropic"
      else
        provider="openai"
      fi
    fi

    case "$provider" in
      openai|openai-env)
        provider_id="openai-env"
        api="openai-completions"
        base_url="''${OPENAI_BASE_URL:-https://api.openai.com/v1}"
        api_key="''${OPENAI_API_KEY:-unused}"
        model="''${OPENAI_MODEL:-''${PI_MODEL:-}}"
        ;;
      anthropic|anthropic-env)
        provider_id="anthropic-env"
        api="anthropic-messages"
        base_url="''${ANTHROPIC_BASE_URL:-https://api.anthropic.com}"
        api_key="''${ANTHROPIC_API_KEY:-unused}"
        model="''${ANTHROPIC_MODEL:-''${PI_MODEL:-}}"
        ;;
      *)
        echo "pi-wrap: unsupported PI_PROVIDER='$provider'" >&2
        exit 2
        ;;
    esac

    if [ -z "$model" ]; then
      echo "pi-wrap: set OPENAI_MODEL, ANTHROPIC_MODEL, or PI_MODEL" >&2
      exit 2
    fi

    extension="$(mktemp --suffix=.js -t pi-wrap-XXXXXX)"
    cleanup() {
      rm -f "$extension"
    }
    trap cleanup EXIT

    provider_json="$(json_string "$provider_id")"
    base_json="$(json_string "$base_url")"
    key_json="$(json_string "$api_key")"
    api_json="$(json_string "$api")"
    model_json="$(json_string "$model")"

    cat > "$extension" <<EOF
    export default function (pi) {
      const model = $model_json;
      pi.registerProvider($provider_json, {
        baseUrl: $base_json,
        apiKey: $key_json,
        api: $api_json,
        models: [{
          id: model,
          name: model,
          reasoning: false,
          input: ["text"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: 32768,
          maxTokens: 2048,
        }],
      });
    }
    EOF

    exec ${pi}/bin/pi -e "$extension" --provider "$provider_id" --model "$model" "$@"
  '';
}
