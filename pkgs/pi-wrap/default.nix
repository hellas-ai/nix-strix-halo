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

    context_window="''${PI_CONTEXT_WINDOW:-32768}"
    max_tokens="''${PI_MAX_TOKENS:-2048}"
    reasoning="''${PI_REASONING:-false}"
    case "$context_window:$max_tokens" in
      *[!0-9:]*|:*|*:|0:*|*:0) echo "pi-wrap: context/token limits must be positive integers" >&2; exit 2 ;;
    esac
    case "$reasoning" in
      true|false) ;;
      *) echo "pi-wrap: PI_REASONING must be true or false" >&2; exit 2 ;;
    esac

    if [ -n "''${PI_WRAP_RUNTIME_DIR:-}" ]; then
      runtime_root="$PI_WRAP_RUNTIME_DIR"
    elif [ -n "''${XDG_RUNTIME_DIR:-}" ]; then
      runtime_root="$XDG_RUNTIME_DIR/pi-wrap"
    else
      echo "pi-wrap: set XDG_RUNTIME_DIR or PI_WRAP_RUNTIME_DIR" >&2
      exit 2
    fi
    case "$runtime_root/" in
      /tmp/*) echo "pi-wrap: refusing a /tmp runtime directory" >&2; exit 2 ;;
    esac
    mkdir -p -- "$runtime_root"
    chmod 700 -- "$runtime_root"
    session_dir="$(mktemp -d "$runtime_root/session.XXXXXX")"
    extension="$session_dir/provider.js"
    cleanup() {
      rm -rf -- "$session_dir"
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
          reasoning: $reasoning,
          input: ["text"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: $context_window,
          maxTokens: $max_tokens,
        }],
      });
    }
    EOF

    ${pi}/bin/pi -e "$extension" --provider "$provider_id" --model "$model" "$@"
  '';
}
