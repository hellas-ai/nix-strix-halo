# Push a rendered dashboard to a live Grafana via the HTTP API.
#
# Usage:
#   push-grafana-dashboard [--file DASHBOARD.json] [--dev] \
#     (--ssh [USER@]HOST | --url GRAFANA_URL)
#
# Modes:
#   --ssh HOST   ssh to the Grafana host and POST via localhost using the
#                admin password file (NixOS services.grafana deployment).
#   --url URL    POST directly; auth from $GRAFANA_TOKEN (service account)
#                or $GRAFANA_USER/$GRAFANA_PASSWORD (basic auth).
#
# --dev pushes under a "-dev" uid / " (dev)" title so the provisioned copy
# is untouched while iterating. Env fallbacks: GRAFANA_SSH_HOST, GRAFANA_URL,
# GRAFANA_HTTP_PORT (default 3005, ssh mode only).

usage() {
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

DASHBOARD="$DEFAULT_DASHBOARD"
DEV=0
SSH_HOST="${GRAFANA_SSH_HOST:-}"
URL="${GRAFANA_URL:-}"
HTTP_PORT="${GRAFANA_HTTP_PORT:-3005}"

while [ $# -gt 0 ]; do
  case "$1" in
  --file)
    DASHBOARD="$2"
    shift 2
    ;;
  --dev)
    DEV=1
    shift
    ;;
  --ssh)
    SSH_HOST="$2"
    shift 2
    ;;
  --url)
    URL="$2"
    shift 2
    ;;
  *)
    usage
    ;;
  esac
done

if [ -z "$SSH_HOST" ] && [ -z "$URL" ]; then
  echo "error: need --ssh HOST or --url URL (or GRAFANA_SSH_HOST / GRAFANA_URL)" >&2
  usage
fi

PAYLOAD=$(jq -c --argjson dev "$DEV" '
  {
    dashboard: (.
      | .id = null
      | if $dev == 1 then .uid = .uid + "-dev" | .title = .title + " (dev)" else . end),
    overwrite: true,
    message: "push-grafana-dashboard"
  }' "$DASHBOARD")

if [ -n "$SSH_HOST" ]; then
  # shellcheck disable=SC2029 # $HTTP_PORT expands client-side by design
  RESPONSE=$(printf '%s' "$PAYLOAD" | ssh "$SSH_HOST" "
    set -eu
    PW=\$(cat /var/lib/grafana/grafana-password.secret)
    curl -fsS -u \"admin:\$PW\" -H 'Content-Type: application/json' \
      -X POST \"http://127.0.0.1:$HTTP_PORT/api/dashboards/db\" --data-binary @-
  ")
else
  AUTH_ARGS=()
  if [ -n "${GRAFANA_TOKEN:-}" ]; then
    AUTH_ARGS=(-H "Authorization: Bearer $GRAFANA_TOKEN")
  elif [ -n "${GRAFANA_USER:-}" ]; then
    AUTH_ARGS=(-u "$GRAFANA_USER:${GRAFANA_PASSWORD:-}")
  fi
  RESPONSE=$(printf '%s' "$PAYLOAD" | curl -fsS "${AUTH_ARGS[@]}" \
    -H 'Content-Type: application/json' \
    -X POST "$URL/api/dashboards/db" --data-binary @-)
fi

echo "$RESPONSE" | jq .
SLUG=$(echo "$RESPONSE" | jq -r .url)
echo "pushed: $SLUG" >&2
