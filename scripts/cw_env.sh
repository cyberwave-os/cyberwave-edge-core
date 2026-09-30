#!/usr/bin/env bash
# cw_env: point this shell (and the edge-core service) at dev, staging or a local stack.
#
# Install:  install -m 755 cw_env.sh ~/.local/bin/cw_env
#           cw_env init --install      # one-time ~/.bashrc hook; see "Two ways to run it"
#
# Usage:
#   cw_env switch dev|staging          use that channel
#   cw_env switch local [HOST[:PORT]]  use a stack on another machine (host is remembered)
#   cw_env switch off                  stop overriding anything
#   cw_env show                        print what is active
#   cw_env update [dev|staging]        install/upgrade every package (default: active channel)
#   cw_env update ... --restart        ...then restart the edge-core service
#   cw_env switch dev --install        switch, then install one package (edge-core)
#   cw_env init [--install]            print (or append) the ~/.bashrc hook
#   cw_env export                      print the saved exports for eval
#
# The channel name on its own still works as a shorthand: `cw_env dev`.
#
# Two ways to run it, and the difference is the one thing to get right:
#   cw_env switch dev       the shell function `init --install` puts in
#                                 ~/.bashrc. Updates this shell and the files.
#   ./cw_env.sh switch dev  a child process: it bypasses that function and
#                                 updates only the files, never the calling shell
#                                 (a process cannot export into its parent).
#                                 Catch up with `eval "$(cw_env export)"`,
#                                 or run it as `source ./cw_env.sh ...`.
# New shells source env.sh from the same hook, so they start on the last channel.
#
# Why the difference bites: the CLI reads $CYBERWAVE_BASE_URL *before*
# credentials.json, so a stale value left in a shell silently outranks a switch
# and every command keeps hitting the old backend. `show` warns when they differ.
#
# A switch rewrites the service's routing in ~/.cyberwave/credentials.json (API
# URL, broker host/port/TLS/login); the service and its drivers follow on the next
# `sudo cyberwave edge restart`. The API token there belongs to one backend, so a
# dev<->staging switch is not complete without a fresh login:
#   cw_env switch staging && cyberwave login && sudo cyberwave edge restart
#
# Local stack (`local`):
#   Points this machine's edge core at a stack running `docker compose -f local.yml up`
#   on another machine (or this one). HOST can be 192.168.1.101, 192.168.1.101:8000
#   or http://192.168.1.101:8000; the API port defaults to 8000, the broker is
#   HOST:1883 without TLS, and it logs in with the stack's static account (test/test).
#   Without HOST it uses $CW_ENV_LOCAL_HOST, then the remembered host, then localhost.
#   `local` is not a package channel, so --install and update are refused. First run:
#     cw_env update dev                  # install the binaries from dev
#     cw_env switch local 192.168.1.101  # point shell and service at the stack
#     cyberwave login                          # get a token from the local backend
#     sudo cyberwave edge restart
#
# Overrides:
#   CW_ENV_PACKAGES      packages `update` walks (default: cyberwave-cli cyberwave-edge-core)
#   CW_ENV_VERSION       pin CYBERWAVE_INSTALL_VERSION (default: latest)
#   CW_ENV_LOCAL_MQTT_{PORT,USERNAME,PASSWORD}   local broker (default: 1883, test, test)
#   CW_ENV_RC_FILE       shell rc `init --install` appends to (default: ~/.bashrc)
#
# Buildkite registry tokens (dev/staging only; `local` does not use them):
#   Read only from ~/.config/cyberwave/<env>.tokens (chmod 600) and exported as
#   CYBERWAVE_INTERNAL_{DEB,PYTHON}_READ_TOKEN for install-internal.sh. env.sh
#   sources that file rather than copying the values, so a shell restored from
#   the snapshot has them too and a rotation takes effect with no re-switch.
#   The switch stops if the file or either token is missing. Get the values from Bitwarden,
#   shared Buildkite account entry (see cyberwave-clis/README.md). Tokens saved with
#   `cyberwave configure --internal-*-read-token` are not read here.
#
# Files:
#   ~/.config/cyberwave/<env>.tokens   registry read tokens, chmod 600 (dev/staging only)
#   ~/.config/cyberwave/local.host     last `local` host (not a secret)
#   ~/.config/cyberwave/env.sh         exports for the active channel, rewritten per switch
#                                      (sources <env>.tokens; holds no secret of its own)
#   ~/.cyberwave/credentials.json      read by the edge-core service; routing rewritten here

CW_ENV_CONFIG_DIR="${CW_ENV_CONFIG_DIR:-$HOME/.config/cyberwave}"
CW_ENV_INSTALL_URL="${CW_ENV_INSTALL_URL:-https://cyberwave.com/install-internal.sh}"
# Space-separated package basenames; the channel suffix is added by the installer.
CW_ENV_PACKAGES="${CW_ENV_PACKAGES:-cyberwave-cli cyberwave-edge-core}"

# The switch is remembered here: shell exports a new shell can source, written
# on every switch. Separate from credentials.json, which carries the same
# routing to the service rather than to a shell.
CW_ENV_FILE="${CW_ENV_FILE:-${CW_ENV_CONFIG_DIR}/env.sh}"

# The local stack is the machine running `docker compose -f local.yml up`.
CW_ENV_LOCAL_HOST_FILE="${CW_ENV_CONFIG_DIR}/local.host"
CW_ENV_LOCAL_DEFAULT_API_PORT=8000
CW_ENV_LOCAL_MQTT_PORT="${CW_ENV_LOCAL_MQTT_PORT:-1883}"

# The local stack's mosquitto keeps a static files-backend account granted
# `topic readwrite #` (local.yml: MQTT_BROKER_USERNAME/PASSWORD, default
# test/test). `local` uses it instead of the API token because the token path
# resolves its principal from a cache keyed on MQTT client id, and a miss
# denies every topic -- which surfaces as SUBACK 128 on webrtc-answer and a
# dead video stream rather than as an auth error. Override both if the stack
# sets MQTT_BROKER_USERNAME/PASSWORD to something else.
CW_ENV_LOCAL_MQTT_USERNAME="${CW_ENV_LOCAL_MQTT_USERNAME:-test}"
CW_ENV_LOCAL_MQTT_PASSWORD="${CW_ENV_LOCAL_MQTT_PASSWORD:-test}"
# Where `cyberwave login` keeps the runtime env ("envs") that the edge-core
# service and the driver containers read; this shell's exports never reach them.
CW_ENV_CREDENTIALS_FILE="${CW_ENV_CREDENTIALS_FILE:-$HOME/.cyberwave/credentials.json}"

# Every variable a switch owns, in one list so `off`, the per-switch reset and
# the env.sh snapshot can never drift apart.
CW_ENV_VARS=(
  CW_ENV_ACTIVE
  CYBERWAVE_BASE_URL
  CYBERWAVE_ENVIRONMENT
  CYBERWAVE_MQTT_HOST
  CYBERWAVE_MQTT_PORT
  CYBERWAVE_MQTT_USE_TLS
  CYBERWAVE_MQTT_USERNAME
  CYBERWAVE_MQTT_PASSWORD
  CYBERWAVE_INSTALL_CHANNEL
  CYBERWAVE_INSTALL_PACKAGE_BASENAME
  CYBERWAVE_INSTALL_PYTHON_PACKAGE
  CYBERWAVE_INSTALL_VERSION
)
# Cleared alongside the rest. Their *values* are never written to env.sh; the
# snapshot sources <env>.tokens instead, so there is only ever one copy on disk.
CW_ENV_SECRET_VARS=(
  CYBERWAVE_INTERNAL_DEB_READ_TOKEN
  CYBERWAVE_INTERNAL_PYTHON_READ_TOKEN
)

# Sourced into a shell, or run as a child process? Decides whether a switch can
# reach the caller's environment, so it is what the closing hint keys on.
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then CW_ENV_SOURCED=0; else CW_ENV_SOURCED=1; fi

cw_env__die() { echo "cw_env: $*" >&2; return 1; }

# Print the header comment, from line 2 up to the first blank line.
cw_env__help() {
  sed -n '2,/^$/{/^$/q;s/^# \{0,1\}//;s/^#$//;p}' "${BASH_SOURCE[0]:-$0}"
}

# Absolute path to this file, so the ~/.bashrc hook keeps working from any cwd.
cw_env__self_path() {
  local self="${BASH_SOURCE[0]:-$0}"
  readlink -f "$self" 2>/dev/null || echo "$self"
}

cw_env__base_url() {
  case "$1" in
    dev)     echo "https://api-dev.cyberwave.com" ;;
    staging) echo "https://api-staging.cyberwave.com" ;;
    local)   echo "http://${CW_ENV_LOCAL_HOST}:${CW_ENV_LOCAL_API_PORT}" ;;
  esac
}

# Same broker `cyberwave login` infers from the dev/staging API URL.
cw_env__mqtt_host() { echo "$1.mqtt.cyberwave.com"; }

cw_env__clear() { unset "${CW_ENV_VARS[@]}" "${CW_ENV_SECRET_VARS[@]}"; }

# Never print a token: just say whether it is there.
cw_env__mask() { [ -n "$1" ] && echo "set (${#1} chars)" || echo "unset"; }

cw_env__show() {
  if [ -z "${CW_ENV_ACTIVE:-}" ]; then
    echo "cw_env: no environment active in this shell"
    if [ -r "$CW_ENV_FILE" ]; then
      echo "  last switch saved in ${CW_ENV_FILE}; load it with: eval \"\$(cw_env export)\""
    fi
    return 0
  fi
  echo "cw_env: ${CW_ENV_ACTIVE}"
  echo "  CYBERWAVE_BASE_URL=${CYBERWAVE_BASE_URL:-}"
  if [ "${CW_ENV_ACTIVE}" = local ]; then
    echo "  CYBERWAVE_ENVIRONMENT=${CYBERWAVE_ENVIRONMENT:-}"
    echo "  CYBERWAVE_MQTT_HOST=${CYBERWAVE_MQTT_HOST:-}"
    echo "  CYBERWAVE_MQTT_PORT=${CYBERWAVE_MQTT_PORT:-}"
    echo "  CYBERWAVE_MQTT_USE_TLS=${CYBERWAVE_MQTT_USE_TLS:-}"
    echo "  CYBERWAVE_MQTT_USERNAME=${CYBERWAVE_MQTT_USERNAME:-}"
    echo "  CYBERWAVE_MQTT_PASSWORD=$(cw_env__mask "${CYBERWAVE_MQTT_PASSWORD:-}")"
    echo "  host read from: ${CW_ENV_LOCAL_HOST_SOURCE:-?}"
    echo "  shell exports: ${CW_ENV_FILE}"
    echo "  service routing: ${CW_ENV_CREDENTIALS_FILE} (applies after 'sudo cyberwave edge restart')"
    echo "  no package channel: --install/update are refused on local"
    return 0
  fi
  echo "  shell exports: ${CW_ENV_FILE}"
  echo "  service routing: ${CW_ENV_CREDENTIALS_FILE} (applies after 'sudo cyberwave edge restart')"
  echo "  CYBERWAVE_INSTALL_CHANNEL=${CYBERWAVE_INSTALL_CHANNEL:-}"
  echo "  CYBERWAVE_INSTALL_PACKAGE_BASENAME=${CYBERWAVE_INSTALL_PACKAGE_BASENAME:-}"
  echo "  CYBERWAVE_INSTALL_PYTHON_PACKAGE=${CYBERWAVE_INSTALL_PYTHON_PACKAGE:-}"
  echo "  CYBERWAVE_INSTALL_VERSION=${CYBERWAVE_INSTALL_VERSION:-<latest>}"
  echo "  deb token=$(cw_env__mask "${CYBERWAVE_INTERNAL_DEB_READ_TOKEN:-}")"
  echo "  python token=$(cw_env__mask "${CYBERWAVE_INTERNAL_PYTHON_READ_TOKEN:-}")"
  # Every path onto a channel loads these, so `unset` is never routine here --
  # it means update/--install would fail, and the two cases need different fixes.
  if [ -z "${CYBERWAVE_INTERNAL_DEB_READ_TOKEN:-}" ] || [ -z "${CYBERWAVE_INTERNAL_PYTHON_READ_TOKEN:-}" ]; then
    local tokens_file="${CW_ENV_CONFIG_DIR}/${CW_ENV_ACTIVE}.tokens"
    if [ -r "$tokens_file" ]; then
      # The snapshot sources the tokens file, so this shell is simply behind it
      # -- which re-running an *executed* switch cannot fix, since that only
      # rewrites the snapshot. Point at the reload, the way __warn_drift does.
      echo "    ^ ${tokens_file} is readable but not loaded in this shell — catch up with:"
      echo "        eval \"\$(cw_env export)\"    (or 'cw_env init --install', once)"
    else
      echo "    ^ ${tokens_file} is missing — update/--install will stop until it exists"
    fi
  fi
}

cw_env__load_tokens() {
  local tokens_file="${CW_ENV_CONFIG_DIR}/${1}.tokens"
  if [ ! -f "$tokens_file" ]; then
    cw_env__die "missing $tokens_file — create it with:
  install -m 600 /dev/null $tokens_file
  cat >> $tokens_file <<'EOF'
CYBERWAVE_INTERNAL_DEB_READ_TOKEN=bkrt_...
CYBERWAVE_INTERNAL_PYTHON_READ_TOKEN=bkrt_...
EOF"
    return 1
  fi
  # shellcheck disable=SC1090
  set -a; . "$tokens_file"; set +a
  [ -n "${CYBERWAVE_INTERNAL_DEB_READ_TOKEN:-}" ] || { cw_env__die "CYBERWAVE_INTERNAL_DEB_READ_TOKEN not set in $tokens_file"; return 1; }
  [ -n "${CYBERWAVE_INTERNAL_PYTHON_READ_TOKEN:-}" ] || { cw_env__die "CYBERWAVE_INTERNAL_PYTHON_READ_TOKEN not set in $tokens_file"; return 1; }
}

# "http://host:8000/path" -> "host:8000". Accepts a bare host or host:port too.
cw_env__authority() {
  local raw="${1#http://}"
  raw="${raw#https://}"
  echo "${raw%%/*}"
}

cw_env__host_of() { local a; a="$(cw_env__authority "$1")"; echo "${a%%:*}"; }

cw_env__port_of() {
  local a; a="$(cw_env__authority "$1")"
  case "$a" in *:*) echo "${a##*:}" ;; esac
}

# Precedence: argument > $CW_ENV_LOCAL_HOST > local.host file > localhost,
# so a bare `cw_env switch local` keeps working after the first time.
cw_env__resolve_local_host() {
  local requested="$1" from="argument"
  if [ -z "$requested" ]; then
    if [ -n "${CW_ENV_LOCAL_HOST:-}" ]; then
      requested="${CW_ENV_LOCAL_HOST}:${CW_ENV_LOCAL_API_PORT:-$CW_ENV_LOCAL_DEFAULT_API_PORT}"
      from="\$CW_ENV_LOCAL_HOST"
    elif [ -r "$CW_ENV_LOCAL_HOST_FILE" ]; then
      requested="$(cat "$CW_ENV_LOCAL_HOST_FILE")"
      from="$CW_ENV_LOCAL_HOST_FILE"
    else
      requested="localhost"
      from="default (same machine)"
    fi
  fi

  CW_ENV_LOCAL_HOST="$(cw_env__host_of "$requested")"
  CW_ENV_LOCAL_API_PORT="$(cw_env__port_of "$requested")"
  CW_ENV_LOCAL_API_PORT="${CW_ENV_LOCAL_API_PORT:-$CW_ENV_LOCAL_DEFAULT_API_PORT}"
  [ -n "$CW_ENV_LOCAL_HOST" ] || { cw_env__die "could not read a host out of '$requested'"; return 1; }
  CW_ENV_LOCAL_HOST_SOURCE="$from"

  # Failing to remember it is not a reason to fail the switch.
  if [ "$from" = argument ]; then
    if mkdir -p "$CW_ENV_CONFIG_DIR" 2>/dev/null; then
      echo "${CW_ENV_LOCAL_HOST}:${CW_ENV_LOCAL_API_PORT}" > "$CW_ENV_LOCAL_HOST_FILE" 2>/dev/null \
        || echo "cw_env: could not remember the host in $CW_ENV_LOCAL_HOST_FILE" >&2
    fi
  fi
}

# Save the switch as shell exports, so a new shell -- or an executed
# `./cw_env switch dev`, which cannot touch its parent's environment --
# can pick the channel up with `. env.sh` / `eval "$(cw_env export)"`.
# The leading unsets make the file a complete description of the active channel:
# sourcing it over a different one cannot leave a stale key behind (`local`'s
# broker login bleeding into dev is the case that matters).
cw_env__write_env_file() {
  local var tmp tokens_file
  mkdir -p "$CW_ENV_CONFIG_DIR" 2>/dev/null || {
    echo "cw_env: could not create $CW_ENV_CONFIG_DIR; this switch is not saved" >&2
    return 0
  }
  tmp="${CW_ENV_FILE}.cw-env-tmp"
  {
    echo "# Written by cw_env; rewritten on every switch. Do not edit."
    echo "# Source it to put a shell on the channel that was switched to last."
    printf 'unset %s\n' "${CW_ENV_VARS[@]}" "${CW_ENV_SECRET_VARS[@]}"
    for var in "${CW_ENV_VARS[@]}"; do
      if [ -n "${!var:-}" ]; then printf 'export %s=%q\n' "$var" "${!var}"; fi
    done
    # Pull the registry tokens in by sourcing their file rather than copying
    # the values, so a shell restored from here is identical to one from a
    # sourced switch -- and "unset" in `show` keeps meaning something is wrong.
    # Copying would put a second copy of a secret on disk to rotate, and would
    # make `cw_env export` print tokens to stdout.
    if [ -n "${CW_ENV_ACTIVE:-}" ] && [ "$CW_ENV_ACTIVE" != local ]; then
      tokens_file="${CW_ENV_CONFIG_DIR}/${CW_ENV_ACTIVE}.tokens"
      printf 'if [ -r %q ]; then set -a; . %q; set +a; fi\n' "$tokens_file" "$tokens_file"
    fi
  } > "$tmp" 2>/dev/null || {
    rm -f "$tmp" 2>/dev/null
    echo "cw_env: could not write $CW_ENV_FILE; this switch is not saved" >&2
    return 0
  }
  chmod 600 "$tmp" 2>/dev/null || true
  mv -f "$tmp" "$CW_ENV_FILE" 2>/dev/null || {
    rm -f "$tmp" 2>/dev/null
    echo "cw_env: could not write $CW_ENV_FILE; this switch is not saved" >&2
  }
  return 0
}

# Read one exported value back out of the snapshot, so `show` can compare the
# channel this shell holds against the channel the last switch chose.
cw_env__saved_value() {
  local key="$1" value
  [ -r "$CW_ENV_FILE" ] || return 0
  value="$(sed -n "s/^export ${key}=//p" "$CW_ENV_FILE" | tail -1)"
  # printf %q leaves plain channel names and URLs bare, but strip a quote pair
  # anyway so the comparison never turns on quoting.
  value="${value#\'}"
  echo "${value%\'}"
}

# The failure this catches is silent: an executed switch updates env.sh and the
# service while the calling shell keeps the channel it was on, and the shell is
# what the CLI reads first -- so `cyberwave login` quietly dials the old backend
# (Connection refused, if that was a local stack that is no longer up).
cw_env__warn_drift() {
  local saved_active
  saved_active="$(cw_env__saved_value CW_ENV_ACTIVE)"
  [ -n "$saved_active" ] || return 0
  [ "${CW_ENV_ACTIVE:-}" = "$saved_active" ] && return 0
  echo "cw_env: this shell is on '${CW_ENV_ACTIVE:-<none>}' but the last switch chose '${saved_active}'." >&2
  echo "  The shell wins for the CLI: 'cyberwave login' would use ${CYBERWAVE_BASE_URL:-<the SDK default>}." >&2
  echo "  Catch up with:  eval \"\$(cw_env export)\"" >&2
}

cw_env__print_env() {
  if [ ! -r "$CW_ENV_FILE" ]; then
    cw_env__die "nothing saved yet — run 'cw_env switch dev|staging|local' first"
    return 1
  fi
  cat "$CW_ENV_FILE"
}

CW_ENV_HOOK_BEGIN="# >>> cw_env >>>"
CW_ENV_HOOK_END="# <<< cw_env <<<"

# The hook does the two things a child process cannot do for itself: it makes
# `cw_env` a shell function, so a switch lands in the calling shell, and
# it starts every new shell on the channel chosen last.
cw_env__hook() {
  local self
  self="$(cw_env__self_path)"
  cat <<HOOK
${CW_ENV_HOOK_BEGIN}
cw_env() { source "${self}" "\$@"; }
if [ -r "${CW_ENV_FILE}" ]; then . "${CW_ENV_FILE}"; fi
${CW_ENV_HOOK_END}
HOOK
}

cw_env__init() {
  local do_install="$1" rc_file="${CW_ENV_RC_FILE:-$HOME/.bashrc}"

  if [ "$do_install" != 1 ]; then
    echo "cw_env: add this to ${rc_file}, or run 'cw_env init --install':"
    echo
    cw_env__hook
    return 0
  fi

  if [ -f "$rc_file" ] && grep -qF "$CW_ENV_HOOK_BEGIN" "$rc_file"; then
    echo "cw_env: hook already present in ${rc_file}"
    return 0
  fi
  { echo; cw_env__hook; } >> "$rc_file" || {
    cw_env__die "could not append the hook to ${rc_file}"
    return 1
  }
  echo "cw_env: hook appended to ${rc_file}"
  echo "  open a new shell (or 'source ${rc_file}'), then 'cw_env switch dev' switches this shell too"
}

# Rewrite the service's routing in the credentials file so it matches the
# channel just activated. `cyberwave login` persists only ENVIRONMENT, BASE_URL,
# MQTT_HOST and MQTT_PORT, and only at login time, so without this a switch would
# leave the service on the previous backend -- or worse, half on each (e.g. the
# local broker login sent to dev's broker). edge-core loads every CYBERWAVE_* key
# from "envs" at startup and driver_launcher forwards them into each driver
# container, so one write reaches the service and every driver. Takes effect on
# the next `cyberwave edge restart`.
#   Arguments: KEY=VALUE to set, -KEY to remove.
cw_env__sync_service_env() {
  if [ ! -f "$CW_ENV_CREDENTIALS_FILE" ]; then
    echo "cw_env: no $CW_ENV_CREDENTIALS_FILE yet; the service is not paired." >&2
    echo "  Run 'cyberwave login', then 'cw_env switch ${CW_ENV_ACTIVE:-<env>}' again." >&2
    return 0
  fi

  CW_FILE="$CW_ENV_CREDENTIALS_FILE" python3 - "$@" <<'PY' \
    || echo "cw_env: could not update $CW_ENV_CREDENTIALS_FILE" >&2
import json, os, pathlib, sys

path = pathlib.Path(os.environ["CW_FILE"])
try:
    data = json.loads(path.read_text())
except (OSError, ValueError):
    raise SystemExit(1)
if not isinstance(data, dict):
    raise SystemExit(1)

envs = data.get("envs")
envs = dict(envs) if isinstance(envs, dict) else {}
before = dict(envs)

for arg in sys.argv[1:]:
    if arg.startswith("-"):
        envs.pop(arg[1:], None)
    else:
        key, _, value = arg.partition("=")
        envs[key] = value

# The token was issued by whichever backend the service last logged in to.
old_url, new_url = before.get("CYBERWAVE_BASE_URL"), envs.get("CYBERWAVE_BASE_URL")
if old_url and old_url != new_url:
    print(f"cw_env: the service token is for {old_url}; run 'cyberwave login' "
          f"to get one for {new_url}, then 'sudo cyberwave edge restart'.", file=sys.stderr)

# The update loop re-activates a channel once per package; don't rewrite the
# file (and don't churn its mtime) when it is already correct.
if envs == before:
    raise SystemExit(0)

data["envs"] = envs
tmp = path.with_name(path.name + ".cw-env-tmp")
# 0600 from the first byte: the file carries the service's API token, and a
# chmod after the write leaves it umask-readable in between. The chmod stays for
# a tmp left behind by an earlier run, whose mode O_CREAT would not change.
fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as handle:
    handle.write(json.dumps(data, indent=2) + "\n")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
}

cw_env__activate() {
  local env_name="$1" host_arg="${2:-}"
  local deb_token="" python_token=""

  # Everything that can fail runs before the clear, so a switch that stops --
  # a missing tokens file, a host with no hostname in it -- leaves this shell on
  # the channel it was already on instead of on no channel at all. The tokens go
  # into locals because the clear unsets what load_tokens exported. Both checks
  # are idempotent; `update` calls activate once per package.
  if [ "$env_name" = local ]; then
    cw_env__resolve_local_host "$host_arg" || return 1
  else
    cw_env__load_tokens "$env_name" || return 1
    deb_token="$CYBERWAVE_INTERNAL_DEB_READ_TOKEN"
    python_token="$CYBERWAVE_INTERNAL_PYTHON_READ_TOKEN"
  fi

  cw_env__clear

  if [ "$env_name" = local ]; then
    export CYBERWAVE_BASE_URL="$(cw_env__base_url local)"
    export CYBERWAVE_ENVIRONMENT="local"
    # Left unset, the SDK falls back to the production broker over TLS, and the
    # CLI only infers a local broker for localhost/127.0.0.1 (never a LAN IP),
    # so a stack reached by IP needs its broker set explicitly.
    export CYBERWAVE_MQTT_HOST="$CW_ENV_LOCAL_HOST"
    export CYBERWAVE_MQTT_PORT="$CW_ENV_LOCAL_MQTT_PORT"
    export CYBERWAVE_MQTT_USE_TLS="false"
    # The API-token auth dev and staging use is not dependable against a local
    # broker; see CW_ENV_LOCAL_MQTT_USERNAME.
    export CYBERWAVE_MQTT_USERNAME="$CW_ENV_LOCAL_MQTT_USERNAME"
    export CYBERWAVE_MQTT_PASSWORD="$CW_ENV_LOCAL_MQTT_PASSWORD"
    export CW_ENV_ACTIVE="local"
    cw_env__sync_service_env \
      "CYBERWAVE_BASE_URL=$CYBERWAVE_BASE_URL" \
      "CYBERWAVE_ENVIRONMENT=local" \
      "CYBERWAVE_MQTT_HOST=$CYBERWAVE_MQTT_HOST" \
      "CYBERWAVE_MQTT_PORT=$CYBERWAVE_MQTT_PORT" \
      "CYBERWAVE_MQTT_USE_TLS=false" \
      "CYBERWAVE_MQTT_USERNAME=$CYBERWAVE_MQTT_USERNAME" \
      "CYBERWAVE_MQTT_PASSWORD=$CYBERWAVE_MQTT_PASSWORD"
    return 0
  fi

  export CYBERWAVE_INTERNAL_DEB_READ_TOKEN="$deb_token"
  export CYBERWAVE_INTERNAL_PYTHON_READ_TOKEN="$python_token"
  export CYBERWAVE_BASE_URL="$(cw_env__base_url "$env_name")"
  export CYBERWAVE_INSTALL_CHANNEL="$env_name"
  export CYBERWAVE_INSTALL_PACKAGE_BASENAME="${CW_ENV_PACKAGE_BASENAME:-cyberwave-edge-core}"
  export CYBERWAVE_INSTALL_PYTHON_PACKAGE="${CW_ENV_PYTHON_PACKAGE:-cyberwave-edge-core}"
  if [ -n "${CW_ENV_VERSION:-}" ]; then export CYBERWAVE_INSTALL_VERSION="$CW_ENV_VERSION"; fi
  export CW_ENV_ACTIVE="$env_name"

  # dev and staging authenticate to the broker with the API token over TLS on
  # 8883 (inferred from the port). A broker login or USE_TLS=false left behind
  # by `local` would override that for the service and every driver.
  cw_env__sync_service_env \
    "CYBERWAVE_BASE_URL=$CYBERWAVE_BASE_URL" \
    "CYBERWAVE_ENVIRONMENT=$env_name" \
    "CYBERWAVE_MQTT_HOST=$(cw_env__mqtt_host "$env_name")" \
    "CYBERWAVE_MQTT_PORT=8883" \
    -CYBERWAVE_MQTT_USE_TLS \
    -CYBERWAVE_MQTT_USERNAME \
    -CYBERWAVE_MQTT_PASSWORD
}

cw_env__off() {
  cw_env__clear
  # An env.sh of nothing but unsets, so new shells stop inheriting the channel
  # too. The service keeps whatever credentials.json says; only a switch moves it.
  cw_env__write_env_file
  echo "cw_env: cleared (the service stays on ${CW_ENV_CREDENTIALS_FILE})"
}

cw_env__no_local_channel() {
  cw_env__die "'local' is not a package channel — the registry publishes stable/dev/staging only.
  Install the binaries from dev, then point them at the local stack:
    cw_env update dev && cw_env switch local ${CW_ENV_LOCAL_HOST:-<stack-ip>}"
}

cw_env__install() {
  [ "${CW_ENV_ACTIVE:-}" = local ] && { cw_env__no_local_channel; return 1; }
  echo "cw_env: installing ${CYBERWAVE_INSTALL_PYTHON_PACKAGE} from channel ${CYBERWAVE_INSTALL_CHANNEL}"
  curl -fsSL "$CW_ENV_INSTALL_URL" | bash
}

# install-internal.sh installs one package per run, so `update` runs it once
# per package in CW_ENV_PACKAGES.
cw_env__update() {
  local env_name="$1" do_restart="$2"
  local pkg rc=0 failed=""
  local saved_basename="${CW_ENV_PACKAGE_BASENAME:-}" saved_python="${CW_ENV_PYTHON_PACKAGE:-}"

  [ "$env_name" = local ] && { cw_env__no_local_channel; return 1; }

  echo "cw_env: updating on channel '${env_name}':"
  for pkg in $CW_ENV_PACKAGES; do echo "  - ${pkg}-${env_name}"; done

  for pkg in $CW_ENV_PACKAGES; do
    echo
    echo "cw_env: ==> ${pkg}"
    CW_ENV_PACKAGE_BASENAME="$pkg"
    CW_ENV_PYTHON_PACKAGE="$pkg"
    if ! cw_env__activate "$env_name"; then
      rc=1; failed="${failed} ${pkg}"
      break
    fi
    if ! cw_env__install; then
      cw_env__die "failed to update ${pkg} on ${env_name}"
      rc=1; failed="${failed} ${pkg}"
    fi
  done

  # Leave the shell on the channel's defaults, not the last package in the loop.
  if [ -n "$saved_basename" ]; then CW_ENV_PACKAGE_BASENAME="$saved_basename"; else unset CW_ENV_PACKAGE_BASENAME; fi
  if [ -n "$saved_python" ]; then CW_ENV_PYTHON_PACKAGE="$saved_python"; else unset CW_ENV_PYTHON_PACKAGE; fi
  cw_env__activate "$env_name" || return 1
  cw_env__write_env_file

  echo
  if [ "$rc" = 0 ]; then
    echo "cw_env: all packages updated on '${env_name}'"
  else
    cw_env__die "update failed for:${failed}"
  fi
  cw_env__show

  if [ "$do_restart" = 1 ]; then
    cw_env__restart_edge || rc=1
  elif [ "$rc" = 0 ]; then
    echo "cw_env: run 'cw_env update ${env_name} --restart' (or 'sudo cyberwave edge restart') to load the new binary"
  fi

  return "$rc"
}

# A restart also re-pulls mutable-tagged (dev/latest/staging/...) worker images.
cw_env__restart_edge() {
  local cyberwave_bin
  cyberwave_bin="$(command -v cyberwave 2>/dev/null)" || true
  if [ -z "$cyberwave_bin" ]; then
    cw_env__die "cyberwave CLI not on PATH — cannot restart edge-core"
    return 1
  fi
  echo "cw_env: restarting edge-core service"
  if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    "$cyberwave_bin" edge restart
  else
    sudo -E "$cyberwave_bin" edge restart
  fi
}

# Run as a child process a switch reaches the service and the saved exports but
# not the caller's environment, so say so rather than let the shell look stale.
cw_env__shell_hint() {
  [ "$CW_ENV_SOURCED" = 1 ] && return 0
  echo
  echo "cw_env: the service routing and ${CW_ENV_FILE} are updated; this shell is not."
  echo "  load it here:  eval \"\$(cw_env export)\""
  echo "  or once:       cw_env init --install   # then a switch reaches the shell directly"
}

cw_env__main() {
  local cmd="" env_name="" host_arg="" do_install=0 do_restart=0

  while [ $# -gt 0 ]; do
    case "$1" in
      switch|use)        cmd="switch" ;;
      update)            cmd="update" ;;
      show|status)       cmd="show" ;;
      off|clear)         cmd="off" ;;
      export|env)        cmd="export" ;;
      init|shell-init)   cmd="init" ;;
      # A bare channel name is the shorthand for `switch <channel>`.
      dev|staging|local) env_name="$1"; [ -n "$cmd" ] || cmd="switch" ;;
      --install|-i)      do_install=1 ;;
      --restart|-r)      do_restart=1 ;;
      -h|--help)         cw_env__help; return 0 ;;
      # Only `local` takes a host, and only after it has been named, so a
      # misspelled channel is an error instead of becoming a hostname.
      *)
        if [ "$env_name" = local ] && [ -z "$host_arg" ]; then
          host_arg="$1"
        else
          cw_env__die "unknown argument '$1' (expected switch|update|show|off|export|init, or dev|staging|local)"; return 1
        fi
        ;;
    esac
    shift
  done

  case "$cmd" in
    show)   cw_env__show; cw_env__warn_drift; return 0 ;;
    off)    cw_env__off; cw_env__shell_hint; return 0 ;;
    export) cw_env__print_env; return $? ;;
    init)   cw_env__init "$do_install"; return $? ;;
  esac

  # `cw_env update` with no channel reuses whatever is already active.
  if [ "$cmd" = update ] && [ -z "$env_name" ]; then
    env_name="${CW_ENV_ACTIVE:-}"
    [ -n "$env_name" ] || { cw_env__die "no active environment — use 'cw_env update dev|staging'"; return 1; }
  fi

  [ -n "$env_name" ] || { cw_env__die "specify a channel: 'cw_env switch dev|staging|local'"; return 1; }

  if [ "$cmd" = update ]; then
    cw_env__update "$env_name" "$do_restart" || return 1
    cw_env__shell_hint
    return 0
  fi

  cw_env__activate "$env_name" "$host_arg" || return 1
  cw_env__write_env_file
  cw_env__show

  if [ "$do_install" = 1 ]; then
    cw_env__install || return 1
  fi

  if [ "$do_restart" = 1 ]; then
    cw_env__restart_edge || return 1
  fi

  cw_env__shell_hint
}

# Executed, not sourced: the exports die with this process, but the switch
# itself (credentials.json, the remembered host, env.sh) does not, so every
# subcommand is allowed and __shell_hint says what the caller still has to do.
if [ "$CW_ENV_SOURCED" = 0 ]; then
  set -euo pipefail
fi
cw_env__main "$@"
