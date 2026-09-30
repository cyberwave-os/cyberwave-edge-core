#!/usr/bin/env bash
# Upgrade a running edge-core service to PACKAGE_FILE under a real systemd
# (plan B1), which test_emulated_binary_package.sh cannot do: without systemd
# the deb's postinst always takes its "nothing to manage" branch.
#
# Starting from the unit an older `cyberwave edge install` wrote, with a custom
# config dir and a drop-in, it checks that `dpkg -i`:
#   - rewrites the unit with the ExecStartPre update-guard hook, keeping
#     ExecStart, the config dir and the drop-in;
#   - runs daemon-reload and does not restart the running service;
#   - is a no-op when the postinst or the install runs again.
# It then restarts the service with the guard missing, succeeding and failing,
# and checks that the service starts every time.
#
# Last, it runs `cyberwave edge update` (the CLI from this checkout) against a
# local apt repository holding the same deb repacked as stable 1.0.0 and 1.0.1:
# --check, an update, an update with nothing new, a rollback with --version,
# and a held package without and with --force. Each step checks the installed
# version, whether the service restarted, and that the unit is left unchanged.
#
# The service's main process is a sleep stub (a drop-in replaces ExecStart),
# because the real orchestrator needs credentials and a backend.
#
# On the runner: PACKAGE_FILE=path/to/cyberwave-edge-core*.deb ./test_deb_upgrade_under_systemd.sh
# It boots a privileged, systemd-enabled Ubuntu 22.04 container and runs this
# same script inside it with --in-container. The container is removed on exit.
# Only the deb and the CLI and SDK sources are mounted, read-only, so nothing
# root-owned is left in the runner's workspace. Installing the CLI needs PyPI.

set -euo pipefail

UNIT_NAME=cyberwave-edge-core.service
UNIT=/etc/systemd/system/${UNIT_NAME}
DROPIN=/etc/systemd/system/${UNIT_NAME}.d/50-ci-stub.conf
CONFIG_DIR=/opt/cyberwave-ci-config
GUARD=/usr/lib/cyberwave-edge-core/update-guard
DEB_IN_CONTAINER=/tmp/cyberwave-edge-core.deb
CLI_SRC_IN_CONTAINER=/src/cyberwave-python-cli
SDK_SRC_IN_CONTAINER=/src/cyberwave-python
STABLE_PACKAGE=cyberwave-edge-core
REPO_DIR=/srv/cyberwave-ci-apt
CLI=/opt/cli-venv/bin/cyberwave

# ---------------------------------------------------------------------------
# Inside the container
# ---------------------------------------------------------------------------

dump_state() {
    echo "----- ${UNIT} -----"
    cat "${UNIT}" || true
    echo "----- systemctl status -----"
    systemctl status "${UNIT_NAME}" --no-pager -l || true
    echo "----- journal -----"
    journalctl -u "${UNIT_NAME}" --no-pager -n 50 || true
}

fail() {
    echo "❌ $*" >&2
    dump_state >&2
    exit 1
}

main_pid() {
    systemctl show -p MainPID --value "${UNIT_NAME}"
}

wait_active() {
    for _ in $(seq 1 30); do
        systemctl is-active --quiet "${UNIT_NAME}" && return 0
        sleep 1
    done
    fail "${UNIT_NAME} did not become active"
}

assert_unit_line() {
    grep -qxF "$1" "${UNIT}" || fail "${UNIT} lacks the line: $1"
}

restart_and_check() {
    local before
    before="$(main_pid)"
    systemctl restart "${UNIT_NAME}" || fail "systemctl restart failed ($1)"
    wait_active
    [ "$(main_pid)" != "${before}" ] || fail "service was not restarted ($1)"
    echo "✅ service restarts with the guard ${1}"
}

run_checks() {
    local package_name postinst dropin_sum unit_sum pid_before

    echo "=== Starting from a unit written by an older 'cyberwave edge install' ==="
    printf '#!/bin/sh\nexec sleep infinity\n' > /usr/local/bin/fake-edge-core
    chmod 755 /usr/local/bin/fake-edge-core
    mkdir -p "${CONFIG_DIR}" "$(dirname "${DROPIN}")"
    # The CLI's pre-B1 template (cyberwave_cli/core.py SYSTEMD_UNIT_TEMPLATE),
    # rendered with a custom config dir.
    cat > "${UNIT}" <<EOF
[Unit]
Description=Cyberwave Edge Core Orchestrator
After=network-online.target docker.service
Wants=network-online.target docker.service

[Service]
Type=notify
NotifyAccess=all
ExecStart=/usr/bin/cyberwave-edge-core
Restart=always
RestartSec=5
WatchdogSec=60
TimeoutStartSec=900
Environment=CYBERWAVE_EDGE_CONFIG_DIR=${CONFIG_DIR}
OOMScoreAdjust=-800
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cyberwave-edge-core

[Install]
WantedBy=multi-user.target
EOF
    cat > "${DROPIN}" <<'EOF'
[Service]
Type=simple
WatchdogSec=0
ExecStart=
ExecStart=/usr/local/bin/fake-edge-core
EOF
    systemctl daemon-reload
    systemctl enable --now "${UNIT_NAME}"
    wait_active
    pid_before="$(main_pid)"
    [ "${pid_before}" -gt 0 ] || fail "no main PID before the upgrade"
    dropin_sum="$(sha256sum "${DROPIN}")"

    echo "=== Installing $(dpkg-deb -f "${DEB_IN_CONTAINER}" Package) $(dpkg-deb -f "${DEB_IN_CONTAINER}" Version) ==="
    dpkg -i "${DEB_IN_CONTAINER}" 2>&1 | tee /tmp/dpkg-install.log
    grep -qF "ensure-systemd-unit: wrote ${UNIT}" /tmp/dpkg-install.log \
        || fail "the postinst did not rewrite the unit"
    grep -qF "systemd reloaded (service not restarted)" /tmp/dpkg-install.log \
        || fail "the postinst did not run daemon-reload"
    assert_unit_line "ExecStartPre=-/bin/sh -c 'if [ -x ${GUARD} ]; then exec ${GUARD}; fi'"
    assert_unit_line "ExecStart=/usr/bin/cyberwave-edge-core"
    assert_unit_line "Environment=CYBERWAVE_EDGE_CONFIG_DIR=${CONFIG_DIR}"
    [ "$(sha256sum "${DROPIN}")" = "${dropin_sum}" ] || fail "the drop-in was changed"
    [ "$(systemctl show -p NeedDaemonReload --value "${UNIT_NAME}")" = "no" ] \
        || fail "systemd was not reloaded after the unit changed"
    systemctl is-active --quiet "${UNIT_NAME}" || fail "the upgrade stopped the service"
    [ "$(main_pid)" = "${pid_before}" ] || fail "the upgrade restarted the service"
    echo "✅ upgrade rewrote the unit, kept ExecStart, config dir and drop-in, did not restart"

    echo "=== Running the postinst and the install again ==="
    package_name="$(dpkg-deb -f "${DEB_IN_CONTAINER}" Package)"
    postinst="/var/lib/dpkg/info/${package_name}.postinst"
    unit_sum="$(sha256sum "${UNIT}")"
    "${postinst}" configure 2>&1 | tee /tmp/postinst-rerun.log
    dpkg -i "${DEB_IN_CONTAINER}" 2>&1 | tee /tmp/dpkg-reinstall.log
    for log in /tmp/postinst-rerun.log /tmp/dpkg-reinstall.log; do
        grep -qF "ensure-systemd-unit: ${UNIT} up to date" "${log}" \
            || fail "$(basename "${log}"): the unit was not reported up to date"
    done
    [ "$(sha256sum "${UNIT}")" = "${unit_sum}" ] || fail "a re-run changed the unit"
    [ "$(main_pid)" = "${pid_before}" ] || fail "a re-run restarted the service"
    echo "✅ re-running the postinst and the install is a no-op"

    echo "=== Restarting through ExecStartPre ==="
    [ ! -e "${GUARD}" ] || fail "${GUARD} unexpectedly exists; this test assumes a pre-B2 package"
    restart_and_check "missing"

    mkdir -p "$(dirname "${GUARD}")"
    printf '#!/bin/sh\necho ran >> /tmp/guard-ran\nexit 0\n' > "${GUARD}"
    chmod 755 "${GUARD}"
    restart_and_check "succeeding"
    grep -qx ran /tmp/guard-ran || fail "the guard did not run on start"

    printf '#!/bin/sh\nexit 1\n' > "${GUARD}"
    restart_and_check "failing"
    rm -f "${GUARD}"

    echo "✅ Edge-core deb upgrade under systemd works"
}

# ---------------------------------------------------------------------------
# cyberwave edge update, against a local apt repository
# ---------------------------------------------------------------------------

installed_version() {
    dpkg-query -W -f='${Version}' "${STABLE_PACKAGE}"
}

assert_version() {
    [ "$(installed_version)" = "$1" ] || fail "installed version is $(installed_version), expected $1 ($2)"
}

assert_log() {
    grep -qF "$2" "$1" || fail "$(basename "$1") lacks: $2"
}

# Run the CLI with its output in /tmp/<name>.log; returns the CLI's exit code.
cli() {
    local name="$1" rc
    shift
    echo "\$ cyberwave $*"
    # Wide enough that rich never wraps a line the checks grep for.
    COLUMNS=200 "${CLI}" "$@" >"/tmp/${name}.log" 2>&1 && rc=0 || rc=$?
    cat "/tmp/${name}.log"
    return "${rc}"
}

# Repack the staged deb as the stable package at version $1. The staged deb's
# Conflicts/Replaces name the stable package, so they are dropped.
repack_as_stable() {
    local work="/tmp/repack-$1"
    rm -rf "${work}"
    dpkg-deb -R "${DEB_IN_CONTAINER}" "${work}"
    sed -i \
        -e "s/^Package: .*/Package: ${STABLE_PACKAGE}/" \
        -e "s/^Version: .*/Version: $1/" \
        -e '/^Conflicts:/d' -e '/^Replaces:/d' \
        "${work}/DEBIAN/control"
    # Uncompressed: this deb only travels to the local repository.
    dpkg-deb -Znone --build "${work}" "${REPO_DIR}/${STABLE_PACKAGE}_$1.deb" >/dev/null
    rm -rf "${work}"
}

run_update_checks() {
    local staged_package unit_sum dropin_sum pid keyring sources

    echo "=== Installing the CLI from this checkout ==="
    cp -r "${SDK_SRC_IN_CONTAINER}" /tmp/sdk-src
    cp -r "${CLI_SRC_IN_CONTAINER}" /tmp/cli-src
    python3 -m venv /opt/cli-venv
    # pip retries a refused connection but not a read that times out mid
    # download, which PyPI does now and then; retry the whole install.
    local attempt
    for attempt in 1 2 3; do
        /opt/cli-venv/bin/pip install --quiet --disable-pip-version-check --timeout 60 \
            /tmp/sdk-src /tmp/cli-src && break
        [ "${attempt}" -lt 3 ] || fail "installing the CLI failed ${attempt} times"
        echo "pip install attempt ${attempt} failed; retrying in 10s"
        sleep 10
    done
    "${CLI}" --version

    staged_package="$(dpkg-deb -f "${DEB_IN_CONTAINER}" Package)"
    unit_sum="$(sha256sum "${UNIT}")"
    dropin_sum="$(sha256sum "${DROPIN}")"

    if [ "${staged_package}" = "${STABLE_PACKAGE}-pr" ]; then
        echo "=== A PR build needs --version ==="
        pid="$(main_pid)"
        if cli update-pr edge update; then
            fail "update of a PR build without --version succeeded"
        fi
        assert_log /tmp/update-pr.log "PR builds are not ordered against each other"
        [ "$(main_pid)" = "${pid}" ] || fail "a refused update restarted the service"
        echo "✅ a PR build without --version is refused"
    fi

    echo "=== Serving ${STABLE_PACKAGE} 1.0.0 and 1.0.1 from ${REPO_DIR} ==="
    # The unit is not a file of the package, so it and the service stay.
    dpkg --purge "${staged_package}"
    mkdir -p "${REPO_DIR}"
    repack_as_stable 1.0.0
    repack_as_stable 1.0.1
    (
        cd "${REPO_DIR}"
        apt-ftparchive packages . >Packages
        # Written outside the repository first, so it does not hash itself half-written.
        apt-ftparchive release . >/tmp/Release
        mv /tmp/Release Release
    )
    # The CLI adds its Buildkite source and key only when they are missing, so
    # pre-seeding both paths makes it install from the local repository.
    read -r keyring sources < <(/opt/cli-venv/bin/python -c '
from cyberwave_cli import core
print(*core._resolve_deb_registry_paths(core.EDGE_CORE_SPEC, "stable"))')
    mkdir -p "$(dirname "${keyring}")" "$(dirname "${sources}")"
    touch "${keyring}"
    echo "deb [trusted=yes] file:${REPO_DIR} ./" >"${sources}"
    # Keep apt off the Ubuntu mirrors: every package the test needs is installed.
    mv /etc/apt/sources.list /etc/apt/sources.list.ci-disabled
    apt-get update -qq
    dpkg -i "${REPO_DIR}/${STABLE_PACKAGE}_1.0.0.deb"
    assert_version 1.0.0 "start"
    systemctl is-active --quiet "${UNIT_NAME}" || fail "the service stopped while switching packages"

    echo "=== --check changes nothing ==="
    pid="$(main_pid)"
    cli check edge update --check || fail "--check failed"
    assert_log /tmp/check.log "Installed: 1.0.0"
    assert_log /tmp/check.log "Available: 1.0.1"
    assert_log /tmp/check.log "An update is available"
    assert_version 1.0.0 "--check"
    [ "$(main_pid)" = "${pid}" ] || fail "--check restarted the service"
    echo "✅ --check reports 1.0.1 and changes nothing"

    echo "=== Updating to the latest version ==="
    cli update edge update || fail "the update failed"
    assert_log /tmp/update.log "Updated ${STABLE_PACKAGE}: 1.0.0 → 1.0.1"
    assert_version 1.0.1 "update"
    wait_active
    [ "$(main_pid)" != "${pid}" ] || fail "the update did not restart the service"
    echo "✅ update installs 1.0.1 and restarts the service"

    echo "=== Updating again with nothing new ==="
    pid="$(main_pid)"
    cli update-noop edge update || fail "the second update failed"
    assert_log /tmp/update-noop.log "already at 1.0.1; nothing to restart"
    assert_version 1.0.1 "second update"
    [ "$(main_pid)" = "${pid}" ] || fail "an update with nothing new restarted the service"
    echo "✅ update with nothing new does not restart"

    echo "=== Rolling back with --version ==="
    cli rollback edge update --version 1.0.0 || fail "the rollback failed"
    assert_log /tmp/rollback.log "Updated ${STABLE_PACKAGE}: 1.0.1 → 1.0.0"
    assert_version 1.0.0 "rollback"
    wait_active
    [ "$(main_pid)" != "${pid}" ] || fail "the rollback did not restart the service"
    echo "✅ --version 1.0.0 downgrades and restarts the service"

    echo "=== A held package ==="
    apt-mark hold "${STABLE_PACKAGE}"
    pid="$(main_pid)"
    if cli update-held edge update; then
        fail "update of a held package without --force succeeded"
    fi
    assert_log /tmp/update-held.log "is held (apt-mark hold)"
    assert_version 1.0.0 "held"
    [ "$(main_pid)" = "${pid}" ] || fail "a refused update restarted the service"
    cli update-force edge update --force || fail "the forced update failed"
    assert_version 1.0.1 "--force"
    wait_active
    [ "$(main_pid)" != "${pid}" ] || fail "the forced update did not restart the service"
    apt-mark unhold "${STABLE_PACKAGE}"
    echo "✅ a held package is refused, and updated with --force"

    # The CLI must not re-template the unit (that would drop the config dir).
    [ "$(sha256sum "${UNIT}")" = "${unit_sum}" ] || fail "cyberwave edge update changed the unit"
    [ "$(sha256sum "${DROPIN}")" = "${dropin_sum}" ] || fail "cyberwave edge update changed the drop-in"
    echo "✅ cyberwave edge update works against a local apt repository"
}

if [ "${1:-}" = "--in-container" ]; then
    run_checks
    run_update_checks
    exit 0
fi

# ---------------------------------------------------------------------------
# On the runner
# ---------------------------------------------------------------------------

PACKAGE_FILE="${PACKAGE_FILE:-}"
[ -n "${PACKAGE_FILE}" ] || { echo "ERROR: PACKAGE_FILE is required." >&2; exit 1; }
[ -f "${PACKAGE_FILE}" ] || { echo "ERROR: package file not found: ${PACKAGE_FILE}" >&2; exit 1; }
PACKAGE_FILE="$(realpath "${PACKAGE_FILE}")"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI_SRC="${REPO_ROOT}/cyberwave-clis/cyberwave-python-cli"
SDK_SRC="${REPO_ROOT}/cyberwave-sdks/cyberwave-python"

IMAGE="${SYSTEMD_TEST_IMAGE:-cyberwave-ci/edge-core-systemd-test:ubuntu-22.04}"
# Self-hosted runners run several jobs side by side and keep containers
# between runs, so the name and label are unique per run and per process.
RUN_LABEL="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}"
CONTAINER="cw-edge-core-systemd-${RUN_LABEL}-$$"

cleanup() {
    docker rm -f -v "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "=== Building ${IMAGE} ==="
# No build context: the image is only systemd on top of ubuntu:22.04, cached
# by the runner's Docker between runs.
docker build --quiet -t "${IMAGE}" - <<'EOF'
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends systemd systemd-sysv dbus \
        apt-utils python3-venv \
    && rm -rf /var/lib/apt/lists/*
# Units that have nothing to manage in a container, or wait on hardware.
RUN systemctl mask \
        systemd-udevd.service systemd-udevd-kernel.socket systemd-udevd-control.socket \
        systemd-logind.service getty.target console-getty.service \
        systemd-remount-fs.service systemd-networkd-wait-online.service
STOPSIGNAL SIGRTMIN+3
CMD ["/sbin/init"]
EOF

# systemd as PID 1 needs a writable cgroup tree: a private cgroup namespace on
# cgroup v2 hosts, the host's (read-only) hierarchy on cgroup v1 hosts.
if [ "$(docker info --format '{{.CgroupVersion}}')" = "1" ]; then
    cgroup_args=(--cgroupns=host -v /sys/fs/cgroup:/sys/fs/cgroup:ro)
else
    cgroup_args=(--cgroupns=private)
fi

echo "=== Booting ${CONTAINER} ==="
docker run -d \
    --name "${CONTAINER}" \
    --label cyberwave.ci=edge-core-systemd-test \
    --label "cyberwave.ci.run=${RUN_LABEL}" \
    --privileged \
    "${cgroup_args[@]}" \
    --tmpfs /run \
    --tmpfs /run/lock \
    -v "${PACKAGE_FILE}:${DEB_IN_CONTAINER}:ro" \
    -v "${CLI_SRC}:${CLI_SRC_IN_CONTAINER}:ro" \
    -v "${SDK_SRC}:${SDK_SRC_IN_CONTAINER}:ro" \
    "${IMAGE}" >/dev/null

state=""
for _ in $(seq 1 60); do
    state="$(docker exec "${CONTAINER}" systemctl is-system-running 2>/dev/null || true)"
    case "${state}" in
        running | degraded) break ;;
    esac
    sleep 1
done
case "${state}" in
    running | degraded) echo "systemd is ${state}" ;;
    starting)
        # A unit unrelated to this test can keep the boot "starting"; the
        # manager already takes commands, so carry on and show what waits.
        echo "::warning::systemd is still starting after 60s; continuing"
        docker exec "${CONTAINER}" systemctl list-jobs --no-pager || true
        ;;
    *)
        echo "ERROR: systemd did not finish booting (state: '${state:-none}')" >&2
        docker logs "${CONTAINER}" >&2 || true
        exit 1
        ;;
esac

# Run from a file, not stdin, so no command in the checks can read the script.
docker cp "$0" "${CONTAINER}:/tmp/test_deb_upgrade_under_systemd.sh"
docker exec "${CONTAINER}" bash /tmp/test_deb_upgrade_under_systemd.sh --in-container
