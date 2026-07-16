#!/bin/sh

set -eu

DEVICE_NAME="${DEVICE_NAME:-unknown-device}"
DEVICE_DISTRO="${DEVICE_DISTRO:-unknown-distro}"
PACKAGE_FILE="${PACKAGE_FILE:-}"

if [ -z "${PACKAGE_FILE}" ]; then
    echo "ERROR: PACKAGE_FILE environment variable is required."
    exit 1
fi

if [ ! -f "${PACKAGE_FILE}" ]; then
    echo "ERROR: package file not found: ${PACKAGE_FILE}"
    exit 1
fi

echo "=== Emulated package install test: ${DEVICE_NAME} (${DEVICE_DISTRO}) ==="
echo "Using prebuilt package: ${PACKAGE_FILE}"

export DEBIAN_FRONTEND=noninteractive

# Install ca-certificates, falling back to the Azure region mirror when the
# default Ubuntu mirror is unreachable. GitHub-hosted runners are Azure VMs, so
# azure.{ports,archive}.ubuntu.com stays reachable even when the canonical
# ports.ubuntu.com / archive.ubuntu.com are slow or blocked. Note that
# `apt-get update` only emits warnings (exit 0) when indexes fail to download,
# so we key the fallback off the install step actually failing.
install_ca_certificates() {
    apt-get update -qq || true
    if apt-get install -y -qq --no-install-recommends ca-certificates; then
        return 0
    fi

    echo "WARNING: apt install failed against the default mirror; retrying via the Azure mirror" >&2
    sed -i \
        -e 's|https\?://ports.ubuntu.com/ubuntu-ports|http://azure.ports.ubuntu.com/ubuntu-ports|g' \
        -e 's|https\?://archive.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
        -e 's|https\?://security.ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' \
        /etc/apt/sources.list 2>/dev/null || true

    apt-get update -qq
    apt-get install -y -qq --no-install-recommends ca-certificates
}

install_ca_certificates

dpkg -i "${PACKAGE_FILE}"

echo "=== Verifying installed binary ==="
/usr/bin/cyberwave-edge-core --help
/usr/bin/cyberwave-edge-core __selfcheck_sdk
echo "✅ Package install works on ${DEVICE_NAME} (${DEVICE_DISTRO})"
