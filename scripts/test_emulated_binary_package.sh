#!/bin/sh

set -eu

DEVICE_NAME="${DEVICE_NAME:-unknown-device}"
DEVICE_DISTRO="${DEVICE_DISTRO:-unknown-distro}"
PACKAGE_NAME="cyberwave-edge-core"

echo "=== Emulated package test: ${DEVICE_NAME} (${DEVICE_DISTRO}) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    bash \
    binutils \
    build-essential \
    ca-certificates \
    dpkg-dev \
    file \
    libffi-dev \
    libssl-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv

if ! command -v pip >/dev/null 2>&1; then
    ln -sf "$(command -v pip3)" /usr/local/bin/pip
fi

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[build]"

chmod +x ./build.sh
./build.sh
./dist/cyberwave-edge-core --version || ./dist/cyberwave-edge-core --help

VERSION="$(python3 - <<'PY'
import re
from pathlib import Path

text = Path("pyproject.toml").read_text(encoding="utf-8")
match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
if not match:
    raise SystemExit("Unable to find version in pyproject.toml")
print(match.group(1))
PY
)"

PKG_DIR="${PACKAGE_NAME}_${VERSION}_${DEVICE_DISTRO}"
rm -rf "${PKG_DIR}" "${PKG_DIR}.deb"
mkdir -p "${PKG_DIR}/DEBIAN" "${PKG_DIR}/usr/bin"

cp ./dist/cyberwave-edge-core "${PKG_DIR}/usr/bin/"
chmod 755 "${PKG_DIR}/usr/bin/cyberwave-edge-core"

cat > "${PKG_DIR}/DEBIAN/control" <<EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: arm64
Maintainer: Cyberwave <info@cyberwave.com>
Description: cyberwave-edge-core emulated install test package
EOF

dpkg-deb --build "${PKG_DIR}"
dpkg -i "${PKG_DIR}.deb"

echo "=== Verifying installed binary ==="
/usr/bin/cyberwave-edge-core --help
echo "✅ ${PACKAGE_NAME} installation works on ${DEVICE_NAME} (${DEVICE_DISTRO})"
