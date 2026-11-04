#!/bin/bash
# Build script for creating standalone Cyberwave CLI binary

set -e

echo "=== Building Cyberwave Edge Core ==="

# Ensure we're in the project directory
cd "$(dirname "$0")"

# Check if pyinstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "Installing build dependencies..."
    pip install -e ".[build]"
fi

# Clean previous builds
rm -rf build dist *.spec __pyinstaller_entry.py __pyinstaller_rthook_ca.py

# Create a wrapper entry point that uses absolute imports
cat > __pyinstaller_entry.py << 'EOF'
"""PyInstaller entry point for Cyberwave CLI."""
from cyberwave_edge_core.main import main

if __name__ == "__main__":
    main()
EOF

# Runs before the entry point: the bundled OpenSSL looks for CAs where the build
# image keeps them, which the host may not (see ca_bundle.py).
cat > __pyinstaller_rthook_ca.py << 'EOF'
from cyberwave_edge_core.ca_bundle import ensure_ca_bundle

ensure_ca_bundle()
EOF

# Build standalone binary
echo "Building standalone binary..."
pyinstaller \
    --onefile \
    --name cyberwave-edge-core \
    --runtime-hook __pyinstaller_rthook_ca.py \
    --hidden-import cyberwave_edge_core \
    --hidden-import cyberwave_edge_core._build_version \
    --hidden-import cyberwave_edge_core.ca_bundle \
    --hidden-import click \
    --hidden-import rich \
    --hidden-import httpx \
    --collect-submodules cyberwave \
    --collect-submodules rich._unicode_data \
    --exclude-module matplotlib \
    __pyinstaller_entry.py

# Clean up the temporary entry point and runtime hook
rm -f __pyinstaller_entry.py __pyinstaller_rthook_ca.py

echo ""
echo "=== Build complete ==="
echo "Binary: dist/cyberwave-edge-core"
echo ""
echo "Test with: ./dist/cyberwave-edge-core --help"
