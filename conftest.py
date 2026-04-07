"""Pytest configuration for cyberwave-edge-core tests.

1. Sets CYBERWAVE_EDGE_CONFIG_DIR to a temporary directory so the module-level
   bootstrap in startup.py does not attempt to read /etc/cyberwave during
   test collection.

2. Registers a minimal cyberwave SDK stub in sys.modules so startup.py can be
   imported without the real SDK being installed in the test environment.
"""
from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("CYBERWAVE_EDGE_CONFIG_DIR", "/tmp/cyberwave-test")

_fake_cw = types.ModuleType("cyberwave")
_fake_cw.__path__ = []  # type: ignore[attr-defined]
_fake_cw.Cyberwave = object  # type: ignore[attr-defined]
_fake_fp = types.ModuleType("cyberwave.fingerprint")
_fake_fp.generate_fingerprint = lambda: "test-fingerprint"  # type: ignore[attr-defined]
_fake_cw.fingerprint = _fake_fp  # type: ignore[attr-defined]

_fake_edge = types.ModuleType("cyberwave.edge")
_fake_edge.__path__ = []  # type: ignore[attr-defined]
_fake_edge_platform = types.ModuleType("cyberwave.edge.platform")
_fake_edge_platform.USBIP_LAUNCHD_LABEL = "com.cyberwave.usbip"  # type: ignore[attr-defined]
_fake_edge_platform.USBIP_PORT = 3240  # type: ignore[attr-defined]
_fake_edge_platform.is_port_listening = lambda port, host="127.0.0.1", timeout=1: False  # type: ignore[attr-defined]
_fake_edge_platform.is_usbip_server_running = lambda: False  # type: ignore[attr-defined]
_fake_edge.platform = _fake_edge_platform  # type: ignore[attr-defined]
_fake_cw.edge = _fake_edge  # type: ignore[attr-defined]

sys.modules.setdefault("cyberwave", _fake_cw)
sys.modules.setdefault("cyberwave.fingerprint", _fake_fp)
sys.modules.setdefault("cyberwave.edge", _fake_edge)
sys.modules.setdefault("cyberwave.edge.platform", _fake_edge_platform)
