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
sys.modules.setdefault("cyberwave", _fake_cw)
sys.modules.setdefault("cyberwave.fingerprint", _fake_fp)
