"""Pytest configuration for cyberwave-edge-core tests.

1. Sets CYBERWAVE_EDGE_CONFIG_DIR to a temporary directory so the module-level
   bootstrap in startup.py does not attempt to read /etc/cyberwave during
   test collection.

2. Ensures the symbols edge-core imports from the ``cyberwave`` SDK are
   resolvable.  Prefers the real SDK when it is installed (typical
   developer environment); otherwise installs minimal stubs so the test
   suite can run in CI/sandbox environments that intentionally exclude
   the SDK dependency.
"""

from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("CYBERWAVE_EDGE_CONFIG_DIR", "/tmp/cyberwave-test")


def _ensure_real_or_stub_cyberwave_sdk() -> None:
    """Try to import the real SDK first; fall back to stubs otherwise."""
    try:
        import cyberwave  # noqa: F401
        import cyberwave.edge  # noqa: F401
        import cyberwave.edge.host_metrics  # noqa: F401
        import cyberwave.edge.platform  # noqa: F401
        import cyberwave.fingerprint  # noqa: F401
    except ImportError:
        pass
    else:
        return

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

    _fake_host_metrics = types.ModuleType("cyberwave.edge.host_metrics")
    _fake_host_metrics.read_host_memory = lambda: None  # type: ignore[attr-defined]
    _fake_host_metrics.read_host_cpu_temperature = lambda: None  # type: ignore[attr-defined]
    _fake_edge.host_metrics = _fake_host_metrics  # type: ignore[attr-defined]

    _fake_cw.edge = _fake_edge  # type: ignore[attr-defined]

    sys.modules.setdefault("cyberwave", _fake_cw)
    sys.modules.setdefault("cyberwave.fingerprint", _fake_fp)
    sys.modules.setdefault("cyberwave.edge", _fake_edge)
    sys.modules.setdefault("cyberwave.edge.platform", _fake_edge_platform)
    sys.modules.setdefault("cyberwave.edge.host_metrics", _fake_host_metrics)


_ensure_real_or_stub_cyberwave_sdk()
