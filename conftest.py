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

import hashlib
import os
import sys
import types

os.environ.setdefault("CYBERWAVE_EDGE_CONFIG_DIR", "/tmp/cyberwave-test")


def _ensure_real_or_stub_cyberwave_sdk() -> None:
    """Try to import the real SDK first; fall back to stubs otherwise."""
    try:
        import cyberwave  # noqa: F401
        import cyberwave.downloads  # noqa: F401  (0.7.5+: the shared model download step)
        import cyberwave.edge  # noqa: F401
        import cyberwave.edge.host_metrics  # noqa: F401
        import cyberwave.edge.platform  # noqa: F401
        import cyberwave.fingerprint  # noqa: F401
        import cyberwave.mqtt_identity  # noqa: F401
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

    _fake_mqtt_identity = types.ModuleType("cyberwave.mqtt_identity")
    _fake_mqtt_identity.mqtt_username_for_token = (  # type: ignore[attr-defined]
        lambda token: "cwh_" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    )
    _fake_cw.mqtt_identity = _fake_mqtt_identity  # type: ignore[attr-defined]

    # ``ModelManager`` calls the shared download step (and its partial-download test,
    # when it is built). Only what those callers need; a test that wants a real download
    # skips itself when ``IS_STUB`` is set.
    class _DownloadError(RuntimeError):
        def __init__(self, message: str = "", *, status: int | None = None) -> None:
            super().__init__(message)
            self.status = status

    class _DownloadAuthError(_DownloadError):
        pass

    def _download_file_stub(*_args: object, **_kwargs: object) -> str:
        raise NotImplementedError("the real cyberwave SDK is not installed")

    _fake_downloads = types.ModuleType("cyberwave.downloads")
    _fake_downloads.IS_STUB = True  # type: ignore[attr-defined]
    _fake_downloads.DownloadError = _DownloadError  # type: ignore[attr-defined]
    _fake_downloads.DownloadAuthError = _DownloadAuthError  # type: ignore[attr-defined]
    _fake_downloads.download_file = _download_file_stub  # type: ignore[attr-defined]
    _fake_downloads.is_partial_download = (  # type: ignore[attr-defined]
        lambda name: name.startswith(".dl_")
    )
    _fake_cw.downloads = _fake_downloads  # type: ignore[attr-defined]

    sys.modules.setdefault("cyberwave", _fake_cw)
    sys.modules.setdefault("cyberwave.fingerprint", _fake_fp)
    sys.modules.setdefault("cyberwave.edge", _fake_edge)
    sys.modules.setdefault("cyberwave.edge.platform", _fake_edge_platform)
    sys.modules.setdefault("cyberwave.edge.host_metrics", _fake_host_metrics)
    sys.modules.setdefault("cyberwave.mqtt_identity", _fake_mqtt_identity)
    sys.modules.setdefault("cyberwave.downloads", _fake_downloads)


_ensure_real_or_stub_cyberwave_sdk()
