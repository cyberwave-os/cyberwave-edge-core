"""Tests for auto-detected worker memory limits on resource-constrained hosts.

Covers the ``_auto_detect_worker_memory_limit()`` function in ``startup.py``
and its integration with ``load_worker_resource_limits()``.
"""

from __future__ import annotations

import pytest

import cyberwave_edge_core.startup as startup
from cyberwave_edge_core.resource_monitor import MemoryInfo


class TestAutoDetectWorkerMemoryLimit:
    def test_returns_none_when_no_memory_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.resource_monitor._read_memory_info", lambda: None
        )
        monkeypatch.setattr(startup, "get_runtime_env_var", lambda *a, **kw: None)
        result = startup._auto_detect_worker_memory_limit()
        assert result is None

    def test_returns_none_for_large_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.resource_monitor._read_memory_info",
            lambda: MemoryInfo(total_mb=16384, available_mb=12000, used_percent=27.0),
        )
        monkeypatch.setattr(startup, "get_runtime_env_var", lambda *a, **kw: None)
        result = startup._auto_detect_worker_memory_limit()
        assert result is None

    def test_returns_limits_for_pi4_sized_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.resource_monitor._read_memory_info",
            lambda: MemoryInfo(total_mb=3800, available_mb=2000, used_percent=47.0),
        )
        monkeypatch.setattr(startup, "get_runtime_env_var", lambda *a, **kw: None)
        result = startup._auto_detect_worker_memory_limit()
        assert result is not None
        assert result.memory_mb is not None
        assert 2500 < result.memory_mb < 3800
        assert result.cpu_quota_percent is None

    def test_opt_out_via_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.resource_monitor._read_memory_info",
            lambda: MemoryInfo(total_mb=3800, available_mb=2000, used_percent=47.0),
        )

        def get_env(name, default=None):
            if name == "CYBERWAVE_WORKER_AUTO_MEMORY_LIMIT":
                return "false"
            return default

        monkeypatch.setattr(startup, "get_runtime_env_var", get_env)
        result = startup._auto_detect_worker_memory_limit()
        assert result is None

    def test_load_worker_resource_limits_uses_auto_when_no_explicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.resource_monitor._read_memory_info",
            lambda: MemoryInfo(total_mb=2048, available_mb=1024, used_percent=50.0),
        )
        monkeypatch.setattr(startup, "get_runtime_env_var", lambda *a, **kw: None)
        result = startup.load_worker_resource_limits()
        assert result is not None
        assert result.memory_mb is not None

    def test_load_worker_resource_limits_explicit_overrides_auto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def get_env(name, default=None):
            if name == "CYBERWAVE_WORKER_MEMORY_MB":
                return "1024"
            return default

        monkeypatch.setattr(startup, "get_runtime_env_var", get_env)
        result = startup.load_worker_resource_limits()
        assert result is not None
        assert result.memory_mb == 1024
