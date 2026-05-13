"""Unit tests for cyberwave_edge_core.resource_monitor module.

Covers:
- MemoryInfo data class properties
- CpuTemperature data class properties
- ResourceSnapshot health/warning states
- _read_memory_info parsing
- _read_cpu_temperature reading
- SystemResourceMonitor check and throttling
- SystemResourceMonitor.suggest_worker_memory_limit_mb
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cyberwave_edge_core.resource_monitor import (
    MEMORY_CRITICAL_PERCENT,
    MEMORY_WARNING_PERCENT,
    CPU_TEMP_CRITICAL_C,
    CPU_TEMP_WARNING_C,
    CpuTemperature,
    MemoryInfo,
    ResourceSnapshot,
    SystemResourceMonitor,
    _read_cpu_temperature,
    _read_memory_info,
)


# ---------------------------------------------------------------------------
# MemoryInfo
# ---------------------------------------------------------------------------


class TestMemoryInfo:
    def test_not_warning_under_threshold(self) -> None:
        mi = MemoryInfo(total_mb=4096, available_mb=2048, used_percent=50.0)
        assert mi.is_warning is False
        assert mi.is_critical is False

    def test_warning_at_threshold(self) -> None:
        mi = MemoryInfo(
            total_mb=4096, available_mb=600, used_percent=MEMORY_WARNING_PERCENT
        )
        assert mi.is_warning is True
        assert mi.is_critical is False

    def test_critical_at_threshold(self) -> None:
        mi = MemoryInfo(
            total_mb=4096, available_mb=300, used_percent=MEMORY_CRITICAL_PERCENT
        )
        assert mi.is_warning is True
        assert mi.is_critical is True


# ---------------------------------------------------------------------------
# CpuTemperature
# ---------------------------------------------------------------------------


class TestCpuTemperature:
    def test_normal_temp(self) -> None:
        ct = CpuTemperature(celsius=55.0, source="thermal_zone0")
        assert ct.is_warning is False
        assert ct.is_critical is False

    def test_warning_temp(self) -> None:
        ct = CpuTemperature(celsius=CPU_TEMP_WARNING_C, source="thermal_zone0")
        assert ct.is_warning is True
        assert ct.is_critical is False

    def test_critical_temp(self) -> None:
        ct = CpuTemperature(celsius=CPU_TEMP_CRITICAL_C, source="thermal_zone0")
        assert ct.is_warning is True
        assert ct.is_critical is True


# ---------------------------------------------------------------------------
# ResourceSnapshot
# ---------------------------------------------------------------------------


class TestResourceSnapshot:
    def test_healthy_when_no_data(self) -> None:
        snap = ResourceSnapshot(timestamp=time.time(), memory=None, cpu_temp=None)
        assert snap.is_healthy is True
        assert snap.has_warnings is False

    def test_healthy_with_normal_values(self) -> None:
        snap = ResourceSnapshot(
            timestamp=time.time(),
            memory=MemoryInfo(total_mb=4096, available_mb=2048, used_percent=50.0),
            cpu_temp=CpuTemperature(celsius=45.0, source="test"),
        )
        assert snap.is_healthy is True
        assert snap.has_warnings is False

    def test_unhealthy_with_critical_memory(self) -> None:
        snap = ResourceSnapshot(
            timestamp=time.time(),
            memory=MemoryInfo(total_mb=4096, available_mb=200, used_percent=95.0),
            cpu_temp=None,
        )
        assert snap.is_healthy is False

    def test_has_warnings_with_elevated_temp(self) -> None:
        snap = ResourceSnapshot(
            timestamp=time.time(),
            memory=None,
            cpu_temp=CpuTemperature(celsius=78.0, source="test"),
        )
        assert snap.is_healthy is True
        assert snap.has_warnings is True


# ---------------------------------------------------------------------------
# _read_memory_info
# ---------------------------------------------------------------------------


class TestReadMemoryInfo:
    def test_returns_none_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert _read_memory_info() is None

    def test_parses_meminfo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        meminfo_content = (
            "MemTotal:        3906292 kB\n"
            "MemFree:          123456 kB\n"
            "MemAvailable:    1024000 kB\n"
            "Buffers:          102400 kB\n"
            "Cached:           512000 kB\n"
        )
        meminfo_path = tmp_path / "meminfo"
        meminfo_path.write_text(meminfo_content)

        import cyberwave_edge_core.resource_monitor as rm_mod

        original_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == "/proc/meminfo":
                return original_open(str(meminfo_path), *args, **kwargs)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        result = _read_memory_info()
        assert result is not None
        assert result.total_mb == pytest.approx(3906292 / 1024, abs=1)
        assert result.available_mb == pytest.approx(1024000 / 1024, abs=1)
        assert result.used_percent > 0


# ---------------------------------------------------------------------------
# _read_cpu_temperature
# ---------------------------------------------------------------------------


class TestReadCpuTemperature:
    def test_returns_none_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert _read_cpu_temperature() is None


# ---------------------------------------------------------------------------
# SystemResourceMonitor
# ---------------------------------------------------------------------------


class TestSystemResourceMonitor:
    def test_check_returns_none_on_throttle(self) -> None:
        monitor = SystemResourceMonitor()
        monitor._last_check = time.monotonic()
        result = monitor.check()
        assert result is monitor._last_snapshot

    def test_check_returns_snapshot_after_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cyberwave_edge_core.resource_monitor as rm_mod

        monkeypatch.setattr(rm_mod, "MONITOR_CHECK_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(rm_mod, "_read_memory_info", lambda: None)
        monkeypatch.setattr(rm_mod, "_read_cpu_temperature", lambda: None)

        monitor = SystemResourceMonitor()
        result = monitor.check()
        assert result is not None
        assert result.memory is None
        assert result.cpu_temp is None

    def test_suggest_worker_memory_limit_for_small_device(self) -> None:
        monitor = SystemResourceMonitor()
        monitor._last_snapshot = ResourceSnapshot(
            timestamp=time.time(),
            memory=MemoryInfo(total_mb=3800, available_mb=2000, used_percent=47.0),
            cpu_temp=None,
        )

        limit = monitor.suggest_worker_memory_limit_mb()
        assert limit is not None
        assert 2500 < limit < 3800

    def test_suggest_worker_memory_limit_returns_none_for_large_device(self) -> None:
        monitor = SystemResourceMonitor()
        monitor._last_snapshot = ResourceSnapshot(
            timestamp=time.time(),
            memory=MemoryInfo(total_mb=16384, available_mb=12000, used_percent=27.0),
            cpu_temp=None,
        )

        limit = monitor.suggest_worker_memory_limit_mb()
        assert limit is None

    def test_consecutive_critical_tracking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cyberwave_edge_core.resource_monitor as rm_mod

        monkeypatch.setattr(rm_mod, "MONITOR_CHECK_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(
            rm_mod,
            "_read_memory_info",
            lambda: MemoryInfo(total_mb=4096, available_mb=200, used_percent=95.0),
        )
        monkeypatch.setattr(rm_mod, "_read_cpu_temperature", lambda: None)

        monitor = SystemResourceMonitor()
        monitor.check()
        assert monitor.consecutive_critical_count == 1
        monitor._last_check = 0.0
        monitor.check()
        assert monitor.consecutive_critical_count == 2

    def test_consecutive_critical_resets_on_healthy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cyberwave_edge_core.resource_monitor as rm_mod

        monkeypatch.setattr(rm_mod, "MONITOR_CHECK_INTERVAL_SECONDS", 0.0)

        monitor = SystemResourceMonitor()
        monitor._consecutive_critical = 5

        monkeypatch.setattr(
            rm_mod,
            "_read_memory_info",
            lambda: MemoryInfo(total_mb=4096, available_mb=2048, used_percent=50.0),
        )
        monkeypatch.setattr(rm_mod, "_read_cpu_temperature", lambda: None)

        monitor.check()
        assert monitor.consecutive_critical_count == 0
