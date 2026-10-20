"""Unit tests for cyberwave_edge_core.resource_monitor module.

Covers:
- MemoryInfo data class properties
- CpuTemperature data class properties
- ResourceSnapshot health/warning states
- read_memory_info parsing
- read_cpu_temperature reading
- SystemResourceMonitor check and throttling
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import pytest

from cyberwave_edge_core.resource_monitor import (
    CPU_TEMP_CRITICAL_C,
    CPU_TEMP_WARNING_C,
    MEMORY_CRITICAL_PERCENT,
    MEMORY_WARNING_PERCENT,
    CpuTemperature,
    MemoryInfo,
    ResourceSnapshot,
    SystemResourceMonitor,
    read_cpu_temperature,
    read_memory_info,
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
        mi = MemoryInfo(total_mb=4096, available_mb=600, used_percent=MEMORY_WARNING_PERCENT)
        assert mi.is_warning is True
        assert mi.is_critical is False

    def test_critical_at_threshold(self) -> None:
        mi = MemoryInfo(total_mb=4096, available_mb=300, used_percent=MEMORY_CRITICAL_PERCENT)
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

    def test_to_publish_dict_omits_absent_sources(self) -> None:
        """``None`` sources must be absent from the published dict.

        The frontend distinguishes "metric absent" from "metric is zero",
        so we omit the key entirely rather than emitting ``0`` or ``null``.
        """
        snap = ResourceSnapshot(timestamp=time.time(), memory=None, cpu_temp=None)
        assert snap.to_publish_dict() == {}

    def test_to_publish_dict_emits_memory_only(self) -> None:
        snap = ResourceSnapshot(
            timestamp=time.time(),
            memory=MemoryInfo(total_mb=4096.0, available_mb=1500.0, used_percent=63.4),
            cpu_temp=None,
        )
        out = snap.to_publish_dict()
        assert out == {
            "host_memory_percent": 63.4,
            "host_memory_available_mb": 1500.0,
        }

    def test_to_publish_dict_emits_temp_only(self) -> None:
        snap = ResourceSnapshot(
            timestamp=time.time(),
            memory=None,
            cpu_temp=CpuTemperature(celsius=58.7, source="thermal_zone0:cpu-thermal"),
        )
        assert snap.to_publish_dict() == {"cpu_temp_c": 58.7}

    def test_to_publish_dict_emits_both_when_available(self) -> None:
        snap = ResourceSnapshot(
            timestamp=time.time(),
            memory=MemoryInfo(total_mb=4096.0, available_mb=200.0, used_percent=95.0),
            cpu_temp=CpuTemperature(celsius=88.0, source="thermal_zone0:cpu-thermal"),
        )
        out = snap.to_publish_dict()
        assert out == {
            "host_memory_percent": 95.0,
            "host_memory_available_mb": 200.0,
            "cpu_temp_c": 88.0,
        }

    def test_to_publish_dict_does_not_leak_threshold_flags(self) -> None:
        """Severity flags stay client-side so we have a single source of truth."""
        snap = ResourceSnapshot(
            timestamp=time.time(),
            memory=MemoryInfo(total_mb=4096.0, available_mb=200.0, used_percent=95.0),
            cpu_temp=CpuTemperature(celsius=88.0, source="thermal_zone0"),
        )
        out = snap.to_publish_dict()
        # is_warning / is_critical must NOT be serialised — consumers re-evaluate
        # the shared thresholds locally.
        assert "is_warning" not in out
        assert "is_critical" not in out


# ---------------------------------------------------------------------------
# read_memory_info
# ---------------------------------------------------------------------------


class TestReadMemoryInfo:
    """Tests for the threshold-aware memory wrapper.

    ``/proc/meminfo`` parsing semantics (MemAvailable fallback, malformed
    input, etc.) are covered exhaustively in the SDK's
    ``test_host_metrics.py``.  The cases here verify that the edge-core
    wrapper:

    1. forwards ``None`` from the SDK reader,
    2. preserves field values without re-rounding, and
    3. attaches the threshold properties.
    """

    def test_returns_none_when_sdk_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.resource_monitor as rm_mod

        monkeypatch.setattr(rm_mod, "read_host_memory", lambda: None)
        assert read_memory_info() is None

    def test_wraps_sdk_result_with_thresholds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cyberwave.edge.host_metrics import HostMemoryInfo

        import cyberwave_edge_core.resource_monitor as rm_mod

        sdk_result = HostMemoryInfo(total_mb=4096.0, available_mb=200.0, used_percent=95.0)
        monkeypatch.setattr(rm_mod, "read_host_memory", lambda: sdk_result)

        result = read_memory_info()
        assert result is not None
        assert result.total_mb == 4096.0
        assert result.available_mb == 200.0
        assert result.used_percent == 95.0
        assert result.is_warning is True
        assert result.is_critical is True

    def test_returns_none_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert read_memory_info() is None

    def test_parses_meminfo(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

        original_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == "/proc/meminfo":
                return original_open(str(meminfo_path), *args, **kwargs)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        result = read_memory_info()
        assert result is not None
        assert result.total_mb == pytest.approx(3906292 / 1024, abs=1)
        assert result.available_mb == pytest.approx(1024000 / 1024, abs=1)
        assert result.used_percent > 0


# ---------------------------------------------------------------------------
# read_cpu_temperature
# ---------------------------------------------------------------------------


class TestReadCpuTemperature:
    """Tests for the threshold-aware wrapper around the SDK reader.

    Zone-discovery semantics (CPU-typed zone preference, multi-zone max,
    fallback to all zones) are covered exhaustively in the SDK's
    ``test_host_metrics.py`` — here we only verify that the wrapper
    translates the SDK's result into the edge-core ``CpuTemperature``
    type (with thresholds) and forwards ``None``.
    """

    def test_returns_none_when_sdk_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.resource_monitor as rm_mod

        monkeypatch.setattr(rm_mod, "read_host_cpu_temperature", lambda: None)
        assert read_cpu_temperature() is None

    def test_wraps_sdk_result_with_thresholds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cyberwave.edge.host_metrics import HostCpuTemperature

        import cyberwave_edge_core.resource_monitor as rm_mod

        sdk_result = HostCpuTemperature(celsius=83.0, source="thermal_zone1:coretemp")
        monkeypatch.setattr(rm_mod, "read_host_cpu_temperature", lambda: sdk_result)

        result = read_cpu_temperature()
        assert result is not None
        assert result.celsius == pytest.approx(83.0)
        assert result.source == "thermal_zone1:coretemp"
        # Threshold properties are still applied by the wrapper.
        assert result.is_warning is True
        assert result.is_critical is True


# ---------------------------------------------------------------------------
# SystemResourceMonitor
# ---------------------------------------------------------------------------


class TestSystemResourceMonitor:
    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.resource_monitor as rm_mod

        monkeypatch.setattr(rm_mod, "read_disk_percent", lambda: None)

    def test_check_returns_cached_snapshot_on_throttle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When called within the throttle window, ``check()`` returns the
        cached ``_last_snapshot`` and does *not* re-read /proc/meminfo or
        the thermal sysfs."""
        import cyberwave_edge_core.resource_monitor as rm_mod

        read_count = [0]
        temp_count = [0]

        def fake_read_mem() -> None:
            read_count[0] += 1
            return None

        def fake_read_temp() -> None:
            temp_count[0] += 1
            return None

        monkeypatch.setattr(rm_mod, "read_memory_info", fake_read_mem)
        monkeypatch.setattr(rm_mod, "read_cpu_temperature", fake_read_temp)

        monitor = SystemResourceMonitor()
        # Pretend we just checked, so the next call is throttled.
        monitor._last_check = time.monotonic()
        sentinel = ResourceSnapshot(timestamp=0.0, memory=None, cpu_temp=None)
        monitor._last_snapshot = sentinel

        result = monitor.check()
        assert result is sentinel
        assert read_count[0] == 0
        assert temp_count[0] == 0

    def test_check_returns_snapshot_after_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.resource_monitor as rm_mod

        monkeypatch.setattr(rm_mod, "MONITOR_CHECK_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(rm_mod, "read_memory_info", lambda: None)
        monkeypatch.setattr(rm_mod, "read_cpu_temperature", lambda: None)

        monitor = SystemResourceMonitor()
        result = monitor.check()
        assert result is not None
        assert result.memory is None
        assert result.cpu_temp is None

    def test_consecutive_critical_tracking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.resource_monitor as rm_mod

        monkeypatch.setattr(rm_mod, "MONITOR_CHECK_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(
            rm_mod,
            "read_memory_info",
            lambda: MemoryInfo(total_mb=4096, available_mb=200, used_percent=95.0),
        )
        monkeypatch.setattr(rm_mod, "read_cpu_temperature", lambda: None)

        monitor = SystemResourceMonitor()
        monitor.check()
        assert monitor.consecutive_critical_count == 1
        monitor._last_check = 0.0
        monitor.check()
        assert monitor.consecutive_critical_count == 2

    def test_consecutive_critical_resets_on_healthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.resource_monitor as rm_mod

        monkeypatch.setattr(rm_mod, "MONITOR_CHECK_INTERVAL_SECONDS", 0.0)

        monitor = SystemResourceMonitor()
        monitor._consecutive_critical = 5

        monkeypatch.setattr(
            rm_mod,
            "read_memory_info",
            lambda: MemoryInfo(total_mb=4096, available_mb=2048, used_percent=50.0),
        )
        monkeypatch.setattr(rm_mod, "read_cpu_temperature", lambda: None)

        monitor.check()
        assert monitor.consecutive_critical_count == 0

    def test_check_reports_cpu_and_disk_percent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.resource_monitor as rm_mod

        monkeypatch.setattr(rm_mod, "MONITOR_CHECK_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(rm_mod, "read_memory_info", lambda: None)
        monkeypatch.setattr(rm_mod, "read_cpu_temperature", lambda: None)
        monkeypatch.setattr(rm_mod, "read_disk_percent", lambda: 96.5)
        times = iter([(100, 1000), (150, 2000), (150, 2000)])
        monkeypatch.setattr(rm_mod, "read_cpu_times", lambda: next(times))

        monitor = SystemResourceMonitor()
        first = monitor.check()
        assert first is not None and first.cpu_percent is None
        assert first.disk_percent == 96.5

        second = monitor.check()
        assert second is not None and second.cpu_percent == pytest.approx(95.0)
        assert second.to_publish_dict() == {"host_cpu_percent": 95.0, "host_disk_percent": 96.5}

        third = monitor.check()
        assert third is not None and third.cpu_percent is None


class TestHostGauges:
    def test_read_cpu_times_parses_the_aggregate_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io

        real_open = open

        def fake_open(path, *args, **kwargs):
            if path == "/proc/stat":
                return io.StringIO("cpu  10 2 30 400 50 6 7 8 9 10\ncpu0 1 1 1 1 1 1 1 1\n")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        from cyberwave_edge_core.resource_monitor import read_cpu_times

        assert read_cpu_times() == (450, 513)

    def test_read_cpu_times_is_none_without_proc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_open(*_args, **_kwargs):
            raise FileNotFoundError("/proc/stat")

        monkeypatch.setattr("builtins.open", fake_open)
        from cyberwave_edge_core.resource_monitor import read_cpu_times

        assert read_cpu_times() is None

    def test_read_disk_percent_measures_the_docker_data_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import shutil
        from collections import namedtuple

        import cyberwave_edge_core.docker_helpers as dh
        from cyberwave_edge_core.resource_monitor import read_disk_percent

        usage = namedtuple("usage", "total used free")
        seen: list[str] = []
        monkeypatch.setattr(dh, "docker_data_root", lambda: str(tmp_path))
        monkeypatch.setattr(
            shutil, "disk_usage", lambda path: seen.append(path) or usage(200, 190, 10)
        )

        assert read_disk_percent() == pytest.approx(95.0)
        assert seen == [str(tmp_path)]


class TestSaturationThreshold:
    @pytest.mark.parametrize(
        ("raw", "expected", "warned"),
        [
            (None, 95.0, False),
            ("80", 80.0, False),
            ("100", 100.0, False),
            ("high", 95.0, True),
            ("150", 95.0, True),
            ("0", 95.0, True),
            ("nan", 95.0, True),
        ],
    )
    def test_env_threshold(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        raw: str | None,
        expected: float,
        warned: bool,
    ) -> None:
        from cyberwave_edge_core.resource_monitor import (
            RESOURCE_SATURATION_PERCENT_ENV,
            resource_saturation_percent,
        )

        if raw is None:
            monkeypatch.delenv(RESOURCE_SATURATION_PERCENT_ENV, raising=False)
        else:
            monkeypatch.setenv(RESOURCE_SATURATION_PERCENT_ENV, raw)

        with caplog.at_level("WARNING"):
            assert resource_saturation_percent() == expected
        assert (RESOURCE_SATURATION_PERCENT_ENV in caplog.text) is warned

    def test_monitor_clears_five_points_below(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cyberwave_edge_core.resource_monitor import RESOURCE_SATURATION_PERCENT_ENV

        monkeypatch.setenv(RESOURCE_SATURATION_PERCENT_ENV, "85")
        monitor = SystemResourceMonitor()

        assert (monitor.saturation_percent, monitor.saturation_clear_percent) == (85.0, 80.0)
