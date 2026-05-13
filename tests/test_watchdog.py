"""Unit tests for cyberwave_edge_core.watchdog module.

Covers:
- SystemdWatchdog: ping, notify_ready, notify_stopping, recommended interval
- HardwareWatchdog: open, ping, close (mocked /dev/watchdog)
- ProcessWatchdog: start, ping, stop combine both layers
- protect_edge_core_oom: OOM score adjustment
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest

from cyberwave_edge_core.watchdog import (
    HardwareWatchdog,
    ProcessWatchdog,
    SystemdWatchdog,
    protect_edge_core_oom,
)

# ---------------------------------------------------------------------------
# SystemdWatchdog
# ---------------------------------------------------------------------------


class TestSystemdWatchdog:
    def test_disabled_when_no_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", None)
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)

        sd = SystemdWatchdog()
        assert sd.enabled is False
        assert sd.recommended_interval_seconds == 0.0

    def test_enabled_when_watchdog_usec_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", 30_000_000)
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")

        sd = SystemdWatchdog()
        assert sd.enabled is True
        assert sd.recommended_interval_seconds == 15.0

    def test_ping_sends_watchdog_notification(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", 10_000_000)
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        sd = SystemdWatchdog()
        sd.ping()
        assert "WATCHDOG=1" in sent

    def test_notify_ready_sends_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        sd = SystemdWatchdog()
        sd.notify_ready()
        assert "READY=1" in sent

    def test_notify_stopping_sends_stopping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        sd = SystemdWatchdog()
        sd.notify_stopping()
        assert "STOPPING=1" in sent

    def test_sd_notify_skipped_when_pid_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_sd_notify`` must not send when this PID is not WATCHDOG_PID."""
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[bytes] = []

        class FakeSocket:
            def __init__(self, *a: object, **kw: object) -> None:
                pass

            def sendto(self, data: bytes, addr: object) -> None:
                sent.append(data)

            def close(self) -> None:
                pass

        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_PID", os.getpid() + 1)
        monkeypatch.setattr(wd_mod.socket, "socket", lambda *a, **kw: FakeSocket())

        wd_mod._sd_notify("WATCHDOG=1")
        assert sent == []

    def test_sd_notify_sends_when_pid_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_sd_notify`` sends when WATCHDOG_PID matches current PID."""
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[bytes] = []

        class FakeSocket:
            def __init__(self, *a: object, **kw: object) -> None:
                pass

            def sendto(self, data: bytes, addr: object) -> None:
                sent.append(data)

            def close(self) -> None:
                pass

        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_PID", os.getpid())
        monkeypatch.setattr(wd_mod.socket, "socket", lambda *a, **kw: FakeSocket())

        wd_mod._sd_notify("WATCHDOG=1")
        assert sent == [b"WATCHDOG=1"]


# ---------------------------------------------------------------------------
# HardwareWatchdog
# ---------------------------------------------------------------------------


class TestHardwareWatchdog:
    def test_open_returns_false_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        hw = HardwareWatchdog()
        assert hw.open() is False
        assert hw.enabled is False

    def test_open_returns_false_when_device_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_DEV", str(tmp_path / "nonexistent"))
        hw = HardwareWatchdog()
        assert hw.open() is False

    def test_open_returns_false_when_disabled_via_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        watchdog_path = tmp_path / "watchdog"
        watchdog_path.touch()
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_DEV", str(watchdog_path))
        monkeypatch.setenv("CYBERWAVE_HARDWARE_WATCHDOG", "false")
        hw = HardwareWatchdog()
        assert hw.open() is False

    def test_close_disarms_watchdog(self) -> None:
        hw = HardwareWatchdog()
        hw._fd = None
        hw.close()
        assert hw.enabled is False


# ---------------------------------------------------------------------------
# ProcessWatchdog
# ---------------------------------------------------------------------------


class TestProcessWatchdog:
    def test_start_and_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", None)

        pw = ProcessWatchdog()
        pw.start()
        pw.ping()
        pw.stop()

    def test_any_enabled_reflects_layers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", None)

        pw = ProcessWatchdog()
        assert pw.any_enabled is False

        pw.hardware.enabled = True
        assert pw.any_enabled is True

    def test_active_layers_empty_when_none_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", None)

        pw = ProcessWatchdog()
        assert pw.active_layers() == []

    def test_active_layers_orders_systemd_then_hardware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Order is stable so consumers can compare snapshots without sorting."""
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", None)

        pw = ProcessWatchdog()
        pw.systemd.enabled = True
        pw.hardware.enabled = True

        assert pw.active_layers() == ["systemd", "hardware"]

    def test_active_layers_subset_when_only_one_layer_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", None)

        pw = ProcessWatchdog()
        pw.systemd.enabled = True
        assert pw.active_layers() == ["systemd"]

        pw.systemd.enabled = False
        pw.hardware.enabled = True
        assert pw.active_layers() == ["hardware"]


# ---------------------------------------------------------------------------
# OOM score adjustment
# ---------------------------------------------------------------------------


class TestProtectEdgeCoreOom:
    def test_noop_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        protect_edge_core_oom()

    def test_writes_oom_score_adj(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")

        oom_file = tmp_path / "oom_score_adj"
        oom_file.write_text("0")

        import cyberwave_edge_core.watchdog as wd_mod

        original_path_class = Path

        monkeypatch.setattr(
            wd_mod,
            "Path",
            lambda p: (
                original_path_class(str(oom_file))
                if "oom_score_adj" in str(p)
                else original_path_class(p)
            ),
        )

        protect_edge_core_oom(score_adj=-800)
        assert oom_file.read_text() == "-800"

    def test_clamps_oom_score_adj_to_valid_range(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Out-of-range values are clamped to [-1000, 1000]."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")

        oom_file = tmp_path / "oom_score_adj"
        oom_file.write_text("0")

        import cyberwave_edge_core.watchdog as wd_mod

        original_path_class = Path
        monkeypatch.setattr(
            wd_mod,
            "Path",
            lambda p: (
                original_path_class(str(oom_file))
                if "oom_score_adj" in str(p)
                else original_path_class(p)
            ),
        )

        protect_edge_core_oom(score_adj=-5000)
        assert oom_file.read_text() == "-1000"
