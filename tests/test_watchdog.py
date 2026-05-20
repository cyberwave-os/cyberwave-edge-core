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

    def test_notify_extend_timeout_sends_usec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        sd = SystemdWatchdog()
        sd.notify_extend_timeout(30_000_000)
        assert "EXTEND_TIMEOUT_USEC=30000000" in sent

    def test_notify_extend_timeout_skips_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        sd = SystemdWatchdog()
        sd.notify_extend_timeout(0)
        assert sent == []

    def test_notify_status_sends_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        sd = SystemdWatchdog()
        sd.notify_status("Pulling driver images: foo:latest")
        assert "STATUS=Pulling driver images: foo:latest" in sent

    def test_notify_stopping_sends_stopping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        sd = SystemdWatchdog()
        sd.notify_stopping()
        assert "STOPPING=1" in sent

    def test_sd_notify_watchdog_skipped_when_pid_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``WATCHDOG=1`` is PID-restricted per ``sd_notify(3)``."""
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

    def test_sd_notify_stopping_skipped_when_pid_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``STOPPING=1`` is PID-restricted per ``sd_notify(3)``."""
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

        wd_mod._sd_notify("STOPPING=1")
        assert sent == []

    def test_sd_notify_ready_allowed_when_pid_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for 0.1.4.1459 → 0.1.4.1463: ``READY=1`` must NOT be PID-gated.

        Under PyInstaller ``--onefile`` the bootloader holds ``MainPID``
        (and therefore systemd's ``$WATCHDOG_PID``) while the actual
        Python app runs in a forked child whose PID differs.  Per
        ``sd_notify(3)`` only ``WATCHDOG=`` and ``STOPPING=`` are
        PID-restricted; readiness is governed by ``NotifyAccess=`` and
        is allowed from any permitted process.  The previous guard
        dropped ``READY=1`` from the child and systemd ``Type=notify``
        timed out at ``TimeoutStartSec=300``.
        """
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

        wd_mod._sd_notify("READY=1")
        assert sent == [b"READY=1"]

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

    def test_notify_main_pid_unblocks_watchdog_pings_from_pid_mismatched_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for the 0.1.4.1463 watchdog-loop kill.

        Under PyInstaller ``--onefile`` the bootloader holds
        ``$WATCHDOG_PID`` while the actual app runs in a forked child.
        The pre-fix ``WATCHDOG=1`` from the child was silently dropped
        by :func:`_sd_notify`'s PID guard, systemd's ``WatchdogSec``
        elapsed, and the unit was ``SIGABRT``-killed every minute.
        After :meth:`SystemdWatchdog.notify_main_pid` rebinds the
        ``MAINPID`` to the current process, subsequent ``WATCHDOG=1``
        notifications from the same process must reach the socket.
        """
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
        # Simulate the onefile shape: $WATCHDOG_PID points at the
        # bootloader, not us.
        monkeypatch.setattr(wd_mod, "_WATCHDOG_PID", os.getpid() + 1)
        monkeypatch.setattr(wd_mod.socket, "socket", lambda *a, **kw: FakeSocket())

        # Pre-rebind: PID-restricted notifications from us are dropped.
        wd_mod._sd_notify("WATCHDOG=1")
        assert sent == [], "guard should drop WATCHDOG=1 before MAINPID rebind"

        # Go through ProcessWatchdog.claim_main_pid (the production
        # caller) rather than SystemdWatchdog.notify_main_pid directly,
        # so a regression that breaks the wrapper is also caught.
        ProcessWatchdog().claim_main_pid()

        # Post-rebind: the MAINPID datagram went out, the in-process
        # cache was refreshed, and subsequent WATCHDOG=1 pings pass
        # through the guard.
        assert sent == [f"MAINPID={os.getpid()}".encode()]
        assert wd_mod._WATCHDOG_PID == os.getpid()

        sent.clear()
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

    def test_mark_ready_emits_ready_even_from_non_watchdog_pid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end regression: ``mark_ready`` must reach the notify socket
        even when this process is **not** ``$WATCHDOG_PID``.

        This is the bug that caused 0.1.4.1459 → 0.1.4.1463 to time out
        under systemd ``Type=notify``: the PyInstaller ``--onefile``
        bootloader is ``MainPID`` while the unpacked Python child sends
        ``READY=1``.
        """
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_PID", os.getpid() + 1)
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", 30_000_000)
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        pw = ProcessWatchdog()
        pw.mark_ready()

        assert "READY=1" in sent

    def test_mark_ready_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling :meth:`mark_ready` twice emits ``READY=1`` exactly once."""
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_PID", os.getpid())
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        pw = ProcessWatchdog()
        pw.mark_ready()
        pw.mark_ready()

        assert sent.count("READY=1") == 1

    def test_start_pinging_emits_first_watchdog_ping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """:meth:`start_pinging` opens the hardware layer and sends the first ping."""
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_PID", os.getpid())
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", 30_000_000)
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        pw = ProcessWatchdog()
        pw.start_pinging()

        assert "WATCHDOG=1" in sent

    def test_start_pinging_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated :meth:`start_pinging` calls don't re-open or double-log."""
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", None)

        pw = ProcessWatchdog()
        opens: list[int] = []
        monkeypatch.setattr(pw.hardware, "open", lambda: opens.append(1) or False)
        monkeypatch.setattr(pw, "ping", lambda: None)

        pw.start_pinging()
        pw.start_pinging()

        assert len(opens) == 1, "hardware.open() must run at most once"

    def test_start_wrapper_calls_both_phases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Backwards-compat :meth:`start` runs both phases in order."""
        import cyberwave_edge_core.watchdog as wd_mod

        order: list[str] = []
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", None)

        pw = ProcessWatchdog()
        monkeypatch.setattr(pw, "mark_ready", lambda: order.append("mark_ready"))
        monkeypatch.setattr(
            pw,
            "start_pinging",
            lambda *, ping_interval_seconds=None: order.append("start_pinging"),
        )

        pw.start()

        assert order == ["mark_ready", "start_pinging"]

    def test_extend_timeout_delegates_to_systemd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``extend_timeout`` converts seconds to microseconds and delegates."""
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        pw = ProcessWatchdog()
        pw.extend_timeout(30.0)
        assert "EXTEND_TIMEOUT_USEC=30000000" in sent

    def test_notify_status_delegates_to_systemd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        sent: list[str] = []
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        monkeypatch.setattr(wd_mod, "_sd_notify", lambda state: sent.append(state))

        pw = ProcessWatchdog()
        pw.notify_status("test status")
        assert "STATUS=test status" in sent

    def test_extend_timeout_not_pid_restricted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``EXTEND_TIMEOUT_USEC=`` must not be PID-restricted like ``WATCHDOG=``."""
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

        wd_mod._sd_notify("EXTEND_TIMEOUT_USEC=30000000")
        assert sent == [b"EXTEND_TIMEOUT_USEC=30000000"]

    def test_any_enabled_reflects_layers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", None)

        pw = ProcessWatchdog()
        assert pw.any_enabled is False

        pw.hardware.enabled = True
        assert pw.any_enabled is True

    def test_active_layers_empty_when_none_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
