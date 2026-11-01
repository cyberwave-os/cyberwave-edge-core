"""Unit tests for cyberwave_edge_core.watchdog module.

Covers:
- SystemdWatchdog: ping, notify_ready, notify_stopping, recommended interval
- HardwareWatchdog: open, ping, close (mocked /dev/watchdog)
- ProcessWatchdog: start, ping, stop combine both layers
- protect_edge_core_oom: OOM score adjustment
"""

from __future__ import annotations

import errno
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


class TestHardwareWatchdogTimeout:
    @pytest.fixture(autouse=True)
    def _no_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CYBERWAVE_HARDWARE_WATCHDOG_TIMEOUT_SECONDS", raising=False)

    def test_twice_watchdog_sec_with_a_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", 60_000_000)
        assert wd_mod.resolve_hardware_watchdog_timeout() == 120
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", 90_000_000)
        assert wd_mod.resolve_hardware_watchdog_timeout() == 180
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        assert wd_mod.resolve_hardware_watchdog_timeout() == 120
        assert HardwareWatchdog()._timeout == 120

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", 60_000_000)
        monkeypatch.setenv("CYBERWAVE_HARDWARE_WATCHDOG_TIMEOUT_SECONDS", "300")
        assert wd_mod.resolve_hardware_watchdog_timeout() == 300
        monkeypatch.setenv("CYBERWAVE_HARDWARE_WATCHDOG_TIMEOUT_SECONDS", "soon")
        assert wd_mod.resolve_hardware_watchdog_timeout() == 120

    def test_open_warns_when_the_driver_applies_a_short_timeout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import fcntl
        import struct

        import cyberwave_edge_core.watchdog as wd_mod

        device = tmp_path / "watchdog"
        device.touch()
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_DEV", str(device))
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", 60_000_000)
        monkeypatch.delenv("CYBERWAVE_HARDWARE_WATCHDOG", raising=False)
        requested: list[int] = []

        def ioctl(fd: int, request: int, arg: bytes) -> bytes:
            requested.append(struct.unpack("i", arg)[0])
            return struct.pack("i", 30)

        monkeypatch.setattr(fcntl, "ioctl", ioctl)
        hw = HardwareWatchdog()
        with caplog.at_level("WARNING"):
            assert hw.open() is True
        hw.close()

        assert requested == [120]
        assert "not above WatchdogSec" in caplog.text
        assert device.read_bytes() == b"V"

    def test_open_warns_when_the_driver_refuses_the_timeout_and_keeps_a_short_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A Raspberry Pi's bcm2835_wdt rejects anything above 15 s with EINVAL."""
        import fcntl
        import struct

        import cyberwave_edge_core.watchdog as wd_mod

        device = tmp_path / "watchdog"
        device.touch()
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_DEV", str(device))
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", 60_000_000)
        monkeypatch.delenv("CYBERWAVE_HARDWARE_WATCHDOG", raising=False)

        def ioctl(fd: int, request: int, arg: bytes) -> bytes:
            if request == wd_mod._WDIOC_SETTIMEOUT:
                raise OSError(errno.EINVAL, "Invalid argument")
            assert request == wd_mod._WDIOC_GETTIMEOUT
            return struct.pack("i", 15)

        monkeypatch.setattr(fcntl, "ioctl", ioctl)
        hw = HardwareWatchdog()
        with caplog.at_level("INFO"):
            assert hw.open() is True
        hw.close()

        assert "refused by the driver, effective=15s" in caplog.text
        assert "Hardware watchdog timeout (15s) is not above WatchdogSec (60s)" in caplog.text

    def test_open_reads_the_kept_timeout_from_sysfs_without_the_ioctl(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import fcntl

        import cyberwave_edge_core.watchdog as wd_mod

        device = tmp_path / "watchdog"
        device.touch()
        sysfs = tmp_path / "watchdog0"
        sysfs.mkdir()
        (sysfs / "timeout").write_text("15\n")
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_DEV", str(device))
        monkeypatch.setattr(wd_mod, "_WATCHDOG_SYSFS", str(sysfs))
        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", 60_000_000)
        monkeypatch.delenv("CYBERWAVE_HARDWARE_WATCHDOG", raising=False)

        def ioctl(fd: int, request: int, arg: bytes) -> bytes:
            raise OSError(errno.ENOTTY, "Inappropriate ioctl for device")

        monkeypatch.setattr(fcntl, "ioctl", ioctl)
        hw = HardwareWatchdog()
        with caplog.at_level("WARNING"):
            assert hw.open() is True
        hw.close()

        assert "Hardware watchdog timeout (15s) is not above WatchdogSec" in caplog.text


class TestHardwareWatchdogDisarm:
    def _pipe(self, hw: HardwareWatchdog) -> int:
        read_fd, write_fd = os.pipe()
        hw._fd = write_fd
        hw.enabled = True
        return read_fd

    def test_close_writes_the_magic_character(self) -> None:
        hw = HardwareWatchdog(timeout_seconds=120)
        read_fd = self._pipe(hw)
        hw.ping()
        hw.close()
        assert os.read(read_fd, 16) == b"1V"
        assert hw.enabled is False
        os.close(read_fd)

    def test_a_close_that_interrupts_a_ping_is_left_to_the_ping(self) -> None:
        """A signal handler runs on the pinging thread: blocking would deadlock,
        and a ping written after the ``V`` would cancel the disarm."""
        hw = HardwareWatchdog(timeout_seconds=120)
        read_fd = self._pipe(hw)
        with hw._lock:
            hw.close()
            assert hw._fd is not None
        hw.ping()
        assert os.read(read_fd, 16) == b"1V"
        assert hw._fd is None
        os.close(read_fd)

    def test_a_close_racing_a_ping_on_another_thread_is_not_lost(self) -> None:
        """The ping released the lock between close's failed acquire and the flag."""

        class _FreedInBetween:
            def __init__(self) -> None:
                self.attempts = 0

            def acquire(self, blocking: bool = True) -> bool:
                self.attempts += 1
                return self.attempts > 1

            def release(self) -> None:
                pass

        hw = HardwareWatchdog(timeout_seconds=120)
        read_fd = self._pipe(hw)
        hw._lock = _FreedInBetween()  # type: ignore[assignment]
        hw.close()
        assert os.read(read_fd, 16) == b"V"
        assert hw._fd is None
        assert hw._close_pending is False
        os.close(read_fd)

    def test_disarm_now_ignores_the_lock(self) -> None:
        hw = HardwareWatchdog(timeout_seconds=120)
        read_fd = self._pipe(hw)
        with hw._lock:
            hw.disarm_now()
        assert os.read(read_fd, 16) == b"V"
        assert hw._fd is None
        os.close(read_fd)

    def _sysfs(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: str, nowayout: str):
        import cyberwave_edge_core.watchdog as wd_mod

        sysfs = tmp_path / "sysfs"
        sysfs.mkdir()
        (sysfs / "state").write_text(f"{state}\n")
        (sysfs / "nowayout").write_text(f"{nowayout}\n")
        device = tmp_path / "watchdog"
        device.touch()
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_SYSFS", str(sysfs))
        monkeypatch.setattr(wd_mod, "_WATCHDOG_DEV", str(device))
        monkeypatch.delenv("CYBERWAVE_HARDWARE_WATCHDOG", raising=False)
        return device

    def test_disarm_stale_stops_a_running_timer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        device = self._sysfs(monkeypatch, tmp_path, "active", "0")
        assert HardwareWatchdog(timeout_seconds=120).disarm_stale() is True
        assert device.read_bytes() == b"V"

    @pytest.mark.parametrize(("state", "nowayout"), [("inactive", "0"), ("active", "1")])
    def test_disarm_stale_never_arms_or_fights_nowayout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: str, nowayout: str
    ) -> None:
        device = self._sysfs(monkeypatch, tmp_path, state, nowayout)
        assert HardwareWatchdog(timeout_seconds=120).disarm_stale() is False
        assert device.read_bytes() == b""

    def test_disarm_stale_without_sysfs_does_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(wd_mod, "_WATCHDOG_SYSFS", str(tmp_path / "missing"))
        monkeypatch.setattr(wd_mod, "_WATCHDOG_DEV", str(tmp_path / "never-opened"))
        assert HardwareWatchdog(timeout_seconds=120).disarm_stale() is False

    def test_sigabrt_disarms_then_dies_with_the_default_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import signal

        import cyberwave_edge_core.watchdog as wd_mod

        handlers: dict[int, object] = {}
        exit_hooks: list[object] = []
        kills: list[tuple[int, int]] = []
        monkeypatch.setattr(wd_mod.signal, "signal", lambda sig, h: handlers.__setitem__(sig, h))
        monkeypatch.setattr(wd_mod.atexit, "register", exit_hooks.append)
        monkeypatch.setattr(wd_mod.os, "kill", lambda pid, sig: kills.append((pid, sig)))
        monkeypatch.setattr(wd_mod.faulthandler, "dump_traceback", lambda **kw: None)
        pw = ProcessWatchdog(hardware_watchdog_timeout=120)
        read_fd = self._pipe(pw.hardware)

        pw.install_disarm_handlers()
        assert exit_hooks == [pw.hardware.close]
        handlers[signal.SIGABRT](signal.SIGABRT, None)

        assert os.read(read_fd, 16) == b"V"
        assert handlers[signal.SIGABRT] is signal.SIG_DFL
        assert kills == [(os.getpid(), signal.SIGABRT)]
        os.close(read_fd)


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

    def test_shutdown_during_open_closes_the_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A SIGTERM between the stopping check and the open must not leave it armed."""
        import cyberwave_edge_core.watchdog as wd_mod

        monkeypatch.setattr(wd_mod, "_WATCHDOG_USEC", None)
        monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", None)

        pw = ProcessWatchdog()
        calls: list[str] = []
        monkeypatch.setattr(
            pw.hardware, "open", lambda: calls.append("open") or pw.begin_shutdown()
        )
        monkeypatch.setattr(pw.hardware, "close", lambda: calls.append("close"))
        monkeypatch.setattr(pw, "ping", lambda: calls.append("ping"))

        pw.start_pinging()

        assert calls == ["open", "close"]

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


def test_startup_extensions_have_absolute_deadline(monkeypatch):
    import cyberwave_edge_core.watchdog as wd_mod

    now = [100.0]
    sent = []
    monkeypatch.setattr(wd_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
    monkeypatch.setattr(wd_mod, "_sd_notify", sent.append)
    watchdog = wd_mod.ProcessWatchdog()
    watchdog.extend_timeout(30)
    assert sent[-1] == "EXTEND_TIMEOUT_USEC=30000000"
    now[0] = 1898
    watchdog.extend_timeout(30)
    assert sent[-1] == "EXTEND_TIMEOUT_USEC=2000000"
    before = list(sent)
    now[0] = 1901
    watchdog.extend_timeout(30)
    assert sent == before


def test_startup_deadline_follows_startup_timeout_setting(monkeypatch):
    import cyberwave_edge_core.watchdog as wd_mod

    monkeypatch.setenv("CYBERWAVE_STARTUP_TIMEOUT_MINUTES", "90")
    monkeypatch.setattr(wd_mod.time, "monotonic", lambda: 100.0)
    assert wd_mod.ProcessWatchdog()._startup_deadline == 100.0 + 90 * 60


def test_shutdown_disables_extensions_and_readiness(monkeypatch):
    import cyberwave_edge_core.watchdog as wd_mod

    sent = []
    monkeypatch.setattr(wd_mod, "_NOTIFY_SOCKET", "/run/systemd/notify")
    monkeypatch.setattr(wd_mod, "_sd_notify", sent.append)
    watchdog = wd_mod.ProcessWatchdog()
    opened = []
    monkeypatch.setattr(watchdog.hardware, "open", lambda: opened.append(True))
    watchdog.begin_shutdown()
    watchdog.extend_timeout(30)
    watchdog.mark_ready()
    watchdog.start_pinging()
    assert sent == ["STOPPING=1"]
    assert not watchdog._ready_signalled
    assert not opened


def _boot_cli(monkeypatch: pytest.MonkeyPatch, calls: list[str]):
    """Run the edge-core CLI's default command with every boot step mocked."""
    from unittest.mock import Mock

    from click.testing import CliRunner

    import cyberwave_edge_core.watchdog as wd_mod
    from cyberwave_edge_core import main, resource_monitor

    process_watchdog = Mock()
    process_watchdog.hardware.disarm_stale.side_effect = lambda: calls.append("disarm_stale")
    process_watchdog.hardware.close.side_effect = lambda: calls.append("close")
    handlers: dict[int, object] = {}
    monkeypatch.setattr(wd_mod, "ProcessWatchdog", lambda: process_watchdog)
    monkeypatch.setattr(wd_mod, "protect_edge_core_oom", lambda: None)
    monkeypatch.setattr(resource_monitor, "SystemResourceMonitor", Mock())
    monkeypatch.setattr(
        main.signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler)
    )
    monkeypatch.setattr(main, "_sigterm_handler", lambda signum, frame: calls.append("sigterm"))
    monkeypatch.setattr(main, "run_runtime_loop", Mock())

    def boot(**_kwargs):
        calls.append("startup")
        return True

    monkeypatch.setattr(main, "run_startup_checks", boot)
    result = CliRunner().invoke(main.cli, [])
    assert result.exit_code == 0, result.output
    return process_watchdog, handlers


def test_boot_disarms_a_stale_hardware_timer_before_the_startup_checks(monkeypatch):
    calls: list[str] = []
    _boot_cli(monkeypatch, calls)
    assert calls.index("disarm_stale") < calls.index("startup")


def test_sigterm_closes_the_hardware_watchdog(monkeypatch):
    """The ordered shutdown stops containers without pinging.

    An armed hardware timer left running through it would reset the board.
    """
    import signal

    calls: list[str] = []
    process_watchdog, handlers = _boot_cli(monkeypatch, calls)
    process_watchdog.install_disarm_handlers.assert_called_once_with()
    calls.clear()
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert calls == ["sigterm", "close"]
    process_watchdog.begin_shutdown.assert_called()
