"""Readiness and bounded lifecycle checks with portable and real-systemd coverage."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest


@contextmanager
def _short_unix_socket_path() -> Iterator[Path]:
    """Yield a short ``Path`` suitable for binding an ``AF_UNIX`` socket.

    Avoids the macOS 104-character path limit that pytest's ``tmp_path``
    routinely blows past when the test name is long.
    """
    base = Path(tempfile.mkdtemp(prefix="cw-wd-", dir="/tmp"))
    try:
        yield base / "notify.sock"
    finally:
        for child in base.iterdir():
            try:
                child.unlink()
            except FileNotFoundError:
                pass
        try:
            base.rmdir()
        except OSError:
            pass


_CHILD_SCRIPT = textwrap.dedent(
    """
    import sys
    from cyberwave_edge_core.watchdog import ProcessWatchdog
    pw = ProcessWatchdog()
    pw.mark_ready()
    sys.exit(0)
    """
).strip()


def test_mark_ready_from_subprocess_with_pid_mismatch() -> None:
    """A PyInstaller child must notify readiness despite a different WATCHDOG_PID."""
    with _short_unix_socket_path() as sock_path:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        server.bind(str(sock_path))
        server.settimeout(10.0)
        try:
            fake_watchdog_pid = os.getpid()

            env = os.environ.copy()
            env["NOTIFY_SOCKET"] = str(sock_path)
            env["WATCHDOG_PID"] = str(fake_watchdog_pid)
            # Inherit the test's PYTHONPATH so the child can import the
            # in-tree package without an install step.
            env["PYTHONPATH"] = os.pathsep.join(sys.path)

            child = subprocess.run(
                [sys.executable, "-c", _CHILD_SCRIPT],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert child.returncode == 0, (
                f"child failed: stdout={child.stdout!r} stderr={child.stderr!r}"
            )
            try:
                data, _addr = server.recvfrom(256)
            except socket.timeout:
                pytest.fail(
                    "no datagram received within 10 s — the PID-mismatch "
                    "silent-drop regression has returned"
                )
            assert data == b"READY=1", (
                f"expected READY=1 from PID-mismatched subprocess, got {data!r}"
            )
        finally:
            server.close()


def _systemd_scope() -> list[str]:
    return [] if os.environ.get("CYBERWAVE_TEST_SYSTEMD_SYSTEM") == "1" else ["--user"]


@pytest.fixture
def live_systemd():
    available = False
    if sys.platform == "linux" and shutil.which("systemd-run") and shutil.which("systemctl"):
        try:
            result = subprocess.run(
                ["systemctl", *_systemd_scope(), "is-system-running"],
                capture_output=True, text=True, timeout=3,
            )
            available = result.stdout.strip() in {"running", "degraded"}
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not available:
        message = "A running Linux systemd manager is required"
        if os.environ.get("CYBERWAVE_REQUIRE_SYSTEMD_TESTS") == "1":
            pytest.fail(message)
        pytest.skip(message)


def test_service_reaches_active_under_systemd_run(tmp_path: Path, live_systemd) -> None:
    unit_name = f"cw-edgecore-notify-regression-{uuid.uuid4().hex[:8]}"

    script_path = tmp_path / "edge_core_notify_simulator.py"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import sys, time
            sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
            from cyberwave_edge_core.watchdog import ProcessWatchdog
            pw = ProcessWatchdog()
            pw.mark_ready()
            pw.start_pinging()
            # Keep the unit alive long enough for is-active to observe it.
            time.sleep(60)
            """
        ).strip()
        + "\n"
    )

    systemd_run = subprocess.run(
        [
            "systemd-run",
            *_systemd_scope(),
            f"--unit={unit_name}",
            "--service-type=notify",
            "--property=TimeoutStartSec=30s",
            "--property=NotifyAccess=main",
            sys.executable,
            str(script_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert systemd_run.returncode == 0, (
        f"systemd-run failed: stdout={systemd_run.stdout!r} stderr={systemd_run.stderr!r}"
    )

    try:
        deadline = time.monotonic() + 30.0
        state = ""
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["systemctl", *_systemd_scope(), "is-active", unit_name],
                capture_output=True,
                text=True,
                timeout=3,
            )
            state = result.stdout.strip()
            if state == "active":
                break
            if state in {"failed", "inactive"}:
                # Capture the unit journal for the failure message.
                journal = subprocess.run(
                    ["journalctl", *_systemd_scope(), "-u", unit_name, "--no-pager"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                pytest.fail(
                    f"unit entered {state!r} before becoming active; journal:\n{journal.stdout}"
                )
            time.sleep(0.5)

        assert state == "active", (
            f"unit did not reach active within 30 s (last state={state!r}) — "
            "this is the 0.1.4.1459 → 0.1.4.1463 regression"
        )
    finally:
        subprocess.run(
            ["systemctl", *_systemd_scope(), "stop", unit_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        subprocess.run(
            ["systemctl", *_systemd_scope(), "reset-failed", unit_name],
            capture_output=True,
            text=True,
            timeout=5,
        )


@pytest.mark.parametrize("action", ["startup_timeout", "restart", "stop"])
def test_stalled_service_lifecycle(tmp_path, live_systemd, action):
    unit = f"cw-edgecore-lifecycle-{uuid.uuid4().hex[:8]}.service"
    script = tmp_path / "stalled_boot.py"
    script.write_text(textwrap.dedent(f"""
        import signal, sys, time
        from pathlib import Path
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from cyberwave_edge_core.watchdog import ProcessWatchdog
        watchdog = ProcessWatchdog()
        watchdog._startup_deadline = time.monotonic() + 3
        signal.signal(signal.SIGTERM, lambda *_: watchdog.begin_shutdown())
        watchdog.extend_timeout(30)
        Path({str(tmp_path / 'booting')!r}).touch()
        while True:
            watchdog.extend_timeout(30)
            time.sleep(0.05)
    """))

    def control(*args):
        return subprocess.run(
            ["systemctl", *_systemd_scope(), *args],
            capture_output=True, text=True, timeout=5,
        )

    def state():
        result = control("show", unit, "--all", "--property=ActiveState,MainPID,Job,Result")
        assert result.returncode == 0, result.stderr
        return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    def wait_for(predicate):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            fields = state()
            if predicate(fields):
                return fields
            time.sleep(0.1)
        pytest.fail(f"Service lifecycle did not complete: {fields}")

    try:
        result = subprocess.run(
            ["systemd-run", *_systemd_scope(), f"--unit={unit}", "--no-block",
             "--service-type=notify", "--property=TimeoutStartSec=2s",
             "--property=TimeoutStopSec=1s", "--property=KillMode=control-group",
             "--property=SendSIGKILL=yes", "--property=Restart=no",
             sys.executable, str(script)],
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0, result.stderr
        initial = wait_for(lambda f: f["ActiveState"] == "activating" and f["MainPID"] != "0")
        wait_for(lambda f: (tmp_path / "booting").exists())
        if action != "startup_timeout":
            result = control(action, "--no-block", unit)
            assert result.returncode == 0, result.stderr
            if action == "restart":
                wait_for(lambda f: f["MainPID"] not in {"0", initial["MainPID"]})
                result = control("stop", "--no-block", unit)
                assert result.returncode == 0, result.stderr
        final = wait_for(lambda f: f["ActiveState"] in {"inactive", "failed"}
                         and f["MainPID"] == "0" and f["Job"] in {"", "0"})
        if action in {"startup_timeout", "stop"}:
            assert final["Result"] == "timeout"
    finally:
        control("stop", "--no-block", unit)
        control("kill", "--signal=SIGKILL", unit)
        control("reset-failed", unit)


def test_sigterm_during_startup_never_announces_ready(monkeypatch):
    import signal
    import threading
    from click.testing import CliRunner
    from cyberwave_edge_core import main, watchdog, startup, resource_monitor

    event = threading.Event()
    monkeypatch.setattr(main, "shutdown_event", event)
    handlers, notifications, cleaned = {}, [], []
    monkeypatch.setattr(main.signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    monkeypatch.setattr(watchdog, "protect_edge_core_oom", lambda: None)
    monkeypatch.setattr(watchdog, "_NOTIFY_SOCKET", "/run/systemd/notify")
    monkeypatch.setattr(watchdog, "_sd_notify", notifications.append)
    monkeypatch.setattr(resource_monitor.SystemResourceMonitor, "check", lambda self: None)
    monkeypatch.setattr(startup, "_graceful_shutdown", lambda worker: cleaned.append(worker))
    def boot(**kwargs):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        kwargs["watchdog"].extend_timeout(30)
        return True
    monkeypatch.setattr(main, "run_startup_checks", boot)
    result = CliRunner().invoke(main.cli, [])
    assert result.exit_code == 0, result.output
    assert cleaned == [None]
    assert "STOPPING=1" in notifications
    assert "READY=1" not in notifications
    assert not any(s.startswith("EXTEND_TIMEOUT") for s in notifications)
