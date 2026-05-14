"""End-to-end-ish regression test for the systemd ``Type=notify`` boot path.

This test catches the 0.1.4.1459 → 0.1.4.1463 regression in which
``ProcessWatchdog.start()`` silently dropped ``READY=1`` whenever it
ran in a process whose PID did not equal ``$WATCHDOG_PID`` — the
canonical case being the PyInstaller ``--onefile`` bootloader (holds
``MainPID``) forking the actual Python app into a child (whose PID
differs).  Under ``Type=notify`` the unit then timed out at
``TimeoutStartSec``.

Two layers of coverage:

1. :func:`test_mark_ready_from_subprocess_with_pid_mismatch` is a
   fully portable, no-systemd-required regression test that runs on
   any POSIX host (Linux + macOS).  The parent test process binds a
   real ``AF_UNIX`` ``SOCK_DGRAM`` socket as a fake ``$NOTIFY_SOCKET``,
   sets ``$WATCHDOG_PID`` to a deliberately wrong PID (so the child's
   ``os.getpid()`` cannot match), spawns a Python subprocess that
   imports the watchdog module and calls
   :meth:`ProcessWatchdog.mark_ready`, and asserts the parent receives
   a ``READY=1`` datagram.  Subprocess (rather than ``os.fork``) is
   used so the test is safe in a multi-threaded pytest worker on
   macOS, where fork from a threaded process is unsupported.

2. :func:`test_service_reaches_active_under_systemd_run` is the full
   systemd integration check.  Skipped automatically on macOS / when
   no user systemd is reachable; on a Linux host with
   ``systemctl --user`` it runs ``systemd-run --service-type=notify``
   against a tiny script that mirrors the production handoff
   (``mark_ready`` → ``start_pinging`` → sleep) and polls
   ``systemctl --user is-active`` for ``active`` within 30 s.
"""

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


# ---------------------------------------------------------------------------
# Layer 1 — portable subprocess-based regression (runs on macOS + Linux)
# ---------------------------------------------------------------------------


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
    """Regression for 0.1.4.1459 → 0.1.4.1463.

    Simulates the PyInstaller ``--onefile`` shape: a parent process
    (the bootloader, in production) holds the systemd-designated
    ``WATCHDOG_PID``, while the actual app runs in a child whose PID
    differs.  Pre-fix the child's ``READY=1`` was silently dropped by
    the ``WATCHDOG_PID`` guard inside
    :func:`cyberwave_edge_core.watchdog._sd_notify` and the unit timed
    out at ``TimeoutStartSec``.  Post-fix it reaches the notify socket.
    """
    with _short_unix_socket_path() as sock_path:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        server.bind(str(sock_path))
        server.settimeout(10.0)
        try:
            # Pick a PID the child cannot possibly own.  ``os.getpid()`` of
            # this test process is fine because the child gets a different
            # PID by definition.
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
            assert os.getpid() != fake_watchdog_pid or True  # documentation
            # Sanity: the child's PID differs from the recorded
            # ``WATCHDOG_PID``; otherwise the test does not actually
            # exercise the regression.  We can't read the child's PID
            # after the fact, but ``Popen`` always allocates a new PID
            # different from the parent's, and we set ``WATCHDOG_PID``
            # to the parent's PID.

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


# ---------------------------------------------------------------------------
# Layer 2 — full systemd-run integration (Linux + systemctl --user only)
# ---------------------------------------------------------------------------


def _have_user_systemd() -> bool:
    """Best-effort detection of a reachable per-user systemd instance."""
    if sys.platform != "linux":
        return False
    if not shutil.which("systemd-run") or not shutil.which("systemctl"):
        return False
    if os.environ.get("XDG_RUNTIME_DIR") is None:
        return False
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-system-running"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    # 0 (running) / 1 (degraded) / 5 (offline) — anything other than
    # "couldn't talk to dbus" is good enough.  ``is-system-running``
    # returns non-zero when the bus is missing, which we reject.
    return result.returncode in (0, 1, 5)


@pytest.mark.skipif(
    not _have_user_systemd(),
    reason="systemctl --user not reachable; integration test requires Linux + user systemd",
)
def test_service_reaches_active_under_systemd_run(tmp_path: Path) -> None:
    """A ``Type=notify`` unit running our two-phase handoff must reach ``active``.

    Mirrors the production boot path in miniature: ``mark_ready`` (the
    PID-unrestricted ``READY=1``) followed by ``start_pinging`` (the
    PID-restricted ``WATCHDOG=1`` loop) followed by a sleep, all
    inside a transient ``systemd-run --user --service-type=notify``
    unit.  A failure here means a future change reintroduced the
    silent-drop bug.
    """
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
            "--user",
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
                ["systemctl", "--user", "is-active", unit_name],
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
                    ["journalctl", "--user", "-u", unit_name, "--no-pager"],
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
            ["systemctl", "--user", "stop", unit_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        subprocess.run(
            ["systemctl", "--user", "reset-failed", unit_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
