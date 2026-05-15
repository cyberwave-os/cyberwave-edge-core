"""Process-level watchdog and OOM protection for Cyberwave Edge Core.

Scope: this module is the **orchestrator/host-level** watchdog.  It is
**not** the per-driver in-process command/teleop watchdog (see
``cyberwave-edge-nodes`` and ``cyberwave-edge-runtime`` for those — same
term, different layer) and it is **not** the worker container health
monitor (see :mod:`cyberwave_edge_core.worker_health`).

Provides three layers of resilience for edge devices running under heavy load
(e.g. Raspberry Pi 4 with YOLO inference):

1. **Systemd watchdog** — when Edge Core runs as a systemd service with
   ``WatchdogSec`` configured, the runtime loop pings the watchdog each
   cycle.  If the process hangs or dies, systemd restarts it automatically.

2. **Hardware watchdog** — on Linux devices with ``/dev/watchdog`` (all
   Raspberry Pi models), a hardware timer reboots the device if the
   edge-core process stops pinging.  This is a last-resort safety net
   for kernel-level hangs or total process death (e.g. OOM kill with no
   systemd recovery).

3. **OOM score adjustment** — lowers the edge-core process OOM score so
   the Linux OOM killer preferentially terminates the ML worker container
   (which Docker restarts automatically) instead of the orchestrator.
"""

from __future__ import annotations

import errno
import logging
import os
import platform
import socket
import struct
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Systemd watchdog (sd_notify protocol)
# ---------------------------------------------------------------------------

_NOTIFY_SOCKET: Optional[str] = os.environ.get("NOTIFY_SOCKET")
_WATCHDOG_USEC: Optional[int] = None
_WATCHDOG_PID: Optional[int] = None

if os.environ.get("WATCHDOG_USEC"):
    try:
        _WATCHDOG_USEC = int(os.environ["WATCHDOG_USEC"])
    except ValueError:
        pass

if os.environ.get("WATCHDOG_PID"):
    try:
        _WATCHDOG_PID = int(os.environ["WATCHDOG_PID"])
    except ValueError:
        pass


def _is_designated_watchdog_pid() -> bool:
    """Return True when this process is the systemd-designated notifier.

    Per ``sd_notify(3)``, ``WATCHDOG=1`` and ``STOPPING=1`` should only
    originate from the process whose PID matches ``$WATCHDOG_PID``
    (defaulting to the main service PID when unset).  ``READY=1`` is **not**
    PID-restricted by systemd and is allowed from any process the unit's
    ``NotifyAccess=`` permits — see :func:`_sd_notify`.
    """
    if _WATCHDOG_PID is None:
        return True
    return _WATCHDOG_PID == os.getpid()


# Notification states that systemd's sd_notify(3) restricts to the
# WATCHDOG_PID process.  READY=1 is deliberately **not** in this set:
# per sd_notify(3), only the periodic watchdog pings and the stopping
# notification are PID-restricted; readiness is governed by the unit's
# ``NotifyAccess=`` (which defaults to ``main`` and works as expected).
_PID_RESTRICTED_PREFIXES: tuple[str, ...] = ("WATCHDOG=", "STOPPING=")


def _sd_notify(state: str) -> None:
    """Send a notification to the systemd notify socket (best-effort).

    PID-restricted notifications (``WATCHDOG=`` / ``STOPPING=``) are
    skipped — and the skip is logged at DEBUG so misuse from a forked
    child is at least diagnosable — when this process is not the
    systemd-designated notifier.  ``READY=1`` is allowed from any PID
    so the boot path works correctly when edge-core runs under a
    PyInstaller ``--onefile`` bootloader (which forks the actual app
    into a child whose PID differs from systemd's ``MainPID``) — see
    :func:`_is_designated_watchdog_pid`.
    """
    if not _NOTIFY_SOCKET:
        return
    if state.startswith(_PID_RESTRICTED_PREFIXES) and not _is_designated_watchdog_pid():
        logger.debug(
            "Skipping %s from non-notifier pid %d (WATCHDOG_PID=%s)",
            state.split("=", 1)[0],
            os.getpid(),
            _WATCHDOG_PID,
        )
        return
    addr = _NOTIFY_SOCKET
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock.sendto(state.encode(), addr)
        finally:
            sock.close()
    except OSError:
        pass


class SystemdWatchdog:
    """Ping the systemd watchdog at the interval dictated by ``WatchdogSec``.

    The systemd recommendation is to ping at half the ``WatchdogSec``
    interval.  This class exposes the recommended interval so the caller
    can decide when to call :meth:`ping`.
    """

    def __init__(self) -> None:
        self.enabled = _WATCHDOG_USEC is not None and _WATCHDOG_USEC > 0
        self._interval_us = _WATCHDOG_USEC or 0
        self._last_ping: float = 0.0

    @property
    def recommended_interval_seconds(self) -> float:
        """Half the configured ``WatchdogSec``, in seconds."""
        if not self.enabled:
            return 0.0
        return self._interval_us / 2_000_000.0

    def notify_ready(self) -> None:
        """Tell systemd that startup is complete (``READY=1``)."""
        _sd_notify("READY=1")

    def notify_main_pid(self, pid: Optional[int] = None) -> None:
        """Tell systemd the real ``MainPID`` is *pid* (default: current process).

        Per ``sd_notify(3)`` ``MAINPID=`` is **not** PID-restricted; it
        only requires that the unit's ``NotifyAccess=`` permits this
        process (``all`` or ``exec`` work).  After systemd processes
        the message, ``$WATCHDOG_PID`` is updated so subsequent
        PID-restricted notifications (``WATCHDOG=`` / ``STOPPING=``)
        from this process are accepted.

        Refreshes the module-level ``_WATCHDOG_PID`` cache so the
        in-process guard inside :func:`_sd_notify` agrees with what
        systemd will see — otherwise the very next ``WATCHDOG=1`` call
        from this same process would still be silently dropped by our
        own guard.

        Used by :meth:`ProcessWatchdog.claim_main_pid` to fix the
        PyInstaller ``--onefile`` shape where the bootloader holds
        ``MainPID`` while the actual app runs in a forked child.
        """
        global _WATCHDOG_PID
        target = pid if pid is not None else os.getpid()
        _sd_notify(f"MAINPID={target}")
        _WATCHDOG_PID = target

    def ping(self) -> None:
        """Send ``WATCHDOG=1`` to the systemd notify socket."""
        if not self.enabled:
            return
        _sd_notify("WATCHDOG=1")
        self._last_ping = time.monotonic()

    def notify_stopping(self) -> None:
        """Tell systemd we are shutting down (``STOPPING=1``)."""
        _sd_notify("STOPPING=1")


# ---------------------------------------------------------------------------
# Hardware watchdog (/dev/watchdog)
# ---------------------------------------------------------------------------

_WATCHDOG_DEV = "/dev/watchdog"
# Only the ``SETTIMEOUT`` ioctl is used; ``KEEPALIVE`` is replaced by a plain
# ``write(1)`` and ``GETSUPPORT`` is unused.  See ``linux/watchdog.h``.
_WDIOC_SETTIMEOUT = 0xC0045706


class HardwareWatchdog:
    """Drive the Linux hardware watchdog timer (``/dev/watchdog``).

    On Raspberry Pi the ``bcm2835_wdt`` kernel module provides a hardware
    watchdog with a configurable timeout (default 15 s).  If the process
    stops writing to ``/dev/watchdog`` within the timeout, the SoC resets
    the board.

    Opening the device starts the watchdog.  The ``V`` magic-close
    convention is honoured: writing ``V`` before close disarms the timer
    so a graceful shutdown does not trigger a reboot.
    """

    def __init__(self, *, timeout_seconds: int = 60) -> None:
        self._fd: Optional[int] = None
        self._timeout = timeout_seconds
        self.enabled = False

    def open(self) -> bool:
        """Open ``/dev/watchdog`` and set the timeout.  Returns True on success."""
        if platform.system() != "Linux":
            return False
        if not Path(_WATCHDOG_DEV).exists():
            logger.debug("Hardware watchdog device %s not found", _WATCHDOG_DEV)
            return False

        env_flag = os.environ.get("CYBERWAVE_HARDWARE_WATCHDOG", "").lower()
        if env_flag in ("0", "false", "no", "off", "disabled"):
            logger.info("Hardware watchdog disabled via CYBERWAVE_HARDWARE_WATCHDOG")
            return False

        try:
            fd = os.open(_WATCHDOG_DEV, os.O_WRONLY)
        except OSError as exc:
            if exc.errno == errno.EACCES:
                logger.info(
                    "Cannot open %s (permission denied); "
                    "run as root or grant CAP_SYS_ADMIN to enable hardware watchdog",
                    _WATCHDOG_DEV,
                )
            elif exc.errno == errno.EBUSY:
                logger.info(
                    "Hardware watchdog %s already in use by another process",
                    _WATCHDOG_DEV,
                )
            else:
                logger.debug("Cannot open hardware watchdog: %s", exc)
            return False

        self._fd = fd
        try:
            buf = struct.pack("i", self._timeout)
            import fcntl

            fcntl.ioctl(fd, _WDIOC_SETTIMEOUT, buf)
            logger.info(
                "Hardware watchdog enabled (timeout=%ds)",
                self._timeout,
            )
        except OSError:
            logger.info(
                "Hardware watchdog opened (could not set custom timeout; using kernel default)",
            )

        self.enabled = True
        return True

    def ping(self) -> None:
        """Keep the hardware watchdog alive."""
        if self._fd is None:
            return
        try:
            os.write(self._fd, b"1")
        except OSError as exc:
            logger.warning("Hardware watchdog ping failed: %s", exc)

    def close(self) -> None:
        """Disarm and close the hardware watchdog (magic close)."""
        if self._fd is None:
            return
        try:
            os.write(self._fd, b"V")
            os.close(self._fd)
            logger.info("Hardware watchdog disarmed and closed")
        except OSError:
            logger.debug("Error closing hardware watchdog", exc_info=True)
        finally:
            self._fd = None
            self.enabled = False


# ---------------------------------------------------------------------------
# Unified process watchdog
# ---------------------------------------------------------------------------


class ProcessWatchdog:
    """Unified watchdog combining systemd and hardware watchdog layers.

    Usage (preferred — three-phase, safe under PyInstaller ``--onefile``)::

        wd = ProcessWatchdog()
        wd.claim_main_pid()           # MAINPID=<self> — fixes onefile fork
        run_blocking_boot_work()      # docker pulls, MQTT, twin sync, ...
        wd.mark_ready()               # READY=1
        wd.start_pinging()            # opens /dev/watchdog + first WATCHDOG=1
        while running:
            do_work()
            wd.ping()                 # every reconcile cycle
        wd.stop()                     # graceful shutdown

    The :meth:`claim_main_pid` call matters when the process tree
    contains a parent that holds the systemd ``MainPID`` (and
    therefore ``$WATCHDOG_PID``) but is not the process that actually
    runs the ping loop.  The canonical case is PyInstaller
    ``--onefile`` binaries: the bootloader stays as ``MainPID`` while
    the unpacked Python app runs in a forked child.  ``MAINPID=`` is
    not PID-restricted by ``sd_notify(3)``, so the child can rebind;
    after that, the PID-restricted ``WATCHDOG=1`` pings emitted by
    :meth:`start_pinging` / :meth:`ping` are accepted by systemd.
    Requires the unit to set ``NotifyAccess=all`` (or ``exec``); the
    systemd default (``main``) rejects ``MAINPID=`` from a non-main
    PID.

    ``READY=1`` is independent — it is never PID-restricted, so
    :meth:`mark_ready` works from any process the unit's
    ``NotifyAccess=`` permits, regardless of whether
    :meth:`claim_main_pid` was called first.

    The legacy :meth:`start` entry point combines readiness and
    pinging for backwards compatibility; new callers should prefer
    the three-phase form so a future regression of "WATCHDOG=1
    silently dropped from a child PID" surfaces immediately.
    """

    def __init__(self, *, hardware_watchdog_timeout: int = 60) -> None:
        self.systemd = SystemdWatchdog()
        self.hardware = HardwareWatchdog(timeout_seconds=hardware_watchdog_timeout)
        self._ready_signalled = False
        self._pinging_started = False

    @property
    def any_enabled(self) -> bool:
        return self.systemd.enabled or self.hardware.enabled

    def active_layers(self) -> list[str]:
        """Return the list of enabled watchdog layer names, in protocol order.

        Used by the bootstrap ``edge_health`` publisher so the dashboard
        can show which protective layers the edge actually has (e.g.
        ``["systemd"]`` on a developer laptop vs ``["systemd",
        "hardware"]`` on a Raspberry Pi).  Order is stable
        (systemd first, hardware second) so consumers can compare
        snapshots without sorting.
        """
        layers: list[str] = []
        if self.systemd.enabled:
            layers.append("systemd")
        if self.hardware.enabled:
            layers.append("hardware")
        return layers

    def claim_main_pid(self) -> None:
        """Tell systemd this process is the real ``MainPID`` (``MAINPID=``).

        Required when the executable is a PyInstaller ``--onefile``
        binary: the bootloader holds systemd's ``MainPID`` /
        ``$WATCHDOG_PID`` while the actual Python app runs in a
        forked child whose PID differs.  Without this rebind every
        ``WATCHDOG=1`` ping from the child is silently dropped by
        the PID guard inside :func:`_sd_notify`, the watchdog times
        out at ``WatchdogSec``, and systemd ``SIGABRT``s the process
        every minute (the 0.1.4.1463 regression).

        ``MAINPID=`` is not itself PID-restricted, so this method is
        safe to call from the forked child.  It does require the unit
        to set ``NotifyAccess=all`` (or ``exec``); the default
        ``main`` rejects ``MAINPID=`` from a non-main PID.

        Idempotent and a no-op when there is no notify socket.  Must
        be called **before** :meth:`mark_ready` and
        :meth:`start_pinging`.
        """
        self.systemd.notify_main_pid()

    def mark_ready(self) -> None:
        """Signal systemd that startup is complete (``READY=1``).

        Idempotent.  Safe to call from any process the unit's
        ``NotifyAccess=`` permits — including a process whose PID does
        not match ``$WATCHDOG_PID`` (e.g. a PyInstaller ``--onefile``
        child).  Must be invoked **after** all blocking boot work has
        finished (Docker images pulled, MQTT connected, twin sync
        reconciled) and **before** any branch that could fork/exec
        into a deeper child whose readiness we'd otherwise miss.

        Does **not** open ``/dev/watchdog`` and does **not** emit
        ``WATCHDOG=1`` — use :meth:`start_pinging` for that, from the
        ``WATCHDOG_PID`` process (call :meth:`claim_main_pid` first
        to make this process the ``WATCHDOG_PID`` when running under
        PyInstaller ``--onefile``).
        """
        if self._ready_signalled:
            return
        self.systemd.notify_ready()
        self._ready_signalled = True

    def start_pinging(self, *, ping_interval_seconds: Optional[float] = None) -> None:
        """Open the hardware watchdog and emit the first ``WATCHDOG=1`` ping.

        Must run in the systemd-designated ``WATCHDOG_PID`` process
        (or a thread inside it).  Calling from a forked child is a
        no-op for the systemd half because ``WATCHDOG=`` is
        PID-restricted by :func:`_sd_notify`; the hardware half
        still opens ``/dev/watchdog`` but the periodic
        :meth:`ping` cadence would then need to come from the main
        process anyway, so this is best avoided.

        ``ping_interval_seconds`` is the cadence at which the caller
        intends to invoke :meth:`ping`.  When supplied, it is compared
        against the systemd-recommended interval (half of
        ``WatchdogSec``) and a warning is logged if the caller pings
        too slowly — a useful sanity check against future config drift
        where the runtime loop or ``WatchdogSec`` change independently.

        Idempotent.
        """
        if self._pinging_started:
            return
        self.hardware.open()
        layers = self.active_layers()
        if layers:
            logger.info("Watchdog ping loop started (layers: %s)", ", ".join(layers))

        if (
            ping_interval_seconds is not None
            and self.systemd.enabled
            and self.systemd.recommended_interval_seconds > 0
            and ping_interval_seconds > self.systemd.recommended_interval_seconds
        ):
            logger.warning(
                "Reconcile interval (%.1fs) exceeds half of WatchdogSec (%.1fs); "
                "systemd may declare the service hung before a ping arrives — "
                "increase WatchdogSec or decrease the loop interval",
                ping_interval_seconds,
                self.systemd.recommended_interval_seconds,
            )

        self.ping()
        self._pinging_started = True

    def start(self, *, ping_interval_seconds: Optional[float] = None) -> None:
        """Initialise both watchdog layers and signal readiness.

        Backwards-compatible wrapper around :meth:`mark_ready` followed
        by :meth:`start_pinging`.  New callers should prefer the
        two-phase form so readiness can be signalled before any
        operation that could fork/exec a child (which would silently
        drop the PID-restricted ``WATCHDOG=1`` pings — see the class
        docstring).
        """
        self.mark_ready()
        self.start_pinging(ping_interval_seconds=ping_interval_seconds)

    def ping(self) -> None:
        """Ping all active watchdog layers."""
        self.systemd.ping()
        self.hardware.ping()

    def stop(self) -> None:
        """Gracefully shut down watchdog layers."""
        self.systemd.notify_stopping()
        self.hardware.close()


# ---------------------------------------------------------------------------
# OOM score adjustment
# ---------------------------------------------------------------------------


def protect_edge_core_oom(score_adj: int = -800) -> None:
    """Lower the edge-core process OOM score so it survives memory pressure.

    The Linux OOM killer assigns each process a score (0–1000, higher =
    more likely to be killed).  By setting ``oom_score_adj`` to a negative
    value we make edge-core a poor target, so the kernel preferentially
    kills the heavier ML worker container instead.

    Docker containers inherit the default ``oom_score_adj`` (0), and ML
    worker containers consume far more memory than the orchestrator, so
    they naturally score higher and get killed first.  This adjustment
    provides an additional safety margin.

    The value ``-800`` keeps edge-core safe under typical conditions
    without requiring ``oom_score_adj=-1000`` (which effectively makes
    the process unkillable and could wedge the system).
    """
    if platform.system() != "Linux":
        return

    oom_path = Path(f"/proc/{os.getpid()}/oom_score_adj")
    if not oom_path.exists():
        return

    try:
        score_adj = max(-1000, min(1000, score_adj))
        oom_path.write_text(str(score_adj))
        logger.info(
            "OOM score adjusted to %d for edge-core process (pid=%d)",
            score_adj,
            os.getpid(),
        )
    except PermissionError:
        logger.debug(
            "Cannot adjust OOM score (requires root); "
            "consider adding OOMScoreAdjust=-800 to the systemd unit"
        )
    except OSError as exc:
        logger.debug("Failed to adjust OOM score: %s", exc)
