"""Worker health monitoring and restart accounting for Cyberwave Edge Core.

Provides:
- ``WorkerHealthMonitor``: polls container health, tracks restart counts and
  reasons, enforces restart rate limiting, and emits structured health snapshots.
- ``WorkerHealthState``: an immutable snapshot of health at a point in time.
- ``RestartRecord``: a single restart event with timestamp and reason.

Design goals
~~~~~~~~~~~~
* Non-blocking: every public method completes quickly and never raises.
* Decoupled: no direct import of startup.py; wired in by the reconcile loop.
* Persistent across reconcile cycles: the monitor is instantiated once by
  ``_reconcile_worker_watcher`` and retained across calls.
* Safe under fault injection: a bad worker update that crashes the container
  is detected within the next reconcile cycle and logged with reasons.

Restart rate limiting
~~~~~~~~~~~~~~~~~~~~~
To protect against crash loops, the monitor caps automatic restarts using an
exponential back-off: after ``max_restarts_in_window`` restarts within
``restart_window_seconds``, further restarts are suppressed until the window
resets.  The circuit-breaker state is surfaced in ``WorkerHealthState`` so
operators can see it in ``worker status``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RESTART_WINDOW_SECONDS: float = 300.0  # 5-minute sliding window
DEFAULT_MAX_RESTARTS_IN_WINDOW: int = 5  # trip circuit-breaker after N restarts


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RestartRecord:
    """Record of a single container restart event."""

    timestamp: float
    reason: str
    success: bool


@dataclass
class WorkerHealthState:
    """Snapshot of worker health at a single point in time."""

    container_name: str
    container_status: str  # running / exited / none / unknown

    # Liveness / readiness
    is_healthy: bool  # container is running and not in crash loop
    is_ready: bool  # container running + readiness probe passed

    # Restart accounting
    restart_count: int  # total restarts tracked by this monitor
    recent_restarts: int  # restarts within the sliding window
    restart_records: list[RestartRecord] = field(default_factory=list)

    # Circuit-breaker
    circuit_breaker_tripped: bool = False
    circuit_breaker_tripped_at: Optional[float] = None

    # Timing
    observed_at: float = field(default_factory=time.time)
    uptime_seconds: Optional[float] = None  # seconds since last successful start

    def summary_line(self) -> str:
        """Return a one-line human-readable summary."""
        if self.circuit_breaker_tripped:
            return (
                f"{self.container_name}: circuit-breaker tripped "
                f"({self.recent_restarts} restarts in window)"
            )
        status_emoji = "✓" if self.is_healthy else "✗"
        return (
            f"{status_emoji} {self.container_name}: {self.container_status} "
            f"(restarts={self.restart_count})"
        )


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class WorkerHealthMonitor:
    """Track worker container health and restart history.

    Intended to be used as a long-lived object that persists across
    reconcile-loop cycles.  Create once; call ``check()`` each cycle.

    Parameters
    ----------
    container_name:
        Docker container name to monitor.
    restart_window_seconds:
        Sliding window size for restart-rate limiting.
    max_restarts_in_window:
        Maximum allowed restarts within the window before circuit-breaker trips.
    readiness_probe:
        Optional callable that returns True when the container is considered
        ready (e.g. a health endpoint check).  If None, readiness = running.
    """

    def __init__(
        self,
        container_name: str,
        *,
        restart_window_seconds: float = DEFAULT_RESTART_WINDOW_SECONDS,
        max_restarts_in_window: int = DEFAULT_MAX_RESTARTS_IN_WINDOW,
        readiness_probe: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._container_name = container_name
        self._restart_window = restart_window_seconds
        self._max_restarts = max_restarts_in_window
        self._readiness_probe = readiness_probe

        self._restart_records: list[RestartRecord] = []
        self._circuit_breaker_tripped: bool = False
        self._circuit_breaker_tripped_at: Optional[float] = None
        self._last_start_time: Optional[float] = None

        # Track the previously-observed container status to detect spontaneous
        # exits (crash loops not triggered by a deliberate restart).
        self._last_container_status: Optional[str] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_restart(self, *, reason: str, success: bool) -> None:
        """Record that a deliberate restart was attempted.

        Call this from ``WorkerManager.restart()`` / ``WorkerWatcher`` after
        triggering a restart so that the monitor can account for it.
        """
        record = RestartRecord(
            timestamp=time.time(),
            reason=reason,
            success=success,
        )
        self._restart_records.append(record)
        if success:
            self._last_start_time = time.time()
        logger.info(
            "Worker restart recorded: container=%s reason=%r success=%s",
            self._container_name,
            reason,
            success,
        )
        self._update_circuit_breaker()

    def record_start(self) -> None:
        """Record that the container was intentionally started (not restarted)."""
        self._last_start_time = time.time()

    def check(self, container_status: Optional[str] = None) -> WorkerHealthState:
        """Probe the current container state and return a health snapshot.

        If *container_status* is not supplied, it is queried from Docker.
        Supplying it avoids a redundant ``docker inspect`` call when the status
        is already known to the caller.

        Side-effects:
        - Detects spontaneous container exits (crash-loops) and logs warnings.
        - Resets circuit-breaker when the window has cleared.
        """
        if container_status is None:
            container_status = self._query_container_status()

        self._detect_spontaneous_exit(container_status)
        self._last_container_status = container_status

        now = time.time()
        recent = self._restarts_in_window(now)

        # Reset circuit-breaker when the window has drained.
        if self._circuit_breaker_tripped and recent < self._max_restarts:
            logger.info(
                "Worker health circuit-breaker reset for %s (recent restarts=%d)",
                self._container_name,
                recent,
            )
            self._circuit_breaker_tripped = False
            self._circuit_breaker_tripped_at = None

        is_running = container_status == "running"
        is_ready = is_running and self._run_readiness_probe()
        is_healthy = is_running and not self._circuit_breaker_tripped

        uptime: Optional[float] = None
        if is_running and self._last_start_time is not None:
            uptime = now - self._last_start_time

        return WorkerHealthState(
            container_name=self._container_name,
            container_status=container_status,
            is_healthy=is_healthy,
            is_ready=is_ready,
            restart_count=len(self._restart_records),
            recent_restarts=recent,
            restart_records=list(self._restart_records),
            circuit_breaker_tripped=self._circuit_breaker_tripped,
            circuit_breaker_tripped_at=self._circuit_breaker_tripped_at,
            observed_at=now,
            uptime_seconds=uptime,
        )

    def is_restart_allowed(self) -> bool:
        """Return True when a restart is permitted (circuit-breaker is closed).

        Should be checked before triggering any automatic restart.
        """
        if not self._circuit_breaker_tripped:
            return True
        # Re-evaluate: maybe the window has cleared.
        now = time.time()
        recent = self._restarts_in_window(now)
        if recent < self._max_restarts:
            self._circuit_breaker_tripped = False
            self._circuit_breaker_tripped_at = None
            return True
        return False

    def reset(self) -> None:
        """Clear all recorded restarts and reset the circuit-breaker.

        Useful after a deliberate full-edge restart or operator intervention.
        """
        self._restart_records.clear()
        self._circuit_breaker_tripped = False
        self._circuit_breaker_tripped_at = None
        self._last_start_time = None
        logger.info("Worker health monitor reset for %s", self._container_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_container_status(self) -> str:
        """Query Docker for the current container status (best-effort)."""
        try:
            from .docker_helpers import docker_container_status

            return docker_container_status(self._container_name)
        except Exception as exc:
            logger.debug("Failed to query container status for %s: %s", self._container_name, exc)
            return "unknown"

    def _restarts_in_window(self, now: float) -> int:
        """Count restart records within the sliding window."""
        cutoff = now - self._restart_window
        return sum(1 for r in self._restart_records if r.timestamp >= cutoff)

    def _update_circuit_breaker(self) -> None:
        """Trip the circuit-breaker if too many restarts occurred recently."""
        now = time.time()
        recent = self._restarts_in_window(now)
        if recent >= self._max_restarts and not self._circuit_breaker_tripped:
            self._circuit_breaker_tripped = True
            self._circuit_breaker_tripped_at = now
            logger.warning(
                "Worker health circuit-breaker TRIPPED for %s: "
                "%d restarts in the last %.0f s — automatic restarts suppressed",
                self._container_name,
                recent,
                self._restart_window,
            )

    def _detect_spontaneous_exit(self, current_status: str) -> None:
        """Log a warning when the container exits without a deliberate restart."""
        if self._last_container_status == "running" and current_status in {"exited", "dead"}:
            logger.warning(
                "Worker container %s exited spontaneously (status=%s); "
                "this may indicate a crash loop",
                self._container_name,
                current_status,
            )

    def _run_readiness_probe(self) -> bool:
        """Run the readiness probe; returns True if none is configured."""
        if self._readiness_probe is None:
            return True
        try:
            return bool(self._readiness_probe())
        except Exception as exc:
            logger.debug("Readiness probe failed for %s: %s", self._container_name, exc)
            return False
