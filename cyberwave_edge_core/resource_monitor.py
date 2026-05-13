"""System resource monitoring for Cyberwave Edge Core.

Scope: monitors **host-level** memory and CPU temperature so the operator
can correlate OOM kills / thermal throttling with edge-core or worker
restarts.  This is independent of:

- ``cyberwave_edge_core.worker_health`` (per-container health probes), and
- ``cyberwave_cli.commands.edge.bench`` / ``cyberwave_cli.monitor`` (one-shot
  diagnostic readers in the CLI).

Provides lightweight, non-blocking monitoring of host system resources
(memory, CPU temperature) to detect conditions that could cause edge-core
or its managed containers to be killed by the OS.

Designed for resource-constrained devices (Raspberry Pi 4 with 1–4 GB RAM)
running ML inference workloads.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from cyberwave.edge.host_metrics import (
    read_host_cpu_temperature,
    read_host_memory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_WARNING_PERCENT = 85.0
MEMORY_CRITICAL_PERCENT = 92.0
CPU_TEMP_WARNING_C = 75.0
CPU_TEMP_CRITICAL_C = 82.0

MONITOR_CHECK_INTERVAL_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemoryInfo:
    """Snapshot of system memory usage from ``/proc/meminfo``."""

    total_mb: float
    available_mb: float
    used_percent: float

    @property
    def is_warning(self) -> bool:
        return self.used_percent >= MEMORY_WARNING_PERCENT

    @property
    def is_critical(self) -> bool:
        return self.used_percent >= MEMORY_CRITICAL_PERCENT


@dataclass
class CpuTemperature:
    """CPU temperature reading."""

    celsius: float
    source: str

    @property
    def is_warning(self) -> bool:
        return self.celsius >= CPU_TEMP_WARNING_C

    @property
    def is_critical(self) -> bool:
        return self.celsius >= CPU_TEMP_CRITICAL_C


@dataclass
class ResourceSnapshot:
    """Combined snapshot of system resources."""

    timestamp: float
    memory: Optional[MemoryInfo]
    cpu_temp: Optional[CpuTemperature]

    @property
    def is_healthy(self) -> bool:
        if self.memory and self.memory.is_critical:
            return False
        if self.cpu_temp and self.cpu_temp.is_critical:
            return False
        return True

    @property
    def has_warnings(self) -> bool:
        if self.memory and self.memory.is_warning:
            return True
        if self.cpu_temp and self.cpu_temp.is_warning:
            return True
        return False


# ---------------------------------------------------------------------------
# Threshold-aware readers
# ---------------------------------------------------------------------------
#
# The low-level parsing of ``/proc/meminfo`` and ``/sys/class/thermal`` lives
# in :mod:`cyberwave.edge.host_metrics` so that the CLI (``cyberwave edge
# bench`` and ``cyberwave monitor``) can share the same primitives.  Here we
# only add the edge-core-specific severity layer (the warning/critical
# thresholds defined above) by wrapping the SDK's plain data carriers.


def read_memory_info() -> Optional[MemoryInfo]:
    """Return a threshold-aware memory snapshot, or ``None`` if unavailable.

    Delegates ``/proc/meminfo`` parsing to
    :func:`cyberwave.edge.host_metrics.read_host_memory` and wraps the
    result with edge-core's warning/critical thresholds.
    """
    raw = read_host_memory()
    if raw is None:
        return None
    return MemoryInfo(
        total_mb=raw.total_mb,
        available_mb=raw.available_mb,
        used_percent=raw.used_percent,
    )


def read_cpu_temperature() -> Optional[CpuTemperature]:
    """Return a threshold-aware CPU temperature reading, or ``None``.

    Delegates sysfs thermal-zone discovery and reading to
    :func:`cyberwave.edge.host_metrics.read_host_cpu_temperature`, which
    enumerates ``/sys/class/thermal/thermal_zone*``, prefers CPU-typed
    zones and returns the hottest matching reading.
    """
    raw = read_host_cpu_temperature()
    if raw is None:
        return None
    return CpuTemperature(celsius=raw.celsius, source=raw.source)


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------


class SystemResourceMonitor:
    """Lightweight host resource monitor for the edge-core runtime loop.

    Call :meth:`check` each reconcile cycle.  The monitor throttles
    actual reads to ``MONITOR_CHECK_INTERVAL_SECONDS`` to avoid overhead.
    """

    def __init__(self) -> None:
        self._last_check: float = 0.0
        self._last_snapshot: Optional[ResourceSnapshot] = None
        self._consecutive_critical: int = 0

        self._warning_logged_at: float = 0.0
        self._critical_logged_at: float = 0.0
        self._log_interval = 60.0

    def check(self) -> Optional[ResourceSnapshot]:
        """Return a resource snapshot, or None if skipped due to throttle.

        The snapshot is always cached as ``last_snapshot`` for callers that
        want the most recent reading regardless of throttle.
        """
        now = time.monotonic()
        if now - self._last_check < MONITOR_CHECK_INTERVAL_SECONDS:
            return self._last_snapshot

        self._last_check = now
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            memory=read_memory_info(),
            cpu_temp=read_cpu_temperature(),
        )
        self._last_snapshot = snapshot
        self._log_resource_state(snapshot, now)
        return snapshot

    @property
    def last_snapshot(self) -> Optional[ResourceSnapshot]:
        return self._last_snapshot

    @property
    def consecutive_critical_count(self) -> int:
        return self._consecutive_critical

    def _log_resource_state(self, snapshot: ResourceSnapshot, now: float) -> None:
        """Log warnings/criticals with rate-limiting to avoid log spam."""
        if not snapshot.is_healthy:
            self._consecutive_critical += 1
            if now - self._critical_logged_at >= self._log_interval:
                self._critical_logged_at = now
                parts = []
                if snapshot.memory and snapshot.memory.is_critical:
                    parts.append(
                        f"memory={snapshot.memory.used_percent:.0f}% "
                        f"(available={snapshot.memory.available_mb:.0f}MB)"
                    )
                if snapshot.cpu_temp and snapshot.cpu_temp.is_critical:
                    parts.append(f"cpu_temp={snapshot.cpu_temp.celsius:.1f}°C")
                logger.warning(
                    "CRITICAL resource pressure: %s — "
                    "edge-core or managed containers may be OOM-killed or throttled "
                    "(consecutive_critical=%d)",
                    ", ".join(parts),
                    self._consecutive_critical,
                )
        elif snapshot.has_warnings:
            self._consecutive_critical = 0
            if now - self._warning_logged_at >= self._log_interval:
                self._warning_logged_at = now
                parts = []
                if snapshot.memory and snapshot.memory.is_warning:
                    parts.append(
                        f"memory={snapshot.memory.used_percent:.0f}% "
                        f"(available={snapshot.memory.available_mb:.0f}MB)"
                    )
                if snapshot.cpu_temp and snapshot.cpu_temp.is_warning:
                    parts.append(f"cpu_temp={snapshot.cpu_temp.celsius:.1f}°C")
                logger.info("Resource pressure elevated: %s", ", ".join(parts))
        else:
            self._consecutive_critical = 0
