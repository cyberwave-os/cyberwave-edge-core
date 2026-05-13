"""System resource monitoring for Cyberwave Edge Core.

Provides lightweight, non-blocking monitoring of host system resources
(memory, CPU temperature) to detect conditions that could cause edge-core
or its managed containers to be killed by the OS.

Designed for resource-constrained devices (Raspberry Pi 4 with 1–4 GB RAM)
running ML inference workloads.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
# Low-level readers (Linux /proc and /sys)
# ---------------------------------------------------------------------------


def _read_memory_info() -> Optional[MemoryInfo]:
    """Parse ``/proc/meminfo`` for total and available memory."""
    if platform.system() != "Linux":
        return None
    try:
        fields: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].rstrip(":") in (
                    "MemTotal",
                    "MemAvailable",
                    "MemFree",
                    "Buffers",
                    "Cached",
                ):
                    fields[parts[0].rstrip(":")] = int(parts[1])

        total_kb = fields.get("MemTotal", 0)
        available_kb = fields.get("MemAvailable")
        if available_kb is None:
            available_kb = (
                fields.get("MemFree", 0)
                + fields.get("Buffers", 0)
                + fields.get("Cached", 0)
            )

        if total_kb == 0:
            return None

        total_mb = total_kb / 1024.0
        available_mb = available_kb / 1024.0
        used_percent = (1.0 - available_mb / total_mb) * 100.0

        return MemoryInfo(
            total_mb=round(total_mb, 1),
            available_mb=round(available_mb, 1),
            used_percent=round(used_percent, 1),
        )
    except OSError:
        return None


def _read_cpu_temperature() -> Optional[CpuTemperature]:
    """Read CPU temperature from sysfs thermal zones.

    Works on Raspberry Pi (``/sys/class/thermal/thermal_zone0/temp``)
    and most other Linux SBCs.
    """
    if platform.system() != "Linux":
        return None

    thermal_paths = [
        ("/sys/class/thermal/thermal_zone0/temp", "thermal_zone0"),
        ("/sys/devices/virtual/thermal/thermal_zone0/temp", "thermal_zone0"),
    ]

    for path_str, source in thermal_paths:
        path = Path(path_str)
        if path.exists():
            try:
                raw = path.read_text().strip()
                millidegrees = int(raw)
                celsius = millidegrees / 1000.0
                return CpuTemperature(celsius=round(celsius, 1), source=source)
            except (ValueError, OSError):
                continue

    return None


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
            memory=_read_memory_info(),
            cpu_temp=_read_cpu_temperature(),
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

    def suggest_worker_memory_limit_mb(self) -> Optional[int]:
        """Suggest a memory limit for the worker container.

        On devices with 4 GB or less, suggests reserving ~25% for the OS
        and edge-core, giving the rest to the worker container.  Returns
        None when total memory is unknown or above 8 GB (where limits are
        less critical).
        """
        if self._last_snapshot is None or self._last_snapshot.memory is None:
            return None

        total_mb = self._last_snapshot.memory.total_mb
        if total_mb > 8192:
            return None

        reserved_mb = max(512, total_mb * 0.25)
        limit = int(total_mb - reserved_mb)
        return max(256, limit)

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
