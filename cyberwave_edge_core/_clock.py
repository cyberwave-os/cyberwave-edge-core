"""Injectable monotonic clock for worker lifecycle timing.

Production uses :func:`time.monotonic`. Integration and chaos tests may
replace it via :func:`set_now_monotonic` (e.g. a :class:`FakeMonotonicClock`)
so cooldown and health-monitor windows can advance without wall-clock sleeps.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

_now_monotonic: Callable[[], float] = time.monotonic


def now_monotonic() -> float:
    """Return the current monotonic timestamp (seconds)."""
    return _now_monotonic()


def set_now_monotonic(fn: Optional[Callable[[], float]] = None) -> None:
    """Install *fn* as the monotonic clock, or reset to :func:`time.monotonic`."""
    global _now_monotonic
    _now_monotonic = time.monotonic if fn is None else fn


class FakeMonotonicClock:
    """Deterministic monotonic clock for tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._value = start

    def __call__(self) -> float:
        return self._value

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot advance clock backwards")
        self._value += seconds
