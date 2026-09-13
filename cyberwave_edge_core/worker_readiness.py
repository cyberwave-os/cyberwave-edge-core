"""Track a worker container from ``running`` to actually usable.

Docker says ``running`` a second or two after ``docker run``, but the process
inside still has to import worker modules (downloading weights on a cold
device), connect to MQTT, subscribe and warm up before it can accept a run.
``_run_container`` used to resolve the ``worker_starting`` alert at
``running``, so that whole span — the longest part of startup — had no alert
open. Two backend consumers read that alert as "the device is still bringing
its worker up": the pending-reason copy under a ``REQUESTED`` run, and
``sweep_stuck_edge_manual_executions``, which defers failing the row while it
is live.

The worker publishes milestones to a JSON file on a bind-mounted tmpfs (see
``cyberwave.workers.status``). ``WorkerManager`` keeps the alert open past
container start and calls :func:`evaluate_readiness` once per reconcile tick
to decide what to do with it.

Everything here is a pure function of (status record, container status,
elapsed, restart count). No thread and no deadline held in memory: elapsed is
anchored to the container's own ``StartedAt``, so the pull that precedes the
container does not eat into the worker's ready budget — a seam the card
deliberately spans. That anchor is also why *restart count* has to be passed
in: a Docker-initiated restart rewrites ``StartedAt``, so a crash loop would
otherwise reset the clock on every iteration and never time out.

The card itself is not stateless: ``WorkerManager`` holds it in a
``_StartupCard`` keyed by container name, and nothing rebuilds a context from
an alert that already exists on the backend. An edge-core killed while a
worker container keeps running therefore drops the card rather than resuming
it (CYB-3868 follow-up).
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WORKER_STATUS_FILENAME = "worker-status.json"

DEFAULT_STATUS_DIR_LINUX = Path("/run/cyberwave")
"""tmpfs on any systemd host: no flash writes, and nothing survives a reboot
to be mistaken for a live worker."""

READY_PHASE = "ready"

#: Milestones the worker publishes, in order. Mirrors the ``PHASE_*``
#: constants in the SDK's ``cyberwave.workers.status``.
WORKER_STARTUP_PHASES: tuple[str, ...] = (
    "booting",
    "connecting_mqtt",
    "loading_models",
    "subscribed",
    "warming_up",
    READY_PHASE,
)

DEFAULT_READY_TIMEOUT_SECONDS = 900
"""Generous on purpose: a cold Jetson loading multi-GB checkpoints can pass
ten minutes, and calling a working device failed is worse than leaving a
progress card up on a wedged one."""

STATUS_FILE_GRACE_SECONDS = 60.0
"""How long to wait for the first record before concluding the worker image
predates status reporting. Well past interpreter start and SDK import on slow
hardware, so a merely-slow worker is never mistaken for an old one."""

#: Container states that mean the worker is never going to report.
_DEAD_CONTAINER_STATES = frozenset({"exited", "dead", "none"})


def default_status_dir(config_dir: Path) -> Path:
    """Host directory holding the worker status file. macOS has no ``/run``,
    so Darwin falls back to the already-mounted config directory."""
    if platform.system() == "Darwin":
        return config_dir / "run"
    return DEFAULT_STATUS_DIR_LINUX


def read_status(status_path: Path) -> dict[str, Any] | None:
    """Read the worker's status record, or ``None`` if unreadable.

    Every malformed state reads as ``None``: the next tick is seconds away,
    and a spurious failure alert costs far more than a skipped sample.
    """
    try:
        raw = status_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        record = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(record, dict):
        return None
    phase = record.get("phase")
    if not isinstance(phase, str) or not phase:
        return None
    return record


def clear_status(status_path: Path) -> None:
    """Remove a stale status file before starting a container, so the next
    poll cannot read the previous worker's ``ready`` and resolve at once."""
    try:
        status_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.debug("Could not clear worker status file %s", status_path, exc_info=True)


@dataclass(frozen=True)
class ReadinessOutcome:
    """What the caller should do to the ``worker_starting`` alert this tick."""

    patch: dict[str, Any] = field(default_factory=dict)
    """Metadata to merge onto the alert. Empty means nothing new to report."""

    close: bool = False
    """Resolve the alert — startup is over, successfully or not."""

    failure: str = ""
    """Non-empty marks the alert failed before resolving."""

    phase: str = ""
    """Failure phase key, selecting the user-facing copy."""


def evaluate_readiness(
    record: dict[str, Any] | None,
    *,
    container_status: str,
    elapsed_seconds: float,
    ready_timeout_seconds: float = DEFAULT_READY_TIMEOUT_SECONDS,
    grace_seconds: float = STATUS_FILE_GRACE_SECONDS,
    restart_count: int = 0,
) -> ReadinessOutcome:
    """Decide the alert's fate from one observation of the worker.

    *elapsed_seconds* is measured from the container's start rather than from
    when this process began watching, so it survives an edge-core restart.

    *restart_count* is Docker's own tally for this container. It is the only
    evidence a crash loop leaves behind for a poller: see below.
    """
    if container_status in _DEAD_CONTAINER_STATES:
        reached = str(record.get("phase")) if record else "unknown"
        return ReadinessOutcome(
            close=True,
            failure=f"container {container_status} during worker startup (phase={reached})",
            phase="worker_startup_exited",
        )

    # A worker that reported ``ready`` is ready, however many restarts it took
    # to get there — checked before the restart rule below so a container that
    # crashed once and then came up is not called failed.
    if record is not None and str(record.get("phase")) == READY_PHASE:
        return ReadinessOutcome(patch=alert_patch(record), close=True)

    if restart_count > 0:
        # The worker runs under ``--restart unless-stopped``. Docker rewrites
        # ``StartedAt`` on every restart, so *elapsed_seconds* resets with it
        # and the timeout below can never fire; and between crash and restart
        # the container parks in ``restarting``, which is not a dead state, so
        # a 15 s poll almost never lands on the ``exited`` moment either. A
        # worker crash-looping on a missing checkpoint would hold the card
        # open forever. The restart count is the evidence that survives both.
        reached = str(record.get("phase")) if record else "unknown"
        return ReadinessOutcome(
            close=True,
            failure=(
                f"worker container restarted {restart_count}x during startup (phase={reached})"
            ),
            phase="worker_startup_exited",
        )

    if record is None:
        if elapsed_seconds >= grace_seconds:
            # Worker image predates status reporting: fall back to the
            # historical "running means started".
            logger.info(
                "No worker status after %.0fs; assuming a worker image "
                "without status reporting and treating running as started",
                elapsed_seconds,
            )
            return ReadinessOutcome(close=True)
        return ReadinessOutcome()

    patch = alert_patch(record)
    if elapsed_seconds >= ready_timeout_seconds:
        return ReadinessOutcome(
            patch=patch,
            close=True,
            failure=(
                f"worker did not become ready within {int(ready_timeout_seconds)}s "
                f"(last phase={record.get('phase')})"
            ),
            phase="worker_ready_timeout",
        )

    return ReadinessOutcome(patch=patch)


def alert_patch(record: dict[str, Any]) -> dict[str, Any]:
    """Translate a worker status record into a ``worker_starting`` patch.

    Only fields the alert consumers understand; ``pid`` and ``boot_id`` stay
    in edge-core's logs.
    """
    patch: dict[str, Any] = {"phase": str(record.get("phase"))}
    for key in ("models_total", "models_ready"):
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            patch[key] = value
    detail = record.get("detail")
    if isinstance(detail, str) and detail:
        patch["detail"] = detail
    return patch
