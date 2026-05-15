"""File-change watcher for the workers directory.

Uses a polling loop (not inotify) for cross-platform compatibility and
Docker bind-mount reliability.  Runs as part of the existing
``run_runtime_loop()`` reconciliation cycle.

Every call to ``reconcile_worker_files()`` checks whether the set of
``*.py`` files in the workers directory has changed since the last call.
If it has, it calls ``model_manager.ensure_models()`` for any new model
requirements and then restarts the worker container.

Restart cool-down
~~~~~~~~~~~~~~~~~
To prevent rapid successive restarts when files are being written incrementally
(e.g. ``rsync`` or ``scp``), the watcher enforces a minimum cool-down period
between automatic restarts (``min_restart_interval_seconds``).  If another
change is detected within the cool-down window, the restart is deferred until
the next reconcile cycle where the cool-down has expired.

Integration with WorkerHealthMonitor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The watcher passes a human-readable *reason* string to ``WorkerManager.restart()``.
The manager propagates this to the attached ``WorkerHealthMonitor`` so that
restart history is annotated with why the restart occurred.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .model_manager import ModelManager
    from .worker_health import WorkerHealthState
    from .worker_manager import WorkerManager

logger = logging.getLogger(__name__)

DEFAULT_MIN_RESTART_INTERVAL: float = 10.0  # seconds between automatic restarts


class WorkerWatcher:
    """Detect changes to the workers directory and trigger container restarts.

    The watcher is stateful: it remembers the last-known directory hash
    across calls to ``reconcile_worker_files()``.  The first call always
    establishes the baseline without triggering a restart.

    Parameters
    ----------
    workers_dir:
        Path to the directory containing ``*.py`` worker scripts.
    worker_manager:
        ``WorkerManager`` instance used to perform restarts.
    model_manager:
        ``ModelManager`` instance used to pre-download models before restarts.
    on_restart:
        Optional callback invoked after a successful restart trigger.
    min_restart_interval_seconds:
        Minimum time that must elapse between two successive automatic restarts.
        Defaults to ``DEFAULT_MIN_RESTART_INTERVAL``.
    mqtt_publish:
        Optional ``(topic: str, payload: bytes) -> None`` callback used
        to publish worker health payloads via MQTT.
    mqtt_health_topic:
        MQTT topic to publish worker health payloads to.
    """

    def __init__(
        self,
        workers_dir: Path,
        worker_manager: "WorkerManager",
        model_manager: "ModelManager",
        *,
        on_restart: Optional[Callable[[], None]] = None,
        min_restart_interval_seconds: float = DEFAULT_MIN_RESTART_INTERVAL,
        mqtt_publish: Optional[Callable[[str, bytes], None]] = None,
        mqtt_health_topic: Optional[str] = None,
    ) -> None:
        self._workers_dir = workers_dir
        self._worker_manager = worker_manager
        self._model_manager = model_manager
        self._on_restart = on_restart
        self._min_restart_interval = min_restart_interval_seconds
        self._mqtt_publish = mqtt_publish
        self._mqtt_health_topic = mqtt_health_topic

        self._last_hash: Optional[str] = None
        self._last_restart_at: Optional[float] = None
        self._pending_restart: bool = False  # True when a restart was deferred

    @property
    def worker_manager(self) -> "WorkerManager":
        """The ``WorkerManager`` used for container restarts."""
        return self._worker_manager

    def check_health(self) -> Optional["WorkerHealthState"]:
        """Run a health probe and return the current health snapshot.

        Returns None when no ``WorkerHealthMonitor`` is attached to the manager.
        Intended to be called each reconcile cycle to detect spontaneous exits.
        Also publishes the health state via MQTT when a publish callback is set.
        """
        hm = self._worker_manager.health_monitor
        if hm is None:
            return None
        state = hm.check()
        if state.circuit_breaker_tripped:
            logger.warning(
                "Worker health circuit-breaker is tripped for %s "
                "(%d restarts in window); automatic restarts are suppressed",
                state.container_name,
                state.recent_restarts,
            )
        self._publish_health_mqtt(state)
        return state

    def _publish_health_mqtt(self, state: "WorkerHealthState") -> None:
        """Best-effort publish of worker health to MQTT."""
        if self._mqtt_publish is None or self._mqtt_health_topic is None:
            return
        try:
            payload = json.dumps(state.to_mqtt_payload(), separators=(",", ":")).encode()
            self._mqtt_publish(self._mqtt_health_topic, payload)
        except Exception:
            logger.debug("Failed to publish worker health via MQTT", exc_info=True)

    def reconcile_worker_files(self) -> bool:
        """Check for worker file changes and restart container if needed.

        Returns True when a restart was triggered in this call, False otherwise.
        Returning False does not mean nothing changed — a restart may be pending
        due to cool-down.
        """
        current_hash = self._compute_directory_hash()

        if self._last_hash is None:
            self._last_hash = current_hash
            logger.debug("Worker watcher: baseline established (hash=%s)", current_hash[:12])
            return False

        changed = current_hash != self._last_hash

        if changed:
            logger.info(
                "Worker files changed (hash %s → %s)",
                self._last_hash[:12],
                current_hash[:12],
            )
            self._last_hash = current_hash
            self._pending_restart = True

        if not self._pending_restart:
            return False

        # Enforce cool-down between restarts.
        now = time.time()
        if self._last_restart_at is not None:
            elapsed = now - self._last_restart_at
            if elapsed < self._min_restart_interval:
                remaining = self._min_restart_interval - elapsed
                logger.debug("Worker restart deferred: cool-down %.1fs remaining", remaining)
                # Returns False here too — callers cannot distinguish "no change" from
                # "change detected but deferred".  _pending_restart stays True so the
                # restart fires on the next reconcile call after cool-down expires.
                return False

        self._pending_restart = False
        self._last_restart_at = now

        reason = "worker-files-changed"
        logger.info("Triggering worker container restart (reason=%r)", reason)
        self._ensure_models()
        ok = self._worker_manager.restart(reason=reason)
        if not ok:
            self._send_worker_start_failure_alert()

        if self._on_restart:
            try:
                self._on_restart()
            except Exception as exc:
                logger.warning("on_restart callback failed: %s", exc)

        return True

    def _compute_directory_hash(self) -> str:
        """Return a stable hash of all *.py files (names + mtimes + sizes)."""
        h = hashlib.sha256()
        if not self._workers_dir.exists():
            return h.hexdigest()

        entries: list[tuple[str, float, int]] = []
        try:
            for py_file in sorted(self._workers_dir.glob("*.py")):
                try:
                    stat = py_file.stat()
                    entries.append((py_file.name, stat.st_mtime, stat.st_size))
                except OSError:
                    pass
        except OSError:
            pass

        for name, mtime, size in entries:
            h.update(f"{name}:{mtime:.3f}:{size}\n".encode())

        return h.hexdigest()

    def _ensure_models(self) -> None:
        """Pre-download any models referenced by updated worker files."""
        try:
            model_ids = self._model_manager.scan_worker_model_ids(self._workers_dir)
            if not model_ids:
                return
            logger.info("Ensuring models before worker restart: %s", model_ids)
            self._model_manager.ensure_models(model_ids)
            self._send_model_failure_alerts()
        except Exception as exc:
            logger.warning(
                "Model pre-download before worker restart failed (will continue): %s", exc
            )

    def _send_model_failure_alerts(self) -> None:
        """Send alerts for any models that failed during the last ensure_models call."""
        failures = getattr(self._model_manager, "last_ensure_failures", None)
        if not failures:
            return
        twin_uuids = getattr(self._worker_manager, "_twin_uuids", [])
        if not twin_uuids:
            return
        try:
            from .startup import _send_model_failure_alerts

            _send_model_failure_alerts(twin_uuids=twin_uuids, failures=failures)
        except Exception:
            logger.debug(
                "Failed to send model-failure alerts from worker watcher",
                exc_info=True,
            )

    def _send_worker_start_failure_alert(self) -> None:
        """Send alerts when the worker container restart fails."""
        twin_uuids = getattr(self._worker_manager, "_twin_uuids", [])
        if not twin_uuids:
            return
        try:
            from .startup import _send_worker_start_failure_alerts

            _send_worker_start_failure_alerts(twin_uuids=twin_uuids)
        except Exception:
            logger.debug(
                "Failed to send worker-start-failure alert from worker watcher",
                exc_info=True,
            )
