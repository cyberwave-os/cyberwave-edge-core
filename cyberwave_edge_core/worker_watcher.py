"""File-change watcher for the workers directory.

Uses a polling loop (not inotify) for cross-platform compatibility and
Docker bind-mount reliability.  Runs as part of the existing
``run_runtime_loop()`` reconciliation cycle.

Every call to ``reconcile_worker_files()`` checks whether the set of
``*.py`` files in the workers directory has changed since the last call.
If it has, it calls ``model_manager.ensure_models()`` for any new model
requirements and then restarts the worker container.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .model_manager import ModelManager
    from .worker_manager import WorkerManager

logger = logging.getLogger(__name__)


class WorkerWatcher:
    """Detect changes to the workers directory and trigger container restarts.

    The watcher is stateful: it remembers the last-known directory hash
    across calls to ``reconcile_worker_files()``.  The first call always
    establishes the baseline without triggering a restart.
    """

    def __init__(
        self,
        workers_dir: Path,
        worker_manager: "WorkerManager",
        model_manager: "ModelManager",
        *,
        on_restart: Optional[Callable[[], None]] = None,
    ) -> None:
        self._workers_dir = workers_dir
        self._worker_manager = worker_manager
        self._model_manager = model_manager
        self._on_restart = on_restart
        self._last_hash: Optional[str] = None

    def reconcile_worker_files(self) -> bool:
        """Check for worker file changes and restart container if needed.

        Returns True when a restart was triggered, False otherwise.
        """
        current_hash = self._compute_directory_hash()

        if self._last_hash is None:
            self._last_hash = current_hash
            logger.debug("Worker watcher: baseline established (hash=%s)", current_hash[:12])
            return False

        if current_hash == self._last_hash:
            return False

        logger.info(
            "Worker files changed (hash %s → %s); triggering worker container restart",
            self._last_hash[:12],
            current_hash[:12],
        )
        self._last_hash = current_hash

        self._ensure_models()
        self._worker_manager.restart()

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
            model_ids = self._model_manager.scan_worker_model_requirements(self._workers_dir)
            if not model_ids:
                return
            logger.info("Ensuring models before worker restart: %s", model_ids)
            self._model_manager.ensure_models(model_ids)
        except Exception as exc:
            logger.warning(
                "Model pre-download before worker restart failed (will continue): %s", exc
            )
