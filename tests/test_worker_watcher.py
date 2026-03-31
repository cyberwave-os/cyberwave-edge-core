"""Unit tests for WorkerWatcher (worker_watcher.py).

Covers:
- First call establishes baseline (no restart)
- No change → no restart
- New .py file added → restart triggered
- Existing .py file modified → restart triggered
- .py file removed → restart triggered
- Non-.py file changes → ignored
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

from cyberwave_edge_core.worker_watcher import WorkerWatcher


def _make_watcher(workers_dir: Path) -> tuple[WorkerWatcher, MagicMock, MagicMock]:
    worker_manager = MagicMock()
    model_manager = MagicMock()
    model_manager.scan_worker_model_ids.return_value = []
    watcher = WorkerWatcher(
        workers_dir=workers_dir,
        worker_manager=worker_manager,
        model_manager=model_manager,
    )
    return watcher, worker_manager, model_manager


class TestWorkerWatcherBaseline:
    def test_first_call_establishes_baseline(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        (workers_dir / "detect.py").write_text("pass")

        watcher, worker_manager, _ = _make_watcher(workers_dir)
        triggered = watcher.reconcile_worker_files()

        assert triggered is False
        worker_manager.restart.assert_not_called()

    def test_no_change_does_not_restart(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        (workers_dir / "detect.py").write_text("pass")

        watcher, worker_manager, _ = _make_watcher(workers_dir)
        watcher.reconcile_worker_files()  # establish baseline
        triggered = watcher.reconcile_worker_files()

        assert triggered is False
        worker_manager.restart.assert_not_called()


class TestWorkerWatcherDetectsAdd:
    def test_new_py_file_triggers_restart(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, _ = _make_watcher(workers_dir)
        watcher.reconcile_worker_files()  # baseline: empty

        (workers_dir / "new_worker.py").write_text("pass")
        triggered = watcher.reconcile_worker_files()

        assert triggered is True
        worker_manager.restart.assert_called_once()


class TestWorkerWatcherDetectsModify:
    def test_modified_py_file_triggers_restart(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        py_file = workers_dir / "detect.py"
        py_file.write_text("version = 1")

        watcher, worker_manager, _ = _make_watcher(workers_dir)
        watcher.reconcile_worker_files()  # baseline

        # Wait briefly then update mtime
        time.sleep(0.05)
        py_file.write_text("version = 2")

        triggered = watcher.reconcile_worker_files()
        assert triggered is True
        worker_manager.restart.assert_called_once()


class TestWorkerWatcherDetectsRemove:
    def test_removed_py_file_triggers_restart(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        py_file = workers_dir / "detect.py"
        py_file.write_text("pass")

        watcher, worker_manager, _ = _make_watcher(workers_dir)
        watcher.reconcile_worker_files()  # baseline

        py_file.unlink()
        triggered = watcher.reconcile_worker_files()

        assert triggered is True
        worker_manager.restart.assert_called_once()


class TestWorkerWatcherIgnoresNonPy:
    def test_non_py_file_does_not_trigger_restart(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        (workers_dir / "detect.py").write_text("pass")

        watcher, worker_manager, _ = _make_watcher(workers_dir)
        watcher.reconcile_worker_files()  # baseline

        (workers_dir / "readme.md").write_text("docs")
        (workers_dir / "config.json").write_text("{}")
        triggered = watcher.reconcile_worker_files()

        assert triggered is False
        worker_manager.restart.assert_not_called()


class TestWorkerWatcherEnsuresModels:
    def test_ensure_models_called_before_restart(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, model_manager = _make_watcher(workers_dir)
        model_manager.scan_worker_model_ids.return_value = ["yolov8n"]
        watcher.reconcile_worker_files()  # baseline

        (workers_dir / "detect.py").write_text('cw.models.load("yolov8n")')
        watcher.reconcile_worker_files()

        model_manager.scan_worker_model_ids.assert_called()
        model_manager.ensure_models.assert_called_with(["yolov8n"])
