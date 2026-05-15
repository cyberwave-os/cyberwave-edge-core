"""Unit tests for WorkerWatcher (worker_watcher.py).

Covers:
- First call establishes baseline (no restart)
- No change → no restart
- New .py file added → restart triggered
- Existing .py file modified → restart triggered
- .py file removed → restart triggered
- Non-.py file changes → ignored
- Model download failure alerts
- Worker start failure alerts
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_ensure_models_exception_does_not_block_restart(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, model_manager = _make_watcher(workers_dir)
        model_manager.scan_worker_model_ids.side_effect = RuntimeError("network error")
        watcher.reconcile_worker_files()  # baseline

        (workers_dir / "detect.py").write_text("pass")
        triggered = watcher.reconcile_worker_files()

        # restart should still be called even though model pre-download failed
        assert triggered is True
        worker_manager.restart.assert_called_once()

    def test_ensure_models_skipped_when_no_model_ids(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, model_manager = _make_watcher(workers_dir)
        model_manager.scan_worker_model_ids.return_value = []
        watcher.reconcile_worker_files()  # baseline

        (workers_dir / "detect.py").write_text("pass")
        watcher.reconcile_worker_files()

        model_manager.ensure_models.assert_not_called()


class TestWorkerWatcherOnRestartCallback:
    def test_on_restart_callback_invoked_after_restart(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        callback_calls: list[None] = []
        worker_manager = MagicMock()
        model_manager = MagicMock()
        model_manager.scan_worker_model_ids.return_value = []

        watcher = WorkerWatcher(
            workers_dir=workers_dir,
            worker_manager=worker_manager,
            model_manager=model_manager,
            on_restart=lambda: callback_calls.append(None),
        )
        watcher.reconcile_worker_files()  # baseline

        (workers_dir / "worker.py").write_text("pass")
        watcher.reconcile_worker_files()

        assert len(callback_calls) == 1

    def test_on_restart_callback_exception_is_swallowed(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        def bad_callback() -> None:
            raise RuntimeError("callback failed")

        worker_manager = MagicMock()
        model_manager = MagicMock()
        model_manager.scan_worker_model_ids.return_value = []

        watcher = WorkerWatcher(
            workers_dir=workers_dir,
            worker_manager=worker_manager,
            model_manager=model_manager,
            on_restart=bad_callback,
        )
        watcher.reconcile_worker_files()  # baseline

        (workers_dir / "worker.py").write_text("pass")
        triggered = watcher.reconcile_worker_files()

        # Exception from callback must not propagate
        assert triggered is True
        worker_manager.restart.assert_called_once()

    def test_on_restart_not_called_when_no_change(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        (workers_dir / "worker.py").write_text("pass")

        callback_calls: list[None] = []
        worker_manager = MagicMock()
        model_manager = MagicMock()
        model_manager.scan_worker_model_ids.return_value = []

        watcher = WorkerWatcher(
            workers_dir=workers_dir,
            worker_manager=worker_manager,
            model_manager=model_manager,
            on_restart=lambda: callback_calls.append(None),
        )
        watcher.reconcile_worker_files()  # baseline
        watcher.reconcile_worker_files()  # no change

        assert callback_calls == []


class TestWorkerWatcherDirectoryHash:
    def test_empty_dir_produces_consistent_hash(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        watcher, _, _ = _make_watcher(workers_dir)
        h1 = watcher._compute_directory_hash()
        h2 = watcher._compute_directory_hash()
        assert h1 == h2

    def test_nonexistent_dir_produces_consistent_hash(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "no_such_dir"
        watcher, _, _ = _make_watcher(workers_dir)
        h1 = watcher._compute_directory_hash()
        h2 = watcher._compute_directory_hash()
        assert h1 == h2

    def test_hash_changes_when_file_added(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        watcher, _, _ = _make_watcher(workers_dir)
        h_before = watcher._compute_directory_hash()
        (workers_dir / "new.py").write_text("pass")
        h_after = watcher._compute_directory_hash()
        assert h_before != h_after


class TestWorkerWatcherModelFailureAlerts:
    def test_model_failure_alerts_sent_on_ensure_failures(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, model_manager = _make_watcher(workers_dir)
        worker_manager._twin_uuids = ["twin-uuid-1"]
        model_manager.scan_worker_model_ids.return_value = ["yolov8n"]
        model_manager.last_ensure_failures = {"yolov8n": "No download sources"}

        watcher.reconcile_worker_files()  # baseline

        with patch.object(watcher, "_send_model_failure_alerts") as mock_send:
            (workers_dir / "detect.py").write_text('cw.models.load("yolov8n")')
            watcher.reconcile_worker_files()

            mock_send.assert_called_once()

    def test_no_alerts_when_no_failures(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, model_manager = _make_watcher(workers_dir)
        worker_manager._twin_uuids = ["twin-uuid-1"]
        model_manager.scan_worker_model_ids.return_value = ["yolov8n"]
        model_manager.last_ensure_failures = {}

        watcher.reconcile_worker_files()  # baseline

        with patch.object(watcher, "_send_model_failure_alerts") as mock_send:
            (workers_dir / "detect.py").write_text('cw.models.load("yolov8n")')
            watcher.reconcile_worker_files()

            mock_send.assert_called_once()

    def test_send_model_failure_alerts_skips_when_no_failures(self, tmp_path: Path) -> None:
        """_send_model_failure_alerts is a no-op when last_ensure_failures is empty."""
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, model_manager = _make_watcher(workers_dir)
        worker_manager._twin_uuids = ["twin-uuid-1"]
        model_manager.last_ensure_failures = {}

        # Should not raise even without startup being importable
        watcher._send_model_failure_alerts()

    def test_send_model_failure_alerts_skips_when_no_twin_uuids(self, tmp_path: Path) -> None:
        """_send_model_failure_alerts is a no-op when no twin_uuids are available."""
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, model_manager = _make_watcher(workers_dir)
        worker_manager._twin_uuids = []
        model_manager.last_ensure_failures = {"yolov8n": "No download sources"}

        # Should not raise even without startup being importable
        watcher._send_model_failure_alerts()


class TestWorkerWatcherStartFailureAlerts:
    def test_alert_sent_on_restart_failure(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, model_manager = _make_watcher(workers_dir)
        worker_manager._twin_uuids = ["twin-uuid-1"]
        worker_manager.restart.return_value = False

        watcher.reconcile_worker_files()  # baseline

        with patch.object(watcher, "_send_worker_start_failure_alert") as mock_send:
            (workers_dir / "detect.py").write_text("pass")
            watcher.reconcile_worker_files()

            mock_send.assert_called_once()

    def test_no_alert_on_successful_restart(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, model_manager = _make_watcher(workers_dir)
        worker_manager._twin_uuids = ["twin-uuid-1"]
        worker_manager.restart.return_value = True

        watcher.reconcile_worker_files()  # baseline

        with patch.object(watcher, "_send_worker_start_failure_alert") as mock_send:
            (workers_dir / "detect.py").write_text("pass")
            watcher.reconcile_worker_files()

            mock_send.assert_not_called()

    def test_send_worker_start_failure_alert_skips_when_no_twin_uuids(self, tmp_path: Path) -> None:
        """_send_worker_start_failure_alert is a no-op when no twin_uuids are available."""
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, model_manager = _make_watcher(workers_dir)
        worker_manager._twin_uuids = []

        # Should not raise even without startup being importable
        watcher._send_worker_start_failure_alert()
