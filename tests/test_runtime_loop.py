"""Tests for _reconcile_worker_watcher() and run_runtime_loop() in startup.py.

Covers:
  1. _reconcile_worker_watcher() — lazy creation, reuse, early returns, method calls
  2. run_runtime_loop() — periodic sync cadence, exception resilience
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import cyberwave_edge_core.startup as startup

# ===========================================================================
# 1. _reconcile_worker_watcher()
# ===========================================================================


class TestReconcileWorkerWatcher:
    """Tests for the _reconcile_worker_watcher() glue function."""

    def test_returns_unchanged_when_no_environment_uuid(self, monkeypatch):
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: None)
        sentinel = object()
        assert startup._reconcile_worker_watcher(sentinel) is sentinel

    def test_returns_unchanged_when_no_token(self, monkeypatch):
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "load_token", lambda: None)
        sentinel = object()
        assert startup._reconcile_worker_watcher(sentinel) is sentinel

    def test_creates_watcher_on_first_call(self, monkeypatch, tmp_path):
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "get_runtime_env_var", lambda *a, **kw: None)

        mock_watcher_cls = MagicMock()
        mock_watcher_instance = MagicMock()
        mock_watcher_cls.return_value = mock_watcher_instance

        mock_wm_cls = MagicMock()
        mock_wm_instance = MagicMock()
        mock_wm_instance.container_name = "cyberwave-worker-test"
        mock_wm_cls.return_value = mock_wm_instance

        mock_hm_cls = MagicMock()
        mock_mm_cls = MagicMock()

        with (
            patch("cyberwave_edge_core.worker_watcher.WorkerWatcher", mock_watcher_cls),
            patch("cyberwave_edge_core.worker_manager.WorkerManager", mock_wm_cls),
            patch("cyberwave_edge_core.worker_health.WorkerHealthMonitor", mock_hm_cls),
            patch("cyberwave_edge_core.model_manager.ModelManager", mock_mm_cls),
        ):
            result = startup._reconcile_worker_watcher(None)

        assert result is mock_watcher_instance
        mock_wm_cls.assert_called_once()
        mock_hm_cls.assert_called_once_with(container_name="cyberwave-worker-test")
        mock_wm_instance.set_health_monitor.assert_called_once()
        mock_watcher_cls.assert_called_once()
        mock_watcher_instance.check_health.assert_called_once()
        mock_watcher_instance.reconcile_worker_files.assert_called_once()

    def test_reuses_existing_watcher(self, monkeypatch, tmp_path):
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

        existing = MagicMock()
        result = startup._reconcile_worker_watcher(existing)

        assert result is existing
        existing.check_health.assert_called_once()
        existing.reconcile_worker_files.assert_called_once()

    def test_calls_check_health_and_reconcile_every_cycle(self, monkeypatch, tmp_path):
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

        existing = MagicMock()
        startup._reconcile_worker_watcher(existing)
        startup._reconcile_worker_watcher(existing)
        startup._reconcile_worker_watcher(existing)

        assert existing.check_health.call_count == 3
        assert existing.reconcile_worker_files.call_count == 3


# ===========================================================================
# 2. run_runtime_loop() — sync cadence
# ===========================================================================


class TestRuntimeLoopSyncCadence:
    """Verify that run_runtime_loop() calls reconcile_worker_sync at the right cadence."""

    def _patch_loop_deps(self, monkeypatch):
        """Stub every reconcile function that the loop calls (except worker sync)."""
        monkeypatch.setattr(
            startup,
            "reconcile_driver_log_streams",
            lambda: 0,
        )
        monkeypatch.setattr(
            startup,
            "reconcile_driver_restart_failures",
            lambda: {"inspected": 0, "flapping": 0, "stopped": 0, "alerts_sent": 0},
        )
        monkeypatch.setattr(
            startup,
            "reconcile_twin_json_file_sync",
            lambda: {"tracked": 0, "changed": 0, "synced": 0},
        )
        monkeypatch.setattr(
            startup,
            "_reconcile_worker_watcher",
            lambda w: w,
        )
        monkeypatch.setattr(
            startup,
            "ensure_edge_command_subscription",
            lambda: None,
        )
        monkeypatch.setattr(
            startup,
            "ensure_twin_command_subscriptions",
            lambda: None,
        )
        monkeypatch.setattr(startup, "_CONTAINER_LOG_THREADS", {})
        monkeypatch.setattr(startup, "_graceful_shutdown", lambda w: None)

    def _stop_after(self, n, monkeypatch):
        """Make the shutdown_event stop the loop after *n* wait() calls."""
        real_event = startup.shutdown_event
        call_count = [0]

        def fake_wait(timeout=None):
            call_count[0] += 1
            if call_count[0] >= n:
                real_event.set()
            return real_event.is_set()

        monkeypatch.setattr(real_event, "wait", fake_wait)

    @pytest.fixture(autouse=True)
    def _reset_shutdown(self):
        """Clear shutdown_event before/after every test."""
        startup.shutdown_event.clear()
        yield
        startup.shutdown_event.clear()

    def test_sync_runs_at_interval(self, monkeypatch):
        self._patch_loop_deps(monkeypatch)
        monkeypatch.setattr(startup, "_WORKER_SYNC_INTERVAL_LOOPS", 3)
        startup._worker_sync_loop_counter = 0

        sync_calls: list[int] = []
        iteration = [0]

        def fake_sync():
            sync_calls.append(iteration[0])
            return {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}

        monkeypatch.setattr(startup, "reconcile_worker_sync", fake_sync)
        self._stop_after(7, monkeypatch)

        startup.run_runtime_loop()

        assert len(sync_calls) == 2

    def test_counter_resets_after_sync(self, monkeypatch):
        self._patch_loop_deps(monkeypatch)
        monkeypatch.setattr(startup, "_WORKER_SYNC_INTERVAL_LOOPS", 2)
        startup._worker_sync_loop_counter = 0

        sync_count = [0]

        def fake_sync():
            sync_count[0] += 1
            return {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}

        monkeypatch.setattr(startup, "reconcile_worker_sync", fake_sync)
        self._stop_after(4, monkeypatch)

        startup.run_runtime_loop()

        assert sync_count[0] == 2

    def test_sync_exception_does_not_crash_loop(self, monkeypatch):
        self._patch_loop_deps(monkeypatch)
        monkeypatch.setattr(startup, "_WORKER_SYNC_INTERVAL_LOOPS", 1)
        startup._worker_sync_loop_counter = 0

        def failing_sync():
            raise RuntimeError("sync boom")

        monkeypatch.setattr(startup, "reconcile_worker_sync", failing_sync)
        self._stop_after(3, monkeypatch)

        startup.run_runtime_loop()

    def test_watcher_exception_does_not_crash_loop(self, monkeypatch):
        self._patch_loop_deps(monkeypatch)
        monkeypatch.setattr(startup, "_WORKER_SYNC_INTERVAL_LOOPS", 100)
        startup._worker_sync_loop_counter = 0

        def failing_watcher(w):
            raise RuntimeError("watcher boom")

        monkeypatch.setattr(startup, "_reconcile_worker_watcher", failing_watcher)
        monkeypatch.setattr(
            startup,
            "reconcile_worker_sync",
            lambda: {"written": 0, "removed": 0, "unchanged": 0, "errors": 0},
        )
        self._stop_after(2, monkeypatch)

        startup.run_runtime_loop()

    def test_watchdog_pinged_each_cycle(self, monkeypatch):
        """Verify the watchdog is pinged every reconcile cycle."""
        self._patch_loop_deps(monkeypatch)
        monkeypatch.setattr(startup, "_WORKER_SYNC_INTERVAL_LOOPS", 100)
        startup._worker_sync_loop_counter = 0

        ping_count = [0]

        class FakeWatchdog:
            def ping(self):
                ping_count[0] += 1

        self._stop_after(3, monkeypatch)
        startup.run_runtime_loop(watchdog=FakeWatchdog())

        assert ping_count[0] == 3

    def test_resource_monitor_checked_each_cycle(self, monkeypatch):
        """Verify the resource monitor is checked every reconcile cycle."""
        self._patch_loop_deps(monkeypatch)
        monkeypatch.setattr(startup, "_WORKER_SYNC_INTERVAL_LOOPS", 100)
        startup._worker_sync_loop_counter = 0

        check_count = [0]

        class FakeResourceMonitor:
            def check(self):
                check_count[0] += 1
                return None

        self._stop_after(3, monkeypatch)
        startup.run_runtime_loop(resource_monitor=FakeResourceMonitor())

        assert check_count[0] == 3

    def test_watchdog_exception_does_not_crash_loop(self, monkeypatch):
        """Verify a watchdog exception doesn't take down the loop."""
        self._patch_loop_deps(monkeypatch)
        monkeypatch.setattr(startup, "_WORKER_SYNC_INTERVAL_LOOPS", 100)
        startup._worker_sync_loop_counter = 0

        class FailingWatchdog:
            def ping(self):
                raise RuntimeError("watchdog ping boom")

        self._stop_after(2, monkeypatch)
        startup.run_runtime_loop(watchdog=FailingWatchdog())
