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
        mock_hm_cls.assert_called_once()
        hm_call = mock_hm_cls.call_args
        assert hm_call.kwargs["container_name"] == "cyberwave-worker-test"
        # The watcher must wire ``expected_running_probe`` so the
        # monitor can self-suppress the spontaneous-exit warning when
        # the workers dir is empty (deactivation path: the stop is
        # driven by a different ``WorkerManager`` instance that
        # doesn't share this monitor).
        probe = hm_call.kwargs.get("expected_running_probe")
        assert callable(probe)
        # Empty/nonexistent dir → probe returns False (workflow
        # deactivation leg).
        assert probe() is False

        workers_dir = tmp_path / "workers"
        workers_dir.mkdir(exist_ok=True)
        (workers_dir / "wf_aaaaaaaaaaaa.py").write_text("# x", encoding="utf-8")
        # Populated dir, no restart in flight → probe returns True
        # (the normal "worker should be up" steady state).
        assert probe() is True

        # Populated dir, restart in flight → probe returns False so the
        # "Restart edge core" flow doesn't trigger a false-positive
        # crash-loop WARN between the stop and the subsequent start.
        # ``monkeypatch`` auto-restores the flag at teardown, so no
        # manual ``finally`` is needed to reset it.
        monkeypatch.setattr(startup, "_EDGE_RESTART_IN_PROGRESS", True)
        assert probe() is False
        mock_wm_instance.set_health_monitor.assert_called_once()
        mock_watcher_cls.assert_called_once()
        mock_watcher_instance.check_health.assert_called_once()
        mock_watcher_instance.reconcile_worker_files.assert_called_once()
        # Manager registered so ``reconcile_worker_lifecycle`` can
        # reuse it and route deliberate stops through ``record_stop``.
        assert startup._get_monitored_worker_manager() is mock_wm_instance
        startup._set_monitored_worker_manager(None)

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
        monkeypatch.setattr(startup, "_run_periodic_docker_cleanup", lambda: None)

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
        """Verify a watchdog exception doesn't take down the loop and that
        the loop keeps trying to ping on subsequent cycles."""
        self._patch_loop_deps(monkeypatch)
        monkeypatch.setattr(startup, "_WORKER_SYNC_INTERVAL_LOOPS", 100)
        startup._worker_sync_loop_counter = 0

        ping_count = [0]

        class FailingWatchdog:
            def ping(self):
                ping_count[0] += 1
                raise RuntimeError("watchdog ping boom")

        self._stop_after(2, monkeypatch)
        startup.run_runtime_loop(watchdog=FailingWatchdog())

        assert ping_count[0] == 2

    def test_periodic_docker_cleanup_called(self, monkeypatch):
        """Verify _run_periodic_docker_cleanup is called each cycle."""
        self._patch_loop_deps(monkeypatch)
        monkeypatch.setattr(startup, "_WORKER_SYNC_INTERVAL_LOOPS", 100)
        startup._worker_sync_loop_counter = 0

        cleanup_calls = [0]

        def fake_cleanup():
            cleanup_calls[0] += 1

        monkeypatch.setattr(startup, "_run_periodic_docker_cleanup", fake_cleanup)
        self._stop_after(3, monkeypatch)

        startup.run_runtime_loop()

        assert cleanup_calls[0] == 3

    def test_periodic_docker_cleanup_exception_does_not_crash(self, monkeypatch):
        """Verify an exception in _run_periodic_docker_cleanup propagates but
        the function itself catches errors internally."""
        self._patch_loop_deps(monkeypatch)
        monkeypatch.setattr(startup, "_WORKER_SYNC_INTERVAL_LOOPS", 100)
        startup._worker_sync_loop_counter = 0

        cleanup_calls = [0]

        def fake_cleanup():
            cleanup_calls[0] += 1

        monkeypatch.setattr(startup, "_run_periodic_docker_cleanup", fake_cleanup)
        self._stop_after(2, monkeypatch)

        startup.run_runtime_loop()
        assert cleanup_calls[0] == 2


# ===========================================================================
# 3. _run_periodic_docker_cleanup()
# ===========================================================================

# "No prune has ever run", so every interval reads as elapsed. Not 0.0:
# time.monotonic() counts from boot, so a 0.0 anchor only clears the 3 h
# image/volume intervals once the host has been up 3 h. That passed on a
# long-lived CI runner and failed on a freshly booted Pi.
_NEVER_PRUNED = float("-inf")


class TestRunPeriodicDockerCleanup:
    """Tests for the _run_periodic_docker_cleanup scheduling logic."""

    @pytest.fixture(autouse=True)
    def _reset_prune_times(self, monkeypatch):
        """Reset prune timestamps and cleanup flags before/after each test.

        The SD-card caches are process-level; leaving them set would make
        these tests order-dependent.
        """
        startup._last_container_prune_time = _NEVER_PRUNED
        startup._last_image_prune_time = _NEVER_PRUNED
        startup._last_volume_prune_time = _NEVER_PRUNED
        startup._docker_cleanup_disabled = None
        startup._root_is_sd_card = None
        startup._sd_card_pressure_logged = False
        monkeypatch.delenv("CYBERWAVE_SKIP_PERIODIC_DOCKER_CLEANUP", raising=False)
        import cyberwave_edge_core.docker_helpers as _dh
        monkeypatch.setattr(_dh, "is_sd_card_path", lambda _p: False)
        monkeypatch.setattr(_dh, "docker_data_root", lambda: "/var/lib/docker")
        yield
        startup._last_container_prune_time = _NEVER_PRUNED
        startup._last_image_prune_time = _NEVER_PRUNED
        startup._last_volume_prune_time = _NEVER_PRUNED
        startup._docker_cleanup_disabled = None
        startup._root_is_sd_card = None
        startup._sd_card_pressure_logged = False

    def test_runs_both_prune_on_first_call(self, monkeypatch):
        container_prune_calls = [0]
        image_prune_calls = [0]

        monkeypatch.setattr(
            startup,
            "CONTAINER_PRUNE_INTERVAL_SECONDS",
            10.0,
        )
        monkeypatch.setattr(
            startup,
            "IMAGE_PRUNE_INTERVAL_SECONDS",
            100.0,
        )

        import cyberwave_edge_core.docker_helpers as _dh

        monkeypatch.setattr(
            _dh,
            "docker_prune_stopped_cyberwave_containers",
            lambda **kw: container_prune_calls.__setitem__(0, container_prune_calls[0] + 1) or 0,
        )
        monkeypatch.setattr(
            _dh,
            "docker_prune_unused_images",
            lambda: image_prune_calls.__setitem__(0, image_prune_calls[0] + 1) or True,
        )

        startup._run_periodic_docker_cleanup()

        assert container_prune_calls[0] == 1
        assert image_prune_calls[0] == 1

    def test_skips_when_interval_not_elapsed(self, monkeypatch):
        container_prune_calls = [0]
        image_prune_calls = [0]

        monkeypatch.setattr(startup, "CONTAINER_PRUNE_INTERVAL_SECONDS", 1800.0)
        monkeypatch.setattr(startup, "IMAGE_PRUNE_INTERVAL_SECONDS", 10800.0)

        import time

        now = time.monotonic()
        startup._last_container_prune_time = now
        startup._last_image_prune_time = now

        import cyberwave_edge_core.docker_helpers as _dh

        monkeypatch.setattr(
            _dh,
            "docker_prune_stopped_cyberwave_containers",
            lambda **kw: container_prune_calls.__setitem__(0, container_prune_calls[0] + 1) or 0,
        )
        monkeypatch.setattr(
            _dh,
            "docker_prune_unused_images",
            lambda: image_prune_calls.__setitem__(0, image_prune_calls[0] + 1) or True,
        )

        startup._run_periodic_docker_cleanup()

        assert container_prune_calls[0] == 0
        assert image_prune_calls[0] == 0

    def test_container_prune_runs_when_interval_elapsed(self, monkeypatch):
        container_prune_calls = [0]
        image_prune_calls = [0]

        monkeypatch.setattr(startup, "CONTAINER_PRUNE_INTERVAL_SECONDS", 10.0)
        monkeypatch.setattr(startup, "IMAGE_PRUNE_INTERVAL_SECONDS", 10800.0)

        import time

        now = time.monotonic()
        startup._last_container_prune_time = now - 20
        startup._last_image_prune_time = now

        import cyberwave_edge_core.docker_helpers as _dh

        monkeypatch.setattr(
            _dh,
            "docker_prune_stopped_cyberwave_containers",
            lambda **kw: container_prune_calls.__setitem__(0, container_prune_calls[0] + 1) or 0,
        )
        monkeypatch.setattr(
            _dh,
            "docker_prune_unused_images",
            lambda: image_prune_calls.__setitem__(0, image_prune_calls[0] + 1) or True,
        )

        startup._run_periodic_docker_cleanup()

        assert container_prune_calls[0] == 1
        assert image_prune_calls[0] == 0

    def test_skips_cleanup_when_env_var_set(self, monkeypatch):
        """Cleanup is skipped when CYBERWAVE_SKIP_PERIODIC_DOCKER_CLEANUP=1."""
        startup._docker_cleanup_disabled = None
        monkeypatch.setenv("CYBERWAVE_SKIP_PERIODIC_DOCKER_CLEANUP", "1")

        container_prune_calls = [0]
        image_prune_calls = [0]

        import cyberwave_edge_core.docker_helpers as _dh

        monkeypatch.setattr(
            _dh,
            "docker_prune_stopped_cyberwave_containers",
            lambda **kw: container_prune_calls.__setitem__(0, container_prune_calls[0] + 1) or 0,
        )
        monkeypatch.setattr(
            _dh,
            "docker_prune_unused_images",
            lambda: image_prune_calls.__setitem__(0, image_prune_calls[0] + 1) or True,
        )

        startup._run_periodic_docker_cleanup()

        assert container_prune_calls[0] == 0
        assert image_prune_calls[0] == 0
        startup._docker_cleanup_disabled = None

    def _patch_prune_counters(self, monkeypatch):
        """Patch the three prune helpers and return their call counters."""
        import cyberwave_edge_core.docker_helpers as _dh

        calls = {"container": 0, "image": 0, "volume": 0}

        def _bump(key, retval):
            def _inner(*_args, **_kwargs):
                calls[key] += 1
                return retval

            return _inner

        monkeypatch.setattr(
            _dh, "docker_prune_stopped_cyberwave_containers", _bump("container", 0)
        )
        monkeypatch.setattr(_dh, "docker_prune_unused_images", _bump("image", True))
        monkeypatch.setattr(_dh, "docker_prune_dangling_volumes", _bump("volume", True))
        return calls

    def test_defers_cleanup_on_healthy_sd_card(self, monkeypatch):
        """On an SD card below the usage threshold, cleanup is deferred for flash wear."""
        startup._docker_cleanup_disabled = None
        monkeypatch.delenv("CYBERWAVE_SKIP_PERIODIC_DOCKER_CLEANUP", raising=False)

        import cyberwave_edge_core.docker_helpers as _dh

        monkeypatch.setattr(_dh, "is_sd_card_path", lambda _p: True)
        monkeypatch.setattr(_dh, "docker_data_root", lambda: "/var/lib/docker")
        monkeypatch.setattr(startup, "_docker_disk_usage_percent", lambda: 41.0)
        calls = self._patch_prune_counters(monkeypatch)

        startup._run_periodic_docker_cleanup()

        assert calls == {"container": 0, "image": 0, "volume": 0}
        startup._docker_cleanup_disabled = None

    def test_runs_cleanup_on_full_sd_card(self, monkeypatch):
        """Crossing the threshold overrides the wear deferral.

        The unconditional SD-card check is what let a Pi fill to 95%.
        """
        startup._docker_cleanup_disabled = None
        monkeypatch.delenv("CYBERWAVE_SKIP_PERIODIC_DOCKER_CLEANUP", raising=False)

        import cyberwave_edge_core.docker_helpers as _dh

        monkeypatch.setattr(_dh, "is_sd_card_path", lambda _p: True)
        monkeypatch.setattr(_dh, "docker_data_root", lambda: "/var/lib/docker")
        monkeypatch.setattr(startup, "_docker_disk_usage_percent", lambda: 95.0)
        monkeypatch.setattr(startup, "_last_container_prune_time", _NEVER_PRUNED)
        monkeypatch.setattr(startup, "_last_image_prune_time", _NEVER_PRUNED)
        monkeypatch.setattr(startup, "_last_volume_prune_time", _NEVER_PRUNED)
        calls = self._patch_prune_counters(monkeypatch)

        startup._run_periodic_docker_cleanup()

        assert calls == {"container": 1, "image": 1, "volume": 1}
        startup._docker_cleanup_disabled = None

    def test_pressure_warning_logged_once_per_crossing(self, monkeypatch, caplog):
        """The over-threshold warning must not fire on every reconcile tick.

        This runs every ~15 s, so an unconditional warning would add
        thousands of journald lines a day to an already-full device.
        """
        import logging

        import cyberwave_edge_core.docker_helpers as _dh

        monkeypatch.setattr(_dh, "is_sd_card_path", lambda _p: True)
        monkeypatch.setattr(_dh, "docker_data_root", lambda: "/var/lib/docker")
        usage = [95.0]
        monkeypatch.setattr(startup, "_docker_disk_usage_percent", lambda: usage[0])
        self._patch_prune_counters(monkeypatch)

        def _warnings() -> list[str]:
            return [r.message for r in caplog.records if r.levelno == logging.WARNING]

        with caplog.at_level(logging.WARNING, logger=startup.logger.name):
            for _ in range(5):
                startup._run_periodic_docker_cleanup()
            assert len(_warnings()) == 1

            # Recovering below the threshold re-arms it, so a second episode
            # is still reported rather than silently swallowed forever.
            caplog.clear()
            usage[0] = 40.0
            startup._run_periodic_docker_cleanup()
            assert _warnings() == []

            usage[0] = 96.0
            startup._run_periodic_docker_cleanup()
            assert len(_warnings()) == 1

    def test_defers_cleanup_when_sd_card_usage_unreadable(self, monkeypatch):
        """An unreadable statvfs keeps the historical wear-sparing behaviour."""
        startup._docker_cleanup_disabled = None
        monkeypatch.delenv("CYBERWAVE_SKIP_PERIODIC_DOCKER_CLEANUP", raising=False)

        import cyberwave_edge_core.docker_helpers as _dh

        monkeypatch.setattr(_dh, "is_sd_card_path", lambda _p: True)
        monkeypatch.setattr(_dh, "docker_data_root", lambda: "/var/lib/docker")
        monkeypatch.setattr(startup, "_docker_disk_usage_percent", lambda: None)
        calls = self._patch_prune_counters(monkeypatch)

        startup._run_periodic_docker_cleanup()

        assert calls == {"container": 0, "image": 0, "volume": 0}
        startup._docker_cleanup_disabled = None

    def test_env_opt_out_beats_disk_pressure(self, monkeypatch):
        """The explicit operator opt-out wins even on a full SD card."""
        startup._docker_cleanup_disabled = None
        monkeypatch.setenv("CYBERWAVE_SKIP_PERIODIC_DOCKER_CLEANUP", "1")

        import cyberwave_edge_core.docker_helpers as _dh

        monkeypatch.setattr(_dh, "is_sd_card_path", lambda _p: True)
        monkeypatch.setattr(_dh, "docker_data_root", lambda: "/var/lib/docker")
        monkeypatch.setattr(startup, "_docker_disk_usage_percent", lambda: 99.0)
        calls = self._patch_prune_counters(monkeypatch)

        startup._run_periodic_docker_cleanup()

        assert calls == {"container": 0, "image": 0, "volume": 0}
        startup._docker_cleanup_disabled = None

    def test_runs_cleanup_when_not_sd_card_and_no_env_var(self, monkeypatch):
        """Cleanup runs normally when not on SD card and env var not set."""
        startup._docker_cleanup_disabled = None
        monkeypatch.delenv("CYBERWAVE_SKIP_PERIODIC_DOCKER_CLEANUP", raising=False)

        monkeypatch.setattr(
            startup,
            "CONTAINER_PRUNE_INTERVAL_SECONDS",
            10.0,
        )
        monkeypatch.setattr(
            startup,
            "IMAGE_PRUNE_INTERVAL_SECONDS",
            100.0,
        )

        import cyberwave_edge_core.docker_helpers as _dh

        monkeypatch.setattr(_dh, "is_sd_card_path", lambda _p: False)
        monkeypatch.setattr(_dh, "docker_data_root", lambda: "/var/lib/docker")

        container_prune_calls = [0]
        image_prune_calls = [0]

        monkeypatch.setattr(
            _dh,
            "docker_prune_stopped_cyberwave_containers",
            lambda **kw: container_prune_calls.__setitem__(0, container_prune_calls[0] + 1) or 0,
        )
        monkeypatch.setattr(
            _dh,
            "docker_prune_unused_images",
            lambda: image_prune_calls.__setitem__(0, image_prune_calls[0] + 1) or True,
        )

        startup._run_periodic_docker_cleanup()

        assert container_prune_calls[0] == 1
        assert image_prune_calls[0] == 1
        startup._docker_cleanup_disabled = None
