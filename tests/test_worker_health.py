"""Unit tests for WorkerHealthMonitor (worker_health.py).

Covers:
- check() returns correct healthy/unhealthy state based on container status
- record_restart() increments counters and updates is_healthy
- Circuit-breaker trips after max_restarts_in_window restarts
- Circuit-breaker resets when window drains
- is_restart_allowed() respects circuit-breaker state
- Spontaneous exit detection (running → exited)
- ``expected_running_probe`` downgrades the spontaneous-exit warning to
  INFO when the workers dir has been emptied — the deactivation flow
  drives ``reconcile_worker_lifecycle.stop()`` through a *different*
  ``WorkerManager`` instance, so the watcher's long-lived monitor can't
  catch the stop via ``record_stop`` and the probe is the suppression
  channel for that cross-instance case.
- ``record_stop`` pre-empts the next spontaneous-exit detection for
  same-instance callers (``WorkerManager.stop`` is the canonical
  wirer-upper), and resets the uptime baseline so the next start
  re-anchors cleanly.
- Readiness probe integration
- reset() clears all state
- RestartRecord fields are populated correctly
- ResourceLimits.to_docker_args() / to_env_args()
- WorkerManager.restart() passes reason to health monitor
- WorkerManager.restart() respects circuit-breaker (blocked when tripped)
- WorkerWatcher cool-down between successive restarts
- WorkerWatcher passes reason to WorkerManager.restart()
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import cyberwave_edge_core.worker_manager as wm_module
from cyberwave_edge_core.worker_health import WorkerHealthMonitor
from cyberwave_edge_core.worker_manager import ResourceLimits, WorkerManager
from cyberwave_edge_core.worker_watcher import WorkerWatcher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor() -> WorkerHealthMonitor:
    return WorkerHealthMonitor(
        container_name="cyberwave-worker-test",
        restart_window_seconds=60.0,
        max_restarts_in_window=3,
    )


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def worker_manager(tmp_config: Path) -> WorkerManager:
    return WorkerManager(
        config_dir=tmp_config,
        environment_uuid="aabbccdd-eeff-0011-2233-445566778899",
        token="testtoken",
    )


# ---------------------------------------------------------------------------
# WorkerHealthMonitor — basic checks
# ---------------------------------------------------------------------------


class TestWorkerhealthMonitorCheck:
    def test_running_container_is_healthy(self, monitor: WorkerHealthMonitor) -> None:
        state = monitor.check(container_status="running")
        assert state.is_healthy is True
        assert state.container_status == "running"
        assert state.circuit_breaker_tripped is False

    def test_exited_container_is_unhealthy(self, monitor: WorkerHealthMonitor) -> None:
        state = monitor.check(container_status="exited")
        assert state.is_healthy is False

    def test_none_container_is_unhealthy(self, monitor: WorkerHealthMonitor) -> None:
        state = monitor.check(container_status="none")
        assert state.is_healthy is False

    def test_observed_at_is_recent(self, monitor: WorkerHealthMonitor) -> None:
        before = time.time()
        state = monitor.check(container_status="running")
        after = time.time()
        assert before <= state.observed_at <= after

    def test_uptime_none_when_no_start_recorded(self, monitor: WorkerHealthMonitor) -> None:
        state = monitor.check(container_status="running")
        assert state.uptime_seconds is None

    def test_uptime_calculated_after_start_recorded(self, monitor: WorkerHealthMonitor) -> None:
        monitor.record_start()
        time.sleep(0.02)
        state = monitor.check(container_status="running")
        assert state.uptime_seconds is not None
        assert state.uptime_seconds >= 0.01

    def test_restart_count_zero_initially(self, monitor: WorkerHealthMonitor) -> None:
        state = monitor.check(container_status="running")
        assert state.restart_count == 0
        assert state.recent_restarts == 0


# ---------------------------------------------------------------------------
# WorkerHealthMonitor — restart accounting
# ---------------------------------------------------------------------------


class TestWorkerhealthMonitorRestartAccounting:
    def test_record_restart_increments_count(self, monitor: WorkerHealthMonitor) -> None:
        monitor.record_restart(reason="test", success=True)
        state = monitor.check(container_status="running")
        assert state.restart_count == 1

    def test_record_restart_populates_record_fields(self, monitor: WorkerHealthMonitor) -> None:
        before = time.monotonic()  # RestartRecord.timestamp uses now_monotonic()
        monitor.record_restart(reason="worker-files-changed", success=True)
        records = monitor.check(container_status="running").restart_records
        assert len(records) == 1
        rec = records[0]
        assert rec.reason == "worker-files-changed"
        assert rec.success is True
        assert rec.timestamp >= before

    def test_failed_restart_record(self, monitor: WorkerHealthMonitor) -> None:
        monitor.record_restart(reason="crash-loop", success=False)
        state = monitor.check(container_status="running")
        assert state.restart_records[0].success is False

    def test_multiple_restarts_accumulated(self, monitor: WorkerHealthMonitor) -> None:
        for _ in range(3):
            monitor.record_restart(reason="test", success=True)
        state = monitor.check(container_status="running")
        assert state.restart_count == 3
        assert state.recent_restarts == 3


# ---------------------------------------------------------------------------
# WorkerHealthMonitor — circuit-breaker
# ---------------------------------------------------------------------------


class TestWorkerhealthMonitorCircuitBreaker:
    def test_circuit_breaker_trips_after_max_restarts(self, monitor: WorkerHealthMonitor) -> None:
        for i in range(3):
            monitor.record_restart(reason=f"crash-{i}", success=True)
        state = monitor.check(container_status="running")
        assert state.circuit_breaker_tripped is True
        assert state.is_healthy is False

    def test_circuit_breaker_not_tripped_below_max(self, monitor: WorkerHealthMonitor) -> None:
        for i in range(2):
            monitor.record_restart(reason=f"crash-{i}", success=True)
        state = monitor.check(container_status="running")
        assert state.circuit_breaker_tripped is False

    def test_circuit_breaker_tripped_at_is_set(self, monitor: WorkerHealthMonitor) -> None:
        before = time.monotonic()  # circuit_breaker_tripped_at uses now_monotonic()
        for _ in range(3):
            monitor.record_restart(reason="crash", success=True)
        state = monitor.check(container_status="running")
        assert state.circuit_breaker_tripped_at is not None
        assert state.circuit_breaker_tripped_at >= before

    def test_circuit_breaker_resets_after_window_clears(self) -> None:
        monitor = WorkerHealthMonitor(
            container_name="test",
            restart_window_seconds=0.05,  # 50 ms window for fast test
            max_restarts_in_window=2,
        )
        for _ in range(2):
            monitor.record_restart(reason="crash", success=True)
        state = monitor.check(container_status="running")
        assert state.circuit_breaker_tripped is True

        time.sleep(0.1)  # wait for window to expire

        state = monitor.check(container_status="running")
        assert state.circuit_breaker_tripped is False

    def test_is_restart_allowed_false_when_tripped(self, monitor: WorkerHealthMonitor) -> None:
        for _ in range(3):
            monitor.record_restart(reason="crash", success=True)
        assert monitor.is_restart_allowed() is False

    def test_is_restart_allowed_true_when_not_tripped(self, monitor: WorkerHealthMonitor) -> None:
        monitor.record_restart(reason="ok", success=True)
        assert monitor.is_restart_allowed() is True

    def test_is_restart_allowed_resets_automatically(self) -> None:
        monitor = WorkerHealthMonitor(
            container_name="test",
            restart_window_seconds=0.05,
            max_restarts_in_window=2,
        )
        for _ in range(2):
            monitor.record_restart(reason="crash", success=True)
        assert monitor.is_restart_allowed() is False

        time.sleep(0.1)

        assert monitor.is_restart_allowed() is True


# ---------------------------------------------------------------------------
# WorkerHealthMonitor — spontaneous exit detection
# ---------------------------------------------------------------------------


class TestWorkerhealthMonitorSpontaneousExit:
    def test_warns_when_running_transitions_to_exited(
        self, monitor: WorkerHealthMonitor, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        monitor.check(container_status="running")
        with caplog.at_level(logging.WARNING):
            monitor.check(container_status="exited")
        assert any("exited spontaneously" in m for m in caplog.messages)

    def test_no_warning_on_first_check(
        self, monitor: WorkerHealthMonitor, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            monitor.check(container_status="exited")
        assert not any("exited spontaneously" in m for m in caplog.messages)

    def test_no_warning_when_status_unchanged(
        self, monitor: WorkerHealthMonitor, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        monitor.check(container_status="running")
        with caplog.at_level(logging.WARNING):
            monitor.check(container_status="running")
        assert not any("exited spontaneously" in m for m in caplog.messages)

    def test_probe_returning_false_suppresses_warning_and_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Deactivation path: the ``reconcile_worker_lifecycle`` stop
        empties the workers dir and shuts the container down through a
        different ``WorkerManager`` instance (no monitor attached). On
        the watcher's next probe the long-lived monitor sees
        running→exited; the dir-empty probe must downgrade the log to
        INFO so operators aren't paged on a deliberate deactivation.
        """
        import logging

        workers_present = True

        def probe() -> bool:
            return workers_present

        monitor = WorkerHealthMonitor(
            container_name="cyberwave-worker-test",
            expected_running_probe=probe,
        )
        monitor.check(container_status="running")

        workers_present = False
        with caplog.at_level(logging.INFO):
            monitor.check(container_status="exited")

        assert not any("exited spontaneously" in msg for msg in caplog.messages)
        assert any("suppressing spontaneous-exit warning" in msg for msg in caplog.messages)

    def test_probe_returning_true_still_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """If the workers dir is non-empty (worker SHOULD be running)
        and the container exits, that IS a crash-loop signal — the
        probe must not swallow it.
        """
        import logging

        monitor = WorkerHealthMonitor(
            container_name="cyberwave-worker-test",
            expected_running_probe=lambda: True,
        )
        monitor.check(container_status="running")
        with caplog.at_level(logging.WARNING):
            monitor.check(container_status="exited")
        assert any("exited spontaneously" in msg for msg in caplog.messages)

    def test_probe_raising_falls_back_to_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Probe failure must not silently swallow a real exit. We
        prefer a false-positive WARN over missing a real crash loop.
        """
        import logging

        def boom() -> bool:
            raise RuntimeError("filesystem down")

        monitor = WorkerHealthMonitor(
            container_name="cyberwave-worker-test",
            expected_running_probe=boom,
        )
        monitor.check(container_status="running")
        with caplog.at_level(logging.WARNING):
            monitor.check(container_status="exited")
        assert any("exited spontaneously" in msg for msg in caplog.messages)

    def test_record_stop_suppresses_next_spontaneous_exit_warning(
        self, monitor: WorkerHealthMonitor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same-instance path: any caller that holds the monitor and
        issues a deliberate stop (``WorkerManager.stop`` is the
        canonical wirer-upper) can call ``record_stop`` to pre-empt
        the false-positive warning even when no probe is configured.
        """
        import logging

        monitor.check(container_status="running")
        monitor.record_stop(reason="test")
        with caplog.at_level(logging.WARNING):
            monitor.check(container_status="exited")
        assert not any("exited spontaneously" in msg for msg in caplog.messages)

    def test_record_stop_logs_info_with_reason(
        self, monitor: WorkerHealthMonitor, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.INFO):
            monitor.record_stop(reason="workers-dir-empty")
        assert any(
            "Worker stop recorded" in msg and "workers-dir-empty" in msg for msg in caplog.messages
        )

    def test_record_stop_clears_uptime_baseline(self, monitor: WorkerHealthMonitor) -> None:
        """Uptime should reset on a deliberate stop so the next
        ``record_start`` re-baselines cleanly."""
        monitor.record_start()
        monitor.record_stop(reason="test")
        state = monitor.check(container_status="running")
        assert state.uptime_seconds is None


# ---------------------------------------------------------------------------
# WorkerHealthMonitor — readiness probe
# ---------------------------------------------------------------------------


class TestWorkerhealthMonitorReadinessProbe:
    def test_ready_true_when_no_probe(self, monitor: WorkerHealthMonitor) -> None:
        state = monitor.check(container_status="running")
        assert state.is_ready is True

    def test_ready_follows_probe_result(self) -> None:
        probe_result = True

        def probe() -> bool:
            return probe_result

        m = WorkerHealthMonitor(container_name="test", readiness_probe=probe)
        state = m.check(container_status="running")
        assert state.is_ready is True

        probe_result = False
        state = m.check(container_status="running")
        assert state.is_ready is False

    def test_ready_false_when_not_running(self) -> None:
        probe_calls = []

        def probe() -> bool:
            probe_calls.append(True)
            return True

        m = WorkerHealthMonitor(container_name="test", readiness_probe=probe)
        state = m.check(container_status="exited")
        assert state.is_ready is False
        assert not probe_calls

    def test_probe_exception_results_in_not_ready(self) -> None:
        def bad_probe() -> bool:
            raise RuntimeError("probe failed")

        m = WorkerHealthMonitor(container_name="test", readiness_probe=bad_probe)
        state = m.check(container_status="running")
        assert state.is_ready is False


# ---------------------------------------------------------------------------
# WorkerHealthMonitor — reset
# ---------------------------------------------------------------------------


class TestWorkerhealthMonitorReset:
    def test_reset_clears_restart_records(self, monitor: WorkerHealthMonitor) -> None:
        for _ in range(5):
            monitor.record_restart(reason="crash", success=True)
        monitor.reset()
        state = monitor.check(container_status="running")
        assert state.restart_count == 0
        assert state.recent_restarts == 0

    def test_reset_clears_circuit_breaker(self, monitor: WorkerHealthMonitor) -> None:
        for _ in range(3):
            monitor.record_restart(reason="crash", success=True)
        assert not monitor.is_restart_allowed()
        monitor.reset()
        assert monitor.is_restart_allowed() is True


# ---------------------------------------------------------------------------
# ResourceLimits
# ---------------------------------------------------------------------------


class TestResourceLimits:
    def test_no_limits_returns_empty_args(self) -> None:
        rl = ResourceLimits()
        assert rl.to_docker_args() == []
        assert rl.to_env_args() == []

    def test_memory_limit_arg(self) -> None:
        rl = ResourceLimits(memory_mb=2048)
        args = rl.to_docker_args()
        assert "--memory" in args
        idx = args.index("--memory")
        assert args[idx + 1] == "2048m"

    def test_cpu_quota_args(self) -> None:
        rl = ResourceLimits(cpu_quota_percent=50.0)
        args = rl.to_docker_args()
        assert "--cpu-period" in args
        assert "--cpu-quota" in args
        period_idx = args.index("--cpu-period")
        quota_idx = args.index("--cpu-quota")
        period = int(args[period_idx + 1])
        quota = int(args[quota_idx + 1])
        assert quota == period // 2  # 50%

    def test_gpu_memory_fraction_env_arg(self) -> None:
        rl = ResourceLimits(gpu_memory_fraction=0.5)
        args = rl.to_env_args()
        assert "-e" in args
        env_idx = args.index("-e")
        assert "CYBERWAVE_GPU_MEM_FRACTION" in args[env_idx + 1]

    def test_combined_limits(self) -> None:
        rl = ResourceLimits(cpu_quota_percent=25.0, memory_mb=1024, gpu_memory_fraction=0.25)
        docker_args = rl.to_docker_args()
        env_args = rl.to_env_args()
        assert "--cpu-quota" in docker_args
        assert "--memory" in docker_args
        assert any("CYBERWAVE_GPU_MEM_FRACTION" in a for a in env_args)


# ---------------------------------------------------------------------------
# WorkerManager — circuit-breaker integration
# ---------------------------------------------------------------------------


class TestWorkermanagerCircuitBreaker:
    def test_restart_blocked_when_circuit_breaker_tripped(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monitor = WorkerHealthMonitor(
            container_name=worker_manager.container_name,
            max_restarts_in_window=2,
            restart_window_seconds=60,
        )
        worker_manager.set_health_monitor(monitor)

        # Trip the circuit-breaker.
        for _ in range(2):
            monitor.record_restart(reason="crash", success=True)

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)

        stop_called: list[str] = []

        def _fake_rm(name: str, **kw: object) -> bool:
            stop_called.append(name)
            return True

        monkeypatch.setattr(wm_module, "docker_rm", _fake_rm)

        result = worker_manager.restart(reason="file-change")
        assert result is False
        assert not stop_called, "stop should not be called when circuit-breaker is tripped"

    def test_restart_allowed_when_circuit_breaker_not_tripped(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monitor = WorkerHealthMonitor(
            container_name=worker_manager.container_name,
            max_restarts_in_window=5,
        )
        worker_manager.set_health_monitor(monitor)

        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "worker.py").write_text("pass")

        run_calls: list[list[str]] = []

        def fake_run(cmd: list, **kwargs: object) -> MagicMock:
            run_calls.append(cmd)
            m = MagicMock()
            m.returncode = 0
            return m

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "none")
        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        with patch.object(wm_module, "docker_container_status", side_effect=["none", "running"]):
            result = worker_manager.restart(reason="test")

        assert result is True

    def test_restart_records_reason_in_health_monitor(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monitor = WorkerHealthMonitor(container_name=worker_manager.container_name)
        worker_manager.set_health_monitor(monitor)

        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "worker.py").write_text("pass")

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
        monkeypatch.setattr(wm_module.subprocess, "run", lambda cmd, **kw: MagicMock(returncode=0))
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        with patch.object(wm_module, "docker_container_status", side_effect=["none", "running"]):
            worker_manager.restart(reason="worker-files-changed")

        state = monitor.check(container_status="running")
        assert state.restart_count == 1
        assert state.restart_records[0].reason == "worker-files-changed"


# ---------------------------------------------------------------------------
# WorkerWatcher — cool-down and reason propagation
# ---------------------------------------------------------------------------


class TestWorkerWatcherCooldown:
    def _make_watcher(
        self, workers_dir: Path, min_interval: float = 0.0
    ) -> tuple[WorkerWatcher, MagicMock, MagicMock]:
        worker_manager = MagicMock()
        model_manager = MagicMock()
        model_manager.scan_worker_model_ids.return_value = []
        watcher = WorkerWatcher(
            workers_dir=workers_dir,
            worker_manager=worker_manager,
            model_manager=model_manager,
            min_restart_interval_seconds=min_interval,
        )
        return watcher, worker_manager, model_manager

    def test_restart_deferred_within_cooldown(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, _ = self._make_watcher(workers_dir, min_interval=60.0)
        watcher.reconcile_worker_files()  # baseline

        (workers_dir / "a.py").write_text("pass")
        triggered1 = watcher.reconcile_worker_files()  # should trigger (first restart)
        assert triggered1 is True
        worker_manager.restart.assert_called_once()

        # Modify a file again immediately — should be deferred.
        import time as _time

        _time.sleep(0.02)
        (workers_dir / "b.py").write_text("pass")
        triggered2 = watcher.reconcile_worker_files()
        assert triggered2 is False
        assert worker_manager.restart.call_count == 1

    def test_restart_triggers_after_cooldown_expires(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, _ = self._make_watcher(workers_dir, min_interval=0.05)
        watcher.reconcile_worker_files()  # baseline

        (workers_dir / "a.py").write_text("pass")
        watcher.reconcile_worker_files()  # triggers first restart
        assert worker_manager.restart.call_count == 1

        # Change another file and wait for cool-down.
        import time as _time

        _time.sleep(0.06)
        (workers_dir / "b.py").write_text("pass")
        triggered = watcher.reconcile_worker_files()
        assert triggered is True
        assert worker_manager.restart.call_count == 2

    def test_restart_called_with_reason(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        watcher, worker_manager, _ = self._make_watcher(workers_dir, min_interval=0.0)
        watcher.reconcile_worker_files()  # baseline

        (workers_dir / "detect.py").write_text("pass")
        watcher.reconcile_worker_files()

        worker_manager.restart.assert_called_once_with(reason="worker-files-changed")

    def test_no_restart_when_no_change(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        (workers_dir / "detect.py").write_text("pass")

        watcher, worker_manager, _ = self._make_watcher(workers_dir, min_interval=0.0)
        watcher.reconcile_worker_files()  # baseline
        watcher.reconcile_worker_files()  # no change

        worker_manager.restart.assert_not_called()


# ---------------------------------------------------------------------------
# WorkerWatcher — check_health integration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WorkerHealthState.summary_line
# ---------------------------------------------------------------------------


class TestWorkerhealthStateSummaryLine:
    def test_summary_healthy(self, monitor: WorkerHealthMonitor) -> None:
        state = monitor.check(container_status="running")
        line = state.summary_line()
        assert "running" in line
        assert "restarts=0" in line

    def test_summary_unhealthy_shows_cross(self, monitor: WorkerHealthMonitor) -> None:
        state = monitor.check(container_status="exited")
        line = state.summary_line()
        assert "✗" in line

    def test_summary_circuit_breaker_tripped(self, monitor: WorkerHealthMonitor) -> None:
        for _ in range(3):
            monitor.record_restart(reason="crash", success=True)
        state = monitor.check(container_status="running")
        line = state.summary_line()
        assert "circuit-breaker tripped" in line
        assert "3" in line


# ---------------------------------------------------------------------------
# WorkerHealthMonitor._query_container_status (via check() without status arg)
# ---------------------------------------------------------------------------


class TestWorkerhealthMonitorQueryContainerStatus:
    def test_check_without_status_queries_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.worker_health as wh_module

        monkeypatch.setattr(
            "cyberwave_edge_core.docker_helpers.docker_container_status",
            lambda name: "running",
        )
        monitor = WorkerHealthMonitor(container_name="cyberwave-worker-test")
        state = monitor.check()  # no container_status supplied
        assert state.container_status == "running"
        assert state.is_healthy is True

    def test_query_returns_unknown_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cyberwave_edge_core.worker_health as wh_module

        def raise_exc(name: str) -> str:
            raise RuntimeError("docker gone")

        monkeypatch.setattr(
            "cyberwave_edge_core.docker_helpers.docker_container_status",
            raise_exc,
        )
        monitor = WorkerHealthMonitor(container_name="cyberwave-worker-test")
        state = monitor.check()
        assert state.container_status == "unknown"


class TestWorkerWatcherCheckHealth:
    def test_check_health_returns_none_without_monitor(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        worker_manager = MagicMock()
        worker_manager.health_monitor = None
        model_manager = MagicMock()
        watcher = WorkerWatcher(
            workers_dir=workers_dir,
            worker_manager=worker_manager,
            model_manager=model_manager,
        )
        assert watcher.check_health() is None

    def test_check_health_calls_monitor(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()

        monitor = MagicMock()
        mock_state = MagicMock()
        mock_state.circuit_breaker_tripped = False
        monitor.check.return_value = mock_state

        worker_manager = MagicMock()
        worker_manager.health_monitor = monitor
        model_manager = MagicMock()

        watcher = WorkerWatcher(
            workers_dir=workers_dir,
            worker_manager=worker_manager,
            model_manager=model_manager,
        )
        result = watcher.check_health()

        monitor.check.assert_called_once()
        assert result is mock_state
