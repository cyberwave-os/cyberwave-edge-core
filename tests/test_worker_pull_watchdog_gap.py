"""Startup timeout and stop-request regressions (CYB-3338 / CYB-4368)."""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from cyberwave_edge_core import startup, watchdog, worker_manager


class _Recorder:
    def __init__(self) -> None:
        self.extensions: list[float] = []
        self.statuses: list[str] = []

    def extend_timeout(self, seconds: float) -> None:
        self.extensions.append(seconds)

    def notify_status(self, status: str) -> None:
        self.statuses.append(status)


def _call_keywords(fn_name: str, callee: str) -> list[list[str]]:
    tree = ast.parse(Path(startup.__file__).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    return [
        [kw.arg for kw in node.keywords]
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == callee
    ]


@pytest.fixture(autouse=True)
def clear_shutdown():
    startup.shutdown_event.clear()
    yield
    startup.shutdown_event.clear()


@pytest.mark.parametrize("callee", ["fetch_and_run_twin_drivers", "_start_worker_after_drivers"])
def test_boot_forwards_watchdog(callee):
    calls = _call_keywords("run_startup_checks", callee)
    assert calls
    assert all("watchdog" in kw for kw in calls)


def test_worker_startup_extends_timeout_while_pull_is_silent(monkeypatch, tmp_path):
    recorder = _Recorder()
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(startup, "_wait_for_driver_readiness", lambda _: {})
    monkeypatch.setattr("cyberwave_edge_core.docker_helpers.docker_container_status", lambda _: "none")
    monkeypatch.setattr(startup, "load_worker_resource_limits", lambda: None)
    real_heartbeat = watchdog.startup_heartbeat
    monkeypatch.setattr(
        watchdog, "startup_heartbeat",
        lambda *args, **kwargs: real_heartbeat(*args, **kwargs, interval_seconds=0.01),
    )
    manager = Mock()
    manager.start.side_effect = lambda: time.sleep(0.08)
    monkeypatch.setattr(worker_manager, "WorkerManager", Mock(return_value=manager))

    startup._start_worker_after_drivers(
        token="test", environment_uuid="test-env", twin_uuids=[], watchdog=recorder,
    )

    manager.start.assert_called_once()
    assert len(recorder.extensions) >= 2
    count = len(recorder.extensions)
    time.sleep(0.03)
    assert len(recorder.extensions) == count


@pytest.mark.parametrize("fail", [False, True])
def test_heartbeat_stops_after_scope(fail):
    recorder = _Recorder()
    stop = threading.Event()
    try:
        with watchdog.startup_heartbeat(
            recorder, shutdown_event=stop, status="Pulling worker", interval_seconds=0.01,
        ):
            if fail:
                raise RuntimeError("pull failed")
    except RuntimeError:
        pass
    count = len(recorder.extensions)
    time.sleep(0.03)
    assert count >= 1
    assert len(recorder.extensions) == count


def test_heartbeat_does_not_extend_requested_stop():
    recorder = _Recorder()
    stop = threading.Event()
    with watchdog.startup_heartbeat(
        recorder, shutdown_event=stop, status="Pulling worker", interval_seconds=0.01,
    ):
        stop.set()
        time.sleep(0.03)
        count = len(recorder.extensions)
        time.sleep(0.03)
        assert len(recorder.extensions) == count


def test_worker_start_skipped_after_sigterm(monkeypatch):
    wait = Mock()
    monkeypatch.setattr(startup, "_wait_for_driver_readiness", wait)
    startup.shutdown_event.set()
    startup._start_worker_after_drivers(token="test", environment_uuid="test-env", twin_uuids=[])
    wait.assert_not_called()


def test_driver_pull_does_not_extend_stop_timeout(monkeypatch):
    recorder = _Recorder()
    startup.shutdown_event.set()
    monkeypatch.setattr(
        startup, "_pull_docker_image_with_progress_multi", lambda *a, **kw: time.sleep(0.04),
    )
    startup._pull_driver_images_parallel(
        ["fake/image:tag"], watchdog=recorder, heartbeat_interval_seconds=0.01,
    )
    assert not recorder.extensions


def test_sigterm_during_boot_never_announces_ready(monkeypatch):
    from click.testing import CliRunner

    from cyberwave_edge_core import main, resource_monitor

    process_watchdog = Mock()
    monkeypatch.setattr(watchdog, "ProcessWatchdog", lambda: process_watchdog)
    monkeypatch.setattr(watchdog, "protect_edge_core_oom", lambda: None)
    monkeypatch.setattr(resource_monitor, "SystemResourceMonitor", Mock())
    monkeypatch.setattr(main.signal, "signal", Mock())
    cleanup = Mock()
    stop_worker = Mock()
    runtime = Mock()
    monkeypatch.setattr(startup, "_graceful_shutdown", cleanup)
    monkeypatch.setattr(startup, "_stop_worker_container_for_restart", stop_worker)
    monkeypatch.setattr(main, "run_runtime_loop", runtime)

    def boot(**kwargs):
        main._sigterm_handler(15, None)
        return True

    monkeypatch.setattr(main, "run_startup_checks", boot)
    result = CliRunner().invoke(main.cli, [])
    assert result.exit_code == 0, result.output
    process_watchdog.mark_ready.assert_not_called()
    runtime.assert_not_called()
    stop_worker.assert_called_once()
    cleanup.assert_called_once_with(None)
    process_watchdog.stop.assert_called_once()


def test_worker_pull_does_not_retry_after_stop(monkeypatch, tmp_path):
    stop = threading.Event()
    manager = worker_manager.WorkerManager(
        config_dir=tmp_path, environment_uuid="test-env", token="test", shutdown_event=stop,
    )
    monkeypatch.setattr(worker_manager, "_docker_image_present", lambda _: False)

    def pull(*args, **kwargs):
        stop.set()
        return False

    attempt = Mock(side_effect=pull)
    monkeypatch.setattr(manager, "_pull_worker_image_with_progress_once", attempt)
    assert manager._pull_worker_image_with_progress("test/stop:latest") is False
    attempt.assert_called_once()


@pytest.mark.parametrize("pulled", [False, True])
def test_stop_during_worker_pull_never_launches_container(monkeypatch, tmp_path, pulled):
    stop = threading.Event()
    manager = worker_manager.WorkerManager(
        config_dir=tmp_path, environment_uuid="test-env", token="test", shutdown_event=stop,
    )
    monkeypatch.setattr(worker_manager, "docker_rm", lambda _: True)
    monkeypatch.setattr(worker_manager, "is_jetson", lambda: False)
    monkeypatch.setattr(worker_manager, "docker_has_nvidia_runtime", lambda: False)
    monkeypatch.setattr(worker_manager, "_hailo_device_present", lambda: False)
    monkeypatch.setattr(worker_manager, "build_user_args", lambda: [])
    monkeypatch.setattr(startup, "get_runtime_env_var", lambda name, default=None: default)
    for method in ("_build_env_vars", "_build_volume_args", "_build_network_args"):
        monkeypatch.setattr(manager, method, lambda: {})
    monkeypatch.setattr(manager, "_reset_readiness_tracking", lambda: None)
    monkeypatch.setattr(manager, "_open_starting_alerts", lambda _: None)
    close_alerts = Mock()
    monkeypatch.setattr(manager, "_close_starting_alerts", close_alerts)
    launch = Mock()
    monkeypatch.setattr(worker_manager.subprocess, "run", launch)

    def pull(image):
        stop.set()
        return pulled

    monkeypatch.setattr(manager, "_pull_worker_image_with_progress", pull)
    assert manager._run_container() is False
    launch.assert_not_called()
    close_alerts.assert_called_once()
