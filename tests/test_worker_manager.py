"""Unit tests for WorkerManager (worker_manager.py).

Covers:
- Worker container env var injection
- Volume mount args
- GPU detection flag
- Zenoh env var helper
- start() skips container when no worker files present
- status() reflects correct state
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import cyberwave_edge_core.worker_manager as wm_module
from cyberwave_edge_core.worker_manager import WorkerManager, get_zenoh_env_vars


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def worker_manager(tmp_config: Path) -> WorkerManager:
    return WorkerManager(
        config_dir=tmp_config,
        environment_uuid="aabbccdd-eeff-0011-2233-445566778899",
        token="testtoken1234",
        twin_uuids=["twin-uuid-1", "twin-uuid-2"],
    )


class TestWorkerManagerEnvVars:
    def test_core_env_vars_present(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.load_credentials_envs",
            lambda: {},
        )
        monkeypatch.setattr("os.environ", {})
        env = worker_manager._build_env_vars()

        assert env["CYBERWAVE_API_KEY"] == "testtoken1234"
        assert env["CYBERWAVE_DATA_BACKEND"] == "zenoh"
        assert env["CYBERWAVE_ENVIRONMENT_UUID"] == "aabbccdd-eeff-0011-2233-445566778899"
        assert env["CYBERWAVE_TWIN_UUIDS"] == "twin-uuid-1,twin-uuid-2"
        assert env["CYBERWAVE_EDGE_CONFIG_DIR"] == "/app/.cyberwave"

    def test_zenoh_connect_injected_when_set(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_get_runtime_env_var(name: str, default: object = None) -> object:
            if name == "ZENOH_CONNECT":
                return "tcp/192.168.1.10:7447"
            return default

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            fake_get_runtime_env_var,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        env = worker_manager._build_env_vars()
        assert env["ZENOH_CONNECT"] == "tcp/192.168.1.10:7447"

    def test_zenoh_connect_absent_when_not_set(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        env = worker_manager._build_env_vars()
        assert "ZENOH_CONNECT" not in env

    def test_zenoh_shm_enabled_on_linux(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})
        monkeypatch.setattr(wm_module.platform, "system", lambda: "Linux")

        env = worker_manager._build_env_vars()
        assert env.get("ZENOH_SHM_ENABLED") == "true"

    def test_zenoh_shm_not_set_on_macos(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})
        monkeypatch.setattr(wm_module.platform, "system", lambda: "Darwin")

        env = worker_manager._build_env_vars()
        assert "ZENOH_SHM_ENABLED" not in env

    def test_non_production_environment_included(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_get_runtime_env_var(name: str, default: object = None) -> object:
            if name == "CYBERWAVE_ENVIRONMENT":
                return "dev"
            return default

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            fake_get_runtime_env_var,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        env = worker_manager._build_env_vars()
        assert env["CYBERWAVE_ENVIRONMENT"] == "dev"

    def test_production_environment_not_included(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_get_runtime_env_var(name: str, default: object = None) -> object:
            if name == "CYBERWAVE_ENVIRONMENT":
                return "production"
            return default

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            fake_get_runtime_env_var,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        env = worker_manager._build_env_vars()
        assert "CYBERWAVE_ENVIRONMENT" not in env


class TestWorkerManagerVolumes:
    def test_volume_args_include_config_dir(
        self, worker_manager: WorkerManager, tmp_config: Path
    ) -> None:
        args = worker_manager._build_volume_args()
        assert f"{tmp_config}:/app/.cyberwave" in args

    def test_volume_args_include_workers_dir_ro(
        self, worker_manager: WorkerManager, tmp_config: Path
    ) -> None:
        args = worker_manager._build_volume_args()
        workers_dir = tmp_config / "workers"
        assert f"{workers_dir}:/app/workers:ro" in args

    def test_volume_args_include_models_dir_ro(
        self, worker_manager: WorkerManager, tmp_config: Path
    ) -> None:
        args = worker_manager._build_volume_args()
        models_dir = tmp_config / "models"
        assert f"{models_dir}:/app/models:ro" in args

    def test_volume_dirs_created(
        self, worker_manager: WorkerManager, tmp_config: Path
    ) -> None:
        worker_manager._build_volume_args()
        assert (tmp_config / "workers").exists()
        assert (tmp_config / "models").exists()


class TestWorkerManagerStartSkipWhenNoWorkers:
    def test_start_returns_true_with_no_workers(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "none")

        result = worker_manager.start()
        assert result is True

    def test_start_returns_true_when_already_running(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "detect.py").write_text("import cw\nmodel = cw.models.load('yolov8n')\n")

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "running")
        monkeypatch.setattr(worker_manager, "_ensure_log_stream", lambda: None)

        result = worker_manager.start()
        assert result is True

    def test_start_returns_false_when_docker_unavailable(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: False)
        result = worker_manager.start()
        assert result is False


class TestWorkerManagerGPU:
    def test_gpu_args_added_when_nvidia_present(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        run_calls: list[list[str]] = []

        def fake_run(cmd: list, **kwargs: object) -> MagicMock:
            run_calls.append(cmd)
            m = MagicMock()
            m.returncode = 0
            return m

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "none")
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)
        monkeypatch.setattr(worker_manager, "_ensure_log_stream", lambda: None)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        with patch.object(
            wm_module, "docker_container_status", side_effect=["none", "running"]
        ):
            worker_manager._run_container()

        docker_run_cmd = next(
            (c for c in run_calls if c and c[0] == "docker" and "run" in c), None
        )
        assert docker_run_cmd is not None
        assert "--gpus" in docker_run_cmd
        assert "all" in docker_run_cmd

    def test_no_gpu_args_when_nvidia_absent(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        run_calls: list[list[str]] = []

        def fake_run(cmd: list, **kwargs: object) -> MagicMock:
            run_calls.append(cmd)
            m = MagicMock()
            m.returncode = 0
            return m

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)
        monkeypatch.setattr(worker_manager, "_ensure_log_stream", lambda: None)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        with patch.object(
            wm_module, "docker_container_status", side_effect=["none", "running"]
        ):
            worker_manager._run_container()

        docker_run_cmd = next(
            (c for c in run_calls if c and c[0] == "docker" and "run" in c), None
        )
        assert docker_run_cmd is not None
        assert "--gpus" not in docker_run_cmd


class TestGetZenohEnvVars:
    def test_data_backend_always_zenoh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: None,
        )
        env = get_zenoh_env_vars()
        assert env["CYBERWAVE_DATA_BACKEND"] == "zenoh"

    def test_zenoh_connect_included_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_get_runtime_env_var(name: str, default: object = None) -> object:
            return "tcp/router:7447" if name == "ZENOH_CONNECT" else None

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            fake_get_runtime_env_var,
        )
        env = get_zenoh_env_vars()
        assert env["ZENOH_CONNECT"] == "tcp/router:7447"

    def test_zenoh_connect_excluded_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: None,
        )
        env = get_zenoh_env_vars()
        assert "ZENOH_CONNECT" not in env


# ---------------------------------------------------------------------------
# WorkerManager.stop
# ---------------------------------------------------------------------------


class TestWorkerManagerStop:
    def test_returns_true_when_docker_unavailable(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: False)
        assert worker_manager.stop() is True

    def test_returns_true_when_container_not_found(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "none")
        assert worker_manager.stop() is True

    def test_calls_docker_rm_when_container_exists(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        removed: list[str] = []
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "running")
        monkeypatch.setattr(
            wm_module, "docker_rm", lambda name, **kw: removed.append(name) or True
        )
        result = worker_manager.stop()
        assert result is True
        assert removed == [worker_manager._container_name]

    def test_returns_false_when_docker_rm_fails(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "exited")
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: False)
        assert worker_manager.stop() is False


# ---------------------------------------------------------------------------
# WorkerManager.status
# ---------------------------------------------------------------------------


class TestWorkerManagerStatus:
    def test_returns_worker_status_object(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cyberwave_edge_core.worker_manager import WorkerStatus

        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "running")
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        s = worker_manager.status()
        assert isinstance(s, WorkerStatus)

    def test_container_name_in_status(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "running")
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        s = worker_manager.status()
        assert s.container_name == worker_manager._container_name

    def test_worker_files_listed(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True)
        (workers_dir / "alpha.py").write_text("pass")
        (workers_dir / "beta.py").write_text("pass")
        (workers_dir / "readme.md").write_text("docs")

        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "running")
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        s = worker_manager.status()
        assert s.worker_files == ["alpha.py", "beta.py"]

    def test_worker_files_empty_when_dir_absent(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "none")
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        s = worker_manager.status()
        assert s.worker_files == []

    def test_gpu_enabled_reflects_nvidia_runtime(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "running")
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        s = worker_manager.status()
        assert s.gpu_enabled is True

    def test_health_fields_populated_when_monitor_attached(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cyberwave_edge_core.worker_health import WorkerHealthMonitor

        monitor = WorkerHealthMonitor(container_name=worker_manager._container_name)
        worker_manager.set_health_monitor(monitor)

        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "running")
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        s = worker_manager.status()
        assert s.health_state is not None
        assert s.circuit_breaker_tripped is False

    def test_health_fields_zero_when_no_monitor(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "running")
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        s = worker_manager.status()
        assert s.health_state is None
        assert s.restart_count == 0


# ---------------------------------------------------------------------------
# WorkerManager.logs
# ---------------------------------------------------------------------------


class TestWorkerManagerLogs:
    def test_logs_returns_early_when_docker_unavailable(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: False)
        run_calls: list[object] = []
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: run_calls.append(a))
        worker_manager.logs()
        assert run_calls == []

    def test_logs_follow_calls_subprocess(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        captured: list[list[str]] = []
        monkeypatch.setattr(
            wm_module.subprocess, "run", lambda cmd, **kw: captured.append(cmd)
        )
        worker_manager.logs(follow=True)
        assert any("-f" in cmd for cmd in captured)

    def test_logs_no_follow_calls_subprocess_without_f(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        captured: list[list[str]] = []
        monkeypatch.setattr(
            wm_module.subprocess, "run", lambda cmd, **kw: captured.append(cmd)
        )
        worker_manager.logs(follow=False)
        assert captured
        assert "-f" not in captured[0]

    def test_logs_follow_swallows_keyboard_interrupt(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)

        def raise_ki(*a: object, **kw: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(wm_module.subprocess, "run", raise_ki)
        worker_manager.logs(follow=True)  # should not propagate

    def test_logs_no_follow_swallows_os_error(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)

        def raise_os(*a: object, **kw: object) -> None:
            raise OSError("docker not found")

        monkeypatch.setattr(wm_module.subprocess, "run", raise_os)
        worker_manager.logs(follow=False)  # should not propagate


# ---------------------------------------------------------------------------
# WorkerManager._build_network_args
# ---------------------------------------------------------------------------


class TestWorkerManagerNetworkArgs:
    def test_darwin_uses_add_host(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module.platform, "system", lambda: "Darwin")
        args = worker_manager._build_network_args()
        assert "--add-host" in args

    def test_linux_uses_host_network(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module.platform, "system", lambda: "Linux")
        args = worker_manager._build_network_args()
        assert "--network" in args
        assert "host" in args


# ---------------------------------------------------------------------------
# WorkerManager._run_container failure paths
# ---------------------------------------------------------------------------


class TestWorkerManagerRunContainerFailures:
    def _setup_run(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "worker.py").write_text("pass")
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
        monkeypatch.setattr(worker_manager, "_ensure_log_stream", lambda: None)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

    def test_returns_false_on_called_process_error(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import subprocess as _sp

        self._setup_run(worker_manager, tmp_config, monkeypatch)

        def raise_cpe(*a: object, **kw: object) -> None:
            raise _sp.CalledProcessError(1, "docker", stderr="image pull failed")

        monkeypatch.setattr(wm_module.subprocess, "run", raise_cpe)
        assert worker_manager._run_container() is False

    def test_returns_false_on_timeout(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import subprocess as _sp

        self._setup_run(worker_manager, tmp_config, monkeypatch)

        def raise_timeout(*a: object, **kw: object) -> None:
            raise _sp.TimeoutExpired("docker", 60)

        monkeypatch.setattr(wm_module.subprocess, "run", raise_timeout)
        assert worker_manager._run_container() is False

    def test_returns_false_when_container_exits_immediately(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._setup_run(worker_manager, tmp_config, monkeypatch)
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)
        # Probe always returns "exited"
        with patch.object(wm_module, "docker_container_status", return_value="exited"):
            assert worker_manager._run_container() is False

    def test_returns_true_when_probe_window_exhausted_without_running(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._setup_run(worker_manager, tmp_config, monkeypatch)
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)
        # Container never reaches "running" or "exited" within the probe window.
        with patch.object(wm_module, "docker_container_status", return_value="unknown"):
            assert worker_manager._run_container() is True


# ---------------------------------------------------------------------------
# WorkerManager.start — health monitor record_start integration
# ---------------------------------------------------------------------------


class TestWorkerManagerStartHealthIntegration:
    def test_start_calls_record_start_on_success(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cyberwave_edge_core.worker_health import WorkerHealthMonitor

        monitor = WorkerHealthMonitor(container_name=worker_manager._container_name)
        worker_manager.set_health_monitor(monitor)

        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "worker.py").write_text("pass")

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)
        monkeypatch.setattr(worker_manager, "_ensure_log_stream", lambda: None)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        with patch.object(
            wm_module, "docker_container_status", side_effect=["none", "running"]
        ):
            result = worker_manager.start()

        assert result is True
        # record_start sets _last_start_time; uptime will be non-None in check()
        state = monitor.check(container_status="running")
        assert state.uptime_seconds is not None


# ---------------------------------------------------------------------------
# _follow_worker_logs free function
# ---------------------------------------------------------------------------


class TestFollowWorkerLogs:
    def test_streams_lines_to_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from cyberwave_edge_core.worker_manager import _follow_worker_logs

        fake_proc = MagicMock()
        fake_proc.stdout = iter(["line one\n", "line two\n"])
        fake_proc.wait.return_value = 0

        with patch.object(wm_module, "docker_logs_follow", return_value=fake_proc):
            with caplog.at_level(logging.INFO):
                _follow_worker_logs("my_container")

        messages = " ".join(caplog.messages)
        assert "line one" in messages
        assert "line two" in messages

    def test_returns_early_when_process_is_none(self) -> None:
        from cyberwave_edge_core.worker_manager import _follow_worker_logs

        with patch.object(wm_module, "docker_logs_follow", return_value=None):
            _follow_worker_logs("my_container")  # should not raise

    def test_handles_exception_in_stdout_iteration(self) -> None:
        from cyberwave_edge_core.worker_manager import _follow_worker_logs

        fake_proc = MagicMock()
        # stdout is truthy but raises when iterated
        fake_proc.stdout = MagicMock()
        fake_proc.stdout.__iter__ = MagicMock(side_effect=RuntimeError("pipe broken"))
        fake_proc.wait.return_value = 0

        with patch.object(wm_module, "docker_logs_follow", return_value=fake_proc):
            _follow_worker_logs("my_container")  # should not raise
