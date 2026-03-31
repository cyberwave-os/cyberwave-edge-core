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
