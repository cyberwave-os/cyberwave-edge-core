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

import threading
import time
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

    def test_macos_worker_rewrites_localhost_base_url_for_container(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_get_runtime_env_var(name: str, default: object = None) -> object:
            if name == "CYBERWAVE_BASE_URL":
                return "http://localhost:8000"
            return default

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            fake_get_runtime_env_var,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})
        monkeypatch.setattr(wm_module.platform, "system", lambda: "Darwin")

        env = worker_manager._build_env_vars()

        assert env["CYBERWAVE_BASE_URL"] == "http://host.docker.internal:8000"

    def test_macos_worker_rewrites_localhost_mqtt_host_for_container(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_get_runtime_env_var(name: str, default: object = None) -> object:
            if name == "CYBERWAVE_MQTT_HOST":
                return "localhost"
            return default

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            fake_get_runtime_env_var,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})
        monkeypatch.setattr(wm_module.platform, "system", lambda: "Darwin")

        env = worker_manager._build_env_vars()

        assert env["CYBERWAVE_MQTT_HOST"] == "host.docker.internal"

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

    def test_zenoh_shm_disabled_by_default_on_linux(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SHM must default to ``false`` on Linux — enabling it requires
        ``--ipc=host`` between containers, which we do not configure by
        default.  The worker and driver paths both route through
        ``build_zenoh_env_vars(ZenohConfig())``, so an explicit opt-in via
        the ``ZENOH_SHARED_MEMORY`` process env propagates consistently to
        both sides.
        """
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})
        monkeypatch.setattr(wm_module.platform, "system", lambda: "Linux")

        env = worker_manager._build_env_vars()
        assert env.get("ZENOH_SHARED_MEMORY") == "false"

    def test_zenoh_shm_disabled_by_default_on_macos(
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
        assert env.get("ZENOH_SHARED_MEMORY") == "false"

    def test_zenoh_shm_opt_in_from_env(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit ``ZENOH_SHARED_MEMORY=true`` in the edge-core process env
        must propagate to the worker container."""
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {"ZENOH_SHARED_MEMORY": "true"})
        monkeypatch.setattr(wm_module.platform, "system", lambda: "Linux")

        env = worker_manager._build_env_vars()
        assert env.get("ZENOH_SHARED_MEMORY") == "true"

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

    def test_volume_args_include_models_dir_rw(
        self, worker_manager: WorkerManager, tmp_config: Path
    ) -> None:
        args = worker_manager._build_volume_args()
        models_dir = tmp_config / "models"
        assert f"{models_dir}:/app/models" in args
        assert f"{models_dir}:/app/models:ro" not in args

    def test_volume_dirs_created(self, worker_manager: WorkerManager, tmp_config: Path) -> None:
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

        result = worker_manager.start()
        assert result is True

    def test_start_returns_true_when_restarting(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Container in Docker's 'restarting' state must not trigger _run_container.

        --restart=unless-stopped causes Docker to automatically restart the
        container when it exits. During that brief 'restarting' window
        start() must yield to Docker rather than racing with its restart
        daemon, which would cause a name-conflict on the subsequent docker run.
        """
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "detect.py").write_text("pass\n")

        run_container_called: list[bool] = []
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "restarting")
        monkeypatch.setattr(
            WorkerManager,
            "_run_container",
            lambda self: run_container_called.append(True) or False,
        )

        result = worker_manager.start()
        assert result is True
        assert run_container_called == [], "_run_container must not be called during restarting"

    def test_concurrent_start_only_runs_container_once(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two threads calling start() simultaneously must only call _run_container once.

        This is the regression test for the per-container-name lock.
        Without the lock, both callers that pass the pre-lock status check
        would each call _run_container(), racing on docker rm + docker run.

        Determinism: a Barrier ensures both threads reach the lock-acquire
        point simultaneously (both see status="none" from the pre-lock check),
        then one wins the lock and calls _run_container (marking state as
        "running"), and the other re-checks inside the lock, sees "running",
        and skips.
        """
        import threading as _th

        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "detect.py").write_text("pass\n")

        run_container_calls: list[int] = []
        container_state: dict[str, str] = {"status": "none"}
        # Both threads must pass through the pre-lock status check before
        # either acquires the lock, ensuring they both see "none".
        barrier = _th.Barrier(2)

        original_status_calls: list[str] = []

        def status_with_barrier(name: str) -> str:
            s = container_state["status"]
            original_status_calls.append(s)
            if len(original_status_calls) <= 2:
                # Synchronise: wait until both threads have called status
                # (pre-lock check), then release them to race for the lock.
                barrier.wait(timeout=2)
            return s

        def fake_run_container(self: WorkerManager) -> bool:
            run_container_calls.append(1)
            container_state["status"] = "running"
            return True

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", status_with_barrier)
        monkeypatch.setattr(WorkerManager, "_run_container", fake_run_container)

        errors: list[BaseException] = []

        def call_start() -> None:
            try:
                worker_manager.start()
            except BaseException as exc:
                errors.append(exc)

        t1 = _th.Thread(target=call_start)
        t2 = _th.Thread(target=call_start)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"
        assert not t1.is_alive() and not t2.is_alive(), "Threads did not finish — possible deadlock"
        # With the lock: only the thread that wins the lock calls _run_container.
        # The other re-checks inside the lock, sees "running", and skips.
        assert len(run_container_calls) == 1, (
            f"Expected exactly 1 _run_container call, got {len(run_container_calls)}. "
            "Lock is missing or not working correctly."
        )

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

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--gpus" in docker_run_cmd
        assert "all" in docker_run_cmd
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest-gpu"

    def test_gpu_image_not_double_suffixed_when_already_selected(
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

        worker_manager._image = "cyberwaveos/edge-ml-worker:latest-gpu"

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "none")
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            assert worker_manager._run_container() is True

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--gpus" in docker_run_cmd
        assert "all" in docker_run_cmd
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest-gpu"

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

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--gpus" not in docker_run_cmd


def _stub_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Common stubs for ``_run_container`` tests that bypass real env / docker."""
    monkeypatch.setattr(wm_module, "docker_available", lambda: True)
    monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "none")
    monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
    monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        "cyberwave_edge_core.startup.get_runtime_env_var",
        lambda name, default=None: default,
    )
    monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
    monkeypatch.setattr("os.environ", {})


class TestWorkerManagerHailo:
    """Verify Hailo accelerator device passthrough + image tag rewrite.

    The host-side signal is the presence of ``/dev/hailo0`` (created by
    HailoRT's PCIe kernel driver). When detected on a worker that's about
    to start the standard ``edge-ml-worker:<tag>`` image, edge-core
    rewrites the tag to the Hailo sibling (``<tag>-hailo``) and adds the
    device + group + Gate-4 env var passthrough.
    """

    def _stub_docker_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        run_calls: list[list[str]],
    ) -> None:
        def fake_run(cmd: list, **kwargs: object) -> MagicMock:
            run_calls.append(cmd)
            m = MagicMock()
            m.returncode = 0
            return m

        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)

    def test_hailo_device_passthrough_when_device_present(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: True)
        monkeypatch.setattr(wm_module, "group_gid", lambda name: 1010 if name == "hailo" else None)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--device" in docker_run_cmd
        assert "/dev/hailo0:/dev/hailo0:rwm" in docker_run_cmd
        assert "--group-add" in docker_run_cmd
        assert "1010" in docker_run_cmd
        assert "CYBERWAVE_REQUIRED_DEVICES=/dev/hailo0" in docker_run_cmd
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest-hailo"

    def test_hailo_skips_group_add_when_group_missing(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HailoRT 4.20+ ships /dev/hailo0 as 0666 and creates no group."""
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: True)
        monkeypatch.setattr(wm_module, "group_gid", lambda name: None)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--device" in docker_run_cmd
        assert "/dev/hailo0:/dev/hailo0:rwm" in docker_run_cmd
        assert "--group-add" not in docker_run_cmd
        assert "CYBERWAVE_REQUIRED_DEVICES=/dev/hailo0" in docker_run_cmd

    def test_no_hailo_args_when_device_absent(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: False)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--device" not in docker_run_cmd
        assert "CYBERWAVE_REQUIRED_DEVICES=/dev/hailo0" not in docker_run_cmd
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest"

    def test_gpu_takes_precedence_over_hailo(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NVIDIA + Hailo on the same host is unsupported; GPU wins."""
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: True)
        monkeypatch.setattr(wm_module, "group_gid", lambda name: 1010)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--gpus" in docker_run_cmd
        assert "--device" not in docker_run_cmd
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest-gpu"

    def test_hailo_image_not_double_suffixed_when_already_selected(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        worker_manager._image = "cyberwaveos/edge-ml-worker:latest-hailo"

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: True)
        monkeypatch.setattr(wm_module, "group_gid", lambda name: None)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--device" in docker_run_cmd
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest-hailo"

    def test_custom_image_override_left_untouched(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``CYBERWAVE_WORKER_IMAGE``-style operator overrides are not rewritten.

        Mirrors the GPU behaviour: device passthrough is added, but the image
        ref is left alone since the operator explicitly chose it.
        """
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        worker_manager._image = "myregistry.local/cyberwave/custom-worker:dev"

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: True)
        monkeypatch.setattr(wm_module, "group_gid", lambda name: None)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--device" in docker_run_cmd
        assert docker_run_cmd[-1] == "myregistry.local/cyberwave/custom-worker:dev"


class TestHailoHelpers:
    """Pure-function tests for the Hailo helpers (no docker subprocess)."""

    def test_apply_hailo_image_tag_rewrites_known_tag(self) -> None:
        assert (
            wm_module._apply_hailo_image_tag("cyberwaveos/edge-ml-worker:latest")
            == "cyberwaveos/edge-ml-worker:latest-hailo"
        )
        assert (
            wm_module._apply_hailo_image_tag("cyberwaveos/edge-ml-worker:dev")
            == "cyberwaveos/edge-ml-worker:dev-hailo"
        )

    def test_apply_hailo_image_tag_leaves_existing_hailo_alone(self) -> None:
        assert (
            wm_module._apply_hailo_image_tag("cyberwaveos/edge-ml-worker:dev-hailo")
            == "cyberwaveos/edge-ml-worker:dev-hailo"
        )

    def test_apply_hailo_image_tag_leaves_gpu_alone(self) -> None:
        assert (
            wm_module._apply_hailo_image_tag("cyberwaveos/edge-ml-worker:latest-gpu")
            == "cyberwaveos/edge-ml-worker:latest-gpu"
        )

    def test_apply_hailo_image_tag_leaves_custom_image_alone(self) -> None:
        assert (
            wm_module._apply_hailo_image_tag("myregistry.local/cyberwave/custom:dev")
            == "myregistry.local/cyberwave/custom:dev"
        )

    def test_build_hailo_passthrough_args_with_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wm_module, "group_gid", lambda name: 1010)
        args = wm_module._build_hailo_passthrough_args()
        assert args[:2] == ["--device", "/dev/hailo0:/dev/hailo0:rwm"]
        assert "--group-add" in args
        assert "1010" in args
        assert args[-1] == "CYBERWAVE_REQUIRED_DEVICES=/dev/hailo0"

    def test_build_hailo_passthrough_args_without_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "group_gid", lambda name: None)
        args = wm_module._build_hailo_passthrough_args()
        assert "--group-add" not in args
        assert "--device" in args
        assert "CYBERWAVE_REQUIRED_DEVICES=/dev/hailo0" in args


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
    """``stop()`` must be non-destructive: ``docker stop``, never ``docker rm``.

    Pre-fix it called ``docker rm -f`` which left every caller (the
    edge-restart flow, ``reconcile_worker_lifecycle``) without a
    container to inspect or restart cheaply.
    """

    @staticmethod
    def _patch_calls(monkeypatch: pytest.MonkeyPatch, status: str) -> dict[str, list[str]]:
        calls: dict[str, list[str]] = {"stop": [], "rm": []}
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda _n: status)
        monkeypatch.setattr(
            wm_module, "docker_stop", lambda name, **_kw: calls["stop"].append(name) or True
        )
        monkeypatch.setattr(
            wm_module, "docker_rm", lambda name, **_kw: calls["rm"].append(name) or True
        )
        return calls

    def test_returns_true_when_docker_unavailable(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: False)
        assert worker_manager.stop() is True

    @pytest.mark.parametrize("status", ["none", "exited", "created"])
    def test_short_circuits_for_non_running_states(
        self,
        worker_manager: WorkerManager,
        monkeypatch: pytest.MonkeyPatch,
        status: str,
    ) -> None:
        calls = self._patch_calls(monkeypatch, status)
        assert worker_manager.stop() is True
        assert calls["stop"] == [], f"docker_stop must not be called for status={status!r}"
        assert calls["rm"] == [], "stop() must never call docker_rm"

    def test_calls_docker_stop_and_never_docker_rm_when_running(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._patch_calls(monkeypatch, "running")
        assert worker_manager.stop() is True
        assert calls["stop"] == [worker_manager.container_name]
        assert calls["rm"] == [], (
            "Regression: stop() must NOT remove the container — "
            "use docker stop only so the container persists for restart/inspection."
        )

    def test_returns_false_when_docker_stop_fails(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda _n: "running")
        monkeypatch.setattr(wm_module, "docker_stop", lambda _name, **_kw: False)
        assert worker_manager.stop() is False

    def test_notifies_health_monitor_on_successful_stop(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful ``docker_stop`` must call
        ``WorkerHealthMonitor.record_stop`` so the next health probe
        doesn't false-positive on the running→exited transition. The
        reason flows through so log greps can correlate the stop with
        the upstream trigger (deactivation, edge restart, etc.).
        """
        from unittest.mock import MagicMock

        monitor = MagicMock()
        worker_manager.set_health_monitor(monitor)
        self._patch_calls(monkeypatch, "running")

        assert worker_manager.stop(reason="workers-dir-empty") is True
        monitor.record_stop.assert_called_once_with(reason="workers-dir-empty")

    def test_does_not_notify_monitor_when_docker_stop_fails(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed stop must not pre-empt the spontaneous-exit
        detector — if docker_stop returned False the container is
        likely still running, and a later real crash should still
        warn.
        """
        from unittest.mock import MagicMock

        monitor = MagicMock()
        worker_manager.set_health_monitor(monitor)
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda _n: "running")
        monkeypatch.setattr(wm_module, "docker_stop", lambda _name, **_kw: False)

        assert worker_manager.stop() is False
        monitor.record_stop.assert_not_called()

    @pytest.mark.parametrize("status", ["none", "exited", "created"])
    def test_does_not_notify_monitor_on_short_circuit_paths(
        self,
        worker_manager: WorkerManager,
        monkeypatch: pytest.MonkeyPatch,
        status: str,
    ) -> None:
        """The short-circuit branches (already non-running) skip the
        notification — there's no running→exited transition to
        suppress, so calling ``record_stop`` would mask a future real
        crash on the next start cycle.
        """
        from unittest.mock import MagicMock

        monitor = MagicMock()
        worker_manager.set_health_monitor(monitor)
        self._patch_calls(monkeypatch, status)

        assert worker_manager.stop() is True
        monitor.record_stop.assert_not_called()

    def test_stop_works_without_monitor_attached(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fallback path of ``reconcile_worker_lifecycle`` (cold
        boot, watcher hasn't ticked yet) and ``_stop_worker_container_for_restart``
        construct a fresh ``WorkerManager`` without a monitor and
        call stop() on it. That path must not blow up trying to
        ``record_stop`` on None.
        """
        self._patch_calls(monkeypatch, "running")
        # No set_health_monitor() call — _health_monitor is None.
        assert worker_manager.stop() is True


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
        assert s.container_name == worker_manager.container_name

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

        monitor = WorkerHealthMonitor(container_name=worker_manager.container_name)
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
    @staticmethod
    def _fake_popen(captured: list[list[str]]):
        """Return a Popen factory that records cmd and returns a no-op proc."""

        def factory(cmd, **kw):
            captured.append(cmd)
            proc = MagicMock()
            proc.stdout = iter([])
            proc.wait.return_value = 0
            return proc

        return factory

    def test_logs_returns_early_when_docker_unavailable(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: False)
        captured: list[list[str]] = []
        monkeypatch.setattr(wm_module.subprocess, "Popen", self._fake_popen(captured))
        worker_manager.logs()
        assert captured == []

    def test_logs_follow_calls_subprocess(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        captured: list[list[str]] = []
        monkeypatch.setattr(wm_module.subprocess, "Popen", self._fake_popen(captured))
        worker_manager.logs(follow=True)
        assert any("-f" in cmd for cmd in captured)

    def test_logs_no_follow_calls_subprocess_without_f(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        captured: list[list[str]] = []
        monkeypatch.setattr(wm_module.subprocess, "Popen", self._fake_popen(captured))
        worker_manager.logs(follow=False)
        assert captured
        assert "-f" not in captured[0]

    def test_logs_follow_swallows_keyboard_interrupt(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)

        def raise_ki(*a: object, **kw: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(wm_module.subprocess, "Popen", raise_ki)
        worker_manager.logs(follow=True)  # should not propagate

    def test_logs_no_follow_swallows_os_error(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)

        def raise_os(*a: object, **kw: object) -> None:
            raise OSError("docker not found")

        monkeypatch.setattr(wm_module.subprocess, "Popen", raise_os)
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
        monkeypatch.setattr(
            WorkerManager,
            "_pull_worker_image_with_progress",
            lambda self, image, timeout=600: True,
        )
        # Stub the full pull path (which makes live API calls via WorkerStartingAlertContext).
        monkeypatch.setattr(
            WorkerManager,
            "_pull_worker_image_with_progress",
            lambda self, image, timeout=600: True,
        )
        monkeypatch.setattr(
            WorkerManager, "_send_startup_failure_alert", lambda self, detail="": None
        )

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
        inspect_data = {
            "State": {"Status": "exited", "ExitCode": 1, "Error": ""},
            "RestartCount": 0,
        }
        with patch.object(wm_module, "docker_inspect", return_value=inspect_data):
            assert worker_manager._run_container() is False

    def test_returns_true_when_probe_window_exhausted_without_running(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Probe timeout returns True (resilient) since Docker will keep trying."""
        self._setup_run(worker_manager, tmp_config, monkeypatch)
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)
        inspect_data = {"State": {"Status": "created"}}
        with patch.object(wm_module, "docker_inspect", return_value=inspect_data):
            assert worker_manager._run_container() is True

    def test_proceeds_when_docker_rm_fails(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._setup_run(worker_manager, tmp_config, monkeypatch)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: False)
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())
        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            assert worker_manager._run_container() is True

    def test_retries_on_name_conflict(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import subprocess as _sp

        self._setup_run(worker_manager, tmp_config, monkeypatch)
        call_count: dict[str, int] = {"run": 0}
        rm_calls: list[str] = []

        def fake_run(*a: object, **kw: object) -> MagicMock:
            call_count["run"] += 1
            if call_count["run"] == 1:
                raise _sp.CalledProcessError(
                    125, "docker", stderr="Conflict. The container name is already in use."
                )
            return MagicMock()

        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)
        monkeypatch.setattr(
            wm_module, "docker_rm", lambda name, **kw: rm_calls.append(name) or True
        )
        # Verification loop after conflict rm must see "none" so the retry proceeds.
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "none")

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            assert worker_manager._run_container() is True

        assert call_count["run"] == 2
        assert len(rm_calls) == 2

    def test_conflict_retry_failure_sends_alert(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import subprocess as _sp

        self._setup_run(worker_manager, tmp_config, monkeypatch)
        alerts: list[str] = []
        monkeypatch.setattr(
            WorkerManager,
            "_send_startup_failure_alert",
            lambda self, detail="": alerts.append(detail),
        )
        monkeypatch.setattr(
            wm_module.subprocess,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(
                _sp.CalledProcessError(125, "docker", stderr="already in use")
            ),
        )

        assert worker_manager._run_container() is False
        assert any("docker run failed" in a for a in alerts)

    def test_conflict_retry_aborts_when_rm_fails(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If docker_rm times out during conflict recovery the run is not retried."""
        import subprocess as _sp

        self._setup_run(worker_manager, tmp_config, monkeypatch)
        alerts: list[str] = []
        monkeypatch.setattr(
            WorkerManager,
            "_send_startup_failure_alert",
            lambda self, detail="": alerts.append(detail),
        )
        call_count: dict[str, int] = {"run": 0}

        def fake_run(*a: object, **kw: object) -> MagicMock:
            call_count["run"] += 1
            raise _sp.CalledProcessError(125, "docker", stderr="Conflict. already in use")

        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)
        # Simulate docker_rm timing out during conflict recovery (returns False).
        rm_results = [True, False]  # initial rm ok, conflict-handler rm times out
        monkeypatch.setattr(
            wm_module, "docker_rm", lambda name, **kw: rm_results.pop(0) if rm_results else False
        )

        result = worker_manager._run_container()

        assert result is False
        # docker run must only have been called once (no retry when rm fails).
        assert call_count["run"] == 1
        assert any("docker rm timed out" in a for a in alerts)

    def test_conflict_retry_aborts_when_container_persists_after_rm(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the container is still present after rm (Docker restart race) we abort cleanly."""
        import subprocess as _sp

        self._setup_run(worker_manager, tmp_config, monkeypatch)
        alerts: list[str] = []
        monkeypatch.setattr(
            WorkerManager,
            "_send_startup_failure_alert",
            lambda self, detail="": alerts.append(detail),
        )

        def fake_run(*a: object, **kw: object) -> MagicMock:
            raise _sp.CalledProcessError(125, "docker", stderr="Conflict. already in use")

        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)
        # docker_container_status always returns "running" (Docker restart daemon keeps recreating).
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "running")

        result = worker_manager._run_container()

        assert result is False
        assert any("still exists" in a for a in alerts)

    def test_non_conflict_error_not_retried(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import subprocess as _sp

        self._setup_run(worker_manager, tmp_config, monkeypatch)
        call_count: dict[str, int] = {"run": 0}

        def fail_once(*a: object, **kw: object) -> None:
            call_count["run"] += 1
            raise _sp.CalledProcessError(1, "docker", stderr="OCI runtime exec failed")

        monkeypatch.setattr(wm_module.subprocess, "run", fail_once)
        assert worker_manager._run_container() is False
        assert call_count["run"] == 1


class TestWorkerStartupProbeWindow:
    """Startup probe window defaults to 30s and is configurable via env var."""

    def _setup_run(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "worker.py").write_text("pass")
        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
        monkeypatch.setattr(
            WorkerManager,
            "_pull_worker_image_with_progress",
            lambda self, image, timeout=600: True,
        )
        monkeypatch.setattr(
            WorkerManager,
            "_pull_worker_image_with_progress",
            lambda self, image, timeout=600: True,
        )
        monkeypatch.setattr(
            WorkerManager, "_send_startup_failure_alert", lambda self, detail="": None
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

    def test_default_probe_window_is_30_seconds(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._setup_run(worker_manager, tmp_config, monkeypatch)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())

        sleep_calls: list[float] = []
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: sleep_calls.append(s))

        inspect_data = {"State": {"Status": "created"}}
        with patch.object(wm_module, "docker_inspect", return_value=inspect_data):
            worker_manager._run_container()

        assert len(sleep_calls) == wm_module._DEFAULT_STARTUP_PROBE_SECONDS

    def test_env_var_overrides_probe_window(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._setup_run(worker_manager, tmp_config, monkeypatch)

        def fake_get_runtime_env_var(name: str, default: object = None) -> object:
            if name == "CYBERWAVE_WORKER_STARTUP_PROBE_SECONDS":
                return "10"
            return default

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            fake_get_runtime_env_var,
        )
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())

        sleep_calls: list[float] = []
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: sleep_calls.append(s))

        inspect_data = {"State": {"Status": "created"}}
        with patch.object(wm_module, "docker_inspect", return_value=inspect_data):
            worker_manager._run_container()

        assert len(sleep_calls) == 10

    def test_invalid_env_var_uses_default(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._setup_run(worker_manager, tmp_config, monkeypatch)

        def fake_get_runtime_env_var(name: str, default: object = None) -> object:
            if name == "CYBERWAVE_WORKER_STARTUP_PROBE_SECONDS":
                return "not-a-number"
            return default

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            fake_get_runtime_env_var,
        )
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())

        sleep_calls: list[float] = []
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: sleep_calls.append(s))

        inspect_data = {"State": {"Status": "created"}}
        with patch.object(wm_module, "docker_inspect", return_value=inspect_data):
            worker_manager._run_container()

        assert len(sleep_calls) == wm_module._DEFAULT_STARTUP_PROBE_SECONDS

    def test_container_becomes_running_mid_probe(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._setup_run(worker_manager, tmp_config, monkeypatch)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)

        inspect_results = [{"State": {"Status": "created"}}] * 10 + [
            {"State": {"Status": "running"}}
        ]
        with patch.object(wm_module, "docker_inspect", side_effect=inspect_results):
            assert worker_manager._run_container() is True

    def test_restarting_container_keeps_waiting(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Container in 'restarting' state should not be treated as failure."""
        self._setup_run(worker_manager, tmp_config, monkeypatch)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: (
                "5" if name == "CYBERWAVE_WORKER_STARTUP_PROBE_SECONDS" else default
            ),
        )
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)

        inspect_results = [{"State": {"Status": "restarting"}}] * 3 + [
            {"State": {"Status": "running"}}
        ]
        with patch.object(wm_module, "docker_inspect", side_effect=inspect_results):
            assert worker_manager._run_container() is True

    def test_exited_with_restarts_keeps_waiting(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exited container with RestartCount > 0 keeps waiting (Docker restart policy active)."""
        self._setup_run(worker_manager, tmp_config, monkeypatch)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: (
                "5" if name == "CYBERWAVE_WORKER_STARTUP_PROBE_SECONDS" else default
            ),
        )
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)

        inspect_results = [
            {"State": {"Status": "exited", "ExitCode": 1, "Error": ""}, "RestartCount": 1},
            {"State": {"Status": "restarting"}, "RestartCount": 1},
            {"State": {"Status": "running"}, "RestartCount": 1},
        ]
        with patch.object(wm_module, "docker_inspect", side_effect=inspect_results):
            assert worker_manager._run_container() is True

    def test_sends_alert_on_probe_timeout(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Alert is sent when startup probe times out."""
        self._setup_run(worker_manager, tmp_config, monkeypatch)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: (
                "3" if name == "CYBERWAVE_WORKER_STARTUP_PROBE_SECONDS" else default
            ),
        )
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)

        alert_calls: list[str] = []
        monkeypatch.setattr(
            WorkerManager,
            "_send_startup_failure_alert",
            lambda self, detail="": alert_calls.append(detail),
        )

        inspect_data = {"State": {"Status": "created"}}
        with patch.object(wm_module, "docker_inspect", return_value=inspect_data):
            result = worker_manager._run_container()

        assert result is True
        assert len(alert_calls) == 1
        assert "timed out" in alert_calls[0]

    def test_sends_alert_on_immediate_exit(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Alert is sent when container exits immediately with no restarts."""
        self._setup_run(worker_manager, tmp_config, monkeypatch)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)

        alert_calls: list[str] = []
        monkeypatch.setattr(
            WorkerManager,
            "_send_startup_failure_alert",
            lambda self, detail="": alert_calls.append(detail),
        )

        inspect_data = {
            "State": {"Status": "exited", "ExitCode": 137, "Error": "OOMKilled"},
            "RestartCount": 0,
        }
        with patch.object(wm_module, "docker_inspect", return_value=inspect_data):
            result = worker_manager._run_container()

        assert result is False
        assert len(alert_calls) == 1
        assert "exit_code=137" in alert_calls[0]


# ---------------------------------------------------------------------------
# WorkerManager.start — health monitor record_start integration
# ---------------------------------------------------------------------------


class TestWorkerManagerStartHealthIntegration:
    def test_start_calls_record_start_on_success(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cyberwave_edge_core.worker_health import WorkerHealthMonitor

        monitor = WorkerHealthMonitor(container_name=worker_manager.container_name)
        worker_manager.set_health_monitor(monitor)

        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "worker.py").write_text("pass")

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
        monkeypatch.setattr(wm_module.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)
        monkeypatch.setattr(
            WorkerManager,
            "_pull_worker_image_with_progress",
            lambda self, image, timeout=600: True,
        )

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        with (
            # start() now checks status twice: once before the lock and once inside it.
            patch.object(wm_module, "docker_container_status", side_effect=["none", "none"]),
            patch.object(
                wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
            ),
        ):
            result = worker_manager.start()

        assert result is True
        # record_start sets _last_start_time; uptime will be non-None in check()
        state = monitor.check(container_status="running")
        assert state.uptime_seconds is not None


# ---------------------------------------------------------------------------
# resolve_worker_image — env / override resolution
# ---------------------------------------------------------------------------


class TestResolveWorkerImage:
    """``resolve_worker_image`` translates env state into the worker image
    reference used by ``WorkerManager._run_container``.

    Resolution order (most specific → least):
      1. ``CYBERWAVE_WORKER_IMAGE`` — explicit local override (mirrors
         ``driver_overrides`` for the worker side).
      2. ``CYBERWAVE_ENVIRONMENT`` — non-production envs map to the
         matching tag (``dev`` → ``...:dev``).
      3. Production / unknown → ``...:latest``.
    """

    @staticmethod
    def _runtime_env(values: dict[str, str | None]):
        def _get(name: str, default=None):
            return values.get(name, default)

        return _get

    def test_unset_falls_back_to_latest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            self._runtime_env({}),
        )
        assert wm_module.resolve_worker_image() == "cyberwaveos/edge-ml-worker:latest"

    def test_environment_dev_maps_to_dev_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            self._runtime_env({"CYBERWAVE_ENVIRONMENT": "dev"}),
        )
        assert wm_module.resolve_worker_image() == "cyberwaveos/edge-ml-worker:dev"

    def test_environment_production_uses_latest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            self._runtime_env({"CYBERWAVE_ENVIRONMENT": "production"}),
        )
        assert wm_module.resolve_worker_image() == "cyberwaveos/edge-ml-worker:latest"

    def test_explicit_override_wins_over_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even when CYBERWAVE_ENVIRONMENT=dev (which would normally pick
        # ``:dev``), the operator-specified image must take precedence.
        # This is the path used by the `:local-gpu` SDK hot-fix loop —
        # `docker commit` to a tag the registry doesn't have, point the
        # override at it, edge-core's pull falls back to local.
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            self._runtime_env(
                {
                    "CYBERWAVE_ENVIRONMENT": "dev",
                    "CYBERWAVE_WORKER_IMAGE": "cyberwaveos/edge-ml-worker:local-gpu",
                }
            ),
        )
        assert wm_module.resolve_worker_image() == "cyberwaveos/edge-ml-worker:local-gpu"

    def test_override_passes_through_third_party_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Override is forwarded verbatim — no rewriting of registry / tag.
        # Operator owns correctness (including any ``-gpu`` suffix when
        # bypassing the cyberwaveos/edge-ml-worker auto-suffix path).
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            self._runtime_env({"CYBERWAVE_WORKER_IMAGE": "myregistry.io:5000/cw/worker:custom"}),
        )
        assert wm_module.resolve_worker_image() == "myregistry.io:5000/cw/worker:custom"

    def test_blank_override_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Whitespace-only override falls through to the env-name path so a
        # stale empty value doesn't accidentally collapse to ``:latest``.
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            self._runtime_env(
                {"CYBERWAVE_WORKER_IMAGE": "   ", "CYBERWAVE_ENVIRONMENT": "staging"}
            ),
        )
        assert wm_module.resolve_worker_image() == "cyberwaveos/edge-ml-worker:staging"


# ---------------------------------------------------------------------------
# Mutable-tag detection and force-pull behavior
# ---------------------------------------------------------------------------


class TestImageTagMutability:
    """:dev / :local / :latest etc. must be re-pulled on every worker start."""

    @pytest.mark.parametrize(
        "image",
        [
            "cyberwaveos/edge-ml-worker",
            "cyberwaveos/edge-ml-worker:latest",
            "cyberwaveos/edge-ml-worker:dev",
            "cyberwaveos/edge-ml-worker:dev-gpu",
            "cyberwaveos/edge-ml-worker:dev-cpu",
            "cyberwaveos/edge-ml-worker:local",
            "cyberwaveos/edge-ml-worker:local-arm64",
            "cyberwaveos/edge-ml-worker:staging",
            "cyberwaveos/edge-ml-worker:nightly",
            "cyberwaveos/edge-ml-worker:edge",
            "cyberwaveos/edge-ml-worker:main",
            "cyberwaveos/edge-ml-worker:master",
            # Registry with a port, no explicit tag → docker treats as :latest.
            "myregistry.io:5000/cyberwaveos/edge-ml-worker",
            # Registry with a port AND a mutable tag.
            "myregistry.io:5000/cyberwaveos/edge-ml-worker:dev-gpu",
        ],
    )
    def test_mutable_tags_are_detected(self, image: str) -> None:
        assert wm_module._image_tag_is_mutable(image) is True

    @pytest.mark.parametrize(
        "image",
        [
            "cyberwaveos/edge-ml-worker:v1.2.3",
            "cyberwaveos/edge-ml-worker:1.2.3",
            "cyberwaveos/edge-ml-worker:20260501",
            "cyberwaveos/edge-ml-worker:release-v1.2.3",
            "cyberwaveos/edge-ml-worker@sha256:" + "0" * 64,
            # Registry with a port AND an immutable tag.
            "myregistry.io:5000/cyberwaveos/edge-ml-worker:v1.2.3",
        ],
    )
    def test_immutable_tags_are_detected(self, image: str) -> None:
        assert wm_module._image_tag_is_mutable(image) is False


class TestEnsureImagePulledForceRePullsMutableTags:
    """Regression: stale ``:dev-gpu`` was used silently before this fix."""

    def _patch_inspect_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            wm_module,
            "_docker_image_present",
            lambda image: True,
        )

    def _patch_inspect_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            wm_module,
            "_docker_image_present",
            lambda image: False,
        )

    def test_mutable_tag_always_pulls_even_when_local_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_inspect_present(monkeypatch)
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)
        assert WorkerManager._ensure_image_pulled("cyberwaveos/edge-ml-worker:dev-gpu") is True
        assert calls == [["docker", "pull", "cyberwaveos/edge-ml-worker:dev-gpu"]]

    def test_immutable_tag_skips_pull_when_local_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_inspect_present(monkeypatch)
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return MagicMock(returncode=0)

        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)
        assert WorkerManager._ensure_image_pulled("cyberwaveos/edge-ml-worker:v1.2.3") is True
        # Immutable + local present → no docker pull issued.
        assert calls == []

    def test_immutable_tag_pulls_when_local_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_inspect_absent(monkeypatch)
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)
        assert WorkerManager._ensure_image_pulled("cyberwaveos/edge-ml-worker:v1.2.3") is True
        assert calls == [["docker", "pull", "cyberwaveos/edge-ml-worker:v1.2.3"]]

    def test_pull_failure_with_local_copy_falls_back_to_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as _sp

        self._patch_inspect_present(monkeypatch)

        def raise_cpe(*a, **kw):
            raise _sp.CalledProcessError(1, "docker", stderr="network unreachable")

        monkeypatch.setattr(wm_module.subprocess, "run", raise_cpe)
        # Mutable tag, local copy present, registry unreachable → keep going.
        assert WorkerManager._ensure_image_pulled("cyberwaveos/edge-ml-worker:dev-gpu") is True

    def test_pull_failure_without_local_copy_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as _sp

        self._patch_inspect_absent(monkeypatch)

        def raise_cpe(*a, **kw):
            raise _sp.CalledProcessError(1, "docker", stderr="not found")

        monkeypatch.setattr(wm_module.subprocess, "run", raise_cpe)
        assert WorkerManager._ensure_image_pulled("cyberwaveos/edge-ml-worker:dev-gpu") is False

    def test_pull_timeout_with_local_copy_falls_back_to_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as _sp

        self._patch_inspect_present(monkeypatch)

        def raise_timeout(*a, **kw):
            raise _sp.TimeoutExpired("docker", 60)

        monkeypatch.setattr(wm_module.subprocess, "run", raise_timeout)
        assert WorkerManager._ensure_image_pulled("cyberwaveos/edge-ml-worker:dev-gpu") is True

    def test_pull_timeout_without_local_copy_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as _sp

        self._patch_inspect_absent(monkeypatch)

        def raise_timeout(*a, **kw):
            raise _sp.TimeoutExpired("docker", 60)

        monkeypatch.setattr(wm_module.subprocess, "run", raise_timeout)
        assert WorkerManager._ensure_image_pulled("cyberwaveos/edge-ml-worker:dev-gpu") is False


class TestWorkerStartupFailureAlertWiring:
    def test_send_startup_failure_alert_skips_while_image_pull_in_progress(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker_manager._twin_uuids = ["twin-uuid-1"]
        alerts: list[str] = []

        def _capture(**kwargs: object) -> None:
            alerts.append(str(kwargs.get("error", "")))

        monkeypatch.setattr(
            "cyberwave_edge_core.startup._send_worker_start_failure_alerts",
            lambda **kwargs: _capture(**kwargs),
        )
        monkeypatch.setattr(
            wm_module,
            "is_worker_image_pull_in_progress",
            lambda image=None: True,
        )

        worker_manager._send_startup_failure_alert("image unavailable and no local copy")
        assert alerts == []

    def test_pull_worker_image_with_progress_waits_for_in_flight_pull(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker_manager._twin_uuids = ["twin-uuid-1"]
        calls: list[str] = []

        def _fake_once(self: WorkerManager, image: str, timeout: int = 600) -> bool:
            calls.append(image)
            time.sleep(0.05)
            return True

        monkeypatch.setattr(
            WorkerManager,
            "_pull_worker_image_with_progress_once",
            _fake_once,
        )

        results: list[bool] = []

        def _owner() -> None:
            results.append(
                worker_manager._pull_worker_image_with_progress(
                    "cyberwaveos/edge-ml-worker:dev"
                )
            )

        def _waiter() -> None:
            results.append(
                worker_manager._pull_worker_image_with_progress(
                    "cyberwaveos/edge-ml-worker:dev"
                )
            )

        owner = threading.Thread(target=_owner)
        waiter = threading.Thread(target=_waiter)
        owner.start()
        time.sleep(0.01)
        waiter.start()
        owner.join(timeout=5)
        waiter.join(timeout=5)

        assert results == [True, True]
        assert calls == ["cyberwaveos/edge-ml-worker:dev"]
