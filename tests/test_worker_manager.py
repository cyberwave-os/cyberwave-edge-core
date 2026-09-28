"""Unit tests for WorkerManager (worker_manager.py).

Covers:
- Worker container env var injection
- Volume mount args
- GPU detection flag
- Zenoh env var helper
- start() skips container when no worker files present
- status() reflects correct state
- container_started_after_worker_writes: the redundant-hot-reload predicate
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import cyberwave_edge_core.driver_selection as driver_selection
import cyberwave_edge_core.worker_manager as wm_module
from cyberwave_edge_core._clock import FakeMonotonicClock
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
            if name == "ZENOH_MODE":
                return "client"
            return default

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            fake_get_runtime_env_var,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        env = worker_manager._build_env_vars()
        assert env["ZENOH_CONNECT"] == "tcp/192.168.1.10:7447"
        assert env["ZENOH_MODE"] == "client"

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
    monkeypatch.setattr(wm_module, "docker_image_id", lambda image: None)
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

    def test_apply_hailo_image_tag_leaves_jetson_alone(self) -> None:
        # A single host is never both a Jetson and a Pi-with-AI-HAT, but
        # the guard exists so a manual ``CYBERWAVE_WORKER_IMAGE=<...>-jetson``
        # override on a Hailo host doesn't get double-suffixed.
        assert (
            wm_module._apply_hailo_image_tag("cyberwaveos/edge-ml-worker:dev-jetson")
            == "cyberwaveos/edge-ml-worker:dev-jetson"
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


class TestJetsonHelpers:
    """Pure-function tests for the Jetson helpers (no docker subprocess)."""

    def test_apply_jetson_image_tag_rewrites_known_tag(self) -> None:
        assert (
            wm_module._apply_jetson_image_tag("cyberwaveos/edge-ml-worker:latest")
            == "cyberwaveos/edge-ml-worker:latest-jetson"
        )
        assert (
            wm_module._apply_jetson_image_tag("cyberwaveos/edge-ml-worker:dev")
            == "cyberwaveos/edge-ml-worker:dev-jetson"
        )

    def test_apply_jetson_image_tag_leaves_existing_jetson_alone(self) -> None:
        assert (
            wm_module._apply_jetson_image_tag("cyberwaveos/edge-ml-worker:dev-jetson")
            == "cyberwaveos/edge-ml-worker:dev-jetson"
        )

    def test_apply_jetson_image_tag_leaves_gpu_alone(self) -> None:
        # Belt-and-braces: an operator override pointing at ``-gpu`` on a
        # Jetson host doesn't get a second suffix. The image will fail to
        # pull (wrong arch) but the resolver won't compound the confusion.
        assert (
            wm_module._apply_jetson_image_tag("cyberwaveos/edge-ml-worker:latest-gpu")
            == "cyberwaveos/edge-ml-worker:latest-gpu"
        )

    def test_apply_jetson_image_tag_leaves_hailo_alone(self) -> None:
        assert (
            wm_module._apply_jetson_image_tag("cyberwaveos/edge-ml-worker:dev-hailo")
            == "cyberwaveos/edge-ml-worker:dev-hailo"
        )

    def test_apply_jetson_image_tag_leaves_custom_image_alone(self) -> None:
        assert (
            wm_module._apply_jetson_image_tag("myregistry.local/cyberwave/custom:dev")
            == "myregistry.local/cyberwave/custom:dev"
        )


class TestWorkerManagerJetson:
    """Verify Jetson image tag rewrite + ``--runtime nvidia`` passthrough.

    On a Jetson host the L4T install ships nvidia-container-runtime, so
    ``docker_has_nvidia_runtime()`` also returns True. The resolver has to
    pick the ``-jetson`` sibling anyway because the amd64 ``-gpu`` blob has
    the wrong ``libcuda.so`` ABI for Tegra. These tests pin that precedence
    with both signals asserted on, and pin the JetPack 7 floor below which
    the whole GPU path is skipped.
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

    def test_jetson_host_uses_jetson_image_not_gpu(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        # Both signals on: real Jetson hosts satisfy both. The resolver
        # must still land on ``-jetson``, not ``-gpu``.
        monkeypatch.setattr(wm_module, "is_jetson", lambda: True)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: False)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert docker_run_cmd[docker_run_cmd.index("--runtime") + 1] == "nvidia"
        assert "NVIDIA_VISIBLE_DEVICES=all" in docker_run_cmd
        assert "--gpus" not in docker_run_cmd
        # No Hailo device passthrough — Jetson-and-Hailo isn't a real host,
        # but the guard is here to catch a resolver refactor that leaks the
        # Hailo branch into the Jetson code path.
        assert "--device" not in docker_run_cmd
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest-jetson"

    def test_jetson_image_not_double_suffixed_when_already_selected(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        worker_manager._image = "cyberwaveos/edge-ml-worker:latest-jetson"

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "is_jetson", lambda: True)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: False)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            assert worker_manager._run_container() is True

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert docker_run_cmd[docker_run_cmd.index("--runtime") + 1] == "nvidia"
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest-jetson"

    def test_custom_image_override_left_untouched_on_jetson(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``CYBERWAVE_WORKER_IMAGE`` overrides bypass the ``-jetson`` rewrite.

        Same shape as :func:`TestWorkerManagerHailo.test_custom_image_override_left_untouched`
        and the GPU counterpart: ``--runtime nvidia`` still lands (the container
        runtime needs it regardless of what image the operator chose), but the
        image reference is left alone.
        """
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        worker_manager._image = "myregistry.local/cyberwave/custom-worker:dev"

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "is_jetson", lambda: True)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: False)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert docker_run_cmd[docker_run_cmd.index("--runtime") + 1] == "nvidia"
        assert docker_run_cmd[-1] == "myregistry.local/cyberwave/custom-worker:dev"

    @pytest.fixture(autouse=True)
    def _stub_image_pull(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep these off the network — the tags here are real public images.

        Every assertion in this class is about the ``docker run`` argv, so the
        pull is pure latency (and a ~10 min class runtime without this).
        """
        monkeypatch.setattr(
            wm_module.WorkerManager,
            "_pull_worker_image_with_progress",
            lambda self, image, timeout=600: True,
        )

    def test_jetpack_6_host_falls_back_to_cpu_image(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """L4T r36 can't load the ``-jetson`` wheels — take the CPU image.

        The plain tag is a multi-arch build, so arm64 CPU inference works on
        the same host. Falling through to the ``--gpus all`` branch would be
        worse than useless: that rewrites to the amd64 ``-gpu`` blob.
        """
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "is_jetson", lambda: True)
        monkeypatch.setattr(wm_module, "jetson_accelerator_supported", lambda: False)
        monkeypatch.setattr(wm_module, "jetson_l4t_major", lambda: 36)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: False)

        with caplog.at_level(logging.WARNING, logger="cyberwave_edge_core.worker_manager"):
            with patch.object(
                wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
            ):
                worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--runtime" not in docker_run_cmd
        assert "--gpus" not in docker_run_cmd
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest"
        assert any("L4T r36" in record.getMessage() for record in caplog.records)

    def test_jetpack_6_keeps_runtime_for_pinned_custom_image(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The L4T floor gates our image, not somebody else's.

        A ``CYBERWAVE_WORKER_IMAGE`` pinned to a third-party build got
        ``--runtime nvidia`` before the floor existed, and nothing here can
        tell whether that build is JetPack-6-correct. Withdrawing the runtime
        would silently halve its inference speed on upgrade and leave the
        operator hunting for an opt-out env var they have no reason to read.
        """
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        worker_manager._image = "myregistry.local/cyberwave/custom-worker:dev"

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "is_jetson", lambda: True)
        monkeypatch.setattr(wm_module, "jetson_accelerator_supported", lambda: False)
        monkeypatch.setattr(wm_module, "jetson_l4t_major", lambda: 36)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: False)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert docker_run_cmd[docker_run_cmd.index("--runtime") + 1] == "nvidia"
        assert "NVIDIA_VISIBLE_DEVICES=all" in docker_run_cmd
        assert docker_run_cmd[-1] == "myregistry.local/cyberwave/custom-worker:dev"

    def test_jetpack_6_demotes_a_pinned_jetson_tag_to_the_plain_one(
        self,
        worker_manager: WorkerManager,
        tmp_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Pinning ``-jetson`` on an r36 board lands on the plain tag.

        Our own image is the one the floor speaks for, however it was chosen.
        Demoting the reference as well as the GPU args keeps the warning
        honest: the image it names is the one the container starts from.
        """
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        worker_manager._image = "cyberwaveos/edge-ml-worker:latest-jetson"

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "is_jetson", lambda: True)
        monkeypatch.setattr(wm_module, "jetson_accelerator_supported", lambda: False)
        monkeypatch.setattr(wm_module, "jetson_l4t_major", lambda: 36)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: False)

        with caplog.at_level(logging.WARNING, logger="cyberwave_edge_core.worker_manager"):
            with patch.object(
                wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
            ):
                worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--runtime" not in docker_run_cmd
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest"
        assert any(
            "Starting cyberwaveos/edge-ml-worker:latest on CPU" in record.getMessage()
            for record in caplog.records
        ), "the warning must name the image the container actually starts from"

    def test_jetpack_6_override_restores_gpu_path(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``CYBERWAVE_JETSON_ALLOW_UNSUPPORTED_L4T`` opts an r36 host back in.

        Runs the real predicate over a stubbed r36 release file rather than
        patching :func:`jetson_accelerator_supported` away.
        """
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        release = tmp_config / "nv_tegra_release"
        release.write_text("# R36 (release), REVISION: 4.7, BOARD: generic\n")
        monkeypatch.setattr(driver_selection, "_NV_TEGRA_RELEASE", release)
        monkeypatch.setattr(driver_selection, "_l4t_major", None)
        monkeypatch.setattr(driver_selection, "_l4t_major_probed", False)

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        # After _stub_runtime_env: it swaps os.environ for an empty dict.
        monkeypatch.setenv("CYBERWAVE_JETSON_ALLOW_UNSUPPORTED_L4T", "1")
        monkeypatch.setattr(wm_module, "is_jetson", lambda: True)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: False)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert docker_run_cmd[docker_run_cmd.index("--runtime") + 1] == "nvidia"
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest-jetson"

    def test_jetson_image_not_gpu_suffixed_on_nvidia_host(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``-jetson`` override on a non-Jetson NVIDIA host must not become ``-jetson-gpu``.

        Jetson implies GPU — there is no non-GPU Jetson variant.  If an operator
        pins ``CYBERWAVE_WORKER_IMAGE=…:dev-jetson`` on a machine where
        ``is_jetson()`` returns False but ``docker_has_nvidia_runtime()`` is True
        (e.g. a misconfigured host or a future edge-core binary running a
        pre-tagged image), the ``-gpu`` suffix must not be appended.
        """
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        worker_manager._image = "cyberwaveos/edge-ml-worker:dev-jetson"

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "is_jetson", lambda: False)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: False)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            assert worker_manager._run_container() is True

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--gpus" in docker_run_cmd
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:dev-jetson"

    def test_no_jetson_args_when_not_jetson_host(
        self, worker_manager: WorkerManager, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-Jetson host: base tag is preserved (or ``-gpu`` if NVIDIA is present)."""
        workers_dir = tmp_config / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")

        run_calls: list[list[str]] = []
        self._stub_docker_run(monkeypatch, run_calls)
        _stub_runtime_env(monkeypatch)
        monkeypatch.setattr(wm_module, "is_jetson", lambda: False)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: False)

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            worker_manager._run_container()

        docker_run_cmd = next((c for c in run_calls if c and c[0] == "docker" and "run" in c), None)
        assert docker_run_cmd is not None
        assert "--gpus" not in docker_run_cmd
        assert docker_run_cmd[-1] == "cyberwaveos/edge-ml-worker:latest"


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
            return {"ZENOH_CONNECT": "tcp/router:7447", "ZENOH_MODE": "client"}.get(name)

        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            fake_get_runtime_env_var,
        )
        env = get_zenoh_env_vars()
        assert env["ZENOH_CONNECT"] == "tcp/router:7447"
        assert env["ZENOH_MODE"] == "client"

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

    def test_explicit_network_wins_on_every_platform(
        self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller naming a network knows where its producer lives.

        cyberwave-sim runs the plant as a container on a user-defined bridge, so
        a worker on host networking never scouts it over multicast.
        """
        manager = WorkerManager(
            config_dir=tmp_config,
            environment_uuid="aabbccdd-eeff-0011-2233-445566778899",
            token="testtoken1234",
            network="cyberwave-sim_default",
        )
        for platform_name in ("Linux", "Darwin"):
            monkeypatch.setattr(wm_module.platform, "system", lambda: platform_name)
            assert manager._build_network_args() == [
                "--network",
                "cyberwave-sim_default",
            ]

    def test_blank_network_falls_back_to_the_platform_default(
        self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unset Terraform/CI variable renders as an empty string."""
        manager = WorkerManager(
            config_dir=tmp_config,
            environment_uuid="aabbccdd-eeff-0011-2233-445566778899",
            token="testtoken1234",
            network="   ",
        )
        monkeypatch.setattr(wm_module.platform, "system", lambda: "Linux")
        assert manager._build_network_args() == ["--network", "host"]


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
        # The spec-hash image lookup is its own subprocess call; keep it out of
        # the docker run call counts these tests assert on.
        monkeypatch.setattr(wm_module, "docker_image_id", lambda image: None)
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
        # The spec-hash image lookup is its own subprocess call; keep it out of
        # the docker run call counts these tests assert on.
        monkeypatch.setattr(wm_module, "docker_image_id", lambda image: None)
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
    """:dev / :latest etc. must be re-pulled on every worker start."""

    @pytest.mark.parametrize(
        "image",
        [
            "cyberwaveos/edge-ml-worker",
            "cyberwaveos/edge-ml-worker:latest",
            "cyberwaveos/edge-ml-worker:dev",
            "cyberwaveos/edge-ml-worker:dev-gpu",
            "cyberwaveos/edge-ml-worker:dev-cpu",
            "cyberwaveos/edge-ml-worker:dev-jetson",
            "cyberwaveos/edge-ml-worker:dev-hailo",
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
            # Never published, so a pull could only fail.
            "cyberwaveos/edge-ml-worker:local",
            "cyberwaveos/edge-ml-worker:local-arm64",
            "cyberwaveos/edge-ml-worker:local-jetson",
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

        def _fake_once(
            self: WorkerManager,
            image: str,
            timeout: int = 600,
            *,
            alert_ctxs: object = None,
            is_final_attempt: bool = True,
        ) -> bool:
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
                worker_manager._pull_worker_image_with_progress("cyberwaveos/edge-ml-worker:dev")
            )

        def _waiter() -> None:
            results.append(
                worker_manager._pull_worker_image_with_progress("cyberwaveos/edge-ml-worker:dev")
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


def _started_at(value_ns: int) -> dict:
    """A minimal ``docker inspect`` payload whose StartedAt is *value_ns*."""
    whole, frac = divmod(value_ns, 1_000_000_000)
    stamp = datetime.fromtimestamp(whole, tz=timezone.utc)
    return {
        "State": {
            "Status": "running",
            "StartedAt": f"{stamp.strftime('%Y-%m-%dT%H:%M:%S')}.{frac:09d}Z",
        }
    }


class TestNewestWorkerWriteNs:
    def test_missing_directory_has_no_answer(self, tmp_path: Path):
        assert wm_module._newest_worker_write_ns(tmp_path / "absent") is None

    def test_empty_directory_falls_back_to_the_directory_itself(self, tmp_path: Path):
        assert wm_module._newest_worker_write_ns(tmp_path) == tmp_path.stat().st_mtime_ns

    def test_takes_the_newest_worker_file(self, tmp_path: Path):
        (tmp_path / "wf_old.py").write_text("# old\n")
        time.sleep(0.02)
        newer = tmp_path / "wf_new.py"
        newer.write_text("# new\n")
        assert wm_module._newest_worker_write_ns(tmp_path) >= newer.stat().st_mtime_ns

    def test_unlink_bumps_the_directory_mtime(self, tmp_path: Path):
        """The OS behaviour the removal case depends on, asserted rather than assumed.

        If a filesystem ever failed to bump the directory mtime on unlink, a
        deactivate would look like "nothing changed" and the worker would keep
        running the workflow the user just switched off. That is the one
        failure this whole predicate must never produce, so it is pinned here.
        """
        worker = tmp_path / "wf_deadbeef.py"
        worker.write_text("# worker\n")
        before = wm_module._newest_worker_write_ns(tmp_path)
        time.sleep(0.02)
        worker.unlink()
        after = wm_module._newest_worker_write_ns(tmp_path)
        assert before is not None and after is not None
        assert after > before


class TestContainerStartedAfterWorkerWrites:
    """The predicate behind skipping a redundant hot-reload.

    Measured on a Jetson: 21 of 25 ``immediate-worker-sync`` restarts fired a
    median 0.04 s *after* the container came up, i.e. they bounced a container
    that already had the new ``wf_*.py`` and threw away a boot in progress.

    Every case that cannot prove the container is newer than the writes must
    return False, because False is the unconditional restart that shipped
    before this check existed.
    """

    @pytest.fixture
    def workers_dir(self, worker_manager: WorkerManager) -> Path:
        path = worker_manager._workers_dir()
        path.mkdir(parents=True, exist_ok=True)
        (path / "wf_deadbeef.py").write_text("# worker\n")
        return path

    def test_container_newer_than_the_writes_is_current(
        self, worker_manager: WorkerManager, workers_dir: Path
    ):
        newest = wm_module._newest_worker_write_ns(workers_dir)
        assert newest is not None
        assert (
            worker_manager.container_started_after_worker_writes(
                _started_at(newest + 5_000_000_000)
            )
            is True
        )

    def test_container_older_than_the_writes_is_stale(
        self, worker_manager: WorkerManager, workers_dir: Path
    ):
        newest = wm_module._newest_worker_write_ns(workers_dir)
        assert newest is not None
        assert (
            worker_manager.container_started_after_worker_writes(
                _started_at(newest - 5_000_000_000)
            )
            is False
        )

    def test_equal_timestamps_are_stale(self, worker_manager: WorkerManager, workers_dir: Path):
        # ext4 stamps from a coarse clock, so a write landing in the same tick
        # as the container start shares its mtime. Equality is not an ordering.
        newest = wm_module._newest_worker_write_ns(workers_dir)
        assert newest is not None
        assert worker_manager.container_started_after_worker_writes(_started_at(newest)) is False

    def test_a_single_nanosecond_is_not_rounded_away(
        self, worker_manager: WorkerManager, workers_dir: Path
    ):
        """Regression: comparing floats or datetimes got this backwards.

        ``st_mtime`` is a float64 with no bits left for nanoseconds at epoch
        magnitude, and ``datetime`` truncates to microseconds. A draft of this
        predicate reported "already current" for a write that landed 1 ns
        after the container started -- i.e. it skipped a restart the edge
        needed, which is the unsafe direction.
        """
        newest = wm_module._newest_worker_write_ns(workers_dir)
        assert newest is not None
        assert (
            worker_manager.container_started_after_worker_writes(_started_at(newest - 1)) is False
        )
        assert worker_manager.container_started_after_worker_writes(_started_at(newest + 1)) is True

    def test_removing_the_last_worker_is_stale(
        self, worker_manager: WorkerManager, workers_dir: Path
    ):
        """The deactivate path, which needs no special-casing.

        The container legitimately has the file, so the predicate says
        "current". Then the unlink bumps the directory mtime past StartedAt
        and it says "stale" -- which is required, because the unlinked module
        stays imported and would keep executing the deactivated workflow.
        """
        newest = wm_module._newest_worker_write_ns(workers_dir)
        assert newest is not None
        inspect = _started_at(newest + 1_000_000)
        assert worker_manager.container_started_after_worker_writes(inspect) is True

        time.sleep(0.05)
        (workers_dir / "wf_deadbeef.py").unlink()
        assert worker_manager.container_started_after_worker_writes(inspect) is False

    @pytest.mark.parametrize(
        "inspect",
        [
            None,
            {},
            {"State": None},
            {"State": "not-a-dict"},
            {"State": {}},
            {"State": {"StartedAt": ""}},
            {"State": {"StartedAt": "0001-01-01T00:00:00Z"}},
            {"State": {"StartedAt": "not a timestamp"}},
        ],
        ids=[
            "inspect-failed",
            "no-state",
            "null-state",
            "state-not-a-dict",
            "no-started-at",
            "empty-started-at",
            "never-started",
            "unparseable",
        ],
    )
    def test_uncertain_inputs_fall_through_to_a_restart(
        self, worker_manager: WorkerManager, workers_dir: Path, inspect
    ):
        assert worker_manager.container_started_after_worker_writes(inspect) is False

    def test_missing_workers_dir_falls_through_to_a_restart(self, worker_manager: WorkerManager):
        # No ``workers_dir`` fixture, so the directory was never created.
        assert not worker_manager._workers_dir().exists()
        assert (
            worker_manager.container_started_after_worker_writes(_started_at(time.time_ns()))
            is False
        )


class _RecordingAlertCtx:
    """Stand-in for ``WorkerStartingAlertContext`` that records its lifecycle."""

    instances: list["_RecordingAlertCtx"] = []

    def __init__(self, *, twin_uuid: str, image: str, container_name: str) -> None:
        self.twin_uuid = twin_uuid
        self.image = image
        self.container_name = container_name
        self.created = False
        self.resolved = False
        self.failed: str | None = None
        self.patches: list[dict] = []
        _RecordingAlertCtx.instances.append(self)

    def create(self) -> None:
        self.created = True

    def update_metadata(self, patch: dict, *, force: bool = False) -> None:
        # The real context drops its alert handle on resolve and every method
        # short-circuits afterwards; mirror that, or a test cannot tell a live
        # card from a dead one still sitting in the list.
        if self.resolved:
            return
        self.patches.append(patch)

    def resolve(self) -> None:
        self.resolved = True

    def mark_failed_and_resolve(self, description: str, *, phase: str = "pull_failed") -> None:
        if self.resolved:
            return
        self.failed = phase
        self.resolved = True

    @property
    def phases(self) -> list[str]:
        return [p["phase"] for p in self.patches if "phase" in p]


class TestWorkerStartingNotification:
    """The ``worker_starting`` card must cover the whole startup.

    It used to be raised inside the image pull and resolved the moment the
    pull finished, so a user watching a worker come up saw nothing at all
    while the container was actually being created and booted — and nothing
    whatsoever when the image was already on disk.
    """

    @pytest.fixture(autouse=True)
    def _stub_alerts(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Patch the attribute rather than the module: ``startup`` imports other
        # names from ``utils``, so swapping the whole module breaks the import.
        import cyberwave_edge_core.utils as utils_module

        _RecordingAlertCtx.instances = []
        monkeypatch.setattr(
            utils_module, "WorkerStartingAlertContext", _RecordingAlertCtx, raising=False
        )
        # Keep the worker status file inside the test's tmp dir rather than
        # the real ``/run/cyberwave``.
        monkeypatch.setattr(wm_module, "default_status_dir", lambda _cfg: tmp_path / "run")

    @staticmethod
    def _publish_status(manager: WorkerManager, phase: str, **extra: object) -> None:
        """Write what the worker inside the container would have written."""
        status_path = manager._status_path()
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps({"version": 1, "phase": phase, **extra}), encoding="utf-8"
        )

    def _stub_container_start(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        status: str = "running",
    ) -> None:
        monkeypatch.setattr(wm_module.subprocess, "run", lambda cmd, **kw: MagicMock(returncode=0))
        monkeypatch.setattr(wm_module, "docker_rm", lambda *_a, **_kw: True)
        monkeypatch.setattr(
            wm_module, "docker_inspect", lambda *_a, **_kw: {"State": {"Status": status}}
        )
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: False)
        monkeypatch.setattr(wm_module, "_hailo_device_present", lambda: False)
        monkeypatch.setattr(wm_module, "is_jetson", lambda: False)
        _stub_runtime_env(monkeypatch)

    def test_alert_opens_before_the_pull_and_survives_it(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pull must adopt the open card, not raise and retire its own."""
        self._stub_container_start(monkeypatch)
        pull_state: dict[str, object] = {}

        def fake_pull(image: str, timeout: int = 600) -> bool:
            # Snapshot what the card looked like mid-pull.
            pull_state["open"] = len(worker_manager._card.ctxs)
            pull_state["resolved"] = [c.resolved for c in _RecordingAlertCtx.instances]
            return True

        monkeypatch.setattr(worker_manager, "_pull_worker_image_with_progress", fake_pull)

        assert worker_manager._run_container() is True

        # One card per linked twin, opened before the pull ran.
        assert len(_RecordingAlertCtx.instances) == 2
        assert pull_state["open"] == 2
        assert pull_state["resolved"] == [False, False]
        assert all(ctx.created for ctx in _RecordingAlertCtx.instances)

    def test_card_follows_the_alert_audience_not_the_sync_set(
        self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One worker container is one startup, not one per twin on the node.

        cyberwave-sim hands the manager every twin in the environment so a
        workflow activated mid-run is reachable. Carding all of them announced
        a worker starting on a table and a pedestal — twins that have no
        worker and no workflow to lose.
        """
        manager = WorkerManager(
            config_dir=tmp_config,
            environment_uuid="aabbccdd-eeff-0011-2233-445566778899",
            token="testtoken1234",
            twin_uuids=["robot", "table", "pedestal"],
            alert_twin_uuids=["robot"],
        )
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True)

        assert manager._run_container() is True

        assert [ctx.twin_uuid for ctx in _RecordingAlertCtx.instances] == ["robot"]
        # Narrowing the audience must not narrow what the worker syncs, or a
        # workflow pointed at the table later would run nowhere.
        assert manager._build_env_vars()["CYBERWAVE_TWIN_UUIDS"] == "robot,table,pedestal"

    def test_no_card_when_the_audience_is_empty(
        self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicitly empty audience is an answer, not a missing value:
        a simulation started before any workflow exists has nobody waiting on
        a run, so there is no progress to report."""
        manager = WorkerManager(
            config_dir=tmp_config,
            environment_uuid="aabbccdd-eeff-0011-2233-445566778899",
            token="testtoken1234",
            twin_uuids=["robot", "table"],
            alert_twin_uuids=[],
        )
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True)

        assert manager._run_container() is True
        assert _RecordingAlertCtx.instances == []

    def test_card_stays_open_when_the_container_reaches_running(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``running`` is not the end of startup: the worker inside still has
        to load and warm up its models. Resolving here — which this code used
        to do — told the pending-reason copy and the stuck-run sweeper that the
        device had finished starting."""
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.phases == ["starting_container", "container_running"]
            assert ctx.resolved is False, "the worker has not reported ready yet"
            assert ctx.failed is None

    def test_card_resolves_on_the_tick_after_the_worker_reports_ready(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole sequence: container up, worker reports, reconcile closes."""
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True
        self._publish_status(worker_manager, "ready")
        worker_manager.poll_startup_readiness()

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.resolved is True
            assert ctx.failed is None
            assert ctx.phases[-1] == "ready"

    def test_card_reports_the_worker_loading_its_models(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The phases that used to be invisible reach the card."""
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True
        self._publish_status(worker_manager, "warming_up", models_ready=1, models_total=3)
        worker_manager.poll_startup_readiness()

        for ctx in _RecordingAlertCtx.instances:
            warm = next(p for p in ctx.patches if p.get("phase") == "warming_up")
            assert warm["models_ready"] == 1
            assert warm["models_total"] == 3
            assert ctx.resolved is False

    def test_repeated_ticks_do_not_republish_an_unchanged_phase(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Alert updates bypass the throttle, so an unchanged phase must not
        cost one API call per reconcile tick for the whole startup."""
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True
        self._publish_status(worker_manager, "loading_models")
        for _ in range(5):
            worker_manager.poll_startup_readiness()

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.phases.count("loading_models") == 1

    def test_a_dead_container_fails_the_card_on_the_next_tick(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True
        self._publish_status(worker_manager, "loading_models")
        monkeypatch.setattr(
            wm_module, "docker_inspect", lambda *_a, **_kw: {"State": {"Status": "exited"}}
        )
        worker_manager.poll_startup_readiness()

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.resolved is True
            assert ctx.failed == "worker_startup_exited"

    def test_an_unanswered_inspect_during_the_pull_does_not_fail_the_card(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The card is opened *before* the pull, so for minutes there is no
        container to inspect. ``docker_inspect`` returns ``None`` for that and
        for a daemon that would not answer alike, and a reconcile tick landing
        in that window must not report a worker that stopped while starting up
        — the pull is still running and will succeed.
        """
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(wm_module, "docker_inspect", lambda *_a, **_kw: None)
        worker_manager._open_starting_alerts("img")

        for _ in range(10):
            worker_manager.poll_startup_readiness()

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.resolved is False
            assert ctx.failed is None

    def test_one_unanswered_inspect_does_not_fail_a_running_worker(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``docker inspect`` has its own 10 s timeout, and a device pulling a
        multi-GB image is exactly where it blows. One unanswered sample is not
        evidence that a worker mid-``loading_models`` has died."""
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True
        self._publish_status(worker_manager, "loading_models")
        worker_manager.poll_startup_readiness()

        monkeypatch.setattr(wm_module, "docker_inspect", lambda *_a, **_kw: None)
        worker_manager.poll_startup_readiness()

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.resolved is False
            assert ctx.failed is None

    def test_a_container_that_stays_unobservable_is_eventually_reported_gone(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tolerating a miss must not mean tolerating every miss: a container
        removed under a live card still has to retire it rather than leave a
        spinner up forever."""
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True
        self._publish_status(worker_manager, "loading_models")
        worker_manager.poll_startup_readiness()

        monkeypatch.setattr(wm_module, "docker_inspect", lambda *_a, **_kw: None)
        for _ in range(wm_module._READINESS_INSPECT_MISS_LIMIT):
            worker_manager.poll_startup_readiness()

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.resolved is True
            assert ctx.failed == "worker_startup_exited"

    def test_a_restart_does_not_inherit_the_previous_container_sighting(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A restart reuses the open card, so the readiness state behind it has
        to start over with the new container.

        ``_open_starting_alerts`` no-ops while a card is live, and the restart
        ``docker rm``s the container before re-pulling. If the sighting of the
        previous container survived that, the polls landing during the re-pull
        would count missed inspects against a container already removed and
        fail a startup that is proceeding normally.
        """
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True
        self._publish_status(worker_manager, "loading_models")
        worker_manager.poll_startup_readiness()  # sees the first container

        def repull_with_reconcile_ticks(*_a: object, **_kw: object) -> bool:
            # The container is gone and the image is downloading: what every
            # tick during a multi-minute re-pull observes.
            monkeypatch.setattr(wm_module, "docker_inspect", lambda *_a, **_kw: None)
            for _ in range(wm_module._READINESS_INSPECT_MISS_LIMIT + 2):
                worker_manager.poll_startup_readiness()
            self._stub_container_start(monkeypatch)
            return True

        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", repull_with_reconcile_ticks
        )

        assert worker_manager._run_container() is True

        # The restart retires the first startup's cards and opens fresh ones
        # (see ``_open_starting_alerts``); those fresh ones are what must have
        # survived the re-pull.
        assert all(ctx.failed is None for ctx in _RecordingAlertCtx.instances)
        live = [ctx for ctx in _RecordingAlertCtx.instances if not ctx.resolved]
        assert len(live) == 2, "one live card per twin covering the restart"
        for ctx in live:
            assert ctx.phases[-1] == "container_running"

    def test_a_phase_that_never_changes_still_keeps_the_card_alive(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``loading_models`` publishes once however long the weights take, so
        without a keepalive the alert goes untouched until the backend's own
        staleness window drops it and fails the run under a working device."""
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )
        clock = FakeMonotonicClock()
        monkeypatch.setattr(wm_module, "now_monotonic", clock)

        assert worker_manager._run_container() is True
        self._publish_status(worker_manager, "loading_models")
        worker_manager.poll_startup_readiness()
        # Two keepalive intervals of the same phase, sampled every 15 s.
        for _ in range(2 * wm_module._READINESS_KEEPALIVE_SECONDS // 15):
            clock.advance(15)
            worker_manager.poll_startup_readiness()

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.phases.count("loading_models") == 3, (
                "the first publish plus one per keepalive interval, not one per reconcile tick"
            )
            assert ctx.resolved is False

    def test_polling_without_an_open_card_is_a_no_op(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The steady-state cost on every reconcile tick of an idle edge."""
        self._stub_container_start(monkeypatch)
        inspected: list[str] = []
        monkeypatch.setattr(
            wm_module,
            "docker_inspect",
            lambda name, *_a, **_kw: inspected.append(name),
        )

        worker_manager.poll_startup_readiness()

        assert inspected == []
        assert _RecordingAlertCtx.instances == []

    def test_a_second_start_is_never_silent(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A restart while the first card is still open must still show one.

        ``_open_starting_alerts`` no-ops while any card is open, so the second
        startup relies on the first card still being live and being re-patched
        rather than on a fresh one.
        """
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True
        assert worker_manager._run_container() is True

        live = [ctx for ctx in _RecordingAlertCtx.instances if not ctx.resolved]
        assert len(live) == 2, "one live card per twin covering the second startup"
        for ctx in live:
            assert ctx.phases[-1] == "container_running"

    def test_notifies_even_when_the_image_is_already_local(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A skipped pull is exactly the case that used to be silent."""
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True
        assert _RecordingAlertCtx.instances
        assert all("starting_container" in ctx.phases for ctx in _RecordingAlertCtx.instances)

    def test_startup_failure_retires_the_card(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A spinner left up forever is worse than a failure message."""
        self._stub_container_start(monkeypatch, status="exited")
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )
        monkeypatch.setattr(wm_module, "docker_inspect", lambda *_a, **_kw: None)
        monkeypatch.setattr(wm_module.time, "sleep", lambda *_a: None)

        worker_manager._run_container()

        assert _RecordingAlertCtx.instances
        for ctx in _RecordingAlertCtx.instances:
            assert ctx.resolved is True
            assert ctx.failed == "container_start_failed"

    def test_second_startup_gets_its_own_card_after_a_leak(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A card left behind by an earlier startup must not suppress the next.

        ``_open_starting_alerts`` used to return early whenever the list was
        non-empty, so one startup that left through a path which never closed
        its card (an exception escaping the probe, say) silenced the progress
        bar for every later startup in the process — it worked once after an
        edge-core restart and never again.
        """
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        # Simulate the leak: a previous startup left its contexts held.
        leaked = _RecordingAlertCtx(
            twin_uuid="leaked-twin",
            image="cyberwaveos/edge-ml-worker:dev",
            container_name=worker_manager._container_name,
        )
        leaked.create()
        worker_manager._card.ctxs = [leaked]
        _RecordingAlertCtx.instances = []

        assert worker_manager._run_container() is True

        # The stale card is retired rather than reused …
        assert leaked.resolved is True
        # … and this startup reports on cards of its own.
        assert len(_RecordingAlertCtx.instances) == 2
        for ctx in _RecordingAlertCtx.instances:
            assert ctx.phases == ["starting_container", "container_running"]
            # Still open: `container_running` is no longer the end of startup,
            # and `poll_startup_readiness` closes the card on a later tick.
            assert ctx.resolved is False

    def test_no_twins_means_no_card(
        self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = WorkerManager(
            config_dir=tmp_config,
            environment_uuid="aabbccdd-eeff-0011-2233-445566778899",
            token="testtoken1234",
            twin_uuids=[],
        )
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True)

        assert manager._run_container() is True
        assert _RecordingAlertCtx.instances == []

    def test_accelerator_fallback_keeps_the_card_alive(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed ``-gpu`` pull must not retire the card the fallback needs.

        The pull helper adopts the card ``_run_container`` opened, so it must
        not mark it failed on its own account — ``_run_container`` may still
        fall back to the CPU image, and the user has to see that download.
        """
        import cyberwave_edge_core.driver_logs as dl_module

        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "_docker_image_present", lambda *_a, **_kw: False)
        monkeypatch.setattr(wm_module, "_image_tag_is_mutable", lambda *_a, **_kw: True)
        monkeypatch.setattr(wm_module, "_begin_worker_image_pull", lambda *_a, **_kw: True)
        monkeypatch.setattr(wm_module, "_finish_worker_image_pull", lambda *_a, **_kw: None)
        monkeypatch.setenv("CYBERWAVE_WORKER_IMAGE_PULL_RETRIES", "1")
        worker_manager._image = "cyberwaveos/edge-ml-worker:dev"

        def fake_multi(image: str, *, contexts, token, timeout):  # noqa: ANN001
            if image.endswith("-gpu"):
                raise subprocess.CalledProcessError(1, "docker pull", stderr="manifest unknown")
            for ctx in contexts:
                if ctx.driver_alert_ctx:
                    ctx.driver_alert_ctx.update_metadata(
                        {"phase": "downloading", "progress_percent": 50.0}, force=True
                    )

        monkeypatch.setattr(dl_module, "_pull_docker_image_with_progress_multi", fake_multi)

        assert worker_manager._run_container() is True

        # One card per twin, still the originals — no second set was raised.
        assert len(_RecordingAlertCtx.instances) == 2
        for ctx in _RecordingAlertCtx.instances:
            assert ctx.failed is None, "a successful startup must not report failure"
            # Still open: readiness polling closes it on a later tick.
            assert ctx.resolved is False
            # The whole sequence lands on the one card, fallback download included.
            assert ctx.phases == [
                "pull_failed",
                "pull_started",
                "downloading",
                "starting_container",
                "container_running",
            ]

    def test_a_card_opened_by_one_manager_is_closed_by_the_reconcile_tick(
        self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The card belongs to the container, not to whichever manager opened it.

        Nothing in edge-core builds one ``WorkerManager`` and keeps it: the
        boot path builds its own (``_start_worker_after_drivers``), the runtime
        loop's watcher builds another — and the watcher's is the only one
        ``poll_startup_readiness`` is ever called on. While the card lived on
        the instance, the ordinary boot path therefore opened one that nothing
        polled: it stayed up until the next edge-core boot, and the run's
        ``pending_reason`` read "Starting the worker" for a worker already
        serving runs.
        """
        self._stub_container_start(monkeypatch)

        def build() -> WorkerManager:
            return WorkerManager(
                config_dir=tmp_config,
                environment_uuid="aabbccdd-eeff-0011-2233-445566778899",
                token="testtoken1234",
                twin_uuids=["twin-uuid-1", "twin-uuid-2"],
            )

        boot_manager = build()
        monkeypatch.setattr(
            boot_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )
        assert boot_manager._run_container() is True
        assert _RecordingAlertCtx.instances
        assert all(not ctx.resolved for ctx in _RecordingAlertCtx.instances)

        watcher_manager = build()
        assert watcher_manager is not boot_manager
        self._publish_status(watcher_manager, "ready")
        watcher_manager.poll_startup_readiness()

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.resolved is True, "the tick that polls is not the one that opened"
            assert ctx.failed is None
            assert ctx.phases[-1] == "ready"

    def test_a_forward_clock_step_does_not_fail_a_healthy_worker(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Elapsed is ``now - StartedAt``, so a clock step lands on it whole.

        A device without an RTC boots at the epoch and gets stepped forward
        the moment NTP lands — and on the warm-image path there is no pull to
        wait for, so the container starts seconds into boot and the step falls
        squarely inside worker startup. A jump past the ready timeout would
        report a worker that is loading normally as failed.
        """
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )
        clock = FakeMonotonicClock()
        monkeypatch.setattr(wm_module, "now_monotonic", clock)

        started_ns = time.time_ns() - 30 * 1_000_000_000
        monkeypatch.setattr(wm_module, "docker_inspect", lambda *_a, **_kw: _started_at(started_ns))

        assert worker_manager._run_container() is True
        self._publish_status(worker_manager, "loading_models")
        worker_manager.poll_startup_readiness()

        # NTP steps the clock an hour forward. Same instant, same container;
        # ``StartedAt`` now reads an hour and a half back.
        stepped_ns = started_ns - 3600 * 1_000_000_000
        monkeypatch.setattr(wm_module, "docker_inspect", lambda *_a, **_kw: _started_at(stepped_ns))
        clock.advance(15)
        worker_manager.poll_startup_readiness()

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.failed is None, "an NTP step is not a worker that took too long"
            assert ctx.resolved is False

    def test_a_restarting_container_does_not_hold_the_card_open_forever(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash loop leaves no trace in the two facts the poll reads.

        The worker runs under ``--restart unless-stopped``: between crash and
        restart Docker parks the container in ``restarting``, which is not a
        dead state, and every restart rewrites ``StartedAt``, which resets
        elapsed. So neither the dead-container rule nor the readiness timeout
        can fire, and a worker crash-looping on a missing checkpoint holds the
        card open until the next edge-core boot.
        """
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True
        self._publish_status(worker_manager, "loading_models")
        monkeypatch.setattr(
            wm_module,
            "docker_inspect",
            lambda *_a, **_kw: {"State": {"Status": "restarting"}, "RestartCount": 3},
        )
        worker_manager.poll_startup_readiness()

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.resolved is True
            assert ctx.failed == "worker_startup_exited"

    def test_stop_closes_a_card_left_by_a_container_that_already_exited(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``stop()`` short-circuits on a container that is already down —
        which is exactly the state a container that died mid-startup is in,
        with its card still open. Returning before the close left the spinner
        up for a worker nobody is bringing back."""
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )
        assert worker_manager._run_container() is True

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda *_a, **_kw: "exited")

        assert worker_manager.stop(reason="workers-dir-empty") is True

        for ctx in _RecordingAlertCtx.instances:
            assert ctx.resolved is True
            assert ctx.failed is None, "an operator stopping the worker is not a failure"

    def test_a_worker_that_has_not_reported_yet_is_not_ready(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``clear_status()`` runs just before ``docker run``, so a new worker
        has no record at all for its first seconds. Absence of a record means
        "old image, assume ready" only once the card has closed — while one is
        open it means the opposite."""
        self._stub_container_start(monkeypatch)
        monkeypatch.setattr(
            worker_manager, "_pull_worker_image_with_progress", lambda *_a, **_kw: True
        )

        assert worker_manager._run_container() is True
        assert worker_manager.is_worker_ready() is False

        # ``evaluate_readiness`` closes the card on the grace window when no
        # record ever arrives — the old-image case, which still needs the
        # optimistic reading.
        worker_manager._close_starting_alerts()
        assert worker_manager.is_worker_ready() is True


class TestStatusDirResolution:
    def test_a_transient_failure_is_not_cached_for_the_process(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``/run`` can be unavailable very early in boot. Remembering that
        answer disabled startup-phase reporting for the whole process — every
        worker started afterwards reported nothing."""
        monkeypatch.setattr(wm_module, "default_status_dir", lambda _cfg: tmp_path / "run")
        broken = {"yes": True}

        def flaky(path: Path) -> None:
            if broken["yes"]:
                raise OSError("/run is not mounted yet")

        monkeypatch.setattr(wm_module, "_ensure_dir_writable_by_container_user", flaky)

        assert worker_manager._status_dir() is None

        broken["yes"] = False
        assert worker_manager._status_dir() == tmp_path / "run"

    def test_a_resolved_directory_is_cached(
        self, worker_manager: WorkerManager, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The success path still resolves once: mkdir and chown per poll
        would be a syscall pair on every reconcile tick."""
        monkeypatch.setattr(wm_module, "default_status_dir", lambda _cfg: tmp_path / "run")
        calls: list[Path] = []
        monkeypatch.setattr(
            wm_module, "_ensure_dir_writable_by_container_user", lambda p: calls.append(p)
        )

        assert worker_manager._status_dir() == tmp_path / "run"
        assert worker_manager._status_dir() == tmp_path / "run"
        assert len(calls) == 1


def test_keepalive_stays_under_the_backend_staleness_window() -> None:
    """Edge-core's keepalive must stay well under the backend's staleness
    window, ``WORKER_STARTING_ALERT_STALE_SECONDS`` (900 s) in
    ``src/app/services/workflow_execution_pending.py``.

    If it ever exceeds it the failures correlate: a device that is working
    goes silent, ``sweep_stuck_edge_manual_executions`` fails the row, *and*
    ``retry_run_workflow_worker_command`` stops re-publishing, so the run
    cannot recover either. Separate deployables, so neither end can import the
    other's constant — this test and its backend twin are the only link.
    """
    assert wm_module._READINESS_KEEPALIVE_SECONDS <= 300
