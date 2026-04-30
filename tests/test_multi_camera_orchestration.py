"""Tests for multi-camera driver orchestration (CYB-1714).

Covers:
- Driver readiness probes (_wait_for_driver_readiness)
- Worker twin-list wiring validation
- Model pre-download in _start_worker_after_drivers
- Driver health reconciliation (reconcile_driver_health_for_worker)
- Multiple camera drivers + one worker container scenario
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import cyberwave_edge_core.startup as startup
from cyberwave_edge_core.worker_manager import WorkerManager


# ---------------------------------------------------------------------------
# Shared fakes (reused from test_startup_child_camera_twins)
# ---------------------------------------------------------------------------


class FakeTwin:
    def __init__(
        self,
        *,
        uuid: str,
        name: str,
        metadata: dict,
        asset_uuid: str,
        attach_to_twin_uuid: str | None = None,
    ) -> None:
        self.uuid = uuid
        self.name = name
        self.metadata = metadata
        self.asset_uuid = asset_uuid
        self.asset_id = asset_uuid
        self.attach_to_twin_uuid = attach_to_twin_uuid

    def to_dict(self) -> dict:
        payload = {"uuid": self.uuid, "name": self.name, "metadata": self.metadata}
        if self.attach_to_twin_uuid:
            payload["attach_to_twin_uuid"] = self.attach_to_twin_uuid
        return payload


class FakeAsset:
    def __init__(self, *, metadata: dict, registry_id: str = "") -> None:
        self.metadata = metadata
        self.registry_id = registry_id


class FakeTwinsAPI:
    def __init__(self, twins: list[FakeTwin]) -> None:
        self._twins = twins
        self._by_uuid = {t.uuid: t for t in twins}

    def list(self, environment_id: str) -> list[FakeTwin]:
        return self._twins

    def get_raw(self, twin_uuid: str) -> dict:
        twin = self._by_uuid[twin_uuid]
        return {"attach_to_twin_uuid": twin.attach_to_twin_uuid}


class FakeAssetsAPI:
    def __init__(self, assets: dict[str, FakeAsset]) -> None:
        self._assets = assets

    def get(self, asset_uuid: str) -> FakeAsset:
        return self._assets[asset_uuid]


def _stub_client(twins: list[FakeTwin], assets: dict[str, FakeAsset]) -> SimpleNamespace:
    return SimpleNamespace(twins=FakeTwinsAPI(twins), assets=FakeAssetsAPI(assets))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TWIN_A = "aaaa1111-1111-1111-1111-111111111111"
TWIN_B = "bbbb2222-2222-2222-2222-222222222222"
TWIN_A_CONTAINER = f"cyberwave-driver-{TWIN_A[:8]}"
TWIN_B_CONTAINER = f"cyberwave-driver-{TWIN_B[:8]}"


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# _wait_for_driver_readiness
# ---------------------------------------------------------------------------


class TestDriverReadinessProbes:
    def test_all_drivers_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [TWIN_A_CONTAINER, TWIN_B_CONTAINER],
        )
        monkeypatch.setattr(
            startup,
            "_get_container_status_fast",
            lambda name: "running",
        )

        result = startup._wait_for_driver_readiness(
            [TWIN_A, TWIN_B], timeout_seconds=5.0
        )
        assert result[TWIN_A_CONTAINER] == "running"
        assert result[TWIN_B_CONTAINER] == "running"

    def test_one_driver_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        statuses = {TWIN_A_CONTAINER: "running", TWIN_B_CONTAINER: "exited"}
        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [TWIN_A_CONTAINER, TWIN_B_CONTAINER],
        )
        monkeypatch.setattr(
            startup,
            "_get_container_status_fast",
            lambda name: statuses.get(name, "none"),
        )

        result = startup._wait_for_driver_readiness(
            [TWIN_A, TWIN_B], timeout_seconds=5.0
        )
        assert result[TWIN_A_CONTAINER] == "running"
        assert result[TWIN_B_CONTAINER] == "exited"

    def test_restarting_driver_treated_as_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [TWIN_A_CONTAINER],
        )
        monkeypatch.setattr(
            startup, "_get_container_status_fast", lambda name: "restarting"
        )

        result = startup._wait_for_driver_readiness(
            [TWIN_A], timeout_seconds=5.0
        )
        assert result[TWIN_A_CONTAINER] == "restarting"

    def test_timeout_on_slow_driver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [TWIN_A_CONTAINER],
        )
        monkeypatch.setattr(
            startup,
            "_get_container_status_fast",
            lambda name: "created",
        )

        result = startup._wait_for_driver_readiness(
            [TWIN_A], timeout_seconds=0.1, poll_interval=0.05
        )
        assert result[TWIN_A_CONTAINER] == "timeout"

    def test_child_twin_no_container_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Twins without a dedicated container (child cameras) are not waited on."""
        child_uuid = "cccc3333-3333-3333-3333-333333333333"
        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [TWIN_A_CONTAINER],
        )
        monkeypatch.setattr(
            startup,
            "_get_container_status_fast",
            lambda name: "running",
        )

        result = startup._wait_for_driver_readiness(
            [TWIN_A, child_uuid], timeout_seconds=5.0
        )
        assert TWIN_A_CONTAINER in result
        assert f"cyberwave-driver-{child_uuid[:8]}" not in result

    def test_empty_twin_list(self) -> None:
        result = startup._wait_for_driver_readiness([])
        assert result == {}

    def test_uuid_prefix_collision_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two UUIDs sharing the same 8-char prefix should trigger a warning."""
        twin_x = "aaaa1111-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        twin_y = "aaaa1111-yyyy-yyyy-yyyy-yyyyyyyyyyyy"
        container_name = f"cyberwave-driver-{twin_x[:8]}"

        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [container_name],
        )
        monkeypatch.setattr(
            startup, "_get_container_status_fast", lambda name: "running"
        )

        import logging
        with caplog.at_level(logging.WARNING):
            startup._wait_for_driver_readiness(
                [twin_x, twin_y], timeout_seconds=5.0
            )

        assert any("prefix collision" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Worker twin-list wiring
# ---------------------------------------------------------------------------


class TestWorkerTwinWiring:
    def test_twin_uuids_env_contains_all_twins(
        self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
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

        wm = WorkerManager(
            config_dir=tmp_config,
            environment_uuid="env-uuid",
            token="token",
            twin_uuids=[TWIN_A, TWIN_B],
        )
        env = wm._build_env_vars()

        assert env["CYBERWAVE_TWIN_UUIDS"] == f"{TWIN_A},{TWIN_B}"
        assert env["CYBERWAVE_TWIN_UUID"] == TWIN_A

    def test_twin_uuids_backward_compat_single(
        self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
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

        wm = WorkerManager(
            config_dir=tmp_config,
            environment_uuid="env-uuid",
            token="token",
            twin_uuids=[TWIN_A],
        )
        env = wm._build_env_vars()

        assert env["CYBERWAVE_TWIN_UUIDS"] == TWIN_A
        assert env["CYBERWAVE_TWIN_UUID"] == TWIN_A

    def test_empty_twins_no_env_vars(
        self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
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

        wm = WorkerManager(
            config_dir=tmp_config,
            environment_uuid="env-uuid",
            token="token",
            twin_uuids=[],
        )
        env = wm._build_env_vars()

        assert "CYBERWAVE_TWIN_UUIDS" not in env
        assert "CYBERWAVE_TWIN_UUID" not in env


# ---------------------------------------------------------------------------
# _start_worker_after_drivers integration
# ---------------------------------------------------------------------------


class TestStartWorkerAfterDrivers:
    def test_passes_all_twin_uuids_to_worker(
        self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_config)
        monkeypatch.setattr(startup, "_wait_for_driver_readiness", lambda *a, **kw: {})
        monkeypatch.setattr(
            startup,
            "get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr(startup, "load_worker_resource_limits", lambda: None)

        captured = {}

        class FakeWorkerManager:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def start(self):
                return True

        monkeypatch.setattr(
            "cyberwave_edge_core.worker_manager.WorkerManager", FakeWorkerManager
        )
        monkeypatch.setattr(
            "cyberwave_edge_core.worker_manager.resolve_worker_image",
            lambda: "test-image",
        )

        startup._start_worker_after_drivers(
            token="token",
            environment_uuid="env-uuid",
            twin_uuids=[TWIN_A, TWIN_B],
        )

        assert captured["twin_uuids"] == [TWIN_A, TWIN_B]

    def test_pre_downloads_models(
        self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cyberwave_edge_core.model_manager import ModelManager

        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_config)
        monkeypatch.setattr(startup, "_wait_for_driver_readiness", lambda *a, **kw: {})
        monkeypatch.setattr(
            startup,
            "get_runtime_env_var",
            lambda name, default=None: default or "http://localhost",
        )
        monkeypatch.setattr(startup, "load_worker_resource_limits", lambda: None)

        workers_dir = tmp_config / "workers"
        workers_dir.mkdir()
        (workers_dir / "worker.py").write_text('model = cw.models.load("yolov8n")\n')

        ensure_calls: list[list[str]] = []
        real_scan = ModelManager.scan_worker_model_ids

        class FakeModelManager:
            def __init__(self, **kwargs):
                pass

            @staticmethod
            def scan_worker_model_ids(workers_dir):
                return real_scan(workers_dir)

            def ensure_models(self, model_ids):
                ensure_calls.append(list(model_ids))

        monkeypatch.setattr("cyberwave_edge_core.model_manager.ModelManager", FakeModelManager)

        class FakeWM:
            def __init__(self, **kwargs):
                pass

            def start(self):
                return True

        monkeypatch.setattr("cyberwave_edge_core.worker_manager.WorkerManager", FakeWM)
        monkeypatch.setattr(
            "cyberwave_edge_core.worker_manager.resolve_worker_image",
            lambda: "test-image",
        )

        startup._start_worker_after_drivers(
            token="token",
            environment_uuid="env-uuid",
            twin_uuids=[TWIN_A],
        )

        assert len(ensure_calls) == 1
        assert ensure_calls[0] == ["yolov8n"]


# ---------------------------------------------------------------------------
# reconcile_driver_health_for_worker
# ---------------------------------------------------------------------------


class TestDriverHealthReconciliation:
    def test_detects_driver_going_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        startup._DRIVER_HEALTH_PREVIOUS.clear()
        startup._DRIVER_HEALTH_PREVIOUS[TWIN_A_CONTAINER] = "running"
        startup._CONTAINER_TWIN_MAP[TWIN_A_CONTAINER] = TWIN_A

        monkeypatch.setattr(
            startup,
            "_list_running_driver_containers",
            lambda: [],
        )
        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [TWIN_A_CONTAINER],
        )
        monkeypatch.setattr(
            startup, "_send_alert_for_twin", MagicMock()
        )

        result = startup.reconcile_driver_health_for_worker()

        assert result[TWIN_A_CONTAINER] == "down"
        startup._send_alert_for_twin.assert_called_once()
        alert_args = startup._send_alert_for_twin.call_args
        assert alert_args[0][0] == TWIN_A
        assert "no longer running" in alert_args[0][2]

    def test_no_alert_when_stable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        startup._DRIVER_HEALTH_PREVIOUS.clear()
        startup._DRIVER_HEALTH_PREVIOUS[TWIN_A_CONTAINER] = "running"

        monkeypatch.setattr(
            startup,
            "_list_running_driver_containers",
            lambda: [TWIN_A_CONTAINER],
        )
        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [TWIN_A_CONTAINER],
        )
        alert_mock = MagicMock()
        monkeypatch.setattr(startup, "_send_alert_for_twin", alert_mock)

        result = startup.reconcile_driver_health_for_worker()

        assert result[TWIN_A_CONTAINER] == "running"
        alert_mock.assert_not_called()

    def test_first_cycle_no_alert(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On the first reconcile cycle there is no previous state so no alerts."""
        startup._DRIVER_HEALTH_PREVIOUS.clear()

        monkeypatch.setattr(
            startup,
            "_list_running_driver_containers",
            lambda: [TWIN_A_CONTAINER],
        )
        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [TWIN_A_CONTAINER],
        )
        alert_mock = MagicMock()
        monkeypatch.setattr(startup, "_send_alert_for_twin", alert_mock)

        startup.reconcile_driver_health_for_worker()

        alert_mock.assert_not_called()

    def test_removed_container_detected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        startup._DRIVER_HEALTH_PREVIOUS.clear()
        startup._DRIVER_HEALTH_PREVIOUS[TWIN_A_CONTAINER] = "running"
        startup._CONTAINER_TWIN_MAP[TWIN_A_CONTAINER] = TWIN_A

        monkeypatch.setattr(startup, "_list_running_driver_containers", lambda: [])
        monkeypatch.setattr(
            startup, "_list_driver_containers", lambda include_stopped: []
        )
        monkeypatch.setattr(startup, "_send_alert_for_twin", MagicMock())

        result = startup.reconcile_driver_health_for_worker()

        assert result[TWIN_A_CONTAINER] == "removed"
        startup._send_alert_for_twin.assert_called_once()


# ---------------------------------------------------------------------------
# Full two-camera + one worker end-to-end scenario
# ---------------------------------------------------------------------------


class TestTwoCameraOneWorkerScenario:
    """Verify that fetch_and_run_twin_drivers starts 2 driver containers.

    Note: as of CYB-1766, ``fetch_and_run_twin_drivers`` no longer starts
    the worker container. Worker start is the responsibility of
    ``run_startup_checks`` (which calls it after ``_sync_workers_for_twins``)
    and of ``reconcile_worker_lifecycle`` in the runtime loop. Those
    integrations are covered by ``test_startup_worker_sync.py`` and
    ``test_worker_lifecycle_reconcile.py``.
    """

    def test_two_independent_cameras(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fingerprint = "edge-fp"
        cam_a_asset = "asset-a-uuid"
        cam_b_asset = "asset-b-uuid"

        twin_a = FakeTwin(
            uuid=TWIN_A,
            name="Camera Front",
            asset_uuid=cam_a_asset,
            metadata={
                "edge_fingerprint": fingerprint,
                "drivers": {"default": {"docker_image": "cyberwaveos/camera-driver"}},
            },
        )
        twin_b = FakeTwin(
            uuid=TWIN_B,
            name="Camera Rear",
            asset_uuid=cam_b_asset,
            metadata={
                "edge_fingerprint": fingerprint,
                "drivers": {"default": {"docker_image": "cyberwaveos/camera-driver"}},
            },
        )
        assets = {
            cam_a_asset: FakeAsset(metadata={}),
            cam_b_asset: FakeAsset(metadata={}),
        }
        fake_client = _stub_client([twin_a, twin_b], assets)

        monkeypatch.setattr(
            startup, "Cyberwave", lambda base_url, api_key: fake_client
        )
        monkeypatch.setattr(
            startup,
            "_check_and_alert_sensors_devices",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            startup, "write_or_update_twin_json_file", lambda *a, **kw: True
        )
        monkeypatch.setattr(
            startup,
            "_pull_driver_images_parallel",
            lambda images, **kw: {img: True for img in images},
        )

        run_calls: list[dict] = []

        def _fake_run(
            image,
            params,
            *,
            twin_uuid,
            token,
            child_camera_twin_uuids=None,
            macos_bridge_device_candidates=None,
            skip_pull=False,
            prefer_gpu=False,
            gpu_spec="all",
            service_name=None,
            command=None,
            service_env=None,
        ):
            run_calls.append({"twin_uuid": twin_uuid, "image": image})
            return True

        monkeypatch.setattr(startup, "_run_docker_image", _fake_run)

        # Guard rail: ``fetch_and_run_twin_drivers`` must NOT start the
        # worker any more (CYB-1766). If somebody re-introduces that call
        # site, this test will fail.
        worker_start_calls: list[tuple] = []

        def _fake_start_worker(*args, **kwargs):
            worker_start_calls.append((args, kwargs))

        monkeypatch.setattr(
            startup, "_start_worker_after_drivers", _fake_start_worker
        )

        results = startup.fetch_and_run_twin_drivers("test-token", "env-uuid", fingerprint)

        assert len(results) == 2
        started_twins = {r["twin_uuid"] for r in results}
        assert started_twins == {TWIN_A, TWIN_B}

        assert worker_start_calls == [], (
            "fetch_and_run_twin_drivers should not start the worker; "
            "that's now run_startup_checks' job (CYB-1766)."
        )


# ---------------------------------------------------------------------------
# _get_container_status_fast
# ---------------------------------------------------------------------------


class TestGetContainerStatusFast:
    def test_returns_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = "running\n"
            return result

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert startup._get_container_status_fast("test") == "running"

    def test_returns_none_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert startup._get_container_status_fast("test") == "none"
