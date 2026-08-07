"""Tests for multi-container driver orchestration.

Covers:
  1. _get_driver_services — parses services array from metadata
  2. _get_driver_services — returns None for single-image metadata
  3. Container naming with service suffix
  4. shared_env + per-service env layering
  5. Custom command passed to docker create
  6. Parallel pull phase collects all service images
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any

import cyberwave_edge_core.driver_selection as driver_selection
import cyberwave_edge_core.startup as startup
from cyberwave_edge_core.docker_args import (
    DEFAULT_DOCKER_LOG_MAX_FILE,
    DEFAULT_DOCKER_LOG_MAX_SIZE,
)
from tests.driver_subprocess_fakes import fake_docker_start_popen


class _FakeAlertContext:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def create(self) -> None:
        pass

    def update_metadata(self, *args: Any, **kwargs: Any) -> None:
        pass

    def resolve(self) -> None:
        pass

    def fail_without_resolve(self, *args: Any, **kwargs: Any) -> None:
        pass

    def mark_failed_and_resolve(self, *args: Any, **kwargs: Any) -> None:
        pass


def _patch_driver_container_launch(monkeypatch: Any, captured_cmds: list[list[str]]) -> None:
    """Patch subprocess and probe helpers for create+start driver launch."""

    def _fake_subprocess_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(startup.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(startup.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(startup.subprocess, "Popen", fake_docker_start_popen(captured_cmds))
    monkeypatch.setattr(
        "cyberwave_edge_core.docker_helpers.docker_inspect",
        lambda _: {"State": {"Status": "running"}},
    )
    # Pre-cleanup removal is covered by test_docker_launch; no-op it here so it
    # doesn't trip over the always-"running" docker_inspect mock above.
    monkeypatch.setattr(
        "cyberwave_edge_core.docker_launch.remove_existing_container",
        lambda *a, **kw: True,
    )
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(startup, "_resolve_driver_image_tag", lambda img: img)
    monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda _: True)
    monkeypatch.setattr(startup, "_build_driver_network_args", lambda _: ["--network", "host"])

    def _fast_probe_env(name: str, default: object = None) -> object:
        if name in {
            "CYBERWAVE_DRIVER_STARTUP_PROBE_SECONDS",
            "CYBERWAVE_DRIVER_CREATE_TIMEOUT_SECONDS",
        }:
            return "1"
        return default

    monkeypatch.setattr(startup, "get_runtime_env_var", _fast_probe_env)
    monkeypatch.setattr(startup, "load_credentials_envs", lambda: {})
    monkeypatch.setattr(startup, "build_zenoh_env_vars", lambda _: {})
    monkeypatch.setattr(startup, "_get_zenoh_config", lambda: None)
    monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: False)
    monkeypatch.setattr(startup, "_run_macos_device_bridge_commands", lambda **kw: (True, {}))
    monkeypatch.setattr(startup, "_normalize_macos_bridge_candidates", lambda _: {})
    monkeypatch.setattr(startup, "_extract_docker_env_map", lambda _: {})
    monkeypatch.setattr(startup, "_stream_container_logs", lambda *a, **kw: None)
    monkeypatch.setattr("cyberwave_edge_core.docker_launch.time.sleep", lambda _: None)
    monkeypatch.setattr(startup, "DriverStartingAlertContext", _FakeAlertContext)
    monkeypatch.setattr(startup, "CONFIG_DIR", startup.Path("/tmp/.cyberwave-test"))


# ===========================================================================
# Helpers
# ===========================================================================

MULTI_SERVICE_METADATA: dict[str, Any] = {
    "linux-aarch64-jetson": {
        "services": [
            {
                "image": "cyberwaveos/go2-ros2-driver:jetson-humble",
                "name": "driver",
                "command": ["ros2", "launch", "go2_driver", "robot.launch.py"],
            },
            {
                "image": "cyberwaveos/go2-ros2-driver:jetson-humble",
                "name": "bridges",
                "env": {"BRIDGE_MODE": "full"},
            },
            {
                "image": "cyberwaveos/ros2-nav2:jetson-humble",
                "name": "nav2",
            },
            {
                "image": "cyberwaveos/ros2-slam:jetson-humble",
                "name": "slam",
            },
            {
                "image": "cyberwaveos/ros2-elevation-mapping:jetson-humble",
                "name": "elevation",
                "prefer_gpu": True,
                "gpu": "1",
            },
        ],
        "shared_env": {
            "ROS_DOMAIN_ID": "0",
            "CONFIG_PROFILE": "jetson",
        },
        "shared_params": ["--network", "host", "-v", "/data:/data"],
    },
    "default": {
        "docker_image": "cyberwaveos/go2-ros2-driver",
        "prefer_gpu": True,
    },
}

SINGLE_IMAGE_METADATA: dict[str, Any] = {
    "default": {
        "docker_image": "cyberwaveos/so101-driver",
        "params": ["--network", "host"],
    },
}


# ===========================================================================
# 1. _get_driver_services parses services array
# ===========================================================================


class TestGetDriverServices:
    def test_services_metadata_parsed(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "aarch64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: True)

        result = driver_selection._get_driver_services(MULTI_SERVICE_METADATA)
        assert result is not None

        services, shared_env, shared_params = result
        assert len(services) == 5
        assert services[0].name == "driver"
        assert services[0].image == "cyberwaveos/go2-ros2-driver:jetson-humble"
        assert services[0].command == ["ros2", "launch", "go2_driver", "robot.launch.py"]
        assert services[1].name == "bridges"
        assert services[1].env == {"BRIDGE_MODE": "full"}
        assert services[4].name == "elevation"
        assert services[4].prefer_gpu is True
        assert services[4].gpu_spec == "1"

        assert shared_env == {"ROS_DOMAIN_ID": "0", "CONFIG_PROFILE": "jetson"}
        assert shared_params == ["--network", "host", "-v", "/data:/data"]

    def test_single_image_returns_none(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: False)

        result = driver_selection._get_driver_services(SINGLE_IMAGE_METADATA)
        assert result is None

    def test_falls_back_to_default_when_platform_unmatched(self, monkeypatch: Any) -> None:
        """When no platform key matches, falls back to 'default'.
        If 'default' has no services, returns None."""
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: False)

        result = driver_selection._get_driver_services(MULTI_SERVICE_METADATA)
        assert result is None

    def test_validates_missing_image(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: False)

        bad_metadata = {
            "default": {
                "services": [{"name": "broken"}],
            },
        }
        try:
            driver_selection._get_driver_services(bad_metadata)
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "image" in str(exc).lower()

    def test_validates_missing_name(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: False)

        bad_metadata = {
            "default": {
                "services": [{"image": "img:latest"}],
            },
        }
        try:
            driver_selection._get_driver_services(bad_metadata)
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "name" in str(exc).lower()

    def test_per_service_params_parsed(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: False)

        metadata = {
            "default": {
                "services": [
                    {
                        "image": "img:latest",
                        "name": "svc",
                        "params": ["-v", "/tmp:/tmp"],
                    },
                ],
            },
        }
        result = driver_selection._get_driver_services(metadata)
        assert result is not None
        services, _, _ = result
        assert services[0].params == ["-v", "/tmp:/tmp"]


# ===========================================================================
# 2. Multi-container naming
# ===========================================================================


class TestDriverContainerLogCap:
    """Driver containers must carry a log-size cap.

    Applied per container rather than via the host's daemon.json, which only
    the Pi image configures.
    """

    def test_log_cap_applied_by_default(self, monkeypatch: Any) -> None:
        captured_cmds: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured_cmds)
        monkeypatch.delenv("CYBERWAVE_DOCKER_LOG_MAX_SIZE", raising=False)
        monkeypatch.delenv("CYBERWAVE_DOCKER_LOG_MAX_FILE", raising=False)

        startup._run_docker_image(
            "cyberwaveos/so101-driver:humble",
            [],
            twin_uuid="aabbccdd-1234-5678-9012-abcdef012345",
            token="test-token",
            skip_pull=True,
        )

        cmd = next(c for c in captured_cmds if c[:2] == ["docker", "create"])
        assert "--log-driver" in cmd
        assert f"max-size={DEFAULT_DOCKER_LOG_MAX_SIZE}" in cmd
        assert f"max-file={DEFAULT_DOCKER_LOG_MAX_FILE}" in cmd

    def test_default_does_not_loosen_the_pi_image_cap(self) -> None:
        """Per-container flags override daemon.json, so ours must not be larger.

        The Pi image ships 10m x 3 daemon-wide
        (devops/raspberry_pi_imager/files/chroot-setup.sh); a bigger default
        here would raise the ceiling on the device with the least headroom.
        """
        assert DEFAULT_DOCKER_LOG_MAX_SIZE == "10m"
        assert DEFAULT_DOCKER_LOG_MAX_FILE == "3"

    def test_driver_params_override_wins(self, monkeypatch: Any) -> None:
        """An explicit --log-driver in docker_run_params wins.

        Suppressed entirely rather than appended: ``--log-opt max-size`` is
        meaningless against a non-json-file driver.
        """
        captured_cmds: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured_cmds)

        startup._run_docker_image(
            "cyberwaveos/so101-driver:humble",
            ["--log-driver", "journald"],
            twin_uuid="aabbccdd-1234-5678-9012-abcdef012345",
            token="test-token",
            skip_pull=True,
        )

        cmd = next(c for c in captured_cmds if c[:2] == ["docker", "create"])
        assert cmd.count("--log-driver") == 1
        assert "journald" in cmd
        assert not any(a.startswith("max-size=") for a in cmd)

    def test_partial_driver_override_keeps_the_other_bound(self, monkeypatch: Any) -> None:
        """A pinned max-size must not take the max-file cap down with it."""
        captured_cmds: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured_cmds)

        startup._run_docker_image(
            "cyberwaveos/so101-driver:humble",
            ["--log-opt", "max-size=1g"],
            twin_uuid="aabbccdd-1234-5678-9012-abcdef012345",
            token="test-token",
            skip_pull=True,
        )

        cmd = next(c for c in captured_cmds if c[:2] == ["docker", "create"])
        assert "max-size=1g" in cmd
        assert "max-file=3" in cmd
        # One value per key — Docker would otherwise silently take the last.
        assert len([a for a in cmd if a.startswith("max-size=")]) == 1


class TestMultiContainerNaming:
    def test_container_name_includes_service_suffix(self, monkeypatch: Any) -> None:
        """When service_name is set, the container name includes
        ``-{service_name}`` as a suffix."""
        captured_cmds: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured_cmds)

        twin_uuid = "aabbccdd-1234-5678-9012-abcdef012345"
        result = startup._run_docker_image(
            "cyberwaveos/go2-ros2-driver:jetson-humble",
            ["--network", "host"],
            twin_uuid=twin_uuid,
            token="test-token",
            skip_pull=True,
            service_name="nav2",
        )

        docker_create_cmd = [c for c in captured_cmds if c[:2] == ["docker", "create"]]
        assert len(docker_create_cmd) >= 1
        cmd = docker_create_cmd[0]
        name_idx = cmd.index("--name")
        container_name = cmd[name_idx + 1]
        assert container_name == f"cyberwave-driver-{twin_uuid[:8]}-nav2"

    def test_single_container_name_unchanged(self, monkeypatch: Any) -> None:
        """When service_name is None, the container name uses the
        original format without any suffix."""
        captured_cmds: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured_cmds)

        twin_uuid = "aabbccdd-1234-5678-9012-abcdef012345"
        startup._run_docker_image(
            "cyberwaveos/so101-driver:humble",
            [],
            twin_uuid=twin_uuid,
            token="test-token",
            skip_pull=True,
        )

        docker_create_cmd = [c for c in captured_cmds if c[:2] == ["docker", "create"]]
        assert len(docker_create_cmd) >= 1
        cmd = docker_create_cmd[0]
        name_idx = cmd.index("--name")
        container_name = cmd[name_idx + 1]
        assert container_name == f"cyberwave-driver-{twin_uuid[:8]}"


# ===========================================================================
# 3. shared_env + per-service env merge
# ===========================================================================


class TestEnvMerge:
    def test_shared_env_merged_with_service_env(self, monkeypatch: Any) -> None:
        """service_env merges into the container env dict."""
        captured_cmds: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured_cmds)

        twin_uuid = "aabbccdd-1234-5678-9012-abcdef012345"
        startup._run_docker_image(
            "cyberwaveos/ros2-nav2:humble",
            [],
            twin_uuid=twin_uuid,
            token="test-token",
            skip_pull=True,
            service_name="nav2",
            service_env={"ROS_DOMAIN_ID": "0", "CONFIG_PROFILE": "jetson"},
        )

        docker_create_cmd = [c for c in captured_cmds if c[:2] == ["docker", "create"]]
        assert len(docker_create_cmd) >= 1
        cmd_str = " ".join(docker_create_cmd[0])
        assert "ROS_DOMAIN_ID=0" in cmd_str
        assert "CONFIG_PROFILE=jetson" in cmd_str


# ===========================================================================
# 4. Custom command appended
# ===========================================================================


class TestCommandAppended:
    def test_command_appended_after_image(self, monkeypatch: Any) -> None:
        captured_cmds: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured_cmds)

        image = "cyberwaveos/go2-ros2-driver:jetson-humble"
        custom_cmd = ["ros2", "launch", "go2_driver", "robot.launch.py"]
        twin_uuid = "aabbccdd-1234-5678-9012-abcdef012345"

        startup._run_docker_image(
            image,
            [],
            twin_uuid=twin_uuid,
            token="test-token",
            skip_pull=True,
            service_name="driver",
            command=custom_cmd,
        )

        docker_create_cmd = [c for c in captured_cmds if c[:2] == ["docker", "create"]]
        assert len(docker_create_cmd) >= 1
        cmd = docker_create_cmd[0]
        image_idx = cmd.index(image)
        trailing = cmd[image_idx + 1:]
        assert trailing == custom_cmd

    def test_no_command_when_none(self, monkeypatch: Any) -> None:
        captured_cmds: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured_cmds)

        image = "cyberwaveos/so101-driver:humble"
        twin_uuid = "aabbccdd-1234-5678-9012-abcdef012345"

        startup._run_docker_image(
            image,
            [],
            twin_uuid=twin_uuid,
            token="test-token",
            skip_pull=True,
        )

        docker_create_cmd = [c for c in captured_cmds if c[:2] == ["docker", "create"]]
        assert len(docker_create_cmd) >= 1
        cmd = docker_create_cmd[0]
        assert cmd[-1] == image


# ===========================================================================
# 5. Pull phase collects all service images
# ===========================================================================


class TestPullCollectsAllImages:
    def test_multi_service_produces_multiple_specs(self, monkeypatch: Any) -> None:
        """When the drivers metadata has a services array, Pass 1 should
        produce one _DriverSpec per service, causing the pull phase to
        see all unique images."""
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "aarch64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: True)

        result = driver_selection._get_driver_services(MULTI_SERVICE_METADATA)
        assert result is not None

        services, _, _ = result
        images = [svc.image for svc in services]
        unique_images = list(dict.fromkeys(images))

        assert len(unique_images) == 4
        assert "cyberwaveos/go2-ros2-driver:jetson-humble" in unique_images
        assert "cyberwaveos/ros2-nav2:jetson-humble" in unique_images
        assert "cyberwaveos/ros2-slam:jetson-humble" in unique_images
        assert "cyberwaveos/ros2-elevation-mapping:jetson-humble" in unique_images


# ===========================================================================
# 6. Shared-image alert differentiation
# ===========================================================================


class TestDriverStartingAlertDifferentiation:
    def test_shared_image_services_create_distinct_alert_contexts(self, monkeypatch: Any) -> None:
        """Services that share an image still emit distinct
        ``driver_starting`` alerts, differentiated by ``service_name``."""

        class _FakeTwin:
            def __init__(self) -> None:
                self.uuid = "11111111-2222-3333-4444-555555555555"
                self.name = "GO2"
                self.asset_uuid = "asset-go2"
                self.asset_id = "asset-go2"
                self.metadata = {
                    "edge_fingerprint": "edge-fp",
                    "drivers": {
                        "default": {
                            "services": [
                                {
                                    "name": "driver",
                                    "image": "cyberwaveos/go2-ros2-driver:jetson-humble-dev",
                                },
                                {
                                    "name": "bridges",
                                    "image": "cyberwaveos/go2-ros2-driver:jetson-humble-dev",
                                },
                                {
                                    "name": "nav2",
                                    "image": "cyberwaveos/ros2-nav2:jetson-humble-dev",
                                },
                            ]
                        }
                    },
                }

        class _FakeTwinsAPI:
            def __init__(self, twin: _FakeTwin) -> None:
                self._twin = twin

            def list(self, environment_id: str) -> list[_FakeTwin]:
                return [self._twin]

            def get_raw(self, twin_uuid: str) -> dict[str, Any]:
                return {}

        class _FakeAssetsAPI:
            def get(self, asset_uuid: str) -> SimpleNamespace:
                return SimpleNamespace(metadata={})

        fake_twin = _FakeTwin()
        fake_client = SimpleNamespace(
            twins=_FakeTwinsAPI(fake_twin),
            assets=_FakeAssetsAPI(),
        )

        created_alert_keys: list[tuple[str, str, str | None]] = []

        class _FakeAlertContext:
            def __init__(
                self,
                *,
                twin_uuid: str,
                image: str,
                service_name: str | None = None,
            ) -> None:
                created_alert_keys.append((twin_uuid, image, service_name))

            def create(self) -> None:
                pass

            def update_metadata(self, patch: dict[str, Any], *, force: bool = False) -> None:
                pass

            def resolve(self) -> None:
                pass

            def mark_failed_and_resolve(
                self,
                description: str,
                *,
                phase: str = "pull_failed",
            ) -> None:
                pass

        run_calls: list[dict[str, Any]] = []

        def _fake_run_docker_image(
            image: str,
            params: list[str],
            *,
            twin_uuid: str,
            token: str,
            child_camera_twin_uuids: list[str] | None = None,
            macos_bridge_device_candidates: list[str] | None = None,
            skip_pull: bool = False,
            prefer_gpu: bool = False,
            gpu_spec: str = "all",
            service_name: str | None = None,
            command: list[str] | None = None,
            service_env: dict[str, str] | None = None,
            driver_alert_ctx: Any = None,
        ) -> bool:
            run_calls.append(
                {
                    "service_name": service_name,
                    "driver_image": image,
                    "alert_ctx_id": id(driver_alert_ctx),
                }
            )
            return True

        monkeypatch.setattr(startup, "Cyberwave", lambda base_url, api_key: fake_client)
        monkeypatch.setattr(startup, "DriverStartingAlertContext", _FakeAlertContext)
        monkeypatch.setattr(startup, "_maybe_rewrite_jetson_tag", lambda image, _: image)
        monkeypatch.setattr(startup, "_check_and_alert_sensors_devices", lambda *a, **kw: None)
        monkeypatch.setattr(startup, "write_or_update_twin_json_file", lambda *a, **kw: True)
        monkeypatch.setattr(startup, "_clear_stale_driver_starting_alerts", lambda *a, **kw: 0)
        monkeypatch.setattr(
            startup,
            "_pull_driver_images_parallel",
            lambda images, **kw: {img: True for img in set(images)},
        )
        monkeypatch.setattr(startup, "_run_docker_image", _fake_run_docker_image)

        results = startup.fetch_and_run_twin_drivers("test-token", "env-uuid", "edge-fp")

        # One alert context per service launch (even when image is shared).
        assert len(run_calls) == 3
        assert len(created_alert_keys) == 3
        assert len(set(created_alert_keys)) == 3

        by_service = {call["service_name"]: call for call in run_calls}
        assert by_service["driver"]["alert_ctx_id"] != by_service["bridges"]["alert_ctx_id"]
        assert by_service["driver"]["driver_image"] == by_service["bridges"]["driver_image"]
        assert by_service["nav2"]["alert_ctx_id"] != by_service["driver"]["alert_ctx_id"]

        assert (
            fake_twin.uuid,
            "cyberwaveos/go2-ros2-driver:jetson-humble-dev",
            "driver",
        ) in created_alert_keys
        assert (
            fake_twin.uuid,
            "cyberwaveos/go2-ros2-driver:jetson-humble-dev",
            "bridges",
        ) in created_alert_keys

        assert len(results) == 3
        assert all(result["success"] for result in results)
