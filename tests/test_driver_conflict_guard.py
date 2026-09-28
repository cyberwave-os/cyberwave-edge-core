"""Tests for the second-driver guard (``driver_conflict``).

Covers:
  1. ``_robot_driver_conflict`` against fake ``docker inspect`` payloads:
     ``ROBOT_IP`` collisions, and what is never counted (the twin's own
     containers, stopped containers, a shared ``ROS_DOMAIN_ID`` on its own)
  2. ``_run_docker_image`` refuses before any ``docker create``, with the
     ``--env-file`` values resolved, and fails closed when docker cannot list
  3. ``_read_docker_env_file`` reads the file as ``docker run --env-file`` does

No docker: the container listing and image ``ENV`` are faked throughout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import cyberwave_edge_core.docker_helpers as docker_helpers
import cyberwave_edge_core.startup as startup
from cyberwave_edge_core.driver_launcher import _read_docker_env_file, _robot_driver_conflict
from tests.test_multi_service_driver import _patch_driver_container_launch

TWIN_UUID = "aabbccdd-1234-5678-9012-abcdef012345"
IMAGE = "cyberwaveos/kinova-gen3-driver@sha256:" + "a" * 64
ROBOT_IP = "192.168.1.10"


def _container(
    name: str,
    env: dict[str, str],
    *,
    network: str = "host",
    status: str = "running",
) -> dict[str, Any]:
    return {
        "Name": f"/{name}",
        "State": {"Status": status},
        "Config": {"Env": [f"{k}={v}" for k, v in env.items()]},
        "HostConfig": {"NetworkMode": network},
    }


# The hand-started real-arm driver, as `docker inspect kinova-zenoh-hardware` shows it.
HARDWARE_ENV = {"ROBOT_IP": ROBOT_IP, "ROS_DOMAIN_ID": "42", "ROS_DISTRO": "jazzy"}
HARDWARE = _container(
    "kinova-zenoh-hardware", {**HARDWARE_ENV, "KINOVA_PASSWORD": "not-for-the-message"}
)


@pytest.fixture
def image_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The driver image's own ``ENV``; empty unless a test fills it."""
    env: dict[str, str] = {}
    monkeypatch.setattr(docker_helpers, "docker_image_env", lambda _image: dict(env))
    return env


def _conflict(running: list[dict[str, Any]], declared: dict[str, str]) -> str | None:
    return _robot_driver_conflict(
        twin_uuid=TWIN_UUID,
        image=IMAGE,
        declared_env=declared,
        running=running,
    )


# ===========================================================================
# 1. _robot_driver_conflict
# ===========================================================================


class TestRobotDriverConflict:
    def test_same_robot_ip_is_refused_naming_the_container(self, image_env: dict) -> None:
        message = _conflict([HARDWARE], {"ROBOT_IP": ROBOT_IP, "ROS_DOMAIN_ID": "7"})
        assert message == (
            "Another container (kinova-zenoh-hardware) is already driving this robot "
            f"(same ROBOT_IP {ROBOT_IP}). Stop it on the edge before running from Cyberwave."
        )
        assert "not-for-the-message" not in message

    def test_same_robot_ip_is_refused_off_the_host_network(self, image_env: dict) -> None:
        other = _container("arm-by-hand", {"ROBOT_IP": ROBOT_IP}, network="bridge")
        message = _conflict([other], {"ROBOT_IP": ROBOT_IP})
        assert message is not None and "same ROBOT_IP" in message

    def test_different_robot_ip_on_another_domain_is_let_through(self, image_env: dict) -> None:
        assert _conflict([HARDWARE], {"ROBOT_IP": "192.168.1.11", "ROS_DOMAIN_ID": "7"}) is None

    def test_another_robot_on_the_same_domain_is_let_through(self, image_env: dict) -> None:
        # Every robot declaration pins ROS_DOMAIN_ID 42: a Go2 next to the Gen3.
        image_env["ROS_DISTRO"] = "humble"
        assert _conflict([HARDWARE], {"ROBOT_IP": "192.168.123.161", "ROS_DOMAIN_ID": "42"}) is None

    def test_a_ros_camera_on_the_same_domain_is_let_through(self, image_env: dict) -> None:
        camera = _container("cw-camera-driver", {"ROS_DOMAIN_ID": "42", "ROS_DISTRO": "jazzy"})
        assert _conflict([camera], {"ROBOT_IP": ROBOT_IP, "ROS_DOMAIN_ID": "42"}) is None
        # ...and the other way round: a camera driver next to the arm.
        assert _conflict([HARDWARE], {"ROS_DOMAIN_ID": "42", "ROS_DISTRO": "jazzy"}) is None

    def test_this_twins_own_containers_are_not_counted(self, image_env: dict) -> None:
        own = docker_helpers.driver_container_name(TWIN_UUID)
        running = [
            _container(own, HARDWARE_ENV),
            _container(docker_helpers.driver_container_name(TWIN_UUID, "driver"), HARDWARE_ENV),
            _container(docker_helpers.driver_container_name(TWIN_UUID, "camera"), HARDWARE_ENV),
        ]
        assert _conflict(running, {"ROBOT_IP": ROBOT_IP, "ROS_DOMAIN_ID": "42"}) is None

    def test_another_twins_driver_is_counted(self, image_env: dict) -> None:
        other = docker_helpers.driver_container_name(
            "ffffffff-1234-5678-9012-abcdef012345", "driver"
        )
        message = _conflict([_container(other, HARDWARE_ENV)], {"ROBOT_IP": ROBOT_IP})
        assert message is not None and f"({other})" in message

    @pytest.mark.parametrize("status", ["exited", "created", "dead"])
    def test_stopped_containers_are_not_counted(self, image_env: dict, status: str) -> None:
        stopped = _container("kinova-zenoh-hardware", HARDWARE_ENV, status=status)
        assert _conflict([stopped], {"ROBOT_IP": ROBOT_IP, "ROS_DOMAIN_ID": "42"}) is None

    def test_robot_ip_from_the_image_env_counts(self, image_env: dict) -> None:
        image_env["ROBOT_IP"] = ROBOT_IP
        assert _conflict([HARDWARE], {"ROS_DOMAIN_ID": "7"}) is not None
        assert _conflict([HARDWARE], {"ROBOT_IP": "192.168.1.11", "ROS_DOMAIN_ID": "7"}) is None

    def test_declared_env_overrides_the_image_env(self, image_env: dict) -> None:
        image_env["ROBOT_IP"] = ROBOT_IP
        assert _conflict([HARDWARE], {}) is not None
        assert _conflict([HARDWARE], {"ROBOT_IP": "192.168.1.11"}) is None


# ===========================================================================
# 2. _run_docker_image
# ===========================================================================


def _run(**kwargs: Any) -> bool:
    return startup._run_docker_image(
        IMAGE,
        [],
        twin_uuid=TWIN_UUID,
        token="test-token",
        skip_pull=True,
        service_name="driver",
        **kwargs,
    )


def _created(captured: list[list[str]]) -> bool:
    return any(c[:2] == ["docker", "create"] for c in captured)


class TestLauncherGuard:
    @pytest.fixture(autouse=True)
    def _seams(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_env: dict) -> None:
        self.captured: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, self.captured)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        (tmp_path / "kinova").mkdir()
        (tmp_path / "kinova" / "robot.env").write_text(
            f"ROBOT_IP={ROBOT_IP}\nKINOVA_PASSWORD=robot-secret\n"
        )
        self.running: list[dict[str, Any]] | None = [HARDWARE]
        monkeypatch.setattr(
            docker_helpers, "docker_inspect_running_containers", lambda: self.running
        )

    def test_robot_ip_from_the_host_env_file_is_refused_before_create(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("ERROR"):
            assert _run(host_env_file="kinova/robot.env") is False
        assert "driver_conflict" in caplog.text
        assert "(kinova-zenoh-hardware)" in caplog.text
        assert f"same ROBOT_IP {ROBOT_IP}" in caplog.text
        assert "robot-secret" not in caplog.text
        assert not _created(self.captured)

    def test_service_env_robot_ip_is_refused(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("ERROR"):
            assert _run(service_env={"ROBOT_IP": ROBOT_IP}) is False
        assert "driver_conflict" in caplog.text
        assert not _created(self.captured)

    def test_a_different_robot_on_the_same_host_domain_starts(self) -> None:
        assert _run(service_env={"ROBOT_IP": "192.168.1.11", "ROS_DOMAIN_ID": "42"}) is True
        assert _created(self.captured)

    def test_refusal_resolves_the_driver_alert_as_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        failures: list[tuple[str, str]] = []

        class _Alert:
            def mark_failed_and_resolve(self, description: str, *, phase: str) -> None:
                failures.append((phase, description))

        assert _run(host_env_file="kinova/robot.env", driver_alert_ctx=_Alert()) is False
        assert [phase for phase, _ in failures] == ["driver_conflict"]
        assert "kinova-zenoh-hardware" in failures[0][1]

    def test_unlistable_containers_fail_closed(self, caplog: pytest.LogCaptureFixture) -> None:
        self.running = None
        with caplog.at_level("ERROR"):
            assert _run() is False
        assert "docker_error" in caplog.text
        assert not _created(self.captured)


# ===========================================================================
# 3. _read_docker_env_file
# ===========================================================================


def test_env_file_is_read_as_docker_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FROM_EDGE", "edge-value")
    monkeypatch.delenv("UNSET_ON_EDGE", raising=False)
    path = tmp_path / "robot.env"
    path.write_text(
        '# robot\n\n  ROBOT_IP=192.168.1.10\nQUOTED="kept"\nEMPTY=\nFROM_EDGE\nUNSET_ON_EDGE\n'
        "ROBOT_IP=192.168.1.12\n"
    )
    assert _read_docker_env_file(path) == {
        "ROBOT_IP": "192.168.1.12",
        "QUOTED": '"kept"',
        "EMPTY": "",
        "FROM_EDGE": "edge-value",
    }
