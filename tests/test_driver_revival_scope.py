"""Driver revival recreates only the containers that are down.

No docker: the SDK client, the pull and the launcher are faked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import cyberwave_edge_core.startup as startup
from cyberwave_edge_core.docker_helpers import driver_container_name

GO2_UUID = "11111111-2222-3333-4444-555555555555"
ARM_UUID = "99999999-8888-7777-6666-555555555555"


def _services_twin(uuid: str, name: str, services: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=uuid,
        name=name,
        asset_uuid=f"asset-{name}",
        asset_id=f"asset-{name}",
        metadata={
            "edge_fingerprint": "edge-fp",
            "drivers": {
                "default": {
                    "services": [
                        {"name": svc, "image": f"cyberwaveos/{name}-{svc}:humble"}
                        for svc in services
                    ]
                }
            },
        },
    )


class _FakeAlertContext:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def create(self) -> None:
        pass

    def update_metadata(self, *args: Any, **kwargs: Any) -> None:
        pass

    def resolve(self) -> None:
        pass

    def mark_failed_and_resolve(self, *args: Any, **kwargs: Any) -> None:
        pass


@pytest.fixture
def edge(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Two linked twins -- a GO2 with two services and a one-service arm."""
    twins = [
        _services_twin(GO2_UUID, "go2", ["driver", "bridges"]),
        _services_twin(ARM_UUID, "arm", ["driver"]),
    ]
    state = SimpleNamespace(launched=[], pulled=[])
    fake_client = SimpleNamespace(
        twins=SimpleNamespace(list=lambda environment_id: twins, get_raw=lambda uuid: {}),
        assets=SimpleNamespace(get=lambda asset_uuid: SimpleNamespace(metadata={})),
    )

    def _fake_pull(images: list[str], **kwargs: Any) -> dict[str, bool]:
        state.pulled.extend(images)
        return {image: True for image in images}

    def _fake_run_docker_image(image: str, params: list[str], **kwargs: Any) -> bool:
        state.launched.append(driver_container_name(kwargs["twin_uuid"], kwargs["service_name"]))
        return True

    monkeypatch.setattr(startup, "Cyberwave", lambda base_url, api_key: fake_client)
    monkeypatch.setattr(startup, "DriverStartingAlertContext", _FakeAlertContext)
    monkeypatch.setattr(startup, "_maybe_rewrite_jetson_tag", lambda image, _: image)
    monkeypatch.setattr(startup, "_check_and_alert_sensors_devices", lambda *a, **kw: None)
    monkeypatch.setattr(startup, "write_or_update_twin_json_file", lambda *a, **kw: True)
    monkeypatch.setattr(startup, "_clear_stale_driver_starting_alerts", lambda *a, **kw: 0)
    monkeypatch.setattr(startup, "_pull_driver_images_parallel", _fake_pull)
    monkeypatch.setattr(startup, "_run_docker_image", _fake_run_docker_image)
    return state


GO2_DRIVER = driver_container_name(GO2_UUID, "driver")
GO2_BRIDGES = driver_container_name(GO2_UUID, "bridges")
ARM_DRIVER = driver_container_name(ARM_UUID, "driver")


def test_unscoped_run_launches_every_linked_driver(edge: SimpleNamespace) -> None:
    results = startup.fetch_and_run_twin_drivers("tok", "env", "edge-fp")

    assert sorted(edge.launched) == sorted([GO2_DRIVER, GO2_BRIDGES, ARM_DRIVER])
    assert len(results) == 3


def test_scoped_run_launches_only_the_named_service(edge: SimpleNamespace) -> None:
    results = startup.fetch_and_run_twin_drivers(
        "tok", "env", "edge-fp", only_container_names={GO2_BRIDGES}
    )

    assert edge.launched == [GO2_BRIDGES]
    assert edge.pulled == ["cyberwaveos/go2-bridges:humble"]
    assert len(results) == 1


@pytest.mark.parametrize(("on_disk", "pulled"), [(True, []), (False, ["cyberwaveos/arm-driver:humble"])])
def test_a_run_without_pulling_relaunches_the_image_on_disk(
    edge: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, on_disk: bool, pulled: list[str]
) -> None:
    monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda _image: on_disk)

    startup.fetch_and_run_twin_drivers(
        "tok", "env", "edge-fp", only_container_names={ARM_DRIVER}, pull_images=False
    )

    assert edge.launched == [ARM_DRIVER]
    assert edge.pulled == pulled, "only a missing image is pulled"


def test_scoped_run_leaves_other_twins_alone(edge: SimpleNamespace) -> None:
    startup.fetch_and_run_twin_drivers("tok", "env", "edge-fp", only_container_names={ARM_DRIVER})

    assert edge.launched == [ARM_DRIVER]
    assert edge.pulled == ["cyberwaveos/arm-driver:humble"]


def test_scoped_run_still_writes_the_revived_twins_child_camera_json(
    edge: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A camera attached to the arm (e.g. linked after boot) and one attached to the GO2.
    arm_camera, go2_camera = (
        "cccccccc-0000-0000-0000-00000000000a",
        "cccccccc-0000-0000-0000-00000000000b",
    )
    twins = startup.Cyberwave("", "").twins.list(environment_id="env") + [
        SimpleNamespace(
            uuid=uuid,
            name=name,
            asset_uuid="asset-camera",
            asset_id="asset-camera",
            attach_to_twin_uuid=parent,
            metadata={"edge_fingerprint": "edge-fp"},
        )
        for uuid, name, parent in [
            (arm_camera, "arm-cam", ARM_UUID),
            (go2_camera, "go2-cam", GO2_UUID),
        ]
    ]
    client = SimpleNamespace(
        twins=SimpleNamespace(list=lambda environment_id: twins, get_raw=lambda uuid: {}),
        assets=SimpleNamespace(
            get=lambda asset_uuid: SimpleNamespace(metadata={}, camera=asset_uuid == "asset-camera")
        ),
    )
    written: list[str] = []
    monkeypatch.setattr(startup, "Cyberwave", lambda base_url, api_key: client)
    monkeypatch.setattr(startup, "_twin_has_rgb_sensor", lambda asset: asset.camera)
    monkeypatch.setattr(
        startup, "write_or_update_twin_json_file", lambda uuid, *a, **kw: written.append(uuid)
    )

    startup.fetch_and_run_twin_drivers("tok", "env", "edge-fp", only_container_names={ARM_DRIVER})

    assert edge.launched == [ARM_DRIVER]
    assert arm_camera in written
    assert go2_camera not in written


def test_repairing_to_another_environment(
    edge: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sudo cyberwave pair`, then `sudo cyberwave pair` again into environment B.

    The second pair detaches the old twins, links B's twins and restarts
    edge-core; its boot pass is unscoped and launches every twin linked in B.
    Revival afterwards only recreates what is down, and a stopped container of
    a twin the pair detached launches nothing.
    """
    rover_uuid = "77777777-6666-5555-4444-333333333333"
    go2, arm = startup.Cyberwave("", "").twins.list(environment_id="env")
    rover = _services_twin(rover_uuid, "rover", ["driver"])
    twins_by_environment = {"env-a": [go2], "env-b": [arm, rover]}
    client = SimpleNamespace(
        twins=SimpleNamespace(
            list=lambda environment_id: twins_by_environment[environment_id],
            get_raw=lambda uuid: {},
        ),
        assets=SimpleNamespace(get=lambda asset_uuid: SimpleNamespace(metadata={})),
    )
    monkeypatch.setattr(startup, "Cyberwave", lambda base_url, api_key: client)
    environment = {"uuid": "env-a"}
    monkeypatch.setattr(startup, "load_token", lambda: "tok")
    monkeypatch.setattr(startup, "load_saved_fingerprint", lambda: "edge-fp")
    monkeypatch.setattr(startup, "load_environment_uuid", lambda: environment["uuid"])
    monkeypatch.setattr(startup, "_DRIVER_HEALTH_PREVIOUS", {})
    monkeypatch.setattr(startup, "_DRIVER_HEALTH_STALE", False)
    monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {})
    monkeypatch.setattr(startup, "_LAST_REVIVAL_ATTEMPT_AT", None)

    # First pair: boot launches the GO2 in environment A.
    startup.fetch_and_run_twin_drivers("tok", "env-a", "edge-fp")
    assert sorted(edge.launched) == sorted([GO2_DRIVER, GO2_BRIDGES])

    # Second pair into B: the CLI detaches the GO2, links the arm and the rover
    # (the rover is new to this edge) and restarts edge-core.
    go2.metadata.pop("edge_fingerprint")
    environment["uuid"] = "env-b"
    edge.launched.clear()
    startup.fetch_and_run_twin_drivers("tok", "env-b", "edge-fp")
    rover_driver = driver_container_name(rover_uuid, "driver")
    assert sorted(edge.launched) == sorted([ARM_DRIVER, rover_driver])

    # The detached GO2's container is down but still known to this process:
    # revival launches nothing and leaves B's drivers running.
    startup._DRIVER_HEALTH_PREVIOUS.update({GO2_DRIVER: "down", ARM_DRIVER: "running"})
    startup._CONTAINER_TWIN_MAP.update({GO2_DRIVER: GO2_UUID, ARM_DRIVER: ARM_UUID})
    edge.launched.clear()
    startup.reconcile_driver_revival()
    assert edge.launched == []

    # A driver of environment B goes down: only that one is recreated.
    startup._DRIVER_HEALTH_PREVIOUS.update({GO2_DRIVER: "removed", ARM_DRIVER: "down"})
    monkeypatch.setattr(startup, "_LAST_REVIVAL_ATTEMPT_AT", None)
    startup.reconcile_driver_revival()
    assert edge.launched == [ARM_DRIVER]
