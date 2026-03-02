from __future__ import annotations

import sys
import types
from types import SimpleNamespace

fake_cyberwave_module = types.ModuleType("cyberwave")
fake_cyberwave_module.__path__ = []  # type: ignore[attr-defined]
fake_cyberwave_module.Cyberwave = object  # type: ignore[attr-defined]
fake_fingerprint_module = types.ModuleType("cyberwave.fingerprint")
fake_fingerprint_module.generate_fingerprint = (  # type: ignore[attr-defined]
    lambda: "test-fingerprint"
)
fake_cyberwave_module.fingerprint = fake_fingerprint_module  # type: ignore[attr-defined]

sys.modules.setdefault("cyberwave", fake_cyberwave_module)
sys.modules.setdefault("cyberwave.fingerprint", fake_fingerprint_module)

import cyberwave_edge_core.startup as startup


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
        payload = {
            "uuid": self.uuid,
            "name": self.name,
            "metadata": self.metadata,
        }
        if self.attach_to_twin_uuid:
            payload["attach_to_twin_uuid"] = self.attach_to_twin_uuid
        return payload


class FakeAsset:
    def __init__(self, *, metadata: dict, registry_id: str = "") -> None:
        self.metadata = metadata
        self.registry_id = registry_id

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "registry_id": self.registry_id,
        }


class FakeTwinsAPI:
    def __init__(self, twins: list[FakeTwin]) -> None:
        self._twins = twins
        self._by_uuid = {twin.uuid: twin for twin in twins}

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
    return SimpleNamespace(
        twins=FakeTwinsAPI(twins),
        assets=FakeAssetsAPI(assets),
    )


def test_camera_child_twin_driver_is_skipped_and_passed_to_parent(monkeypatch) -> None:
    fingerprint = "edge-fingerprint"
    parent_uuid = "11111111-1111-1111-1111-111111111111"
    child_uuid = "22222222-2222-2222-2222-222222222222"

    parent_asset_uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    child_asset_uuid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    parent_twin = FakeTwin(
        uuid=parent_uuid,
        name="SO101 parent",
        asset_uuid=parent_asset_uuid,
        metadata={
            "edge_fingerprint": fingerprint,
            "drivers": {"default": {"docker_image": "cyberwaveos/so101-driver"}},
        },
    )
    child_camera_twin = FakeTwin(
        uuid=child_uuid,
        name="Wrist camera child",
        asset_uuid=child_asset_uuid,
        attach_to_twin_uuid=parent_uuid,
        metadata={
            "edge_fingerprint": fingerprint,
            "drivers": {"default": {"docker_image": "cyberwaveos/camera-driver"}},
        },
    )

    assets = {
        parent_asset_uuid: FakeAsset(
            metadata={},
            registry_id="the-robot-studio/so101",
        ),
        child_asset_uuid: FakeAsset(
            metadata={
                "universal_schema": {
                    "sensors": [
                        {"id": "camera", "type": "camera"},
                    ]
                }
            },
            registry_id="cyberwave/standard-cam",
        ),
    }
    fake_client = _stub_client([parent_twin, child_camera_twin], assets)

    monkeypatch.setattr(startup, "Cyberwave", lambda base_url, token: fake_client)
    monkeypatch.setattr(startup, "_check_and_alert_sensors_devices", lambda *args, **kwargs: None)

    written_twins: list[str] = []

    def _fake_write(twin_uuid: str, twin_data: dict, asset_data: dict) -> bool:
        written_twins.append(twin_uuid)
        return True

    monkeypatch.setattr(startup, "write_or_update_twin_json_file", _fake_write)

    run_calls: list[dict] = []

    def _fake_run(
        image: str,
        params: list[str],
        *,
        twin_uuid: str,
        token: str,
        child_camera_twin_uuids: list[str] | None = None,
    ) -> bool:
        run_calls.append(
            {
                "image": image,
                "twin_uuid": twin_uuid,
                "child_camera_twin_uuids": child_camera_twin_uuids or [],
            }
        )
        return True

    monkeypatch.setattr(startup, "_run_docker_image", _fake_run)

    results = startup.fetch_and_run_twin_drivers("test-token", "env-uuid", fingerprint)

    assert [result["twin_uuid"] for result in results] == [parent_uuid]
    assert len(run_calls) == 1
    assert run_calls[0]["twin_uuid"] == parent_uuid
    assert run_calls[0]["child_camera_twin_uuids"] == [child_uuid]
    assert set(written_twins) == {parent_uuid, child_uuid}


def test_non_camera_child_twin_is_not_skipped(monkeypatch) -> None:
    fingerprint = "edge-fingerprint"
    parent_uuid = "33333333-3333-3333-3333-333333333333"
    child_uuid = "44444444-4444-4444-4444-444444444444"

    parent_asset_uuid = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    child_asset_uuid = "dddddddd-dddd-dddd-dddd-dddddddddddd"

    parent_twin = FakeTwin(
        uuid=parent_uuid,
        name="Robot parent",
        asset_uuid=parent_asset_uuid,
        metadata={
            "edge_fingerprint": fingerprint,
            "drivers": {"default": {"docker_image": "cyberwaveos/robot-driver"}},
        },
    )
    child_non_camera_twin = FakeTwin(
        uuid=child_uuid,
        name="Auxiliary child",
        asset_uuid=child_asset_uuid,
        attach_to_twin_uuid=parent_uuid,
        metadata={
            "edge_fingerprint": fingerprint,
            "drivers": {"default": {"docker_image": "cyberwaveos/aux-driver"}},
        },
    )

    assets = {
        parent_asset_uuid: FakeAsset(
            metadata={},
            registry_id="the-robot-studio/so101",
        ),
        child_asset_uuid: FakeAsset(
            metadata={
                "universal_schema": {
                    "sensors": [
                        {"id": "laser", "type": "lidar"},
                    ]
                }
            },
            registry_id="cyberwave/lidar",
        ),
    }
    fake_client = _stub_client([parent_twin, child_non_camera_twin], assets)

    monkeypatch.setattr(startup, "Cyberwave", lambda base_url, token: fake_client)
    monkeypatch.setattr(startup, "_check_and_alert_sensors_devices", lambda *args, **kwargs: None)
    monkeypatch.setattr(startup, "write_or_update_twin_json_file", lambda *args, **kwargs: True)

    run_calls: list[dict] = []

    def _fake_run(
        image: str,
        params: list[str],
        *,
        twin_uuid: str,
        token: str,
        child_camera_twin_uuids: list[str] | None = None,
    ) -> bool:
        run_calls.append(
            {
                "image": image,
                "twin_uuid": twin_uuid,
                "child_camera_twin_uuids": child_camera_twin_uuids or [],
            }
        )
        return True

    monkeypatch.setattr(startup, "_run_docker_image", _fake_run)

    results = startup.fetch_and_run_twin_drivers("test-token", "env-uuid", fingerprint)

    assert len(results) == 2
    assert {result["twin_uuid"] for result in results} == {parent_uuid, child_uuid}
    assert len(run_calls) == 2
    by_twin = {call["twin_uuid"]: call for call in run_calls}
    assert by_twin[parent_uuid]["child_camera_twin_uuids"] == []
    assert by_twin[child_uuid]["child_camera_twin_uuids"] == []
