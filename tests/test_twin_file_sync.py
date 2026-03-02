import importlib
import json
import sys
import types
import uuid

# Provide a lightweight cyberwave stub before importing startup.
cyberwave_stub = types.ModuleType("cyberwave")
cyberwave_stub.Cyberwave = object
cyberwave_fingerprint_stub = types.ModuleType("cyberwave.fingerprint")
cyberwave_fingerprint_stub.generate_fingerprint = lambda: "test-fingerprint"
sys.modules["cyberwave"] = cyberwave_stub
sys.modules["cyberwave.fingerprint"] = cyberwave_fingerprint_stub

startup = importlib.import_module("cyberwave_edge_core.startup")


def test_extract_twin_update_payload_filters_unknown_fields_and_infers_asset_uuid() -> None:
    payload = startup._extract_twin_update_payload(
        {
            "name": "edge twin",
            "metadata": {"camera_id": "front"},
            "asset": {"uuid": "asset-123"},
            "local_only": {"do_not_sync": True},
        }
    )

    assert payload == {
        "name": "edge twin",
        "metadata": {"camera_id": "front"},
        "asset_uuid": "asset-123",
    }


def test_reconcile_twin_json_file_sync_tracks_then_syncs_changed_file(
    tmp_path, monkeypatch
) -> None:
    twin_uuid = str(uuid.uuid4())
    twin_file = tmp_path / f"{twin_uuid}.json"
    calls: list[tuple[str, dict]] = []

    class FakeTwins:
        def update(self, twin_id: str, **kwargs) -> None:
            calls.append((twin_id, kwargs))

    class FakeClient:
        def __init__(self, *, base_url: str, token: str) -> None:
            self.base_url = base_url
            self.token = token
            self.twins = FakeTwins()

    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(startup, "_TWIN_FILE_CHECKSUMS", {})
    monkeypatch.setattr(startup, "load_token", lambda: "token-123")
    monkeypatch.setattr(
        startup,
        "get_runtime_env_var",
        lambda _name, default=None: default,
    )
    monkeypatch.setattr(startup, "Cyberwave", FakeClient)

    twin_file.write_text(
        json.dumps(
            {
                "uuid": twin_uuid,
                "metadata": {"edge_value": 1},
                "asset": {"uuid": "asset-abc"},
                "local_only": {"should_not_be_sent": True},
            }
        )
    )

    summary = startup.reconcile_twin_json_file_sync()
    assert summary == {"tracked": 1, "changed": 0, "synced": 0}
    assert calls == []

    twin_file.write_text(
        json.dumps(
            {
                "uuid": twin_uuid,
                "metadata": {"edge_value": 2},
                "asset": {"uuid": "asset-abc"},
                "local_only": {"should_not_be_sent": False},
            }
        )
    )

    summary = startup.reconcile_twin_json_file_sync()
    assert summary == {"tracked": 1, "changed": 1, "synced": 1}
    assert len(calls) == 1
    sent_twin_uuid, payload = calls[0]
    assert sent_twin_uuid == twin_uuid
    assert payload["metadata"] == {"edge_value": 2}
    assert payload["asset_uuid"] == "asset-abc"
    assert "local_only" not in payload

