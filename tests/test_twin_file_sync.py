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


def test_extract_twin_update_payload_filters_unknown_fields() -> None:
    payload = startup._extract_twin_update_payload(
        {
            "name": "edge twin",
            "metadata": {"camera_id": "front"},
            "asset": {"uuid": "asset-123"},
            "asset_uuid": "asset-123",
            "environment_uuid": "env-456",
            "local_only": {"do_not_sync": True},
        }
    )

    assert payload == {
        "name": "edge twin",
        "metadata": {"camera_id": "front"},
    }
    assert "asset_uuid" not in payload
    assert "environment_uuid" not in payload
    assert "local_only" not in payload


class FakeTwins:
    """Records ``update`` calls and serves canned ``get_raw`` responses."""

    def __init__(self, get_responses: dict[str, dict] | None = None) -> None:
        self.update_calls: list[tuple[str, dict]] = []
        self.get_raw_calls: list[str] = []
        self.get_responses: dict[str, dict] = get_responses or {}

    def update(self, twin_id: str, **kwargs) -> None:
        self.update_calls.append((twin_id, kwargs))

    def get_raw(self, twin_id: str) -> dict:
        self.get_raw_calls.append(twin_id)
        if twin_id not in self.get_responses:
            raise LookupError(f"no fake response for {twin_id}")
        return self.get_responses[twin_id]


def _install_fake_client(monkeypatch, twins: FakeTwins, *, tmp_path) -> None:
    class FakeClient:
        def __init__(self, *, base_url: str, token: str) -> None:
            self.base_url = base_url
            self.token = token
            self.twins = twins

    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(startup, "_TWIN_FILE_CHECKSUMS", {})
    monkeypatch.setattr(startup, "load_token", lambda: "token-123")
    monkeypatch.setattr(
        startup,
        "get_runtime_env_var",
        lambda _name, default=None: default,
    )
    monkeypatch.setattr(startup, "Cyberwave", FakeClient)


def test_reconcile_twin_json_file_sync_tracks_then_syncs_changed_file(
    tmp_path, monkeypatch
) -> None:
    twin_uuid = str(uuid.uuid4())
    twin_file = tmp_path / f"{twin_uuid}.json"
    # Backend mirrors the initial local metadata so the first pull is a no-op.
    twins = FakeTwins(
        get_responses={
            twin_uuid: {
                "uuid": twin_uuid,
                "metadata": {"edge_value": 1},
            }
        }
    )
    _install_fake_client(monkeypatch, twins, tmp_path=tmp_path)

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

    # First cycle tracks the file. The pull leg fires but is a no-op because
    # local metadata already matches the backend.
    summary = startup.reconcile_twin_json_file_sync()
    assert summary["tracked"] == 1
    assert summary["changed"] == 0
    assert summary["synced"] == 0
    assert summary["pulled"] == 0
    assert twins.update_calls == []
    assert twins.get_raw_calls == [twin_uuid]

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
    assert summary["tracked"] == 1
    assert summary["changed"] == 1
    assert summary["synced"] == 1
    # Push happened first; pull is skipped for files that just pushed.
    assert summary["pulled"] == 0
    assert len(twins.update_calls) == 1
    sent_twin_uuid, payload = twins.update_calls[0]
    assert sent_twin_uuid == twin_uuid
    assert payload["metadata"] == {"edge_value": 2}
    assert "asset_uuid" not in payload
    assert "local_only" not in payload
    # No additional get_raw this cycle (push leg only).
    assert twins.get_raw_calls == [twin_uuid]


def test_reconcile_pulls_backend_metadata_into_local_when_unchanged_locally(
    tmp_path, monkeypatch
) -> None:
    twin_uuid = str(uuid.uuid4())
    twin_file = tmp_path / f"{twin_uuid}.json"
    backend_metadata = {
        "frame_filter_enabled": True,
        "ui_set_value": "from-backend",
    }
    twins = FakeTwins(
        get_responses={
            twin_uuid: {
                "uuid": twin_uuid,
                "metadata": backend_metadata,
                # A field outside the pull allowlist must be left alone.
                "name": "Renamed by UI but not pullable",
            }
        }
    )
    _install_fake_client(monkeypatch, twins, tmp_path=tmp_path)

    local_payload = {
        "uuid": twin_uuid,
        "name": "edge twin",
        "metadata": {"frame_filter_enabled": False},
        "asset": {"uuid": "asset-abc"},
    }
    twin_file.write_text(json.dumps(local_payload, indent=2))

    # First cycle: tracks the file AND pulls backend metadata in one go.
    # This is the user-facing behavior: a UI-set flag converges to the edge
    # without requiring an edge-core restart.
    summary = startup.reconcile_twin_json_file_sync()
    assert summary["tracked"] == 1
    assert summary["pulled"] == 1
    assert summary["synced"] == 0
    assert twins.get_raw_calls == [twin_uuid]
    assert twins.update_calls == []

    on_disk = json.loads(twin_file.read_text())
    assert on_disk["metadata"] == backend_metadata
    # Pull is allowlist-only: name must NOT have been overwritten by backend.
    assert on_disk["name"] == "edge twin"
    assert on_disk["asset"] == {"uuid": "asset-abc"}

    # Second cycle: backend value already matches local -> no further pull.
    summary = startup.reconcile_twin_json_file_sync()
    assert summary["pulled"] == 0
    assert summary["synced"] == 0


def test_reconcile_skips_pull_for_pushed_files_in_same_cycle(tmp_path, monkeypatch) -> None:
    twin_uuid = str(uuid.uuid4())
    twin_file = tmp_path / f"{twin_uuid}.json"
    twins = FakeTwins(
        get_responses={
            twin_uuid: {
                "uuid": twin_uuid,
                "metadata": {"server_only": True},
            }
        }
    )
    _install_fake_client(monkeypatch, twins, tmp_path=tmp_path)

    twin_file.write_text(json.dumps({"uuid": twin_uuid, "metadata": {"v": 1}}))
    startup.reconcile_twin_json_file_sync()  # initial track + first pull
    initial_pull_calls = list(twins.get_raw_calls)

    twin_file.write_text(json.dumps({"uuid": twin_uuid, "metadata": {"v": 2}}))
    summary = startup.reconcile_twin_json_file_sync()
    assert summary["synced"] == 1
    # Pushed file: skip the pull leg this cycle.
    assert summary["pulled"] == 0
    # No additional get_raw beyond the initial track cycle.
    assert twins.get_raw_calls == initial_pull_calls


def test_reconcile_pull_failure_is_swallowed(tmp_path, monkeypatch) -> None:
    twin_uuid = str(uuid.uuid4())
    twin_file = tmp_path / f"{twin_uuid}.json"

    class FailingTwins(FakeTwins):
        def get_raw(self, twin_id: str) -> dict:
            self.get_raw_calls.append(twin_id)
            raise RuntimeError("backend down")

    twins = FailingTwins()
    _install_fake_client(monkeypatch, twins, tmp_path=tmp_path)

    twin_file.write_text(json.dumps({"uuid": twin_uuid, "metadata": {"v": 1}}))
    startup.reconcile_twin_json_file_sync()  # initial track (also attempts pull)

    summary = startup.reconcile_twin_json_file_sync()
    assert summary["pulled"] == 0
    assert summary["synced"] == 0
    # We attempted the pull every cycle (so connectivity issues stay visible
    # in logs); the failures must not raise out of the reconcile.
    assert twins.get_raw_calls == [twin_uuid, twin_uuid]
    # Local file is untouched.
    assert json.loads(twin_file.read_text()) == {"uuid": twin_uuid, "metadata": {"v": 1}}


def test_coerce_twin_to_dict_handles_object_with_to_dict() -> None:
    class FakeTwin:
        def __init__(self, data: dict) -> None:
            self._data = data

        def to_dict(self) -> dict:
            return self._data

    payload = {"uuid": "abc", "metadata": {"k": "v"}}
    assert startup._coerce_twin_to_dict(FakeTwin(payload)) == payload


def test_coerce_twin_to_dict_returns_dict_unchanged() -> None:
    payload = {"uuid": "abc"}
    assert startup._coerce_twin_to_dict(payload) is payload


def test_coerce_twin_to_dict_returns_none_when_unsupported() -> None:
    class Opaque:
        __slots__ = ()

    assert startup._coerce_twin_to_dict(Opaque()) is None
