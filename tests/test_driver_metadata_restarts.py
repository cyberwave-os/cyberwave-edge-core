"""Edge-core restarts sensor drivers after a twin metadata change."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import pytest

from cyberwave_edge_core import startup

TWIN = "11111111-2222-4333-8444-555555555555"
CHILD = "22222222-2222-4333-8444-555555555555"
CONTAINER = "cyberwave-driver-11111111"
MIC_IMAGE = "cyberwaveos/generic-microphone-driver:latest"
ASSET = {"uuid": "asset", "registry_id": "cyberwave/standard-cam"}


def _rfc3339(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@pytest.fixture
def edge(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    state: Any = type("Edge", (), {})()
    state.now = 1000.0
    state.image = MIC_IMAGE
    state.labels = {}
    state.started = time.time() - 3600
    state.restarts = []
    state.alerts = []
    state.inspected = []
    state.edge_restarting = False

    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(startup, "_TWIN_FILE_CHECKSUMS", {})
    monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {CONTAINER: TWIN})
    monkeypatch.setattr(startup, "_DRIVER_METADATA_STATE", {})
    monkeypatch.setattr(startup, "_METADATA_RESTART_TIMES", {})
    monkeypatch.setattr(startup, "_METADATA_RESTART_SUPPRESSED", set())
    monkeypatch.setattr(startup, "_TWIN_JSON_LAST_GOOD", {})
    monkeypatch.setattr(startup, "_TWIN_JSON_REPAIR_WARNED", set())
    monkeypatch.setattr(startup.time, "monotonic", lambda: state.now)
    real_time = time.time
    monkeypatch.setattr(startup.time, "time", lambda: real_time() + state.now - 1000.0)
    monkeypatch.setattr(startup, "_edge_core_restart_in_progress", lambda: state.edge_restarting)
    monkeypatch.setattr(startup, "load_token", lambda: "token")
    monkeypatch.setattr(startup, "load_saved_fingerprint", lambda: "fp")
    monkeypatch.setattr(startup, "load_environment_uuid", lambda **_: "env")

    state.images = {}
    state.started_by = {}
    state.fetch_kwargs = []
    state.watchdog = None
    state.uninspectable = set()
    state.fetch_error = None
    state.launch_ok = True

    def _inspect(name: str) -> dict[str, Any] | None:
        state.inspected.append(name)
        if name in state.uninspectable:
            return None
        return {
            "Config": {"Image": state.images.get(name, state.image), "Labels": state.labels},
            "State": {
                "Status": "running",
                "StartedAt": _rfc3339(state.started_by.get(name, state.started)),
            },
        }

    monkeypatch.setattr(startup, "_inspect_driver_container", _inspect)

    def _fetch_and_run(token, environment_uuid, fingerprint, *, only_container_names=None, **kwargs):
        state.restarts.append(set(only_container_names or ()))
        state.fetch_kwargs.append(kwargs)
        if state.fetch_error:
            raise state.fetch_error
        results = []
        for name in only_container_names or ():
            twin = startup._CONTAINER_TWIN_MAP[name]
            service = name[len(startup.driver_container_name(twin)) + 1 :] or None
            results.append({"twin_uuid": twin, "service_name": service, "success": state.launch_ok})
        return results

    monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", _fetch_and_run)
    state.alert_fails = False

    def _send_alert(twin: str, name: str, desc: str, kind: str, **kw: Any) -> None:
        if state.alert_fails:
            raise RuntimeError("backend unreachable")
        state.alerts.append((twin, kind))

    monkeypatch.setattr(startup, "_send_alert_for_twin", _send_alert)

    def write(twin: str = TWIN, **metadata: Any) -> None:
        path = tmp_path / f"{twin}.json"
        data = json.loads(path.read_text()) if path.exists() else {"uuid": twin}
        data["metadata"] = {**data.get("metadata", {}), **metadata}
        path.write_text(json.dumps(data))

    def launch(twin: str = TWIN, **metadata: Any) -> None:
        startup.write_or_update_twin_json_file(
            twin, {"uuid": twin, "metadata": metadata}, dict(ASSET)
        )

    def tick(seconds: float = 0.0, **changes: Any) -> None:
        state.now += seconds
        startup.reconcile_driver_metadata_restarts(
            {twin: frozenset(keys) for twin, keys in changes.items()}, watchdog=state.watchdog
        )

    state.write, state.launch, state.tick = write, launch, tick
    return state


def _change(edge: Any, key: str, value: Any, twin: str = TWIN) -> None:
    edge.write(twin, **{key: value})
    edge.tick(**{twin: {key}})


def test_settings_change_restarts_the_sensor_driver_once_after_the_debounce(edge: Any) -> None:
    edge.launch(audio_gain_db=0)

    _change(edge, "audio_gain_db", 6)
    edge.tick(5)
    assert edge.restarts == []

    edge.tick(10)
    assert edge.restarts == [{CONTAINER}]
    edge.tick(15)
    edge.tick(600)
    assert edge.restarts == [{CONTAINER}]
    assert edge.alerts == []


def test_a_settings_restart_relaunches_the_image_on_disk_and_pings_the_loop_watchdog(
    edge: Any,
) -> None:
    edge.watchdog = object()
    edge.launch(audio_gain_db=0)

    _change(edge, "audio_gain_db", 6)
    edge.tick(15)

    assert edge.restarts == [{CONTAINER}]
    assert edge.fetch_kwargs == [{"watchdog": edge.watchdog, "pull_images": False}]


def test_saves_inside_the_debounce_become_one_restart(edge: Any) -> None:
    edge.launch(audio_gain_db=0)

    _change(edge, "audio_gain_db", 3)
    edge.now += 8
    _change(edge, "audio_gain_db", 6)
    edge.tick(7)
    assert edge.restarts == []

    edge.tick(8)
    assert edge.restarts == [{CONTAINER}]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "online"),
        ("performance", {"cpu": 40}),
        ("active_map_uuid", "map-1"),
        ("speaker_device", "hw:1,0"),
        ("audio_routing", {"out": "left"}),
        ("audio_device", {"name": "C920", "index": 2}),
    ],
)
def test_runtime_keys_never_restart(edge: Any, key: str, value: Any) -> None:
    edge.launch(audio_gain_db=0)

    _change(edge, key, value)
    edge.tick(30)

    assert edge.restarts == []
    assert edge.inspected == []


def test_a_string_audio_device_is_a_pin(edge: Any) -> None:
    edge.launch(audio_device="hw:1,0")

    _change(edge, "audio_device", "hw:2,0")
    edge.tick(15)

    assert edge.restarts == [{CONTAINER}]


def test_robot_driver_gets_one_alert_and_no_restart(edge: Any) -> None:
    edge.image = "cyberwaveos/go2-driver:latest"
    edge.launch(max_speed=1)

    _change(edge, "max_speed", 2)
    edge.tick(15)
    edge.tick(15)

    assert edge.restarts == []
    assert edge.alerts == [(TWIN, startup.DRIVER_RESTART_NEEDED_ALERT)]


def test_restart_floor_service_gets_an_alert_instead(edge: Any) -> None:
    edge.labels = {"cyberwave.min_restart_interval_s": "60"}
    edge.launch(audio_gain_db=0)

    _change(edge, "audio_gain_db", 6)
    edge.tick(15)

    assert edge.restarts == []
    assert edge.alerts == [(TWIN, startup.DRIVER_RESTART_NEEDED_ALERT)]


@pytest.mark.parametrize(
    "image",
    [
        "cyberwaveos/camera-driver:latest",
        "cyberwaveos/generic-speaker-driver:latest",
        "cyberwaveos/generic-webcam-driver:latest",
    ],
)
def test_every_sensor_driver_is_restarted(edge: Any, image: str) -> None:
    edge.image = image
    edge.launch(fps=10)

    _change(edge, "fps", 15)
    edge.tick(15)

    assert edge.restarts == [{CONTAINER}]


def test_an_edit_in_the_drivers_first_90_s_restarts_once_after_them(edge: Any) -> None:
    edge.started = time.time() - 30
    edge.launch(audio_gain_db=0)

    _change(edge, "audio_gain_db", 6)
    edge.tick(15)
    edge.tick(15, **{TWIN: {"status"}})
    assert edge.restarts == []

    edge.tick(30)
    assert edge.restarts == [{CONTAINER}]
    edge.tick(15)
    edge.tick(600)
    assert edge.restarts == [{CONTAINER}]


def test_a_link_or_unlink_never_restarts(edge: Any) -> None:
    edge.launch(audio_gain_db=0, edge_fingerprint="fp")

    edge.write(edge_fingerprint=None, audio_gain_db=6)
    edge.tick(**{TWIN: {"edge_fingerprint", "audio_gain_db"}})
    edge.tick(15)

    assert edge.restarts == []


def test_waits_for_an_edge_core_restart(edge: Any) -> None:
    edge.launch(audio_gain_db=0)
    edge.edge_restarting = True

    _change(edge, "audio_gain_db", 6)
    edge.tick(15)
    assert edge.restarts == []

    edge.edge_restarting = False
    edge.tick(15)
    assert edge.restarts == [{CONTAINER}]


def test_a_child_camera_restarts_its_parents_containers(edge: Any, tmp_path: Any) -> None:
    edge.image = "cyberwaveos/camera-driver:latest"
    edge.launch(fps=10)
    startup.write_or_update_twin_json_file(
        CHILD,
        {"uuid": CHILD, "attach_to_twin_uuid": TWIN, "metadata": {"fps": 10}},
        {"uuid": "asset"},
    )

    _change(edge, "fps", 30, twin=CHILD)
    edge.tick(15)

    assert edge.restarts == [{CONTAINER}]


def test_restarts_are_rate_limited_then_suppressed_with_one_alert(edge: Any) -> None:
    edge.launch(audio_gain_db=0)

    for gain in (1, 2, 3):
        _change(edge, "audio_gain_db", gain)
        edge.tick(15)
        edge.tick(15)
    assert len(edge.restarts) == 1
    edge.tick(120)
    assert len(edge.restarts) == 2

    edge.now += 200
    _change(edge, "audio_gain_db", 4)
    edge.tick(15)
    assert len(edge.restarts) == 3

    for gain in (5, 6):
        edge.now += 200
        _change(edge, "audio_gain_db", gain)
        edge.tick(15)
    assert len(edge.restarts) == 3
    assert edge.alerts == [(TWIN, startup.DRIVER_METADATA_RESTART_SUPPRESSED_ALERT)]

    edge.now += 1800
    _change(edge, "audio_gain_db", 7)
    edge.tick(15)
    assert len(edge.restarts) == 4


class _FakeTwins:
    def __init__(self) -> None:
        self.backend: dict[str, Any] = {}
        self.updates: list[str] = []

    def update(self, twin_id: str, **kwargs: Any) -> None:
        self.updates.append(twin_id)
        self.backend = json.loads(json.dumps(kwargs["metadata"]))

    def get_raw(self, twin_id: str) -> dict[str, Any]:
        return {"uuid": twin_id, "metadata": self.backend}


def test_a_driver_write_pushed_by_edge_core_never_restarts(
    edge: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    twins = _FakeTwins()

    class _Client:
        def __init__(self, **_: Any) -> None:
            self.twins = twins

    monkeypatch.setattr(startup, "_build_twin_sync_client", lambda: _Client())
    edge.launch(audio_gain_db=0)
    twins.backend = {"audio_gain_db": 0}

    def sync_tick(seconds: float) -> None:
        edge.now += seconds
        summary = startup.reconcile_twin_json_file_sync()
        startup.reconcile_driver_metadata_restarts(summary["metadata_changes"])

    sync_tick(0)
    edge.write(sensors=[{"id": "mic"}])
    sync_tick(15)
    assert twins.updates == [TWIN]
    twins.backend = {**twins.backend, "status": "online"}
    for _ in range(4):
        sync_tick(15)

    assert edge.restarts == []


def test_pull_rewrites_in_place_and_reports_changed_keys(
    edge: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    twins = _FakeTwins()

    class _Client:
        twins = None

    _Client.twins = twins
    monkeypatch.setattr(startup, "_build_twin_sync_client", lambda: _Client())
    edge.launch(audio_gain_db=0, status="idle")
    path = tmp_path / f"{TWIN}.json"
    inode = path.stat().st_ino
    twins.backend = {"audio_gain_db": 6, "status": "idle", "highpass_hz": 80}

    summary = startup.reconcile_twin_json_file_sync()

    assert summary["metadata_changes"] == {TWIN: frozenset({"audio_gain_db", "highpass_hz"})}
    assert path.stat().st_ino == inode
    assert json.loads(path.read_text())["metadata"] == twins.backend


def test_an_uninspectable_sibling_holds_the_restart(
    edge: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sibling = f"{CONTAINER}-base"
    monkeypatch.setitem(startup._CONTAINER_TWIN_MAP, sibling, TWIN)
    edge.images = {sibling: "cyberwaveos/go2-driver:latest"}
    edge.uninspectable = {sibling}
    edge.launch(audio_gain_db=0)

    _change(edge, "audio_gain_db", 6)
    edge.tick(15)
    edge.tick(15)
    assert edge.restarts == []
    assert edge.alerts == []

    edge.uninspectable = set()
    edge.tick(15)
    assert edge.restarts == []
    assert edge.alerts == [(TWIN, startup.DRIVER_RESTART_NEEDED_ALERT)]


def test_a_failed_relaunch_is_retried_after_the_minimum_gap(edge: Any) -> None:
    edge.launch(audio_gain_db=0)
    edge.fetch_error = RuntimeError("twin list failed")

    _change(edge, "audio_gain_db", 6)
    edge.tick(15)
    assert len(edge.restarts) == 1
    edge.fetch_error = None
    edge.tick(60)
    assert len(edge.restarts) == 1

    edge.launch_ok = False
    edge.tick(60)
    assert len(edge.restarts) == 2

    edge.launch_ok = True
    edge.tick(120)
    assert len(edge.restarts) == 3
    edge.tick(300)
    assert len(edge.restarts) == 3


def test_a_relaunch_that_rewrites_the_file_but_fails_to_start_is_retried(
    edge: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    edge.launch(audio_gain_db=0)

    def _rewrite_then_fail(token, environment_uuid, fingerprint, *, only_container_names=None, **_kw):
        edge.restarts.append(set(only_container_names))
        data = json.loads((tmp_path / f"{TWIN}.json").read_text())
        startup.write_or_update_twin_json_file(TWIN, data, {"uuid": "asset"})
        return [{"twin_uuid": TWIN, "success": len(edge.restarts) > 1}]

    monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", _rewrite_then_fail)
    _change(edge, "audio_gain_db", 6)
    edge.tick(15)
    edge.tick(130)

    assert len(edge.restarts) == 2


def test_an_unreadable_read_keeps_the_pending_change(edge: Any, tmp_path: Any) -> None:
    edge.launch(audio_gain_db=0)
    path = tmp_path / f"{TWIN}.json"
    _change(edge, "audio_gain_db", 6)
    good = path.read_text()

    path.write_text(good[: len(good) // 2])
    edge.tick(15)
    assert edge.restarts == []
    assert TWIN in startup._DRIVER_METADATA_STATE

    path.write_text(good)
    edge.tick(15)
    assert edge.restarts == [{CONTAINER}]

    path.unlink()
    startup._DRIVER_METADATA_STATE[TWIN].pending = "x"
    edge.tick(15)
    assert TWIN not in startup._DRIVER_METADATA_STATE


def test_a_sensor_the_driver_registers_through_the_backend_never_restarts(
    edge: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    twins = _FakeTwins()

    class _Client:
        def __init__(self, **_: Any) -> None:
            self.twins = twins

    monkeypatch.setattr(startup, "_build_twin_sync_client", lambda: _Client())
    declared = {"id": "ch0", "type": "microphone", "parameters": {"channel_index": 0}}
    edge.launch(sensors=[declared])
    twins.backend = {"sensors": [declared]}

    def sync_tick(seconds: float) -> None:
        edge.now += seconds
        summary = startup.reconcile_twin_json_file_sync()
        startup.reconcile_driver_metadata_restarts(summary["metadata_changes"])

    sync_tick(0)
    twins.backend = {"sensors": [declared, {"id": "mic", "type": "audio"}]}
    for _ in range(3):
        sync_tick(15)
    assert edge.restarts == []

    tuned = {**declared, "parameters": {"channel_index": 0, "gain_db": 6}}
    twins.backend = {"sensors": [tuned, {"id": "mic", "type": "audio"}]}
    sync_tick(15)
    sync_tick(15)
    assert edge.restarts == [{CONTAINER}]

    edge.now += 200
    added = {"id": "ch1", "type": "microphone", "parameters": {"channel_index": 1}}
    twins.backend = {"sensors": [tuned, {"id": "mic", "type": "audio"}, added]}
    sync_tick(15)
    sync_tick(15)
    assert len(edge.restarts) == 2


def _sync_client(monkeypatch: pytest.MonkeyPatch) -> _FakeTwins:
    twins = _FakeTwins()

    class _Client:
        def __init__(self, **_: Any) -> None:
            self.twins = twins

    monkeypatch.setattr(startup, "_build_twin_sync_client", lambda: _Client())
    return twins


def test_an_unreadable_file_is_repaired_from_the_last_intact_copy(
    edge: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    twins = _sync_client(monkeypatch)
    edge.launch(audio_gain_db=0)
    path = tmp_path / f"{TWIN}.json"
    inode = path.stat().st_ino
    path.write_text('{"uuid": "' + TWIN + '", "metadata": {"audio_gain')
    twins.backend = {"audio_gain_db": 6}

    summary = startup.reconcile_twin_json_file_sync()

    assert summary["metadata_changes"] == {TWIN: frozenset({"audio_gain_db"})}
    assert json.loads(path.read_text()) == {
        "uuid": TWIN,
        "metadata": {"audio_gain_db": 6},
        "asset": ASSET,
    }
    assert path.stat().st_ino == inode


def test_without_an_intact_copy_the_repair_waits_for_the_next_launch(
    edge: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    twins = _sync_client(monkeypatch)
    path = tmp_path / f"{TWIN}.json"
    path.write_text('{"uuid": "' + TWIN)
    twins.backend = {"audio_gain_db": 6}

    with caplog.at_level("WARNING", logger=startup.logger.name):
        for _ in range(3):
            assert startup.reconcile_twin_json_file_sync()["pulled"] == 0

    assert path.read_text() == '{"uuid": "' + TWIN
    assert caplog.text.count("holds no intact copy") == 1


def test_a_full_disk_leaves_the_original_intact(
    edge: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import errno

    twins = _FakeTwins()

    class _Client:
        def __init__(self, **_: Any) -> None:
            self.twins = twins

    monkeypatch.setattr(startup, "_build_twin_sync_client", lambda: _Client())
    edge.launch(audio_gain_db=0)
    path = tmp_path / f"{TWIN}.json"
    before = path.read_bytes()
    twins.backend = {"audio_gain_db": 6}

    def _no_space(_fd: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(startup.os, "fsync", _no_space)
    summary = startup.reconcile_twin_json_file_sync()

    assert summary["pulled"] == 0
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_the_in_place_rewrite_never_truncates_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    path = tmp_path / f"{TWIN}.json"
    path.write_text(json.dumps({"metadata": {"long": "x" * 1000}}))
    inode = path.stat().st_ino
    flags: list[int] = []
    real_open = startup.os.open

    def _open(file: Any, flag: int, *args: Any) -> int:
        if str(file) == str(path):
            flags.append(flag)
        return real_open(file, flag, *args)

    monkeypatch.setattr(startup.os, "open", _open)
    startup._rewrite_twin_json_in_place(path, '{"metadata": {}}\n')

    assert path.read_text() == '{"metadata": {}}\n'
    assert path.stat().st_ino == inode
    assert flags and not any(flag & startup.os.O_TRUNC for flag in flags)


def test_a_first_self_registered_sensor_never_restarts(
    edge: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    twins = _sync_client(monkeypatch)
    edge.launch(audio_gain_db=0)
    twins.backend = {"audio_gain_db": 0}

    def sync_tick(seconds: float) -> None:
        edge.now += seconds
        summary = startup.reconcile_twin_json_file_sync()
        startup.reconcile_driver_metadata_restarts(summary["metadata_changes"])

    sync_tick(0)
    twins.backend = {"audio_gain_db": 0, "sensors": [{"id": "mic", "type": "audio"}]}
    for _ in range(3):
        sync_tick(15)

    assert edge.restarts == []


def test_a_shorter_rewrite_stays_valid_json_until_the_tail_is_cut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    path = tmp_path / f"{TWIN}.json"
    path.write_text(json.dumps({"metadata": {"long": "x" * 1000}}))
    seen: list[Any] = []
    real_ftruncate = startup.os.ftruncate

    def _ftruncate(fd: int, length: int) -> None:
        seen.append(json.loads(path.read_text()))
        real_ftruncate(fd, length)

    monkeypatch.setattr(startup.os, "ftruncate", _ftruncate)
    startup._rewrite_twin_json_in_place(path, '{"metadata": {}}\n')

    assert seen == [{"metadata": {}}]
    assert path.read_text() == '{"metadata": {}}\n'


def test_a_change_first_seen_unreadable_is_revisited(edge: Any, tmp_path: Any) -> None:
    edge.launch(audio_gain_db=0)
    path = tmp_path / f"{TWIN}.json"
    edge.write(audio_gain_db=6)
    good = path.read_text()

    path.write_text(good[:20])
    edge.tick(**{TWIN: {"audio_gain_db"}})
    edge.tick(15)
    assert edge.restarts == []

    path.write_text(good)
    edge.tick(15)
    edge.tick(15)
    assert edge.restarts == [{CONTAINER}]


def test_a_partial_two_service_relaunch_keeps_the_change_pending(
    edge: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = f"{CONTAINER}-left", f"{CONTAINER}-right"
    monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {first: TWIN, second: TWIN})
    edge.launch(audio_gain_db=0)

    def _relaunch(token, environment_uuid, fingerprint, *, only_container_names=None, **_kw):
        edge.restarts.append(set(only_container_names))
        edge.started_by[first] = time.time()
        return [
            {"twin_uuid": TWIN, "service_name": "left", "success": True},
            {"twin_uuid": TWIN, "service_name": "right", "success": len(edge.restarts) > 1},
        ]

    monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", _relaunch)
    _change(edge, "audio_gain_db", 6)
    edge.tick(15)
    assert edge.restarts == [{first, second}]

    edge.tick(60)
    assert len(edge.restarts) == 1
    edge.tick(60)
    assert len(edge.restarts) == 2
    edge.tick(300)
    assert len(edge.restarts) == 2


def test_an_undelivered_restart_needed_alert_keeps_the_change_pending(edge: Any) -> None:
    edge.image = "cyberwaveos/go2-driver:latest"
    edge.launch(max_speed=1)
    edge.alert_fails = True

    _change(edge, "max_speed", 2)
    edge.tick(15)
    edge.tick(15)
    assert edge.alerts == []
    assert startup._DRIVER_METADATA_STATE[TWIN].pending is not None

    edge.alert_fails = False
    edge.tick(15)
    edge.tick(15)
    assert edge.alerts == [(TWIN, startup.DRIVER_RESTART_NEEDED_ALERT)]
    assert edge.restarts == []


def test_an_undelivered_suppression_alert_is_retried(edge: Any) -> None:
    edge.launch(audio_gain_db=0)
    for gain in (1, 2, 3):
        edge.now += 200
        _change(edge, "audio_gain_db", gain)
        edge.tick(15)
    assert len(edge.restarts) == 3

    edge.alert_fails = True
    edge.now += 200
    _change(edge, "audio_gain_db", 4)
    edge.tick(15)
    assert edge.alerts == []

    edge.alert_fails = False
    edge.tick(15)
    edge.tick(15)
    assert edge.alerts == [(TWIN, startup.DRIVER_METADATA_RESTART_SUPPRESSED_ALERT)]
    assert len(edge.restarts) == 3


def _backend_ticks(edge: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeTwins, Any]:
    twins = _sync_client(monkeypatch)

    def sync_tick(seconds: float) -> None:
        edge.now += seconds
        summary = startup.reconcile_twin_json_file_sync()
        startup.reconcile_driver_metadata_restarts(summary["metadata_changes"])

    return twins, sync_tick


def test_a_driver_startup_write_through_the_backend_never_restarts(
    edge: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    twins, sync_tick = _backend_ticks(edge, monkeypatch)
    edge.image = "cyberwaveos/generic-webcam-driver:latest"
    edge.started = time.time() - 10
    edge.launch(audio_gain_db=0)
    twins.backend = {"audio_gain_db": 0}
    sync_tick(0)

    twins.backend = {
        "audio_gain_db": 0,
        "audio_device": {"name": "C920", "index": 2},
        "sensors": [{"id": "mic", "type": "audio"}],
        "speaker_device": "hw:2,0",
    }
    for _ in range(10):
        sync_tick(15)

    assert edge.restarts == []


def test_a_ui_edit_right_after_a_restart_reaches_the_driver(
    edge: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    twins, sync_tick = _backend_ticks(edge, monkeypatch)
    edge.started = time.time() - 5
    edge.launch(audio_gain_db=0)
    twins.backend = {"audio_gain_db": 0}
    sync_tick(0)

    twins.backend = {"audio_gain_db": 6}
    sync_tick(15)
    sync_tick(15)
    sync_tick(15)
    assert edge.restarts == []

    for _ in range(4):
        sync_tick(15)
    assert edge.restarts == [{CONTAINER}]
