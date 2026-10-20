from __future__ import annotations

from typing import Any

import pytest

from cyberwave_edge_core import startup

TWIN = "11111111-2222-4333-8444-555555555555"
OTHER = "99999999-2222-4333-8444-555555555555"
BY_ID = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0"
MISSING = "sensor_device_missing"


@pytest.fixture
def edge(monkeypatch: pytest.MonkeyPatch) -> Any:
    state: Any = type("Edge", (), {})()
    state.config = {"selected_device": 0, "twin_to_device": {TWIN: BY_ID}}
    state.sent = []
    state.resolved = []
    state.present = {BY_ID}
    state.checked = []
    monkeypatch.setattr(startup, "_read_cameras_config", lambda: state.config)
    monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {"cyberwave-driver-11111111": TWIN})
    monkeypatch.setattr(startup, "_MISSING_SENSOR_DEVICES", {})
    monkeypatch.setattr(startup, "_SENSOR_DEVICE_STREAKS", {})
    state.cleared = {TWIN}
    monkeypatch.setattr(startup, "_SENSOR_DEVICE_ALERTS_CLEARED", state.cleared)
    monkeypatch.setattr(
        startup,
        "_send_alert_for_twin",
        lambda twin, title, desc, kind, **kw: state.sent.append((twin, kind, kw["metadata"])),
    )
    state.resolve_fails = False

    def _resolve(twin: str, kind: str, *, strict: bool = False) -> int:
        if state.resolve_fails:
            raise RuntimeError("backend unreachable")
        state.resolved.append((twin, kind))
        return 1

    monkeypatch.setattr(startup, "resolve_active_alerts_for_twin", _resolve)

    def _exists(device: str) -> bool:
        state.checked.append(device)
        return device in state.present

    state.check = lambda: startup.reconcile_sensor_device_presence(exists=_exists)
    return state


@pytest.mark.parametrize(
    ("pin", "watched"),
    [
        (2, ["/dev/video2"]),
        ("2", ["/dev/video2"]),
        (BY_ID, [BY_ID]),
        ("/dev/video2", ["/dev/video0"]),
        ("serial:213722070420", []),
        ("rtsp://camera.local/stream", []),
    ],
)
def test_watches_the_device_the_launcher_injects(edge: Any, pin: Any, watched: list[str]) -> None:
    edge.config["twin_to_device"] = {TWIN: pin, OTHER: BY_ID}
    edge.check()
    assert edge.checked == watched


def test_alerts_after_two_missing_checks_and_resolves_after_two_present(edge: Any) -> None:
    edge.check()
    edge.present.clear()
    edge.check()
    assert edge.sent == []
    edge.check()
    edge.check()
    assert edge.sent == [(TWIN, MISSING, {"device": BY_ID})]

    edge.present.add(BY_ID)
    edge.check()
    assert edge.resolved == []
    edge.check()
    assert edge.resolved == [(TWIN, MISSING)]


def test_alternating_presence_never_alerts(edge: Any) -> None:
    for i in range(10):
        edge.present = {BY_ID} if i % 2 else set()
        edge.check()
    assert edge.sent == []


@pytest.mark.parametrize(
    "unwatch",
    [
        lambda edge: edge.config["twin_to_device"].clear(),
        lambda edge: edge.config["twin_to_device"].update({TWIN: "serial:213722070420"}),
        lambda edge: startup._CONTAINER_TWIN_MAP.clear(),
    ],
    ids=["unmapped", "repinned-to-serial", "driver-gone"],
)
def test_alert_is_resolved_once_the_twin_is_no_longer_watched(edge: Any, unwatch: Any) -> None:
    edge.present.clear()
    edge.check()
    edge.check()
    assert len(edge.sent) == 1

    unwatch(edge)
    edge.check()
    assert edge.resolved == [(TWIN, MISSING)]
    edge.check()
    assert edge.resolved == [(TWIN, MISSING)]
    assert startup._MISSING_SENSOR_DEVICES == {}


def test_a_restart_clears_an_alert_left_by_the_previous_process(edge: Any) -> None:
    edge.cleared.clear()
    edge.check()
    assert edge.resolved == [(TWIN, MISSING)]
    edge.check()
    assert edge.resolved == [(TWIN, MISSING)]


def test_a_failed_resolve_is_retried_until_it_succeeds(edge: Any) -> None:
    edge.present.clear()
    edge.check()
    edge.check()
    assert len(edge.sent) == 1

    edge.present.add(BY_ID)
    edge.resolve_fails = True
    edge.check()
    edge.check()
    assert edge.resolved == []
    assert TWIN in startup._MISSING_SENSOR_DEVICES

    edge.resolve_fails = False
    edge.check()
    edge.check()
    assert edge.resolved == [(TWIN, MISSING)]
    assert TWIN not in startup._MISSING_SENSOR_DEVICES
