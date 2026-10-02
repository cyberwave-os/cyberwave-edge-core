"""A per-twin twin_to_device change restarts drivers without a global selected_device."""

from __future__ import annotations

import json

import pytest

import cyberwave_edge_core.startup as startup

TWIN = "11111111-1111-4111-8111-111111111111"
PINNED = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_8E3A1B2F-video-index0"


@pytest.fixture
def drift(tmp_path, monkeypatch):
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(startup, "_read_edge_json", lambda: None)
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(startup, "_edge_json_mtime", None)
    monkeypatch.setattr(startup, "load_token", lambda: "tok")
    restarts: list[str] = []
    monkeypatch.setattr(startup, "_start_camera_config_drift_restart", restarts.append)

    def setup(container_device, twin_to_device):
        cameras = tmp_path / "cameras.json"
        cameras.write_text(json.dumps({"twin_to_device": twin_to_device}))
        monkeypatch.setattr(startup, "_cameras_json_mtime", cameras.stat().st_mtime - 10)
        monkeypatch.setattr(startup, "_list_running_driver_containers", lambda: ["c"])
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda name: {
                "Config": {
                    "Env": [
                        f"CYBERWAVE_METADATA_VIDEO_DEVICE={container_device}",
                        f"CYBERWAVE_TWIN_UUID={TWIN}",
                    ]
                }
            },
        )
        return restarts

    return setup


def test_repinned_twin_triggers_restart(drift):
    restarts = drift("/dev/v4l/by-id/usb-OLD-video-index0", {TWIN: PINNED})
    assert startup.reconcile_camera_config_drift() is True
    assert restarts == ["tok"]


def test_matching_container_does_not(drift):
    restarts = drift(PINNED, {TWIN: PINNED})
    assert startup.reconcile_camera_config_drift() is False
    assert restarts == []


def test_container_of_an_unmapped_twin_is_left_alone(drift):
    restarts = drift("/dev/video0", {"other-twin": 1})
    assert startup.reconcile_camera_config_drift() is False
    assert restarts == []
