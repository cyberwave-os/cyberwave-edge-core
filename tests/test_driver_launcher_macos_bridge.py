"""Unit tests for macOS device bridge helpers in ``driver_launcher``."""

from __future__ import annotations

from unittest.mock import MagicMock

import cyberwave_edge_core.driver_launcher as driver_launcher
import cyberwave_edge_core.startup as startup


def test_macos_bridge_non_darwin_is_noop(monkeypatch):
    monkeypatch.setattr(driver_launcher.platform, "system", lambda: "Linux")

    ok, resolved = driver_launcher._run_macos_device_bridge_commands(
        params=["--device", "/dev/video0:/dev/video0"],
        twin_uuid="twin-1",
        container_name="cyberwave-driver-twin-1",
    )

    assert ok is True
    assert resolved == {}


def test_macos_bridge_skips_video_devices_when_usbip_active(monkeypatch):
    monkeypatch.setattr(driver_launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        startup,
        "_extract_docker_device_mappings",
        lambda params: [("/dev/video0", "/dev/video0")],
    )
    monkeypatch.setattr(
        startup,
        "_is_video_device_path",
        lambda path: path.startswith("/dev/video"),
    )
    monkeypatch.setattr(
        startup,
        "get_runtime_env_var",
        lambda key, default="": "",
    )

    ok, resolved = driver_launcher._run_macos_device_bridge_commands(
        params=[],
        twin_uuid="twin-1",
        container_name="cyberwave-driver-twin-1",
        usbip_active=True,
    )

    assert ok is True
    assert resolved == {"/dev/video0": "/dev/video0"}


def test_macos_bridge_dedupes_duplicate_device_mappings(monkeypatch):
    monkeypatch.setattr(driver_launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        startup,
        "_extract_docker_device_mappings",
        lambda params: [("/dev/ttyUSB0", "/dev/ttyUSB0")],
    )
    monkeypatch.setattr(startup, "_is_video_device_path", lambda path: False)
    monkeypatch.setattr(
        startup,
        "get_runtime_env_var",
        lambda key, default="": "echo {host_device}",
    )

    run_mock = MagicMock(
        return_value=MagicMock(stdout="", stderr="", returncode=0),
    )
    monkeypatch.setattr(driver_launcher.subprocess, "run", run_mock)

    ok, resolved = driver_launcher._run_macos_device_bridge_commands(
        params=[],
        twin_uuid="twin-1",
        container_name="cyberwave-driver-twin-1",
        additional_device_mappings=[
            ("/dev/ttyUSB0", "/dev/ttyUSB0"),
            ("/dev/ttyUSB0", "/dev/ttyUSB0"),
        ],
    )

    assert ok is True
    assert resolved == {"/dev/ttyUSB0": "/dev/ttyUSB0"}
    assert run_mock.call_count == 1
