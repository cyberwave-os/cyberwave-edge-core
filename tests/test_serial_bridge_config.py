"""edge-core side of the macOS serial bridge.

The CLI publishes each serial device as a TCP listener on the host and records
the mapping in ``~/.cyberwave/serial_bridges.json`` (same shape as
``camera_streams.json``). edge-core reads it and tells the driver container
which ports to turn back into PTYs.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import cyberwave_edge_core.driver_launcher as driver_launcher
import cyberwave_edge_core.startup as startup

from .test_multi_service_driver import _patch_driver_container_launch


@pytest.fixture(autouse=True)
def _assume_bridges_are_listening(monkeypatch):
    """Liveness is asserted by its own tests below."""
    monkeypatch.setattr(startup, "_probe_macos_host_tcp_port", lambda h, p, **k: True)


def _write_bridges(tmp_path, payload: Any) -> None:
    (tmp_path / "serial_bridges.json").write_text(json.dumps(payload))


def test_ports_are_read_in_declared_order(monkeypatch, tmp_path):
    """Order decides which arm becomes ttyACM0 vs ttyACM1, so it must be stable."""
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    _write_bridges(
        tmp_path,
        {
            "bridges": [
                {"slot": 0, "device": "/dev/cu.usbmodemA", "port": 8100},
                {"slot": 1, "device": "/dev/cu.usbmodemB", "port": 8101},
            ]
        },
    )

    assert startup._load_serial_bridge_ports() == [8100, 8101]


def test_missing_file_yields_no_ports(monkeypatch, tmp_path):
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

    assert startup._load_serial_bridge_ports() == []


def test_malformed_file_yields_no_ports(monkeypatch, tmp_path):
    """A corrupt config must degrade to "no bridge", not crash driver startup."""
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    (tmp_path / "serial_bridges.json").write_text("{not json")

    assert startup._load_serial_bridge_ports() == []


@pytest.mark.parametrize("payload", [[{"port": 8300}], "stale", 1])
def test_wrong_top_level_type_yields_no_ports(monkeypatch, tmp_path, payload):
    """Valid JSON in an old or corrupt shape must use the documented USB/IP
    fallback instead of aborting every driver launch with ``AttributeError``."""
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    _write_bridges(tmp_path, payload)

    assert startup._load_serial_bridge_ports() == []


@pytest.mark.parametrize("bridges", [1, "stale", {"port": 8300}])
def test_wrong_bridges_collection_type_yields_no_ports(monkeypatch, tmp_path, bridges):
    """The top-level object can still carry a corrupt nested value; iteration
    must not abort driver replacement before the USB/IP fallback starts."""
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    _write_bridges(tmp_path, {"bridges": bridges})

    assert startup._load_serial_bridge_ports() == []


def test_entries_without_a_usable_port_are_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    _write_bridges(
        tmp_path,
        {
            "bridges": [
                {"slot": 0, "device": "/dev/cu.usbmodemA"},
                {"slot": 1, "device": "/dev/cu.usbmodemB", "port": "8101"},
                {"slot": 2, "device": "/dev/cu.usbmodemC", "port": 0},
            ]
        },
    )

    assert startup._load_serial_bridge_ports() == [8101]


def _launch_on_macos(
    monkeypatch: Any,
    ports: list[int],
    *,
    image: str = "cyberwaveos/so101-driver:latest",
    supports_serial_bridge: bool = True,
    child_camera_uuids: list[str] | None = None,
    child_camera_urls: dict[str, str] | None = None,
) -> list[list[str]]:
    captured_cmds: list[list[str]] = []
    _patch_driver_container_launch(monkeypatch, captured_cmds)
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: True)
    monkeypatch.setattr(startup, "_load_serial_bridge_ports", lambda: list(ports))
    monkeypatch.setattr(
        driver_launcher,
        "_image_supports_serial_bridge",
        lambda candidate: supports_serial_bridge,
        raising=False,
    )
    if child_camera_urls is not None:
        monkeypatch.setattr(
            startup,
            "_load_camera_stream_urls_for_twins",
            lambda uuids: {
                uuid: child_camera_urls[uuid]
                for uuid in uuids
                if uuid in child_camera_urls
            },
        )
    monkeypatch.setattr(
        driver_launcher,
        "_usbip_preattach_serial_devices",
        lambda **kw: ["/dev/ttyACM0"],
    )

    startup._run_docker_image(
        image,
        [],
        twin_uuid="aabbccdd-1234-5678-9012-abcdef012345",
        token="test-token",
        skip_pull=True,
        child_camera_twin_uuids=child_camera_uuids,
    )
    return [c for c in captured_cmds if c[:2] == ["docker", "create"]]


def _env_map(cmd: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i, tok in enumerate(cmd):
        if tok == "-e" and i + 1 < len(cmd):
            key, sep, value = cmd[i + 1].partition("=")
            if sep:
                out[key] = value
    return out


def test_configured_ports_reach_the_container(monkeypatch):
    create_cmds = _launch_on_macos(monkeypatch, [8100, 8101])

    assert _env_map(create_cmds[0]).get("CYBERWAVE_SERIAL_BRIDGE_PORTS") == "8100,8101"


def test_child_camera_stream_urls_reach_the_parent_container(monkeypatch):
    urls = {
        "camera-a": "http://host.docker.internal:8091",
        "camera-b": "http://host.docker.internal:8092",
        "camera-c": "http://host.docker.internal:8093",
    }
    create_cmds = _launch_on_macos(
        monkeypatch,
        [8300, 8301],
        child_camera_uuids=list(urls),
        child_camera_urls=urls,
    )

    assert json.loads(
        _env_map(create_cmds[0])["CYBERWAVE_CHILD_CAMERA_STREAM_URLS"]
    ) == urls


def test_usbip_serial_preattach_is_skipped_when_bridging(monkeypatch):
    """Both paths create /dev/ttyACM*; pre-attaching as well would collide and
    waste a helper container on every driver start."""
    create_cmds = _launch_on_macos(monkeypatch, [8100])

    env = _env_map(create_cmds[0])
    assert "CYBERWAVE_USBIP_PREATTACHED" not in env
    tty_devices = [
        create_cmds[0][i + 1]
        for i, tok in enumerate(create_cmds[0])
        if tok == "--device" and i + 1 < len(create_cmds[0]) and "ttyACM" in create_cmds[0][i + 1]
    ]
    assert tty_devices == []


def test_usbip_still_used_when_no_bridge_is_configured(monkeypatch):
    """Linux hosts and un-bridged macOS setups keep the existing behaviour."""
    create_cmds = _launch_on_macos(monkeypatch, [])

    env = _env_map(create_cmds[0])
    assert "CYBERWAVE_SERIAL_BRIDGE_PORTS" not in env
    assert env.get("CYBERWAVE_USBIP_PREATTACHED") == "1"


def test_serial_bridge_config_does_not_disable_usbip_for_unsupported_images(monkeypatch):
    """Bridge ports describe SO101 devices, not a host-wide transport choice.
    Camera and cached legacy images that do not declare bridge support must
    retain their existing USB/IP path."""
    create_cmds = _launch_on_macos(
        monkeypatch,
        [8300],
        image="cyberwaveos/camera-driver:latest",
        supports_serial_bridge=False,
    )

    cmd = create_cmds[0]
    env = _env_map(cmd)
    assert "CYBERWAVE_SERIAL_BRIDGE_PORTS" not in env
    assert env.get("CYBERWAVE_USBIP_ENABLED") == "1"
    assert "--pid=host" in cmd


def test_ports_with_no_listener_are_dropped(monkeypatch, tmp_path):
    """serial_bridges.json is a snapshot from install time with a /dev/cu.* path
    baked into each wrapper. macOS callout names encode USB location, so moving
    an arm to another port leaves the wrapper waiting forever on a device that
    never appears — nothing listens, yet a non-empty config would still suppress
    the USB/IP fallback entirely."""
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    _write_bridges(
        tmp_path,
        {
            "bridges": [
                {"slot": 0, "device": "/dev/cu.usbmodemA", "port": 8300},
                {"slot": 1, "device": "/dev/cu.usbmodemB", "port": 8301},
            ]
        },
    )
    monkeypatch.setattr(
        startup, "_probe_macos_host_tcp_port", lambda host, port, **k: port == 8300
    )

    assert startup._load_serial_bridge_ports() == [8300]


def test_config_with_nothing_listening_falls_back_to_usbip(monkeypatch, tmp_path):
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    _write_bridges(
        tmp_path, {"bridges": [{"slot": 0, "device": "/dev/cu.usbmodemA", "port": 8300}]}
    )
    monkeypatch.setattr(startup, "_probe_macos_host_tcp_port", lambda *a, **k: False)

    assert startup._load_serial_bridge_ports() == []


def test_bridged_launch_does_not_claim_usbip_is_active(monkeypatch):
    """With a bridge configured the entrypoint skips the USB/IP block entirely,
    so every decision premised on an attach happening is false: --pid=host is
    unnecessary, the video readiness wait never runs, and suppressing the
    "camera not configured" warning hides the one diagnostic that explains a
    missing camera."""
    create_cmds = _launch_on_macos(monkeypatch, [8300, 8301])

    cmd = create_cmds[0]
    env = _env_map(cmd)
    assert "CYBERWAVE_USBIP_ENABLED" not in env
    assert "CYBERWAVE_USBIP_VIDEO_TIMEOUT_SECS" not in env
    assert "--pid=host" not in cmd


def test_unbridged_launch_still_enables_usbip(monkeypatch):
    create_cmds = _launch_on_macos(monkeypatch, [])

    cmd = create_cmds[0]
    assert _env_map(cmd).get("CYBERWAVE_USBIP_ENABLED") == "1"
    assert "--pid=host" in cmd
