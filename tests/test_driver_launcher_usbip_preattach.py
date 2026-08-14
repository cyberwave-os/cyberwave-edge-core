"""Unit tests for the macOS USB/IP host-side pre-attach in ``driver_launcher``.

Docker snapshots the VM's device list into a privileged container's private
tmpfs ``/dev`` at *creation* time. The driver container therefore cannot see
``/dev/ttyACM*`` nodes that its own entrypoint attaches after start. These
tests cover the pre-attach that makes the nodes exist beforehand.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

import cyberwave_edge_core.driver_launcher as driver_launcher
import cyberwave_edge_core.startup as startup

from .test_multi_service_driver import _patch_driver_container_launch

_REAL_CAPABILITY_PROBE = driver_launcher._image_supports_usbip_attach_only


@pytest.fixture(autouse=True)
def _assume_image_supports_attach_only(monkeypatch):
    """These tests exercise the attach itself; the capability gate has its own
    tests below, which call the real probe via _REAL_CAPABILITY_PROBE."""
    monkeypatch.setattr(
        driver_launcher, "_image_supports_usbip_attach_only", lambda img: True
    )


def _fake_completed(stdout: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_preattach_returns_device_paths_reported_by_helper(monkeypatch):
    """The helper prints ``USBIP_DEVICE=`` lines; each becomes a VM path."""
    monkeypatch.setattr(
        driver_launcher.subprocess,
        "run",
        lambda *a, **k: _fake_completed(
            "[usbip] Attaching device 0-1-1\nUSBIP_DEVICE=/dev/ttyACM0\nUSBIP_DEVICE=/dev/ttyACM1\n"
        ),
    )

    devices = driver_launcher._usbip_preattach_serial_devices(image="so101:latest")

    assert devices == ["/dev/ttyACM0", "/dev/ttyACM1"]


def test_preattach_deduplicates_repeated_paths(monkeypatch):
    monkeypatch.setattr(
        driver_launcher.subprocess,
        "run",
        lambda *a, **k: _fake_completed("USBIP_DEVICE=/dev/ttyACM0\nUSBIP_DEVICE=/dev/ttyACM0\n"),
    )

    assert driver_launcher._usbip_preattach_serial_devices(image="so101:latest") == ["/dev/ttyACM0"]


def test_preattach_returns_empty_when_helper_fails(monkeypatch):
    """A non-zero helper must degrade to the in-container fallback, not raise."""
    monkeypatch.setattr(
        driver_launcher.subprocess,
        "run",
        lambda *a, **k: _fake_completed("USBIP_DEVICE=/dev/ttyACM0\n", returncode=1),
    )

    assert driver_launcher._usbip_preattach_serial_devices(image="so101:latest") == []


def test_preattach_returns_empty_when_docker_unavailable(monkeypatch):
    def _boom(*a, **k):
        raise OSError("docker not found")

    monkeypatch.setattr(driver_launcher.subprocess, "run", _boom)

    assert driver_launcher._usbip_preattach_serial_devices(image="so101:latest") == []


def test_preattach_returns_empty_on_timeout(monkeypatch):
    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=1)

    monkeypatch.setattr(driver_launcher.subprocess, "run", _timeout)

    assert driver_launcher._usbip_preattach_serial_devices(image="so101:latest") == []


def test_preattach_helper_runs_privileged_with_host_pid_namespace(monkeypatch):
    """nsenter into the VM requires --pid=host; --rm keeps no stray container."""
    captured: dict[str, list[str]] = {}

    def _capture(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        return _fake_completed("")

    monkeypatch.setattr(driver_launcher.subprocess, "run", _capture)

    driver_launcher._usbip_preattach_serial_devices(image="so101:latest")

    cmd = captured["cmd"]
    assert cmd[:3] == ["docker", "run"] + ["--rm"]
    assert "--privileged" in cmd
    assert "--pid=host" in cmd
    assert "so101:latest" in cmd
    assert driver_launcher._USBIP_ATTACH_ONLY_FLAG in cmd
    # The image must come before the flag so it is parsed as a container arg.
    assert cmd.index("so101:latest") < cmd.index(driver_launcher._USBIP_ATTACH_ONLY_FLAG)


def _launch_on_macos_with_usbip(monkeypatch: Any, preattached: list[str]) -> list[list[str]]:
    """Run a driver launch on a simulated macOS+USB/IP host, return docker cmds."""
    captured_cmds: list[list[str]] = []
    _patch_driver_container_launch(monkeypatch, captured_cmds)
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: True)
    monkeypatch.setattr(
        driver_launcher,
        "_usbip_preattach_serial_devices",
        lambda **kw: list(preattached),
    )

    startup._run_docker_image(
        "cyberwaveos/so101-driver:latest",
        [],
        twin_uuid="aabbccdd-1234-5678-9012-abcdef012345",
        token="test-token",
        skip_pull=True,
    )
    return [c for c in captured_cmds if c[:2] == ["docker", "create"]]


def test_preattached_nodes_are_mapped_into_the_driver_container(monkeypatch):
    """The whole point: nodes attached before create must reach the container."""
    create_cmds = _launch_on_macos_with_usbip(monkeypatch, ["/dev/ttyACM0", "/dev/ttyACM1"])

    assert create_cmds, "expected a docker create command"
    cmd = create_cmds[0]
    pairs = {cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--device" and i + 1 < len(cmd)}
    assert "/dev/ttyACM0:/dev/ttyACM0" in pairs
    assert "/dev/ttyACM1:/dev/ttyACM1" in pairs


def _env_map(cmd: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i, tok in enumerate(cmd):
        if tok == "-e" and i + 1 < len(cmd):
            key, sep, value = cmd[i + 1].partition("=")
            if sep:
                out[key] = value
    return out


def test_container_is_told_devices_were_preattached(monkeypatch):
    """Without this flag the entrypoint would detach the very imports
    edge-core just created for it, and duplicate the rest."""
    create_cmds = _launch_on_macos_with_usbip(monkeypatch, ["/dev/ttyACM0"])

    assert _env_map(create_cmds[0]).get("CYBERWAVE_USBIP_PREATTACHED") == "1"


def test_preattached_flag_absent_when_nothing_was_attached(monkeypatch):
    """The entrypoint must run its own full attach when the pre-attach found
    nothing, so it must not be told the work was already done."""
    create_cmds = _launch_on_macos_with_usbip(monkeypatch, [])

    assert "CYBERWAVE_USBIP_PREATTACHED" not in _env_map(create_cmds[0])


def test_no_device_flags_added_when_preattach_finds_nothing(monkeypatch):
    """A failed pre-attach must not inject bogus --device flags that would make
    docker create fail outright; the in-container fallback handles it."""
    create_cmds = _launch_on_macos_with_usbip(monkeypatch, [])

    assert create_cmds, "expected a docker create command"
    cmd = create_cmds[0]
    tty_devices = [
        cmd[i + 1]
        for i, tok in enumerate(cmd)
        if tok == "--device" and i + 1 < len(cmd) and "ttyACM" in cmd[i + 1]
    ]
    assert tty_devices == []


# --- the helper must only run against images that understand the flag --------


def test_helper_is_skipped_for_images_that_do_not_declare_support(monkeypatch):
    """_run_docker_image is the generic launcher for every driver. Running the
    helper against an arbitrary image passes --usbip-attach-only straight to
    that driver's main.py, booting a second privileged --pid=host copy of it
    with no environment, which then outlives the client timeout as an orphan."""
    monkeypatch.setattr(driver_launcher, "_image_supports_usbip_attach_only", lambda img: False)
    ran: list[list[str]] = []
    monkeypatch.setattr(
        driver_launcher.subprocess, "run", lambda cmd, *a, **k: ran.append(list(cmd))
    )

    assert driver_launcher._usbip_preattach_serial_devices(image="camera:latest") == []
    assert ran == [], "no container may be launched for an unsupported image"


def test_helper_runs_for_images_that_declare_support(monkeypatch):
    monkeypatch.setattr(driver_launcher, "_image_supports_usbip_attach_only", lambda img: True)
    monkeypatch.setattr(
        driver_launcher.subprocess,
        "run",
        lambda *a, **k: _fake_completed("USBIP_DEVICE=/dev/ttyACM0\n"),
    )

    assert driver_launcher._usbip_preattach_serial_devices(image="so101:latest") == [
        "/dev/ttyACM0"
    ]


def test_capability_probe_reads_the_image_label(monkeypatch):
    captured: dict[str, list[str]] = {}

    def _capture(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        return _fake_completed("1\n")

    monkeypatch.setattr(driver_launcher.subprocess, "run", _capture)

    assert _REAL_CAPABILITY_PROBE("so101:latest") is True
    assert "inspect" in captured["cmd"]
    assert driver_launcher._USBIP_ATTACH_ONLY_LABEL in " ".join(captured["cmd"])


def test_capability_probe_is_false_when_label_absent(monkeypatch):
    monkeypatch.setattr(
        driver_launcher.subprocess, "run", lambda *a, **k: _fake_completed("<no value>\n")
    )

    assert _REAL_CAPABILITY_PROBE("camera:latest") is False


def test_capability_probe_is_false_when_docker_fails(monkeypatch):
    def _boom(*a, **k):
        raise OSError("docker gone")

    monkeypatch.setattr(driver_launcher.subprocess, "run", _boom)

    assert _REAL_CAPABILITY_PROBE("so101:latest") is False


def test_helper_forwards_usbip_configuration(monkeypatch):
    """CYBERWAVE_USBIP_* knobs are documented operator controls; the helper
    performs the attach, so ignoring them silently overrides the operator (e.g.
    _BUSID opts out of attaching everything, _DETACH_STALE=0 protects other
    imports in the shared VM)."""
    monkeypatch.setattr(driver_launcher, "_image_supports_usbip_attach_only", lambda img: True)
    captured: dict[str, list[str]] = {}

    def _capture(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        return _fake_completed("")

    monkeypatch.setattr(driver_launcher.subprocess, "run", _capture)

    driver_launcher._usbip_preattach_serial_devices(
        image="so101:latest",
        env={
            "CYBERWAVE_USBIP_BUSID": "1-1-1",
            "CYBERWAVE_USBIP_DETACH_STALE": "0",
            "CYBERWAVE_USBIP_HOST": "10.0.0.1",
            "UNRELATED": "x",
        },
    )

    cmd = captured["cmd"]
    pairs = {cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-e" and i + 1 < len(cmd)}
    assert "CYBERWAVE_USBIP_BUSID=1-1-1" in pairs
    assert "CYBERWAVE_USBIP_DETACH_STALE=0" in pairs
    assert "CYBERWAVE_USBIP_HOST=10.0.0.1" in pairs
    assert not any(p.startswith("UNRELATED") for p in pairs)
