"""Tests for ``_load_selected_camera_device``, the cameras.json -> env bridge.

This is the second half of the camera pin round trip. The CLI resolves a twin
to a camera once and records it; this function renders that record into
``CYBERWAVE_METADATA_VIDEO_DEVICE``, which the camera driver's entrypoint
treats as authoritative -- it will not export the twin's own
``metadata.video_device`` over an env var that is already set.

That makes the *kind* of value stored here load-bearing, and it had no
coverage: a mapping can look correct and still hand the driver a positional
path it is free to substitute away from.
"""

from __future__ import annotations

import json

import cyberwave_edge_core.startup as startup

TWIN = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
OTHER_TWIN = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
BY_ID = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0"


def _write_cameras(monkeypatch, tmp_path, payload: dict) -> None:
    (tmp_path / "cameras.json").write_text(json.dumps(payload))
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    # edge.json would otherwise take precedence over cameras.json.
    monkeypatch.setattr(startup, "_read_edge_json", lambda: None)


def test_index_entry_renders_as_a_dev_node(monkeypatch, tmp_path):
    """The long-standing contract for positionally-mapped twins."""
    _write_cameras(monkeypatch, tmp_path, {"twin_to_device": {TWIN: 6}})

    assert startup._load_selected_camera_device(TWIN) == "/dev/video6"


def test_stable_entry_is_passed_through_verbatim(monkeypatch, tmp_path):
    """A by-id pin must reach the driver intact.

    Rendering it as ``/dev/video{N}`` -- or dropping it and falling back to
    the global index -- downgrades a stable identifier into a positional one,
    which is exactly what the pin exists to prevent.
    """
    _write_cameras(
        monkeypatch, tmp_path, {"twin_to_device": {TWIN: BY_ID}, "selected_device": 0}
    )

    assert startup._load_selected_camera_device(TWIN) == BY_ID


def test_stream_entry_is_passed_through_verbatim(monkeypatch, tmp_path):
    """A network camera twin must not be handed a local device."""
    _write_cameras(
        monkeypatch,
        tmp_path,
        {"twin_to_device": {TWIN: "rtsp://10.0.0.2/avc"}, "selected_device": 0},
    )

    assert startup._load_selected_camera_device(TWIN) == "rtsp://10.0.0.2/avc"


def test_numeric_string_still_counts_as_an_index(monkeypatch, tmp_path):
    """JSON round trips and older writers can leave the index as a string."""
    _write_cameras(monkeypatch, tmp_path, {"twin_to_device": {TWIN: "6"}})

    assert startup._load_selected_camera_device(TWIN) == "/dev/video6"


def test_unmapped_twin_falls_back_to_the_global_device(monkeypatch, tmp_path):
    """Back-compat: the fallback is why a pinned twin must be in the mapping."""
    _write_cameras(
        monkeypatch,
        tmp_path,
        {"twin_to_device": {OTHER_TWIN: 6}, "selected_device": 0},
    )

    assert startup._load_selected_camera_device(TWIN) == "/dev/video0"


def test_blank_stable_entry_does_not_shadow_the_global_device(monkeypatch, tmp_path):
    """An empty string is not a pin; it must not resolve to itself."""
    _write_cameras(
        monkeypatch, tmp_path, {"twin_to_device": {TWIN: "   "}, "selected_device": 3}
    )

    assert startup._load_selected_camera_device(TWIN) == "/dev/video3"


def test_driver_stable_identifier_is_not_replaced_by_the_global_device(
    monkeypatch, tmp_path
):
    """Pass-through matches the driver's rule: non-index, non-/dev/ is a pin.

    The three definitions of "stable" -- here, in the CLI installer, and in the
    driver's ``_is_stable_device_identifier`` -- have to agree. When this one
    was a narrower allow-list, a value the CLI accepted as a pin was dropped
    here and replaced by the global ``selected_device``; that injected env var
    then won over the twin's own ``metadata.video_device``, so the twin opened
    a substitute camera silently. Failing loudly in the driver on an
    unresolvable pin is the point of pinning.
    """
    _write_cameras(
        monkeypatch,
        tmp_path,
        {"twin_to_device": {TWIN: "some-stable-name"}, "selected_device": 1},
    )

    assert startup._load_selected_camera_device(TWIN) == "some-stable-name"


def test_a_serial_entry_never_renders_as_a_device_node(monkeypatch, tmp_path):
    """A librealsense serial is all digits, and must not become a device.

    ``_coerce_video_index`` accepts ``"213722070420"`` happily, which rendered
    ``/dev/video213722070420`` -- a node that cannot exist. The driver then
    logged "device does not exist inside the container" on every healthy depth
    camera. The serial reaches the driver as
    ``CYBERWAVE_METADATA_SERIAL_NUMBER`` instead, so the right answer here is
    to inject no video device at all.

    Back-compat: a ``cameras.json`` written by an older CLI holds the untagged
    serial, which still coerces to an index here. That degrades to the previous
    behaviour (a spurious warning, no functional effect, since the depth path
    ignores ``camera_id``) and self-heals the next time the installer runs.
    """
    _write_cameras(
        monkeypatch,
        tmp_path,
        {"twin_to_device": {TWIN: "serial:213722070420"}, "selected_device": 0},
    )

    assert startup._load_selected_camera_device(TWIN) is None


def test_non_rtsp_url_schemes_are_recognised(monkeypatch, tmp_path):
    """The driver treats any explicit source as pinned; so must this."""
    _write_cameras(
        monkeypatch,
        tmp_path,
        {"twin_to_device": {TWIN: "udp://239.0.0.1:5000"}, "selected_device": 1},
    )

    assert startup._load_selected_camera_device(TWIN) == "udp://239.0.0.1:5000"
