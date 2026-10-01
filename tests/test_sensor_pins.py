"""Unit tests for ``sensor_pins``: which udev links a twin pins, and are they there."""

from __future__ import annotations

import os

import pytest

from cyberwave_edge_core import sensor_pins as sp

CAM_NAME = "usb-046d_HD_Pro_Webcam_C920_8E3A1B2F-video-index0"
MIC_NAME = "usb-miniDSP_UMA16v2_00026-00"


@pytest.fixture
def udev(tmp_path, monkeypatch):
    """Fixture /dev and /proc/asound trees."""
    v4l = tmp_path / "v4l" / "by-id"
    v4l_path = tmp_path / "v4l" / "by-path"
    alsa = tmp_path / "snd" / "by-id"
    asound = tmp_path / "asound"
    for d in (v4l, v4l_path, alsa, asound):
        d.mkdir(parents=True)
    (tmp_path / "video0").write_text("")
    (tmp_path / "snd" / "controlC2").write_text("")
    monkeypatch.setattr(sp, "V4L_BY_ID_DIR", str(v4l))
    monkeypatch.setattr(sp, "V4L_BY_PATH_DIR", str(v4l_path))
    monkeypatch.setattr(sp, "ALSA_BY_ID_DIR", str(alsa))
    monkeypatch.setattr(sp, "ASOUND_DIR", str(asound))
    return tmp_path


def _link(directory: str, name: str, target: str) -> str:
    path = os.path.join(directory, name)
    os.symlink(target, path)
    return path


class TestCameraPins:
    def test_video_device_by_id_path(self, udev):
        path = f"{sp.V4L_BY_ID_DIR}/{CAM_NAME}"
        pins = sp.resolve_sensor_pins(sp.KIND_CAMERA, {"video_device": path})
        assert [(p.path, p.source) for p in pins] == [(path, "metadata.video_device")]

    def test_cameras_json_value_comes_first_and_is_deduplicated(self, udev):
        path = f"{sp.V4L_BY_ID_DIR}/{CAM_NAME}"
        pins = sp.resolve_sensor_pins(
            sp.KIND_CAMERA, {"video_device": path}, extra_sources={"cameras.json": path}
        )
        assert [(p.path, p.source) for p in pins] == [(path, "cameras.json")]

    def test_two_different_sources_are_both_validated(self, udev):
        a = f"{sp.V4L_BY_ID_DIR}/usb-A-video-index0"
        b = f"{sp.V4L_BY_ID_DIR}/usb-B-video-index0"
        pins = sp.resolve_sensor_pins(
            sp.KIND_CAMERA, {"video_device": b}, extra_sources={"cameras.json": a}
        )
        assert [p.path for p in pins] == [a, b]

    def test_by_path_is_a_pin(self, udev):
        path = f"{sp.V4L_BY_PATH_DIR}/pci-0000:00:14.0-usb-0:2:1.0-video-index0"
        pins = sp.resolve_sensor_pins(sp.KIND_CAMERA, {"video_device": path})
        assert [p.path for p in pins] == [path]

    def test_udev_name_in_serial_number_is_a_pin(self, udev):
        pins = sp.resolve_sensor_pins(sp.KIND_CAMERA, {"serial_number": CAM_NAME})
        assert [(p.path, p.source) for p in pins] == [
            (f"{sp.V4L_BY_ID_DIR}/{CAM_NAME}", "metadata.serial_number")
        ]

    @pytest.mark.parametrize(
        "metadata",
        [
            {},
            {"serial_number": "926223022293"},  # raw hardware serial: driver's job
            {"serial_number": ""},
            {"video_device": "/dev/video0"},  # positional
            {"video_device": 0},
            {"video_device": "rtsp://10.0.0.2/avc"},
            {"video_device": CAM_NAME},  # bare name is not a camera source
            None,
            "not-a-dict",
        ],
    )
    def test_not_a_pin(self, udev, metadata):
        assert sp.resolve_sensor_pins(sp.KIND_CAMERA, metadata) == []

    @pytest.mark.parametrize(
        "value",
        [
            "/dev/v4l/by-id/../../etc/passwd",
            "/dev/v4l/by-id/sub/usb-x",
            "/dev/v4l/by-id/",
            "/dev/v4l/by-id/..",
        ],
    )
    def test_path_traversal_is_rejected(self, udev, value):
        value = value.replace("/dev/v4l/by-id", sp.V4L_BY_ID_DIR)
        assert sp.resolve_sensor_pins(sp.KIND_CAMERA, {"video_device": value}) == []

    def test_cameras_json_stream_value_is_ignored(self, udev):
        assert (
            sp.resolve_sensor_pins(
                sp.KIND_CAMERA, {}, extra_sources={"cameras.json": "rtsp://10.0.0.2/avc"}
            )
            == []
        )


class TestAudioPins:
    def test_audio_device_path_and_serial_collapse_to_one_pin(self, udev):
        path = f"{sp.ALSA_BY_ID_DIR}/{MIC_NAME}"
        pins = sp.resolve_sensor_pins(
            sp.KIND_AUDIO, {"audio_device": path, "serial_number": MIC_NAME}
        )
        assert [(p.path, p.source) for p in pins] == [(path, "metadata.audio_device")]

    def test_bare_name_in_audio_device(self, udev):
        pins = sp.resolve_sensor_pins(sp.KIND_AUDIO, {"audio_device": MIC_NAME})
        assert [p.path for p in pins] == [f"{sp.ALSA_BY_ID_DIR}/{MIC_NAME}"]

    def test_serial_number_only(self, udev):
        pins = sp.resolve_sensor_pins(sp.KIND_AUDIO, {"serial_number": MIC_NAME})
        assert [p.source for p in pins] == ["metadata.serial_number"]

    @pytest.mark.parametrize(
        "metadata",
        [
            {},
            {"serial_number": "00026"},
            {"audio_device": "hw:2,0"},
            {"audio_device": "http://host.docker.internal:8000/pcm"},
            {"audio_device": 3},
        ],
    )
    def test_not_a_pin(self, udev, metadata):
        assert sp.resolve_sensor_pins(sp.KIND_AUDIO, metadata) == []

    def test_camera_fields_are_not_audio_pins(self, udev):
        meta = {"video_device": f"{sp.V4L_BY_ID_DIR}/{CAM_NAME}"}
        assert sp.resolve_sensor_pins(sp.KIND_AUDIO, meta) == []


class TestPresence:
    def test_present_when_link_resolves(self, udev):
        path = _link(sp.V4L_BY_ID_DIR, CAM_NAME, str(udev / "video0"))
        assert sp.SensorPin(sp.KIND_CAMERA, path, "x").present

    def test_dangling_link_is_absent(self, udev):
        path = _link(sp.V4L_BY_ID_DIR, CAM_NAME, str(udev / "gone"))
        assert not sp.SensorPin(sp.KIND_CAMERA, path, "x").present

    def test_missing_link_is_absent(self, udev):
        pin = sp.SensorPin(sp.KIND_CAMERA, f"{sp.V4L_BY_ID_DIR}/nope", "x")
        assert not pin.present
        assert sp.missing_pins([pin]) == [pin]

    def test_name_and_description(self, udev):
        pin = sp.SensorPin(
            sp.KIND_AUDIO, f"{sp.ALSA_BY_ID_DIR}/{MIC_NAME}", "metadata.audio_device"
        )
        assert pin.name == MIC_NAME
        assert sp.describe_missing([pin]) == f"{MIC_NAME} (from metadata.audio_device)"


class TestAudioLaunchEnv:
    def _pin(self, udev, target):
        path = _link(sp.ALSA_BY_ID_DIR, MIC_NAME, target)
        return sp.SensorPin(sp.KIND_AUDIO, path, "metadata.audio_device")

    def test_resolves_card_number_and_id(self, udev):
        card = udev / "asound" / "card2"
        card.mkdir()
        (card / "id").write_text("UMA16v2\n")
        env = sp.audio_launch_env([self._pin(udev, str(udev / "snd" / "controlC2"))])
        assert env == {
            sp.AUDIO_BY_ID_ENV: f"{sp.ALSA_BY_ID_DIR}/{MIC_NAME}",
            sp.AUDIO_ALSA_CARD_ENV: "2",
            sp.AUDIO_ALSA_DEVICE_ENV: "hw:2",
            sp.AUDIO_ALSA_CARD_ID_ENV: "UMA16v2",
        }

    def test_card_id_is_optional(self, udev):
        env = sp.audio_launch_env([self._pin(udev, str(udev / "snd" / "controlC2"))])
        assert env[sp.AUDIO_ALSA_DEVICE_ENV] == "hw:2"
        assert sp.AUDIO_ALSA_CARD_ID_ENV not in env

    def test_relative_link_like_udev_makes(self, udev):
        env = sp.audio_launch_env([self._pin(udev, "../controlC2")])
        assert env[sp.AUDIO_ALSA_CARD_ENV] == "2"

    def test_target_that_is_not_a_control_node_is_an_error(self, udev):
        with pytest.raises(sp.SensorPinError, match="ALSA card"):
            sp.audio_launch_env([self._pin(udev, str(udev / "video0"))])

    def test_microphone_role_gets_the_bare_serial(self, udev):
        env = sp.audio_launch_env(
            [self._pin(udev, str(udev / "snd" / "controlC2"))], sp.ROLE_MICROPHONE
        )
        assert env[sp.SERIAL_NUMBER_ENV] == "UMA16v2_00026"
        assert sp.AUDIO_DEVICE_ENV not in env

    def test_speaker_role_gets_an_unambiguous_portaudio_selector(self, udev):
        env = sp.audio_launch_env(
            [self._pin(udev, str(udev / "snd" / "controlC2"))], sp.ROLE_SPEAKER
        )
        selector = env[sp.AUDIO_DEVICE_ENV]
        assert selector == "(hw:2,"
        assert sp.SERIAL_NUMBER_ENV not in env
        assert selector in "UMA16v2: USB Audio (hw:2,0)".lower()
        assert selector not in "Other: USB Audio (hw:20,0)".lower()

    def test_camera_pins_yield_nothing(self, udev):
        path = _link(sp.V4L_BY_ID_DIR, CAM_NAME, str(udev / "video0"))
        assert sp.audio_launch_env([sp.SensorPin(sp.KIND_CAMERA, path, "x")]) == {}


class TestAudioSerialToken:
    """The microphone driver only understands ``*_<serial>-<n>`` in by-id."""

    def _pin(self, name):
        return sp.SensorPin(sp.KIND_AUDIO, f"{sp.ALSA_BY_ID_DIR}/{name}", "x")

    def test_token_selects_exactly_the_pinned_entry(self, udev):
        _link(sp.ALSA_BY_ID_DIR, MIC_NAME, "../controlC2")
        _link(sp.ALSA_BY_ID_DIR, "usb-miniDSP_UMA16v2_00099-00", "../controlC3")
        assert sp.audio_serial_token(self._pin(MIC_NAME)) == "UMA16v2_00026"

    def test_serial_less_device_still_resolves_when_unique(self, udev):
        _link(sp.ALSA_BY_ID_DIR, "usb-Generic_USB_Audio-00", "../controlC2")
        assert sp.audio_serial_token(self._pin("usb-Generic_USB_Audio-00")) == "USB_Audio"

    def test_indistinguishable_units_are_refused(self, udev):
        _link(sp.ALSA_BY_ID_DIR, "usb-A_Generic_USB_Audio-00", "../controlC2")
        _link(sp.ALSA_BY_ID_DIR, "usb-B_Generic_USB_Audio-00", "../controlC3")
        with pytest.raises(sp.SensorPinError, match="cannot be told apart"):
            sp.audio_serial_token(self._pin("usb-A_Generic_USB_Audio-00"))

    def test_name_without_underscore_is_refused(self, udev):
        _link(sp.ALSA_BY_ID_DIR, "usb-Weird-00", "../controlC2")
        with pytest.raises(sp.SensorPinError):
            sp.audio_serial_token(self._pin("usb-Weird-00"))

    def test_missing_tree_is_refused(self, udev, monkeypatch):
        monkeypatch.setattr(sp, "ALSA_BY_ID_DIR", str(udev / "nope"))
        with pytest.raises(sp.SensorPinError, match="cannot list"):
            sp.audio_serial_token(self._pin(MIC_NAME))
