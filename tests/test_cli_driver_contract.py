"""Contract between the CLI pins, edge-core and the real microphone/speaker drivers.

Loads the actual CLI and driver modules from sibling checkouts; skips when absent.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

import cyberwave_edge_core.startup as startup
from cyberwave_edge_core import sensor_pins as sp

TWIN = "11111111-1111-4111-8111-111111111111"
CAM = "usb-046d_HD_Pro_Webcam_C920_8E3A1B2F-video-index0"
MIC = "usb-miniDSP_UMA16v2_00026-00"

_DRIVERS = (
    Path.home()
    / "Documents/monorepos/4-fourth/cyberwave/cyberwave-edge-runtime"
    / "runtime-services/drivers/native/cyberwave"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture
def udev(tmp_path, monkeypatch):
    root = tmp_path / "dev"
    v4l = root / "v4l" / "by-id"
    alsa = root / "snd" / "by-id"
    for d in (v4l, alsa, tmp_path / "asound" / "card3", tmp_path / "config"):
        d.mkdir(parents=True)
    (root / "video2").write_text("")
    (root / "snd" / "controlC3").write_text("")
    (tmp_path / "asound" / "card3" / "id").write_text("UMA16v2\n")
    os.symlink("../../video2", v4l / CAM)
    os.symlink("../controlC3", alsa / MIC)
    monkeypatch.setattr(sp, "V4L_BY_ID_DIR", str(v4l))
    monkeypatch.setattr(sp, "ALSA_BY_ID_DIR", str(alsa))
    monkeypatch.setattr(sp, "V4L_BY_PATH_DIR", str(root / "v4l" / "by-path"))
    monkeypatch.setattr(sp, "ASOUND_DIR", str(tmp_path / "asound"))
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(startup, "_read_edge_json", lambda: None)
    return tmp_path


class TestCliToEdgeCore:
    @pytest.fixture
    def cli(self, udev, monkeypatch):
        core = pytest.importorskip("cyberwave_cli.core")
        monkeypatch.setattr(core, "_V4L_BY_ID_DIR", sp.V4L_BY_ID_DIR)
        monkeypatch.setattr(core, "_ALSA_BY_ID_DIR", sp.ALSA_BY_ID_DIR)
        return core

    def test_camera_pin_round_trips_through_twin_and_cameras_json(self, cli, udev):
        from cyberwave_cli.device_utils import write_cameras_json

        metadata = cli._camera_pin_metadata(CAM)
        write_cameras_json(
            [],
            udev / "config",
            selected_index=None,
            twin_to_device={TWIN: f"{cli._V4L_BY_ID_DIR}/{CAM}"},
        )
        selected = startup._load_selected_camera_device(TWIN)
        assert selected == f"{sp.V4L_BY_ID_DIR}/{CAM}"

        pins = sp.resolve_sensor_pins(sp.KIND_CAMERA, metadata, selected_camera_device=selected)
        assert [p.path for p in pins] == [f"{sp.V4L_BY_ID_DIR}/{CAM}"]
        assert pins[0].present
        assert metadata["serial_number"] == ""  # blanked, must not be read as a pin

    def test_audio_pin_round_trips_and_is_one_pin(self, cli, udev):
        metadata = cli._audio_pin_metadata(MIC)
        pins = sp.resolve_sensor_pins(sp.KIND_AUDIO, metadata)
        assert [p.path for p in pins] == [f"{sp.ALSA_BY_ID_DIR}/{MIC}"]
        assert pins[0].present

    def test_every_device_the_cli_can_offer_is_a_valid_pin(self, cli, udev):
        cams = cli._list_v4l_by_id()
        mics = cli._list_alsa_by_id()
        assert [d.name for d in cams] == [CAM] and [d.name for d in mics] == [MIC]
        for dev in cams:
            [pin] = sp.resolve_sensor_pins(sp.KIND_CAMERA, cli._camera_pin_metadata(dev.name))
            assert pin.present
        for dev in mics:
            [pin] = sp.resolve_sensor_pins(sp.KIND_AUDIO, cli._audio_pin_metadata(dev.name))
            assert pin.present

    def test_legacy_raw_serial_pin_is_not_enforced_by_edge_core(self, cli):
        metadata = cli._camera_pin_metadata("926223022293")
        assert metadata == {"serial_number": "926223022293"}
        assert sp.resolve_sensor_pins(sp.KIND_CAMERA, metadata) == []

    def test_cleared_pin_is_no_pin(self, cli):
        cleared = {"serial_number": "", "video_device": ""}
        assert sp.resolve_sensor_pins(sp.KIND_CAMERA, cleared) == []
        assert sp.resolve_sensor_pins(sp.KIND_AUDIO, cleared) == []


MIC_CONFIG = _DRIVERS / "generic-microphone/src/audio_config.py"
SPEAKER_CONFIG = _DRIVERS / "generic-speaker/src/audio_config.py"


@pytest.mark.skipif(not MIC_CONFIG.exists(), reason="generic-microphone driver not checked out")
class TestMicrophoneDriver:
    @pytest.fixture
    def driver(self, udev, monkeypatch):
        module = _load(MIC_CONFIG, "mic_audio_config")
        monkeypatch.setattr(module, "_SND_BY_ID_DIR", sp.ALSA_BY_ID_DIR)
        return module

    def _pin(self):
        return sp.SensorPin(sp.KIND_AUDIO, f"{sp.ALSA_BY_ID_DIR}/{MIC}", "x")

    def test_the_cli_value_alone_is_not_understood_by_the_driver(self, driver):
        """The stored by-id name is not a bare serial, hence the translation."""
        with pytest.raises(RuntimeError, match="No USB audio device"):
            driver.resolve_device_selector(MIC, None)

    def test_the_injected_serial_resolves_to_the_pinned_card(self, driver):
        env = sp.audio_launch_env([self._pin()], sp.ROLE_MICROPHONE)
        assert driver.resolve_device_selector(env[sp.SERIAL_NUMBER_ENV], None) == "hw:3"
        assert env[sp.AUDIO_ALSA_DEVICE_ENV] == "hw:3"

    def test_selected_card_follows_a_replug_to_another_index(self, driver, udev):
        os.unlink(f"{sp.ALSA_BY_ID_DIR}/{MIC}")
        (udev / "dev" / "snd" / "controlC5").write_text("")
        os.symlink("../controlC5", f"{sp.ALSA_BY_ID_DIR}/{MIC}")
        env = sp.audio_launch_env([self._pin()], sp.ROLE_MICROPHONE)
        assert driver.resolve_device_selector(env[sp.SERIAL_NUMBER_ENV], None) == "hw:5"

    def test_precedence_in_the_driver_is_serial_over_audio_device(self, driver):
        env = sp.audio_launch_env([self._pin()], sp.ROLE_MICROPHONE)
        assert driver.resolve_device_selector(env[sp.SERIAL_NUMBER_ENV], "2") == "hw:3"


@pytest.mark.skipif(not SPEAKER_CONFIG.exists(), reason="generic-speaker driver not checked out")
class TestSpeakerDriver:
    DEVICES = [
        {"index": 0, "name": "Built-in Output: HDA Intel PCH (hw:0,0)"},
        {"index": 1, "name": "Other USB Audio: USB Audio (hw:30,0)"},
        {"index": 2, "name": "UMA16v2: USB Audio (hw:3,0)"},
        {"index": 3, "name": "Another: USB Audio (hw:13,0)"},
    ]

    def test_selector_opens_the_pinned_card_and_nothing_else(self, udev):
        driver = _load(SPEAKER_CONFIG, "speaker_audio_config")
        pin = sp.SensorPin(sp.KIND_AUDIO, f"{sp.ALSA_BY_ID_DIR}/{MIC}", "x")
        env = sp.audio_launch_env([pin], sp.ROLE_SPEAKER)
        selector = driver.parse_audio_device_selector(env[sp.AUDIO_DEVICE_ENV])
        index = driver.resolve_output_device_index(selector, self.DEVICES, default_index=0)
        assert index == 2  # not the default (0), not hw:30 or hw:13

    def test_the_raw_by_id_path_would_silently_fall_back_to_the_default(self):
        """A by-id path matches no PortAudio name and silently selects the default."""
        driver = _load(SPEAKER_CONFIG, "speaker_audio_config")
        selector = driver.parse_audio_device_selector(f"/dev/snd/by-id/{MIC}")
        assert driver.resolve_output_device_index(selector, self.DEVICES, default_index=0) == 0
