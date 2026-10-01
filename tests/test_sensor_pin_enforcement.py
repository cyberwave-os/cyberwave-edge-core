"""Pre-launch gate, runtime monitor and cameras.json drift for udev by-id pins."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import cyberwave_edge_core.startup as startup
from cyberwave_edge_core import sensor_pins as sp

FINGERPRINT = "edge-fingerprint"
_REAL_FETCH = startup.fetch_and_run_twin_drivers
TWIN = "11111111-1111-4111-8111-111111111111"
ASSET = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
CAM_NAME = "usb-046d_HD_Pro_Webcam_C920_8E3A1B2F-video-index0"
MIC_NAME = "usb-miniDSP_UMA16v2_00026-00"
CAMERA_IMAGE = "cyberwaveos/camera-driver"
MIC_IMAGE = "cyberwaveos/generic-microphone"
SPEAKER_IMAGE = "cyberwaveos/generic-speaker"


class _Twin:
    def __init__(self, metadata, name="Cam A"):
        self.uuid = TWIN
        self.name = name
        self.metadata = metadata
        self.asset_uuid = ASSET
        self.asset_id = ASSET


class _Asset:
    metadata: dict = {}
    registry_id = ""


@pytest.fixture
def udev(tmp_path, monkeypatch):
    """Fixture /dev + /proc/asound trees and a Linux edge-core."""
    root = tmp_path / "dev"
    dirs = {
        "v4l": root / "v4l" / "by-id",
        "v4l_path": root / "v4l" / "by-path",
        "alsa": root / "snd" / "by-id",
        "asound": tmp_path / "asound",
    }
    for d in dirs.values():
        d.mkdir(parents=True)
    (root / "video0").write_text("")
    (root / "snd" / "controlC3").write_text("")
    monkeypatch.setattr(sp, "V4L_BY_ID_DIR", str(dirs["v4l"]))
    monkeypatch.setattr(sp, "V4L_BY_PATH_DIR", str(dirs["v4l_path"]))
    monkeypatch.setattr(sp, "ALSA_BY_ID_DIR", str(dirs["alsa"]))
    monkeypatch.setattr(sp, "ASOUND_DIR", str(dirs["asound"]))

    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setattr(startup, "CONFIG_DIR", config)
    monkeypatch.setattr(startup, "_read_edge_json", lambda: None)
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
    startup._SENSOR_PIN_WATCH.clear()
    monkeypatch.setattr(startup, "_LAST_SENSOR_RELAUNCH_AT", None)

    class Tree:
        cam_dir = dirs["v4l"]
        mic_dir = dirs["alsa"]
        config_dir = config

        @property
        def cam_path(self):
            return f"{dirs['v4l']}/{CAM_NAME}"

        @property
        def mic_path(self):
            return f"{dirs['alsa']}/{MIC_NAME}"

        def plug_camera(self):
            os.symlink(str(root / "video0"), f"{dirs['v4l']}/{CAM_NAME}")

        def unplug_camera(self):
            os.unlink(f"{dirs['v4l']}/{CAM_NAME}")

        def plug_mic(self):
            os.symlink("../controlC3", f"{dirs['alsa']}/{MIC_NAME}")

        def unplug_mic(self):
            os.unlink(f"{dirs['alsa']}/{MIC_NAME}")

        def write_twin_json(self, metadata):
            (config / f"{TWIN}.json").write_text(json.dumps({"uuid": TWIN, "metadata": metadata}))

    yield Tree()
    startup._SENSOR_PIN_WATCH.clear()


@pytest.fixture
def alerts(monkeypatch):
    """Record alerts instead of calling the backend."""
    sent: list[dict] = []
    resolved: list[tuple[str, str]] = []

    def _send(twin_uuid, title, description, alert_type, severity="warning", metadata=None):
        sent.append(
            {
                "twin": twin_uuid,
                "type": alert_type,
                "title": title,
                "description": description,
                "severity": severity,
                "metadata": metadata,
            }
        )

    monkeypatch.setattr(startup, "_send_alert_for_twin", _send)
    monkeypatch.setattr(
        startup,
        "resolve_active_alerts_for_twin",
        lambda twin_uuid, alert_type: resolved.append((twin_uuid, alert_type)) or 1,
    )
    return SimpleNamespace(sent=sent, resolved=resolved)


class _AlertCtx:
    def __init__(self):
        self.failed: list[tuple[str, str]] = []

    def mark_failed_and_resolve(self, message, phase=None):
        self.failed.append((phase, message))

    def create(self):
        pass

    def update_metadata(self, *a, **k):
        pass


def _launch(monkeypatch, image, metadata, *, run_ok=True, driver_params=None):
    """Drive the real ``fetch_and_run_twin_drivers`` with one twin."""
    driver = {"docker_image": image}
    if driver_params:
        driver["params"] = list(driver_params)
    twin = _Twin(
        {
            "edge_fingerprint": FINGERPRINT,
            "drivers": {"default": driver},
            **metadata,
        }
    )
    client = SimpleNamespace(
        twins=SimpleNamespace(
            list=lambda environment_id: [twin],
            get_raw=lambda uuid: {"attach_to_twin_uuid": None},
        ),
        assets=SimpleNamespace(get=lambda uuid: _Asset()),
    )
    monkeypatch.setattr(startup, "Cyberwave", lambda base_url, api_key: client)
    monkeypatch.setattr(startup, "_check_and_alert_sensors_devices", lambda *a, **k: None)
    monkeypatch.setattr(
        startup, "_pull_driver_images_parallel", lambda images, **kw: {i: True for i in images}
    )
    monkeypatch.setattr(startup, "write_or_update_twin_json_file", lambda *a, **k: True)
    ctx = _AlertCtx()
    monkeypatch.setattr(startup, "DriverStartingAlertContext", lambda **kw: ctx, raising=False)

    runs: list[dict] = []
    real_fetch = _REAL_FETCH

    def _fake_run(image, params, *, twin_uuid, token, service_env=None, **kw):
        runs.append(
            {"image": image, "twin_uuid": twin_uuid, "service_env": service_env, "params": params}
        )
        return run_ok

    monkeypatch.setattr(startup, "_run_docker_image", _fake_run)
    results = real_fetch("tok", "env", FINGERPRINT)
    return SimpleNamespace(results=results, runs=runs, ctx=ctx)


class TestLaunchGateCamera:
    def test_connected_pin_starts_the_driver(self, udev, alerts, monkeypatch):
        udev.plug_camera()
        out = _launch(monkeypatch, CAMERA_IMAGE, {"video_device": udev.cam_path})
        assert len(out.runs) == 1
        assert out.results[0]["success"] is True
        assert alerts.sent == []
        assert startup._SENSOR_PIN_WATCH[TWIN].held is False

    def test_missing_pin_holds_the_driver_and_alerts(self, udev, alerts, monkeypatch):
        out = _launch(monkeypatch, CAMERA_IMAGE, {"video_device": udev.cam_path})
        assert out.runs == []  # never started on a substitute device
        assert out.results[0]["success"] is False
        assert out.results[0]["skipped_reason"] == "sensor_device_missing"
        [alert] = alerts.sent
        assert alert["type"] == "sensor_device_missing"
        assert alert["severity"] == "error"
        assert alert["twin"] == TWIN
        assert CAM_NAME in alert["description"]
        assert alert["metadata"] == {"missing_devices": [udev.cam_path]}
        assert out.ctx.failed and out.ctx.failed[0][0] == "sensor_device_missing"
        assert startup._SENSOR_PIN_WATCH[TWIN].held is True

    def test_dangling_link_counts_as_missing(self, udev, alerts, monkeypatch):
        os.symlink(str(udev.cam_dir / "gone"), udev.cam_path)
        out = _launch(monkeypatch, CAMERA_IMAGE, {"video_device": udev.cam_path})
        assert out.runs == []

    def test_cameras_json_pin_is_validated_too(self, udev, alerts, monkeypatch):
        udev.plug_camera()
        gone = f"{sp.V4L_BY_ID_DIR}/usb-OTHER-video-index0"
        (udev.config_dir / "cameras.json").write_text(json.dumps({"twin_to_device": {TWIN: gone}}))
        out = _launch(monkeypatch, CAMERA_IMAGE, {"video_device": udev.cam_path})
        assert out.runs == []
        assert "usb-OTHER-video-index0 (from cameras.json)" in alerts.sent[0]["description"]

    def test_raw_serial_is_left_to_the_driver(self, udev, alerts, monkeypatch):
        out = _launch(monkeypatch, CAMERA_IMAGE, {"serial_number": "926223022293"})
        assert len(out.runs) == 1 and alerts.sent == []
        assert TWIN not in startup._SENSOR_PIN_WATCH

    def test_unpinned_twin_is_not_gated(self, udev, alerts, monkeypatch):
        out = _launch(monkeypatch, CAMERA_IMAGE, {"video_device": "/dev/video0"})
        assert len(out.runs) == 1
        assert out.runs[0]["service_env"] is None  # untouched when nothing to add
        assert TWIN not in startup._SENSOR_PIN_WATCH

    def test_non_sensor_image_is_never_gated(self, udev, alerts, monkeypatch):
        out = _launch(monkeypatch, "cyberwaveos/so101-driver", {"video_device": udev.cam_path})
        assert len(out.runs) == 1 and alerts.sent == []

    def test_not_enforced_off_linux(self, udev, alerts, monkeypatch):
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        out = _launch(monkeypatch, CAMERA_IMAGE, {"video_device": udev.cam_path})
        assert len(out.runs) == 1 and alerts.sent == []

    def test_successful_start_clears_a_stale_alert(self, udev, alerts, monkeypatch):
        _launch(monkeypatch, CAMERA_IMAGE, {"video_device": udev.cam_path})  # held
        udev.plug_camera()
        out = _launch(monkeypatch, CAMERA_IMAGE, {"video_device": udev.cam_path})
        assert len(out.runs) == 1
        assert (TWIN, "sensor_device_missing") in alerts.resolved
        assert startup._SENSOR_PIN_WATCH[TWIN].held is False
        assert startup._SENSOR_PIN_WATCH[TWIN].alerted is False

    def test_unlinked_twin_is_dropped_from_the_watch(self, udev, alerts, monkeypatch):
        startup._SENSOR_PIN_WATCH["gone-twin"] = startup._SensorPinWatch("camera", "old")
        udev.plug_camera()
        _launch(monkeypatch, CAMERA_IMAGE, {"video_device": udev.cam_path})
        assert set(startup._SENSOR_PIN_WATCH) == {TWIN}


class TestLaunchGateAudio:
    def _asound(self, udev):
        card = udev.config_dir.parent / "asound" / "card3"
        card.mkdir()
        (card / "id").write_text("UMA16v2\n")

    def test_microphone_gets_the_bare_serial_the_driver_resolves(self, udev, alerts, monkeypatch):
        udev.plug_mic()
        self._asound(udev)
        out = _launch(
            monkeypatch, MIC_IMAGE, {"audio_device": udev.mic_path, "serial_number": MIC_NAME}
        )
        assert len(out.runs) == 1
        env = out.runs[0]["service_env"]
        assert env[sp.SERIAL_NUMBER_ENV] == "UMA16v2_00026"
        assert env[sp.AUDIO_ALSA_DEVICE_ENV] == "hw:3"
        assert env[sp.AUDIO_ALSA_CARD_ID_ENV] == "UMA16v2"
        assert sp.AUDIO_DEVICE_ENV not in env

    def test_speaker_gets_a_portaudio_selector_for_its_card(self, udev, alerts, monkeypatch):
        udev.plug_mic()
        out = _launch(monkeypatch, SPEAKER_IMAGE, {"audio_device": udev.mic_path})
        env = out.runs[0]["service_env"]
        assert env[sp.AUDIO_DEVICE_ENV] == "(hw:3,"
        assert sp.SERIAL_NUMBER_ENV not in env

    def test_explicit_params_env_still_wins(self, udev, alerts, monkeypatch):
        udev.plug_mic()
        out = _launch(
            monkeypatch,
            SPEAKER_IMAGE,
            {"audio_device": udev.mic_path},
            driver_params=["-e", "CYBERWAVE_METADATA_AUDIO_DEVICE=2"],
        )
        assert sp.AUDIO_DEVICE_ENV not in out.runs[0]["service_env"]

    def test_microphone_that_the_driver_cannot_tell_apart_is_held(self, udev, alerts, monkeypatch):
        udev.plug_mic()
        os.symlink("../controlC4", f"{udev.mic_dir}/usb-other_UMA16v2_00026-00")
        os.symlink("../controlC5", f"{udev.mic_dir}/usb-third_UMA16v2_00026-00")
        out = _launch(monkeypatch, MIC_IMAGE, {"audio_device": udev.mic_path})
        assert out.runs == []
        assert "cannot be told apart" in alerts.sent[0]["description"]
        assert startup._SENSOR_PIN_WATCH[TWIN].held is True

    def test_missing_mic_holds_the_driver(self, udev, alerts, monkeypatch):
        out = _launch(monkeypatch, MIC_IMAGE, {"audio_device": udev.mic_path}, run_ok=True)
        assert out.runs == []
        assert alerts.sent[0]["type"] == "sensor_device_missing"
        assert MIC_NAME in alerts.sent[0]["description"]

    def test_speaker_is_gated_like_a_microphone(self, udev, alerts, monkeypatch):
        out = _launch(monkeypatch, SPEAKER_IMAGE, {"audio_device": udev.mic_path})
        assert out.runs == []

    def test_camera_pin_does_not_gate_an_audio_driver(self, udev, alerts, monkeypatch):
        out = _launch(monkeypatch, MIC_IMAGE, {"video_device": udev.cam_path})
        assert len(out.runs) == 1


class TestCameraNodePassthrough:
    """The by-id link resolves inside the container only if its node is passed."""

    def test_pinned_node_is_passed_through(self, udev, alerts, monkeypatch):
        udev.plug_camera()
        node = os.path.realpath(udev.cam_path)
        out = _launch(
            monkeypatch,
            CAMERA_IMAGE,
            {"video_device": udev.cam_path},
            driver_params=["--device", "/dev/video0:/dev/video0"],
        )
        params = out.runs[0]["params"]
        assert params[-2:] == ["--device", f"{node}:{node}"]
        assert params[:2] == ["--device", "/dev/video0:/dev/video0"]

    def test_not_duplicated_when_the_asset_already_passes_it(self, udev, alerts, monkeypatch):
        udev.plug_camera()
        node = os.path.realpath(udev.cam_path)
        out = _launch(
            monkeypatch,
            CAMERA_IMAGE,
            {"video_device": udev.cam_path},
            driver_params=[f"--device={node}"],
        )
        assert out.runs[0]["params"] == [f"--device={node}"]

    def test_unpinned_camera_params_are_untouched(self, udev, alerts, monkeypatch):
        out = _launch(
            monkeypatch,
            CAMERA_IMAGE,
            {"video_device": "/dev/video0"},
            driver_params=["--device", "/dev/video0"],
        )
        assert out.runs[0]["params"] == ["--device", "/dev/video0"]


class TestRuntimeMonitor:
    @pytest.fixture
    def relaunches(self, monkeypatch):
        calls: list[tuple] = []
        monkeypatch.setattr(
            startup, "fetch_and_run_twin_drivers", lambda *a, **k: calls.append(a) or []
        )
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_saved_fingerprint", lambda: FINGERPRINT)
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env")
        return calls

    def _watch(self, udev, kind="camera", **kw):
        startup._SENSOR_PIN_WATCH[TWIN] = startup._SensorPinWatch(kind, "Cam A", **kw)

    def test_nothing_watched_is_a_noop(self, udev, alerts, relaunches):
        assert startup.reconcile_sensor_pins()["watched"] == 0
        assert alerts.sent == [] and relaunches == []

    def test_unplug_alerts_once_per_outage(self, udev, alerts, relaunches, caplog):
        udev.plug_camera()
        udev.write_twin_json({"video_device": udev.cam_path})
        self._watch(udev)
        assert startup.reconcile_sensor_pins()["missing"] == 0
        udev.unplug_camera()
        with caplog.at_level("ERROR"):
            first = startup.reconcile_sensor_pins()
        second = startup.reconcile_sensor_pins()
        assert first["alerted"] == 1 and second["alerted"] == 0
        assert second["missing"] == 1
        assert len(alerts.sent) == 1
        assert CAM_NAME in caplog.text and "Cam A" in caplog.text
        assert relaunches == []  # nothing to relaunch while it is still gone

    def test_replug_resolves_alert_and_relaunches(self, udev, alerts, relaunches):
        udev.write_twin_json({"video_device": udev.cam_path})
        self._watch(udev)
        startup.reconcile_sensor_pins()  # missing -> alert
        udev.plug_camera()
        summary = startup.reconcile_sensor_pins()
        assert summary["recovered"] == 1 and summary["relaunch_attempted"] == 1
        assert (TWIN, "sensor_device_missing") in alerts.resolved
        assert relaunches == [("tok", "env", FINGERPRINT)]
        assert startup._SENSOR_PIN_WATCH[TWIN].alerted is False

    def test_second_outage_alerts_again(self, udev, alerts, relaunches):
        udev.plug_camera()
        udev.write_twin_json({"video_device": udev.cam_path})
        self._watch(udev)
        udev.unplug_camera()
        startup.reconcile_sensor_pins()
        udev.plug_camera()
        startup.reconcile_sensor_pins()
        udev.unplug_camera()
        startup.reconcile_sensor_pins()
        assert len(alerts.sent) == 2

    def test_held_twin_launches_when_device_appears(self, udev, alerts, relaunches):
        udev.write_twin_json({"audio_device": udev.mic_path})
        self._watch(udev, "audio", held=True, alerted=True)
        udev.plug_mic()
        summary = startup.reconcile_sensor_pins()
        assert summary["relaunch_attempted"] == 1
        assert len(relaunches) == 1

    def test_held_twin_whose_pin_was_removed_is_relaunched(self, udev, alerts, relaunches):
        udev.write_twin_json({"video_device": "/dev/video0"})
        self._watch(udev, held=True, alerted=True)
        startup.reconcile_sensor_pins()
        assert len(relaunches) == 1

    def test_relaunch_is_debounced(self, udev, alerts, relaunches):
        udev.write_twin_json({"video_device": udev.cam_path})
        self._watch(udev, held=True, alerted=True)
        udev.plug_camera()
        startup.reconcile_sensor_pins()
        startup._SENSOR_PIN_WATCH[TWIN].held = True  # still held after a failed attempt
        again = startup.reconcile_sensor_pins()
        assert again["relaunch_attempted"] == 0
        assert len(relaunches) == 1

    def test_camera_that_changed_node_is_recreated(self, udev, alerts, relaunches):
        udev.plug_camera()
        udev.write_twin_json({"video_device": udev.cam_path})
        startup._SENSOR_PIN_WATCH[TWIN] = startup._SensorPinWatch(
            "camera", "Cam A", targets={udev.cam_path: "/dev/video7"}
        )
        summary = startup.reconcile_sensor_pins()
        assert summary["relaunch_attempted"] == 1
        assert alerts.sent == []  # connected, so not an outage

    def test_audio_card_renumbering_is_recreated(self, udev, alerts, relaunches):
        udev.plug_mic()
        udev.write_twin_json({"audio_device": udev.mic_path})
        startup._SENSOR_PIN_WATCH[TWIN] = startup._SensorPinWatch(
            "audio", "Mic", targets={udev.mic_path: "/dev/snd/controlC9"}
        )
        assert startup.reconcile_sensor_pins()["relaunch_attempted"] == 1

    def test_unchanged_node_is_left_alone(self, udev, alerts, relaunches):
        udev.plug_camera()
        udev.write_twin_json({"video_device": udev.cam_path})
        startup._SENSOR_PIN_WATCH[TWIN] = startup._SensorPinWatch(
            "camera", "Cam A", targets={udev.cam_path: os.path.realpath(udev.cam_path)}
        )
        assert startup.reconcile_sensor_pins()["relaunch_attempted"] == 0

    def test_launch_records_the_node_it_gave_the_driver(self, udev, alerts, monkeypatch):
        udev.plug_camera()
        _launch(monkeypatch, CAMERA_IMAGE, {"video_device": udev.cam_path})
        assert startup._SENSOR_PIN_WATCH[TWIN].targets == {
            udev.cam_path: os.path.realpath(udev.cam_path)
        }

    def test_unreadable_twin_json_is_not_treated_as_recovery(self, udev, alerts, relaunches):
        self._watch(udev, held=True, alerted=True)  # no twin json on disk
        startup.reconcile_sensor_pins()
        assert relaunches == [] and alerts.resolved == []
        assert startup._SENSOR_PIN_WATCH[TWIN].alerted is True

    def test_repin_is_picked_up_from_the_synced_twin_json(self, udev, alerts, relaunches):
        udev.plug_camera()
        udev.write_twin_json({"video_device": udev.cam_path})
        self._watch(udev)
        startup.reconcile_sensor_pins()
        udev.write_twin_json({"video_device": f"{sp.V4L_BY_ID_DIR}/usb-NEW-video-index0"})
        startup.reconcile_sensor_pins()
        assert "usb-NEW-video-index0" in alerts.sent[0]["description"]

    def test_cameras_json_repin_is_picked_up(self, udev, alerts, relaunches):
        udev.plug_camera()
        udev.write_twin_json({"video_device": udev.cam_path})
        self._watch(udev)
        (udev.config_dir / "cameras.json").write_text(
            json.dumps({"twin_to_device": {TWIN: f"{sp.V4L_BY_ID_DIR}/usb-NEW-video-index0"}})
        )
        startup.reconcile_sensor_pins()
        assert alerts.sent and "cameras.json" in alerts.sent[0]["description"]

    def test_no_credentials_means_no_relaunch(self, udev, alerts, relaunches, monkeypatch):
        monkeypatch.setattr(startup, "load_token", lambda: None)
        udev.write_twin_json({"video_device": udev.cam_path})
        self._watch(udev, held=True, alerted=True)
        udev.plug_camera()
        assert startup.reconcile_sensor_pins()["relaunch_attempted"] == 0
        assert relaunches == []

    def test_relaunch_failure_does_not_escape(self, udev, alerts, relaunches, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("docker down")

        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", boom)
        udev.write_twin_json({"video_device": udev.cam_path})
        self._watch(udev, held=True, alerted=True)
        udev.plug_camera()
        assert startup.reconcile_sensor_pins()["relaunch_attempted"] == 1

    def test_alert_send_failure_does_not_escape(self, udev, relaunches, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("api down")

        monkeypatch.setattr(startup, "_send_alert_for_twin", boom)
        udev.write_twin_json({"video_device": udev.cam_path})
        self._watch(udev)
        startup.reconcile_sensor_pins()  # must not raise

    def test_not_run_off_linux(self, udev, alerts, relaunches, monkeypatch):
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        udev.write_twin_json({"video_device": udev.cam_path})
        self._watch(udev)
        assert startup.reconcile_sensor_pins()["watched"] == 0
        assert alerts.sent == []


class TestHeldThenRecoveredEndToEnd:
    def test_hold_then_replug_then_relaunch_starts_the_driver(self, udev, alerts, monkeypatch):
        meta = {"video_device": udev.cam_path}
        out = _launch(monkeypatch, CAMERA_IMAGE, meta)
        assert out.runs == []
        udev.write_twin_json(meta)

        udev.plug_camera()
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_saved_fingerprint", lambda: FINGERPRINT)
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env")
        relaunch_out = {}

        def _relaunch(*a, **k):
            relaunch_out["o"] = _launch(monkeypatch, CAMERA_IMAGE, meta)
            return []

        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", _relaunch)
        startup.reconcile_sensor_pins()
        assert len(relaunch_out["o"].runs) == 1
        assert startup._SENSOR_PIN_WATCH[TWIN].held is False
        assert (TWIN, "sensor_device_missing") in alerts.resolved


class TestCameraDriftWithoutGlobalDevice:
    """A per-twin change restarts drivers even without a global ``selected_device``."""

    def _setup(self, udev, monkeypatch, container_device):
        cams = udev.config_dir / "cameras.json"
        cams.write_text(json.dumps({"twin_to_device": {TWIN: udev.cam_path}}))
        monkeypatch.setattr(startup, "_cameras_json_mtime", cams.stat().st_mtime - 10)
        monkeypatch.setattr(startup, "_edge_json_mtime", None)
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
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        calls = []
        monkeypatch.setattr(startup, "_start_camera_config_drift_restart", calls.append)
        return calls

    def test_repinned_twin_triggers_restart(self, udev, monkeypatch):
        calls = self._setup(udev, monkeypatch, "/dev/v4l/by-id/usb-OLD-video-index0")
        assert startup.reconcile_camera_config_drift() is True
        assert calls == ["tok"]

    def test_matching_container_does_not(self, udev, monkeypatch):
        calls = self._setup(udev, monkeypatch, udev.cam_path)
        assert startup.reconcile_camera_config_drift() is False
        assert calls == []

    def test_container_of_an_unmapped_twin_is_left_alone(self, udev, monkeypatch):
        calls = self._setup(udev, monkeypatch, "/dev/video0")
        (udev.config_dir / "cameras.json").write_text(json.dumps({"twin_to_device": {"other": 1}}))
        monkeypatch.setattr(startup, "_cameras_json_mtime", 0.0)
        assert startup.reconcile_camera_config_drift() is False
        assert calls == []
