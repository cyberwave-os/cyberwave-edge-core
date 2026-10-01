"""udev by-id pins for camera and audio sensors (Linux).

The installer writes the pin; edge-core only enforces it, so a pinned device
that is missing holds the twin instead of letting its driver pick another one.

Pin fields written by the CLI: cameras use ``video_device`` (and cameras.json
``twin_to_device``), audio uses ``audio_device`` and ``serial_number`` with the
by-id name. A raw hardware serial is not a link and is left to the driver.

The audio drivers do not understand a by-id name, so it is translated:

* microphone: ``serial_number`` is a bare serial matched as ``*_<serial>-<n>``;
  the bare form is injected as ``CYBERWAVE_METADATA_SERIAL_NUMBER``.
* speaker: ``CYBERWAVE_METADATA_AUDIO_DEVICE`` is a PortAudio name substring
  that silently falls back to the default output on a miss; ``(hw:<card>,`` is
  injected instead.

The ALSA card number changes on replug, so it is resolved at launch only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

KIND_CAMERA = "camera"
KIND_AUDIO = "audio"

V4L_BY_ID_DIR = "/dev/v4l/by-id"
V4L_BY_PATH_DIR = "/dev/v4l/by-path"
ALSA_BY_ID_DIR = "/dev/snd/by-id"
ASOUND_DIR = "/proc/asound"

_UDEV_NAME_PREFIXES = ("usb-", "pci-", "platform-")
_CONTROL_NODE_RE = re.compile(r"^controlC(\d+)$")

AUDIO_BY_ID_ENV = "CYBERWAVE_EDGE_AUDIO_BY_ID"
AUDIO_ALSA_CARD_ENV = "CYBERWAVE_EDGE_AUDIO_ALSA_CARD"
AUDIO_ALSA_DEVICE_ENV = "CYBERWAVE_EDGE_AUDIO_ALSA_DEVICE"
AUDIO_ALSA_CARD_ID_ENV = "CYBERWAVE_EDGE_AUDIO_ALSA_CARD_ID"
SERIAL_NUMBER_ENV = "CYBERWAVE_METADATA_SERIAL_NUMBER"
AUDIO_DEVICE_ENV = "CYBERWAVE_METADATA_AUDIO_DEVICE"

ROLE_MICROPHONE = "microphone"
ROLE_SPEAKER = "speaker"

_INTERFACE_SUFFIX_RE = re.compile(r"-\d+$")


class SensorPinError(Exception):
    """A pin that is present but cannot be handed to the driver unambiguously."""


@dataclass(frozen=True)
class SensorPin:
    """One udev link a twin is pinned to."""

    kind: str
    path: str
    source: str

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def present(self) -> bool:
        """True when the link exists *and* resolves (a dangling link is absent)."""
        return os.path.exists(self.path)


def looks_like_udev_name(value: str) -> bool:
    """True when *value* is a udev stable name rather than a hardware serial."""
    return value.strip().startswith(_UDEV_NAME_PREFIXES)


def _link_in_tree(value: Any, directory: str) -> Optional[str]:
    """Link path for a bare name or full path directly under *directory*, else None."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    prefix = directory.rstrip("/") + "/"
    if candidate.startswith(prefix):
        candidate = candidate[len(prefix) :]
    if not candidate or "/" in candidate or candidate in {".", ".."}:
        return None
    return prefix + candidate


def _metadata_dict(metadata: Any) -> dict[str, Any]:
    return metadata if isinstance(metadata, dict) else {}


def resolve_sensor_pins(
    kind: str,
    metadata: Any,
    *,
    selected_camera_device: Optional[str] = None,
) -> list[SensorPin]:
    """Deduplicated udev links the twin pins for *kind*.

    *selected_camera_device* (from ``cameras.json``) outranks ``video_device``
    inside the container, so it is validated too.
    """
    meta = _metadata_dict(metadata)
    pins: list[SensorPin] = []

    def add(path: Optional[str], source: str) -> None:
        if path and all(p.path != path for p in pins):
            pins.append(SensorPin(kind=kind, path=path, source=source))

    if kind == KIND_CAMERA:
        for value, source in (
            (selected_camera_device, "cameras.json"),
            (meta.get("video_device"), "metadata.video_device"),
        ):
            # Only full paths: indexes, /dev/videoN and URLs have no link to check.
            for directory in (V4L_BY_ID_DIR, V4L_BY_PATH_DIR):
                if _starts_with_dir(value, directory):
                    add(_link_in_tree(value, directory), source)
        serial = meta.get("serial_number")
        if isinstance(serial, str) and looks_like_udev_name(serial):
            add(_link_in_tree(serial, V4L_BY_ID_DIR), "metadata.serial_number")
    elif kind == KIND_AUDIO:
        audio_device = meta.get("audio_device")
        if isinstance(audio_device, str) and (
            _starts_with_dir(audio_device, ALSA_BY_ID_DIR) or looks_like_udev_name(audio_device)
        ):
            add(_link_in_tree(audio_device, ALSA_BY_ID_DIR), "metadata.audio_device")
        serial = meta.get("serial_number")
        if isinstance(serial, str) and looks_like_udev_name(serial):
            add(_link_in_tree(serial, ALSA_BY_ID_DIR), "metadata.serial_number")
    return pins


def _starts_with_dir(value: Any, directory: str) -> bool:
    return isinstance(value, str) and value.strip().startswith(directory.rstrip("/") + "/")


def missing_pins(pins: Iterable[SensorPin]) -> list[SensorPin]:
    return [pin for pin in pins if not pin.present]


def describe_missing(pins: Iterable[SensorPin]) -> str:
    return ", ".join(f"{pin.name} (from {pin.source})" for pin in pins)


def audio_card_index(pin: SensorPin) -> Optional[int]:
    """ALSA card number behind an audio by-id link (``controlC<N>``)."""
    match = _CONTROL_NODE_RE.match(os.path.basename(os.path.realpath(pin.path)))
    return int(match.group(1)) if match else None


def pin_target(pin: SensorPin) -> str:
    """Where the link resolves right now (``/dev/video2``, ``/dev/snd/controlC3``)."""
    return os.path.realpath(pin.path)


def audio_serial_token(pin: SensorPin) -> str:
    """Bare serial the microphone driver resolves back to *pin* and to nothing else.

    ``usb-miniDSP_UMA16v2_00026-00`` -> ``UMA16v2_00026``. Raises when another
    entry matches too, since the driver could then open a different unit.
    """
    name = _INTERFACE_SUFFIX_RE.sub("", pin.name)
    if "_" not in name:
        raise SensorPinError(f"{pin.name} has no serial-like part the driver can match")
    token = name.split("_", 1)[1]
    pattern = re.compile(rf"_{re.escape(token)}-\d+$")
    try:
        entries = sorted(os.listdir(ALSA_BY_ID_DIR))
    except OSError as exc:
        raise SensorPinError(f"cannot list {ALSA_BY_ID_DIR}: {exc}") from exc
    matches = [entry for entry in entries if pattern.search(entry)]
    if matches != [pin.name]:
        raise SensorPinError(
            f"{pin.name} cannot be told apart from {', '.join(matches) or 'nothing'} "
            "by the driver's serial lookup; pin a unit with a distinct serial"
        )
    return token


def audio_launch_env(pins: Iterable[SensorPin], role: Optional[str] = None) -> dict[str, str]:
    """Env that points an audio driver at its pinned card (selector depends on *role*)."""
    for pin in pins:
        if pin.kind != KIND_AUDIO:
            continue
        index = audio_card_index(pin)
        if index is None:
            raise SensorPinError(f"{pin.name} does not resolve to an ALSA card ({pin_target(pin)})")
        env = {
            AUDIO_BY_ID_ENV: pin.path,
            AUDIO_ALSA_CARD_ENV: str(index),
            AUDIO_ALSA_DEVICE_ENV: f"hw:{index}",
        }
        try:
            with open(os.path.join(ASOUND_DIR, f"card{index}", "id"), encoding="utf-8") as fh:
                card_id = fh.read().strip()
        except OSError:
            card_id = ""
        if card_id:
            env[AUDIO_ALSA_CARD_ID_ENV] = card_id
        if role == ROLE_MICROPHONE:
            env[SERIAL_NUMBER_ENV] = audio_serial_token(pin)
        elif role == ROLE_SPEAKER:
            # The comma keeps hw:1 from matching hw:10.
            env[AUDIO_DEVICE_ENV] = f"(hw:{index},"
        return env
    return {}
