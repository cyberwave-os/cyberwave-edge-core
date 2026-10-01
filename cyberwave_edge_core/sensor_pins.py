"""udev by-id pins for camera and audio sensors (Linux).

The installer writes the pin; edge-core only enforces it, so a pinned device
that is missing holds the twin instead of letting its driver pick another one.

Which sensors can be pinned, where their links live and which metadata fields
carry a pin is declared once in :func:`sensor_kinds`; everything else here is
generic. A raw hardware serial is not a link and is left to the driver.

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
from typing import Any, Callable, Iterable, Mapping, Optional

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


def _starts_with_dir(value: Any, directory: str) -> bool:
    return isinstance(value, str) and value.strip().startswith(directory.rstrip("/") + "/")


def resolve_sensor_pins(
    kind: str,
    metadata: Any,
    *,
    extra_sources: Optional[Mapping[str, Any]] = None,
) -> list[SensorPin]:
    """Deduplicated udev links the twin pins for *kind*.

    *extra_sources* maps a source label to a value that can override the
    metadata inside the container (e.g. the ``cameras.json`` selection), so it
    is validated too. Only full paths into the kind's link trees count there.
    """
    spec = sensor_kinds()[kind]
    meta = metadata if isinstance(metadata, dict) else {}
    pins: list[SensorPin] = []

    def add(path: Optional[str], source: str) -> None:
        if path and all(p.path != path for p in pins):
            pins.append(SensorPin(kind=kind, path=path, source=source))

    def add_path(value: Any, source: str) -> None:
        for directory in spec.link_dirs:
            if _starts_with_dir(value, directory):
                add(_link_in_tree(value, directory), source)

    for source, value in (extra_sources or {}).items():
        add_path(value, source)
    for field in spec.fields:
        value = meta.get(field.name)
        source = f"metadata.{field.name}"
        add_path(value, source)
        # A bare by-id name (not a hardware serial) resolves in the first tree.
        if field.accepts_bare_name and isinstance(value, str) and looks_like_udev_name(value):
            add(_link_in_tree(value, spec.link_dirs[0]), source)
    return pins


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
    audio_pins = [pin for pin in pins if pin.kind == KIND_AUDIO]
    if len({pin_target(pin) for pin in audio_pins}) > 1:
        raise SensorPinError(
            "conflicting audio pins: "
            + ", ".join(f"{pin.name} ({pin.source})" for pin in audio_pins)
        )
    for pin in audio_pins:
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


@dataclass(frozen=True)
class PinField:
    name: str
    # Whether a bare by-id name (not only a full path) in this field is a pin.
    accepts_bare_name: bool = False


@dataclass(frozen=True)
class SensorKind:
    name: str
    # Link trees a pin may live in; bare names resolve in the first one.
    link_dirs: tuple[str, ...]
    fields: tuple[PinField, ...]
    # Nodes with this basename prefix are passed to the container via --device.
    node_prefix: Optional[str] = None
    # Translates connected pins for the driver; raises SensorPinError if it cannot.
    launch_env: Optional[Callable[[Iterable[SensorPin], Optional[str]], dict[str, str]]] = None


def sensor_kinds() -> dict[str, SensorKind]:
    """The pinnable sensor kinds. Add an entry here to support a new one."""
    return {
        KIND_CAMERA: SensorKind(
            name=KIND_CAMERA,
            link_dirs=(V4L_BY_ID_DIR, V4L_BY_PATH_DIR),
            fields=(PinField("video_device"), PinField("serial_number", accepts_bare_name=True)),
            node_prefix="video",
        ),
        KIND_AUDIO: SensorKind(
            name=KIND_AUDIO,
            link_dirs=(ALSA_BY_ID_DIR,),
            fields=(
                PinField("audio_device", accepts_bare_name=True),
                PinField("serial_number", accepts_bare_name=True),
            ),
            launch_env=audio_launch_env,
        ),
    }
