"""Machine-dependent device utilities for the Cyberwave Edge Core.

This module provides utilities for discovering and enumerating hardware devices
on the edge machine. Currently supports Linux via v4l2-ctl.

Future: Add support for macOS (AVFoundation), Windows (DirectShow), etc.
"""

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CameraDevice:
    """Represents a discovered camera device."""

    card: str
    bus_info: str
    paths: list[str] = field(default_factory=list)
    driver: Optional[str] = None
    serial: Optional[str] = None

    @property
    def primary_path(self) -> Optional[str]:
        """The primary /dev/video* path (usually the first one)."""
        return self.paths[0] if self.paths else None

    @property
    def index(self) -> Optional[int]:
        """Extract the numeric index from the primary path (e.g. /dev/video2 -> 2)."""
        if not self.primary_path:
            return None
        match = re.search(r"/dev/video(\d+)", self.primary_path)
        return int(match.group(1)) if match else None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {
            "card": self.card,
            "bus_info": self.bus_info,
            "paths": self.paths,
            "primary_path": self.primary_path,
            "index": self.index,
            "driver": self.driver,
            "serial": self.serial,
        }


def _parse_v4l2_list_devices(output: str) -> list[CameraDevice]:
    """Parse the output of `v4l2-ctl --list-devices`.

    Example output:
        HD USB Camera: HD USB Camera (usb-0000:01:00.0-1.2):
            /dev/video0
            /dev/video1
            /dev/media0

        Logitech C920 (usb-0000:01:00.0-1.4):
            /dev/video2
            /dev/video3
            /dev/media1

    Returns a list of CameraDevice objects.
    """
    devices: list[CameraDevice] = []
    current_device: Optional[CameraDevice] = None

    for line in output.splitlines():
        line = line.rstrip()
        if not line:
            continue

        if line.startswith("\t") or line.startswith(" "):
            path = line.strip()
            if current_device and path.startswith("/dev/video"):
                current_device.paths.append(path)
        else:
            match = re.match(r"^(.+?)\s*\(([^)]+)\):\s*$", line)
            if match:
                card = match.group(1).strip()
                bus_info = match.group(2).strip()
                current_device = CameraDevice(card=card, bus_info=bus_info)
                devices.append(current_device)
            else:
                card = line.rstrip(":").strip()
                current_device = CameraDevice(card=card, bus_info="")
                devices.append(current_device)

    return [d for d in devices if d.paths]


def _get_v4l2_device_info(device_path: str) -> dict:
    """Get detailed info for a specific v4l2 device using `v4l2-ctl --device=X --all`.

    v4l2-ctl outputs "Driver name", "Card type", "Bus info" (and sometimes "Serial").
    We normalize these to driver, card, bus_info, serial.
    """
    if not shutil.which("v4l2-ctl"):
        return {}

    try:
        result = subprocess.run(
            ["v4l2-ctl", f"--device={device_path}", "--all"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}

        # v4l2-ctl uses "Driver name", "Card type", "Bus info" - normalize to our keys
        key_aliases = {
            "driver_name": "driver",
            "driver": "driver",
            "card_type": "card",
            "card": "card",
            "bus_info": "bus_info",
            "serial": "serial",
            "serial_number": "serial",
        }
        info: dict = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                raw_key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                if raw_key in key_aliases and value:
                    info[key_aliases[raw_key]] = value
        return info
    except Exception as exc:
        logger.debug("Failed to get v4l2 device info for %s: %s", device_path, exc)
        return {}


def discover_usb_cameras_v4l2() -> list[CameraDevice]:
    """Discover USB cameras using v4l2-ctl (Linux only).

    Runs `v4l2-ctl --list-devices` and parses the output to get a list of
    connected camera devices with their paths and metadata.

    Returns:
        List of CameraDevice objects, empty if v4l2-ctl is not available or fails.
    """
    if not shutil.which("v4l2-ctl"):
        logger.warning("v4l2-ctl not found; cannot enumerate cameras (install v4l-utils)")
        return []

    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 and result.stderr:
            logger.warning(
                "v4l2-ctl reported errors (e.g. unplugged device) but continuing: %s",
                result.stderr.strip(),
            )

        # Parse stdout regardless of return code - v4l2-ctl may output valid devices
        # before failing on an inaccessible one (e.g. unplugged /dev/video0)
        devices = _parse_v4l2_list_devices(result.stdout or "")

        for device in devices:
            if device.primary_path:
                info = _get_v4l2_device_info(device.primary_path)
                if info.get("driver"):
                    device.driver = info["driver"]
                if info.get("serial"):
                    device.serial = info["serial"]

        logger.info("Discovered %d USB camera(s) via v4l2-ctl", len(devices))
        return devices

    except subprocess.TimeoutExpired:
        logger.warning("v4l2-ctl timed out")
        return []
    except Exception as exc:
        logger.warning("Failed to discover USB cameras: %s", exc)
        return []


def discover_usb_cameras() -> list[CameraDevice]:
    """Discover USB cameras on the system.

    Platform-specific:
    - Linux: Uses v4l2-ctl
    - macOS: Not yet implemented (future: AVFoundation)
    - Windows: Not yet implemented (future: DirectShow)

    Returns:
        List of CameraDevice objects.
    """
    import platform

    system = platform.system()
    if system == "Linux":
        return discover_usb_cameras_v4l2()
    elif system == "Darwin":
        logger.warning("macOS camera discovery not yet implemented")
        return []
    elif system == "Windows":
        logger.warning("Windows camera discovery not yet implemented")
        return []
    else:
        logger.warning("Unsupported platform for camera discovery: %s", system)
        return []


def list_serial_ports() -> list[str]:
    """List serial ports that may be robot devices.

    Linux: /dev/ttyACM*, /dev/ttyUSB*
    macOS: /dev/tty.usbmodem*, /dev/tty.usbserial*
    Windows: COM* (not yet implemented)

    Returns:
        Sorted list of port paths.
    """
    import platform

    dev = Path("/dev")
    if not dev.exists():
        return []

    system = platform.system()
    candidates: list[str] = []

    if system == "Linux":
        for pattern in ("ttyACM*", "ttyUSB*"):
            candidates.extend(str(p) for p in dev.glob(pattern) if p.exists())
    elif system == "Darwin":
        for pattern in ("tty.usbmodem*", "tty.usbserial*"):
            candidates.extend(str(p) for p in dev.glob(pattern) if p.exists())
    else:
        logger.warning("Serial port listing not implemented for %s", system)

    return sorted(set(candidates))


def write_cameras_json(cameras: list[CameraDevice], config_dir: Path) -> Path:
    """Write discovered cameras to a JSON file in the config directory.

    Args:
        cameras: List of discovered CameraDevice objects.
        config_dir: Directory to write the cameras.json file.

    Returns:
        Path to the written cameras.json file.
    """
    import json

    cameras_file = config_dir / "cameras.json"
    data = {
        "devices": [cam.to_dict() for cam in cameras],
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(cameras_file, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote cameras.json to %s (%d devices)", cameras_file, len(cameras))
    return cameras_file


def load_cameras_json(config_dir: Path) -> list[dict]:
    """Load cameras from cameras.json in the config directory.

    Args:
        config_dir: Directory containing cameras.json.

    Returns:
        List of camera dicts, empty if file doesn't exist or is invalid.
    """
    import json

    cameras_file = config_dir / "cameras.json"
    if not cameras_file.exists():
        return []
    try:
        with open(cameras_file) as f:
            data = json.load(f)
        return data.get("devices", [])
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load cameras.json: %s", exc)
        return []
