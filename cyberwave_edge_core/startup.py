"""Boot-time startup checks for the Cyberwave Edge Core.

On every boot the edge core must:
  1. Read the API token from ``~/.cyberwave/credentials.json``
  2. Validate the token against the Cyberwave REST API
  3. Verify that it can connect to the MQTT broker
  4. Check whether an environment is linked via ``~/.cyberwave/environment.json``

The config directory defaults to ``~/.cyberwave`` on all platforms
(under the invoking user's home, even when running via ``sudo``).
It can be overridden with the ``CYBERWAVE_EDGE_CONFIG_DIR`` environment variable.
Legacy installs using ``/etc/cyberwave`` are migrated automatically.

This module exposes each check individually (for the ``status`` command)
and a single ``run_startup_checks()`` orchestrator for the boot path.
"""

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from cyberwave import Cyberwave
from cyberwave.edge.platform import is_usbip_server_running as _is_usbip_server_running
from cyberwave.fingerprint import generate_fingerprint
from rich.console import Console

from . import __version__ as PACKAGE_EDGE_CORE_VERSION
from .utils import (
    EDGE_CORE_RESTART_PHASE_COMPLETED,
    EDGE_CORE_RESTART_PHASE_FAILED,
    EDGE_CORE_RESTART_PHASE_IN_PROGRESS,
    DriverStartingAlertContext,
    EdgeCoreRestartAlertContext,
)
from .zenoh_config import (
    ZenohConfig,
    build_zenoh_env_vars,
    log_zenoh_diagnostics,
    start_zenoh_router,
    stop_zenoh_router,
)


def _resolve_sudo_user_home() -> Optional[Path]:
    """Return invoking user's home when running via sudo (best effort)."""
    sudo_user = os.getenv("SUDO_USER", "").strip()
    if not sudo_user:
        return None

    try:
        import pwd

        home = pwd.getpwnam(sudo_user).pw_dir
    except Exception:
        return None
    if not home:
        return None
    return Path(home)


def _resolve_default_config_dir() -> Path:
    """Return default edge config directory for this platform.

    All platforms now resolve to ``~/.cyberwave`` (under the invoking
    user's home, even when running via ``sudo``).  The legacy Linux
    path ``/etc/cyberwave`` is handled by migration.
    """
    sudo_home = _resolve_sudo_user_home()
    base_home = sudo_home or Path.home()
    return base_home / ".cyberwave"


def _resolve_config_dir() -> Path:
    """Resolve config dir honoring explicit environment override first."""
    override = os.getenv("CYBERWAVE_EDGE_CONFIG_DIR", "").strip()
    if override:
        return Path(override)
    return _resolve_default_config_dir()


_LEGACY_SYSTEM_CONFIG_DIR = Path("/etc/cyberwave")


def _migrate_legacy_config(config_dir: Path) -> None:
    """Best-effort migration from legacy /etc/cyberwave to user config dir.

    Older versions stored config under ``/etc/cyberwave`` on Linux (and
    briefly on macOS).  Now all platforms use ``~/.cyberwave``.  This
    copies JSON files from the legacy path so users don't lose their
    configuration after an upgrade.  Existing files are never overwritten.
    """
    if os.getenv("CYBERWAVE_EDGE_CONFIG_DIR", "").strip():
        return
    if config_dir == _LEGACY_SYSTEM_CONFIG_DIR:
        return
    if not _LEGACY_SYSTEM_CONFIG_DIR.exists():
        return

    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    bootstrap_logger = logging.getLogger(__name__)
    copied_files = 0
    for json_file in _LEGACY_SYSTEM_CONFIG_DIR.glob("*.json"):
        if not json_file.is_file():
            continue
        target_file = config_dir / json_file.name
        if target_file.exists():
            continue
        try:
            shutil.copy2(json_file, target_file)
            if os.name != "nt":
                os.chmod(target_file, 0o600)
            copied_files += 1
        except OSError:
            continue
    if copied_files:
        bootstrap_logger.info(
            "Migrated %d legacy edge config file(s) from %s to %s",
            copied_files,
            _LEGACY_SYSTEM_CONFIG_DIR,
            config_dir,
        )


def _bootstrap_runtime_env_vars() -> None:
    """Load persisted runtime env vars into process env for child imports."""
    if os.name != "nt":
        os.umask(0o077)
    config_dir = _resolve_config_dir()
    _migrate_legacy_config(config_dir)
    credentials_file = config_dir / "credentials.json"
    try:
        if not credentials_file.exists():
            return
        with open(credentials_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    if not isinstance(data, dict):
        return

    envs: dict[str, str] = {}
    raw_envs = data.get("envs")
    if isinstance(raw_envs, dict):
        for key, value in raw_envs.items():
            if isinstance(key, str) and isinstance(value, str) and value.strip():
                envs[key] = value.strip()

    for key, value in envs.items():
        os.environ.setdefault(key, value)


_bootstrap_runtime_env_vars()

logger = logging.getLogger(__name__)
_edge_log_level_name = os.getenv("CYBERWAVE_EDGE_LOG_LEVEL", "info").strip().upper()
logger.setLevel(getattr(logging, _edge_log_level_name, logging.INFO))
console = Console()


def _resolve_package_version(package_name: str, fallback: str | None = None) -> str | None:
    """Resolve an installed package version with an optional fallback."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return fallback
    except Exception:
        return fallback


try:
    from cyberwave import __version__ as PACKAGE_CYBERWAVE_SDK_VERSION
except Exception:
    PACKAGE_CYBERWAVE_SDK_VERSION = None


EDGE_CORE_VERSION = _resolve_package_version(
    "cyberwave-edge-core",
    fallback=PACKAGE_EDGE_CORE_VERSION,
)
CYBERWAVE_SDK_VERSION = _resolve_package_version(
    "cyberwave",
    fallback=PACKAGE_CYBERWAVE_SDK_VERSION,
)

# Re-exported from driver_logs for backward compat and in-module use.
from .driver_logs import (  # noqa: E402
    _build_driver_log_payload,
    _CONTAINER_LOG_LAST_SEEN,
    _CONTAINER_LOG_THREADS,
    _log_and_publish_driver_message,
    _pull_docker_image_with_progress,
    _stream_container_logs,
    reconcile_driver_log_streams,
)

# Map container names to twin UUIDs so log threads can publish telemetry.
_CONTAINER_TWIN_MAP: dict[str, str] = {}

# Shared MQTT client for publishing driver log telemetry.
_shared_mqtt_client: Optional[Any] = None
_shared_mqtt_lock = threading.Lock()

# Module-level shutdown event; set by SIGTERM handler or KeyboardInterrupt
# in main.py to signal run_runtime_loop to stop.
shutdown_event = threading.Event()

# Track which token/path combination has already been announced to avoid
# repeated "Loaded token ..." info logs during steady-state polling.
_last_logged_token_signature: Optional[str] = None
_token_log_lock = threading.Lock()

# ---- constants ---------------------------------------------------------------

# Edge config directory. The systemd unit sets CYBERWAVE_EDGE_CONFIG_DIR on
# Linux. For manual invocation, defaults to ~/.cyberwave on all platforms.
CONFIG_DIR = _resolve_config_dir()
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
FINGERPRINT_FILE = CONFIG_DIR / "fingerprint.json"
ENVIRONMENT_FILE = CONFIG_DIR / "environment.json"
DEFAULT_API_URL = "https://api.cyberwave.com"
DEFAULT_ENVIRONMENT = "production"
DRIVER_CONTAINER_PREFIX = "cyberwave-driver-"
LOG_FOLLOWER_RECONCILE_INTERVAL_SECONDS = 15.0
EDGE_COMMAND_RESTART = "restart_edge_core"
DRIVER_RESTART_LOOP_THRESHOLD = int(os.getenv("CYBERWAVE_DRIVER_RESTART_LOOP_THRESHOLD", "4"))
DRIVER_RESTART_LOOP_WINDOW_SECONDS = float(
    os.getenv("CYBERWAVE_DRIVER_RESTART_LOOP_WINDOW_SECONDS", "60")
)

CONTAINER_PRUNE_INTERVAL_SECONDS = float(
    os.getenv("CYBERWAVE_CONTAINER_PRUNE_INTERVAL_SECONDS", "1800")  # 30 minutes
)
IMAGE_PRUNE_INTERVAL_SECONDS = float(
    os.getenv("CYBERWAVE_IMAGE_PRUNE_INTERVAL_SECONDS", "10800")  # 3 hours
)


def _atomic_write_json(path: Path, data: Any, *, mode: int = 0o600) -> None:
    """Atomically write *data* as JSON to *path* with restrictive permissions.

    Uses a sibling temp file + ``os.replace`` so readers never see a
    half-written file.  On POSIX the file is ``chmod``-ed to *mode*
    **before** the rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if os.name != "nt":
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def resolve_config_owner_uid_gid(config_dir: "Path | None" = None) -> "tuple[int, int] | None":
    """Return the ``(uid, gid)`` that should own files inside *config_dir*.

    Used to repair ownership when edge-core runs as root — either via
    ``sudo`` (where ``SUDO_UID`` is set) or via ``systemd`` (where it is
    not).  Resolution order:

    1. If ``SUDO_UID`` is present, use it (falling back to ``SUDO_UID``
       for the gid when ``SUDO_GID`` is absent).  Matches the previous
       ``sudo``-only behavior exactly.
    2. Otherwise, look at the owner of ``config_dir.parent`` (e.g.
       ``/home/alice`` for ``/home/alice/.cyberwave``).  When that
       directory is owned by a non-root user, return its uid/gid.  This
       is the systemd-case fallback: the unit file bakes in
       ``CYBERWAVE_EDGE_CONFIG_DIR=/home/<user>/.cyberwave`` at install
       time, so the home-directory owner is the user we want to chown to.

    Returns ``None`` when the process is not root on Linux or when no
    sensible non-root owner can be determined.
    """
    if platform.system() != "Linux" or os.getuid() != 0:
        return None

    sudo_uid = os.environ.get("SUDO_UID", "").strip()
    if sudo_uid:
        sudo_gid = os.environ.get("SUDO_GID", "").strip()
        try:
            uid = int(sudo_uid)
            gid = int(sudo_gid) if sudo_gid else uid
        except ValueError:
            return None
        return uid, gid

    cfg = config_dir if config_dir is not None else CONFIG_DIR
    try:
        st = cfg.parent.stat()
    except OSError:
        return None
    if st.st_uid == 0:
        return None
    return st.st_uid, st.st_gid

DEFAULT_DRIVER_TROUBLESHOOTING_URL = "https://docs.cyberwave.com"
DRIVER_TROUBLESHOOTING_URL = (
    os.getenv("CYBERWAVE_DRIVER_TROUBLESHOOTING_URL", DEFAULT_DRIVER_TROUBLESHOOTING_URL).strip()
    or DEFAULT_DRIVER_TROUBLESHOOTING_URL
)
_PROTECTED_CONFIG_JSON_FILES = {
    "credentials.json",
    "fingerprint.json",
    "environment.json",
}
_EDGE_COMMAND_SUBSCRIBED = False
_EDGE_COMMAND_SUBSCRIPTION_LOCK = threading.Lock()
_EDGE_RESTART_LOCK = threading.Lock()
_EDGE_RESTART_IN_PROGRESS = False
_HANDLED_EDGE_COMMAND_REQUEST_IDS: set[str] = set()
# Twin UUIDs whose ``cyberwave/twin/{uuid}/command`` topic is currently
# subscribed. Tracked as a set rather than a boolean so that twins paired
# *after* edge-core started are picked up automatically and twins that get
# unpaired stop being listened to (CYB-1766 follow-up).
_SUBSCRIBED_TWIN_COMMAND_UUIDS: set[str] = set()
_TWIN_COMMAND_SUBSCRIPTION_LOCK = threading.Lock()
TWIN_COMMAND_SYNC_WORKFLOWS = "sync_workflows"
_TWIN_FILE_CHECKSUMS: dict[str, str] = {}
_CONTAINER_LAST_RESTART_COUNT: dict[str, int] = {}
_CONTAINER_RESTART_HISTORY: dict[str, deque[float]] = {}
_EDGE_HEALTH_CHECK: Optional[Any] = None
_EDGE_HEALTH_CHECK_LOCK = threading.Lock()

# Resolved once per process at first use; shared by all container launches.
_ZENOH_CONFIG: Optional[ZenohConfig] = None
_ZENOH_CONFIG_LOCK = threading.Lock()
_TWIN_UPDATE_ALLOWED_FIELDS = frozenset(
    {
        "name",
        "description",
        "position_x",
        "position_y",
        "position_z",
        "rotation_w",
        "rotation_x",
        "rotation_y",
        "rotation_z",
        "scale_x",
        "scale_y",
        "scale_z",
        "kinematics_override",
        "joint_calibration",
        "metadata",
        "controller_policy_uuid",
        "attach_to_twin_uuid",
        "attach_to_link",
        "attach_offset_x",
        "attach_offset_y",
        "attach_offset_z",
        "attach_offset_rotation_w",
        "attach_offset_rotation_x",
        "attach_offset_rotation_y",
        "attach_offset_rotation_z",
        "fixed_base",
    }
)

# Sensor types that require camera device selection (RGB cameras)
RGB_SENSOR_TYPES = frozenset({"rgb", "camera", "rgb_camera", "rgbd"})
EDGE_HEALTH_PUBLISH_INTERVAL_SECONDS = max(
    1,
    int(os.getenv("CYBERWAVE_EDGE_HEALTH_PUBLISH_INTERVAL_SECONDS", "5")),
)


def _twin_has_rgb_sensor(asset: Any) -> bool:
    """Return True if the asset has an RGB sensor (camera).

    Checks universal_schema.sensors, metadata.capabilities.sensors, or
    registry_id for known camera assets.
    """
    return len(_get_device_requiring_sensor_ids(asset)) > 0


def _get_asset_registry_id(asset: Any) -> str:
    """Return normalized asset registry id or an empty string."""
    metadata = getattr(asset, "metadata", None) or {}
    registry_id = getattr(asset, "registry_id", None)
    if not registry_id and isinstance(metadata, dict):
        registry_id = metadata.get("registry_id")
    return str(registry_id).strip() if registry_id else ""


def _get_device_requiring_sensor_ids(asset: Any) -> list[str]:
    """Return sensor IDs that require a device port (e.g. /dev/video0).

    Sensors with type in RGB_SENSOR_TYPES need a port in metadata.sensors_devices.
    Uses "id" from schema if present, else "sensor_0", "sensor_1", etc.
    """
    sensor_ids: list[str] = []
    schema = None
    metadata = getattr(asset, "metadata", None) or {}
    if isinstance(metadata, dict):
        schema = metadata.get("universal_schema")
    if not schema:
        schema = getattr(asset, "universal_schema", None)
    if schema and isinstance(schema, dict):
        sensors = schema.get("sensors", [])
        if isinstance(sensors, list):
            for i, s in enumerate(sensors):
                if isinstance(s, dict) and (s.get("type") or "").lower() in RGB_SENSOR_TYPES:
                    sid = s.get("id") or f"sensor_{i}"
                    sensor_ids.append(str(sid))

    # Fallback: check capabilities.sensors from metadata
    if not sensor_ids:
        caps = metadata.get("capabilities", {}) if isinstance(metadata, dict) else {}
        if isinstance(caps, dict):
            sensors = caps.get("sensors", [])
            if isinstance(sensors, list):
                for i, s in enumerate(sensors):
                    if isinstance(s, dict) and (s.get("type") or "").lower() in RGB_SENSOR_TYPES:
                        sid = s.get("id") or f"sensor_{i}"
                        sensor_ids.append(str(sid))

    # Fallback: known camera registry IDs - assume single "camera" sensor
    if not sensor_ids:
        rid = _get_asset_registry_id(asset).lower()
        if "standard-cam" in rid or "realsense" in rid or "camera" in rid:
            sensor_ids.append("camera")

    return sensor_ids


def _get_unassigned_sensor_ids(twin_metadata: dict, sensor_ids: list[str]) -> list[str]:
    """Return sensor IDs that need a device port but have none in metadata.sensors_devices."""
    sensors_devices = twin_metadata.get("sensors_devices") or {}
    if not isinstance(sensors_devices, dict):
        return list(sensor_ids)
    unassigned: list[str] = []
    for sid in sensor_ids:
        port = sensors_devices.get(sid)
        if not port or not str(port).strip():
            unassigned.append(sid)
    return unassigned


def _check_and_alert_sensors_devices(
    twin_uuid: str, twin_name: str, asset: Any, twin_metadata: dict
) -> None:
    """If twin has device-requiring sensors but any lack a port in sensors_devices, send alert."""
    sensor_ids = _get_device_requiring_sensor_ids(asset)
    if not sensor_ids:
        return
    unassigned = _get_unassigned_sensor_ids(twin_metadata, sensor_ids)
    if unassigned:
        _send_alert_for_twin(
            twin_uuid,
            "Sensor device not assigned",
            f"Twin '{twin_name}' has sensors requiring device ports but no port is assigned "
            f"for: {', '.join(unassigned)}. Set metadata.sensors_devices (e.g. "
            '{{"camera": "/dev/video0"}}) via the frontend.',
            "sensors_devices",
            severity="warning",
        )


def _resolve_macos_camera_bridge_candidates(asset: Any, twin_metadata: dict[str, Any]) -> list[str]:
    """Resolve macOS camera bridge candidates for default metadata flows.

    This keeps Linux-oriented camera drivers transparent on macOS by providing
    default ``/dev/video*`` targets for the bridge command even when driver
    metadata has no explicit ``--device`` params.
    """
    candidate_devices: list[str] = []
    sensor_ids = _get_device_requiring_sensor_ids(asset)

    sensors_devices = twin_metadata.get("sensors_devices")
    if isinstance(sensors_devices, dict):
        for sensor_id in sensor_ids:
            value = sensors_devices.get(sensor_id)
            if isinstance(value, str) and value.strip().startswith("/dev/video"):
                candidate_devices.append(value.strip())

    video_device = twin_metadata.get("video_device")
    if isinstance(video_device, str) and video_device.strip().startswith("/dev/video"):
        candidate_devices.append(video_device.strip())

    if not candidate_devices and sensor_ids:
        candidate_devices.append("/dev/video0")

    return list(dict.fromkeys(candidate_devices))


def _read_cameras_config() -> Optional[dict]:
    """Return the raw ``cameras.json`` payload, or ``None`` when unavailable."""
    cameras_file = CONFIG_DIR / "cameras.json"
    if not cameras_file.exists():
        return None
    try:
        with open(cameras_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not read cameras.json")
        return None


def _coerce_video_index(value: Any) -> Optional[int]:
    """Best-effort conversion of a stored video index to ``int``."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _load_camera_stream_url_for_twin(twin_uuid: Optional[str]) -> Optional[str]:
    """Return the MJPEG stream URL assigned to *twin_uuid* on macOS, if any.

    Reads ``~/.cyberwave/camera_streams.json`` (written by the CLI installer
    when multiple camera twins are mapped to different AVFoundation cameras)
    and returns the entry for *twin_uuid*.  Returns ``None`` when no mapping
    exists or the file is missing/invalid.
    """
    if not twin_uuid:
        return None
    streams_file = CONFIG_DIR / "camera_streams.json"
    if not streams_file.exists():
        return None
    try:
        with open(streams_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not read camera_streams.json")
        return None
    mapping = data.get("twin_to_stream_url") or {}
    url = mapping.get(str(twin_uuid))
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


def _load_selected_camera_device(twin_uuid: Optional[str] = None) -> Optional[str]:
    """Read the selected video device from ``cameras.json``.

    When ``twin_uuid`` is provided, first look it up in the optional
    ``twin_to_device`` mapping persisted by the CLI.  Fall back to the global
    ``selected_device`` value for backward compatibility.  Returns the
    ``/dev/video<N>`` path, or ``None`` when no selection is available.
    """
    data = _read_cameras_config()
    if data is None:
        return None

    if twin_uuid:
        mapping = data.get("twin_to_device") or {}
        mapped = _coerce_video_index(mapping.get(str(twin_uuid)))
        if mapped is not None:
            return f"/dev/video{mapped}"

    selected_index = _coerce_video_index(data.get("selected_device"))
    if selected_index is None:
        return None
    return f"/dev/video{selected_index}"


def _resolve_mqtt_kwargs() -> dict[str, Any]:
    """Derive MQTT connection kwargs from runtime config.

    When the backend base URL is a local or private-network address the MQTT
    broker is assumed to be co-located on the same host at port 1883 (no TLS),
    matching the standard ``local.yml`` Docker Compose layout.  Otherwise the
    persisted ``CYBERWAVE_MQTT_HOST`` value is forwarded.
    """
    import ipaddress
    from urllib.parse import urlparse

    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()

    is_local = hostname in ("localhost", "0.0.0.0")
    if not is_local:
        try:
            is_local = ipaddress.ip_address(hostname).is_private
        except ValueError:
            pass

    if is_local:
        return {"mqtt_host": parsed.hostname, "mqtt_port": 1883}

    mqtt_host = get_runtime_env_var("CYBERWAVE_MQTT_HOST")
    if mqtt_host:
        return {"mqtt_host": mqtt_host}
    return {}


def _get_shared_mqtt_client(token: str) -> Any:
    """Return a shared MQTT client, creating and connecting it on first call."""
    global _shared_mqtt_client
    with _shared_mqtt_lock:
        if _shared_mqtt_client is not None and _shared_mqtt_client.mqtt.connected:
            return _shared_mqtt_client
        base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
        mqtt_kwargs = _resolve_mqtt_kwargs()
        try:
            client = Cyberwave(base_url=base_url, api_key=token, **mqtt_kwargs)
            client.mqtt.connect()
            _shared_mqtt_client = client
            logger.info("Shared MQTT client connected for log forwarding")
            return client
        except Exception as exc:
            logger.warning("Failed to create shared MQTT client: %s", exc)
            return None


def _mask_token(token: str) -> str:
    """Return a masked token string suitable for logs."""
    return f"{token[:6]}…{token[-4:]}" if len(token) > 12 else "***"


def _reset_logged_token_signature() -> None:
    """Clear token log dedupe state after missing/invalid credential reads."""
    global _last_logged_token_signature
    with _token_log_lock:
        _last_logged_token_signature = None


def _log_loaded_token_once(token: str) -> None:
    """Log the loaded token once per credentials path + token value."""
    global _last_logged_token_signature
    signature = f"{CREDENTIALS_FILE}:{hashlib.sha256(token.encode()).hexdigest()}"
    with _token_log_lock:
        if _last_logged_token_signature == signature:
            logger.debug("Token reloaded from %s without changes", CREDENTIALS_FILE)
            return
        _last_logged_token_signature = signature
    logger.info("Loaded token from %s (token: %s)", CREDENTIALS_FILE, _mask_token(token))


def load_token() -> Optional[str]:
    """Load the API token from the edge config credentials file.

    Returns the token string, or ``None`` if the file is missing or
    cannot be parsed.
    """
    if not CREDENTIALS_FILE.exists():
        _reset_logged_token_signature()
        logger.warning("Credentials file not found: %s", CREDENTIALS_FILE)
        return None
    try:
        with open(CREDENTIALS_FILE) as f:
            data = json.load(f)
        token = data.get("token") or None
        if token:
            _log_loaded_token_once(token)
        else:
            _reset_logged_token_signature()
            logger.warning(
                "Credentials file %s exists but has no 'token' field. Keys present: %s",
                CREDENTIALS_FILE,
                list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
        return token
    except (json.JSONDecodeError, OSError) as exc:
        _reset_logged_token_signature()
        logger.warning("Failed to read credentials file %s: %s", CREDENTIALS_FILE, exc)
        return None


def load_credentials_envs() -> dict[str, str]:
    """Load persisted runtime env vars from credentials.json.

    Expected schema:
        {"envs": {"CYBERWAVE_BASE_URL": "..."}}
    """
    if not CREDENTIALS_FILE.exists():
        return {}
    try:
        with open(CREDENTIALS_FILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}

    envs: dict[str, str] = {}
    raw_envs = data.get("envs")
    if isinstance(raw_envs, dict):
        for key, value in raw_envs.items():
            if (
                isinstance(key, str)
                and key.startswith("CYBERWAVE_")
                and isinstance(value, str)
                and value.strip()
            ):
                envs[key] = value.strip()
    return envs


def load_driver_overrides() -> dict[str, str]:
    """Load local driver image overrides from credentials.json.

    Allows overriding the cloud-configured driver image for a specific twin
    without needing cloud write access.  Useful when the asset's driver in the
    cloud cannot be updated (e.g. permission denied) but a different local image
    should be used instead.

    Expected schema:
        {"driver_overrides": {"<twin_uuid>": "<docker_image>"}}
    """
    if not CREDENTIALS_FILE.exists():
        return {}
    try:
        with open(CREDENTIALS_FILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}

    overrides: dict[str, str] = {}
    raw = data.get("driver_overrides")
    if isinstance(raw, dict):
        for twin_uuid, image in raw.items():
            if isinstance(twin_uuid, str) and isinstance(image, str) and image.strip():
                overrides[twin_uuid.strip()] = image.strip()
    return overrides


def load_worker_resource_limits() -> "Optional[Any]":
    """Build ResourceLimits for the worker container from credentials envs.

    Reads ``CYBERWAVE_WORKER_CPU_PERCENT`` (float, percent of one CPU; e.g.
    ``100`` = 1 CPU core, ``200`` = 2 cores) and
    ``CYBERWAVE_WORKER_MEMORY_MB`` (int, MiB) from the resolved env.

    When neither is explicitly set, auto-detects memory-constrained hosts
    (<=4 GB) and applies a safe default limit to prevent the worker from
    consuming all available memory and triggering the OOM killer.
    """
    from .worker_manager import ResourceLimits

    cpu_str = get_runtime_env_var("CYBERWAVE_WORKER_CPU_PERCENT")
    mem_str = get_runtime_env_var("CYBERWAVE_WORKER_MEMORY_MB")

    cpu: Optional[float] = None
    mem: Optional[int] = None

    if cpu_str:
        try:
            cpu = float(cpu_str)
        except ValueError:
            logger.warning("Invalid CYBERWAVE_WORKER_CPU_PERCENT=%r; ignoring", cpu_str)

    if mem_str:
        try:
            mem = int(mem_str)
        except ValueError:
            logger.warning("Invalid CYBERWAVE_WORKER_MEMORY_MB=%r; ignoring", mem_str)

    if cpu is None and mem is None:
        auto_limits = _auto_detect_worker_memory_limit()
        if auto_limits is not None:
            return auto_limits
        return None

    return ResourceLimits(cpu_quota_percent=cpu, memory_mb=mem)


def _auto_detect_worker_memory_limit() -> "Optional[Any]":
    """Auto-detect and apply a memory limit on memory-constrained hosts.

    On devices with <=4 GB total RAM (e.g. Raspberry Pi 4), running an ML
    inference worker without a memory limit risks the OOM killer terminating
    the edge-core orchestrator process.  This function reserves ~25% of
    total memory for the OS and edge-core, capping the worker at the rest.

    Opt out via ``CYBERWAVE_WORKER_AUTO_MEMORY_LIMIT=false``.

    Known limitation: ``/proc/meminfo`` reports the **host** total even
    when edge-core runs inside a cgroup-constrained container, so this
    function may misclassify edge-core-in-a-container hosts.  We don't
    run edge-core in a container today; if that changes, add a fallback
    that reads ``/sys/fs/cgroup/memory.max`` (cgroup v2) or
    ``memory.limit_in_bytes`` (cgroup v1) and uses the smaller value.
    """
    from .resource_monitor import read_memory_info
    from .worker_manager import ResourceLimits

    opt_out = (
        get_runtime_env_var("CYBERWAVE_WORKER_AUTO_MEMORY_LIMIT") or ""
    ).lower()
    if opt_out in ("0", "false", "no", "off"):
        return None

    mem_info = read_memory_info()
    if mem_info is None:
        return None

    total_mb = mem_info.total_mb
    if total_mb > 4096:
        return None

    reserved_mb = max(512, total_mb * 0.25)
    limit_mb = int(total_mb - reserved_mb)
    limit_mb = max(256, limit_mb)

    logger.info(
        "Auto-detected memory-constrained host (total=%.0fMB); "
        "setting worker container memory limit to %dMB "
        "(override with CYBERWAVE_WORKER_MEMORY_MB)",
        total_mb,
        limit_mb,
    )
    return ResourceLimits(memory_mb=limit_mb)


def get_runtime_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve runtime env var preferring process env, then credentials envs."""
    process_value = os.getenv(name)
    if isinstance(process_value, str) and process_value.strip():
        return process_value.strip()

    credentials_value = load_credentials_envs().get(name)
    if isinstance(credentials_value, str) and credentials_value.strip():
        return credentials_value.strip()
    return default


def validate_token(token: str, *, base_url: Optional[str] = None) -> bool:
    """Validate *token* by listing workspaces via the Cyberwave SDK.

    Returns ``True`` when the SDK call succeeds (i.e. the token is valid).
    """
    base_url = base_url or get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL)
    masked_token = f"{token[:6]}…{token[-4:]}" if len(token) > 12 else "***"
    logger.info("Validating token against %s via SDK (token: %s)", base_url, masked_token)
    try:
        client = Cyberwave(base_url=base_url, api_key=token)
        client.workspaces.list()
        logger.info("Token validated successfully (workspaces listed)")
        return True
    except Exception as exc:
        logger.warning("Token validation failed (%s): %s", base_url, exc)
        return False


def check_mqtt_connection(token: str) -> bool:
    """Try to connect to the MQTT broker via the Cyberwave Python SDK.

    The SDK reads broker host / port / credentials from environment
    variables (``CYBERWAVE_MQTT_HOST``, etc.) and falls back to sensible
    defaults.  Returns ``True`` if the connection succeeds.
    """
    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    mqtt_kwargs = _resolve_mqtt_kwargs()
    logger.info(
        "Attempting MQTT connection (base_url=%s, mqtt_kwargs=%s)",
        base_url,
        mqtt_kwargs,
    )
    try:
        client = Cyberwave(base_url=base_url, api_key=token, **mqtt_kwargs)
        client.mqtt.connect()
        connected: bool = client.mqtt.connected
        if connected:
            logger.info("MQTT connection successful")
            client.mqtt.disconnect()
        else:
            logger.warning("MQTT client connected but reports not connected")
        return connected
    except Exception as exc:
        logger.warning("MQTT connection check failed: %s: %s", type(exc).__name__, exc)
        return False


def load_environment_uuid(*, retries: int = 0, retry_delay_seconds: float = 0.2) -> Optional[str]:
    """Load linked environment UUID from the edge config environment file.

    Expected format:
        {"uuid": "unique-uuid-of-the-environment"}
    """
    if not ENVIRONMENT_FILE.exists():
        return None

    max_attempts = max(1, retries + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            with open(ENVIRONMENT_FILE) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("environment.json should contain a JSON object")
                return None

            env_uuid = data.get("uuid")
            if not isinstance(env_uuid, str) or not env_uuid.strip():
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds)
                    continue
                logger.warning("environment.json must contain a non-empty 'uuid' field")
                return None

            normalized_uuid = str(uuid.UUID(env_uuid.strip()))
            return normalized_uuid
        except ValueError:
            logger.warning("environment.json contains an invalid UUID format")
            return None
        except (json.JSONDecodeError, OSError) as exc:
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            logger.warning("Failed to read environment file: %s", exc)
            return None
    raise RuntimeError("Failed to load environment UUID from environment.json")


# ---- orchestrator ------------------------------------------------------------


def load_saved_fingerprint() -> Optional[str]:
    """Load a previously persisted fingerprint from disk."""
    if not FINGERPRINT_FILE.exists():
        return None
    try:
        with open(FINGERPRINT_FILE) as f:
            data = json.load(f)
        fingerprint = data.get("fingerprint")
        if isinstance(fingerprint, str) and fingerprint.strip():
            return fingerprint.strip()
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read fingerprint file: %s", exc)
        return None


def save_fingerprint(fingerprint: str) -> bool:
    """Persist fingerprint to the edge config directory.

    Written with mode ``0o644`` (world-readable) because the fingerprint
    is a hardware identifier, not a secret.  This matters when edge-core
    runs as root (systemd) but the invoking user later reads the file
    via the CLI: a restrictive ``0o600`` owned by root would make the
    CLI silently fall back to regenerating a fresh fingerprint,
    desynchronising it from the one registered with the backend.
    """
    try:
        _atomic_write_json(FINGERPRINT_FILE, {"fingerprint": fingerprint}, mode=0o644)
        return True
    except OSError as exc:
        logger.warning("Failed to save fingerprint file: %s", exc)
        return False


def get_or_create_fingerprint() -> Optional[str]:
    """Load fingerprint from disk, or generate and persist a new one."""
    saved = load_saved_fingerprint()
    if saved:
        return saved
    fingerprint = generate_fingerprint()
    if not save_fingerprint(fingerprint):
        return None
    return fingerprint


# Re-exported from docker_args for backward compat.
from .docker_args import (  # noqa: E402
    _build_driver_network_args,
    _docker_params_include_add_host,
    _extract_docker_device_mappings,
    _extract_docker_env_map,
    _is_video_device_path,
    _normalize_macos_bridge_candidates,
    _parse_bridge_resolved_device,
    _resolve_bool_env_var,
    _rewrite_macos_container_base_url,
    _rewrite_macos_container_hostname,
    _strip_video_device_mappings,
)


def _run_macos_device_bridge_commands(
    *,
    params: list[str],
    twin_uuid: str,
    container_name: str,
    additional_device_mappings: Optional[list[tuple[str, str]]] = None,
    usbip_active: bool = False,
) -> tuple[bool, dict[str, str]]:
    """Best-effort macOS host-bridge hook for linux-only ``--device`` mappings.

    If ``CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND`` is set, each extracted device
    mapping triggers one command execution. Supported template variables:
      - ``{host_device}``
      - ``{container_device}``
      - ``{twin_uuid}``
      - ``{container_name}``
      - ``{config_dir}``

    When *usbip_active* is True, ``/dev/video*`` mappings are skipped because
    USB/IP handles them transparently via the container entrypoint.
    """
    if platform.system() != "Darwin":
        return True, {}

    explicit_device_mappings = _extract_docker_device_mappings(params)
    device_mappings: list[tuple[str, str]] = []
    usbip_handled_video_devices: dict[str, str] = {}
    seen_mappings: set[tuple[str, str]] = set()
    for mapping in explicit_device_mappings + (additional_device_mappings or []):
        if mapping in seen_mappings:
            continue
        seen_mappings.add(mapping)
        host_device, container_device = mapping
        if usbip_active and (
            _is_video_device_path(host_device) or _is_video_device_path(container_device)
        ):
            logger.info(
                "USB/IP active — skipping bridge command for video device %s:%s "
                "(will be attached via USB/IP in container entrypoint)",
                host_device,
                container_device,
            )
            usbip_handled_video_devices[container_device] = container_device
            continue
        device_mappings.append(mapping)
    if not device_mappings:
        return True, usbip_handled_video_devices

    bridge_command_template = (
        get_runtime_env_var("CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND", "") or ""
    ).strip()
    if not bridge_command_template:
        logger.warning(
            "Driver uses --device mappings on macOS but CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND "
            "is not configured; hardware access will likely fail"
        )
        resolved = {container: container for _, container in device_mappings}
        resolved.update(usbip_handled_video_devices)
        return True, resolved

    resolved_device_map: dict[str, str] = dict(usbip_handled_video_devices)

    for host_device, container_device in device_mappings:
        try:
            rendered_command = bridge_command_template.format(
                host_device=host_device,
                container_device=container_device,
                twin_uuid=twin_uuid,
                container_name=container_name,
                config_dir=str(CONFIG_DIR),
            )
        except Exception as exc:
            logger.error(
                "Invalid CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND template %r: %s",
                bridge_command_template,
                exc,
            )
            return False, {}

        try:
            command_parts = shlex.split(rendered_command)
            if not command_parts:
                logger.error(
                    "CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND rendered to an empty command"
                )
                return False, {}
            result = subprocess.run(
                command_parts,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            resolved_device_map[container_device] = _parse_bridge_resolved_device(
                result.stdout or "",
                fallback=container_device,
            )
        except (
            ValueError,
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            logger.error(
                "Failed to run macOS hardware bridge command for device %s (%s): %s",
                host_device,
                container_device,
                exc,
            )
            return False, {}
    return True, resolved_device_map


def _get_zenoh_config() -> ZenohConfig:
    """Return the process-wide Zenoh configuration, resolved once on first call."""
    global _ZENOH_CONFIG
    if _ZENOH_CONFIG is not None:
        return _ZENOH_CONFIG
    with _ZENOH_CONFIG_LOCK:
        if _ZENOH_CONFIG is None:
            _ZENOH_CONFIG = ZenohConfig()
    return _ZENOH_CONFIG


def _run_docker_image(
    image: str,
    params: list[str],
    *,
    twin_uuid: str,
    token: str,
    child_camera_twin_uuids: Optional[list[str]] = None,
    macos_bridge_device_candidates: Optional[list[str]] = None,
    skip_pull: bool = False,
    prefer_gpu: bool = False,
    gpu_spec: str = "all",
    service_name: str | None = None,
    command: list[str] | None = None,
    service_env: dict[str, str] | None = None,
    driver_alert_ctx: Optional["DriverStartingAlertContext"] = None,
) -> bool:
    """Run a driver Docker container for a twin.

    When *skip_pull* is False (the default) the image is pulled first.
    Set *skip_pull* to True when images have already been fetched by an
    earlier parallel-pull phase.

    *driver_alert_ctx*, when provided, reuses an alert that was already
    created during the parallel-pull phase so the user sees a continuous
    ``driver_starting`` lifecycle.  When ``None`` a fresh alert is
    created (backwards-compatible with callers that skip the bulk pull).

    The container is started in detached mode with ``--restart unless-stopped``
    so it persists across reboots.  Environment variables are passed so the
    driver can authenticate with the Cyberwave backend and know which twin it
    controls.

    Returns ``True`` if the container was started successfully.
    """
    if not shutil.which("docker"):
        logger.error("Docker is not installed or not in PATH")
        return False

    if service_name:
        container_name = f"cyberwave-driver-{twin_uuid[:8]}-{service_name}"
    else:
        container_name = f"cyberwave-driver-{twin_uuid[:8]}"
    image = _resolve_driver_image_tag(image)
    runtime_environment = (
        get_runtime_env_var("CYBERWAVE_ENVIRONMENT", DEFAULT_ENVIRONMENT) or DEFAULT_ENVIRONMENT
    ).lower()

    # Remove any existing container with the same name (idempotent re-runs)
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=30,
    )

    if driver_alert_ctx is None:
        driver_alert_ctx = DriverStartingAlertContext(twin_uuid=twin_uuid, image=image)
        driver_alert_ctx.create()

    if skip_pull:
        if not _docker_image_exists_locally(image):
            logger.error("Image %s not available locally (skip_pull=True)", image)
            driver_alert_ctx.mark_failed_and_resolve(
                f"Driver image {image} not available locally after pull phase.",
                phase="image_missing",
            )
            return False
        driver_alert_ctx.update_metadata({"phase": "pull_skipped"}, force=True)
    else:
        try:
            _pull_docker_image_with_progress(
                image,
                container_name=container_name,
                twin_uuid=twin_uuid,
                token=token,
                driver_alert_ctx=driver_alert_ctx,
            )
        except subprocess.CalledProcessError as exc:
            err_tail = (exc.stderr or "").strip() or "unknown error"
            if _docker_image_exists_locally(image):
                logger.warning(
                    "Failed to pull docker image %s (%s); using local image copy",
                    image,
                    err_tail,
                )
                driver_alert_ctx.update_metadata(
                    {
                        "phase": "pull_failed_using_local",
                        "last_error": err_tail[:500],
                    },
                    force=True,
                )
            else:
                logger.error("Failed to pull docker image %s: %s", image, exc.stderr)
                driver_alert_ctx.mark_failed_and_resolve(
                    f"Failed to pull driver image {image}. {err_tail[:300]}",
                    phase="pull_failed",
                )
                return False
        except subprocess.TimeoutExpired:
            if _docker_image_exists_locally(image):
                logger.warning(
                    "Docker pull timed out for image %s; using local image copy",
                    image,
                )
                driver_alert_ctx.update_metadata(
                    {"phase": "pull_timeout_using_local"}, force=True
                )
            else:
                logger.error("Docker pull timed out for image: %s", image)
                driver_alert_ctx.mark_failed_and_resolve(
                    f"Timed out pulling driver image {image}.",
                    phase="pull_timeout",
                )
                return False
        except OSError as exc:
            if _docker_image_exists_locally(image):
                logger.warning(
                    "Docker pull OS error for image %s; using local image copy: %s",
                    image,
                    exc,
                )
                driver_alert_ctx.update_metadata(
                    {"phase": "pull_oserror_using_local", "last_error": str(exc)[:500]},
                    force=True,
                )
            else:
                logger.error("Docker pull failed for image %s: %s", image, exc)
                driver_alert_ctx.mark_failed_and_resolve(
                    f"Could not pull driver image {image}: {exc}",
                    phase="pull_oserror",
                )
                return False
        else:
            # NOTE: deliberately not "pull_complete" — the frontend treats that
            # as a terminal phase and would briefly drop the spinner between
            # pull-finished and container-running.  ``starting_container`` keeps
            # the spinner on through the docker-run + health-probe window.
            driver_alert_ctx.update_metadata(
                {"phase": "starting_container"}, force=True
            )

    # Build env vars for the container
    container_env: dict[str, str] = {
        "CYBERWAVE_TWIN_UUID": twin_uuid,
        "CYBERWAVE_API_KEY": token,
        "CYBERWAVE_EDGE_HOST_PLATFORM": platform.system().lower(),
    }
    if child_camera_twin_uuids:
        normalized_child_uuids = [str(child_uuid).strip() for child_uuid in child_camera_twin_uuids]
        normalized_child_uuids = [child_uuid for child_uuid in normalized_child_uuids if child_uuid]
        if normalized_child_uuids:
            child_uuids_csv = ",".join(dict.fromkeys(normalized_child_uuids))
            container_env["CYBERWAVE_CHILD_TWIN_UUIDS"] = child_uuids_csv

    explicit_params_env = _extract_docker_env_map(params)

    # On Linux, read the selected camera device from cameras.json so camera
    # drivers open the correct /dev/video* instead of defaulting to index 0.
    if platform.system() == "Linux" and "CYBERWAVE_METADATA_VIDEO_DEVICE" not in explicit_params_env:
        selected_video_device = _load_selected_camera_device(twin_uuid)
        if selected_video_device is not None:
            container_env.setdefault("CYBERWAVE_METADATA_VIDEO_DEVICE", selected_video_device)

    macos_bridge_mappings = _normalize_macos_bridge_candidates(macos_bridge_device_candidates)

    # Determine USB/IP state early so the bridge function can skip video
    # devices that USB/IP will handle transparently inside the container.
    usbip_active = platform.system() == "Darwin" and _is_usbip_server_running()

    # Check for an explicit MJPEG camera stream URL early.  When the user has
    # configured one, video device bridge resolution is pointless because the
    # driver will consume the HTTP stream instead of /dev/video*.
    #
    # Resolution order (most-specific wins):
    #   1) ``camera_streams.json['twin_to_stream_url'][twin_uuid]`` — set by
    #      the CLI installer when the user mapped multiple camera twins to
    #      distinct AVFoundation cameras.
    #   2) ``CYBERWAVE_MACOS_CAMERA_STREAM_URL`` runtime env var — legacy
    #      single-camera fallback.
    _macos_camera_stream_url: Optional[str] = None
    if platform.system() == "Darwin":
        _per_twin = _load_camera_stream_url_for_twin(twin_uuid)
        if _per_twin:
            _macos_camera_stream_url = _per_twin
        else:
            _raw = get_runtime_env_var("CYBERWAVE_MACOS_CAMERA_STREAM_URL")
            if _raw and _raw.strip():
                _macos_camera_stream_url = _raw.strip()

    macos_bridge_ok, macos_resolved_devices = _run_macos_device_bridge_commands(
        params=params,
        twin_uuid=twin_uuid,
        container_name=container_name,
        additional_device_mappings=macos_bridge_mappings,
        usbip_active=usbip_active,
    )
    if not macos_bridge_ok:
        driver_alert_ctx.mark_failed_and_resolve(
            f"macOS device bridge setup failed for image {image}.",
            phase="macos_bridge_failed",
        )
        return False

    if platform.system() == "Darwin" and not _macos_camera_stream_url:
        video_device_map = {
            container_device: resolved_device
            for container_device, resolved_device in macos_resolved_devices.items()
            if _is_video_device_path(container_device)
        }
        if video_device_map:
            container_env.setdefault(
                "CYBERWAVE_EDGE_VIDEO_DEVICE_MAP",
                json.dumps(video_device_map, separators=(",", ":")),
            )
            first_resolved_video_device = next(iter(video_device_map.values()))
            if "CYBERWAVE_METADATA_VIDEO_DEVICE" not in explicit_params_env:
                container_env.setdefault(
                    "CYBERWAVE_METADATA_VIDEO_DEVICE",
                    first_resolved_video_device,
                )

            # When USB/IP handles video devices, the container will see
            # /dev/video* natively after entrypoint attachment — don't strip
            # --device params and don't rewrite paths to RTSP URLs.
            if not usbip_active:
                should_strip_video_devices = _resolve_bool_env_var(
                    "CYBERWAVE_MACOS_STRIP_VIDEO_DEVICE_PARAMS",
                    default=True,
                )
                if should_strip_video_devices and any(
                    resolved != container_device
                    for container_device, resolved in video_device_map.items()
                ):
                    params = _strip_video_device_mappings(params)

    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL")
    if base_url:
        container_env["CYBERWAVE_BASE_URL"] = _rewrite_macos_container_base_url(base_url)
    mqtt_host = get_runtime_env_var("CYBERWAVE_MQTT_HOST")
    if mqtt_host:
        container_env["CYBERWAVE_MQTT_HOST"] = _rewrite_macos_container_hostname(
            mqtt_host
        )
    if runtime_environment != "production":
        container_env["CYBERWAVE_ENVIRONMENT"] = runtime_environment

    # Also forward additional CYBERWAVE_* env vars persisted by the CLI.
    for key, value in load_credentials_envs().items():
        if key.startswith("CYBERWAVE_"):
            if key in explicit_params_env:
                continue
            container_env.setdefault(key, value)

    # Forward CYBERWAVE_* from the edge core process environment so that
    # host-set vars (e.g. systemd Environment=, /etc/environment) reach
    # the driver container. E.g. CYBERWAVE_GO2_IP_ADDR for the Go2 driver.
    # setdefault avoids overwriting vars we or credentials already set.
    for key, value in os.environ.items():
        if key.startswith("CYBERWAVE_") and isinstance(value, str) and value.strip():
            if key in explicit_params_env:
                continue
            container_env.setdefault(key, value.strip())

    # Auto-infer MQTT TLS when port 8883 is configured but USE_TLS is absent.
    # Port 8883 is the IANA-assigned MQTT-over-TLS port; C++ drivers (unlike the
    # Python SDK) do not auto-detect this and need the explicit flag.
    if (
        "CYBERWAVE_MQTT_USE_TLS" not in container_env
        and "CYBERWAVE_MQTT_USE_TLS" not in explicit_params_env
    ):
        mqtt_port = container_env.get("CYBERWAVE_MQTT_PORT", "")
        if mqtt_port == "8883":
            container_env["CYBERWAVE_MQTT_USE_TLS"] = "true"

    # Driver reads setup.json from so101_lib under this dir (mounted CONFIG_DIR)
    container_env["CYBERWAVE_EDGE_CONFIG_DIR"] = "/app/.cyberwave"

    # Inject Zenoh transport configuration so drivers that use cw.data.publish()
    # automatically pick up the correct backend and router settings.  Drivers
    # that do not use the SDK data layer simply ignore these variables.
    # Use setdefault so that any per-driver override in explicit_params_env takes
    # precedence (driver metadata can always override with -e KEY=val in params).
    zenoh_env = build_zenoh_env_vars(_get_zenoh_config())
    for key, value in zenoh_env.items():
        if key not in explicit_params_env:
            container_env.setdefault(key, value)

    # On macOS, enable USB/IP passthrough when the host server is running.
    # --pid=host lets the container use nsenter to access Docker Desktop's
    # pre-installed usbip tools; CYBERWAVE_USBIP_ENABLED tells the entrypoint
    # to auto-attach devices (serial + video).
    pid_args: list[str] = []
    if usbip_active:
        pid_args = ["--pid=host"]
        container_env.setdefault("CYBERWAVE_USBIP_ENABLED", "1")
        has_video_devices = any(
            _is_video_device_path(d) for d in macos_resolved_devices
        )
        if has_video_devices and not _macos_camera_stream_url:
            container_env.setdefault("CYBERWAVE_USBIP_VIDEO_TIMEOUT_SECS", "8")

    # When the user has configured a macOS MJPEG camera stream URL, force
    # it as the video device.  This takes priority over bridge-resolved
    # /dev/video* paths and USB/IP video passthrough (which is often
    # unreliable for high-bandwidth video).
    if _macos_camera_stream_url:
        container_env["CYBERWAVE_METADATA_VIDEO_DEVICE"] = _macos_camera_stream_url
        logger.info(
            "macOS camera stream URL override: %s (usbip_active=%s)",
            _macos_camera_stream_url,
            usbip_active,
        )
    elif platform.system() == "Darwin" and macos_bridge_mappings and not usbip_active:
        logger.warning(
            "macOS camera twin %s has no MJPEG stream URL configured. "
            "The driver container will likely fail to open /dev/video* "
            "because Docker Desktop does not expose host cameras. "
            "Run: cyberwave edge install --reconfigure-camera",
            twin_uuid[:8],
        )
        try:
            _send_alert_for_twin(
                twin_uuid,
                "Camera not configured for macOS",
                "This camera twin has no MJPEG stream URL configured. Docker "
                "Desktop on macOS cannot pass /dev/video* devices to containers. "
                "Run 'cyberwave edge install --reconfigure-camera' to set up "
                "camera streaming.",
                "macos_camera_not_configured",
                severity="warning",
            )
        except Exception as exc:
            logger.debug(
                "Could not send macos_camera_not_configured alert: %s", exc
            )

    if service_env:
        container_env.update(service_env)

    env_vars: List[str] = []
    for key, value in container_env.items():
        env_vars += ["-e", f"{key}={value}"]

    twin_json_file = CONFIG_DIR / f"{twin_uuid}.json"
    if twin_json_file.is_file():
        env_vars += ["-v", f"{twin_json_file}:/app/{twin_uuid}.json"]
        env_vars += ["-e", f"CYBERWAVE_TWIN_JSON_FILE=/app/{twin_uuid}.json"]
    # Mount the edge config directory read-only so driver containers cannot
    # tamper with credentials.json or other sensitive config files.
    env_vars += ["-v", f"{CONFIG_DIR}:/app/.cyberwave:ro"]
    # SO101 drivers need write access to so101_lib/ for calibrations and URDF
    # downloads.  Mount that subdirectory read-write as an overlay.
    so101_lib_dir = CONFIG_DIR / "so101_lib"
    if so101_lib_dir.is_dir() or "so101" in image.lower():
        so101_lib_dir.mkdir(parents=True, exist_ok=True)
        env_vars += ["-v", f"{so101_lib_dir}:/app/.cyberwave/so101_lib"]

    network_args = _build_driver_network_args(params)

    gpu_args: list[str] = []
    if prefer_gpu and platform.system() == "Linux":
        from .docker_helpers import docker_has_nvidia_default_runtime, docker_has_nvidia_runtime

        if docker_has_nvidia_runtime() and docker_has_nvidia_default_runtime():
            gpu_value = gpu_spec or "all"
            gpu_args = ["--gpus", gpu_value]
            logger.info(
                "NVIDIA runtime detected with default daemon config — "
                "enabling GPU passthrough (--gpus %s) for %s",
                gpu_value,
                container_name,
            )
        elif docker_has_nvidia_runtime():
            logger.info(
                "NVIDIA runtime is available but not the default in "
                "/etc/docker/daemon.json — skipping --gpus for %s. "
                "Set \"default-runtime\": \"nvidia\" in "
                "/etc/docker/daemon.json to enable GPU passthrough.",
                container_name,
            )
        else:
            logger.debug(
                "prefer_gpu is set for %s but no NVIDIA runtime found",
                container_name,
            )

    cmd = [
        "docker",
        "run",
        "--detach",
        "--init",
        "--stop-timeout",
        "5",
        "--restart",
        "unless-stopped",
        "--privileged",
        *gpu_args,
        *pid_args,
        *network_args,
        "--name",
        container_name,
        *params,
        *env_vars,
        image,
        *(command or []),
    ]
    if logger.isEnabledFor(logging.DEBUG):
        debug_env_vars: list[str] = []
        for index, item in enumerate(env_vars):
            if item != "-e" or index + 1 >= len(env_vars):
                continue
            key, sep, value = env_vars[index + 1].partition("=")
            if sep and key == "CYBERWAVE_API_KEY":
                value = f"{value[:6]}…{value[-4:]}" if len(value) > 12 else "***"
            debug_env_vars.append(f"{key}{sep}{value}" if sep else env_vars[index + 1])

        debug_cmd = [
            (
                f"CYBERWAVE_API_KEY={arg.split('=', 1)[1][:6]}…{arg.split('=', 1)[1][-4:]}"
                if arg.startswith("CYBERWAVE_API_KEY=") and len(arg.split("=", 1)[1]) > 12
                else "CYBERWAVE_API_KEY=***"
                if arg.startswith("CYBERWAVE_API_KEY=")
                else arg
            )
            for arg in cmd
        ]
        logger.debug(
            "Docker run debug inputs for %s: image=%s params=%s env_vars=%s",
            container_name,
            image,
            params,
            debug_env_vars,
        )
        logger.debug("Docker run command args for %s: %s", container_name, debug_cmd)
    logger.info("Starting docker container %s from image %s", container_name, image)
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        _CONTAINER_TWIN_MAP[container_name] = twin_uuid
        _stream_container_logs(container_name, twin_uuid=twin_uuid, token=token)

        # A detached `docker run` can still fail immediately (e.g. missing USB
        # hardware causes rapid crashes). Verify that the container reaches and
        # stays in a running state for a brief window.
        for _ in range(5):
            inspect_data = _inspect_driver_container(container_name)
            if not inspect_data:
                time.sleep(1.0)
                continue
            state = inspect_data.get("State") if isinstance(inspect_data.get("State"), dict) else {}
            status = str(state.get("Status", "")).lower()
            if status == "running":
                driver_alert_ctx.update_metadata(
                    {"phase": "container_running"}, force=True
                )
                driver_alert_ctx.resolve()
                return True
            if status in {"restarting", "exited", "dead"}:
                logger.error(
                    "Driver container %s failed to start cleanly (status=%s error=%s)",
                    container_name,
                    status,
                    str(state.get("Error", "")).strip() or "none",
                )
                driver_alert_ctx.mark_failed_and_resolve(
                    (
                        f"Driver container {container_name} failed to start cleanly "
                        f"(status={status})."
                    ),
                    phase="container_unhealthy",
                )
                return False
            time.sleep(1.0)

        logger.warning(
            "Driver container %s did not reach a stable running state within startup probe window",
            container_name,
        )
        # Probe window elapsed without confirmation; the container may still
        # come up successfully, so close the alert as resolved (the caller
        # surfaces a separate ``driver_start_failure`` alert if needed).
        driver_alert_ctx.update_metadata(
            {"phase": "container_probe_unconfirmed"}, force=True
        )
        driver_alert_ctx.resolve()
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to start container %s: %s", container_name, exc.stderr)
        driver_alert_ctx.mark_failed_and_resolve(
            f"Failed to start container {container_name}.",
            phase="docker_run_failed",
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("Docker run timed out for image: %s", image)
        driver_alert_ctx.mark_failed_and_resolve(
            f"Docker run timed out for image {image}.",
            phase="docker_run_timeout",
        )
        return False


def _list_driver_containers(*, include_stopped: bool) -> list[str]:
    """Return edge-core managed driver container names."""
    if not shutil.which("docker"):
        return []

    command = ["docker", "ps"]
    if include_stopped:
        command.append("-a")
    command.extend(
        [
            "--format",
            "{{.Names}}",
            "--filter",
            f"name=^{DRIVER_CONTAINER_PREFIX}",
        ]
    )

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Failed to list running driver containers: %s", exc)
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _list_running_driver_containers() -> list[str]:
    """Return running driver container names managed by edge-core."""
    return _list_driver_containers(include_stopped=False)


def _docker_image_exists_locally(image: str) -> bool:
    """Return True when Docker already has *image* locally."""
    if not shutil.which("docker"):
        return False
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def _resolve_driver_image_tag(image: str) -> str:
    """Append the environment tag to a driver image reference when missing."""
    if ":" in image:
        return image
    runtime_environment = (
        get_runtime_env_var("CYBERWAVE_ENVIRONMENT", DEFAULT_ENVIRONMENT) or DEFAULT_ENVIRONMENT
    ).lower()
    if runtime_environment != "production":
        return f"{image}:{runtime_environment}"
    return image


def _maybe_rewrite_jetson_tag(image: str, twin_name: str = "") -> str:
    """On Jetson hardware, try the ``jetson-`` prefixed image tag.

    For example ``cyberwaveos/go2-ros2-driver:humble`` becomes
    ``cyberwaveos/go2-ros2-driver:jetson-humble``.

    If the tag already starts with ``jetson-`` or the platform is not Jetson,
    the image reference is returned unchanged.
    """
    from .driver_selection import is_jetson

    if not is_jetson():
        return image

    if ":" not in image:
        logger.info(
            "[Jetson] Detected Jetson platform for twin '%s' but image "
            "%s has no explicit tag — keeping as-is",
            twin_name,
            image,
        )
        return image

    repo, tag = image.rsplit(":", 1)

    if tag.startswith("jetson-"):
        return image

    jetson_tag = f"jetson-{tag}"
    jetson_image = f"{repo}:{jetson_tag}"
    logger.info(
        "[Jetson] Detected Jetson platform — rewriting driver tag for "
        "twin '%s': %s -> %s (will fall back to %s if pull fails)",
        twin_name,
        image,
        jetson_image,
        image,
    )
    return jetson_image


def _pull_driver_images_parallel(
    images: list[str],
    *,
    timeout: int = 600,
) -> dict[str, bool]:
    """Pull unique driver images in parallel with periodic progress logging.

    Returns a mapping of image -> success boolean.  Images that are already
    present locally are not re-pulled (but ``docker pull`` is still attempted
    to pick up newer tags — failure with a local copy is treated as success).
    """
    from concurrent.futures import ThreadPoolExecutor

    unique_images = list(dict.fromkeys(images))
    if not unique_images:
        return {}

    results: dict[str, bool] = {}

    def _pull_one(image: str) -> tuple[str, bool]:
        try:
            process = subprocess.Popen(
                ["docker", "pull", image],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            process.wait(timeout=timeout)
            return image, process.returncode == 0
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            return image, False
        except OSError:
            return image, False

    logger.info(
        "Pulling %d unique driver image(s) in parallel: %s",
        len(unique_images),
        ", ".join(unique_images),
    )

    futures_map: dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=min(len(unique_images), 4)) as pool:
        for img in unique_images:
            future = pool.submit(_pull_one, img)
            futures_map[future] = img

        pending = set(futures_map.keys())
        while pending:
            # Log a dot-style heartbeat while pulls are in progress.
            done_batch: set[Any] = set()
            for future in list(pending):
                if future.done():
                    done_batch.add(future)
            if done_batch:
                for future in done_batch:
                    pending.discard(future)
                    img = futures_map[future]
                    img_name, pulled = future.result()
                    if pulled:
                        logger.info("Pulled %s", img_name)
                        results[img_name] = True
                    elif _docker_image_exists_locally(img_name):
                        logger.warning(
                            "Pull failed for %s; using local copy", img_name
                        )
                        results[img_name] = True
                    else:
                        logger.error("Failed to pull %s and no local copy", img_name)
                        results[img_name] = False
            else:
                pulling_names = [futures_map[f] for f in pending]
                logger.info("Still pulling: %s ...", ", ".join(pulling_names))
                time.sleep(5)

    pulled_count = sum(1 for v in results.values() if v)
    failed_count = len(results) - pulled_count
    if failed_count:
        logger.warning(
            "Image pull complete: %d succeeded, %d failed", pulled_count, failed_count
        )
    else:
        logger.info("All %d driver image(s) ready", pulled_count)

    return results


def _inspect_driver_container(container_name: str) -> Optional[dict[str, Any]]:
    """Return raw ``docker inspect`` payload for one driver container."""
    if not shutil.which("docker"):
        return None
    try:
        result = subprocess.run(
            ["docker", "inspect", container_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("Invalid docker inspect JSON for container %s", container_name)
        return None
    if not isinstance(payload, list) or not payload:
        return None
    inspect_data = payload[0]
    return inspect_data if isinstance(inspect_data, dict) else None


def _resolve_container_twin_uuid(
    container_name: str, inspect_data: Optional[dict[str, Any]] = None
) -> Optional[str]:
    """Resolve twin UUID for a driver container from cache or inspect env vars."""
    cached = _CONTAINER_TWIN_MAP.get(container_name)
    if cached:
        return cached

    config = (inspect_data or {}).get("Config")
    envs = config.get("Env") if isinstance(config, dict) else None
    if not isinstance(envs, list):
        return None
    for env in envs:
        if not isinstance(env, str):
            continue
        if not env.startswith("CYBERWAVE_TWIN_UUID="):
            continue
        twin_uuid = env.split("=", 1)[1].strip()
        if twin_uuid:
            _CONTAINER_TWIN_MAP[container_name] = twin_uuid
            return twin_uuid
    return None


def _resolve_container_driver_image(
    inspect_data: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve the configured driver image for a container from inspect data."""
    config = (inspect_data or {}).get("Config")
    config_image = config.get("Image") if isinstance(config, dict) else None
    if isinstance(config_image, str) and config_image.strip():
        return config_image.strip()

    image_id = (inspect_data or {}).get("Image")
    if isinstance(image_id, str) and image_id.strip():
        return image_id.strip()
    return None


def _track_container_restarts(container_name: str, restart_count: int) -> tuple[int, int]:
    """Track per-container restart events and return (new_restarts, restarts_in_window)."""
    now = time.time()
    window_start = now - DRIVER_RESTART_LOOP_WINDOW_SECONDS
    history = _CONTAINER_RESTART_HISTORY.setdefault(container_name, deque())
    while history and history[0] < window_start:
        history.popleft()

    previous_count = _CONTAINER_LAST_RESTART_COUNT.get(container_name)
    _CONTAINER_LAST_RESTART_COUNT[container_name] = restart_count
    if previous_count is None:
        return 0, len(history)

    if restart_count < previous_count:
        # Container was recreated; reset local restart tracking baseline.
        history.clear()
        return 0, 0

    new_restarts = restart_count - previous_count
    if new_restarts > 0:
        for _ in range(min(new_restarts, DRIVER_RESTART_LOOP_THRESHOLD + 1)):
            history.append(now)
        while history and history[0] < window_start:
            history.popleft()
    return new_restarts, len(history)


def _stop_driver_container(container_name: str) -> bool:
    """Stop one flapping driver container and disable its restart policy."""
    try:
        subprocess.run(
            ["docker", "update", "--restart=no", container_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        # Best-effort: continue with stop even if update is not available.
        logger.debug("Could not set restart=no for %s", container_name, exc_info=True)

    try:
        subprocess.run(
            ["docker", "stop", container_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        _CONTAINER_LOG_THREADS.pop(container_name, None)
        _CONTAINER_LOG_LAST_SEEN.pop(container_name, None)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Failed to stop flapping driver container %s: %s", container_name, exc)
        return False


def _build_driver_restart_loop_message(
    *,
    twin_name: str,
    container_name: str,
    restart_count: int,
    restart_window_count: int,
) -> str:
    return (
        f"Driver container '{container_name}' for twin '{twin_name}' restarted "
        f"{restart_window_count} times in the last "
        f"{int(DRIVER_RESTART_LOOP_WINDOW_SECONDS)} seconds "
        f"(total restarts reported by Docker: {restart_count}). "
        f"The container was stopped automatically to prevent continuous rebooting. "
        f"Troubleshooting: {DRIVER_TROUBLESHOOTING_URL}"
    )


def reconcile_driver_restart_failures() -> dict[str, int]:
    """Detect flapping drivers and stop them after too many restarts."""
    all_containers = _list_driver_containers(include_stopped=True)
    active_names = set(all_containers)

    for stale in set(_CONTAINER_LAST_RESTART_COUNT) - active_names:
        _CONTAINER_LAST_RESTART_COUNT.pop(stale, None)
    for stale in set(_CONTAINER_RESTART_HISTORY) - active_names:
        _CONTAINER_RESTART_HISTORY.pop(stale, None)

    summary = {"inspected": 0, "flapping": 0, "stopped": 0, "alerts_sent": 0}
    for container_name in all_containers:
        inspect_data = _inspect_driver_container(container_name)
        if not inspect_data:
            continue
        summary["inspected"] += 1

        try:
            restart_count = int(inspect_data.get("RestartCount") or 0)
        except (TypeError, ValueError):
            restart_count = 0
        new_restarts, restarts_in_window = _track_container_restarts(container_name, restart_count)
        if new_restarts <= 0:
            continue
        if restarts_in_window <= DRIVER_RESTART_LOOP_THRESHOLD:
            continue

        state = inspect_data.get("State") if isinstance(inspect_data.get("State"), dict) else {}
        state_status = str(state.get("Status", "")).lower()
        state_error = str(state.get("Error", "")).strip()
        twin_uuid = _resolve_container_twin_uuid(container_name, inspect_data)
        twin_name = f"twin-{(twin_uuid or 'unknown')[:8]}"
        summary["flapping"] += 1

        stopped = _stop_driver_container(container_name)
        if stopped:
            summary["stopped"] += 1

        _CONTAINER_RESTART_HISTORY.pop(container_name, None)
        logger.error(
            (
                "Driver container %s exceeded restart threshold (%d > %d in %ss). "
                "status=%s docker_error=%s stopped=%s"
            ),
            container_name,
            restarts_in_window,
            DRIVER_RESTART_LOOP_THRESHOLD,
            int(DRIVER_RESTART_LOOP_WINDOW_SECONDS),
            state_status or "unknown",
            state_error or "none",
            stopped,
        )

        if not twin_uuid:
            continue
        try:
            _send_alert_for_twin(
                twin_uuid,
                "Driver restart loop detected",
                _build_driver_restart_loop_message(
                    twin_name=twin_name,
                    container_name=container_name,
                    restart_count=restart_count,
                    restart_window_count=restarts_in_window,
                ),
                "driver_restart_loop",
                severity="error",
            )
            summary["alerts_sent"] += 1
        except Exception as exc:
            logger.warning(
                "Failed to send restart-loop alert for twin %s (container=%s): %s",
                twin_uuid,
                container_name,
                exc,
            )
    return summary


_DRIVER_HEALTH_PREVIOUS: dict[str, str] = {}


def reconcile_driver_health_for_worker() -> dict[str, str]:
    """Check whether any driver container has gone down since the last cycle.

    Returns a mapping of ``container_name -> status`` for all tracked drivers.
    When a driver transitions from ``running`` to ``exited``/``dead``/``none``,
    a warning is logged and an alert is sent to the corresponding twin.

    This complements ``reconcile_driver_restart_failures`` which handles
    crash-loop detection — this function catches clean exits and removals.

    Note: if a driver is stopped by the restart-loop handler in the same
    reconcile cycle, both this function and the restart handler may send
    alerts.  The alert types differ (``driver_health`` vs
    ``driver_restart_loop``), so this is informative rather than redundant.
    """
    running = set(_list_running_driver_containers())
    all_containers = set(_list_driver_containers(include_stopped=True))
    known = set(_DRIVER_HEALTH_PREVIOUS.keys())

    statuses: dict[str, str] = {}
    for name in all_containers | known:
        if name in running:
            statuses[name] = "running"
        elif name in all_containers:
            statuses[name] = "down"
        else:
            statuses[name] = "removed"

    for name, status in statuses.items():
        prev = _DRIVER_HEALTH_PREVIOUS.get(name)
        if prev == "running" and status in {"down", "removed"}:
            twin_uuid = _CONTAINER_TWIN_MAP.get(name)
            logger.warning(
                "Driver container %s (twin %s) went %s while worker may be running",
                name,
                (twin_uuid or "unknown")[:8],
                status,
            )
            if twin_uuid:
                try:
                    _send_alert_for_twin(
                        twin_uuid,
                        "Driver container stopped",
                        f"Driver container '{name}' is no longer running ({status}). "
                        f"Frames from this camera will not be available to the worker.",
                        "driver_health",
                        severity="warning",
                    )
                except Exception as exc:
                    logger.debug("Could not send driver-down alert: %s", exc)

    _DRIVER_HEALTH_PREVIOUS.clear()
    _DRIVER_HEALTH_PREVIOUS.update(statuses)
    return statuses


# Debounce state for ``reconcile_driver_revival``.  Module-level so the
# 60s minimum gap between revival attempts persists across loop ticks.
# A monotonic timestamp is used so wall-clock jumps cannot bypass the
# debounce.
_LAST_REVIVAL_ATTEMPT_AT: Optional[float] = None
_REVIVAL_BACKOFF_SECONDS = 60


def reconcile_driver_revival(
    *,
    skip_revival: bool = False,
) -> dict[str, int]:
    """Re-spawn driver containers that exited cleanly.

    Reads the driver health snapshot populated by
    :func:`reconcile_driver_health_for_worker` earlier in the same
    runtime-loop tick.  When at least one driver is in ``down`` state
    (the container still exists but is not running — e.g. clean exit
    via SIGTERM) and the debounce window has elapsed, re-runs
    :func:`fetch_and_run_twin_drivers` to bring the missing container
    back up.

    **Fully removed containers are intentionally not revived.**  When a
    driver is gone from Docker entirely (``docker rm -f``,
    ``docker system prune``, manual cleanup) the health snapshot
    reports status ``removed`` and this reconciler leaves it alone.
    Removal is treated as an explicit operator signal — auto-respawning
    would fight the operator on planned takedowns, image swaps, and
    debug sessions.  To bring such a container back, restart edge-core
    or re-link the twin.

    Only containers that this edge-core process is managing (tracked in
    ``_CONTAINER_TWIN_MAP``) are eligible for revival.  Stopped driver
    containers belonging to twins that are no longer linked to this edge
    are treated as orphans and skipped, otherwise every revival cycle
    would re-run :func:`fetch_and_run_twin_drivers` — which forcibly
    recreates the *currently healthy* drivers as a side effect of
    iterating linked twins (CYB-2231).

    This complements:

    * :func:`reconcile_driver_restart_failures` — only *stops* flapping
      drivers; it never starts anything.
    * The Docker ``--restart unless-stopped`` policy on driver
      containers — ``unless-stopped`` does not auto-revive containers
      that exited cleanly (exit 0 after SIGTERM), only crashed ones.

    Without this reconciler a driver that's stopped via ``docker stop``
    (or by ``_graceful_shutdown`` followed by an edge-core restart that
    didn't actually re-run boot-time startup) stays down forever.

    Caller-supplied *skip_revival* lets the runtime loop opt out for one
    tick — used when the flap detector just stopped a driver, so we
    don't immediately undo its decision.

    Returns a summary dict.  All values default to 0 so callers can log
    a single line regardless of which branch fired.
    """
    global _LAST_REVIVAL_ATTEMPT_AT
    summary = {
        "down": 0,
        "skipped_orphan": 0,
        "skipped_flap_protection": 0,
        "skipped_debounce": 0,
        "skipped_no_credentials": 0,
        "revived_attempted": 0,
    }

    all_down_names = [n for n, s in _DRIVER_HEALTH_PREVIOUS.items() if s == "down"]
    if not all_down_names:
        return summary

    # Skip stopped containers we don't manage.  ``_CONTAINER_TWIN_MAP`` is
    # populated only when this process successfully starts a driver, so
    # membership identifies containers belonging to twins currently linked
    # to this edge.  Without this filter, a leftover stopped container from
    # an unlinked twin keeps re-triggering ``fetch_and_run_twin_drivers``
    # and force-recreating the healthy drivers via the idempotent
    # ``docker rm -f`` step (CYB-2231).
    orphan_names = [n for n in all_down_names if n not in _CONTAINER_TWIN_MAP]
    down_names = [n for n in all_down_names if n in _CONTAINER_TWIN_MAP]
    if orphan_names:
        summary["skipped_orphan"] = len(orphan_names)
        logger.debug(
            "Driver revival ignoring %d orphan stopped container(s) "
            "(twin no longer linked to this edge): %s",
            len(orphan_names),
            ", ".join(orphan_names),
        )
    if not down_names:
        return summary
    summary["down"] = len(down_names)

    if skip_revival:
        summary["skipped_flap_protection"] = len(down_names)
        return summary

    now = time.monotonic()
    if (
        _LAST_REVIVAL_ATTEMPT_AT is not None
        and (now - _LAST_REVIVAL_ATTEMPT_AT) < _REVIVAL_BACKOFF_SECONDS
    ):
        summary["skipped_debounce"] = len(down_names)
        return summary

    token = load_token()
    fingerprint = load_saved_fingerprint()
    environment_uuid = load_environment_uuid()
    if not (token and fingerprint and environment_uuid):
        summary["skipped_no_credentials"] = len(down_names)
        return summary

    _LAST_REVIVAL_ATTEMPT_AT = now
    summary["revived_attempted"] = len(down_names)

    logger.info(
        "Reviving %d down driver container(s): %s",
        len(down_names),
        ", ".join(down_names),
    )
    try:
        fetch_and_run_twin_drivers(token, environment_uuid, fingerprint)
    except Exception:
        logger.exception("Driver revival run failed")

    return summary


_cameras_json_mtime: Optional[float] = None


def _get_container_env_var(inspect_data: dict[str, Any], key: str) -> Optional[str]:
    """Extract a single env var value from ``docker inspect`` output."""
    config = inspect_data.get("Config") or {}
    for entry in config.get("Env") or []:
        if isinstance(entry, str) and entry.startswith(f"{key}="):
            return entry.split("=", 1)[1]
    return None


def reconcile_camera_config_drift() -> bool:
    """Detect ``cameras.json`` changes and trigger a targeted driver restart.

    Compares the video device in ``cameras.json`` against what each running
    driver container was launched with.  When they diverge, the stale
    container is removed and edge-core re-runs driver startup so the new
    device is picked up — no full service restart required.

    NOTE (CYB-2004): this function reads **only** ``cameras.json`` mtime
    and ``docker inspect`` env vars.  It does not inspect any
    ``edge_health`` payload, MQTT subscription, or per-stream
    ``stream_config`` block.  Changes to the ``edge_health`` schema are
    therefore behaviourally invisible here.  A regression test in
    ``tests/test_startup_core.py`` pins this property so a future
    refactor that would couple the two cannot land without an explicit
    edit to that test.

    Returns True if a restart was triggered.
    """
    global _cameras_json_mtime

    if platform.system() != "Linux":
        return False

    cameras_file = CONFIG_DIR / "cameras.json"
    if not cameras_file.exists():
        return False

    try:
        current_mtime = cameras_file.stat().st_mtime
    except OSError:
        return False

    if _cameras_json_mtime is None:
        _cameras_json_mtime = current_mtime
        return False

    if current_mtime == _cameras_json_mtime:
        return False

    _cameras_json_mtime = current_mtime

    # Fallback to the legacy global device when no per-twin mapping exists;
    # otherwise each container is compared against the device its twin is
    # bound to.
    fallback_device = _load_selected_camera_device()
    if fallback_device is None:
        return False

    containers = _list_running_driver_containers()
    stale_containers: list[str] = []
    restart_reason_device = fallback_device

    for container_name in containers:
        inspect_data = _inspect_driver_container(container_name)
        if not inspect_data:
            continue
        current_device = _get_container_env_var(
            inspect_data, "CYBERWAVE_METADATA_VIDEO_DEVICE"
        )
        if current_device is None:
            continue
        container_twin_uuid = _get_container_env_var(
            inspect_data, "CYBERWAVE_TWIN_UUID"
        )
        desired_device = (
            _load_selected_camera_device(container_twin_uuid)
            if container_twin_uuid
            else fallback_device
        ) or fallback_device
        if current_device != desired_device:
            logger.info(
                "Camera config drift detected for %s (twin=%s): "
                "container has %s, cameras.json wants %s",
                container_name,
                container_twin_uuid or "<unknown>",
                current_device,
                desired_device,
            )
            stale_containers.append(container_name)
            restart_reason_device = desired_device

    if not stale_containers:
        return False

    logger.info(
        "Triggering edge restart to apply camera config change (%s)",
        restart_reason_device,
    )
    token = load_token()
    if not token:
        logger.warning("Cannot restart drivers for camera config change: no token")
        return False

    _perform_edge_core_restart(token)
    return True


def _stop_and_prune_driver_containers() -> list[str]:
    """Force-remove edge-core driver containers and prune stopped containers."""
    containers = _list_driver_containers(include_stopped=True)
    removed: list[str] = []
    for container_name in containers:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            removed.append(container_name)
            _CONTAINER_TWIN_MAP.pop(container_name, None)
            _CONTAINER_LOG_THREADS.pop(container_name, None)
            _CONTAINER_LOG_LAST_SEEN.pop(container_name, None)
            _CONTAINER_LAST_RESTART_COUNT.pop(container_name, None)
            _CONTAINER_RESTART_HISTORY.pop(container_name, None)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Failed to remove driver container %s: %s", container_name, exc)

    if shutil.which("docker"):
        try:
            subprocess.run(
                ["docker", "container", "prune", "--force"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Failed to prune stopped containers: %s", exc)

    return removed


def _remove_cached_twin_json_files() -> list[str]:
    """Remove cached twin JSON objects so they can be re-downloaded."""
    removed: list[str] = []
    for json_file in CONFIG_DIR.glob("*.json"):
        if json_file.name in _PROTECTED_CONFIG_JSON_FILES:
            continue

        # Driver twin object files are UUID-based (<twin_uuid>.json). Skip any
        # user/system JSON files that don't match that naming contract.
        try:
            uuid.UUID(json_file.stem)
        except ValueError:
            continue

        try:
            if json_file.is_file() or json_file.is_symlink():
                json_file.unlink()
            elif json_file.is_dir():
                shutil.rmtree(json_file)
            removed.append(json_file.name)
        except OSError as exc:
            logger.warning("Failed to remove cached twin object %s: %s", json_file, exc)
    return removed


def _resolve_attach_to_twin_uuid(client: Any, twin: Any, twin_metadata: dict) -> Optional[str]:
    """Resolve attach_to_twin_uuid from list payload, metadata, or raw twin fetch."""
    attach_to = getattr(twin, "attach_to_twin_uuid", None)
    if not attach_to and hasattr(twin, "_data"):
        data = twin._data
        attach_to = (
            getattr(data, "attach_to_twin_uuid", None)
            if not isinstance(data, dict)
            else data.get("attach_to_twin_uuid")
        )
    if not attach_to:
        attach_to = twin_metadata.get("attach_to_twin_uuid")
    if not attach_to:
        try:
            full = client.twins.get_raw(str(getattr(twin, "uuid", "")))
            if hasattr(full, "attach_to_twin_uuid"):
                attach_to = full.attach_to_twin_uuid
            elif isinstance(full, dict):
                attach_to = full.get("attach_to_twin_uuid")
        except Exception:
            pass
    return str(attach_to) if attach_to else None


def _persist_twin_json_for_driver(twin: Any, twin_uuid: str, asset: Any) -> None:
    """Persist the twin+asset JSON file consumed by edge drivers."""
    twin_data = (
        twin.to_dict()
        if hasattr(twin, "to_dict")
        else {"uuid": twin_uuid, "name": getattr(twin, "name", None)}
    )
    asset_data = asset.to_dict() if hasattr(asset, "to_dict") else {}
    write_or_update_twin_json_file(twin_uuid, twin_data, asset_data)


def _is_legacy_edge_configs_map(edge_configs: dict[str, Any]) -> bool:
    """Return True for legacy edge_configs maps keyed by fingerprint."""
    if not edge_configs:
        return False
    if "edge_fingerprint" in edge_configs or "camera_config" in edge_configs:
        return False
    return all(isinstance(entry, dict) for entry in edge_configs.values())


def _is_twin_linked_to_fingerprint(twin_metadata: dict[str, Any], fingerprint: str) -> bool:
    """Return True when twin metadata indicates linkage to *fingerprint*."""
    candidate = str(twin_metadata.get("edge_fingerprint", "")).strip()
    if candidate and candidate == fingerprint:
        return True

    edge_configs = twin_metadata.get("edge_configs")
    if not isinstance(edge_configs, dict):
        return False

    nested_fingerprint = str(edge_configs.get("edge_fingerprint", "")).strip()
    if nested_fingerprint and nested_fingerprint == fingerprint:
        return True

    if _is_legacy_edge_configs_map(edge_configs):
        return fingerprint in edge_configs
    return False


def _list_linked_twin_uuids_for_fingerprint(
    token: str,
    environment_uuid: str,
    fingerprint: str,
) -> list[str]:
    """Resolve linked twin UUIDs for one edge fingerprint."""
    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    client = Cyberwave(base_url=base_url, api_key=token)
    twins = client.twins.list(environment_id=environment_uuid)
    if not twins:
        return []

    linked: list[str] = []
    for twin in twins:
        twin_uuid = str(getattr(twin, "uuid", "")).strip()
        if not twin_uuid:
            continue
        twin_metadata = twin.metadata if isinstance(twin.metadata, dict) else {}
        if _is_twin_linked_to_fingerprint(twin_metadata, fingerprint):
            linked.append(twin_uuid)
    return list(dict.fromkeys(linked))


def load_selected_twin_uuids() -> Optional[list[str]]:
    """Twin UUIDs the operator picked at install time, from ``environment.json``.

    Returns a (possibly empty) list when ``twin_uuids`` is present, or ``None``
    when the file or field is missing so callers can fall back to fingerprint
    discovery for installs that predate the field (pre–Feb 2026).
    """
    try:
        data = json.loads(ENVIRONMENT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = data.get("twin_uuids") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    return list(dict.fromkeys(str(u).strip() for u in raw if isinstance(u, str) and u.strip()))


def _resolve_worker_sync_twin_uuids(
    token: str,
    environment_uuid: str,
    fingerprint: str,
) -> list[str]:
    """Twin UUIDs to pull workflow workers for.

    The operator's selection in ``environment.json`` wins when present (incl.
    an explicit empty list). Falls back to fingerprint metadata discovery
    only for legacy installs missing the field.
    """
    selected = load_selected_twin_uuids()
    if selected is not None:
        return selected
    return _list_linked_twin_uuids_for_fingerprint(token, environment_uuid, fingerprint)


def _build_host_metrics_provider(
    resource_monitor: Optional[Any],
    watchdog: Optional[Any],
) -> Optional[Any]:
    """Return a zero-arg provider that merges host pressure into edge_health.

    Returns ``None`` when neither ``resource_monitor`` nor ``watchdog`` is
    available so :class:`EdgeHealthCheck` can keep its existing minimal
    payload shape (any subclass / older edge node without the host
    instrumentation continues to publish unchanged).

    The closure is intentionally tolerant: each subreader is wrapped so
    that a single misbehaving source (e.g. a resource monitor that has
    never been ``check()``-ed yet) cannot suppress the whole payload.
    """
    if resource_monitor is None and watchdog is None:
        return None

    def _provider() -> dict[str, Any]:
        out: dict[str, Any] = {}
        if resource_monitor is not None:
            try:
                snap = resource_monitor.last_snapshot
                if snap is not None:
                    out.update(snap.to_publish_dict())
                out["consecutive_critical"] = resource_monitor.consecutive_critical_count
            except Exception:  # pragma: no cover - defensive
                logger.debug("resource_monitor metrics unavailable", exc_info=True)
        if watchdog is not None:
            try:
                out["watchdog_layers"] = watchdog.active_layers()
            except Exception:  # pragma: no cover - defensive
                logger.debug("watchdog active_layers() raised", exc_info=True)
        return out

    return _provider


def _start_bootstrap_edge_health_publisher(
    token: str,
    twin_uuids: list[str],
    *,
    edge_id: str,
    resource_monitor: Optional[Any] = None,
    watchdog: Optional[Any] = None,
) -> bool:
    """Start (or refresh) a lightweight edge_health publisher for linked twins.

    ``resource_monitor`` and ``watchdog``, when supplied, are folded into a
    provider closure that augments every published payload with host-level
    metrics (memory %, CPU temp, watchdog layers, consecutive-critical
    counter).  Only the bootstrap publisher running on the edge host should
    pass these; driver containers see their container's ``/proc``, not the
    host's, so they must publish without a provider.
    """
    global _EDGE_HEALTH_CHECK
    if not twin_uuids:
        return False

    try:
        from cyberwave.edge.health import EdgeHealthCheck
    except Exception as exc:
        logger.warning("Cannot start edge health publisher: %s", exc)
        return False

    mqtt_client = _get_shared_mqtt_client(token)
    if not mqtt_client or not getattr(mqtt_client, "mqtt", None):
        logger.warning("Cannot start edge health publisher: shared MQTT client unavailable")
        return False

    normalized_twin_uuids = list(
        dict.fromkeys(
            [str(twin_uuid).strip() for twin_uuid in twin_uuids if str(twin_uuid).strip()]
        )
    )
    if not normalized_twin_uuids:
        return False

    host_metrics_provider = _build_host_metrics_provider(resource_monitor, watchdog)

    with _EDGE_HEALTH_CHECK_LOCK:
        if _EDGE_HEALTH_CHECK is not None:
            existing = list(getattr(_EDGE_HEALTH_CHECK, "twin_uuids", []) or [])
            _EDGE_HEALTH_CHECK.twin_uuids = list(dict.fromkeys(existing + normalized_twin_uuids))
            _EDGE_HEALTH_CHECK.edge_id = edge_id
            # Refresh the provider when a caller upgraded from a no-monitor
            # invocation to one that has the monitor wired (legitimate when
            # the runtime loop spins up after early bootstrap).
            if host_metrics_provider is not None:
                _EDGE_HEALTH_CHECK.host_metrics_provider = host_metrics_provider
            _EDGE_HEALTH_CHECK.start()
            return True

        _EDGE_HEALTH_CHECK = EdgeHealthCheck(
            mqtt_client=mqtt_client.mqtt,
            twin_uuids=normalized_twin_uuids,
            edge_id=edge_id,
            interval=EDGE_HEALTH_PUBLISH_INTERVAL_SECONDS,
            host_metrics_provider=host_metrics_provider,
        )
        _EDGE_HEALTH_CHECK.start()
        logger.info(
            "Started bootstrap edge health publisher for %d twin(s) (edge_id=%s, host_metrics=%s)",
            len(normalized_twin_uuids),
            edge_id,
            "on" if host_metrics_provider is not None else "off",
        )
        return True


def _stop_bootstrap_edge_health_publisher() -> None:
    """Stop the bootstrap edge_health publisher.

    Called once driver containers are running because drivers publish their
    own health — keeping the bootstrap publisher alive produces a duplicate
    ``edge_id`` that confuses the frontend.
    """
    global _EDGE_HEALTH_CHECK
    with _EDGE_HEALTH_CHECK_LOCK:
        if _EDGE_HEALTH_CHECK is not None:
            try:
                _EDGE_HEALTH_CHECK.stop()
            except Exception:
                pass
            _EDGE_HEALTH_CHECK = None
            logger.info("Stopped bootstrap edge health publisher (drivers running)")


def _clear_stale_driver_starting_alerts(
    twin_uuids: Iterable[str],
    *,
    log_context: str,
) -> int:
    """Resolve orphan ``driver_starting`` alerts for the given twins.

    Best-effort: failures are logged and never raised.  Returns the number
    of alerts resolved.  *twin_uuids* may contain duplicates; each twin is
    processed at most once.
    """
    seen: set[str] = set()
    cleared = 0
    for twin_uuid in twin_uuids:
        if not twin_uuid or twin_uuid in seen:
            continue
        seen.add(twin_uuid)
        try:
            cleared += DriverStartingAlertContext.resolve_active_for_twin(twin_uuid)
        except Exception:
            logger.debug(
                "Failed to clear stale driver_starting alerts for twin %s",
                twin_uuid,
                exc_info=True,
            )
    if cleared:
        logger.info(
            "Cleared %d stale driver_starting alert(s) before %s",
            cleared,
            log_context,
        )
    return cleared


def fetch_and_run_twin_drivers(
    token: str,
    environment_uuid: str,
    fingerprint: str,
) -> List[Dict[str, Any]]:
    """Fetch twins for the environment, match by edge fingerprint, and run drivers.

    For each twin in the environment whose ``metadata.edge_fingerprint``
    matches the local fingerprint, this function fetches the twin's asset,
    looks for a ``driver_docker_image`` key in the asset metadata, and starts
    the corresponding Docker container.

    Returns a list of result dicts with twin info and whether the container
    started successfully.
    """
    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    client = Cyberwave(base_url=base_url, api_key=token)

    # List twins for the environment via the SDK
    twins = client.twins.list(environment_id=environment_uuid)
    if not twins:
        logger.info("No twins found for environment %s", environment_uuid)
        return []

    linked_twin_uuids: set[str] = set()
    assets_by_twin_uuid: dict[str, Any] = {}
    attach_to_by_twin_uuid: dict[str, str] = {}
    camera_children_by_parent: dict[str, list[str]] = {}
    child_registry_ids_by_parent: dict[str, set[str]] = {}

    for twin in twins:
        twin_uuid = str(getattr(twin, "uuid", ""))
        if not twin_uuid:
            continue

        twin_metadata = twin.metadata if isinstance(twin.metadata, dict) else {}
        if twin_metadata.get("edge_fingerprint") != fingerprint:
            continue
        linked_twin_uuids.add(twin_uuid)

        attach_to = _resolve_attach_to_twin_uuid(client, twin, twin_metadata)
        if attach_to:
            attach_to_by_twin_uuid[twin_uuid] = attach_to

        asset_uuid = getattr(twin, "asset_uuid", None) or getattr(twin, "asset_id", "")
        if not asset_uuid:
            continue
        try:
            asset = client.assets.get(asset_uuid)
        except Exception as exc:
            logger.warning(
                "Failed to get asset %s for twin %s while collecting child twin maps: %s",
                asset_uuid,
                twin_uuid,
                exc,
            )
            continue

        assets_by_twin_uuid[twin_uuid] = asset
        if attach_to:
            child_registry_id = _get_asset_registry_id(asset)
            if child_registry_id:
                child_registry_ids_by_parent.setdefault(attach_to, set()).add(child_registry_id)
        if attach_to and _twin_has_rgb_sensor(asset):
            camera_children_by_parent.setdefault(attach_to, []).append(twin_uuid)

    child_camera_twin_uuid_set = {
        child_uuid
        for parent_uuid, child_uuids in camera_children_by_parent.items()
        if parent_uuid in linked_twin_uuids
        for child_uuid in child_uuids
    }

    # ------------------------------------------------------------------
    # Pass 1: Resolve driver images and collect per-twin launch specs.
    # ------------------------------------------------------------------

    @dataclass
    class _DriverSpec:
        twin: Any
        twin_uuid: str
        driver_image: str
        driver_params: list[str]
        child_camera_twin_uuids: list[str]
        macos_bridge_candidates: list[str]
        prefer_gpu: bool = False
        gpu_spec: str = "all"
        # Multi-container fields (None/empty for single-container mode)
        service_name: str | None = None
        command: list[str] | None = None
        service_env: dict[str, str] | None = None

    driver_specs: list[_DriverSpec] = []

    for twin in twins:
        twin_uuid = str(getattr(twin, "uuid", ""))
        if not twin_uuid:
            continue

        twin_metadata = twin.metadata if isinstance(twin.metadata, dict) else {}
        if twin_metadata.get("edge_fingerprint") != fingerprint:
            continue

        logger.info(
            "Twin '%s' (%s) is linked to this edge (fingerprint=%s)",
            twin.name,
            twin_uuid,
            fingerprint,
        )

        asset_uuid = getattr(twin, "asset_uuid", None) or getattr(twin, "asset_id", "")
        if not asset_uuid:
            # The backend stores no asset on this twin, so there is no driver
            # metadata to read and no container we could possibly spawn. Make
            # it loud — this was previously a silent ``continue`` that made
            # ``No twins with driver images matched this edge.`` unhelpful.
            logger.error(
                "Twin '%s' (%s) has no asset attached on the backend — "
                "driver startup is impossible. Attach an asset via the "
                "dashboard and restart edge-core.",
                twin.name,
                twin_uuid,
            )
            _send_alert_for_twin(
                twin_uuid,
                "Twin has no asset",
                (
                    f"Twin '{twin.name}' has no asset attached on the backend. "
                    "The edge cannot spawn a driver for this twin until an "
                    "asset is attached via the dashboard."
                ),
                "driver_start_failure",
                severity="error",
            )
            continue

        asset = assets_by_twin_uuid.get(twin_uuid)
        if asset is None:
            try:
                asset = client.assets.get(asset_uuid)
                assets_by_twin_uuid[twin_uuid] = asset
            except Exception as exc:
                logger.warning(
                    "Failed to get asset %s for twin %s: %s",
                    asset_uuid,
                    twin_uuid,
                    exc,
                )
                continue

        attach_to = attach_to_by_twin_uuid.get(twin_uuid)
        if attach_to is None:
            attach_to = _resolve_attach_to_twin_uuid(client, twin, twin_metadata)
            if attach_to:
                attach_to_by_twin_uuid[twin_uuid] = attach_to

        if twin_uuid in child_camera_twin_uuid_set and attach_to in linked_twin_uuids:
            logger.info(
                "Twin '%s' (%s) is a child camera of parent twin %s; "
                "writing JSON and skipping dedicated driver startup",
                twin.name,
                twin_uuid,
                attach_to,
            )
            _persist_twin_json_for_driver(twin, twin_uuid, asset)
            continue

        drivers = twin_metadata.get("drivers")
        asset_metadata = getattr(asset, "metadata", {}) or {}
        if not isinstance(asset_metadata, dict):
            asset_metadata = {}
        if not drivers:
            drivers = asset_metadata.get("drivers")
            if not drivers:
                if attach_to:
                    logger.info(
                        "Twin '%s' has no driver but is attached to %s; "
                        "writing JSON for parent driver to use",
                        twin.name,
                        attach_to,
                    )
                    _persist_twin_json_for_driver(twin, twin_uuid, asset)
                    continue

                logger.warning("No drivers specified in asset metadata for twin '%s'", twin.name)
                _send_alert_for_twin(
                    twin_uuid,
                    "No drivers specified",
                    f"No drivers specified in asset metadata for twin '{twin.name}'",
                    "driver_start_failure",
                    severity="error",
                )
                raise ValueError(
                    f"No drivers specified in asset metadata for paired twin '{twin.name}'"
                )
            else:
                logger.warning(
                    (
                        "No drivers specified in twin metadata for twin '%s', "
                        "found drivers in asset metadata"
                    ),
                    twin.name,
                )
        _persist_twin_json_for_driver(twin, twin_uuid, asset)

        child_camera_uuids = list(dict.fromkeys(camera_children_by_parent.get(twin_uuid, [])))
        if child_camera_uuids:
            logger.info(
                "Passing %d child camera twin UUID(s) to parent twin '%s': %s",
                len(child_camera_uuids),
                twin.name,
                ",".join(child_camera_uuids),
            )

        macos_bridge_candidates: list[str] = []
        if platform.system() == "Darwin":
            macos_bridge_candidates = _resolve_macos_camera_bridge_candidates(asset, twin_metadata)
            if macos_bridge_candidates:
                logger.info(
                    "Resolved %d macOS camera bridge candidate(s) for twin '%s': %s",
                    len(macos_bridge_candidates),
                    twin.name,
                    ",".join(macos_bridge_candidates),
                )

        # --- Multi-container mode: services array -----------------------
        multi = _get_driver_services(
            drivers,
            child_registry_ids=child_registry_ids_by_parent.get(twin_uuid, set()),
        )
        if multi is not None:
            svc_specs, shared_env, shared_params = multi
            logger.info(
                "Multi-container mode for twin '%s': %d service(s)",
                twin.name,
                len(svc_specs),
            )
            for svc in svc_specs:
                svc_image = _resolve_driver_image_tag(svc.image)
                svc_image = _maybe_rewrite_jetson_tag(svc_image, twin.name)
                merged_env = {**shared_env, **svc.env}
                merged_params = shared_params + svc.params
                driver_specs.append(_DriverSpec(
                    twin=twin,
                    twin_uuid=twin_uuid,
                    driver_image=svc_image,
                    driver_params=merged_params,
                    child_camera_twin_uuids=child_camera_uuids,
                    macos_bridge_candidates=macos_bridge_candidates,
                    prefer_gpu=svc.prefer_gpu,
                    gpu_spec=svc.gpu_spec,
                    service_name=svc.name,
                    command=svc.command,
                    service_env=merged_env,
                ))
            continue

        # --- Single-container mode (existing path) ----------------------
        driver_image, driver_params, driver_prefer_gpu, driver_gpu_spec = (
            _get_best_driver_image_and_params(
                drivers,
                child_registry_ids=child_registry_ids_by_parent.get(twin_uuid, set()),
            )
        )

        _driver_overrides = load_driver_overrides()
        if twin_uuid in _driver_overrides:
            override_image = _driver_overrides[twin_uuid]
            logger.info(
                "Applying local driver override for twin '%s': %s -> %s",
                twin.name,
                driver_image,
                override_image,
            )
            driver_image = override_image
            driver_params = []

        if not driver_image:
            logger.info("No driver_docker_image in asset metadata for twin '%s'", twin.name)
            _send_alert_for_twin(
                twin_uuid,
                "No driver_docker_image in asset metadata",
                f"No driver_docker_image in asset metadata for twin '{twin.name}'",
                "driver_start_failure",
                severity="error",
            )
            raise ValueError(
                f"No drivers specified in asset metadata for paired twin '{twin.name}'"
            )

        driver_image = _resolve_driver_image_tag(driver_image)
        driver_image = _maybe_rewrite_jetson_tag(driver_image, twin.name)

        driver_specs.append(_DriverSpec(
            twin=twin,
            twin_uuid=twin_uuid,
            driver_image=driver_image,
            driver_params=driver_params,
            child_camera_twin_uuids=child_camera_uuids,
            macos_bridge_candidates=macos_bridge_candidates,
            prefer_gpu=driver_prefer_gpu,
            gpu_spec=driver_gpu_spec,
        ))

    # ------------------------------------------------------------------
    # Pass 1a: Drop orphan ``driver_starting`` alerts left by interrupted
    #          prior attempts (watchdog kill, crash loop, OOM, etc.).
    #          Fresh alerts are created with ``force=True`` in pass 1b, so
    #          without this cleanup the dashboard keeps stale "Downloading
    #          driver image …" banners after a successful recovery boot.
    # ------------------------------------------------------------------

    _clear_stale_driver_starting_alerts(
        (spec.twin_uuid for spec in driver_specs),
        log_context="driver startup",
    )

    # ------------------------------------------------------------------
    # Pass 1b: Create driver_starting alerts *before* the pull so the
    #          user sees "Downloading driver image …" during the actual
    #          download, not only after it finishes.
    # ------------------------------------------------------------------

    alert_by_spec_index: dict[int, DriverStartingAlertContext] = {}
    for idx, spec in enumerate(driver_specs):
        ctx = DriverStartingAlertContext(twin_uuid=spec.twin_uuid, image=spec.driver_image)
        ctx.create()
        alert_by_spec_index[idx] = ctx

    # ------------------------------------------------------------------
    # Pass 2: Pull all unique driver images in parallel.
    # ------------------------------------------------------------------

    if driver_specs:
        images_to_pull = [spec.driver_image for spec in driver_specs]
        pull_results = _pull_driver_images_parallel(images_to_pull)
    else:
        pull_results = {}

    # Update alert metadata now that the pull phase has finished so the
    # dashboard shows "pull_complete" while the container is being created.
    for idx, spec in enumerate(driver_specs):
        ctx = alert_by_spec_index.get(idx)
        if ctx is None:
            continue
        if pull_results.get(spec.driver_image, False):
            ctx.update_metadata({"phase": "pull_complete"}, force=True)

    # If a Jetson-rewritten tag failed to pull, fall back to the
    # original (non-jetson) tag and re-pull.
    from .driver_selection import is_jetson as _is_jetson_platform

    if _is_jetson_platform():
        fallback_needed: list[tuple[_DriverSpec, str]] = []
        for spec in driver_specs:
            if not pull_results.get(spec.driver_image, False) and ":" in spec.driver_image:
                repo, tag = spec.driver_image.rsplit(":", 1)
                if tag.startswith("jetson-"):
                    original_tag = tag[len("jetson-"):]
                    original_image = f"{repo}:{original_tag}"
                    logger.warning(
                        "[Jetson] jetson-prefixed image %s not found — "
                        "falling back to %s for twin '%s'",
                        spec.driver_image,
                        original_image,
                        spec.twin.name,
                    )
                    fallback_needed.append((spec, original_image))

        if fallback_needed:
            fallback_images = [img for _, img in fallback_needed]
            fallback_results = _pull_driver_images_parallel(fallback_images)
            for spec, original_image in fallback_needed:
                if fallback_results.get(original_image, False):
                    pull_results[original_image] = True
                    spec.driver_image = original_image

    # ------------------------------------------------------------------
    # Pass 3: Start containers (images are already local).
    # ------------------------------------------------------------------

    results: List[Dict[str, Any]] = []

    for idx, spec in enumerate(driver_specs):
        alert_ctx = alert_by_spec_index.get(idx)

        if not pull_results.get(spec.driver_image, False):
            logger.error(
                "Skipping driver for twin '%s' — image %s not available",
                spec.twin.name,
                spec.driver_image,
            )
            if alert_ctx is not None:
                alert_ctx.mark_failed_and_resolve(
                    f"Driver image '{spec.driver_image}' could not be pulled for twin "
                    f"'{spec.twin.name}'.",
                    phase="pull_failed",
                )
            _send_alert_for_twin(
                spec.twin_uuid,
                "Driver image not available",
                f"Driver image '{spec.driver_image}' could not be pulled for twin "
                f"'{spec.twin.name}'. Troubleshooting: {DRIVER_TROUBLESHOOTING_URL}",
                "driver_start_failure",
                severity="error",
            )
            fail_entry: Dict[str, Any] = {
                "twin_uuid": spec.twin_uuid,
                "twin_name": spec.twin.name,
                "driver_image": spec.driver_image,
                "success": False,
            }
            if spec.service_name:
                fail_entry["service_name"] = spec.service_name
            results.append(fail_entry)
            continue

        logger.info(
            "Starting driver container %s%s for twin '%s'",
            spec.driver_image,
            f" (service={spec.service_name})" if spec.service_name else "",
            spec.twin.name,
        )
        try:
            success = _run_docker_image(
                spec.driver_image,
                spec.driver_params,
                twin_uuid=spec.twin_uuid,
                token=token,
                child_camera_twin_uuids=spec.child_camera_twin_uuids,
                macos_bridge_device_candidates=spec.macos_bridge_candidates,
                skip_pull=True,
                prefer_gpu=spec.prefer_gpu,
                gpu_spec=spec.gpu_spec,
                service_name=spec.service_name,
                command=spec.command,
                service_env=spec.service_env,
                driver_alert_ctx=alert_ctx,
            )
            result_entry: Dict[str, Any] = {
                "twin_uuid": spec.twin_uuid,
                "twin_name": spec.twin.name,
                "driver_image": spec.driver_image,
                "success": success,
            }
            if spec.service_name:
                result_entry["service_name"] = spec.service_name
            results.append(result_entry)
            if not success:
                try:
                    startup_failure_message = (
                        f"Driver image '{spec.driver_image}' for twin '{spec.twin.name}' "
                        "failed to start on this edge. Check that required hardware is "
                        f"connected and accessible. Troubleshooting: {DRIVER_TROUBLESHOOTING_URL}"
                    )
                    _send_alert_for_twin(
                        spec.twin_uuid,
                        "Driver failed to start",
                        startup_failure_message,
                        "driver_start_failure",
                        severity="error",
                    )
                except Exception as alert_exc:
                    logger.warning(
                        "Could not send startup-failure alert for twin %s: %s",
                        spec.twin_uuid,
                        alert_exc,
                    )
        except Exception as exc:
            _send_alert_for_twin(
                spec.twin_uuid,
                "Failed to run driver docker image",
                f"Failed to run driver docker image for twin '{spec.twin.name}': {exc}",
                "driver_start_failure",
                severity="error",
            )
            logger.error(
                "Failed to run driver docker image %s for twin '%s': %s",
                spec.driver_image,
                spec.twin.name,
                exc,
            )

    # Worker start has been moved to ``run_startup_checks``. It now runs
    # *after* ``_sync_workers_for_twins`` so that the workers/ directory
    # reflects the currently-active workflows for the twins the operator
    # actually selected (env.json) — rather than firing against a stale
    # snapshot left over from a previous activation. This gates the
    # ``cyberwaveos/edge-ml-worker`` image pull on the presence of active
    # workflows (CYB-1766).
    return results


def _wait_for_driver_readiness(
    twin_uuids: list[str],
    *,
    timeout_seconds: float = 30.0,
    poll_interval: float = 2.0,
) -> dict[str, str]:
    """Wait for all expected driver containers to reach a running state.

    Returns a mapping of ``container_name -> status`` for each driver.
    Containers that do not exist (e.g. child-camera twins with no dedicated
    driver) are silently skipped.
    """
    expected_containers: dict[str, str] = {}
    for tu in twin_uuids:
        container_name = f"{DRIVER_CONTAINER_PREFIX}{tu[:8]}"
        if container_name in expected_containers:
            logger.warning(
                "UUID prefix collision: twins %s and %s both map to container %s",
                expected_containers[container_name][:12],
                tu[:12],
                container_name,
            )
        expected_containers[container_name] = tu
    if not expected_containers:
        return {}

    # Identify which containers actually exist (some twins are children and
    # share a parent's driver, so they have no dedicated container).
    all_driver_names = set(_list_driver_containers(include_stopped=True))
    relevant = {
        name: tu
        for name, tu in expected_containers.items()
        if name in all_driver_names
    }
    if not relevant:
        logger.debug("No driver containers found for twin UUIDs; skipping readiness wait")
        return {}

    deadline = time.time() + timeout_seconds
    final_statuses: dict[str, str] = {}

    while time.time() < deadline:
        pending = []
        for container_name in relevant:
            if container_name in final_statuses:
                continue
            status = _get_container_status_fast(container_name)
            if status == "running":
                final_statuses[container_name] = "running"
            elif status in {"exited", "dead", "restarting"}:
                final_statuses[container_name] = status
            else:
                pending.append(container_name)

        if not pending:
            break
        time.sleep(poll_interval)

    for container_name in relevant:
        if container_name not in final_statuses:
            final_statuses[container_name] = "timeout"

    healthy = [n for n, s in final_statuses.items() if s == "running"]
    unhealthy = {n: s for n, s in final_statuses.items() if s != "running"}

    logger.info(
        "Driver readiness check complete: %d/%d healthy",
        len(healthy),
        len(final_statuses),
    )
    for name in healthy:
        logger.info("  ✓ %s (twin %s): running", name, relevant[name][:8])
    for name, status in unhealthy.items():
        logger.warning("  ✗ %s (twin %s): %s", name, relevant[name][:8], status)

    return final_statuses


def _get_container_status_fast(container_name: str) -> str:
    """Return the container status string without a full inspect."""
    try:
        result = subprocess.run(
            [
                "docker", "inspect",
                "--format", "{{.State.Status}}",
                container_name,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip().lower()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return "none"


def _start_worker_after_drivers(
    *,
    token: str,
    environment_uuid: str,
    twin_uuids: list[str],
) -> None:
    """Start the worker container after verifying driver readiness.

    Waits for all driver containers to reach a stable state (running or
    failed), pre-downloads required ML models, then starts the worker.
    Proceeds even if some drivers are unhealthy so that healthy cameras
    can still be utilized.

    Callers are expected to invoke this *after* ``_sync_workers_for_twins``
    so that ``WorkerManager.start()`` sees the up-to-date set of
    ``wf_*.py`` files and can correctly skip the image pull when no active
    workflows exist (CYB-1766).
    """
    try:
        logger.info(
            "Preparing worker startup with %d twin(s): %s",
            len(twin_uuids),
            ", ".join(tu[:8] + "..." for tu in twin_uuids) if twin_uuids else "(none)",
        )

        driver_statuses = _wait_for_driver_readiness(twin_uuids)
        unhealthy = {n: s for n, s in driver_statuses.items() if s != "running"}
        if unhealthy:
            logger.warning(
                "Starting worker despite %d unhealthy driver(s): %s",
                len(unhealthy),
                ", ".join(f"{n}={s}" for n, s in unhealthy.items()),
            )

        from .model_manager import ModelManager
        from .worker_manager import WorkerManager, resolve_worker_image

        workers_dir = CONFIG_DIR / "workers"
        models_dir = CONFIG_DIR / "models"
        if workers_dir.is_dir():
            model_ids = ModelManager.scan_worker_model_ids(workers_dir)
            if model_ids:
                logger.info("Pre-downloading %d model(s) before worker startup: %s", len(model_ids), model_ids)
                base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
                mm = ModelManager(cache_dir=models_dir, api_token=token, base_url=base_url)
                mm.ensure_models(model_ids)
                _send_model_failure_alerts(
                    twin_uuids=twin_uuids,
                    failures=mm.last_ensure_failures,
                )

        worker_manager = WorkerManager(
            config_dir=CONFIG_DIR,
            environment_uuid=environment_uuid,
            token=token,
            twin_uuids=twin_uuids,
            image=resolve_worker_image(),
            resource_limits=load_worker_resource_limits(),
        )
        worker_manager.start()
    except Exception as exc:
        logger.warning("Failed to start worker container after driver startup: %s", exc)
        _send_worker_start_failure_alerts(twin_uuids=twin_uuids, error=str(exc))


def _send_alert_for_twin(
    twin_uuid: str,
    alert_title: str,
    alert_description: str,
    alert_type: str,
    severity: str = "warning",
) -> None:
    """
    Send an alert to the twin.
    """
    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    client = Cyberwave(base_url=base_url, api_key=load_token())
    twin = client.twin(twin_id=twin_uuid)
    # Create an alert
    twin.alerts.create(
        name=alert_title,
        description=alert_description,
        severity=severity,  # info | warning | error | critical
        alert_type=alert_type,
        source_type="edge",  # edge | cloud | workflow
    )


def _send_model_failure_alerts(
    *,
    twin_uuids: list[str],
    failures: dict[str, str],
) -> None:
    """Send a technical alert for each model that failed to download.

    Alerts are sent to every twin associated with the current edge so
    that operators can see the failure in each twin's alert feed.
    """
    if not failures:
        return
    for model_id, error_msg in failures.items():
        for twin_uuid in twin_uuids:
            try:
                _send_alert_for_twin(
                    twin_uuid,
                    f"Model not available: {model_id}",
                    (
                        f"Failed to download model '{model_id}' required by an "
                        f"active workflow. The worker will start without this "
                        f"model; inference steps that depend on it will be "
                        f"skipped. Error: {error_msg}"
                    ),
                    "model_download_failure",
                    severity="warning",
                )
            except Exception as alert_exc:
                logger.debug(
                    "Failed to send model-download alert for twin %s, model %s: %s",
                    twin_uuid,
                    model_id,
                    alert_exc,
                )


def _send_worker_start_failure_alerts(
    *,
    twin_uuids: list[str],
    error: str = "",
) -> None:
    """Send a technical alert when the worker container fails to start."""
    detail = f" Error: {error}" if error else ""
    for twin_uuid in twin_uuids:
        try:
            _send_alert_for_twin(
                twin_uuid,
                "Worker container failed to start",
                (
                    f"The edge ML worker container could not be started. "
                    f"Workflows will not run until this is resolved.{detail}"
                ),
                "worker_start_failure",
                severity="warning",
            )
        except Exception as alert_exc:
            logger.debug(
                "Failed to send worker-start-failure alert for twin %s: %s",
                twin_uuid,
                alert_exc,
            )


# Re-exported from driver_selection for backward compat.
from .driver_selection import _get_best_driver_image_and_params as _get_best_driver_image_and_params  # noqa: E402
from .driver_selection import _get_driver_services as _get_driver_services  # noqa: E402


def register_edge(token: str) -> bool:
    fingerprint = get_or_create_fingerprint()
    if not fingerprint:
        logger.warning("Could not load or create edge fingerprint")
        return False

    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    logger.info("Registering edge with fingerprint=%s at %s", fingerprint, base_url)
    try:
        client = Cyberwave(base_url=base_url, api_key=token)
        edge = client.edges.create(
            fingerprint=fingerprint,
        )
        if edge:
            logger.info("Edge registered successfully")
        else:
            logger.warning("Edge registration returned falsy response")
        return bool(edge)
    except Exception as exc:
        logger.warning("Edge registration failed: %s: %s", type(exc).__name__, exc)
        return False


# Period for the REST keepalive that bumps ``Edge.last_seen_at``. The
# backend's "Standby" window is 90 s, so a 30 s cadence tolerates two
# missed posts before the row drops to "Offline" — the same 3-strikes
# rule we use elsewhere (e.g. MQTT debounce).
HOST_FACTS_KEEPALIVE_PERIOD_SECONDS = 30.0

# Module-level singletons so repeat callers don't spawn duplicate
# threads. The thread is a daemon so process exit doesn't hang on it.
_HOST_FACTS_KEEPALIVE_THREAD: Optional[threading.Thread] = None
_HOST_FACTS_KEEPALIVE_STOP: Optional[threading.Event] = None
_HOST_FACTS_KEEPALIVE_LOCK = threading.Lock()


def _post_host_facts_once(token: str) -> bool:
    """Single ``POST /api/v1/edges/discover`` with current host facts.

    Returns ``True`` when the call succeeded. Failure is non-fatal at
    every call site: we just lose one keepalive cycle (the dashboard
    will hold "Online/Standby" within its window) and the next 30 s
    tick retries.
    """
    try:
        from cyberwave.edge.host_metrics import read_host_facts
    except Exception as exc:
        logger.debug("Host facts uploader: read_host_facts unavailable: %s", exc)
        return False

    fingerprint = get_or_create_fingerprint()
    if not fingerprint:
        logger.debug("Host facts uploader: missing fingerprint")
        return False

    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    try:
        facts = read_host_facts().to_dict()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Host facts uploader: read_host_facts() raised: %s", exc)
        return False

    hostname = socket.gethostname() or ""
    try:
        plat = f"{platform.system()}-{platform.machine()}".strip("-")
    except Exception:
        plat = ""

    try:
        client = Cyberwave(base_url=base_url, api_key=token)
        client.edges.discover(
            fingerprint=fingerprint,
            hostname=hostname,
            platform=plat,
            host_facts=facts,
        )
        logger.debug(
            "Uploaded host_facts (%d keys) for edge fingerprint=%s",
            len(facts),
            fingerprint,
        )
        return True
    except Exception as exc:
        logger.warning(
            "Host facts upload failed (%s: %s); will retry on next keepalive tick",
            type(exc).__name__,
            exc,
        )
        return False


def _start_host_facts_keepalive(
    token: str,
    *,
    period_seconds: float = HOST_FACTS_KEEPALIVE_PERIOD_SECONDS,
    fire_immediately: bool = True,
) -> bool:
    """Spawn (once) a daemon thread that POSTs host facts to ``/discover``.

    The thread fires its first tick **immediately** when
    ``fire_immediately=True`` (the default), then every ``period_seconds``
    after that. Returns ``True`` synchronously as soon as the thread is
    scheduled — the caller never waits for the first POST.

    Doing the bootstrap POST in-thread (instead of synchronously on the
    caller's stack) is critical for systemd ``Type=notify`` units: the
    boot path can fire ``READY=1`` before any network I/O completes, so
    ``systemctl restart`` returns in seconds rather than waiting for the
    backend round-trip — important on slow / flaky links where the POST
    can otherwise sit on its TCP timeout for tens of seconds and push
    the unit past ``TimeoutStartSec``.

    This is also what powers the "Standby" state on the dashboard: the
    backend bumps ``Edge.last_seen_at`` on every ``/discover`` call, and
    the keepalive guarantees that signal stays fresh even when an edge
    has no bound twins (so no MQTT ``edge_health`` is being published).

    Idempotent — repeat calls return immediately if a keepalive is
    already running.
    """
    global _HOST_FACTS_KEEPALIVE_THREAD, _HOST_FACTS_KEEPALIVE_STOP

    with _HOST_FACTS_KEEPALIVE_LOCK:
        existing = _HOST_FACTS_KEEPALIVE_THREAD
        if existing is not None and existing.is_alive():
            return True

        stop_event = threading.Event()
        _HOST_FACTS_KEEPALIVE_STOP = stop_event

        def _loop() -> None:
            # Bootstrap tick: do the first POST on entry when requested.
            # We swallow exceptions both here and on subsequent ticks —
            # a failed upload is recoverable on the next iteration, and
            # propagating would kill the daemon thread silently.
            if fire_immediately:
                try:
                    _post_host_facts_once(token)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Host facts bootstrap tick raised: %s: %s",
                        type(exc).__name__,
                        exc,
                    )

            while not stop_event.wait(timeout=period_seconds):
                try:
                    _post_host_facts_once(token)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Host facts keepalive tick raised: %s: %s",
                        type(exc).__name__,
                        exc,
                    )

        thread = threading.Thread(
            target=_loop,
            name="cyberwave-edge-core.host-facts-keepalive",
            daemon=True,
        )
        _HOST_FACTS_KEEPALIVE_THREAD = thread
        thread.start()

    logger.info(
        "Started host-facts keepalive (period=%.0fs, fire_immediately=%s) — "
        "Edge.last_seen_at will stay fresh independent of twin MQTT activity",
        period_seconds,
        fire_immediately,
    )
    return True


def _stop_host_facts_keepalive() -> None:
    """Signal the keepalive thread to stop. Test/teardown only."""
    global _HOST_FACTS_KEEPALIVE_THREAD, _HOST_FACTS_KEEPALIVE_STOP
    with _HOST_FACTS_KEEPALIVE_LOCK:
        stop_event = _HOST_FACTS_KEEPALIVE_STOP
        thread = _HOST_FACTS_KEEPALIVE_THREAD
        _HOST_FACTS_KEEPALIVE_STOP = None
        _HOST_FACTS_KEEPALIVE_THREAD = None
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)


def _upload_host_facts_on_startup(token: str) -> bool:
    """Schedule background host-facts upload and the 30 s REST keepalive.

    Called once near the end of :func:`run_startup_checks`. Returns as
    soon as the keepalive thread is *scheduled* — no network I/O happens
    on the caller's stack. The first ``/discover`` POST is performed by
    the keepalive thread itself on entry (see
    :func:`_start_host_facts_keepalive`), and then again every
    ``HOST_FACTS_KEEPALIVE_PERIOD_SECONDS``.

    Why off-stack:

    - Under systemd ``Type=notify``, ``run_startup_checks`` runs before
      ``READY=1`` is sent. A blocking REST POST against a slow or flaky
      backend can push past ``TimeoutStartSec`` and cause
      ``systemctl restart`` to time out with a confusing "job failed"
      error. Doing the bootstrap upload from a daemon thread means the
      service signals ready in seconds even when the network is misbehaving.
    - The previous synchronous-then-keepalive pattern had no upside —
      both paths POSTed the exact same payload to the same endpoint,
      so collapsing them into one in-thread bootstrap loses nothing.

    The function always returns ``True`` because the upload outcome is
    not known synchronously. Visibility of the upload result is via the
    keepalive thread's logs.
    """
    _start_host_facts_keepalive(token, fire_immediately=True)
    return True


def _build_cyberwave_client(token: str) -> Cyberwave:
    """Create a configured SDK client using runtime environment settings."""
    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    return Cyberwave(base_url=base_url, api_key=token)


def _resolve_edge_for_fingerprint(client: Cyberwave, fingerprint: str) -> Optional[Any]:
    """Resolve the current edge record by fingerprint, creating it if needed."""
    try:
        for edge in client.edges.list():
            if getattr(edge, "fingerprint", None) == fingerprint:
                return edge
    except Exception as exc:
        logger.warning("Failed to list edges while resolving fingerprint %s: %s", fingerprint, exc)

    try:
        return client.edges.create(fingerprint=fingerprint)
    except Exception as exc:
        logger.warning(
            "Failed to create edge while resolving fingerprint %s: %s",
            fingerprint,
            exc,
        )
        return None


def _stop_worker_container_for_restart() -> None:
    """Best-effort graceful stop of the worker container before a full edge restart.

    Calls :meth:`WorkerManager.stop`, which performs a ``docker stop``
    (not ``docker rm``) — the container is left in ``exited`` state so
    operators can ``docker logs`` it and the symmetric
    :func:`_start_worker_container_after_restart` brings it back up on
    a clean ``docker run`` cycle.
    """
    try:
        from .worker_manager import WorkerManager

        environment_uuid = load_environment_uuid() or ""
        if not environment_uuid:
            return
        token = load_token()
        if not token:
            return
        worker_manager = WorkerManager(
            config_dir=CONFIG_DIR,
            environment_uuid=environment_uuid,
            token=token,
        )
        worker_manager.stop()
    except Exception as exc:
        logger.warning("Failed to stop worker container before restart: %s", exc)


def _start_worker_container_after_restart(
    token: str,
    environment_uuid: str,
    fingerprint: str,
) -> bool:
    """Best-effort start of the worker container after a full edge restart.

    Symmetric counterpart to :func:`_stop_worker_container_for_restart`.

    Without this step every edge-core restart command — for example one
    triggered by an admin "Restart edge" action or an operator
    deactivate/reactivate cycle — would tear the worker container down
    via :func:`_stop_worker_container_for_restart` and leave it down
    until :func:`reconcile_worker_lifecycle` ticked again on the runtime
    loop (~5 minutes by default, ``CYBERWAVE_WORKER_SYNC_INTERVAL_LOOPS``
    × loop period). During that window every workflow-driven feature
    that depends on the worker (frame filter, ML inference, detection
    overlays, etc.) silently degrades — the camera driver visibly logs
    ``[FRAME_FILTER] 100.0% of N frames in last 30 s emitted blank
    fallback ... Worker likely down or not publishing on this channel``
    while the user has no immediate way to recover other than waiting.

    The Zenoh router is already restarted on the symmetric path
    (:func:`stop_zenoh_router` / :func:`start_zenoh_router`); this
    function closes the analogous gap for workers.

    Behaves as a no-op when there are no active worker files in
    ``CONFIG_DIR/workers/`` — matching the steady-state semantics of
    :func:`reconcile_worker_lifecycle`, which deliberately leaves the
    worker stopped when no workflows are linked to avoid pulling the
    ``cyberwaveos/edge-ml-worker`` image needlessly (CYB-1766).

    Returns ``True`` when the worker was started, ``False`` when the
    start was skipped (no worker files) or when the start failed
    (best-effort: failure is logged at WARNING and does not propagate
    so the calling restart flow still completes successfully — the
    runtime loop's :func:`reconcile_worker_lifecycle` will retry on the
    next tick, which is exactly the recovery path that exists today).
    """
    try:
        workers_dir = CONFIG_DIR / "workers"
        has_active_workers = workers_dir.is_dir() and any(workers_dir.glob("*.py"))
        if not has_active_workers:
            logger.debug(
                "Skipping worker container start after restart: no active workflow files in %s",
                workers_dir,
            )
            return False

        try:
            twin_uuids = _resolve_worker_sync_twin_uuids(token, environment_uuid, fingerprint)
        except Exception:
            logger.warning(
                "Could not resolve twins for worker container start after "
                "restart; runtime loop reconcile will retry shortly",
                exc_info=True,
            )
            return False

        _start_worker_after_drivers(
            token=token,
            environment_uuid=environment_uuid,
            twin_uuids=twin_uuids,
        )
        return True
    except Exception as exc:
        logger.warning(
            "Failed to start worker container after restart "
            "(runtime loop will retry on next reconcile): %s",
            exc,
        )
        return False


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp string for ``Alert.metadata`` annotations.

    Centralised so the edge-core side matches the backend (which uses
    ``timezone.now().isoformat()``).  Mixing unix-epoch floats with
    ISO strings inside the same ``metadata`` bag would force every
    downstream reader (UI, analytics, CLI) to handle both — keep it
    one shape, the ISO string.
    """
    return datetime.now(tz=timezone.utc).isoformat()


def _perform_edge_core_restart(
    token: str,
    *,
    restart_alert: EdgeCoreRestartAlertContext | None = None,
) -> dict[str, Any]:
    """Execute restart workflow: cleanup local state and re-run driver startup.

    ``restart_alert`` is the lifecycle alert created by the backend.
    When present, this function transitions it to ``in_progress`` at
    entry and ``completed`` (with resolve) on a successful return.  The
    ``failed`` transition is owned by the caller
    (:func:`_run_edge_core_restart_worker`) because it needs the
    exception object for the metadata annotation.
    """
    if restart_alert is not None:
        restart_alert.transition(
            EDGE_CORE_RESTART_PHASE_IN_PROGRESS,
            resolve=False,
            extra_metadata={"in_progress_at": _utc_now_iso()},
        )

    # Capture the twins whose driver containers we are about to tear down so
    # we can clear any in-flight ``driver_starting`` alerts that would
    # otherwise be orphaned by the restart.  ``_stop_and_prune_driver_containers``
    # mutates ``_CONTAINER_TWIN_MAP`` in place, so we snapshot it first.
    twin_uuids_to_clear = {
        twin_uuid for twin_uuid in _CONTAINER_TWIN_MAP.values() if twin_uuid
    }

    _stop_worker_container_for_restart()
    removed_json_files = _remove_cached_twin_json_files()
    removed_containers = _stop_and_prune_driver_containers()

    _clear_stale_driver_starting_alerts(
        twin_uuids_to_clear,
        log_context="edge-core restart",
    )

    environment_uuid = load_environment_uuid(retries=5, retry_delay_seconds=0.2)

    # Stop the Zenoh router container so it is restarted cleanly alongside
    # the driver containers.  Best-effort: failure does not block the restart.
    if environment_uuid:
        stop_zenoh_router(environment_uuid)

    if not environment_uuid:
        logger.warning("No linked environment found; restart completed with cleanup only")
        return {
            "environment_uuid": None,
            "removed_twin_json_files": removed_json_files,
            "removed_driver_containers": removed_containers,
            "drivers_started": 0,
            "drivers_discovered": 0,
        }

    fingerprint = get_or_create_fingerprint()
    if not fingerprint:
        raise RuntimeError("Could not load or create edge fingerprint")

    # Re-start Zenoh router before driver containers if configured.
    zenoh_cfg = _get_zenoh_config()
    if zenoh_cfg.router_enabled:
        router_ok = start_zenoh_router(zenoh_cfg, environment_uuid)
        if not router_ok:
            logger.warning(
                "Zenoh router failed to start during restart; "
                "driver containers will use peer-to-peer discovery"
            )

    results = fetch_and_run_twin_drivers(token, environment_uuid, fingerprint)
    started = sum(1 for result in results if result.get("success"))

    if started > 0:
        _stop_bootstrap_edge_health_publisher()

    worker_started = _start_worker_container_after_restart(
        token, environment_uuid, fingerprint
    )

    summary = {
        "environment_uuid": environment_uuid,
        "removed_twin_json_files": removed_json_files,
        "removed_driver_containers": removed_containers,
        "drivers_started": started,
        "drivers_discovered": len(results),
        "worker_started": worker_started,
    }
    logger.info(
        "Edge-core restart complete: env=%s removed_json=%d removed_containers=%d "
        "started=%d/%d worker_started=%s",
        environment_uuid,
        len(removed_json_files),
        len(removed_containers),
        started,
        len(results),
        worker_started,
    )
    return summary


def _run_edge_core_restart_worker(
    request_id: str, alert_uuid: Optional[str] = None
) -> None:
    """Execute restart flow in a background thread.

    Owns the terminal phase transitions for the ``edge_core_restart``
    alert created by the backend:

    - ``completed`` (with resolve) on a clean return.
    - ``failed``    (with resolve) when ``_perform_edge_core_restart``
      raises.  The exception text is recorded under
      ``metadata.error`` so operators can diagnose without combing
      through journalctl.
    - ``failed``    (with resolve, ``metadata.reason='concurrent_restart'``)
      when another restart is already in flight in this process — without
      this branch the new alert would orphan in ``requested`` until the
      backend reaper times it out (5 min), confusing operators staring at
      what looks like a stuck restart.

    The ``in_progress`` transition lives inside
    :func:`_perform_edge_core_restart` itself so it fires *after* we
    have committed to doing work but *before* any side-effects, which
    is the right semantic for "I have started".
    """
    global _EDGE_RESTART_IN_PROGRESS

    # Build the alert context before we touch the lock so we can use it
    # in the "already in progress" branch below without holding the
    # lock across HTTP calls (transition() reaches out to the backend).
    restart_alert = EdgeCoreRestartAlertContext(alert_uuid=alert_uuid)

    with _EDGE_RESTART_LOCK:
        already_in_progress = _EDGE_RESTART_IN_PROGRESS
        if not already_in_progress:
            _EDGE_RESTART_IN_PROGRESS = True

    if already_in_progress:
        logger.info(
            "Ignoring restart request %s: restart already in progress",
            request_id or "no-request-id",
        )
        restart_alert.transition(
            EDGE_CORE_RESTART_PHASE_FAILED,
            resolve=True,
            extra_metadata={
                "reason": "concurrent_restart",
                "error": "another edge-core restart was already in progress",
                "failed_at": _utc_now_iso(),
            },
        )
        return

    try:
        token = load_token()
        if not token:
            logger.warning(
                "Ignoring restart request %s: no token available",
                request_id or "no-request-id",
            )
            # Without a token we cannot transition the alert either —
            # the backend reaper will eventually time it out.
            return
        try:
            _perform_edge_core_restart(token, restart_alert=restart_alert)
        except Exception as exc:
            logger.exception(
                "Edge-core restart request %s failed",
                request_id or "no-request-id",
            )
            restart_alert.transition(
                EDGE_CORE_RESTART_PHASE_FAILED,
                resolve=True,
                extra_metadata={
                    "error": str(exc)[:500],
                    "failed_at": _utc_now_iso(),
                },
            )
            return
        restart_alert.transition(
            EDGE_CORE_RESTART_PHASE_COMPLETED,
            resolve=True,
            extra_metadata={"completed_at": _utc_now_iso()},
        )
    finally:
        with _EDGE_RESTART_LOCK:
            _EDGE_RESTART_IN_PROGRESS = False


def _handle_edge_command_message(*args: Any) -> None:
    """Handle MQTT command message for this edge."""
    if len(args) == 1:
        payload = args[0]
    elif len(args) >= 2:
        payload = args[1]
    else:
        return

    if not isinstance(payload, dict):
        logger.warning("Ignoring edge command with non-dict payload: %r", payload)
        return

    command = str(payload.get("command", "")).strip().lower()
    if command != EDGE_COMMAND_RESTART:
        return

    request_id = str(payload.get("request_id", "")).strip()
    if request_id:
        if request_id in _HANDLED_EDGE_COMMAND_REQUEST_IDS:
            return
        _HANDLED_EDGE_COMMAND_REQUEST_IDS.add(request_id)

    # ``alert_uuid`` is the lifecycle alert created by the backend in
    # ``POST /api/v1/edges/{uuid}/restart-core``.  Missing when the
    # restart was triggered by something other than that endpoint
    # (direct CLI publish, smoke-test harness, …) — the worker treats
    # the missing UUID as a no-op for alert updates.
    alert_uuid = str(payload.get("alert_uuid", "")).strip() or None

    logger.info(
        "Received edge restart command request_id=%s alert=%s",
        request_id or "none",
        alert_uuid or "none",
    )
    worker = threading.Thread(
        target=_run_edge_core_restart_worker,
        args=(request_id, alert_uuid),
        name=f"edge-core-restart-{(request_id or 'no-id')[:12]}",
        daemon=True,
    )
    worker.start()


def _resolve_edge_command_topic(token: str) -> Optional[str]:
    """Resolve the MQTT topic used for edge command messages."""
    fingerprint = get_or_create_fingerprint()
    if not fingerprint:
        logger.warning("Cannot subscribe to edge commands: edge fingerprint unavailable")
        return None

    client = _build_cyberwave_client(token)
    edge = _resolve_edge_for_fingerprint(client, fingerprint)
    if not edge:
        return None

    edge_uuid = str(getattr(edge, "uuid", "") or "")
    if not edge_uuid:
        logger.warning("Cannot subscribe to edge commands: resolved edge has no UUID")
        return None

    mqtt_client = _get_shared_mqtt_client(token)
    if not mqtt_client:
        logger.warning("Cannot subscribe to edge commands: MQTT client unavailable")
        return None

    return f"{mqtt_client.mqtt.topic_prefix}edges/{edge_uuid}/command"


def ensure_edge_command_subscription() -> bool:
    """Subscribe once to this edge's MQTT command topic."""
    global _EDGE_COMMAND_SUBSCRIBED
    if _EDGE_COMMAND_SUBSCRIBED:
        return True

    token = load_token()
    if not token:
        return False

    with _EDGE_COMMAND_SUBSCRIPTION_LOCK:
        if _EDGE_COMMAND_SUBSCRIBED:
            return True

        topic = _resolve_edge_command_topic(token)
        if not topic:
            return False

        mqtt_client = _get_shared_mqtt_client(token)
        if not mqtt_client:
            return False

        mqtt_client.mqtt.subscribe(topic, _handle_edge_command_message)
        _EDGE_COMMAND_SUBSCRIBED = True
        logger.info("Subscribed to edge command topic: %s", topic)
        return True


# ---------------------------------------------------------------------------
# Twin command subscription (handles sync_workflows from CLI)
# ---------------------------------------------------------------------------


def _handle_twin_command_message(*args: Any) -> None:
    """Handle MQTT command message received on a twin command topic."""
    if len(args) == 1:
        payload = args[0]
    elif len(args) >= 2:
        payload = args[1]
    else:
        return

    if not isinstance(payload, dict):
        logger.warning("Ignoring twin command with non-dict payload: %r", payload)
        return

    command = str(payload.get("command", "")).strip().lower()
    if command != TWIN_COMMAND_SYNC_WORKFLOWS:
        logger.debug("Ignoring unknown twin command: %s", command)
        return

    logger.info("Received %s command via MQTT — triggering immediate worker sync", command)
    worker = threading.Thread(
        target=_run_immediate_worker_sync,
        name="twin-cmd-sync-workflows",
        daemon=True,
    )
    worker.start()


def _run_immediate_worker_sync() -> None:
    """Run an immediate worker sync in a background thread."""
    try:
        summary = reconcile_worker_sync()
        logger.info(
            "Immediate worker sync complete: written=%d removed=%d unchanged=%d errors=%d",
            summary["written"],
            summary["removed"],
            summary["unchanged"],
            summary["errors"],
        )
    except Exception:
        logger.exception("Immediate worker sync failed")


def ensure_twin_command_subscriptions() -> bool:
    """Reconcile MQTT command topic subscriptions with the linked twin set.

    Subscribes to ``cyberwave/twin/{twin_uuid}/command`` for every twin
    currently linked to this edge, and unsubscribes from topics for twins
    that are no longer linked. Idempotent and cheap when nothing changed.

    Listens for ``sync_workflows`` commands published either by the CLI
    (``cyberwave workflow sync`` / ``cyberwave edge sync-workflows``) or
    by the dashboard (``POST /api/v1/twins/{uuid}/sync-workflows``).

    Returns ``True`` when the subscription set is in sync with the API
    (including the case where there are no linked twins), and ``False``
    when reconciliation could not be attempted (missing token / env /
    fingerprint, API failure, or no MQTT client available).
    """
    token = load_token()
    if not token:
        return False

    with _TWIN_COMMAND_SUBSCRIPTION_LOCK:
        environment_uuid = load_environment_uuid()
        if not environment_uuid:
            return False

        fingerprint = get_or_create_fingerprint()
        if not fingerprint:
            return False

        try:
            desired_uuids = set(
                _resolve_worker_sync_twin_uuids(
                    token, environment_uuid, fingerprint
                )
            )
        except Exception:
            logger.exception("Could not list linked twins for command subscription")
            return False

        # Fast path: nothing to add or remove. Avoids touching the MQTT
        # client (and creating one when no twins are linked yet).
        if desired_uuids == _SUBSCRIBED_TWIN_COMMAND_UUIDS:
            return True

        mqtt_client = _get_shared_mqtt_client(token)
        if not mqtt_client:
            return False

        prefix = mqtt_client.mqtt.topic_prefix
        to_subscribe = desired_uuids - _SUBSCRIBED_TWIN_COMMAND_UUIDS
        to_unsubscribe = _SUBSCRIBED_TWIN_COMMAND_UUIDS - desired_uuids

        for twin_uuid in sorted(to_subscribe):
            topic = f"{prefix}cyberwave/twin/{twin_uuid}/command"
            try:
                mqtt_client.mqtt.subscribe(topic, _handle_twin_command_message)
            except Exception:
                logger.exception(
                    "Failed to subscribe to twin command topic: %s", topic
                )
                continue
            _SUBSCRIBED_TWIN_COMMAND_UUIDS.add(twin_uuid)
            logger.info("Subscribed to twin command topic: %s", topic)

        for twin_uuid in sorted(to_unsubscribe):
            topic = f"{prefix}cyberwave/twin/{twin_uuid}/command"
            try:
                mqtt_client.mqtt.unsubscribe(topic)
            except Exception:
                logger.exception(
                    "Failed to unsubscribe from twin command topic: %s", topic
                )
                # Drop from the tracked set anyway so we don't loop forever
                # trying to unsubscribe a topic the broker already forgot.
            _SUBSCRIBED_TWIN_COMMAND_UUIDS.discard(twin_uuid)
            logger.info("Unsubscribed from twin command topic: %s", topic)

        return True


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* and return the result.

    - Dict values are merged recursively.
    - All other values in *override* take precedence over *base*.
    - Keys that only exist in *base* are preserved.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def write_or_update_twin_json_file(twin_uuid: str, twin_data: dict, asset_data: dict) -> bool:
    """
    Writes the content of the JSON twin into the disk, so that the docker container can read it
    and use it to start the driver correctly.

    If the JSON file already exists on disk, the new data is deep-merged on top
    of the existing content so that any locally-written keys are preserved.
    Existing files are rewritten in place so bind-mounted driver containers keep
    seeing the same inode.
    """
    twin_data["asset"] = asset_data
    twin_json_file = CONFIG_DIR / f"{twin_uuid}.json"

    # Docker bind mounts create a directory when the source path doesn't exist.
    # Clean up so we can write a regular file.
    if twin_json_file.is_dir():
        logger.warning(
            "Twin file path %s is a directory (likely from a Docker bind mount), removing it",
            twin_json_file,
        )
        shutil.rmtree(twin_json_file)

    # Merge with existing data so local-only keys are not lost.
    if twin_json_file.exists():
        try:
            with open(twin_json_file) as f:
                existing_data: dict = json.load(f)
            twin_data = _deep_merge(existing_data, twin_data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read existing twin file %s, overwriting: %s",
                twin_json_file,
                exc,
            )

    def _json_default(obj: Any) -> Any:
        """Handle non-serializable types (e.g. datetime from SDK responses)."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    rendered_json = json.dumps(twin_data, indent=2, default=_json_default) + "\n"

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if twin_json_file.exists():
        with open(twin_json_file, "r+", encoding="utf-8") as file_handle:
            file_handle.seek(0)
            file_handle.write(rendered_json)
            file_handle.truncate()
            file_handle.flush()
            os.fsync(file_handle.fileno())
        if os.name != "nt":
            try:
                os.chmod(twin_json_file, 0o600)
            except OSError:
                pass
    else:
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                dir=CONFIG_DIR,
                prefix=f"{twin_uuid}.",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                temp_file.write(rendered_json)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_path = temp_file.name
            if os.name != "nt":
                os.chmod(temp_path, 0o600)
            os.replace(temp_path, twin_json_file)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    logger.debug(
                        "Failed to remove temp twin JSON file %s",
                        temp_path,
                        exc_info=True,
                    )

    checksum = _calculate_file_checksum(twin_json_file)
    if checksum:
        _TWIN_FILE_CHECKSUMS[twin_uuid] = checksum
    else:
        _TWIN_FILE_CHECKSUMS.pop(twin_uuid, None)
    return True


def _is_driver_twin_json_file(path: Path) -> bool:
    """Return True when *path* is a managed twin JSON object file."""
    if not path.is_file() or path.name in _PROTECTED_CONFIG_JSON_FILES:
        return False

    try:
        uuid.UUID(path.stem)
        return True
    except ValueError:
        return False


def _calculate_file_checksum(path: Path) -> Optional[str]:
    """Return SHA-256 checksum for *path* or None on read failures."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(8192), b""):
                digest.update(chunk)
    except OSError as exc:
        logger.warning("Failed to read twin JSON file %s: %s", path, exc)
        return None
    return digest.hexdigest()


def _extract_twin_update_payload(twin_json_data: dict[str, Any]) -> dict[str, Any]:
    """Build safe payload for PUT /api/v1/twins/{uuid} from local twin JSON.

    Only fields the edge legitimately owns are included.  Asset and
    environment assignments are managed via the UI/API, not the edge.
    """
    return {
        key: twin_json_data[key] for key in _TWIN_UPDATE_ALLOWED_FIELDS if key in twin_json_data
    }


def _sync_twin_json_file_with_backend(
    client: Cyberwave, twin_uuid: str, twin_json_file: Path
) -> bool:
    """Push one changed twin JSON file to backend using the REST twin update."""
    try:
        with open(twin_json_file) as file_handle:
            twin_json_data = json.load(file_handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Twin JSON sync skipped for %s: invalid JSON (%s)", twin_json_file, exc)
        return False

    if not isinstance(twin_json_data, dict):
        logger.warning(
            "Twin JSON sync skipped for %s: expected object root",
            twin_json_file,
        )
        return False

    payload = _extract_twin_update_payload(twin_json_data)
    if not payload:
        logger.warning(
            "Twin JSON sync skipped for %s: no updatable fields found",
            twin_json_file,
        )
        return False

    try:
        client.twins.update(twin_uuid, **payload)
        logger.info(
            "Synced updated twin JSON for %s (fields=%s)",
            twin_uuid,
            sorted(payload.keys()),
        )
        return True
    except Exception as exc:
        logger.warning("Failed to sync twin JSON for %s: %s", twin_uuid, exc)
        return False


# Fields the edge will overwrite from the backend twin into the local JSON
# file when the local file has not changed since the last cycle. Kept narrow
# on purpose:
#
#   * ``metadata`` is the canonical case — operator-driven flags such as
#     ``frame_filter_enabled`` are toggled in the UI and need to reach the
#     driver container's environment via ``entrypoint.sh``. Without a pull
#     leg the only way to push a UI change to the edge was to restart
#     edge-core (which re-fetches twins on startup).
#
# Adding more fields here is OK as long as they are *backend-owned* — i.e.
# the edge never legitimately mutates them locally. Pulling a field that
# the edge also writes would silently clobber the local edit on the next
# cycle.
_TWIN_PULL_ALLOWED_FIELDS = frozenset({"metadata"})


def _coerce_twin_to_dict(twin: Any) -> dict[str, Any] | None:
    """Best-effort conversion of an SDK twin object to a plain dict."""
    if isinstance(twin, dict):
        return twin
    for attr in ("to_dict", "model_dump", "dict"):
        method = getattr(twin, attr, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                continue
            if isinstance(value, dict):
                return value
    if hasattr(twin, "__dict__") and isinstance(twin.__dict__, dict):
        return {k: v for k, v in twin.__dict__.items() if not k.startswith("_")}
    return None


def _atomic_write_twin_json(twin_json_file: Path, rendered: str) -> bool:
    """Write *rendered* into *twin_json_file* atomically via a same-dir temp."""
    twin_json_file.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=twin_json_file.parent,
            prefix=f"{twin_json_file.stem}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            temp_file.write(rendered)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = temp_file.name
        if os.name != "nt":
            try:
                os.chmod(temp_path, 0o600)
            except OSError:
                pass
        os.replace(temp_path, twin_json_file)
        return True
    except OSError as exc:
        logger.warning("Failed to write twin JSON %s: %s", twin_json_file, exc)
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                logger.debug("Could not remove temp twin JSON %s", temp_path, exc_info=True)


def _pull_twin_json_file_from_backend(
    client: Cyberwave, twin_uuid: str, twin_json_file: Path
) -> bool:
    """Apply backend-managed twin fields to the local JSON file.

    Returns True when the local file was actually modified. Backend fields
    listed in :data:`_TWIN_PULL_ALLOWED_FIELDS` win over the local value;
    everything else (asset, environment, edge-owned positions, kinematics,
    capabilities, ...) is left untouched.

    Errors (network, JSON, missing twin) are logged at debug and treated as
    no-op so the reconcile loop keeps running.
    """
    if not twin_json_file.exists():
        return False

    try:
        with open(twin_json_file) as file_handle:
            local_data = json.load(file_handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Twin pull skipped for %s: invalid local JSON (%s)", twin_uuid, exc)
        return False
    if not isinstance(local_data, dict):
        logger.warning("Twin pull skipped for %s: expected object root in local JSON", twin_uuid)
        return False

    try:
        twin = client.twins.get_raw(twin_uuid)
    except Exception as exc:
        logger.debug("Twin pull skipped for %s: backend fetch failed (%s)", twin_uuid, exc)
        return False

    backend_data = _coerce_twin_to_dict(twin)
    if backend_data is None:
        logger.debug("Twin pull skipped for %s: cannot serialize backend twin", twin_uuid)
        return False

    changed_fields: list[str] = []
    for field in _TWIN_PULL_ALLOWED_FIELDS:
        if field not in backend_data:
            continue
        if local_data.get(field) == backend_data[field]:
            continue
        local_data[field] = backend_data[field]
        changed_fields.append(field)

    if not changed_fields:
        return False

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    rendered = json.dumps(local_data, indent=2, default=_json_default) + "\n"
    if not _atomic_write_twin_json(twin_json_file, rendered):
        return False

    logger.info(
        "Pulled twin JSON updates for %s from backend (fields=%s)",
        twin_uuid,
        sorted(changed_fields),
    )
    return True


def _build_twin_sync_client() -> Optional[Cyberwave]:
    """Construct an SDK client for twin sync, returning None on failure."""
    token = load_token()
    if not token:
        logger.warning("Cannot reconcile twin JSON files: no API token available")
        return None

    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    try:
        return Cyberwave(base_url=base_url, token=token)
    except Exception as exc:
        logger.warning("Cannot reconcile twin JSON files: failed to create client (%s)", exc)
        return None


def reconcile_twin_json_file_sync() -> dict[str, int]:
    """Reconcile local twin JSON files with the backend in both directions.

    The reconcile is bidirectional:

    * **Push** (legacy behaviour): a local file whose checksum changed since
      the last cycle is pushed to the backend via the twin REST update
      endpoint. The set of fields the edge is allowed to push is constrained
      by :data:`_TWIN_UPDATE_ALLOWED_FIELDS`.

    * **Pull** (new behaviour): for every tracked twin file that did *not*
      change locally this cycle, the latest twin is fetched from the backend
      and the fields in :data:`_TWIN_PULL_ALLOWED_FIELDS` (currently just
      ``metadata``) are merged in. This closes the gap that previously
      forced an edge-core restart for UI-driven metadata edits to reach the
      driver container's environment via ``entrypoint.sh``.

    Push wins for the current cycle; the next cycle's pull will reflect any
    backend-side reconciliation. The summary dict is the only return value
    and is used by the runtime loop for log output.
    """
    changed_candidates: list[tuple[str, Path, str]] = []
    active_twin_uuids: set[str] = set()

    for json_file in sorted(CONFIG_DIR.glob("*.json")):
        if not _is_driver_twin_json_file(json_file):
            continue

        twin_uuid = json_file.stem
        active_twin_uuids.add(twin_uuid)
        checksum = _calculate_file_checksum(json_file)
        if not checksum:
            continue

        previous_checksum = _TWIN_FILE_CHECKSUMS.get(twin_uuid)
        if previous_checksum is None:
            _TWIN_FILE_CHECKSUMS[twin_uuid] = checksum
            continue
        if previous_checksum == checksum:
            continue

        changed_candidates.append((twin_uuid, json_file, checksum))

    for stale_twin_uuid in set(_TWIN_FILE_CHECKSUMS) - active_twin_uuids:
        _TWIN_FILE_CHECKSUMS.pop(stale_twin_uuid, None)

    summary = {
        "tracked": len(active_twin_uuids),
        "changed": len(changed_candidates),
        "synced": 0,
        "pulled": 0,
    }
    if not active_twin_uuids:
        return summary

    client = _build_twin_sync_client()
    if client is None:
        return summary

    pushed_uuids: set[str] = set()
    for twin_uuid, twin_json_file, checksum in changed_candidates:
        if _sync_twin_json_file_with_backend(client, twin_uuid, twin_json_file):
            _TWIN_FILE_CHECKSUMS[twin_uuid] = checksum
            summary["synced"] += 1
            pushed_uuids.add(twin_uuid)

    for twin_uuid in sorted(active_twin_uuids):
        if twin_uuid in pushed_uuids:
            # Already in sync this cycle; the next cycle's pull will surface
            # any concurrent backend edits.
            continue
        twin_json_file = CONFIG_DIR / f"{twin_uuid}.json"
        if _pull_twin_json_file_from_backend(client, twin_uuid, twin_json_file):
            new_checksum = _calculate_file_checksum(twin_json_file)
            if new_checksum:
                _TWIN_FILE_CHECKSUMS[twin_uuid] = new_checksum
            summary["pulled"] += 1

    return summary


def _fix_config_dir_ownership() -> None:
    """Re-chown CONFIG_DIR entries that were written as root back to the invoking user.

    Only runs on Linux when the process is root.  The target uid/gid is
    resolved via :func:`resolve_config_owner_uid_gid`, which handles both
    the ``sudo`` case (``SUDO_UID`` set) and the ``systemd`` case (no
    ``SUDO_UID``, but ``CONFIG_DIR.parent`` is owned by a non-root user).
    Non-root processes cannot chown files they don't own, so they skip
    the fix as well.
    """
    target = resolve_config_owner_uid_gid()
    if target is None:
        return
    target_uid, target_gid = target
    fixed = 0

    try:
        for dirpath, _dirnames, filenames in os.walk(CONFIG_DIR):
            for name in [dirpath] + [os.path.join(dirpath, f) for f in filenames]:
                try:
                    st = os.lstat(name)
                except OSError:
                    continue
                if st.st_uid != target_uid:
                    try:
                        os.lchown(name, target_uid, target_gid)
                        fixed += 1
                    except OSError as exc:
                        logger.debug("Cannot chown %s: %s", name, exc)
    except OSError as exc:
        logger.debug("Cannot walk %s for ownership fix: %s", CONFIG_DIR, exc)

    if fixed:
        logger.info(
            "Fixed ownership on %d file(s) in %s (uid=%d, gid=%d)",
            fixed,
            CONFIG_DIR,
            target_uid,
            target_gid,
        )


def _ensure_config_subdirs() -> None:
    """Eagerly create standard subdirectories under CONFIG_DIR.

    ``workers/`` and ``models/`` were historically created lazily — the
    first at worker-container launch, the second at the first model
    download.  On an edge node with no linked twins or a failed workflow
    sync, neither path runs and the directories simply never exist,
    leaving users and standalone SDK scripts confused about where to
    drop pre-staged weights.

    Creating them eagerly at startup (with ownership matching
    :func:`resolve_config_owner_uid_gid`) gives operators a predictable
    layout they can inspect from a regular shell.  ``CONFIG_DIR``
    itself is also chowned when newly created so the subdirectories do
    not live inside a root-only parent.
    """
    target = resolve_config_owner_uid_gid()

    def _chown_if_needed(path: Path) -> None:
        if target is None:
            return
        try:
            st = path.stat()
            if st.st_uid != target[0] or st.st_gid != target[1]:
                os.chown(path, target[0], target[1])
        except OSError as exc:
            logger.debug("Cannot chown %s: %s", path, exc)

    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug("Cannot create %s: %s", CONFIG_DIR, exc)
        return
    _chown_if_needed(CONFIG_DIR)

    for name in ("workers", "models"):
        subdir = CONFIG_DIR / name
        try:
            subdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.debug("Cannot create %s: %s", subdir, exc)
            continue
        _chown_if_needed(subdir)


def run_startup_checks(
    *,
    resource_monitor: Optional[Any] = None,
    watchdog: Optional[Any] = None,
) -> bool:
    """Execute every boot-time check in sequence.

    Prints a Rich-formatted report to the console.
    Returns ``True`` only when **all** checks pass.

    ``resource_monitor`` and ``watchdog``, when provided, are folded into
    the bootstrap ``edge_health`` publisher so the very first heartbeat
    payload already carries host pressure data.  Both default to ``None``
    so callers that just want the legacy startup behaviour (e.g. tests,
    one-shot CLI flows) are unaffected.
    """
    _fix_config_dir_ownership()
    _ensure_config_subdirs()

    console.print("\n[bold]Cyberwave Edge Core — Startup Checks[/bold]\n")

    # Log resolved configuration for troubleshooting
    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    runtime_environment = (
        get_runtime_env_var("CYBERWAVE_ENVIRONMENT", DEFAULT_ENVIRONMENT) or DEFAULT_ENVIRONMENT
    )
    console.print(f"  [dim]Config dir:  {CONFIG_DIR}[/dim]")
    console.print(f"  [dim]Base URL:    {base_url}[/dim]")
    console.print(f"  [dim]Environment: {runtime_environment}[/dim]")
    console.print()

    # 1 — credentials file
    _t0 = time.perf_counter()
    token = load_token()
    if not token:
        console.print(f"  [red]✗[/red] Credentials [dim]({time.perf_counter() - _t0:.3f}s)[/dim]")
        console.print(f"  [red]No credentials found at {CREDENTIALS_FILE}[/red]")
        console.print("  [dim]Run 'cyberwave login' on this device first.[/dim]")
        return False
    console.print(f"  [green]✓[/green] Credentials [dim]({time.perf_counter() - _t0:.3f}s)[/dim]")

    # 2 — token validity
    _t0 = time.perf_counter()
    token_valid = validate_token(token)
    if token_valid:
        console.print(f"  [green]✓[/green] Token [dim]({time.perf_counter() - _t0:.3f}s)[/dim]")
    else:
        console.print(f"  [red]✗[/red] Token [dim]({time.perf_counter() - _t0:.3f}s)[/dim]")
        console.print(f"  [red]Token validation failed against {base_url}[/red]")
        console.print("  [dim]Check 'journalctl -u cyberwave-edge-core' for details.[/dim]")
        console.print("  [dim]Run 'cyberwave login' to refresh your credentials.[/dim]")
        return False

    # 3 — MQTT broker
    _t0 = time.perf_counter()
    mqtt_ok = check_mqtt_connection(token)
    if mqtt_ok:
        console.print(
            f"  [green]✓[/green] MQTT broker [dim]({time.perf_counter() - _t0:.3f}s)[/dim]"
        )
    else:
        console.print(f"  [red]✗[/red] MQTT broker [dim]({time.perf_counter() - _t0:.3f}s)[/dim]")
        console.print("  [red]Could not connect to the MQTT broker.[/red]")
        console.print("  [dim]Check network connectivity and MQTT configuration.[/dim]")

    # 4: Edge registering
    _t0 = time.perf_counter()
    edge_ok = register_edge(token)
    if edge_ok:
        console.print(
            f"  [green]✓[/green] Edge registration [dim]({time.perf_counter() - _t0:.3f}s)[/dim]"
        )
    else:
        console.print(
            f"  [red]✗[/red] Edge registration [dim]({time.perf_counter() - _t0:.3f}s)[/dim]"
        )
        console.print("  [red]Could not register the edge.[/red]")
        return False

    # 4b — refresh static host facts on Edge.metadata. Non-fatal: the
    # dashboard's "what hardware is this" row simply lags by one boot
    # cycle if the call fails.
    _t0 = time.perf_counter()
    host_facts_ok = _upload_host_facts_on_startup(token)
    elapsed = time.perf_counter() - _t0
    if host_facts_ok:
        console.print(f"  [green]✓[/green] Host facts [dim]({elapsed:.3f}s)[/dim]")
    else:
        console.print(f"  [yellow]⚠[/yellow] Host facts [dim]({elapsed:.3f}s)[/dim]")

    # 5 — linked environment
    _t0 = time.perf_counter()
    environment_uuid = load_environment_uuid(retries=5, retry_delay_seconds=0.2)
    if environment_uuid:
        _elapsed = time.perf_counter() - _t0
        console.print(
            f"  [green]✓[/green] Environment [dim]({environment_uuid}, {_elapsed:.3f}s)[/dim]"
        )
    else:
        console.print(
            f"  [yellow]⚠[/yellow] Environment [dim]({time.perf_counter() - _t0:.3f}s)[/dim]"
        )
        console.print(f"  [yellow]No linked environment found in {ENVIRONMENT_FILE}[/yellow]")
        console.print("  [dim]Expected format: {'uuid': 'unique-uuid-of-the-environment'}[/dim]")

    # 6 — fetch twins, match by fingerprint, write JSON file, run drivers
    if environment_uuid:
        fingerprint = get_or_create_fingerprint()
        if not fingerprint:
            console.print("  [red]✗[/red] Edge fingerprint")
            console.print("  [red]Could not determine edge fingerprint.[/red]")
        else:
            _t0 = time.perf_counter()
            heartbeat_ok = False
            linked_twin_uuids: list[str] = []
            try:
                linked_twin_uuids = _list_linked_twin_uuids_for_fingerprint(
                    token, environment_uuid, fingerprint
                )
                heartbeat_ok = _start_bootstrap_edge_health_publisher(
                    token,
                    linked_twin_uuids,
                    edge_id=fingerprint,
                    resource_monitor=resource_monitor,
                    watchdog=watchdog,
                )
            except Exception as exc:
                logger.warning("Early edge heartbeat bootstrap failed: %s", exc)

            elapsed = time.perf_counter() - _t0
            if heartbeat_ok:
                console.print(
                    (
                        "  [green]✓[/green] Edge heartbeat "
                        f"[dim]({len(linked_twin_uuids)} twin(s), {elapsed:.3f}s)[/dim]"
                    )
                )
            else:
                console.print(f"  [yellow]⚠[/yellow] Edge heartbeat [dim]({elapsed:.3f}s)[/dim]")
                console.print(
                    "  [dim]Could not start early edge heartbeat; continuing startup.[/dim]"
                )

            # 6b — Zenoh infrastructure diagnostics
            zenoh_cfg = _get_zenoh_config()
            zenoh_diag = log_zenoh_diagnostics(zenoh_cfg)
            zenoh_icon = "[green]✓[/green]" if zenoh_cfg.is_zenoh else "[yellow]⚠[/yellow]"
            console.print(f"  {zenoh_icon} Zenoh [dim]({zenoh_diag.mode})[/dim]")
            for w in zenoh_diag.warnings:
                console.print(f"  [yellow]  ↳ {w}[/yellow]")

            # 6c — optional Zenoh router container (must start before drivers)
            if zenoh_cfg.router_enabled:
                _t0 = time.perf_counter()
                router_ok = start_zenoh_router(zenoh_cfg, environment_uuid)
                elapsed = time.perf_counter() - _t0
                if router_ok:
                    container_name = zenoh_cfg.router_container_name(environment_uuid)
                    console.print(
                        f"  [green]✓[/green] Zenoh router "
                        f"[dim]({container_name}, {elapsed:.3f}s)[/dim]"
                    )
                else:
                    console.print(
                        f"  [yellow]⚠[/yellow] Zenoh router "
                        f"[dim](failed to start, {elapsed:.3f}s)[/dim]"
                    )
                    console.print(
                        "  [dim]Driver containers will still start; "
                        "peer-to-peer discovery will be used as fallback.[/dim]"
                    )

            _t0 = time.perf_counter()
            results = fetch_and_run_twin_drivers(token, environment_uuid, fingerprint)
            _elapsed = time.perf_counter() - _t0
            if not results:
                console.print(
                    f"  [yellow]⚠[/yellow] Twin drivers [dim]({_elapsed:.3f}s)[/dim]"
                )
                console.print("  [dim]No twins with driver images matched this edge.[/dim]")
            else:
                started = sum(1 for r in results if r["success"])
                console.print(
                    f"  [green]✓[/green] Twin drivers "
                    f"[dim]({started}/{len(results)}, {_elapsed:.3f}s)[/dim]"
                )
                for r in results:
                    status = "[green]✓[/green]" if r["success"] else "[red]✗[/red]"
                    console.print(f"    {r['twin_name']} → {r['driver_image']} {status}")

                if started > 0:
                    _stop_bootstrap_edge_health_publisher()

            # 7 — workflow worker sync (pull generated wf_*.py files from backend)
            #
            # Scope here is the operator-selected twin list from
            # ``environment.json``, not the fingerprint-discovered set used
            # for drivers/health.  The two can legitimately diverge (e.g.
            # stale ``metadata.edge_fingerprint`` from a previous install),
            # and we want the edge to pull workers strictly for the twins
            # the user picked in ``cyberwave edge install``.
            sync_twin_uuids = _resolve_worker_sync_twin_uuids(
                token, environment_uuid, fingerprint
            )
            if sync_twin_uuids:
                _t0 = time.perf_counter()
                base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
                sync_summary = _sync_workers_for_twins(
                    token=token,
                    twin_uuids=sync_twin_uuids,
                    base_url=base_url,
                )
                total_written = sum(r["written"] for r in sync_summary.values())
                total_removed = sum(r["removed"] for r in sync_summary.values())
                total_errors = sum(r["errors"] for r in sync_summary.values())
                elapsed = time.perf_counter() - _t0
                if total_errors:
                    console.print(
                        f"  [yellow]⚠[/yellow] Worker sync "
                        f"[dim](written={total_written}, removed={total_removed}, errors={total_errors}, {elapsed:.3f}s)[/dim]"
                    )
                else:
                    console.print(
                        f"  [green]✓[/green] Worker sync "
                        f"[dim](written={total_written}, removed={total_removed}, {elapsed:.3f}s)[/dim]"
                    )

                # 8 — start the worker container only if active workflows exist.
                #
                # We base this on the actual on-disk state of ``workers/``
                # rather than the sync counts, so that pre-existing files
                # from a previous boot still trigger a worker start when
                # this boot's sync errored. ``WorkerManager.start()`` is
                # itself a no-op when ``workers/*.py`` is empty (and
                # therefore skips the ``cyberwaveos/edge-ml-worker`` image
                # pull — CYB-1766), so we only need the gate here for the
                # operator-friendly log.
                workers_dir = CONFIG_DIR / "workers"
                has_active_workers = workers_dir.is_dir() and any(
                    workers_dir.glob("*.py")
                )
                if has_active_workers:
                    _start_worker_after_drivers(
                        token=token,
                        environment_uuid=environment_uuid,
                        twin_uuids=sync_twin_uuids,
                    )
                elif total_errors:
                    logger.warning(
                        "Could not determine active workflows due to %d sync "
                        "error(s); worker container not started. Will retry on "
                        "the next reconcile cycle.",
                        total_errors,
                    )
                    console.print(
                        "  [yellow]⚠[/yellow] Sync errors prevented worker "
                        "startup; will retry."
                    )
                else:
                    from .worker_manager import resolve_worker_image as _resolve_worker_image  # noqa: PLC0415

                    logger.info(
                        "No active workflows for any of the %d connected twin(s); "
                        "skipping worker container start (no '%s' image pull).",
                        len(sync_twin_uuids),
                        _resolve_worker_image(),
                    )
                    console.print(
                        "  [dim]No active workflows for connected twins; "
                        "worker container not started.[/dim]"
                    )

    console.print("\n[green]All startup checks passed.[/green]\n")
    return True


# Persistent two-strikes state for worker cleanup. Owned by the
# ``EdgeSyncClient`` instances that ``_sync_workers_for_twins``
# constructs, but stored at module scope so the strike count survives
# across the periodic-sync cycle (each cycle creates a fresh client).
# A worker file disappearing from one sync response is recorded here;
# only on a second consecutive miss is the file actually removed.
# Cleared on edge-core process restart, which gives every existing
# ``wf_*.py`` one strike of grace after a cold start.
_WORKER_SYNC_PREVIOUSLY_MISSING: set[str] = set()


def _sync_workers_for_twins(
    *,
    token: str,
    twin_uuids: list[str],
    base_url: str,
) -> dict[str, dict[str, int]]:
    """Pull generated wf_*.py files from the backend for each linked twin.

    Uses ``sync_all`` so that stale-file cleanup happens only after every
    twin's payload has been fetched — preventing twin A's sync from
    deleting twin B's workers.

    Returns a dict mapping twin_uuid → {written, removed, unchanged, errors}.
    """
    from cyberwave_edge_core.edge_sync_client import EdgeSyncClient

    workers_dir = CONFIG_DIR / "workers"
    client = EdgeSyncClient(
        workers_dir=workers_dir,
        base_url=base_url,
        token=token,
        previously_missing=_WORKER_SYNC_PREVIOUSLY_MISSING,
    )

    summary: dict[str, dict[str, int]] = {}
    try:
        results = client.sync_all(twin_uuids)
        for result in results:
            summary[result.twin_uuid] = {
                "written": len(result.written),
                "removed": len(result.removed),
                "unchanged": len(result.unchanged),
                "errors": len(result.errors),
            }
    except Exception:
        logger.exception("Worker sync_all failed for twins %s", twin_uuids)
        for twin_uuid in twin_uuids:
            summary[twin_uuid] = {
                "written": 0,
                "removed": 0,
                "unchanged": 0,
                "errors": 1,
            }
    return summary


def reconcile_worker_sync() -> dict[str, int]:
    """Periodic reconciliation: pull updated wf_*.py files for all linked twins.

    Called from the runtime loop alongside other reconcile functions.
    Returns a summary dict: {written, removed, unchanged, errors}.
    """
    token = load_token()
    if not token:
        return {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}

    environment_uuid = load_environment_uuid()
    if not environment_uuid:
        return {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}

    fingerprint = get_or_create_fingerprint()
    if not fingerprint:
        return {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}

    try:
        twin_uuids = _resolve_worker_sync_twin_uuids(
            token, environment_uuid, fingerprint
        )
    except Exception:
        logger.exception("Could not list linked twins for worker sync reconcile")
        return {"written": 0, "removed": 0, "unchanged": 0, "errors": 1}

    if not twin_uuids:
        return {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}

    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    per_twin = _sync_workers_for_twins(
        token=token,
        twin_uuids=twin_uuids,
        base_url=base_url,
    )

    totals: dict[str, int] = {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}
    for stats in per_twin.values():
        for key in totals:
            totals[key] += stats.get(key, 0)
    return totals


def reconcile_worker_lifecycle(sync_summary: dict[str, int]) -> None:
    """Start the worker if active workflows produced files; stop it otherwise.

    Called from the runtime loop right after :func:`reconcile_worker_sync`
    so that mid-run activations/deactivations are picked up promptly:

    * Files appeared (workflow activated) → ``WorkerManager.start()`` brings
      the container up and pulls ``cyberwaveos/edge-ml-worker`` on first
      activation.
    * Files disappeared (workflow deactivated) → ``WorkerManager.stop()``
      gracefully stops the container (it is left in ``exited`` state for
      diagnostics; the next start re-creates it cleanly).

    Both ``start`` and ``stop`` are idempotent in their respective steady
    states (already running / already stopped), so the periodic call is
    cheap. We bail out early when sync reported any errors to avoid
    churning the worker on transient API failures (CYB-1766).
    """
    if sync_summary.get("errors"):
        return

    token = load_token()
    if not token:
        return

    environment_uuid = load_environment_uuid()
    if not environment_uuid:
        return

    fingerprint = get_or_create_fingerprint()
    if not fingerprint:
        return

    try:
        twin_uuids = _resolve_worker_sync_twin_uuids(
            token, environment_uuid, fingerprint
        )
    except Exception:
        logger.exception("Could not resolve twins for worker lifecycle reconcile")
        return

    workers_dir = CONFIG_DIR / "workers"
    has_files = workers_dir.is_dir() and any(workers_dir.glob("*.py"))

    from .worker_manager import WorkerManager, resolve_worker_image  # noqa: PLC0415

    worker_manager = WorkerManager(
        config_dir=CONFIG_DIR,
        environment_uuid=environment_uuid,
        token=token,
        twin_uuids=twin_uuids,
        image=resolve_worker_image(),
        resource_limits=load_worker_resource_limits(),
    )

    if has_files:
        worker_manager.start()
    else:
        worker_manager.stop()


# Interval at which worker sync reconciliation runs (every N runtime loops).
# At 15 s per loop this defaults to ~5 minutes.
_WORKER_SYNC_INTERVAL_LOOPS = int(
    os.getenv("CYBERWAVE_WORKER_SYNC_INTERVAL_LOOPS", "20")
)
_worker_sync_loop_counter = 0

_last_container_prune_time: float = 0.0
_last_image_prune_time: float = 0.0


def _run_periodic_docker_cleanup() -> None:
    """Prune stopped cyberwave containers and unused images on a schedule.

    Container pruning runs every ``CONTAINER_PRUNE_INTERVAL_SECONDS`` (default
    30 min). Image pruning runs every ``IMAGE_PRUNE_INTERVAL_SECONDS`` (default
    3 h).  Both are no-ops when Docker is unavailable.
    """
    global _last_container_prune_time, _last_image_prune_time

    from .docker_helpers import docker_prune_stopped_cyberwave_containers, docker_prune_unused_images

    now = time.monotonic()

    if now - _last_container_prune_time >= CONTAINER_PRUNE_INTERVAL_SECONDS:
        try:
            docker_prune_stopped_cyberwave_containers()
        except Exception:
            logger.exception("Unexpected error during stopped-container prune")
        _last_container_prune_time = now

    if now - _last_image_prune_time >= IMAGE_PRUNE_INTERVAL_SECONDS:
        try:
            docker_prune_unused_images()
        except Exception:
            logger.exception("Unexpected error during unused-image prune")
        _last_image_prune_time = now


def run_runtime_loop(
    *,
    watchdog: Optional[Any] = None,
    resource_monitor: Optional[Any] = None,
) -> None:
    """Keep edge-core alive and continuously reconcile driver log forwarding.

    Parameters
    ----------
    watchdog:
        Optional :class:`~cyberwave_edge_core.watchdog.ProcessWatchdog`
        to ping each reconcile cycle so systemd and/or the hardware
        watchdog know the process is alive.
    resource_monitor:
        Optional :class:`~cyberwave_edge_core.resource_monitor.SystemResourceMonitor`
        to check host memory/temperature each cycle and log warnings
        when resources are critically low.
    """
    global _worker_sync_loop_counter

    logger.info(
        "Entering edge-core runtime loop (interval=%.1fs)",
        LOG_FOLLOWER_RECONCILE_INTERVAL_SECONDS,
    )

    worker_watcher: Optional[Any] = None

    while not shutdown_event.is_set():
        attached = reconcile_driver_log_streams()
        logger.debug(
            "Driver log follower reconcile complete (active_streams=%d, tracked=%d)",
            attached,
            len(_CONTAINER_LOG_THREADS),
        )
        restart_summary = reconcile_driver_restart_failures()
        logger.debug(
            (
                "Driver restart reconcile complete "
                "(inspected=%d, flapping=%d, stopped=%d, alerts_sent=%d)"
            ),
            restart_summary["inspected"],
            restart_summary["flapping"],
            restart_summary["stopped"],
            restart_summary["alerts_sent"],
        )

        try:
            driver_health = reconcile_driver_health_for_worker()
            unhealthy_drivers = [
                f"{n}={s}" for n, s in driver_health.items() if s != "running"
            ]
            if unhealthy_drivers:
                logger.debug(
                    "Driver health: %d not running (%s)",
                    len(unhealthy_drivers),
                    ", ".join(unhealthy_drivers),
                )
        except Exception:
            logger.exception("Unexpected error in driver health reconciliation")

        try:
            # Skip revival in the same tick the flap detector stopped a
            # driver — otherwise we'd immediately undo its decision and
            # start a tug-of-war.  Debounce inside reconcile_driver_revival
            # handles the steady-state "always-down" twin case.
            revival_summary = reconcile_driver_revival(
                skip_revival=restart_summary["stopped"] > 0,
            )
            if revival_summary["revived_attempted"]:
                logger.info(
                    "Driver revival reconcile (down=%d, revived_attempted=%d)",
                    revival_summary["down"],
                    revival_summary["revived_attempted"],
                )
            elif revival_summary["down"] or revival_summary["skipped_orphan"]:
                logger.debug(
                    "Driver revival reconcile skipped "
                    "(down=%d, orphans=%d, flap_protected=%d, "
                    "debounced=%d, no_credentials=%d)",
                    revival_summary["down"],
                    revival_summary["skipped_orphan"],
                    revival_summary["skipped_flap_protection"],
                    revival_summary["skipped_debounce"],
                    revival_summary["skipped_no_credentials"],
                )
        except Exception:
            logger.exception("Unexpected error in driver revival reconciliation")

        twin_sync_summary = reconcile_twin_json_file_sync()
        logger.debug(
            "Twin JSON sync reconcile complete "
            "(tracked=%d, changed=%d, synced=%d, pulled=%d)",
            twin_sync_summary["tracked"],
            twin_sync_summary["changed"],
            twin_sync_summary["synced"],
            twin_sync_summary.get("pulled", 0),
        )

        try:
            if reconcile_camera_config_drift():
                continue
        except Exception:
            logger.exception("Unexpected error in camera config drift reconciliation")

        # Reconcile worker file changes and health probes each cycle.
        try:
            worker_watcher = _reconcile_worker_watcher(worker_watcher)
        except Exception:
            logger.exception("Unexpected error in worker file reconciliation")

        try:
            ensure_edge_command_subscription()
        except Exception:
            logger.exception("Unexpected error while ensuring edge command subscription")

        try:
            ensure_twin_command_subscriptions()
        except Exception:
            logger.exception("Unexpected error while ensuring twin command subscriptions")

        # Periodically pull updated generated worker files from the backend.
        _worker_sync_loop_counter += 1
        if _worker_sync_loop_counter >= _WORKER_SYNC_INTERVAL_LOOPS:
            _worker_sync_loop_counter = 0
            try:
                worker_sync_summary = reconcile_worker_sync()
                if worker_sync_summary.get("written") or worker_sync_summary.get("removed"):
                    logger.info(
                        "Worker sync reconcile: written=%d removed=%d unchanged=%d errors=%d",
                        worker_sync_summary["written"],
                        worker_sync_summary["removed"],
                        worker_sync_summary["unchanged"],
                        worker_sync_summary["errors"],
                    )
                else:
                    logger.debug(
                        "Worker sync reconcile: no changes (unchanged=%d, errors=%d)",
                        worker_sync_summary["unchanged"],
                        worker_sync_summary["errors"],
                    )
                # Start/stop the worker container based on whether any
                # active-workflow files exist after the sync. Idempotent —
                # both start() and stop() are no-ops in their respective
                # steady states (CYB-1766).
                try:
                    reconcile_worker_lifecycle(worker_sync_summary)
                except Exception:
                    logger.exception(
                        "Unexpected error during worker lifecycle reconcile"
                    )
            except Exception:
                logger.exception("Unexpected error during worker sync reconcile")

        # -- Periodic Docker cleanup (CYB-1996) ------------------------------
        _run_periodic_docker_cleanup()

        # -- Watchdog and resource monitoring at end of each cycle -----------
        if watchdog is not None:
            try:
                watchdog.ping()
            except Exception:
                logger.debug("Watchdog ping failed", exc_info=True)

        if resource_monitor is not None:
            try:
                resource_monitor.check()
            except Exception:
                logger.debug("Resource monitor check failed", exc_info=True)

        shutdown_event.wait(LOG_FOLLOWER_RECONCILE_INTERVAL_SECONDS)

    # -- Ordered shutdown: worker container first, then driver containers -----
    _graceful_shutdown(worker_watcher)


def _graceful_shutdown(worker_watcher: Optional[Any]) -> None:
    """Ordered shutdown: stop worker container, then driver containers."""
    logger.info("Graceful shutdown initiated — stopping managed containers")

    # 1. Stop worker container via WorkerManager (sends SIGTERM to container).
    if worker_watcher is not None:
        try:
            wm = worker_watcher.worker_manager
            logger.info("Stopping worker container: %s", wm.container_name)
            wm.stop()
            logger.info("Worker container stopped")
        except Exception:
            logger.exception("Error stopping worker container during shutdown")

    # 2. Stop driver containers (they have --stop-timeout already).
    driver_containers = list(_CONTAINER_TWIN_MAP.keys())
    if driver_containers:
        import subprocess as _sp

        for name in driver_containers:
            try:
                logger.info("Stopping driver container: %s", name)
                result = _sp.run(
                    ["docker", "stop", name],
                    timeout=15,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.debug(
                        "docker stop %s exited with %d: %s",
                        name,
                        result.returncode,
                        result.stderr.strip(),
                    )
            except Exception:
                logger.debug("Error stopping driver container %s", name, exc_info=True)

    logger.info("Graceful shutdown complete")


def _reconcile_worker_watcher(
    existing_watcher: Optional[Any],
) -> Optional[Any]:
    """Lazily create and run the worker file watcher + health monitor each reconcile cycle."""
    from .model_manager import ModelManager
    from .worker_health import WorkerHealthMonitor
    from .worker_manager import WorkerManager, resolve_worker_image
    from .worker_watcher import WorkerWatcher

    environment_uuid = load_environment_uuid()
    if not environment_uuid:
        return existing_watcher

    token = load_token()
    if not token:
        return existing_watcher

    workers_dir = CONFIG_DIR / "workers"

    if existing_watcher is None:
        twin_uuids: list[str] = []
        try:
            fingerprint = get_or_create_fingerprint()
            if fingerprint:
                twin_uuids = _resolve_worker_sync_twin_uuids(
                    token, environment_uuid, fingerprint
                )
        except Exception:
            logger.debug("Could not resolve twin_uuids for worker watcher")

        worker_manager = WorkerManager(
            config_dir=CONFIG_DIR,
            environment_uuid=environment_uuid,
            token=token,
            twin_uuids=twin_uuids,
            image=resolve_worker_image(),
            resource_limits=load_worker_resource_limits(),
        )
        # Attach health monitor so that restarts are accounted and rate-limited.
        health_monitor = WorkerHealthMonitor(
            container_name=worker_manager.container_name,
        )
        worker_manager.set_health_monitor(health_monitor)

        model_manager = ModelManager(
            cache_dir=CONFIG_DIR / "models",
            api_token=token,
            base_url=get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL,
        )
        mqtt_publish: Optional[Any] = None
        mqtt_health_topic: Optional[str] = None
        try:
            cw_mqtt = _get_shared_mqtt_client(token)
            if cw_mqtt and getattr(cw_mqtt, "mqtt", None):
                prefix = cw_mqtt.mqtt.topic_prefix
                first_twin = twin_uuids[0] if twin_uuids else environment_uuid
                mqtt_health_topic = f"{prefix}cyberwave/twin/{first_twin}/worker_health"
                mqtt_publish = cw_mqtt.mqtt.publish
        except Exception:
            logger.debug("Could not set up MQTT for worker health publishing")

        existing_watcher = WorkerWatcher(
            workers_dir=workers_dir,
            worker_manager=worker_manager,
            model_manager=model_manager,
            mqtt_publish=mqtt_publish,
            mqtt_health_topic=mqtt_health_topic,
        )
        logger.debug("Worker file watcher + health monitor initialised for %s", workers_dir)

    # Run health probe each cycle to detect spontaneous exits / crash loops.
    existing_watcher.check_health()

    existing_watcher.reconcile_worker_files()
    return existing_watcher
