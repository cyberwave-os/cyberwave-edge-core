"""Boot-time startup checks for the Cyberwave Edge Core.

On every boot the edge core must:
  1. Read the API token from ``/etc/cyberwave/credentials.json``
  2. Validate the token against the Cyberwave REST API
  3. Verify that it can connect to the MQTT broker
  4. Check whether an environment is linked via ``/etc/cyberwave/environment.json``

The config directory defaults to:
  - ``/etc/cyberwave`` on Linux
  - ``~/.cyberwave`` on macOS
and can be overridden with the ``CYBERWAVE_EDGE_CONFIG_DIR`` environment variable.

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
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from cyberwave import Cyberwave
from cyberwave.edge.platform import is_usbip_server_running as _is_usbip_server_running
from cyberwave.fingerprint import generate_fingerprint
from rich.console import Console

from . import __version__ as PACKAGE_EDGE_CORE_VERSION
from .utils import DriverStartingAlertContext
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
    """Return default edge config directory for this platform."""
    if platform.system() == "Darwin":
        # Docker Desktop cannot reliably bind-mount /etc paths on macOS.
        sudo_home = _resolve_sudo_user_home()
        base_home = sudo_home or Path.home()
        return base_home / ".cyberwave"
    return Path("/etc/cyberwave")


def _resolve_config_dir() -> Path:
    """Resolve config dir honoring explicit environment override first."""
    override = os.getenv("CYBERWAVE_EDGE_CONFIG_DIR", "").strip()
    if override:
        return Path(override)
    return _resolve_default_config_dir()


_LEGACY_MACOS_CONFIG_DIR = Path("/etc/cyberwave")


def _migrate_legacy_macos_config(config_dir: Path) -> None:
    """Best-effort migration from legacy /etc/cyberwave to macOS user config dir.

    This keeps existing macOS installs working after moving defaults away from
    /etc for Docker bind-mount compatibility.
    """
    if platform.system() != "Darwin":
        return
    if os.getenv("CYBERWAVE_EDGE_CONFIG_DIR", "").strip():
        return
    if config_dir == _LEGACY_MACOS_CONFIG_DIR:
        return
    if not _LEGACY_MACOS_CONFIG_DIR.exists():
        return
    if (config_dir / "credentials.json").exists():
        return

    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    bootstrap_logger = logging.getLogger(__name__)
    copied_files = 0
    for json_file in _LEGACY_MACOS_CONFIG_DIR.glob("*.json"):
        if not json_file.is_file():
            continue
        target_file = config_dir / json_file.name
        if target_file.exists():
            continue
        try:
            shutil.copy2(json_file, target_file)
            copied_files += 1
        except OSError:
            continue
    if copied_files:
        bootstrap_logger.info(
            "Migrated %d legacy macOS edge config file(s) from %s to %s",
            copied_files,
            _LEGACY_MACOS_CONFIG_DIR,
            config_dir,
        )


def _bootstrap_runtime_env_vars() -> None:
    """Load persisted runtime env vars into process env for child imports."""
    config_dir = _resolve_config_dir()
    _migrate_legacy_macos_config(config_dir)
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

# Track active log streaming threads per container to avoid duplicates.
_CONTAINER_LOG_THREADS: dict[str, threading.Thread] = {}

# Map container names to twin UUIDs so log threads can publish telemetry.
_CONTAINER_TWIN_MAP: dict[str, str] = {}

# Shared MQTT client for publishing driver log telemetry.
_shared_mqtt_client: Optional[Any] = None
_shared_mqtt_lock = threading.Lock()

# Track which token/path combination has already been announced to avoid
# repeated "Loaded token ..." info logs during steady-state polling.
_last_logged_token_signature: Optional[str] = None
_token_log_lock = threading.Lock()

# ---- constants ---------------------------------------------------------------

# Edge config directory. The systemd unit sets CYBERWAVE_EDGE_CONFIG_DIR on
# Linux. For manual invocation, defaults to /etc/cyberwave on Linux and to the
# invoking user's ~/.cyberwave on macOS for Docker bind-mount compatibility.
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
        "asset_uuid",
        "environment_uuid",
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


def _get_shared_mqtt_client(token: str) -> Any:
    """Return a shared MQTT client, creating and connecting it on first call."""
    global _shared_mqtt_client
    with _shared_mqtt_lock:
        if _shared_mqtt_client is not None and _shared_mqtt_client.mqtt.connected:
            return _shared_mqtt_client
        base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
        mqtt_host = get_runtime_env_var("CYBERWAVE_MQTT_HOST")
        try:
            client = Cyberwave(base_url=base_url, api_key=token, mqtt_host=mqtt_host)
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
    mqtt_host = get_runtime_env_var("CYBERWAVE_MQTT_HOST")
    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    logger.info(
        "Attempting MQTT connection (base_url=%s, mqtt_host=%s)",
        base_url,
        mqtt_host,
    )
    try:
        client = Cyberwave(base_url=base_url, api_key=token, mqtt_host=mqtt_host)
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
    """Persist fingerprint to the edge config directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(FINGERPRINT_FILE, "w") as f:
            json.dump({"fingerprint": fingerprint}, f, indent=2)
            f.write("\n")
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


def _extract_docker_device_mappings(params: list[str]) -> list[tuple[str, str]]:
    """Extract ``--device`` mappings from docker run params.

    Supports:
      - ``--device /dev/ttyACM0:/dev/ttyACM0``
      - ``--device=/dev/video0:/dev/video0``
      - ``--device /dev/video0`` (same path in container)
    """
    mappings: list[tuple[str, str]] = []
    i = 0
    while i < len(params):
        param = params[i]
        raw_mapping: Optional[str] = None
        if param == "--device" and i + 1 < len(params):
            raw_mapping = params[i + 1]
            i += 1
        elif param.startswith("--device="):
            raw_mapping = param.split("=", 1)[1]

        if raw_mapping:
            parts = [part.strip() for part in raw_mapping.split(":")]
            if len(parts) >= 2:
                host_device = parts[0]
                container_device = parts[1]
            else:
                host_device = parts[0]
                container_device = parts[0]
            if host_device and container_device:
                mappings.append((host_device, container_device))
        i += 1
    return mappings


def _extract_docker_env_map(params: list[str]) -> dict[str, str]:
    """Extract ``KEY=value`` pairs provided via docker ``-e/--env`` params."""
    env_map: dict[str, str] = {}
    i = 0
    while i < len(params):
        param = params[i]
        raw_env: Optional[str] = None
        if param in {"-e", "--env"} and i + 1 < len(params):
            raw_env = params[i + 1]
            i += 1
        elif param.startswith("--env="):
            raw_env = param.split("=", 1)[1]

        if raw_env:
            key, sep, value = raw_env.partition("=")
            if sep and key:
                env_map[key] = value
        i += 1
    return env_map


def _resolve_bool_env_var(name: str, default: bool) -> bool:
    """Parse an optional runtime env var as boolean with sensible defaults."""
    raw_value = get_runtime_env_var(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_bridge_resolved_device(stdout: str, fallback: str) -> str:
    """Parse resolved device/source emitted by bridge command stdout."""
    payload = stdout.strip()
    if not payload:
        return fallback

    # Preferred format: JSON object with explicit key.
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        candidate = parsed.get("resolved_device") or parsed.get("video_source")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    # Alternate format: key-value line.
    for line in reversed(payload.splitlines()):
        candidate_line = line.strip()
        if not candidate_line:
            continue
        if candidate_line.startswith("resolved_device="):
            _, _, candidate = candidate_line.partition("=")
            candidate = candidate.strip()
            if candidate:
                return candidate

        # Last fallback: accept one-token values only to avoid accidentally
        # parsing verbose log lines as the resolved device.
        if " " not in candidate_line and "\t" not in candidate_line:
            return candidate_line

    return fallback


def _is_video_device_path(value: str) -> bool:
    """Return True when *value* looks like a Linux V4L2 /dev/video path."""
    return value.strip().startswith("/dev/video")


def _strip_video_device_mappings(params: list[str]) -> list[str]:
    """Remove ``--device`` mappings targeting ``/dev/video*`` from docker params."""
    rewritten: list[str] = []
    i = 0
    while i < len(params):
        param = params[i]

        if param == "--device" and i + 1 < len(params):
            mapping_value = params[i + 1]
            host_device, _, container_device = mapping_value.partition(":")
            if _is_video_device_path(host_device) or _is_video_device_path(container_device):
                i += 2
                continue
            rewritten.extend([param, mapping_value])
            i += 2
            continue

        if param.startswith("--device="):
            mapping_value = param.split("=", 1)[1]
            host_device, _, container_device = mapping_value.partition(":")
            if _is_video_device_path(host_device) or _is_video_device_path(container_device):
                i += 1
                continue

        rewritten.append(param)
        i += 1
    return rewritten


def _normalize_macos_bridge_candidates(
    candidates: Optional[list[str]],
) -> list[tuple[str, str]]:
    """Normalize additional macOS bridge candidates into host/container mappings."""
    normalized: list[tuple[str, str]] = []
    for candidate in candidates or []:
        if not isinstance(candidate, str):
            continue
        value = candidate.strip()
        if not value:
            continue
        if ":" in value:
            host_device, _, container_device = value.partition(":")
            host_device = host_device.strip()
            container_device = container_device.strip() or host_device
        else:
            host_device = value
            container_device = value
        if host_device and container_device:
            normalized.append((host_device, container_device))
    return normalized


def _docker_params_include_add_host(params: list[str]) -> bool:
    """Return True when docker params already include ``--add-host``."""
    for i, param in enumerate(params):
        if param.startswith("--add-host="):
            return True
        if param == "--add-host" and i + 1 < len(params):
            return True
    return False


def _build_driver_network_args(params: list[str]) -> list[str]:
    """Build docker network-related args with platform-aware defaults."""
    if platform.system() == "Darwin":
        # Docker Desktop on macOS does not expose Linux host networking semantics.
        # Ensure drivers can still reach host-side bridge services.
        if _docker_params_include_add_host(params):
            return []
        return ["--add-host", "host.docker.internal:host-gateway"]
    return ["--network", "host"]


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
) -> bool:
    """Pull and run a driver Docker container for a twin.

    The container is started in detached mode with ``--restart unless-stopped``
    so it persists across reboots.  Environment variables are passed so the
    driver can authenticate with the Cyberwave backend and know which twin it
    controls.

    Returns ``True`` if the container was started successfully.
    """
    if not shutil.which("docker"):
        logger.error("Docker is not installed or not in PATH")
        return False

    container_name = f"cyberwave-driver-{twin_uuid[:8]}"
    runtime_environment = (
        get_runtime_env_var("CYBERWAVE_ENVIRONMENT", DEFAULT_ENVIRONMENT) or DEFAULT_ENVIRONMENT
    ).lower()

    # check if the docker image has a tag first
    if ":" not in image:
        if runtime_environment != "production":
            # Example: cyberwaveos/cyberwave-edge-so101:dev
            image = f"{image}:{runtime_environment}"

    # Remove any existing container with the same name (idempotent re-runs)
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=30,
    )

    driver_alert_ctx = DriverStartingAlertContext(twin_uuid=twin_uuid, image=image)
    driver_alert_ctx.create()

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
            driver_alert_ctx.resolve()
        else:
            logger.error("Failed to pull docker image %s: %s", image, exc.stderr)
            driver_alert_ctx.fail_without_resolve(
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
            driver_alert_ctx.update_metadata({"phase": "pull_timeout_using_local"}, force=True)
            driver_alert_ctx.resolve()
        else:
            logger.error("Docker pull timed out for image: %s", image)
            driver_alert_ctx.fail_without_resolve(
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
            driver_alert_ctx.resolve()
        else:
            logger.error("Docker pull failed for image %s: %s", image, exc)
            driver_alert_ctx.fail_without_resolve(
                f"Could not pull driver image {image}: {exc}",
                phase="pull_oserror",
            )
            return False
    else:
        driver_alert_ctx.update_metadata({"phase": "pull_complete"}, force=True)
        driver_alert_ctx.resolve()

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
    macos_bridge_mappings = _normalize_macos_bridge_candidates(macos_bridge_device_candidates)

    # Determine USB/IP state early so the bridge function can skip video
    # devices that USB/IP will handle transparently inside the container.
    usbip_active = platform.system() == "Darwin" and _is_usbip_server_running()

    macos_bridge_ok, macos_resolved_devices = _run_macos_device_bridge_commands(
        params=params,
        twin_uuid=twin_uuid,
        container_name=container_name,
        additional_device_mappings=macos_bridge_mappings,
        usbip_active=usbip_active,
    )
    if not macos_bridge_ok:
        return False

    if platform.system() == "Darwin":
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
        container_env["CYBERWAVE_BASE_URL"] = base_url
    mqtt_host = get_runtime_env_var("CYBERWAVE_MQTT_HOST")
    if mqtt_host:
        container_env["CYBERWAVE_MQTT_HOST"] = mqtt_host
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
        if has_video_devices:
            container_env.setdefault("CYBERWAVE_USBIP_VIDEO_TIMEOUT_SECS", "8")

    # Optional MJPEG camera stream fallback for macOS. When the user has
    # configured a stream URL (e.g. because USB/IP bandwidth is too low
    # for video), pass it to the container so cv2.VideoCapture uses the
    # HTTP stream instead of a /dev/video* device.
    if platform.system() == "Darwin":
        camera_stream_url = get_runtime_env_var("CYBERWAVE_MACOS_CAMERA_STREAM_URL")
        if camera_stream_url and camera_stream_url.strip():
            camera_stream_url = camera_stream_url.strip()
            if "CYBERWAVE_METADATA_VIDEO_DEVICE" not in explicit_params_env:
                container_env.setdefault(
                    "CYBERWAVE_METADATA_VIDEO_DEVICE", camera_stream_url
                )
            logger.info(
                "macOS camera stream URL configured: %s",
                camera_stream_url,
            )

    env_vars: List[str] = []
    for key, value in container_env.items():
        env_vars += ["-e", f"{key}={value}"]

    twin_json_file = CONFIG_DIR / f"{twin_uuid}.json"
    if twin_json_file.is_file():
        env_vars += ["-v", f"{twin_json_file}:/app/{twin_uuid}.json"]
        env_vars += ["-e", f"CYBERWAVE_TWIN_JSON_FILE=/app/{twin_uuid}.json"]
    # sync the edge config directory into the container
    env_vars += ["-v", f"{CONFIG_DIR}:/app/.cyberwave"]

    network_args = _build_driver_network_args(params)

    # Run the container.  --init injects tini as PID 1 so signals are
    # forwarded properly even if the driver blocks on serial/USB I/O.
    # --stop-timeout limits the SIGTERM grace period so Docker escalates
    # to SIGKILL faster when USB devices cause unresponsive I/O.
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
        *pid_args,
        *network_args,
        "--name",
        container_name,
        *params,
        *env_vars,
        image,
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
                return True
            if status in {"restarting", "exited", "dead"}:
                logger.error(
                    "Driver container %s failed to start cleanly (status=%s error=%s)",
                    container_name,
                    status,
                    str(state.get("Error", "")).strip() or "none",
                )
                return False
            time.sleep(1.0)

        logger.warning(
            "Driver container %s did not reach a stable running state within startup probe window",
            container_name,
        )
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to start container %s: %s", container_name, exc.stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Docker run timed out for image: %s", image)
        return False


def _stream_container_logs(
    container_name: str,
    *,
    twin_uuid: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    """Stream container logs into this service logger in the background."""
    existing = _CONTAINER_LOG_THREADS.get(container_name)
    if existing and existing.is_alive():
        return

    thread = threading.Thread(
        target=_follow_container_logs,
        args=(container_name,),
        kwargs={"twin_uuid": twin_uuid, "token": token},
        name=f"docker-logs-{container_name}",
        daemon=True,
    )
    _CONTAINER_LOG_THREADS[container_name] = thread
    thread.start()


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


def reconcile_driver_log_streams() -> int:
    """Ensure active driver containers have an attached log-forwarding thread."""
    running_containers = _list_running_driver_containers()
    running_set = set(running_containers)

    # Drop finished thread handles so we can re-attach later if needed.
    stale = [
        name
        for name, thread in _CONTAINER_LOG_THREADS.items()
        if not thread.is_alive() and name not in running_set
    ]
    for name in stale:
        _CONTAINER_LOG_THREADS.pop(name, None)

    attached = 0
    token: Optional[str] = None
    for container_name in running_containers:
        thread = _CONTAINER_LOG_THREADS.get(container_name)
        if thread and thread.is_alive():
            attached += 1
            continue
        if token is None:
            token = load_token()
        twin_uuid = _CONTAINER_TWIN_MAP.get(container_name)
        _stream_container_logs(container_name, twin_uuid=twin_uuid, token=token)
        thread = _CONTAINER_LOG_THREADS.get(container_name)
        if thread and thread.is_alive():
            attached += 1
    return attached


def _parse_log_level(message: str) -> str:
    """Best-effort extraction of log level from a driver log line."""
    upper = message[:80].upper()
    for level in ("ERROR", "CRITICAL", "WARNING", "WARN", "DEBUG", "INFO"):
        if level in upper:
            return "WARNING" if level == "WARN" else level
    return "INFO"


def _build_driver_log_payload(
    message: str,
    container_name: str,
    *,
    driver_image: str | None = None,
) -> dict[str, Any]:
    """Build the MQTT payload for a forwarded driver log line."""
    payload: dict[str, Any] = {
        "type": "driver_log",
        "message": message,
        "level": _parse_log_level(message),
        "container_name": container_name,
        "source": "edge",
        "timestamp": time.time(),
        "edge_core_version": EDGE_CORE_VERSION,
    }
    if CYBERWAVE_SDK_VERSION:
        payload["sdk_version"] = CYBERWAVE_SDK_VERSION
    if driver_image:
        payload["driver_image"] = driver_image
    return payload


def _resolve_driver_log_publish_context(
    *,
    twin_uuid: Optional[str],
    token: Optional[str],
) -> tuple[Optional[Any], Optional[str]]:
    """Create the MQTT publish context used for driver log forwarding."""
    if not twin_uuid or not token:
        return None, None

    mqtt_client = _get_shared_mqtt_client(token)
    if not mqtt_client:
        return None, None

    prefix = mqtt_client.mqtt.topic_prefix
    mqtt_topic = f"{prefix}cyberwave/twin/{twin_uuid}/driverlog"
    return mqtt_client, mqtt_topic


def _publish_driver_log_message(
    message: str,
    container_name: str,
    *,
    mqtt_client: Optional[Any] = None,
    mqtt_topic: Optional[str] = None,
    driver_image: str | None = None,
) -> None:
    """Publish a driver log line to MQTT when a publish context is available."""
    if not mqtt_client or not mqtt_topic:
        return

    try:
        mqtt_client.mqtt.publish(
            mqtt_topic,
            _build_driver_log_payload(
                message,
                container_name,
                driver_image=driver_image,
            ),
        )
    except Exception:
        logger.debug(
            "Failed to publish driver log to MQTT for %s",
            container_name,
            exc_info=True,
        )


def _log_and_publish_driver_message(
    message: str,
    container_name: str,
    *,
    mqtt_client: Optional[Any] = None,
    mqtt_topic: Optional[str] = None,
    driver_image: str | None = None,
) -> None:
    """Mirror a driver log line into local service logs and MQTT."""
    logger.info("[driver:%s] %s", container_name, message)
    _publish_driver_log_message(
        message,
        container_name,
        mqtt_client=mqtt_client,
        mqtt_topic=mqtt_topic,
        driver_image=driver_image,
    )


def _iter_stream_messages(stream: Any) -> Any:
    """Yield messages delimited by newlines or carriage returns."""
    buffer: list[str] = []
    while True:
        chunk = stream.read(1)
        if chunk == "":
            break
        if chunk in {"\r", "\n"}:
            message = "".join(buffer).strip()
            if message:
                yield message
            buffer.clear()
            continue
        buffer.append(chunk)

    message = "".join(buffer).strip()
    if message:
        yield message


def _pull_docker_image_with_progress(
    image: str,
    *,
    container_name: str,
    twin_uuid: str,
    token: str,
    timeout: int = 600,
    driver_alert_ctx: Optional[DriverStartingAlertContext] = None,
) -> None:
    """Pull a driver image while streaming progress to local logs and MQTT."""
    mqtt_client, mqtt_topic = _resolve_driver_log_publish_context(
        twin_uuid=twin_uuid,
        token=token,
    )
    logger.info("Pulling docker image: %s", image)
    if driver_alert_ctx:
        driver_alert_ctx.update_metadata(
            {"phase": "pulling", "last_message": f"docker pull started for {image}"},
            force=True,
        )
    _log_and_publish_driver_message(
        f"docker pull started for image {image}",
        container_name,
        mqtt_client=mqtt_client,
        mqtt_topic=mqtt_topic,
        driver_image=image,
    )

    try:
        process = subprocess.Popen(
            ["docker", "pull", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        if driver_alert_ctx:
            driver_alert_ctx.update_metadata(
                {"phase": "pull_spawn_failed", "last_error": str(exc)[:500]},
                force=True,
            )
        _log_and_publish_driver_message(
            f"docker pull failed for image {image}: {exc}",
            container_name,
            mqtt_client=mqtt_client,
            mqtt_topic=mqtt_topic,
            driver_image=image,
        )
        raise

    recent_messages: deque[str] = deque(maxlen=20)
    last_message: Optional[str] = None
    try:
        if process.stdout:
            for message in _iter_stream_messages(process.stdout):
                if message == last_message:
                    continue
                last_message = message
                recent_messages.append(message)
                if driver_alert_ctx:
                    driver_alert_ctx.update_metadata(
                        {
                            "phase": "downloading",
                            "last_docker_pull_line": message[:500],
                            "recent_pull_lines": list(recent_messages)[-5:],
                        },
                    )
                _log_and_publish_driver_message(
                    f"docker pull: {message}",
                    container_name,
                    mqtt_client=mqtt_client,
                    mqtt_topic=mqtt_topic,
                    driver_image=image,
                )

        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        if driver_alert_ctx:
            driver_alert_ctx.update_metadata(
                {
                    "phase": "pull_timed_out",
                    "last_message": f"docker pull timed out for {image}",
                },
                force=True,
            )
        _log_and_publish_driver_message(
            f"docker pull timed out for image {image}",
            container_name,
            mqtt_client=mqtt_client,
            mqtt_topic=mqtt_topic,
            driver_image=image,
        )
        raise

    if return_code != 0:
        error_output = "\n".join(recent_messages) or f"docker pull exited with code {return_code}"
        if driver_alert_ctx:
            driver_alert_ctx.update_metadata(
                {
                    "phase": "pull_exit_error",
                    "last_error": error_output[:500],
                    "recent_pull_lines": list(recent_messages)[-5:],
                },
                force=True,
            )
        _log_and_publish_driver_message(
            f"docker pull failed for image {image}: {error_output}",
            container_name,
            mqtt_client=mqtt_client,
            mqtt_topic=mqtt_topic,
            driver_image=image,
        )
        raise subprocess.CalledProcessError(
            return_code,
            ["docker", "pull", image],
            stderr=error_output,
        )

    if driver_alert_ctx:
        driver_alert_ctx.update_metadata(
            {
                "phase": "pull_stream_finished",
                "last_message": f"docker pull completed for {image}",
            },
            force=True,
        )
    _log_and_publish_driver_message(
        f"docker pull completed for image {image}",
        container_name,
        mqtt_client=mqtt_client,
        mqtt_topic=mqtt_topic,
        driver_image=image,
    )


def _follow_container_logs(
    container_name: str,
    *,
    twin_uuid: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    """Follow `docker logs -f` and forward lines to the service logger.

    When *twin_uuid* and *token* are provided, each log line is also
    published to the backend via MQTT as a ``driver_log`` event.
    """
    if not shutil.which("docker"):
        logger.warning("Cannot stream logs: Docker is not installed")
        return

    logger.info("Forwarding logs for container %s to service logs", container_name)
    debug_log_stream = logger.isEnabledFor(logging.DEBUG)
    received_lines = 0

    mqtt_client: Optional[Any] = None
    mqtt_topic: Optional[str] = None
    driver_image: Optional[str] = None
    mqtt_client, mqtt_topic = _resolve_driver_log_publish_context(
        twin_uuid=twin_uuid,
        token=token,
    )
    if mqtt_topic:
        logger.info("Driver logs for %s will be published to %s", container_name, mqtt_topic)
        driver_image = _resolve_container_driver_image(
            _inspect_driver_container(container_name)
        )

    try:
        process = subprocess.Popen(
            ["docker", "logs", "-f", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        logger.warning("Failed to start docker log streaming for %s: %s", container_name, exc)
        return

    try:
        if not process.stdout:
            logger.warning("No stdout stream when following logs for %s", container_name)
            return

        for line in process.stdout:
            message = line.rstrip()
            if message:
                received_lines += 1
                _log_and_publish_driver_message(
                    message,
                    container_name,
                    mqtt_client=mqtt_client,
                    mqtt_topic=mqtt_topic,
                    driver_image=driver_image,
                )

                if debug_log_stream:
                    logger.debug(
                        "Container log line received (container=%s, line=%d, chars=%d)",
                        container_name,
                        received_lines,
                        len(message),
                    )
    except Exception as exc:
        logger.warning("Error while streaming logs for %s: %s", container_name, exc)
    finally:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        logger.info(
            "Stopped forwarding logs for container %s (lines_received=%d)",
            container_name,
            received_lines,
        )


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


def _start_bootstrap_edge_health_publisher(
    token: str,
    twin_uuids: list[str],
    *,
    edge_id: str,
) -> bool:
    """Start (or refresh) a lightweight edge_health publisher for linked twins."""
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

    with _EDGE_HEALTH_CHECK_LOCK:
        if _EDGE_HEALTH_CHECK is not None:
            existing = list(getattr(_EDGE_HEALTH_CHECK, "twin_uuids", []) or [])
            _EDGE_HEALTH_CHECK.twin_uuids = list(dict.fromkeys(existing + normalized_twin_uuids))
            _EDGE_HEALTH_CHECK.edge_id = edge_id
            _EDGE_HEALTH_CHECK.start()
            return True

        _EDGE_HEALTH_CHECK = EdgeHealthCheck(
            mqtt_client=mqtt_client.mqtt,
            twin_uuids=normalized_twin_uuids,
            edge_id=edge_id,
            interval=EDGE_HEALTH_PUBLISH_INTERVAL_SECONDS,
        )
        _EDGE_HEALTH_CHECK.start()
        logger.info(
            "Started bootstrap edge health publisher for %d twin(s) (edge_id=%s)",
            len(normalized_twin_uuids),
            edge_id,
        )
        return True


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

    results: List[Dict[str, Any]] = []

    for twin in twins:
        twin_uuid = str(getattr(twin, "uuid", ""))
        if not twin_uuid:
            continue

        # The edge writes edge_fingerprint into twin metadata when the user
        # selects which twins this edge controls.  Match on that field.
        twin_metadata = twin.metadata if isinstance(twin.metadata, dict) else {}
        if twin_metadata.get("edge_fingerprint") != fingerprint:
            continue

        logger.info(
            "Twin '%s' (%s) is linked to this edge (fingerprint=%s)",
            twin.name,
            twin_uuid,
            fingerprint,
        )

        # Get the asset to check for driver_docker_image
        asset = assets_by_twin_uuid.get(twin_uuid)
        if asset is None:
            asset_uuid = getattr(twin, "asset_uuid", None) or getattr(twin, "asset_id", "")
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
            # _check_and_alert_sensors_devices(
            #     twin_uuid,
            #     twin.name or f"twin-{twin_uuid[:8]}",
            #     asset,
            #     twin_metadata,
            # )
            _persist_twin_json_for_driver(twin, twin_uuid, asset)
            continue

        drivers = twin_metadata.get("drivers")
        asset_metadata = getattr(asset, "metadata", {}) or {}
        if not isinstance(asset_metadata, dict):
            asset_metadata = {}
        if not drivers:
            # try fallback to asset metadata
            drivers = asset_metadata.get("drivers")
            if not drivers:
                # Check if this twin is attached to another twin (e.g. camera on SO101).
                # If so, skip running a driver but still write the JSON file so the
                # parent driver can discover and use it.
                if attach_to:
                    # Twin is attached to another - write JSON and skip (parent driver handles it)
                    logger.info(
                        "Twin '%s' has no driver but is attached to %s; "
                        "writing JSON for parent driver to use",
                        twin.name,
                        attach_to,
                    )
                    # _check_and_alert_sensors_devices(
                    #     twin_uuid,
                    #     twin.name or f"twin-{twin_uuid[:8]}",
                    #     asset,
                    #     twin_metadata,
                    # )
                    _persist_twin_json_for_driver(twin, twin_uuid, asset)
                    continue

                # No drivers and not attached to anything - this is an error
                logger.warning("No drivers specified in asset metadata for twin '%s'", twin.name)
                _send_alert_for_twin(
                    twin_uuid,
                    "No drivers specified",
                    f"No drivers specified in asset metadata for twin '{twin.name}'",
                    "error",
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
        driver_image, driver_params = _get_best_driver_image_and_params(
            drivers,
            child_registry_ids=child_registry_ids_by_parent.get(twin_uuid, set()),
        )

        # _check_and_alert_sensors_devices(
        #     twin_uuid,
        #     twin.name or f"twin-{twin_uuid[:8]}",
        #     asset,
        #     twin_metadata,
        # )

        _persist_twin_json_for_driver(twin, twin_uuid, asset)

        if not driver_image:
            logger.info("No driver_docker_image in asset metadata for twin '%s'", twin.name)
            _send_alert_for_twin(
                twin_uuid,
                "No driver_docker_image in asset metadata",
                f"No driver_docker_image in asset metadata for twin '{twin.name}'",
                "error",
            )
            raise ValueError(
                f"No drivers specified in asset metadata for paired twin '{twin.name}'"
            )

        child_camera_twin_uuids = list(dict.fromkeys(camera_children_by_parent.get(twin_uuid, [])))
        if child_camera_twin_uuids:
            logger.info(
                "Passing %d child camera twin UUID(s) to parent twin '%s': %s",
                len(child_camera_twin_uuids),
                twin.name,
                ",".join(child_camera_twin_uuids),
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

        logger.info("Running driver docker image %s for twin '%s'", driver_image, twin.name)
        try:
            success = _run_docker_image(
                driver_image,
                driver_params,
                twin_uuid=twin_uuid,
                token=token,
                child_camera_twin_uuids=child_camera_twin_uuids,
                macos_bridge_device_candidates=macos_bridge_candidates,
            )
            results.append(
                {
                    "twin_uuid": twin_uuid,
                    "twin_name": twin.name,
                    "driver_image": driver_image,
                    "success": success,
                }
            )
            if not success:
                try:
                    startup_failure_message = (
                        f"Driver image '{driver_image}' for twin '{twin.name}' failed to start "
                        "on this edge. Check that required hardware is connected and accessible. "
                        f"Troubleshooting: {DRIVER_TROUBLESHOOTING_URL}"
                    )
                    _send_alert_for_twin(
                        twin_uuid,
                        "Driver failed to start",
                        startup_failure_message,
                        "driver_start_failure",
                        severity="error",
                    )
                except Exception as alert_exc:
                    logger.warning(
                        "Could not send startup-failure alert for twin %s: %s",
                        twin_uuid,
                        alert_exc,
                    )
        except Exception as exc:
            _send_alert_for_twin(
                twin_uuid,
                "Failed to run driver docker image",
                f"Failed to run driver docker image for twin '{twin.name}': {exc}",
                "error",
            )
            logger.error(
                "Failed to run driver docker image %s for twin '%s': %s",
                driver_image,
                twin.name,
                exc,
            )

    _start_worker_after_drivers(token, environment_uuid, list(linked_twin_uuids))
    return results


def _start_worker_after_drivers(
    token: str,
    environment_uuid: str,
    twin_uuids: list[str],
) -> None:
    """Start the worker container after driver startup (best-effort, non-blocking)."""
    try:
        from .worker_manager import WorkerManager

        worker_manager = WorkerManager(
            config_dir=CONFIG_DIR,
            environment_uuid=environment_uuid,
            token=token,
            twin_uuids=twin_uuids,
        )
        worker_manager.start()
    except Exception as exc:
        logger.warning("Failed to start worker container after driver startup: %s", exc)


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


def _get_best_driver_image_and_params(
    drivers: Dict[str, Dict[str, Any]],
    child_registry_ids: Optional[set[str]] = None,
) -> tuple[str, list[str]]:
    """
    Given a list of drivers specified in the metadata of the asset,
    and given the hardware where the edge is running,
    Returns:
    - The best driver to run.
    - A list of parameters to pass to the driver when doing docker run
    If any non-default driver key matches one of the child asset registry IDs,
    that driver is preferred over ``default``.

    "drivers": {
        "default": {
            "docker_image": "helloworld",
            "version": "0.1.0",
            "params": ["--param1", "--param2"],
        },
        "mac": {
            "docker_image": "helloworld",
            "version": "0.1.0",
            "params": ["--param1", "--param2"],
        },
    },
    """
    def _extract_driver_image_and_params(
        driver_name: str,
        driver_config: Any,
    ) -> tuple[str, list[str]]:
        if not isinstance(driver_config, dict):
            raise ValueError(f"Invalid config for driver '{driver_name}'")
        if not driver_config.get("docker_image") or not isinstance(
            driver_config["docker_image"], str
        ):
            raise ValueError(f"No docker_image specified for driver '{driver_name}'")
        raw_params = driver_config.get("params")
        if raw_params is None:
            params: list[str] = []
        elif isinstance(raw_params, list) and all(isinstance(param, str) for param in raw_params):
            params = raw_params
        else:
            raise ValueError(f"Invalid params for driver '{driver_name}'")
        return driver_config["docker_image"], params

    def _resolve_platform_driver_keys() -> list[str]:
        system_name = platform.system().lower()
        machine_name = platform.machine().lower()

        platform_aliases: list[str]
        if system_name == "darwin":
            platform_aliases = ["darwin", "macos", "mac", "osx"]
        elif system_name == "linux":
            platform_aliases = ["linux"]
        elif system_name == "windows":
            platform_aliases = ["windows", "win32"]
        else:
            platform_aliases = [system_name]

        if machine_name:
            machine_specific = [f"{alias}-{machine_name}" for alias in platform_aliases]
            return machine_specific + platform_aliases
        return platform_aliases

    normalized_child_registry_ids = {
        registry_id.strip()
        for registry_id in (child_registry_ids or set())
        if isinstance(registry_id, str) and registry_id.strip()
    }
    if normalized_child_registry_ids and len(drivers) > 1:
        for driver_name, driver_config in drivers.items():
            if driver_name == "default":
                continue
            if driver_name not in normalized_child_registry_ids:
                continue
            return _extract_driver_image_and_params(driver_name, driver_config)

    for platform_key in _resolve_platform_driver_keys():
        if platform_key not in drivers:
            continue
        return _extract_driver_image_and_params(platform_key, drivers[platform_key])

    default_driver = drivers.get("default")
    return _extract_driver_image_and_params("default", default_driver)


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
    """Best-effort stop of the worker container before a full edge restart."""
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


def _perform_edge_core_restart(token: str) -> dict[str, Any]:
    """Execute restart workflow: cleanup local state and re-run driver startup."""
    _stop_worker_container_for_restart()
    removed_json_files = _remove_cached_twin_json_files()
    removed_containers = _stop_and_prune_driver_containers()

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

    summary = {
        "environment_uuid": environment_uuid,
        "removed_twin_json_files": removed_json_files,
        "removed_driver_containers": removed_containers,
        "drivers_started": started,
        "drivers_discovered": len(results),
    }
    logger.info(
        "Edge-core restart complete: env=%s removed_json=%d removed_containers=%d started=%d/%d",
        environment_uuid,
        len(removed_json_files),
        len(removed_containers),
        started,
        len(results),
    )
    return summary


def _run_edge_core_restart_worker(request_id: str) -> None:
    """Execute restart flow in a background thread."""
    global _EDGE_RESTART_IN_PROGRESS

    with _EDGE_RESTART_LOCK:
        if _EDGE_RESTART_IN_PROGRESS:
            logger.info(
                "Ignoring restart request %s: restart already in progress",
                request_id or "no-request-id",
            )
            return
        _EDGE_RESTART_IN_PROGRESS = True

    try:
        token = load_token()
        if not token:
            logger.warning(
                "Ignoring restart request %s: no token available",
                request_id or "no-request-id",
            )
            return
        _perform_edge_core_restart(token)
    except Exception:
        logger.exception(
            "Edge-core restart request %s failed",
            request_id or "no-request-id",
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

    logger.info("Received edge restart command request_id=%s", request_id or "none")
    worker = threading.Thread(
        target=_run_edge_core_restart_worker,
        args=(request_id,),
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
    """Build safe payload for PUT /api/v1/twins/{uuid} from local twin JSON."""
    payload = {
        key: twin_json_data[key] for key in _TWIN_UPDATE_ALLOWED_FIELDS if key in twin_json_data
    }

    # Drivers receive twin + asset in one file. If the SDK payload does not
    # include asset_uuid, infer it from the embedded asset object.
    if "asset_uuid" not in payload:
        asset_data = twin_json_data.get("asset")
        if isinstance(asset_data, dict):
            asset_uuid = asset_data.get("uuid")
            if isinstance(asset_uuid, str) and asset_uuid.strip():
                payload["asset_uuid"] = asset_uuid.strip()

    return payload


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


def reconcile_twin_json_file_sync() -> dict[str, int]:
    """Detect and sync local twin JSON changes to the backend."""
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
    }
    if not changed_candidates:
        return summary

    token = load_token()
    if not token:
        logger.warning("Cannot sync changed twin JSON files: no API token available")
        return summary

    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    try:
        client = Cyberwave(base_url=base_url, token=token)
    except Exception as exc:
        logger.warning("Cannot sync changed twin JSON files: failed to create client (%s)", exc)
        return summary

    for twin_uuid, twin_json_file, checksum in changed_candidates:
        if _sync_twin_json_file_with_backend(client, twin_uuid, twin_json_file):
            _TWIN_FILE_CHECKSUMS[twin_uuid] = checksum
            summary["synced"] += 1

    return summary


def run_startup_checks() -> bool:
    """Execute every boot-time check in sequence.

    Prints a Rich-formatted report to the console.
    Returns ``True`` only when **all** checks pass.
    """
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

            # 7 — workflow worker sync (pull generated wf_*.py files from backend)
            if linked_twin_uuids:
                _t0 = time.perf_counter()
                base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
                sync_summary = _sync_workers_for_twins(
                    token=token,
                    twin_uuids=linked_twin_uuids,
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

    console.print("\n[green]All startup checks passed.[/green]\n")
    return True


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
        twin_uuids = _list_linked_twin_uuids_for_fingerprint(
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


# Interval at which worker sync reconciliation runs (every N runtime loops).
# At 15 s per loop this defaults to ~5 minutes.
_WORKER_SYNC_INTERVAL_LOOPS = int(
    os.getenv("CYBERWAVE_WORKER_SYNC_INTERVAL_LOOPS", "20")
)
_worker_sync_loop_counter = 0


def run_runtime_loop() -> None:
    """Keep edge-core alive and continuously reconcile driver log forwarding."""
    global _worker_sync_loop_counter

    logger.info(
        "Entering edge-core runtime loop (interval=%.1fs)",
        LOG_FOLLOWER_RECONCILE_INTERVAL_SECONDS,
    )

    worker_watcher: Optional[Any] = None

    while True:
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
        twin_sync_summary = reconcile_twin_json_file_sync()
        logger.debug(
            "Twin JSON sync reconcile complete (tracked=%d, changed=%d, synced=%d)",
            twin_sync_summary["tracked"],
            twin_sync_summary["changed"],
            twin_sync_summary["synced"],
        )

        # Reconcile worker file changes and health probes each cycle.
        try:
            worker_watcher = _reconcile_worker_watcher(worker_watcher)
        except Exception:
            logger.exception("Unexpected error in worker file reconciliation")

        try:
            ensure_edge_command_subscription()
        except Exception:
            logger.exception("Unexpected error while ensuring edge command subscription")

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
            except Exception:
                logger.exception("Unexpected error during worker sync reconcile")

        time.sleep(LOG_FOLLOWER_RECONCILE_INTERVAL_SECONDS)


def _reconcile_worker_watcher(
    existing_watcher: Optional[Any],
) -> Optional[Any]:
    """Lazily create and run the worker file watcher + health monitor each reconcile cycle."""
    from .model_manager import ModelManager
    from .worker_health import WorkerHealthMonitor
    from .worker_manager import WorkerManager
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
                twin_uuids = _list_linked_twin_uuids_for_fingerprint(
                    token, environment_uuid, fingerprint
                )
        except Exception:
            logger.debug("Could not resolve twin_uuids for worker watcher")

        worker_manager = WorkerManager(
            config_dir=CONFIG_DIR,
            environment_uuid=environment_uuid,
            token=token,
            twin_uuids=twin_uuids,
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
        existing_watcher = WorkerWatcher(
            workers_dir=workers_dir,
            worker_manager=worker_manager,
            model_manager=model_manager,
        )
        logger.debug("Worker file watcher + health monitor initialised for %s", workers_dir)

    # Run health probe each cycle to detect spontaneous exits / crash loops.
    existing_watcher.check_health()

    existing_watcher.reconcile_worker_files()
    return existing_watcher
