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
import json
import logging
import os
import platform
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from cyberwave import Cyberwave
from cyberwave.fingerprint import generate_fingerprint
from rich.console import Console


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
    if not credentials_file.exists():
        return

    try:
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

# Track active log streaming threads per container to avoid duplicates.
_CONTAINER_LOG_THREADS: dict[str, threading.Thread] = {}

# Map container names to twin UUIDs so log threads can publish telemetry.
_CONTAINER_TWIN_MAP: dict[str, str] = {}

# Shared MQTT client for publishing driver log telemetry.
_shared_mqtt_client: Optional[Any] = None
_shared_mqtt_lock = threading.Lock()

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


def _get_shared_mqtt_client(token: str) -> Any:
    """Return a shared MQTT client, creating and connecting it on first call."""
    global _shared_mqtt_client
    with _shared_mqtt_lock:
        if _shared_mqtt_client is not None and _shared_mqtt_client.mqtt.connected:
            return _shared_mqtt_client
        base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
        try:
            client = Cyberwave(base_url=base_url, api_key=token)
            client.mqtt.connect()
            _shared_mqtt_client = client
            logger.info("Shared MQTT client connected for log forwarding")
            return client
        except Exception as exc:
            logger.warning("Failed to create shared MQTT client: %s", exc)
            return None


def load_token() -> Optional[str]:
    """Load the API token from the edge config credentials file.

    Returns the token string, or ``None`` if the file is missing or
    cannot be parsed.
    """
    if not CREDENTIALS_FILE.exists():
        logger.warning("Credentials file not found: %s", CREDENTIALS_FILE)
        return None
    try:
        with open(CREDENTIALS_FILE) as f:
            data = json.load(f)
        token = data.get("token") or None
        if token:
            masked = f"{token[:6]}…{token[-4:]}" if len(token) > 12 else "***"
            logger.info("Loaded token from %s (token: %s)", CREDENTIALS_FILE, masked)
        else:
            logger.warning(
                "Credentials file %s exists but has no 'token' field. Keys present: %s",
                CREDENTIALS_FILE,
                list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
        return token
    except (json.JSONDecodeError, OSError) as exc:
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
    mqtt_host = get_runtime_env_var("CYBERWAVE_MQTT_HOST", "(default)")
    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
    logger.info(
        "Attempting MQTT connection (base_url=%s, mqtt_host=%s)",
        base_url,
        mqtt_host,
    )
    try:
        client = Cyberwave(base_url=base_url, api_key=token)
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


def _run_docker_image(
    image: str,
    params: list[str],
    *,
    twin_uuid: str,
    token: str,
    child_camera_twin_uuids: Optional[list[str]] = None,
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

    # Pull the image
    logger.info("Pulling docker image: %s", image)
    try:
        subprocess.run(
            ["docker", "pull", image],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as exc:
        if _docker_image_exists_locally(image):
            logger.warning(
                "Failed to pull docker image %s (%s); using local image copy",
                image,
                (exc.stderr or "").strip() or "unknown error",
            )
        else:
            logger.error("Failed to pull docker image %s: %s", image, exc.stderr)
            return False
    except subprocess.TimeoutExpired:
        if _docker_image_exists_locally(image):
            logger.warning(
                "Docker pull timed out for image %s; using local image copy",
                image,
            )
        else:
            logger.error("Docker pull timed out for image: %s", image)
            return False

    # Build env vars for the container
    container_env: dict[str, str] = {
        "CYBERWAVE_TWIN_UUID": twin_uuid,
        "CYBERWAVE_API_KEY": token,
    }
    if child_camera_twin_uuids:
        normalized_child_uuids = [str(child_uuid).strip() for child_uuid in child_camera_twin_uuids]
        normalized_child_uuids = [child_uuid for child_uuid in normalized_child_uuids if child_uuid]
        if normalized_child_uuids:
            child_uuids_csv = ",".join(dict.fromkeys(normalized_child_uuids))
            container_env["CYBERWAVE_CHILD_TWIN_UUIDS"] = child_uuids_csv

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
            container_env.setdefault(key, value)

    # Driver reads setup.json from so101_lib under this dir (mounted CONFIG_DIR)
    container_env["CYBERWAVE_EDGE_CONFIG_DIR"] = "/app/.cyberwave"

    env_vars: List[str] = []
    for key, value in container_env.items():
        env_vars += ["-e", f"{key}={value}"]

    twin_json_file = CONFIG_DIR / f"{twin_uuid}.json"
    if twin_json_file.is_file():
        env_vars += ["-v", f"{twin_json_file}:/app/{twin_uuid}.json"]
        env_vars += ["-e", f"CYBERWAVE_TWIN_JSON_FILE=/app/{twin_uuid}.json"]
    # sync the edge config directory into the container
    env_vars += ["-v", f"{CONFIG_DIR}:/app/.cyberwave"]

    # Run the container
    cmd = [
        "docker",
        "run",
        "--detach",
        "--restart",
        "unless-stopped",
        "--privileged",
        "--network",
        "host",
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

    token = load_token()
    attached = 0
    for container_name in running_containers:
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


def _follow_container_logs(
    container_name: str,
    *,
    twin_uuid: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    """Follow `docker logs -f` and forward lines to the service logger.

    When *twin_uuid* and *token* are provided, each log line is also
    published to the backend via MQTT as a ``driver_log`` telemetry event.
    """
    if not shutil.which("docker"):
        logger.warning("Cannot stream logs: Docker is not installed")
        return

    logger.info("Forwarding logs for container %s to service logs", container_name)
    debug_log_stream = logger.isEnabledFor(logging.DEBUG)
    received_lines = 0

    mqtt_client: Optional[Any] = None
    mqtt_topic: Optional[str] = None
    if twin_uuid and token:
        mqtt_client = _get_shared_mqtt_client(token)
        if mqtt_client:
            prefix = mqtt_client.mqtt.topic_prefix
            mqtt_topic = f"{prefix}cyberwave/twin/{twin_uuid}/telemetry"
            logger.info("Driver logs for %s will be published to %s", container_name, mqtt_topic)

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
                logger.info("[driver:%s] %s", container_name, message)

                if mqtt_client and mqtt_topic:
                    try:
                        mqtt_client.mqtt.publish(
                            mqtt_topic,
                            {
                                "type": "driver_log",
                                "message": message,
                                "level": _parse_log_level(message),
                                "container_name": container_name,
                                "source": "edge",
                                "timestamp": time.time(),
                            },
                        )
                    except Exception:
                        logger.debug(
                            "Failed to publish driver log to MQTT for %s",
                            container_name,
                            exc_info=True,
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
                    _check_and_alert_sensors_devices(
                        twin_uuid,
                        twin.name or f"twin-{twin_uuid[:8]}",
                        asset,
                        twin_metadata,
                    )
                    _persist_twin_json_for_driver(twin, twin_uuid, asset)
                    continue

                # No drivers and not attached to anything - this is an error
                logger.warning("No drivers specified in asset metadata for twin '%s'", twin.name)
                _send_alert_for_twin(
                    twin_uuid,
                    "No drivers specified",
                    "No drivers specified in asset metadata for twin '%s'",
                    "error",
                )
                raise ValueError(
                    "No drivers specified in asset metadata for paired twin '%s'", twin.name
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

        _check_and_alert_sensors_devices(
            twin_uuid,
            twin.name or f"twin-{twin_uuid[:8]}",
            asset,
            twin_metadata,
        )

        _persist_twin_json_for_driver(twin, twin_uuid, asset)

        if not driver_image:
            logger.info("No driver_docker_image in asset metadata for twin '%s'", twin.name)
            _send_alert_for_twin(
                twin_uuid,
                "No driver_docker_image in asset metadata",
                "No driver_docker_image in asset metadata for twin '%s'",
                "error",
            )
            raise ValueError(
                "No drivers specified in asset metadata for paired twin '%s'", twin.name
            )

        child_camera_twin_uuids = list(dict.fromkeys(camera_children_by_parent.get(twin_uuid, [])))
        if child_camera_twin_uuids:
            logger.info(
                "Passing %d child camera twin UUID(s) to parent twin '%s': %s",
                len(child_camera_twin_uuids),
                twin.name,
                ",".join(child_camera_twin_uuids),
            )

        logger.info("Running driver docker image %s for twin '%s'", driver_image, twin.name)
        try:
            success = _run_docker_image(
                driver_image,
                driver_params,
                twin_uuid=twin_uuid,
                token=token,
                child_camera_twin_uuids=child_camera_twin_uuids,
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
                "Failed to run driver docker image for twin '%s': %s",
                "error",
            )
            logger.error(
                "Failed to run driver docker image %s for twin '%s': %s",
                driver_image,
                twin.name,
                exc,
            )

    return results


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
    default_driver = drivers.get("default")
    if not isinstance(default_driver, dict):
        raise ValueError("No default driver specified")

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
            if not isinstance(driver_config, dict):
                raise ValueError(f"Invalid config for driver '{driver_name}'")
            if not driver_config.get("docker_image") or not isinstance(
                driver_config["docker_image"], str
            ):
                raise ValueError(f"No docker_image specified for driver '{driver_name}'")
            raw_params = driver_config.get("params")
            if raw_params is None:
                params: list[str] = []
            elif isinstance(raw_params, list) and all(
                isinstance(param, str) for param in raw_params
            ):
                params = raw_params
            else:
                raise ValueError(f"Invalid params for driver '{driver_name}'")
            return driver_config["docker_image"], params

    if not default_driver.get("docker_image") or not isinstance(
        default_driver["docker_image"], str
    ):
        raise ValueError("No docker_image specified for default driver")
    raw_default_params = default_driver.get("params")
    if raw_default_params is None:
        default_params: list[str] = []
    elif isinstance(raw_default_params, list) and all(
        isinstance(param, str) for param in raw_default_params
    ):
        default_params = raw_default_params
    else:
        raise ValueError("Invalid params for default driver")
    return default_driver["docker_image"], default_params


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


def _perform_edge_core_restart(token: str) -> dict[str, Any]:
    """Execute restart workflow: cleanup local state and re-run driver startup."""
    removed_json_files = _remove_cached_twin_json_files()
    removed_containers = _stop_and_prune_driver_containers()

    environment_uuid = load_environment_uuid(retries=5, retry_delay_seconds=0.2)
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

    with open(twin_json_file, "w") as f:
        json.dump(twin_data, f, indent=2, default=_json_default)
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
        console.print(
            f"  [green]✓[/green] Environment [dim]({environment_uuid}, {time.perf_counter() - _t0:.3f}s)[/dim]"
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
            results = fetch_and_run_twin_drivers(token, environment_uuid, fingerprint)
            if not results:
                console.print(
                    f"  [yellow]⚠[/yellow] Twin drivers [dim]({time.perf_counter() - _t0:.3f}s)[/dim]"
                )
                console.print("  [dim]No twins with driver images matched this edge.[/dim]")
            else:
                started = sum(1 for r in results if r["success"])
                console.print(
                    f"  [green]✓[/green] Twin drivers [dim]({started}/{len(results)}, {time.perf_counter() - _t0:.3f}s)[/dim]"
                )
                for r in results:
                    status = "[green]✓[/green]" if r["success"] else "[red]✗[/red]"
                    console.print(f"    {r['twin_name']} → {r['driver_image']} {status}")

    console.print("\n[green]All startup checks passed.[/green]\n")
    return True


def run_runtime_loop() -> None:
    """Keep edge-core alive and continuously reconcile driver log forwarding."""
    logger.info(
        "Entering edge-core runtime loop (interval=%.1fs)",
        LOG_FOLLOWER_RECONCILE_INTERVAL_SECONDS,
    )
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
        try:
            ensure_edge_command_subscription()
        except Exception:
            logger.exception("Unexpected error while ensuring edge command subscription")
        time.sleep(LOG_FOLLOWER_RECONCILE_INTERVAL_SECONDS)
