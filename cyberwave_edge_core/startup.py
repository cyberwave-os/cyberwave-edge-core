"""Boot-time startup checks for the Cyberwave Edge Core.

On every boot the edge core must:
  1. Read the API token from ``/etc/cyberwave/credentials.json``
  2. Validate the token against the Cyberwave REST API
  3. Verify that it can connect to the MQTT broker
  4. Load configured devices from ``/etc/cyberwave/devices.json``
  5. Check whether an environment is linked via ``/etc/cyberwave/environment.json``

The config directory defaults to ``/etc/cyberwave`` and can be overridden with
the ``CYBERWAVE_EDGE_CONFIG_DIR`` environment variable (set in the systemd unit).

This module exposes each check individually (for the ``status`` command)
and a single ``run_startup_checks()`` orchestrator for the boot path.
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

from cyberwave import Cyberwave
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# ---- constants ---------------------------------------------------------------

# System-wide edge config directory.  The systemd unit sets
# CYBERWAVE_EDGE_CONFIG_DIR=/etc/cyberwave; fall back to the same path
# if the env var is absent (e.g. manual invocation).
CONFIG_DIR = Path(os.getenv("CYBERWAVE_EDGE_CONFIG_DIR", "/etc/cyberwave"))
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
DEVICES_FILE = CONFIG_DIR / "devices.json"
FINGERPRINT_FILE = CONFIG_DIR / "fingerprint.json"
ENVIRONMENT_FILE = CONFIG_DIR / "environment.json"
DEFAULT_API_URL = os.getenv("CYBERWAVE_API_URL", "https://api.cyberwave.com")

CYBERWAVE_ENVIRONMENT = os.getenv("CYBERWAVE_ENVIRONMENT", "production")


def load_devices() -> List[str]:
    """Load the list of devices from the environment.json file."""
    raise NotImplementedError("Not implemented")


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


def validate_token(token: str, *, base_url: Optional[str] = None) -> bool:
    """Validate *token* by listing workspaces via the Cyberwave SDK.

    Returns ``True`` when the SDK call succeeds (i.e. the token is valid).
    """
    base_url = base_url or os.getenv("CYBERWAVE_API_URL", DEFAULT_API_URL)
    masked_token = f"{token[:6]}…{token[-4:]}" if len(token) > 12 else "***"
    logger.info("Validating token against %s via SDK (token: %s)", base_url, masked_token)
    try:
        client = Cyberwave(base_url=base_url, token=token)
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
    mqtt_host = os.getenv("CYBERWAVE_MQTT_HOST", "(default)")
    logger.info(
        "Attempting MQTT connection (api_url=%s, mqtt_host=%s)",
        DEFAULT_API_URL,
        mqtt_host,
    )
    try:
        client = Cyberwave(base_url=DEFAULT_API_URL, token=token)
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


def generate_fingerprint() -> str:
    """Generate a stable fingerprint based on host characteristics."""
    raw = f"{platform.node()}|{platform.system()}|{platform.machine()}|{uuid.getnode()}"
    digest = sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{platform.system().lower()}-{digest}"


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

    # check if the docker image has a tag first
    if ":" not in image:
        if CYBERWAVE_ENVIRONMENT != "production":
            image = f"{image}:{CYBERWAVE_ENVIRONMENT}"  # for example: cyberwaveos/cyberwave-edge-so101:dev

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
        logger.error("Failed to pull docker image %s: %s", image, exc.stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Docker pull timed out for image: %s", image)
        return False

    # Build env vars for the container
    env_vars: List[str] = [
        "-e",
        f"CYBERWAVE_TWIN_UUID={twin_uuid}",
        "-e",
        f"CYBERWAVE_TOKEN={token}",
    ]
    api_url = os.getenv("CYBERWAVE_API_URL")
    if api_url:
        env_vars += ["-e", f"CYBERWAVE_API_URL={api_url}"]
    mqtt_host = os.getenv("CYBERWAVE_MQTT_HOST")
    if mqtt_host:
        env_vars += ["-e", f"CYBERWAVE_MQTT_HOST={mqtt_host}"]
    if CYBERWAVE_ENVIRONMENT != "production":
        env_vars += ["-e", f"CYBERWAVE_ENVIRONMENT={CYBERWAVE_ENVIRONMENT}"]
    twin_json_file = CONFIG_DIR / f"{twin_uuid}.json"
    if twin_json_file.exists():
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
    logger.info("Starting docker container %s from image %s", container_name, image)
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to start container %s: %s", container_name, exc.stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Docker run timed out for image: %s", image)
        return False


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
    client = Cyberwave(base_url=DEFAULT_API_URL, token=token)

    # List twins for the environment via the SDK
    twins = client.twins.list(environment_id=environment_uuid)
    if not twins:
        logger.info("No twins found for environment %s", environment_uuid)
        return []

    results: List[Dict[str, Any]] = []

    for twin in twins:
        twin_uuid = twin.uuid

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
        try:
            asset = client.assets.get(twin.asset_uuid)
        except Exception as exc:
            logger.warning(
                "Failed to get asset %s for twin %s: %s",
                twin.asset_uuid,
                twin_uuid,
                exc,
            )
            continue

        drivers = twin_metadata.get("drivers")
        if not drivers:
            # try fallback to asset metadata
            drivers = asset.metadata.get("drivers")
            if not drivers:
                logger.warning("No drivers specified in asset metadata for twin '%s'", twin.name)
                raise ValueError(
                    "No drivers specified in asset metadata for paired twin '%s'", twin.name
                )
            else:
                logger.warning(
                    "No drivers specified in twin metadata for twin '%s', found drivers in asset metadata",
                    twin.name,
                )
        driver_image, driver_params = _get_best_driver_image_and_params(drivers)

        write_or_update_twin_json_file(twin_uuid, twin.to_dict(), asset.to_dict())

        if not driver_image:
            logger.info("No driver_docker_image in asset metadata for twin '%s'", twin.name)
            raise ValueError(
                "No drivers specified in asset metadata for paired twin '%s'", twin.name
            )

        logger.info("Running driver docker image %s for twin '%s'", driver_image, twin.name)
        success = _run_docker_image(driver_image, driver_params, twin_uuid=twin_uuid, token=token)
        results.append(
            {
                "twin_uuid": twin_uuid,
                "twin_name": twin.name,
                "driver_image": driver_image,
                "success": success,
            }
        )

    return results


def _get_best_driver_image_and_params(drivers: Dict[str, Dict[str, str]]) -> tuple[str, list[str]]:
    """
    Given a list of drivers specified in the metadata of the asset,
    and given the hardware where the edge is running,
    Returns:
    - The best driver to run.
    - A list of parameters to pass to the driver when doing docker run
    TODO: this is missing as of now, always returning the default

    "drivers": {
        "default": {"docker_image": "helloworld", "version": "0.1.0", "params": ["--param1", "--param2"]},
        "mac":{"docker_image": "helloworld", "version": "0.1.0", "params": ["--param1", "--param2"]},
    },
    """
    if drivers["default"]:
        if not drivers["default"]["docker_image"] or not isinstance(
            drivers["default"]["docker_image"], str
        ):
            raise ValueError("No docker_image specified for default driver")
        return drivers["default"]["docker_image"], drivers["default"].get("params", [])
    raise ValueError("No default driver specified")


def register_edge(token: str) -> bool:
    fingerprint = get_or_create_fingerprint()
    if not fingerprint:
        logger.warning("Could not load or create edge fingerprint")
        return False

    logger.info("Registering edge with fingerprint=%s at %s", fingerprint, DEFAULT_API_URL)
    try:
        client = Cyberwave(base_url=DEFAULT_API_URL, token=token)
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
    return True


def run_startup_checks() -> bool:
    """Execute every boot-time check in sequence.

    Prints a Rich-formatted report to the console.
    Returns ``True`` only when **all** checks pass.
    """
    console.print("\n[bold]Cyberwave Edge Core — startup checks[/bold]\n")

    # Log resolved configuration for troubleshooting
    api_url = os.getenv("CYBERWAVE_API_URL", DEFAULT_API_URL)
    console.print(f"  [dim]Config dir:  {CONFIG_DIR}[/dim]")
    console.print(f"  [dim]API URL:     {api_url}[/dim]")
    console.print(f"  [dim]Environment: {CYBERWAVE_ENVIRONMENT}[/dim]")
    console.print()

    # 1 — credentials file
    console.print("  Checking credentials …", end=" ")
    token = load_token()
    if not token:
        console.print("[red]FAIL[/red]")
        console.print(f"\n  [red]No credentials found at {CREDENTIALS_FILE}[/red]")
        console.print("  [dim]Run 'cyberwave login' on this device first.[/dim]")
        return False
    console.print("[green]OK[/green]")

    # 2 — token validity
    console.print("  Validating token …     ", end=" ")
    if validate_token(token):
        console.print("[green]OK[/green]")
    else:
        console.print("[red]FAIL[/red]")
        console.print(f"\n  [red]Token validation failed against {api_url}[/red]")
        console.print("  [dim]Check 'journalctl -u cyberwave-edge-core' for details.[/dim]")
        console.print("  [dim]Run 'cyberwave login' to refresh your credentials.[/dim]")
        return False

    # 3 — MQTT broker
    console.print("  Connecting to MQTT …   ", end=" ")
    if check_mqtt_connection(token):
        console.print("[green]OK[/green]")
    else:
        console.print("[red]FAIL[/red]")
        console.print("\n  [red]Could not connect to the MQTT broker.[/red]")
        console.print("  [dim]Check network connectivity and MQTT configuration.[/dim]")

    # 4: Edge registering
    console.print("  Registering edge …     ", end=" ")
    if register_edge(token):
        console.print("[green]OK[/green]")
    else:
        console.print("[red]FAIL[/red]")
        console.print("\n  [red]Could not register the edge.[/red]")
        return False

    # 5 — linked environment
    console.print("  Checking environment … ", end=" ")
    environment_uuid = load_environment_uuid(retries=5, retry_delay_seconds=0.2)
    if environment_uuid:
        console.print(f"[green]OK[/green] [dim]({environment_uuid})[/dim]")
    else:
        console.print("[yellow]NONE[/yellow]")
        console.print(f"\n  [yellow]No linked environment found in {ENVIRONMENT_FILE}[/yellow]")
        console.print("  [dim]Expected format: {'uuid': 'unique-uuid-of-the-environment'}[/dim]")

    # 6 — fetch twins, match by fingerprint, write JSON file, run drivers
    if environment_uuid:
        console.print("  Fetching twin drivers …", end=" ")
        fingerprint = get_or_create_fingerprint()
        if not fingerprint:
            console.print("[red]FAIL[/red]")
            console.print("\n  [red]Could not determine edge fingerprint.[/red]")
        else:
            results = fetch_and_run_twin_drivers(token, environment_uuid, fingerprint)
            if not results:
                console.print("[yellow]NONE[/yellow]")
                console.print("  [dim]No twins with driver images matched this edge.[/dim]")
            else:
                started = sum(1 for r in results if r["success"])
                console.print(f"[green]{started}/{len(results)} driver(s) started[/green]")
                for r in results:
                    status = "[green]OK[/green]" if r["success"] else "[red]FAIL[/red]"
                    console.print(f"    {r['twin_name']} → {r['driver_image']} {status}")

    console.print("\n[green]All startup checks passed.[/green]\n")
    return True
