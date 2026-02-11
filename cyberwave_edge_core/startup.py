"""Boot-time startup checks for the Cyberwave Edge Core.

On every boot the edge core must:
  1. Read the API token from ``~/.cyberwave/credentials.json``
  2. Validate the token against the Cyberwave REST API
  3. Verify that it can connect to the MQTT broker
  4. Load configured devices from ``~/.cyberwave/devices.json``
  5. Check whether an environment is linked via ``~/.cyberwave/environment.json``

This module exposes each check individually (for the ``status`` command)
and a single ``run_startup_checks()`` orchestrator for the boot path.
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from cyberwave import Cyberwave
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# ---- constants ---------------------------------------------------------------

CONFIG_DIR = Path.home() / ".cyberwave"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
DEVICES_FILE = CONFIG_DIR / "devices.json"
FINGERPRINT_FILE = CONFIG_DIR / "fingerprint.json"
ENVIRONMENT_FILE = CONFIG_DIR / "environment.json"
DEFAULT_API_URL = os.getenv("CYBERWAVE_API_URL", "https://api.cyberwave.com")
AUTH_USER_ENDPOINT = "/dj-rest-auth/user/"


# ---- data models -------------------------------------------------------------


@dataclass
class Device:
    """A single device configured on this edge node.

    Attributes:
        type: Device type identifier (e.g. ``"rgb-camera"``, ``"lidar"``).
        name: Human-readable name (e.g. ``"camera1"``).
        port: System device path (e.g. ``"/dev/video0"``).

    TODO: This data structure is temporary; this will eventually be the same as the twin
    """

    type: str
    name: str
    port: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Device":
        return cls(
            type=data.get("type", ""),
            name=data.get("name", ""),
            port=data.get("port", ""),
        )


# ---- individual checks -------------------------------------------------------


def load_token() -> Optional[str]:
    """Load the API token from *~/.cyberwave/credentials.json*.

    Returns the token string, or ``None`` if the file is missing or
    cannot be parsed.
    """
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        with open(CREDENTIALS_FILE) as f:
            data = json.load(f)
        return data.get("token") or None
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to read credentials file: %s", exc)
        return None


def validate_token(token: str, *, base_url: Optional[str] = None) -> bool:
    """Validate *token* by calling the backend ``/dj-rest-auth/user/`` endpoint.

    Returns ``True`` when the backend responds with HTTP 200.
    """
    base_url = base_url or os.getenv("CYBERWAVE_API_URL", DEFAULT_API_URL)
    try:
        resp = httpx.get(
            f"{base_url}{AUTH_USER_ENDPOINT}",
            headers={"Authorization": f"Token {token}"},
            timeout=15.0,
        )
        return resp.status_code == 200
    except httpx.RequestError as exc:
        logger.warning("API unreachable during token validation: %s", exc)
        return False


def check_mqtt_connection(token: str) -> bool:
    """Try to connect to the MQTT broker via the Cyberwave Python SDK.

    The SDK reads broker host / port / credentials from environment
    variables (``CYBERWAVE_MQTT_HOST``, etc.) and falls back to sensible
    defaults.  Returns ``True`` if the connection succeeds.
    """
    try:
        client = Cyberwave(base_url=DEFAULT_API_URL, token=token)
        client.mqtt.connect()
        connected: bool = client.mqtt.connected
        if connected:
            client.mqtt.disconnect()
        return connected
    except Exception as exc:
        logger.warning("MQTT connection check failed: %s", exc)
        return False


def load_devices() -> List[Device]:
    f"""Load the device list from *~/.cyberwave/devices.json*.

    The file is expected to contain a JSON array of device objects::

        [
          {"slug": "the-robot-studio/so101",
            "metadata": {"type": "follower-arm"},
            "name": "so101",
            "port": "/dev/ttty"
          },
          {"slug": "cyberwave/camera",
            "metadata": {"type": "camera"},
            "name": "camera1",
            "port": "/dev/video0"
          }
        ]

    Returns an empty list when the file is missing or cannot be parsed.
    """
    if not DEVICES_FILE.exists():
        return []
    try:
        with open(DEVICES_FILE) as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            logger.warning("devices.json should contain a JSON array")
            return []
        return [Device.from_dict(entry) for entry in raw if isinstance(entry, dict)]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read devices file: %s", exc)
        return []


def load_environment_uuid() -> Optional[str]:
    """Load linked environment UUID from ~/.cyberwave/environment.json.

    Expected format:
        {"uuid": "unique-uuid-of-the-environment"}
    """
    if not ENVIRONMENT_FILE.exists():
        return None
    try:
        with open(ENVIRONMENT_FILE) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("environment.json should contain a JSON object")
            return None

        env_uuid = data.get("uuid")
        if not isinstance(env_uuid, str) or not env_uuid.strip():
            logger.warning("environment.json must contain a non-empty 'uuid' field")
            return None

        normalized_uuid = str(uuid.UUID(env_uuid.strip()))
        return normalized_uuid
    except ValueError:
        logger.warning("environment.json contains an invalid UUID format")
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read environment file: %s", exc)
        return None


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
    """Persist fingerprint to ~/.cyberwave/fingerprint.json."""
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
    twin_json_file = CONFIG_DIR / f"{twin_uuid}.json"
    if twin_json_file.exists():
        env_vars += ["-v", f"{twin_json_file}:/app/{twin_uuid}.json"]
        env_vars += ["-e", f"CYBERWAVE_TWIN_JSON_FILE=/app/{twin_uuid}.json"]

    # Run the container
    cmd = [
        "docker",
        "run",
        "--detach",
        "--restart",
        "unless-stopped",
        "--network",
        "host",
        "--name",
        container_name,
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

        # The CLI writes edge_fingerprint into twin metadata when the user
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

        asset_metadata = asset.metadata or {}
        driver_image = asset_metadata.get("driver_docker_image")

        write_or_update_twin_json_file(twin_uuid, twin.to_dict(), asset.to_dict())

        if not driver_image:
            logger.info("No driver_docker_image in asset metadata for twin '%s'", twin.name)
            continue

        logger.info("Running driver docker image %s for twin '%s'", driver_image, twin.name)
        success = _run_docker_image(driver_image, twin_uuid=twin_uuid, token=token)
        results.append(
            {
                "twin_uuid": twin_uuid,
                "twin_name": twin.name,
                "driver_image": driver_image,
                "success": success,
            }
        )

    return results


def register_edge(token: str) -> bool:
    fingerprint = get_or_create_fingerprint()
    if not fingerprint:
        logger.warning("Could not load or create edge fingerprint")
        return False

    client = Cyberwave(base_url=DEFAULT_API_URL, token=token)
    edge = client.edges.create(
        fingerprint=fingerprint,
    )
    return bool(edge)


def write_or_update_twin_json_file(twin_uuid: str, twin_data: dict, asset_data: dict) -> bool:
    """
    Writes the content of the JSON twin into the disk, so that the docker container can read it
    and use it to start the driver correctly.
    """
    twin_data["asset"] = asset_data
    twin_json_file = CONFIG_DIR / f"{twin_uuid}.json"
    with open(twin_json_file, "w") as f:
        json.dump(twin_data, f, indent=2)
    return True


def run_startup_checks() -> bool:
    """Execute every boot-time check in sequence.

    Prints a Rich-formatted report to the console.
    Returns ``True`` only when **all** checks pass.
    """
    console.print("\n[bold]Cyberwave Edge Core — startup checks[/bold]\n")

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
        console.print("\n  [red]Token is invalid or the API is unreachable.[/red]")
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
        return False

    # 3b: Edge registering
    console.print("  Registering edge …     ", end=" ")
    if register_edge(token):
        console.print("[green]OK[/green]")
    else:
        console.print("[red]FAIL[/red]")
        console.print("\n  [red]Could not register the edge.[/red]")
        return False

    # 4 — configured devices
    console.print("  Loading devices …      ", end=" ")
    devices = load_devices()
    if not devices:
        console.print("[yellow]NONE[/yellow]")
        console.print(f"\n  [yellow]No devices configured in {DEVICES_FILE}[/yellow]")
        console.print("  [dim]Add devices to the file or run 'cyberwave edge pull'.[/dim]")
        # Not a fatal error — the edge core can run without devices
    else:
        console.print(f"[green]{len(devices)} device(s)[/green]")
        for dev in devices:
            console.print(f"    {dev.name} [dim]({dev.type})[/dim] @ {dev.port}")

    # 5 — linked environment
    console.print("  Checking environment … ", end=" ")
    environment_uuid = load_environment_uuid()
    if environment_uuid:
        console.print(f"[green]OK[/green] [dim]({environment_uuid})[/dim]")
    else:
        console.print("[yellow]NONE[/yellow]")
        console.print(f"\n  [yellow]No linked environment found in {ENVIRONMENT_FILE}[/yellow]")
        console.print("  [dim]Expected format: {'uuid': 'unique-uuid-of-the-environment'}[/dim]")

    # 6 — fetch twins, match by fingerprint, write twin.json file, and run driver docker images
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
