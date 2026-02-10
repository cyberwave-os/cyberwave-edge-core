"""Boot-time startup checks for the Cyberwave Edge Core.

On every boot the edge core must:
  1. Read the API token from ``~/.cyberwave/credentials.json``
  2. Validate the token against the Cyberwave REST API
  3. Verify that it can connect to the MQTT broker
  4. Load configured devices from ``~/.cyberwave/devices.json``

This module exposes each check individually (for the ``status`` command)
and a single ``run_startup_checks()`` orchestrator for the boot path.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import httpx
from rich.console import Console
from cyberwave import Cyberwave

logger = logging.getLogger(__name__)
console = Console()

# ---- constants ---------------------------------------------------------------

CONFIG_DIR = Path.home() / ".cyberwave"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
DEVICES_FILE = CONFIG_DIR / "devices.json"
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
        client = Cyberwave(token=token)
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
          {"slug": "the-robot-studio/so101", "metadata": {"type": "follower-arm"}, "name": "so101", "port": "/dev/ttty"},
          {"slug": "cyberwave/camera", "metadata": {"type": "camera"}, "name": "camera1", "port": "/dev/video0"}
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


# ---- orchestrator ------------------------------------------------------------


def generate_fingerprint() -> str:
    # TODO: Generate a fingerprint for the edge based on the device information
    return "ghaasa"


def register_edge(token: str) -> bool:
    client = Cyberwave(token=token)
    edge = client.edges.create(
        fingerprint=generate_fingerprint(),
    )
    return edge


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

    console.print("\n[green]All startup checks passed.[/green]\n")
    return True
