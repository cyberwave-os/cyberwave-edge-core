"""Internal Docker CLI helpers shared between driver and worker container management.

These functions are implementation plumbing — not user-facing. They wrap raw
``subprocess.run(["docker", ...])`` calls so that startup.py and worker_manager.py
don't duplicate Docker subprocess logic.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_user_args() -> list[str]:
    """Return ``--user uid:gid`` flags on Linux so container writes match the host user.

    On macOS, Docker Desktop transparently maps UIDs, so no flags are needed.
    """
    if platform.system() != "Linux":
        return []
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


def docker_available() -> bool:
    """Return True when the ``docker`` binary is in PATH."""
    return shutil.which("docker") is not None


def docker_rm(container_name: str, *, timeout: int = 30) -> bool:
    """Force-remove a container. Returns True on success (including not-found)."""
    if not docker_available():
        return False
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return True
    except subprocess.CalledProcessError:
        return True
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Failed to remove container %s: %s", container_name, exc)
        return False


def docker_stop(container_name: str, *, timeout: int = 30) -> bool:
    """Stop a container gracefully. Returns True on success or not-running."""
    if not docker_available():
        return False
    try:
        subprocess.run(
            ["docker", "stop", container_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return True
    except subprocess.CalledProcessError:
        return True
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Failed to stop container %s: %s", container_name, exc)
        return False


def docker_inspect(container_name: str) -> Optional[dict[str, Any]]:
    """Return the first element of ``docker inspect`` output, or None."""
    if not docker_available():
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
    data = payload[0]
    return data if isinstance(data, dict) else None


def docker_image_exists_locally(image: str) -> bool:
    """Return True when Docker has *image* cached locally."""
    if not docker_available():
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


def docker_container_status(container_name: str) -> str:
    """Return the container status string (e.g. 'running', 'exited', 'none')."""
    data = docker_inspect(container_name)
    if data is None:
        return "none"
    state = data.get("State")
    if not isinstance(state, dict):
        return "unknown"
    return str(state.get("Status", "unknown")).lower()


def docker_ps_by_prefix(prefix: str, *, include_stopped: bool = False) -> list[str]:
    """Return container names whose name starts with *prefix*."""
    if not docker_available():
        return []

    command = ["docker", "ps"]
    if include_stopped:
        command.append("-a")
    command.extend(
        [
            "--format",
            "{{.Names}}",
            "--filter",
            f"name=^{prefix}",
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
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Failed to list containers with prefix %r: %s", prefix, exc)
        return []


def docker_has_nvidia_runtime() -> bool:
    """Return True when the NVIDIA container runtime is available on this host."""
    if not docker_available():
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        runtimes_json = result.stdout.strip()
        if not runtimes_json:
            return False
        runtimes = json.loads(runtimes_json)
        return isinstance(runtimes, dict) and "nvidia" in runtimes
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
        json.JSONDecodeError,
    ):
        return False


def docker_logs_follow(container_name: str) -> Optional[subprocess.Popen]:  # type: ignore[type-arg]
    """Start a ``docker logs -f`` process and return its Popen handle."""
    if not docker_available():
        return None
    try:
        process = subprocess.Popen(
            ["docker", "logs", "-f", "--tail", "50", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return process
    except OSError as exc:
        logger.warning("Failed to start log streaming for %s: %s", container_name, exc)
        return None
