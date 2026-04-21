"""Docker ``run`` argument parsing and building helpers.

Pure string/list helpers that construct or inspect fragments of
``docker run`` command lines.
"""

from __future__ import annotations

import json
import platform
from typing import Optional
from urllib.parse import urlsplit, urlunsplit


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
    from .startup import get_runtime_env_var

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

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        candidate = parsed.get("resolved_device") or parsed.get("video_source")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    for line in reversed(payload.splitlines()):
        candidate_line = line.strip()
        if not candidate_line:
            continue
        if candidate_line.startswith("resolved_device="):
            _, _, candidate = candidate_line.partition("=")
            candidate = candidate.strip()
            if candidate:
                return candidate

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


def _docker_params_include_network(params: list[str]) -> bool:
    """Return True when docker params already include ``--network``."""
    for i, param in enumerate(params):
        if param.startswith("--network="):
            return True
        if param == "--network" and i + 1 < len(params):
            return True
    return False


def _build_driver_network_args(params: list[str]) -> list[str]:
    """Build docker network-related args with platform-aware defaults."""
    if _docker_params_include_network(params):
        return []
    if platform.system() == "Darwin":
        if _docker_params_include_add_host(params):
            return []
        return ["--add-host", "host.docker.internal:host-gateway"]
    return ["--network", "host"]


def _rewrite_macos_container_hostname(hostname: str) -> str:
    """Rewrite host-local names for Docker Desktop containers on macOS."""
    normalized = hostname.strip()
    if platform.system() != "Darwin":
        return normalized
    if normalized.lower() in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        return "host.docker.internal"
    return normalized


def _rewrite_macos_container_base_url(base_url: str) -> str:
    """Rewrite host-local backend URLs for Docker Desktop containers on macOS."""
    if platform.system() != "Darwin":
        return base_url

    try:
        parsed = urlsplit(base_url.strip())
    except ValueError:
        return base_url

    rewritten_hostname = _rewrite_macos_container_hostname(parsed.hostname or "")
    if rewritten_hostname == (parsed.hostname or "").strip():
        return base_url

    netloc = rewritten_hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )
