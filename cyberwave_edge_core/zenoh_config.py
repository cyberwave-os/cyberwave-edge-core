"""Centralized Zenoh runtime configuration for edge services.

This module provides a single source of truth for all Zenoh-related configuration
in Edge Core.  It is consumed by ``startup.py`` when building environment variables
for driver and worker containers and when optionally starting a Zenoh router.

Environment variables read by this module
------------------------------------------
CYBERWAVE_DATA_BACKEND
    Select the data transport: ``"zenoh"`` (default) or ``"filesystem"``.
    When set to ``"zenoh"`` all managed containers receive the full Zenoh env
    block; containers that do not use the SDK data layer simply ignore it.

ZENOH_CONNECT
    Comma-separated list of Zenoh router endpoint URLs that containers should
    connect to (e.g. ``"tcp/10.0.0.1:7447"``).  When empty / absent, Zenoh
    operates in peer-to-peer discovery mode (multicast on ``--network host``)
    which is the preferred default for same-host deployments.

ZENOH_SHARED_MEMORY
    ``"1"`` / ``"true"`` / ``"yes"`` to enable Zenoh shared-memory transport
    for zero-copy delivery between containers on the same Docker host.
    Defaults to ``"false"`` (safe fallback to TCP).  Set to ``"true"`` on Linux
    hosts where all service containers share the same kernel and SHM namespace.

ZENOH_ROUTER_ENABLED
    ``"1"`` / ``"true"`` to start an optional Zenoh router container before
    driver containers.  Required only for MQTT bridge or multi-hop topologies.
    Defaults to ``"false"``.

ZENOH_ROUTER_IMAGE
    Docker image to use for the router container.
    Defaults to ``"eclipse/zenoh:latest"``.

ZENOH_ROUTER_PORT
    Host port on which the Zenoh router listens.
    Defaults to ``7447``.

ZENOH_ROUTER_CONTAINER_NAME_SUFFIX
    Custom suffix appended to the router container name
    ``cyberwave-zenoh-router-{suffix}``.  Defaults to the first eight
    characters of the environment UUID if available.

Public API
----------
ZenohConfig
    Dataclass holding the resolved configuration for one Edge Core session.

build_zenoh_env_vars(config)
    Returns a ``dict[str, str]`` of the Zenoh-related environment variables
    that must be injected into every managed container.

validate_zenoh_config(config)
    Returns a ``ZenohDiagnostics`` named-tuple summarising the active mode and
    any warnings the operator should be aware of.

start_zenoh_router(config, env_uuid)
    Attempt to start the optional Zenoh router container.  Returns ``True`` on
    success or when the router is already running, ``False`` on failure.

stop_zenoh_router(env_uuid)
    Stop and remove the Zenoh router container if it is running.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TRUE_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _parse_endpoints(value: str | None) -> list[str]:
    if not value or not value.strip():
        return []
    return [ep.strip() for ep in value.split(",") if ep.strip()]


# ---------------------------------------------------------------------------
# ZenohConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class ZenohConfig:
    """Resolved Zenoh configuration for one Edge Core session.

    All fields default to ``None`` / empty so that callers can construct a
    config manually in tests without needing to set real environment variables.
    Values are resolved from the environment inside ``__post_init__`` for any
    field left at its sentinel default.
    """

    data_backend: str = ""
    """``"zenoh"`` or ``"filesystem"``.  Env: ``CYBERWAVE_DATA_BACKEND``."""

    connect_endpoints: list[str] = field(default_factory=list)
    """Router endpoints.  Env: ``ZENOH_CONNECT`` (comma-separated)."""

    shared_memory: bool | None = None
    """Enable shared-memory transport.  Env: ``ZENOH_SHARED_MEMORY``."""

    router_enabled: bool | None = None
    """Start a Zenoh router container.  Env: ``ZENOH_ROUTER_ENABLED``."""

    router_image: str = ""
    """Docker image for the router.  Env: ``ZENOH_ROUTER_IMAGE``."""

    router_port: int = 0
    """Host port for the router.  Env: ``ZENOH_ROUTER_PORT``."""

    def __post_init__(self) -> None:
        if not self.data_backend:
            self.data_backend = os.environ.get("CYBERWAVE_DATA_BACKEND", "zenoh").strip() or "zenoh"

        if not self.connect_endpoints:
            self.connect_endpoints = _parse_endpoints(os.environ.get("ZENOH_CONNECT", ""))

        if self.shared_memory is None:
            self.shared_memory = _parse_bool(os.environ.get("ZENOH_SHARED_MEMORY"))

        if self.router_enabled is None:
            self.router_enabled = _parse_bool(os.environ.get("ZENOH_ROUTER_ENABLED"))

        if not self.router_image:
            self.router_image = (
                os.environ.get("ZENOH_ROUTER_IMAGE", "").strip() or "eclipse/zenoh:latest"
            )

        if not self.router_port:
            try:
                self.router_port = int(os.environ.get("ZENOH_ROUTER_PORT", "0") or "0")
            except ValueError:
                self.router_port = 0
            if not self.router_port:
                self.router_port = 7447

    @property
    def is_zenoh(self) -> bool:
        """Return True when the data backend is Zenoh."""
        return self.data_backend == "zenoh"

    @property
    def peer_to_peer(self) -> bool:
        """Return True when Zenoh runs in peer-to-peer (no router) mode."""
        return not self.connect_endpoints

    def router_container_name(self, env_uuid: str) -> str:
        """Return the deterministic container name for the Zenoh router."""
        suffix = env_uuid[:8] if env_uuid else "default"
        return f"cyberwave-zenoh-router-{suffix}"


# ---------------------------------------------------------------------------
# Env-var builder
# ---------------------------------------------------------------------------

ZENOH_ENV_VARS_ALWAYS: tuple[str, ...] = (
    "CYBERWAVE_DATA_BACKEND",
    "ZENOH_SHARED_MEMORY",
)
"""Env vars injected into every managed container regardless of mode."""

ZENOH_ENV_VARS_ROUTER: tuple[str, ...] = ("ZENOH_CONNECT",)
"""Additional env vars injected only when a router is configured."""


def build_zenoh_env_vars(config: ZenohConfig) -> dict[str, str]:
    """Build the dict of Zenoh environment variables for managed containers.

    The returned dict uses ``str`` values ready to be passed directly to
    ``docker run -e KEY=VALUE`` flags.  Call this once per Edge Core session
    and inject the result into every driver and worker container.

    Args:
        config: Resolved :class:`ZenohConfig` instance.

    Returns:
        Mapping of environment variable names to their string values.
        When Zenoh is not the active backend (``config.data_backend != "zenoh"``)
        only ``CYBERWAVE_DATA_BACKEND`` is returned so containers learn the
        fallback mode.
    """
    env: dict[str, str] = {"CYBERWAVE_DATA_BACKEND": config.data_backend}

    if not config.is_zenoh:
        return env

    env["ZENOH_SHARED_MEMORY"] = "true" if config.shared_memory else "false"

    if config.connect_endpoints:
        env["ZENOH_CONNECT"] = ",".join(config.connect_endpoints)

    return env


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class ZenohDiagnostics(NamedTuple):
    """Summary of the active Zenoh configuration and any operator warnings."""

    mode: str
    """Human-readable description of the active mode."""

    shared_memory_active: bool
    """Whether shared-memory transport has been requested."""

    router_enabled: bool
    """Whether a Zenoh router container will be started."""

    connect_endpoints: list[str]
    """Router endpoints that containers will connect to (empty = P2P)."""

    warnings: list[str]
    """Operator-facing warnings that do not prevent startup."""


def validate_zenoh_config(config: ZenohConfig) -> ZenohDiagnostics:
    """Inspect *config* and return a diagnostics summary with any warnings.

    This function is **pure** (no side effects) so it is safe to call during
    status reporting or test assertions.

    Args:
        config: Resolved :class:`ZenohConfig` instance.

    Returns:
        :class:`ZenohDiagnostics` named-tuple.
    """
    warnings: list[str] = []

    if not config.is_zenoh:
        return ZenohDiagnostics(
            mode=f"filesystem (CYBERWAVE_DATA_BACKEND={config.data_backend!r})",
            shared_memory_active=False,
            router_enabled=bool(config.router_enabled),
            connect_endpoints=list(config.connect_endpoints),
            warnings=[
                "CYBERWAVE_DATA_BACKEND is not 'zenoh'; "
                "Zenoh transport is disabled for all managed containers."
            ],
        )

    if not config.shared_memory:
        warnings.append(
            "ZENOH_SHARED_MEMORY is not enabled.  "
            "Binary stream channels (frames, depth) will use TCP loopback, "
            "which is significantly slower than shared-memory transport.  "
            "Set ZENOH_SHARED_MEMORY=true on Linux hosts where all "
            "containers run on the same kernel."
        )

    if config.router_enabled and not config.connect_endpoints:
        warnings.append(
            "ZENOH_ROUTER_ENABLED=true but ZENOH_CONNECT is empty.  "
            "Containers will not connect to the router automatically.  "
            "Set ZENOH_CONNECT to the router endpoint (e.g. tcp/localhost:7447)."
        )

    if config.connect_endpoints and not config.router_enabled:
        warnings.append(
            "ZENOH_CONNECT is set but ZENOH_ROUTER_ENABLED is false.  "
            "Containers will attempt to connect to the configured endpoints "
            "but no router will be started by Edge Core.  "
            "Ensure an external router is reachable at: " + ", ".join(config.connect_endpoints)
        )

    if platform.system() == "Darwin" and config.shared_memory:
        warnings.append(
            "ZENOH_SHARED_MEMORY=true on macOS (Docker Desktop).  "
            "Shared-memory transport between containers requires Linux "
            "kernel SHM namespacing; this setting has no effect on macOS "
            "and may cause session open errors.  "
            "Consider unsetting ZENOH_SHARED_MEMORY on macOS."
        )

    if config.peer_to_peer:
        mode = "peer-to-peer (multicast discovery, no router)"
    else:
        mode = "router-connected (endpoints: " + ", ".join(config.connect_endpoints) + ")"

    if config.router_enabled:
        mode = "router-managed (" + mode + ")"

    return ZenohDiagnostics(
        mode=mode,
        shared_memory_active=bool(config.shared_memory),
        router_enabled=bool(config.router_enabled),
        connect_endpoints=list(config.connect_endpoints),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Router container management
# ---------------------------------------------------------------------------

ZENOH_ROUTER_CONTAINER_PREFIX = "cyberwave-zenoh-router-"


def _is_router_running(container_name: str) -> bool:
    """Return True when the named router container is in a running state."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Status}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip().lower() == "running"
    except (subprocess.TimeoutExpired, OSError):
        return False


def start_zenoh_router(config: ZenohConfig, env_uuid: str) -> bool:
    """Start the optional Zenoh router container if not already running.

    This is a **no-op** (returns ``True``) when:
    - ``config.router_enabled`` is ``False``.
    - The router container is already running.

    The container is started with ``--restart unless-stopped`` so it
    persists across reboots until explicitly stopped.

    Args:
        config: Resolved :class:`ZenohConfig` instance.
        env_uuid: Environment UUID used to build a deterministic container name.

    Returns:
        ``True`` when the router is running (started or already up),
        ``False`` when startup failed.
    """
    if not config.router_enabled:
        return True

    if not shutil.which("docker"):
        logger.error("Docker is not installed or not in PATH; cannot start Zenoh router")
        return False

    container_name = config.router_container_name(env_uuid)

    if _is_router_running(container_name):
        logger.debug("Zenoh router container %s is already running", container_name)
        return True

    # Remove any stale stopped container with the same name.
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=30,
    )

    # Determine network args — match the same logic used for driver containers.
    # On Linux --network host gives the container access to all host ports without
    # explicit -p publish flags (which are ignored by Docker when host networking
    # is active and generate a harmless but noisy warning).
    network_args: list[str]
    port_args: list[str]
    if platform.system() == "Darwin":
        network_args = ["--add-host", "host.docker.internal:host-gateway"]
        port_args = [
            "-p",
            f"{config.router_port}:{config.router_port}/tcp",
            "-p",
            f"{config.router_port}:{config.router_port}/udp",
        ]
    else:
        network_args = ["--network", "host"]
        port_args = []  # redundant with --network host; omit to avoid Docker warning

    cmd = [
        "docker",
        "run",
        "--detach",
        "--restart",
        "unless-stopped",
        "--name",
        container_name,
        *port_args,
        *network_args,
        config.router_image,
    ]

    logger.info(
        "Starting Zenoh router container %s (image=%s, port=%d)",
        container_name,
        config.router_image,
        config.router_port,
    )
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        logger.info("Zenoh router container %s started successfully", container_name)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Failed to start Zenoh router container %s: %s",
            container_name,
            (exc.stderr or "").strip(),
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("Timed out starting Zenoh router container %s", container_name)
        return False
    except OSError as exc:
        logger.error("OS error starting Zenoh router container %s: %s", container_name, exc)
        return False


def stop_zenoh_router(env_uuid: str) -> bool:
    """Stop and remove the Zenoh router container for the given environment.

    This is a best-effort operation; errors are logged but do not raise.

    Args:
        env_uuid: Environment UUID used to derive the container name.

    Returns:
        ``True`` when the container was removed (or was not present),
        ``False`` when removal failed.
    """
    if not shutil.which("docker"):
        return True

    suffix = env_uuid[:8] if env_uuid else "default"
    container_name = f"{ZENOH_ROUTER_CONTAINER_PREFIX}{suffix}"

    try:
        result = subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("Stopped and removed Zenoh router container %s", container_name)
        else:
            # Container was not running — this is fine.
            logger.debug(
                "Zenoh router container %s not found or already removed: %s",
                container_name,
                result.stderr.strip(),
            )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Failed to remove Zenoh router container %s: %s", container_name, exc)
        return False


# ---------------------------------------------------------------------------
# Convenience: log diagnostics at INFO level
# ---------------------------------------------------------------------------


def log_zenoh_diagnostics(config: ZenohConfig) -> ZenohDiagnostics:
    """Run :func:`validate_zenoh_config`, log the results, and return them.

    Intended to be called once during Edge Core startup so that the operator
    can see the active Zenoh configuration in the service log.

    Args:
        config: Resolved :class:`ZenohConfig` instance.

    Returns:
        The :class:`ZenohDiagnostics` produced by :func:`validate_zenoh_config`.
    """
    diag = validate_zenoh_config(config)
    logger.info("Zenoh configuration: %s", diag.mode)
    if diag.shared_memory_active:
        logger.info("Zenoh shared-memory transport: enabled")
    else:
        logger.info("Zenoh shared-memory transport: disabled (TCP fallback)")
    for warning in diag.warnings:
        logger.warning("Zenoh config warning: %s", warning)
    return diag
