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
    """Force-remove a container and its anonymous volumes.

    ``-v`` only reaps volumes Docker created implicitly from an image's
    ``VOLUME`` directive — never named volumes or bind mounts. Without it,
    recreating such a container orphans a directory no prune reclaims. Our own
    images declare none, but third-party ones (eclipse/zenoh, worker) may.

    No state is lost: a recreate is ``rm`` + ``create``, and ``create`` mints a
    fresh anonymous volume rather than reattaching the old one, so the previous
    behaviour leaked the directory rather than preserving it.

    Returns True on success (including not-found).
    """
    if not docker_available():
        return False
    try:
        subprocess.run(
            ["docker", "rm", "-f", "-v", container_name],
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


def group_gid(group_name: str) -> Optional[int]:
    """Return the numeric GID for *group_name* on the host, or None when absent.

    Used to derive the ``--group-add <gid>`` flag for ``docker run`` when a
    host hardware accelerator's device node is restricted to a specific Unix
    group (e.g. older HailoRT installs use ``hailo`` as the device group; v4.20+
    instead ships ``/dev/hailo0`` with 0666 permissions and no group, so the
    flag is unnecessary on those systems and this helper returns None).

    Linux-only. On other platforms — and when the ``grp`` module is unavailable
    (CPython on Windows builds) — this returns None so callers can skip the
    group-add step unconditionally.
    """
    if platform.system() != "Linux":
        return None
    try:
        import grp  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return int(grp.getgrnam(group_name).gr_gid)
    except KeyError:
        return None


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


def docker_has_nvidia_default_runtime() -> bool:
    """Return True when ``/etc/docker/daemon.json`` uses nvidia as the default runtime.

    Even if NVIDIA runtime is available (``docker_has_nvidia_runtime()``),
    ``--gpus all`` is only reliable when nvidia is the *default* runtime
    configured in the Docker daemon.
    """
    if platform.system() != "Linux":
        return False
    try:
        with open("/etc/docker/daemon.json", encoding="utf-8") as fh:
            daemon_cfg = json.load(fh)
        return daemon_cfg.get("default-runtime") == "nvidia"
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def docker_prune_stopped_cyberwave_containers(prefix: str = "cyberwave") -> int:
    """Remove all stopped containers whose name starts with *prefix*.

    Returns the number of containers successfully removed.
    """
    if not docker_available():
        return 0

    stopped = docker_ps_by_prefix(prefix, include_stopped=True)
    running = set(docker_ps_by_prefix(prefix, include_stopped=False))
    to_remove = [name for name in stopped if name not in running]

    if not to_remove:
        logger.debug("No stopped cyberwave containers to prune")
        return 0

    removed = 0
    for name in to_remove:
        if docker_rm(name):
            removed += 1
            logger.debug("Pruned stopped container: %s", name)
        else:
            logger.warning("Failed to prune stopped container: %s", name)

    logger.info("Pruned %d/%d stopped cyberwave container(s)", removed, len(to_remove))
    return removed


def docker_prune_unused_images() -> bool:
    """Remove unused Docker images that are older than 2 hours.

    The ``--filter until=2h`` guard protects freshly-built or freshly-pulled
    local images (e.g. a dev test image tagged moments before a service
    restart) from being collected before the worker container has a chance to
    start and reference them.

    Returns True on success.
    """
    if not docker_available():
        return False
    try:
        subprocess.run(
            ["docker", "image", "prune", "--all", "--force", "--filter", "until=2h"],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        logger.info("Docker unused image prune completed")
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning("Docker image prune failed: %s", exc)
        return False
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Docker image prune error: %s", exc)
        return False


# Docker 23.0 narrowed bare ``docker volume prune`` to anonymous volumes and
# moved the sweep-everything behaviour behind ``--all``.
ANONYMOUS_ONLY_VOLUME_PRUNE_MIN_VERSION = (23, 0)

# Cached on success only. Edge Core can come up before dockerd, and caching
# that miss would disable volume pruning for the whole process lifetime.
_docker_server_version: Optional[tuple[int, int]] = None


def docker_server_version() -> Optional[tuple[int, int]]:
    """Return the daemon's ``(major, minor)`` version, or None if unknown.

    None covers daemon-down and unparseable output alike; callers must read it
    as "assume the older, more destructive semantics".
    """
    global _docker_server_version
    if _docker_server_version is not None:
        return _docker_server_version

    resolved: Optional[tuple[int, int]] = None
    if docker_available():
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            raw = result.stdout.strip()
            if result.returncode == 0 and raw:
                parts = raw.split("-", 1)[0].split(".")
                resolved = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
        except (subprocess.TimeoutExpired, OSError, ValueError, IndexError) as exc:
            logger.debug("Could not determine Docker server version: %s", exc)

    if resolved is not None:
        _docker_server_version = resolved
    return resolved


def volume_prune_is_anonymous_only() -> bool:
    """Return True when bare ``docker volume prune`` spares named volumes.

    Older daemons remove *every* unused volume on the host, including other
    workloads' named ones — too high a price for reclaiming our own logs. The
    distro ``docker.io`` package still ships 20.10. Unknown counts as old.
    """
    version = docker_server_version()
    return version is not None and version >= ANONYMOUS_ONLY_VOLUME_PRUNE_MIN_VERSION


def docker_prune_dangling_volumes() -> bool:
    """Remove anonymous volumes no container references.

    Omitting ``--all`` keeps named volumes, but only on new enough daemons, so
    :func:`volume_prune_is_anonymous_only` gates the call. No ``until`` filter,
    unlike :func:`docker_prune_unused_images`: the daemon rejects it here, and
    ``docker create`` attaches volumes before start, so a mid-recreate
    container is never a candidate anyway.

    Returns True on success. A version-gated skip is not a failure.
    """
    if not docker_available():
        return False
    if not volume_prune_is_anonymous_only():
        logger.debug(
            "Skipping volume prune: Docker %s predates the anonymous-only default "
            "(>= %d.%d), so a bare prune would also delete named volumes",
            docker_server_version() or "version unknown",
            *ANONYMOUS_ONLY_VOLUME_PRUNE_MIN_VERSION,
        )
        return True
    try:
        subprocess.run(
            ["docker", "volume", "prune", "--force"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        logger.info("Docker dangling volume prune completed")
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning("Docker volume prune failed: %s", exc)
        return False
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Docker volume prune error: %s", exc)
        return False


def backing_device_for(path: str) -> Optional[str]:
    """Return the block device backing *path*, via the longest mountpoint match.

    Matching on ``/`` alone misattributes any path on a separate mount — e.g. a
    Docker data root relocated to an external SSD.
    """
    if platform.system() != "Linux":
        return None
    target = os.path.realpath(path)
    best_device: Optional[str] = None
    best_len = -1
    try:
        with open("/proc/mounts", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                device, mountpoint = parts[0], parts[1]
                if target == mountpoint or target.startswith(mountpoint.rstrip("/") + "/"):
                    if len(mountpoint) > best_len:
                        best_device, best_len = device, len(mountpoint)
    except OSError:
        return None
    return best_device


def is_sd_card_path(path: str) -> bool:
    """Return True when *path* lives on an SD card (``mmcblk`` device).

    Lets callers avoid disk I/O that accelerates flash wear.
    """
    device = backing_device_for(path)
    return device is not None and "mmcblk" in device


def is_sd_card_root() -> bool:
    """Return True when the root filesystem is on an SD card.

    Prefer :func:`is_sd_card_path` — wear and free space are per-filesystem.
    """
    return is_sd_card_path("/")


# Cached: the data root cannot move under a running dockerd, and callers hit
# this every reconcile cycle.
_docker_data_root: Optional[str] = None


def docker_data_root() -> str:
    """Return the directory Docker stores images, containers and volumes in.

    This — not ``/`` — is the filesystem that fills and that prune I/O wears.
    Resolved from ``docker info``, then ``data-root`` in
    ``/etc/docker/daemon.json``, then ``/var/lib/docker``, then ``/``.
    """
    global _docker_data_root
    if _docker_data_root is not None:
        return _docker_data_root

    resolved: Optional[str] = None
    if docker_available():
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.DockerRootDir}}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            candidate = result.stdout.strip()
            if result.returncode == 0 and candidate:
                resolved = candidate
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("Could not read Docker data root from `docker info`: %s", exc)

    if resolved is None:
        try:
            with open("/etc/docker/daemon.json", encoding="utf-8") as fh:
                candidate = json.load(fh).get("data-root")
            if isinstance(candidate, str) and candidate.strip():
                resolved = candidate.strip()
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    if resolved is None:
        resolved = "/var/lib/docker"
    if not os.path.isdir(resolved):
        resolved = "/"

    _docker_data_root = resolved
    logger.debug("Resolved Docker data root to %s", resolved)
    return resolved


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
