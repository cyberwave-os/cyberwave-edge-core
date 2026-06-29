"""Resilient create+start+probe launch for detached Docker containers.

Slow ``docker run --detach`` calls (common on SD-card edge hosts after large
image pulls) can exceed a fixed subprocess timeout and leave containers stuck
in ``created`` without registering them for reconcile.  This module splits
launch into a fast ``docker create``, a non-blocking ``docker start``, and a
configurable startup probe — the same pattern operators use manually when
recovering orphaned containers.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import docker_helpers

logger = logging.getLogger(__name__)

_DEFAULT_DRIVER_CREATE_TIMEOUT_SECONDS = 30
_DEFAULT_DRIVER_STARTUP_PROBE_SECONDS = 120
_DEFAULT_DRIVER_START_RECOVERY_SECONDS = 30
_DEFAULT_DRIVER_REMOVE_TIMEOUT_SECONDS = 60

GetRuntimeEnvVar = Callable[[str, Any], Any]


def driver_startup_probe_seconds(get_runtime_env_var: GetRuntimeEnvVar) -> int:
    """Return the driver startup probe window in seconds."""
    override = get_runtime_env_var("CYBERWAVE_DRIVER_STARTUP_PROBE_SECONDS", None)
    if override:
        try:
            return max(1, int(str(override).strip()))
        except ValueError:
            logger.warning(
                "Invalid CYBERWAVE_DRIVER_STARTUP_PROBE_SECONDS=%r; using default %ds",
                override,
                _DEFAULT_DRIVER_STARTUP_PROBE_SECONDS,
            )
    return _DEFAULT_DRIVER_STARTUP_PROBE_SECONDS


def driver_create_timeout_seconds(get_runtime_env_var: GetRuntimeEnvVar) -> int:
    """Return the ``docker create`` subprocess timeout in seconds."""
    override = get_runtime_env_var("CYBERWAVE_DRIVER_CREATE_TIMEOUT_SECONDS", None)
    if override:
        try:
            return max(1, int(str(override).strip()))
        except ValueError:
            logger.warning(
                "Invalid CYBERWAVE_DRIVER_CREATE_TIMEOUT_SECONDS=%r; using default %ds",
                override,
                _DEFAULT_DRIVER_CREATE_TIMEOUT_SECONDS,
            )
    return _DEFAULT_DRIVER_CREATE_TIMEOUT_SECONDS


def driver_remove_timeout_seconds(get_runtime_env_var: GetRuntimeEnvVar) -> int:
    """Return the timeout for removing a same-named container before create."""
    override = get_runtime_env_var("CYBERWAVE_DRIVER_REMOVE_TIMEOUT_SECONDS", None)
    if override:
        try:
            return max(1, int(str(override).strip()))
        except ValueError:
            logger.warning(
                "Invalid CYBERWAVE_DRIVER_REMOVE_TIMEOUT_SECONDS=%r; using default %ds",
                override,
                _DEFAULT_DRIVER_REMOVE_TIMEOUT_SECONDS,
            )
    return _DEFAULT_DRIVER_REMOVE_TIMEOUT_SECONDS


def remove_existing_container(
    container_name: str,
    *,
    get_runtime_env_var: GetRuntimeEnvVar,
) -> bool:
    """Idempotently remove a same-named container before ``docker create``.

    1. Skip immediately when no such container exists.
    2. ``docker stop`` first (bounded) so the entrypoint can release held
       devices cleanly instead of being ``SIGKILL``'d mid-I/O.
    3. ``docker rm -f`` with a generous, configurable timeout — neither call
       raises (both go through ``docker_helpers``).

    Returns ``True`` when no container with that name remains afterwards.
    """
    if docker_helpers.docker_inspect(container_name) is None:
        return True

    timeout = driver_remove_timeout_seconds(get_runtime_env_var)
    docker_helpers.docker_stop(container_name, timeout=timeout)
    docker_helpers.docker_rm(container_name, timeout=timeout)

    if docker_helpers.docker_inspect(container_name) is not None:
        logger.error(
            "Could not remove existing container %s within %ds; it may be "
            "wedged on a held device (USB audio, etc.)",
            container_name,
            timeout,
        )
        return False
    return True


def docker_create_argv_from_run_argv(run_argv: list[str]) -> list[str]:
    """Convert a ``docker run --detach ...`` argv into ``docker create ...``."""
    if len(run_argv) < 2 or run_argv[0] != "docker" or run_argv[1] != "run":
        raise ValueError(f"Expected docker run argv, got: {run_argv[:4]!r}...")
    rest = run_argv[2:]
    if rest and rest[0] == "--detach":
        rest = rest[1:]
    return ["docker", "create", *rest]


@dataclass(frozen=True)
class ContainerProbeResult:
    success: bool
    last_status: str


def probe_container_startup(
    container_name: str,
    *,
    probe_seconds: int,
    poll_interval: float = 1.0,
) -> ContainerProbeResult:
    """Poll container state until running, failed, or the probe budget expires."""
    last_status = "unknown"
    for _ in range(probe_seconds):
        inspect_data = docker_helpers.docker_inspect(container_name)
        if inspect_data is None:
            last_status = "none"
            time.sleep(poll_interval)
            continue

        state = inspect_data.get("State")
        if not isinstance(state, dict):
            last_status = "unknown"
            time.sleep(poll_interval)
            continue

        last_status = str(state.get("Status", "unknown")).lower()

        if last_status == "running":
            return ContainerProbeResult(success=True, last_status=last_status)

        # A zero exit code means the container ran to completion cleanly
        # (e.g. hello-world).  Treat as success so stream_logs() can forward
        # any output before Docker's restart policy relaunches the container.
        # We check both "exited" and "restarting" because the 1-second poll
        # may catch the container in either state between restart cycles.
        if last_status in {"exited", "restarting"}:
            exit_code = int(state.get("ExitCode", -1))
            if exit_code == 0:
                return ContainerProbeResult(success=True, last_status=last_status)

        if last_status in {"exited", "dead"}:
            restart_count = int(inspect_data.get("RestartCount", 0))
            if restart_count > 0 and last_status == "exited":
                logger.debug(
                    "Container %s exited (restarts=%d); restart policy may revive it",
                    container_name,
                    restart_count,
                )
                time.sleep(poll_interval)
                continue
            error_msg = str(state.get("Error", "")).strip() or "none"
            exit_code = state.get("ExitCode", "?")
            logger.error(
                "Container %s failed to start (status=%s, exit_code=%s, error=%s)",
                container_name,
                last_status,
                exit_code,
                error_msg,
            )
            return ContainerProbeResult(success=False, last_status=last_status)

        time.sleep(poll_interval)

    return ContainerProbeResult(success=False, last_status=last_status)


def _terminate_start_process(start_proc: subprocess.Popen[str] | None) -> None:
    if start_proc is None:
        return
    if start_proc.poll() is not None:
        return
    try:
        start_proc.terminate()
        start_proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        try:
            start_proc.kill()
        except OSError:
            pass


def _reap_start_process(start_proc: subprocess.Popen[str] | None) -> str:
    """Reap the detached ``docker start`` process and return its stderr.

    ``docker start <name>`` exits on its own almost immediately, but the Popen
    child must be waited on or it lingers as a zombie — on a long-lived edge
    host that relaunches many drivers these accumulate.  Draining the pipes
    also surfaces the daemon's error message (device / permission / cgroup
    failures) which is otherwise discarded.
    """
    if start_proc is None:
        return ""
    try:
        _, stderr = start_proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _terminate_start_process(start_proc)
        return ""
    except (OSError, ValueError):
        return ""
    return (stderr or "").strip()


def _docker_start_popen(container_name: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["docker", "start", container_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _retry_docker_start(container_name: str) -> bool:
    try:
        subprocess.run(
            ["docker", "start", container_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Recovery docker start failed for %s: %s", container_name, exc)
        return False


def launch_detached_container(
    *,
    container_name: str,
    run_argv: list[str],
    get_runtime_env_var: GetRuntimeEnvVar,
    on_container_created: Callable[[], None],
    on_running: Callable[[], None],
    on_failure: Callable[[str, str], None],
    stream_logs: Callable[[], None] | None = None,
    on_removed: Callable[[], None] | None = None,
) -> bool:
    """Create a container, start it without blocking, and probe until running.

    On definitive failure the container is force-removed so reconcile does not
    leave orphans in ``created``.  ``on_removed`` (when provided) fires right
    after that force-removal so the caller can drop any state it registered in
    ``on_container_created`` (e.g. the container→twin map).
    """
    create_argv = docker_create_argv_from_run_argv(run_argv)
    create_timeout = driver_create_timeout_seconds(get_runtime_env_var)
    probe_seconds = driver_startup_probe_seconds(get_runtime_env_var)
    recovery_seconds = _DEFAULT_DRIVER_START_RECOVERY_SECONDS

    try:
        subprocess.run(
            create_argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=create_timeout,
        )
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Failed to create container %s: %s",
            container_name,
            (exc.stderr or "").strip() or exc,
        )
        on_failure(f"Failed to create container {container_name}.", "docker_create_failed")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Docker create timed out for container %s", container_name)
        on_failure(
            f"Docker create timed out for container {container_name}.",
            "docker_create_timeout",
        )
        return False

    on_container_created()

    try:
        start_proc: subprocess.Popen[str] | None = _docker_start_popen(container_name)
    except OSError as exc:
        logger.error("Failed to start container %s: %s", container_name, exc)
        docker_helpers.docker_rm(container_name)
        if on_removed is not None:
            on_removed()
        on_failure(f"Failed to start container {container_name}.", "docker_start_failed")
        return False

    probe = probe_container_startup(container_name, probe_seconds=probe_seconds)

    if not probe.success and probe.last_status == "created":
        # The detached start is wedged; reap it and retry synchronously.
        _terminate_start_process(start_proc)
        start_proc = None
        if _retry_docker_start(container_name):
            probe = probe_container_startup(
                container_name,
                probe_seconds=recovery_seconds,
            )

    # Reap the detached ``docker start`` (avoids zombies) and capture any
    # error it printed so a failed launch is diagnosable.
    start_stderr = _reap_start_process(start_proc)

    if probe.success:
        if stream_logs is not None:
            stream_logs()
        on_running()
        return True

    logger.error(
        "Container %s did not reach running within startup budget "
        "(probe=%ds + recovery=%ds, last_status=%s, docker_start_stderr=%s)",
        container_name,
        probe_seconds,
        recovery_seconds,
        probe.last_status,
        start_stderr or "none",
    )
    docker_helpers.docker_rm(container_name)
    if on_removed is not None:
        on_removed()
    phase = (
        "container_unhealthy"
        if probe.last_status in {"exited", "dead", "none"}
        else "container_startup_timeout"
    )
    on_failure(
        (
            f"Driver container {container_name} failed to start cleanly "
            f"(status={probe.last_status})."
        ),
        phase,
    )
    return False
