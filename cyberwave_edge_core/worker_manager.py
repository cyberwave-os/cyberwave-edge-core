"""Worker container lifecycle management for Cyberwave Edge Core.

Manages a single ``cyberwave-worker-{env_uuid[:8]}`` container per edge device.
The worker container runs ML worker scripts from the local workers directory
and shares model weights from the local models cache.

One worker container per edge device (not per twin): workers consume data from
all twins in the environment, so a shared container is simpler and avoids
model duplication.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .docker_helpers import (
    docker_available,
    docker_container_status,
    docker_has_nvidia_runtime,
    docker_logs_follow,
    docker_rm,
)

if TYPE_CHECKING:
    from .worker_health import WorkerHealthMonitor, WorkerHealthState

logger = logging.getLogger(__name__)

WORKER_CONTAINER_PREFIX = "cyberwave-worker-"
DEFAULT_WORKER_IMAGE = "cyberwaveos/ml-worker:latest"


@dataclass
class ResourceLimits:
    """Optional resource constraints for the worker container.

    Set any field to a non-None value to pass the corresponding ``docker run``
    flag.  Fields that remain None are omitted (Docker defaults apply).

    Examples::

        ResourceLimits(cpu_quota_percent=50, memory_mb=2048)
        ResourceLimits(memory_mb=4096, gpu_memory_fraction=0.5)
    """

    cpu_quota_percent: Optional[float] = None  # 0–100; maps to --cpu-quota / --cpu-period
    memory_mb: Optional[int] = None  # --memory
    gpu_memory_fraction: Optional[float] = None  # passed as CYBERWAVE_GPU_MEM_FRACTION env var

    def to_docker_args(self) -> list[str]:
        """Return the docker run flags corresponding to these limits."""
        args: list[str] = []
        if self.cpu_quota_percent is not None:
            period = 100_000  # 100 ms in microseconds
            quota = int(period * self.cpu_quota_percent / 100)
            args += ["--cpu-period", str(period), "--cpu-quota", str(quota)]
        if self.memory_mb is not None:
            args += ["--memory", f"{self.memory_mb}m"]
        return args

    def to_env_args(self) -> list[str]:
        """Return extra -e flags for soft limits that live in env vars."""
        args: list[str] = []
        if self.gpu_memory_fraction is not None:
            args += ["-e", f"CYBERWAVE_GPU_MEM_FRACTION={self.gpu_memory_fraction:.4f}"]
        return args


@dataclass
class WorkerStatus:
    """Snapshot of the worker container state."""

    container_name: str
    status: str  # running / exited / none / unknown
    workers_dir: Path
    models_dir: Path
    worker_files: list[str] = field(default_factory=list)
    gpu_enabled: bool = False
    image: str = DEFAULT_WORKER_IMAGE
    # Health / restart fields populated when a WorkerHealthMonitor is attached.
    restart_count: int = 0
    recent_restarts: int = 0
    circuit_breaker_tripped: bool = False
    health_state: Optional["WorkerHealthState"] = None


class WorkerManager:
    """Manage the lifecycle of the edge worker container.

    All operations are idempotent and non-blocking from the caller's
    perspective — container operations run synchronously within the method
    call but do not block the runtime loop beyond their own duration.

    The manager is intentionally stateless (reads config on every call) so
    it works correctly after a process restart without persisting extra state.

    An optional ``WorkerHealthMonitor`` can be attached via
    ``set_health_monitor()``.  When attached, ``restart()`` records restart
    events and respects circuit-breaker state to prevent crash loops.
    """

    def __init__(
        self,
        *,
        config_dir: Path,
        environment_uuid: str,
        token: str,
        twin_uuids: Optional[list[str]] = None,
        image: str = DEFAULT_WORKER_IMAGE,
        resource_limits: Optional[ResourceLimits] = None,
    ) -> None:
        self._config_dir = config_dir
        self._environment_uuid = environment_uuid
        self._token = token
        self._twin_uuids: list[str] = list(twin_uuids or [])
        self._image = image
        self._resource_limits = resource_limits
        self._container_name = f"{WORKER_CONTAINER_PREFIX}{environment_uuid[:8]}"
        self._health_monitor: Optional["WorkerHealthMonitor"] = None
        self._log_thread: Optional[threading.Thread] = None
        self._log_thread_lock = threading.Lock()

    def set_health_monitor(self, monitor: "WorkerHealthMonitor") -> None:
        """Attach a WorkerHealthMonitor to this manager.

        Once attached, ``restart()`` will:
        - Check ``monitor.is_restart_allowed()`` before proceeding.
        - Record each restart attempt via ``monitor.record_restart()``.
        - Record successful starts via ``monitor.record_start()``.
        """
        self._health_monitor = monitor

    @property
    def container_name(self) -> str:
        """Return the Docker container name used by this manager."""
        return self._container_name

    @property
    def health_monitor(self) -> Optional["WorkerHealthMonitor"]:
        """Return the attached WorkerHealthMonitor, or None if not set."""
        return self._health_monitor

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the worker container if workers exist and it isn't already running.

        Returns True if the container is running after this call (including
        the case where it was already running).  Returns False only on error.
        """
        if not docker_available():
            logger.error("Docker is not available; cannot start worker container")
            return False

        workers_dir = self._workers_dir()
        if not self._has_worker_files(workers_dir):
            logger.info(
                "No worker files found in %s; skipping worker container startup", workers_dir
            )
            return True

        current_status = docker_container_status(self._container_name)
        if current_status == "running":
            logger.info("Worker container %s is already running", self._container_name)
            self._ensure_log_stream()
            return True

        logger.info("Starting worker container %s (image=%s)", self._container_name, self._image)
        ok = self._run_container()
        if ok and self._health_monitor is not None:
            self._health_monitor.record_start()
        return ok

    def stop(self) -> bool:
        """Stop and remove the worker container.

        Returns True when the container is gone after the call.
        """
        if not docker_available():
            return True

        current_status = docker_container_status(self._container_name)
        if current_status == "none":
            logger.debug("Worker container %s not found; nothing to stop", self._container_name)
            return True

        logger.info("Stopping worker container %s", self._container_name)
        return docker_rm(self._container_name)

    def restart(self, *, reason: str = "requested") -> bool:
        """Stop and re-start the worker container.

        Model re-download is handled by callers (WorkerWatcher) before calling
        this method.

        When a ``WorkerHealthMonitor`` is attached:
        - The restart is blocked if the circuit-breaker is tripped.
        - The restart event is recorded (with *reason*) regardless of outcome.

        Returns True on successful restart.
        """
        if self._health_monitor is not None and not self._health_monitor.is_restart_allowed():
            logger.warning(
                "Worker restart blocked for %s: circuit-breaker is tripped (reason=%r)",
                self._container_name,
                reason,
            )
            return False

        logger.info("Restarting worker container %s (reason=%r)", self._container_name, reason)
        if not self.stop():
            logger.warning(
                "Failed to stop worker container %s; attempting fresh start anyway",
                self._container_name,
            )
        ok = self.start()

        if self._health_monitor is not None:
            self._health_monitor.record_restart(reason=reason, success=ok)

        return ok

    def status(self) -> WorkerStatus:
        """Return a snapshot of the current worker container state."""
        workers_dir = self._workers_dir()
        models_dir = self._models_dir()

        worker_files = (
            sorted(p.name for p in workers_dir.glob("*.py") if p.is_file())
            if workers_dir.exists()
            else []
        )

        container_status = docker_container_status(self._container_name)
        gpu_enabled = docker_has_nvidia_runtime()

        health_state = None
        restart_count = 0
        recent_restarts = 0
        circuit_breaker_tripped = False

        if self._health_monitor is not None:
            health_state = self._health_monitor.check(container_status)
            restart_count = health_state.restart_count
            recent_restarts = health_state.recent_restarts
            circuit_breaker_tripped = health_state.circuit_breaker_tripped

        return WorkerStatus(
            container_name=self._container_name,
            status=container_status,
            workers_dir=workers_dir,
            models_dir=models_dir,
            worker_files=worker_files,
            gpu_enabled=gpu_enabled,
            image=self._image,
            restart_count=restart_count,
            recent_restarts=recent_restarts,
            circuit_breaker_tripped=circuit_breaker_tripped,
            health_state=health_state,
        )

    def logs(self, *, follow: bool = True) -> None:
        """Stream worker container logs to stdout (blocking when follow=True)."""
        if not docker_available():
            logger.error("Docker is not available")
            return

        if follow:
            try:
                subprocess.run(
                    ["docker", "logs", "-f", "--tail", "100", self._container_name],
                    check=False,
                )
            except (OSError, KeyboardInterrupt):
                pass
        else:
            try:
                subprocess.run(
                    ["docker", "logs", "--tail", "100", self._container_name],
                    check=False,
                )
            except OSError as exc:
                logger.error("Failed to fetch container logs: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _workers_dir(self) -> Path:
        return self._config_dir / "workers"

    def _models_dir(self) -> Path:
        return self._config_dir / "models"

    def _has_worker_files(self, workers_dir: Path) -> bool:
        if not workers_dir.exists():
            return False
        return any(workers_dir.glob("*.py"))

    def _build_env_vars(self) -> dict[str, str]:
        """Build environment variables for the worker container."""
        from .startup import (
            DEFAULT_API_URL,
            DEFAULT_ENVIRONMENT,
            get_runtime_env_var,
            load_credentials_envs,
        )

        env: dict[str, str] = {
            "CYBERWAVE_API_KEY": self._token,
            "CYBERWAVE_EDGE_CONFIG_DIR": "/app/.cyberwave",
            "CYBERWAVE_DATA_BACKEND": "zenoh",
        }

        if self._environment_uuid:
            env["CYBERWAVE_ENVIRONMENT_UUID"] = self._environment_uuid

        if self._twin_uuids:
            env["CYBERWAVE_TWIN_UUIDS"] = ",".join(self._twin_uuids)

        base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
        env["CYBERWAVE_BASE_URL"] = base_url

        mqtt_host = get_runtime_env_var("CYBERWAVE_MQTT_HOST")
        if mqtt_host:
            env["CYBERWAVE_MQTT_HOST"] = mqtt_host

        runtime_environment = (
            get_runtime_env_var("CYBERWAVE_ENVIRONMENT", DEFAULT_ENVIRONMENT) or DEFAULT_ENVIRONMENT
        ).lower()
        if runtime_environment != "production":
            env["CYBERWAVE_ENVIRONMENT"] = runtime_environment

        zenoh_connect = get_runtime_env_var("ZENOH_CONNECT")
        if zenoh_connect:
            env["ZENOH_CONNECT"] = zenoh_connect

        if platform.system() == "Linux":
            env["ZENOH_SHM_ENABLED"] = "true"

        for key, value in load_credentials_envs().items():
            if key.startswith("CYBERWAVE_"):
                env.setdefault(key, value)

        for key, value in os.environ.items():
            if key.startswith("CYBERWAVE_") and isinstance(value, str) and value.strip():
                env.setdefault(key, value.strip())

        env["CYBERWAVE_EDGE_HOST_PLATFORM"] = platform.system().lower()
        return env

    def _build_volume_args(self) -> list[str]:
        """Build -v mount arguments for the worker container."""
        config_dir = self._config_dir
        workers_dir = self._workers_dir()
        models_dir = self._models_dir()

        workers_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        return [
            "-v",
            f"{config_dir}:/app/.cyberwave",
            "-v",
            f"{workers_dir}:/app/workers:ro",
            "-v",
            f"{models_dir}:/app/models:ro",
        ]

    def _build_network_args(self) -> list[str]:
        """Build network args; mirrors driver container networking."""
        if platform.system() == "Darwin":
            return ["--add-host", "host.docker.internal:host-gateway"]
        return ["--network", "host"]

    def _run_container(self) -> bool:
        """Pull (if needed) and run the worker container. Returns True on success."""
        from .startup import DEFAULT_ENVIRONMENT, get_runtime_env_var

        image = self._image
        runtime_environment = (
            get_runtime_env_var("CYBERWAVE_ENVIRONMENT", DEFAULT_ENVIRONMENT) or DEFAULT_ENVIRONMENT
        ).lower()

        if ":" not in image and runtime_environment != "production":
            image = f"{image}:{runtime_environment}"

        docker_rm(self._container_name)

        env_vars = self._build_env_vars()
        env_args: list[str] = []
        for key, value in env_vars.items():
            env_args += ["-e", f"{key}={value}"]

        volume_args = self._build_volume_args()
        network_args = self._build_network_args()

        gpu_args: list[str] = []
        if docker_has_nvidia_runtime():
            gpu_args = ["--gpus", "all"]
            logger.info("NVIDIA runtime detected; adding --gpus all to worker container")

        resource_args: list[str] = []
        resource_env_args: list[str] = []
        if self._resource_limits is not None:
            resource_args = self._resource_limits.to_docker_args()
            resource_env_args = self._resource_limits.to_env_args()
            if resource_args or resource_env_args:
                logger.info(
                    "Applying resource limits to worker container %s: %s",
                    self._container_name,
                    self._resource_limits,
                )

        cmd = [
            "docker",
            "run",
            "--detach",
            "--restart",
            "unless-stopped",
            *network_args,
            "--name",
            self._container_name,
            *gpu_args,
            *resource_args,
            *volume_args,
            *env_args,
            *resource_env_args,
            image,
        ]

        logger.info("Starting worker container %s from image %s", self._container_name, image)
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed to start worker container %s: %s", self._container_name, exc.stderr
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("Docker run timed out for worker container %s", self._container_name)
            return False

        self._ensure_log_stream()

        for _ in range(5):
            status = docker_container_status(self._container_name)
            if status == "running":
                logger.info("Worker container %s is running", self._container_name)
                return True
            if status in {"exited", "dead"}:
                logger.error(
                    "Worker container %s failed to start (status=%s)",
                    self._container_name,
                    status,
                )
                return False
            time.sleep(1.0)

        logger.error(
            "Worker container %s did not reach running state within startup probe window",
            self._container_name,
        )
        return False

    def _ensure_log_stream(self) -> None:
        """Attach a background log-streaming thread if not already running."""
        with self._log_thread_lock:
            if self._log_thread and self._log_thread.is_alive():
                return

            thread = threading.Thread(
                target=_follow_worker_logs,
                args=(self._container_name,),
                name=f"worker-logs-{self._container_name}",
                daemon=True,
            )
            self._log_thread = thread
            thread.start()


def _follow_worker_logs(container_name: str) -> None:
    """Background thread: stream worker container logs into the service logger."""
    process = docker_logs_follow(container_name)
    if process is None:
        return
    try:
        if process.stdout:
            for line in process.stdout:
                message = line.rstrip()
                if message:
                    logger.info("[worker:%s] %s", container_name, message)
    except Exception:
        pass
    finally:
        try:
            process.wait(timeout=2)
        except Exception:
            pass


def get_zenoh_env_vars() -> dict[str, str]:
    """Return Zenoh-related env vars to inject into any managed container.

    This is additive — existing drivers that don't use the SDK data layer
    simply ignore these vars. Drivers adopting ``cw.data.publish()`` will
    pick them up automatically.
    """
    from .startup import get_runtime_env_var

    env: dict[str, str] = {
        "CYBERWAVE_DATA_BACKEND": "zenoh",
    }

    zenoh_connect = get_runtime_env_var("ZENOH_CONNECT")
    if zenoh_connect:
        env["ZENOH_CONNECT"] = zenoh_connect

    return env
