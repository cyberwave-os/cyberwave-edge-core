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
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .docker_args import (
    _rewrite_macos_container_base_url,
    _rewrite_macos_container_hostname,
)
from .docker_helpers import (
    build_user_args,
    docker_available,
    docker_container_status,
    docker_has_nvidia_runtime,
    docker_inspect,
    docker_rm,
    docker_stop,
    group_gid,
)
from .zenoh_config import ZenohConfig, build_zenoh_env_vars

if TYPE_CHECKING:
    from .worker_health import WorkerHealthMonitor, WorkerHealthState

logger = logging.getLogger(__name__)

# Tag basenames that operators expect to roll forward over time. Anything
# matching one of these (optionally suffixed with ``-gpu``/``-cpu``/etc.)
# is treated as mutable and re-pulled on every worker (re)start so a
# freshly published image is picked up without the operator having to
# remember to ``docker rmi`` first. Immutable tags (e.g. ``v1.2.3``,
# sha digests, dated build IDs) keep the previous fast-path that skips
# the pull when the image is already present locally.
_MUTABLE_TAG_BASENAMES = frozenset(
    {"latest", "dev", "local", "staging", "nightly", "edge", "main", "master"}
)

_DEFAULT_STARTUP_PROBE_SECONDS = 30


def _image_tag_is_mutable(image: str) -> bool:
    """Return True for image references whose tag rolls forward over time.

    Examples:
        ``cyberwaveos/edge-ml-worker``           -> True (no tag → ``latest``)
        ``cyberwaveos/edge-ml-worker:latest``    -> True
        ``cyberwaveos/edge-ml-worker:dev``       -> True
        ``cyberwaveos/edge-ml-worker:dev-gpu``   -> True
        ``cyberwaveos/edge-ml-worker:local``     -> True
        ``cyberwaveos/edge-ml-worker:v1.2.3``    -> False
        ``cyberwaveos/edge-ml-worker@sha256:...`` -> False
        ``myregistry.io:5000/cyberwaveos/img``   -> True (registry port, no tag)
        ``myregistry.io:5000/cyberwaveos/img:dev`` -> True
    """
    if "@" in image:
        # Pinned-by-digest references are immutable by definition.
        return False
    if ":" not in image:
        # Bare image refs default to ``:latest`` which is mutable.
        return True
    tag = image.rsplit(":", 1)[1]
    if not tag or "/" in tag:
        # Either no actual tag, or the colon was part of a registry port
        # (e.g. ``myregistry.io:5000/img``); docker resolves both to
        # ``:latest`` which is mutable.
        return True
    # Strip arch/runtime suffixes such as ``-gpu``, ``-cpu``, ``-arm64``.
    base = tag.split("-", 1)[0].lower()
    return base in _MUTABLE_TAG_BASENAMES


def _docker_image_present(image: str) -> bool:
    """Return True if ``docker image inspect`` finds *image* locally."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def _ensure_dir_writable_by_container_user(path: Path) -> None:
    """Best-effort chown so the container user (``os.getuid()``) can write.

    When edge-core runs as root (via systemd or ``sudo``), directories
    it creates are root-owned.  The worker container runs as the invoking
    user via ``build_user_args()``, so it cannot write to root-owned
    mounts.

    Target uid/gid resolution:

    * Non-root caller → use ``os.getuid()``/``os.getgid()`` directly.
    * Root caller → defer to
      :func:`cyberwave_edge_core.startup.resolve_config_owner_uid_gid`,
      which understands both ``sudo`` (``SUDO_UID`` set) and systemd
      (fall back to the owner of ``CONFIG_DIR.parent``).
    """
    if platform.system() != "Linux":
        return
    try:
        st = path.stat()
    except OSError:
        return

    if os.getuid() != 0:
        target_uid = os.getuid()
        target_gid = os.getgid()
    else:
        from .startup import resolve_config_owner_uid_gid

        resolved = resolve_config_owner_uid_gid()
        if resolved is None:
            return
        target_uid, target_gid = resolved

    if st.st_uid != target_uid:
        try:
            os.chown(path, target_uid, target_gid)
        except OSError:
            logger.debug("Cannot chown %s to uid %d", path, target_uid)


WORKER_CONTAINER_PREFIX = "cyberwave-worker-"
_WORKER_IMAGE_BASE = "cyberwaveos/edge-ml-worker"
DEFAULT_WORKER_IMAGE = f"{_WORKER_IMAGE_BASE}:latest"

# One lock per container name. Prevents concurrent start() / _run_container()
# calls — from MQTT handlers, the runtime reconcile loop, the watcher, and
# edge-restart flows — from racing on docker rm + docker run for the same name.
_CONTAINER_START_LOCKS: dict[str, threading.Lock] = {}
_CONTAINER_START_LOCKS_MUTEX = threading.Lock()


def _get_container_start_lock(container_name: str) -> threading.Lock:
    """Return (creating if needed) the per-container start lock."""
    with _CONTAINER_START_LOCKS_MUTEX:
        if container_name not in _CONTAINER_START_LOCKS:
            _CONTAINER_START_LOCKS[container_name] = threading.Lock()
        return _CONTAINER_START_LOCKS[container_name]

#: Host device node exposed by the Hailo PCIe driver (HailoRT). Presence of
#: this path is the cheapest way to detect a Hailo accelerator on the host
#: without paying the cost of forking ``hailortcli``.
HAILO_DEVICE_PATH = "/dev/hailo0"

#: Env var read by the Hailo worker image's entrypoint. Listed device paths
#: that don't exist inside the container cause the entrypoint to exit
#: with a clear "Gate 4" message instead of crashing inside HailoRT.
_REQUIRED_DEVICES_ENV = "CYBERWAVE_REQUIRED_DEVICES"


def _hailo_device_present() -> bool:
    """Return True when ``/dev/hailo0`` exists on the host.

    Linux-only check (the device node is only created by the Hailo PCIe
    kernel driver). Operators on macOS / Windows never get the Hailo
    passthrough, which matches reality: HailoRT only supports Linux.
    """
    if platform.system() != "Linux":
        return False
    return Path(HAILO_DEVICE_PATH).exists()


def _apply_hailo_image_tag(image: str) -> str:
    """Append ``-hailo`` to the worker image tag when applicable.

    Mirrors the ``-gpu`` rewrite in :meth:`WorkerManager._run_container`:
    only ``cyberwaveos/edge-ml-worker:<tag>`` references are rewritten,
    and only when the tag is not already a Hailo variant. Custom
    operator overrides (``CYBERWAVE_WORKER_IMAGE``) and the ``-gpu``
    fork are left untouched — Hailo + NVIDIA on the same host is not
    a supported combination and ``-gpu`` wins because it's the
    higher-priority accelerator for the rest of the catalog.
    """
    if not image.startswith(f"{_WORKER_IMAGE_BASE}:"):
        return image
    if image.endswith("-hailo") or image.endswith("-gpu"):
        return image
    return f"{image}-hailo"


def _build_hailo_passthrough_args() -> list[str]:
    """Return the ``docker run`` flags that expose ``/dev/hailo0`` to the worker.

    Always emits the ``--device`` flag and the ``CYBERWAVE_REQUIRED_DEVICES``
    env var (consumed by the Hailo image's entrypoint for Gate 4).
    Conditionally adds ``--group-add <gid>`` for the ``hailo`` group
    when it exists on the host: HailoRT versions <4.20 ship the device
    node with ``hailo``-group ownership, while 4.20+ make it
    world-accessible (0666) and the group isn't created. ``group_gid``
    returns ``None`` in the latter case and we skip the flag.
    """
    args: list[str] = [
        "--device",
        f"{HAILO_DEVICE_PATH}:{HAILO_DEVICE_PATH}:rwm",
    ]
    gid = group_gid("hailo")
    if gid is not None:
        args += ["--group-add", str(gid)]
    args += ["-e", f"{_REQUIRED_DEVICES_ENV}={HAILO_DEVICE_PATH}"]
    return args


def resolve_worker_image() -> str:
    """Return the worker image reference for the current environment.

    Resolution order:

    1. ``CYBERWAVE_WORKER_IMAGE`` env var — explicit local override.  The
       same escape hatch :func:`load_driver_overrides` provides for
       drivers via ``credentials.json``: lets an operator pin a custom
       (e.g. ``cyberwaveos/edge-ml-worker:local-gpu``) build without
       round-tripping through cloud config.  Useful for hot-patches and
       SDK debugging.  When the override resolves to a tag that the
       registry does not have, ``_ensure_image_pulled`` falls back to
       the locally-present image (same code path the ``:local`` tag
       relies on for camera drivers).
    2. ``CYBERWAVE_ENVIRONMENT`` env var (``dev``, ``staging`` …) maps
       to the matching image tag.
    3. Production and unknown environments use ``:latest``.
    """
    from .startup import get_runtime_env_var

    override = (get_runtime_env_var("CYBERWAVE_WORKER_IMAGE") or "").strip()
    if override:
        return override

    env_name = (get_runtime_env_var("CYBERWAVE_ENVIRONMENT") or "").strip().lower()
    if env_name and env_name != "production":
        return f"{_WORKER_IMAGE_BASE}:{env_name}"
    return DEFAULT_WORKER_IMAGE


_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"
_DIM = "\033[2m"
_LOG_RE = re.compile(
    r"(\[(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\])"
    r"(\s+\[[^\]]+\])"
)


def _colorize_log_line(line: str) -> str:
    """Add ANSI colors to a plain log line at display time."""
    m = _LOG_RE.search(line)
    if not m:
        return line
    level_tag = m.group(1)
    name_tag = m.group(2)
    level = level_tag[1:-1]
    color = _LEVEL_COLORS.get(level, "")
    colored = f"{color}{level_tag}{_RESET}{_DIM}{name_tag}{_RESET}"
    return line[: m.start()] + colored + line[m.end() :]


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

        Returns True if the container is running (or is expected to reach
        running state) after this call.  Returns False only on definitive
        errors where the container cannot possibly come up (e.g. Docker
        unavailable, image missing, container immediately exited/dead).
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
        if current_status in {"running", "restarting"}:
            logger.info(
                "Worker container %s is already running/restarting (status=%s)",
                self._container_name,
                current_status,
            )
            return True

        logger.info("Starting worker container %s (image=%s)", self._container_name, self._image)
        lock = _get_container_start_lock(self._container_name)
        with lock:
            # Re-check inside the lock: a concurrent caller may have started it
            # while we were waiting.
            current_status = docker_container_status(self._container_name)
            if current_status in {"running", "restarting"}:
                logger.info(
                    "Worker container %s reached running/restarting state while waiting "
                    "for start lock (status=%s); skipping redundant start",
                    self._container_name,
                    current_status,
                )
                return True
            ok = self._run_container()
        if ok and self._health_monitor is not None:
            self._health_monitor.record_start()
        return ok

    def stop(self, *, reason: str = "requested") -> bool:
        """Gracefully stop the worker container without removing it.

        Leaves the container in ``exited`` state so operators can still
        ``docker logs`` it for diagnostics and so callers that want the
        container brought back up (e.g. ``WorkerManager.restart``,
        :func:`reconcile_worker_lifecycle`, the symmetric edge-restart
        flow) can do so cheaply via the regular start path.

        Returns ``True`` when the container is no longer running after
        the call (including the no-op cases where it didn't exist or was
        already stopped). The destructive ``docker rm`` happens later,
        only inside :meth:`_run_container`, so the next start always gets
        a freshly-created container.

        When a ``WorkerHealthMonitor`` is attached AND ``docker_stop``
        actually transitioned a running container, ``record_stop()``
        is called so the next health probe doesn't false-positive on
        the running→exited transition. The short-circuit paths
        (container already non-running) skip the notification because
        there's no transition to suppress. *reason* is logged on the
        monitor for traceability.
        """
        if not docker_available():
            return True

        current_status = docker_container_status(self._container_name)
        if current_status in {"none", "exited", "created"}:
            logger.debug(
                "Worker container %s already in non-running state %r; nothing to stop",
                self._container_name,
                current_status,
            )
            return True

        logger.info(
            "Stopping worker container %s (status=%s) reason=%r",
            self._container_name,
            current_status,
            reason,
        )
        ok = docker_stop(self._container_name)
        if ok and self._health_monitor is not None:
            self._health_monitor.record_stop(reason=reason)
        return ok

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

        cmd = ["docker", "logs", "--tail", "100"]
        if follow:
            cmd.append("-f")
        cmd.append(self._container_name)

        use_color = sys.stdout.isatty()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            if proc.stdout:
                for line in proc.stdout:
                    text = line.rstrip()
                    if text:
                        print(_colorize_log_line(text) if use_color else text)
            proc.wait()
        except (OSError, KeyboardInterrupt):
            pass

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
            env.setdefault("CYBERWAVE_TWIN_UUID", self._twin_uuids[0])

        base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
        env["CYBERWAVE_BASE_URL"] = _rewrite_macos_container_base_url(base_url)

        for mqtt_key in (
            "CYBERWAVE_MQTT_HOST",
            "CYBERWAVE_MQTT_PORT",
            "CYBERWAVE_MQTT_USERNAME",
            "CYBERWAVE_MQTT_USE_TLS",
        ):
            mqtt_val = get_runtime_env_var(mqtt_key)
            if mqtt_val:
                if mqtt_key == "CYBERWAVE_MQTT_HOST":
                    mqtt_val = _rewrite_macos_container_hostname(mqtt_val)
                env[mqtt_key] = mqtt_val

        # The Python SDK uses CYBERWAVE_API_KEY as the MQTT password.
        # Workers only need MQTT (no REST), so when a dedicated MQTT
        # password is configured, inject it as the API key.
        mqtt_password = get_runtime_env_var("CYBERWAVE_MQTT_PASSWORD")
        if mqtt_password:
            env["CYBERWAVE_API_KEY"] = mqtt_password

        runtime_environment = (
            get_runtime_env_var("CYBERWAVE_ENVIRONMENT", DEFAULT_ENVIRONMENT) or DEFAULT_ENVIRONMENT
        ).lower()
        if runtime_environment != "production":
            env["CYBERWAVE_ENVIRONMENT"] = runtime_environment

        zenoh_connect = get_runtime_env_var("ZENOH_CONNECT")
        if zenoh_connect:
            env["ZENOH_CONNECT"] = zenoh_connect

        # Expose a TCP listener so the CLI monitor can connect from the host.
        zenoh_listen = get_runtime_env_var("ZENOH_LISTEN")
        env["ZENOH_LISTEN"] = zenoh_listen or "tcp/0.0.0.0:7447"

        # Route the worker through the same Zenoh env builder that driver
        # containers use (see ``startup._run_docker_image``). This keeps the
        # two sides in lock-step: ``ZENOH_SHARED_MEMORY`` defaults to
        # ``"false"`` because SHM between Docker containers requires
        # ``--ipc=host``, which weakens isolation and has historically caused
        # instability. Operators opt in by exporting
        # ``ZENOH_SHARED_MEMORY=true`` in the edge-core process env.
        for key, value in build_zenoh_env_vars(ZenohConfig()).items():
            env.setdefault(key, value)

        for key, value in load_credentials_envs().items():
            if key.startswith("CYBERWAVE_"):
                env.setdefault(key, value)

        for key, value in os.environ.items():
            if key.startswith("CYBERWAVE_") and isinstance(value, str) and value.strip():
                env.setdefault(key, value.strip())

        env["CYBERWAVE_EDGE_HOST_PLATFORM"] = platform.system().lower()

        # Auto-infer MQTT TLS when port 8883 is configured but USE_TLS is absent.
        if "CYBERWAVE_MQTT_USE_TLS" not in env:
            if env.get("CYBERWAVE_MQTT_PORT") == "8883":
                env["CYBERWAVE_MQTT_USE_TLS"] = "true"

        return env

    def _build_volume_args(self) -> list[str]:
        """Build -v mount arguments for the worker container."""
        config_dir = self._config_dir
        workers_dir = self._workers_dir()
        models_dir = self._models_dir()

        workers_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        _ensure_dir_writable_by_container_user(models_dir)

        return [
            "-v",
            f"{config_dir}:/app/.cyberwave",
            "-v",
            f"{workers_dir}:/app/workers:ro",
            "-v",
            f"{models_dir}:/app/models",
        ]

    def _build_network_args(self) -> list[str]:
        """Build network args; mirrors driver container networking."""
        if platform.system() == "Darwin":
            return [
                "--add-host",
                "host.docker.internal:host-gateway",
                "-p",
                "7447:7447/tcp",
            ]
        return ["--network", "host"]

    @staticmethod
    def _ensure_image_pulled(image: str, timeout: int = 600) -> bool:
        """Pull *image* and make it ready for ``docker run``.

        For mutable tags (``latest``, ``dev``, ``local``, ``staging``,
        ``nightly``, ``edge`` and any of those with arch/runtime suffixes
        like ``dev-gpu``) we always issue ``docker pull`` so a developer
        who just rebuilt or pushed a new image gets the new digest on the
        next worker recycle. For immutable tags (e.g. ``v1.2.3``, sha256
        digests) we keep the previous fast-path of skipping the pull when
        the image is already present locally.

        If ``docker pull`` fails but a local copy exists, we fall back to
        the local copy and warn — losing connectivity should not knock
        the worker offline.

        Uses a generous timeout (default 10 min) to accommodate large GPU
        images on slow connections.  Returns True when the image is
        available locally.
        """
        has_local = _docker_image_present(image)
        mutable = _image_tag_is_mutable(image)

        if has_local and not mutable:
            logger.debug(
                "Image %s already present locally (immutable tag); skipping pull",
                image,
            )
            return True

        logger.info(
            "Pulling worker image %s (timeout=%ds, mutable_tag=%s, local_present=%s)...",
            image,
            timeout,
            mutable,
            has_local,
        )
        try:
            subprocess.run(
                ["docker", "pull", image],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            logger.info("Successfully pulled %s", image)
            return True
        except subprocess.CalledProcessError as exc:
            if has_local:
                logger.warning(
                    "Pull failed for %s; using local copy. stderr=%s",
                    image,
                    exc.stderr,
                )
                return True
            logger.error("Failed to pull worker image %s: %s", image, exc.stderr)
            return False
        except subprocess.TimeoutExpired:
            if has_local:
                logger.warning(
                    "Timed out pulling %s after %ds; using local copy",
                    image,
                    timeout,
                )
                return True
            logger.error("Timed out pulling worker image %s after %ds", image, timeout)
            return False

    def _pull_worker_image_with_progress(self, image: str, timeout: int = 600) -> bool:
        """Pull *image* with progress reporting to all linked twins.

        Creates a ``worker_starting`` alert on each linked twin for the
        duration of the pull so the workbench can display a progress bar
        (byte counts, percent) in analogy with the driver image pull.

        Falls back to the plain subprocess pull
        (:meth:`_ensure_image_pulled`) when no twin UUIDs are attached to
        this manager.  Fallback semantics for a failed pull with a local
        copy present are identical to those of :meth:`_ensure_image_pulled`.
        """
        if not self._twin_uuids:
            return self._ensure_image_pulled(image, timeout=timeout)

        from .driver_logs import (  # noqa: PLC0415
            _pull_docker_image_with_progress_multi,
            _PullDeliveryContext,
        )
        from .utils import WorkerStartingAlertContext  # noqa: PLC0415

        has_local = _docker_image_present(image)
        mutable = _image_tag_is_mutable(image)

        if has_local and not mutable:
            logger.debug(
                "Image %s already present locally (immutable tag); skipping pull",
                image,
            )
            return True

        logger.info(
            "Pulling worker image %s with progress "
            "(timeout=%ds, mutable_tag=%s, local_present=%s)...",
            image,
            timeout,
            mutable,
            has_local,
        )

        alert_ctxs: list[WorkerStartingAlertContext] = []
        for twin_uuid in self._twin_uuids:
            ctx = WorkerStartingAlertContext(
                twin_uuid=twin_uuid,
                image=image,
                container_name=self._container_name,
            )
            ctx.create()
            alert_ctxs.append(ctx)

        contexts = [
            _PullDeliveryContext(
                twin_uuid=twin_uuid,
                container_name=self._container_name,
                driver_alert_ctx=alert_ctxs[i],
            )
            for i, twin_uuid in enumerate(self._twin_uuids)
        ]

        try:
            _pull_docker_image_with_progress_multi(
                image,
                contexts=contexts,
                token=self._token,
                timeout=timeout,
            )
            logger.info("Successfully pulled %s", image)
            for ctx in alert_ctxs:
                ctx.resolve()
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if has_local:
                logger.warning(
                    "Pull failed/timed out for %s; using local copy. error=%s",
                    image,
                    exc,
                )
                for ctx in alert_ctxs:
                    ctx.resolve()
                return True
            error_str = (getattr(exc, "stderr", None) or str(exc))[:500]
            logger.error("Failed to pull worker image %s: %s", image, error_str)
            for ctx in alert_ctxs:
                ctx.mark_failed_and_resolve(error_str)
            return False
        except OSError as exc:
            if has_local:
                logger.warning(
                    "Pull failed for %s (OS error); using local copy. error=%s",
                    image,
                    exc,
                )
                for ctx in alert_ctxs:
                    ctx.resolve()
                return True
            logger.error("Failed to pull worker image %s: %s", image, exc)
            for ctx in alert_ctxs:
                ctx.mark_failed_and_resolve(str(exc)[:500])
            return False

    def _send_startup_failure_alert(self, detail: str = "") -> None:
        """Best-effort alert to all linked twins when worker startup fails."""
        if not self._twin_uuids:
            return
        try:
            from .startup import _send_worker_start_failure_alerts  # noqa: PLC0415

            _send_worker_start_failure_alerts(
                twin_uuids=self._twin_uuids,
                error=detail,
            )
        except Exception:
            logger.debug(
                "Failed to send worker-start-failure alert from WorkerManager",
                exc_info=True,
            )

    def _run_container(self) -> bool:
        """Pull (if needed) and run the worker container. Returns True on success."""
        from .startup import DEFAULT_ENVIRONMENT, get_runtime_env_var

        image = self._image
        runtime_environment = (
            get_runtime_env_var("CYBERWAVE_ENVIRONMENT", DEFAULT_ENVIRONMENT) or DEFAULT_ENVIRONMENT
        ).lower()

        if ":" not in image and runtime_environment != "production":
            image = f"{image}:{runtime_environment}"

        if not docker_rm(self._container_name):
            logger.warning(
                "docker rm timed out for %s; will recover on conflict if needed",
                self._container_name,
            )

        env_vars = self._build_env_vars()
        env_args: list[str] = []
        for key, value in env_vars.items():
            env_args += ["-e", f"{key}={value}"]

        volume_args = self._build_volume_args()
        network_args = self._build_network_args()

        gpu_args: list[str] = []
        non_gpu_image: str | None = None
        if docker_has_nvidia_runtime():
            gpu_args = ["--gpus", "all"]
            if image.startswith("cyberwaveos/edge-ml-worker:") and not image.endswith("-gpu"):
                non_gpu_image = image
                image = f"{image}-gpu"
                logger.info("NVIDIA runtime detected; using GPU image %s", image)
            else:
                logger.info("NVIDIA runtime detected; adding --gpus all to worker container")

        hailo_args: list[str] = []
        non_hailo_image: str | None = None
        if _hailo_device_present() and not gpu_args:
            hailo_args = _build_hailo_passthrough_args()
            rewritten = _apply_hailo_image_tag(image)
            if rewritten != image:
                non_hailo_image = image
                image = rewritten
                logger.info(
                    "Hailo accelerator detected at %s; using Hailo image %s",
                    HAILO_DEVICE_PATH,
                    image,
                )
            else:
                logger.info(
                    "Hailo accelerator detected at %s; adding device passthrough to %s",
                    HAILO_DEVICE_PATH,
                    image,
                )

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

        user_args = build_user_args()

        cmd = [
            "docker",
            "run",
            "--detach",
            "--restart",
            "unless-stopped",
            *network_args,
            *user_args,
            "--name",
            self._container_name,
            *gpu_args,
            *hailo_args,
            *resource_args,
            *volume_args,
            *env_args,
            *resource_env_args,
            image,
        ]

        logger.info("Starting worker container %s from image %s", self._container_name, image)

        if not self._pull_worker_image_with_progress(image):
            if non_gpu_image:
                logger.warning("GPU image %s unavailable; falling back to %s", image, non_gpu_image)
                image = non_gpu_image
                cmd[-1] = image
                if not self._pull_worker_image_with_progress(image):
                    self._send_startup_failure_alert(f"image {image} unavailable and no local copy")
                    return False
            elif non_hailo_image:
                # The Hailo image hasn't been published / is unreachable; fall
                # back to the base image. The --device flag stays in place
                # (harmless when hailo_platform isn't installed; the worker
                # will fail loudly at HailoRuntime.is_available() instead of
                # silently swallowing the user's .hef-selecting workflow).
                logger.warning(
                    "Hailo image %s unavailable; falling back to %s",
                    image,
                    non_hailo_image,
                )
                image = non_hailo_image
                cmd[-1] = image
                if not self._pull_worker_image_with_progress(image):
                    self._send_startup_failure_alert(f"image {image} unavailable and no local copy")
                    return False
            else:
                self._send_startup_failure_alert(f"image {image} unavailable and no local copy")
                return False

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or ""
            if "already in use" in stderr or "Conflict" in stderr:
                logger.warning(
                    "Worker container %s name conflict; force-removing and retrying once",
                    self._container_name,
                )
                if not docker_rm(self._container_name):
                    logger.error(
                        "docker rm timed out for %s during conflict recovery; cannot retry",
                        self._container_name,
                    )
                    self._send_startup_failure_alert(
                        "docker rm timed out during conflict recovery"
                    )
                    return False
                # Brief pause so Docker's daemon finishes cleaning up the container
                # record before we retry docker run.  The race window is sub-second;
                # one fixed wait is sufficient and avoids holding the lock for a
                # long polling loop.
                time.sleep(1.0)
                remaining_status = docker_container_status(self._container_name)
                if remaining_status != "none":
                    logger.error(
                        "Worker container %s still exists (status=%s) after rm; aborting retry",
                        self._container_name,
                        remaining_status,
                    )
                    self._send_startup_failure_alert(
                        f"container still exists (status={remaining_status}) after force-remove"
                    )
                    return False
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
                except subprocess.CalledProcessError as retry_exc:
                    logger.error(
                        "Failed to start worker container %s after conflict retry: %s",
                        self._container_name,
                        retry_exc.stderr,
                    )
                    self._send_startup_failure_alert(f"docker run failed: {retry_exc.stderr}")
                    return False
                except subprocess.TimeoutExpired:
                    logger.error(
                        "Docker run timed out for worker container %s (conflict retry)",
                        self._container_name,
                    )
                    self._send_startup_failure_alert("docker run timed out after 60s")
                    return False
            else:
                logger.error(
                    "Failed to start worker container %s: %s", self._container_name, exc.stderr
                )
                self._send_startup_failure_alert(f"docker run failed: {exc.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Docker run timed out for worker container %s", self._container_name)
            self._send_startup_failure_alert("docker run timed out after 60s")
            return False

        probe_seconds = _DEFAULT_STARTUP_PROBE_SECONDS
        probe_override = get_runtime_env_var("CYBERWAVE_WORKER_STARTUP_PROBE_SECONDS")
        if probe_override:
            try:
                probe_seconds = max(1, int(probe_override))
            except ValueError:
                logger.warning(
                    "Invalid CYBERWAVE_WORKER_STARTUP_PROBE_SECONDS=%r; using default %ds",
                    probe_override,
                    _DEFAULT_STARTUP_PROBE_SECONDS,
                )

        last_status = "unknown"
        for i in range(probe_seconds):
            inspect_data = docker_inspect(self._container_name)
            if inspect_data is None:
                last_status = "none"
                time.sleep(1.0)
                continue

            state = inspect_data.get("State")
            if not isinstance(state, dict):
                last_status = "unknown"
                time.sleep(1.0)
                continue

            last_status = str(state.get("Status", "unknown")).lower()

            if last_status == "running":
                logger.info("Worker container %s is running", self._container_name)
                return True

            if last_status in {"exited", "dead"}:
                error_msg = str(state.get("Error", "")).strip() or "none"
                exit_code = state.get("ExitCode", "?")
                restart_count = int(inspect_data.get("RestartCount", 0))
                if restart_count > 0 and last_status == "exited":
                    logger.debug(
                        "Worker container %s exited (code=%s, restarts=%d); "
                        "Docker restart policy may revive it — continuing probe",
                        self._container_name,
                        exit_code,
                        restart_count,
                    )
                    time.sleep(1.0)
                    continue
                logger.error(
                    "Worker container %s failed to start (status=%s, exit_code=%s, error=%s)",
                    self._container_name,
                    last_status,
                    exit_code,
                    error_msg,
                )
                self._send_startup_failure_alert(
                    f"status={last_status}, exit_code={exit_code}, error={error_msg}"
                )
                return False

            time.sleep(1.0)

        # Probe window elapsed without the container reaching "running".
        # The container has --restart=unless-stopped so Docker will keep
        # trying.  Return True (like driver containers) so the rest of
        # edge-core startup is not blocked; the health monitor and periodic
        # reconcile loop will track eventual state.
        logger.warning(
            "Worker container %s did not reach running state within "
            "startup probe window (%ds, last_status=%s). The container "
            "uses --restart=unless-stopped so Docker will keep trying. "
            "Override probe duration with CYBERWAVE_WORKER_STARTUP_PROBE_SECONDS.",
            self._container_name,
            probe_seconds,
            last_status,
        )
        self._send_startup_failure_alert(
            f"startup probe timed out after {probe_seconds}s, last_status={last_status}"
        )
        return True


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
