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
from typing import TYPE_CHECKING, Any, Mapping, Optional

from ._clock import now_monotonic
from .container_spec import WORKER_SERVICE, DockerRunSpec
from .docker_args import (
    _rewrite_macos_container_base_url,
    _rewrite_macos_container_hostname,
    build_log_args,
)
from .docker_helpers import (
    build_user_args,
    docker_available,
    docker_container_status,
    docker_has_nvidia_runtime,
    docker_image_id,
    docker_inspect,
    docker_rm,
    docker_stop,
    group_gid,
    parse_docker_timestamp_ns,
)
from .driver_selection import (
    MIN_JETSON_L4T_MAJOR,
    is_jetson,
    jetson_accelerator_supported,
    jetson_l4t_major,
)
from .worker_readiness import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    READY_PHASE,
    WORKER_STATUS_FILENAME,
    clear_status,
    default_status_dir,
    evaluate_readiness,
    read_status,
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
#
# ``local`` is absent on purpose: nothing publishes it, so a ``:local``
# image is whatever the last local build produced and a pull can only fail.
# A missing ``:local`` is still pulled, like any other absent tag.
_MUTABLE_TAG_BASENAMES = frozenset(
    {"latest", "dev", "staging", "nightly", "edge", "main", "master"}
)

_DEFAULT_STARTUP_PROBE_SECONDS = 30

#: Consecutive unobservable readiness ticks tolerated *after* the container
#: has been seen once, before it is reported gone. ``docker_inspect`` cannot
#: tell "no such container" from "the daemon would not answer", so a single
#: miss is not evidence of a death.
_READINESS_INSPECT_MISS_LIMIT = 4

#: Re-send an unchanged startup patch at least this often. A phase that
#: publishes once (``loading_models`` runs for as long as the weights take)
#: would otherwise leave the alert untouched until the backend's own
#: staleness window drops it and the run is failed under a working device.
_READINESS_KEEPALIVE_SECONDS = 5 * 60


def _readiness_timeout_seconds() -> float:
    """How long to wait for the worker to report ``ready``. Override with
    ``CYBERWAVE_WORKER_READY_TIMEOUT_SECONDS`` on slow-loading devices."""
    from .startup import get_runtime_env_var

    override = get_runtime_env_var("CYBERWAVE_WORKER_READY_TIMEOUT_SECONDS")
    if override:
        try:
            return float(max(1, int(override)))
        except ValueError:
            logger.warning(
                "Invalid CYBERWAVE_WORKER_READY_TIMEOUT_SECONDS=%r; using default %ds",
                override,
                DEFAULT_READY_TIMEOUT_SECONDS,
            )
    return float(DEFAULT_READY_TIMEOUT_SECONDS)


def _inspect_status(inspect: dict[str, Any] | None) -> str:
    """Container status from a ``docker inspect`` payload, ``"none"`` if absent."""
    if inspect is None:
        return "none"
    state = inspect.get("State")
    if not isinstance(state, dict):
        return "unknown"
    return str(state.get("Status", "unknown")).lower()


def _container_uptime_seconds(inspect: dict[str, Any] | None) -> float | None:
    """Seconds since the container started, from its own ``StartedAt``.

    Docker is the clock, not this process, so the readiness timeout keeps
    running across an edge-core restart. ``None`` when Docker did not say —
    per ``parse_docker_timestamp_ns``, unknown is not zero, and the caller
    decides what to substitute.
    """
    if inspect is None:
        return None
    state = inspect.get("State")
    if not isinstance(state, dict):
        return None
    started_ns = parse_docker_timestamp_ns(str(state.get("StartedAt") or ""))
    if started_ns is None:
        return None
    return max(0.0, time.time() - started_ns / 1_000_000_000)


def _container_restart_count(inspect: dict[str, Any] | None) -> int:
    """Docker's own count of restarts for this container, 0 when unknown."""
    if inspect is None:
        return 0
    value = inspect.get("RestartCount")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


@dataclass
class _StartupCard:
    """The ``worker_starting`` alerts held open for one container, and the
    readiness state describing them.

    Module-level and keyed by container name rather than held per
    ``WorkerManager``, because the object that *opens* a card is routinely not
    the object that closes it: the boot path builds one manager
    (``_start_worker_after_drivers``), the runtime loop's watcher builds
    another and is the only caller of ``poll_startup_readiness``, and the
    edge-restart helper builds a third. A card that belonged to whichever
    instance happened to open it was simply never polled on the ordinary boot
    path, leaving a spinner up until the next edge-core boot.
    """

    ctxs: list[Any] = field(default_factory=list)
    #: Last readiness patch pushed, so a tick that observes no change doesn't
    #: re-send it (alert updates bypass the throttle), and when it went out,
    #: so a phase that never changes still keeps the card alive for the
    #: backend.
    last_patch: dict[str, Any] | None = None
    last_patch_at: float = 0.0
    #: Whether ``docker inspect`` has answered for this startup yet, and how
    #: many ticks running it has not. Until it answers once, "no container" is
    #: the pull still being in flight, not a death.
    container_seen: bool = False
    inspect_misses: int = 0
    #: Container age at our first sighting, and the monotonic reading taken
    #: alongside it. Together they project an elapsed time that a wall-clock
    #: step cannot move — see :meth:`WorkerManager._readiness_elapsed_seconds`.
    uptime_baseline: float = 0.0
    uptime_baseline_at: float | None = None

    def reset_tracking(self) -> None:
        """Forget everything learned about the startup currently being watched,
        keeping the open contexts. Called wherever the card this state
        describes is about to be re-used for a fresh container."""
        self.last_patch = None
        self.last_patch_at = 0.0
        self.container_seen = False
        self.inspect_misses = 0
        self.uptime_baseline = 0.0
        self.uptime_baseline_at = None

    # A card is now reachable from more than one thread — the runtime loop
    # polls it while an MQTT command handler can be starting a container. The
    # two mutations are locked so a context opened by one cannot be dropped by
    # the other's detach. Resolving is deliberately left outside the lock: it
    # is network I/O.
    def add_ctx(self, ctx: Any) -> None:
        """Hold one more alert context open on this card."""
        with _STARTUP_CARD_LOCK:
            self.ctxs.append(ctx)

    def take_ctxs(self) -> list[Any]:
        """Detach every open context, leaving the card empty."""
        with _STARTUP_CARD_LOCK:
            ctxs, self.ctxs = self.ctxs, []
        return ctxs


_STARTUP_CARD_LOCK = threading.Lock()
_STARTUP_CARDS: dict[str, _StartupCard] = {}


def _startup_card(container_name: str) -> _StartupCard:
    """The shared startup card for *container_name*, created on first use."""
    with _STARTUP_CARD_LOCK:
        return _STARTUP_CARDS.setdefault(container_name, _StartupCard())


# Coordinates concurrent worker-image pulls so a second ``WorkerManager.start()``
# waits for an in-flight pull instead of failing immediately with
# ``unavailable and no local copy`` and spamming ``worker_start_failure``.
_WORKER_IMAGE_PULL_LOCK = threading.Lock()
_WORKER_IMAGE_PULL_DONE: dict[str, threading.Event] = {}
_WORKER_IMAGE_PULL_RESULTS: dict[str, bool] = {}


def is_worker_image_pull_in_progress(image: str | None = None) -> bool:
    """Return True when a worker image pull is currently in flight."""
    with _WORKER_IMAGE_PULL_LOCK:
        if image is None:
            return bool(_WORKER_IMAGE_PULL_DONE)
        return image in _WORKER_IMAGE_PULL_DONE


def _begin_worker_image_pull(image: str) -> bool:
    """Register *image* as being pulled. Returns True if this caller owns the pull."""
    with _WORKER_IMAGE_PULL_LOCK:
        if image in _WORKER_IMAGE_PULL_DONE:
            return False
        _WORKER_IMAGE_PULL_DONE[image] = threading.Event()
        return True


def _finish_worker_image_pull(image: str, result: bool) -> None:
    """Publish *result* to any waiters and clear the in-flight slot."""
    with _WORKER_IMAGE_PULL_LOCK:
        event = _WORKER_IMAGE_PULL_DONE.pop(image, None)
        _WORKER_IMAGE_PULL_RESULTS[image] = result
    if event is not None:
        event.set()


def _wait_for_worker_image_pull(image: str, *, timeout: float) -> bool | None:
    """Wait for another caller's pull of *image* to finish.

    Returns the recorded pull result, or ``None`` when no pull was in flight.
    """
    with _WORKER_IMAGE_PULL_LOCK:
        event = _WORKER_IMAGE_PULL_DONE.get(image)
    if event is None:
        return _WORKER_IMAGE_PULL_RESULTS.get(image)
    finished = event.wait(timeout=max(1.0, timeout))
    if not finished:
        logger.warning(
            "Timed out waiting %.0fs for in-progress worker image pull of %s",
            timeout,
            image,
        )
        return None
    return _WORKER_IMAGE_PULL_RESULTS.get(image)


def _image_tag_is_mutable(image: str) -> bool:
    """Return True for image references whose tag rolls forward over time.

    Examples:
        ``cyberwaveos/edge-ml-worker``           -> True (no tag → ``latest``)
        ``cyberwaveos/edge-ml-worker:latest``    -> True
        ``cyberwaveos/edge-ml-worker:dev``       -> True
        ``cyberwaveos/edge-ml-worker:dev-gpu``   -> True
        ``cyberwaveos/edge-ml-worker:local``     -> False (built on the box)
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
    # Strip arch/runtime suffixes such as ``-gpu``, ``-cpu``, ``-arm64``,
    # ``-hailo``, ``-jetson``.
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


def _newest_worker_write_ns(workers_dir: Path) -> int | None:
    """Newest ``st_mtime_ns`` across *workers_dir* and its ``*.py`` files.

    The directory itself is included because ``unlink`` bumps its mtime. That
    is what lets one timestamp cover both changes a sync can make: a worker
    written, and a worker removed — the latter leaving nothing behind to stat.

    Returns None when the directory cannot be read, or when a file disappears
    mid-scan. Both mean "no reliable answer", which callers must treat as
    "assume the container is stale" rather than as a zero timestamp.
    """
    try:
        newest = workers_dir.stat().st_mtime_ns
    except OSError:
        return None
    for path in workers_dir.glob("*.py"):
        try:
            newest = max(newest, path.stat().st_mtime_ns)
        except OSError:
            return None
    return newest


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
    operator overrides (``CYBERWAVE_WORKER_IMAGE``) and the ``-gpu`` /
    ``-jetson`` forks are left untouched — Hailo + NVIDIA on the same
    host is not a supported combination and ``-gpu``/``-jetson`` win
    because they are the higher-priority accelerators for the rest of
    the catalog.
    """
    if not image.startswith(f"{_WORKER_IMAGE_BASE}:"):
        return image
    if image.endswith("-hailo") or image.endswith("-gpu") or image.endswith("-jetson"):
        return image
    return f"{image}-hailo"


def _apply_jetson_image_tag(image: str) -> str:
    """Append ``-jetson`` to the worker image tag on Jetson hosts.

    Mirrors :func:`_apply_hailo_image_tag`: only rewrites
    ``cyberwaveos/edge-ml-worker:<tag>`` references, leaves custom
    ``CYBERWAVE_WORKER_IMAGE`` overrides alone, and refuses to double-suffix
    an already-Jetson tag. The ``-hailo`` fork is also left untouched — a
    single host can't be both a Jetson and a Pi-with-AI-HAT — but this is a
    defensive check rather than a real code path.

    Never combines with ``-gpu``: Jetson is arm64 and the amd64 ``-gpu``
    blob has the wrong ``libcuda.so`` ABI (desktop/server CUDA vs Tegra
    CUDA). :meth:`WorkerManager._run_container` picks Jetson before GPU for
    the same reason.
    """
    if not image.startswith(f"{_WORKER_IMAGE_BASE}:"):
        return image
    if image.endswith("-jetson") or image.endswith("-hailo") or image.endswith("-gpu"):
        return image
    return f"{image}-jetson"


def _strip_jetson_image_tag(image: str) -> str:
    """Inverse of :func:`_apply_jetson_image_tag`, for the L4T-floor demotion.

    Only touches our own worker image. An operator who pinned
    ``cyberwaveos/edge-ml-worker:<tag>-jetson`` on a pre-JetPack-7 board gets
    the plain multi-arch tag, which is the one that actually runs there — so
    the demotion warning names the image the container really starts from.
    """
    if not image.startswith(f"{_WORKER_IMAGE_BASE}:"):
        return image
    if not image.endswith("-jetson"):
        return image
    return image[: -len("-jetson")]


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
       the locally-present image — and a ``:local`` tag skips the doomed
       pull altogether, being immutable.
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

    ``network`` overrides the default host networking with a named Docker
    network.  An edge device wants ``--network host`` — the drivers publishing
    its data run there.  A caller whose producer is itself a container on a
    user-defined bridge (cyberwave-sim's plant) has to put the worker on that
    bridge instead, or the two Zenoh peers never scout each other.

    ``alert_twin_uuids`` narrows who hears about a startup, for a caller whose
    ``twin_uuids`` is deliberately wider than the twins a workflow references
    (cyberwave-sim syncs every twin in the environment so a workflow activated
    mid-run is reachable).  One worker container raises one card per twin it is
    given — the ``worker_starting`` progress card and the failure card alike —
    so the wide set announces a worker starting on twins that have no worker
    and no workflow to lose.  ``None`` defaults to ``twin_uuids``, so an edge
    device is unaffected; an empty list means exactly that, and silences the
    fan-out.
    """

    def __init__(
        self,
        *,
        config_dir: Path,
        environment_uuid: str,
        token: str,
        twin_uuids: Optional[list[str]] = None,
        alert_twin_uuids: Optional[list[str]] = None,
        image: str = DEFAULT_WORKER_IMAGE,
        resource_limits: Optional[ResourceLimits] = None,
        network: Optional[str] = None,
        labels: Optional[Mapping[str, str]] = None,
        shutdown_event: Optional[threading.Event] = None,
    ) -> None:
        self._shutdown_event = shutdown_event
        self._config_dir = config_dir
        self._environment_uuid = environment_uuid
        self._token = token
        self._twin_uuids: list[str] = list(twin_uuids or [])
        # ``None`` defaults to the full set, which is what an edge device wants:
        # every twin bound to the node is one whose workflows this worker runs.
        # An explicitly empty list is a different answer and is honoured as
        # given — "none of these twins has a workflow to lose yet" — which is
        # what cyberwave-sim passes for a simulation started before any workflow
        # was written against it.
        self._alert_twin_uuids: list[str] = list(
            self._twin_uuids if alert_twin_uuids is None else alert_twin_uuids
        )
        self._image = image
        self._resource_limits = resource_limits
        self._network = (network or "").strip() or None
        # Extra container labels, e.g. container_spec.workload_labels().
        self._labels: dict[str, str] = dict(labels or {})
        self._container_name = f"{WORKER_CONTAINER_PREFIX}{environment_uuid[:8]}"
        self._health_monitor: Optional["WorkerHealthMonitor"] = None
        # Resolved once (mkdir/chown) and reused; derived from ``config_dir``.
        self._status_dir_cache: Optional[Path] = None
        self._status_dir_resolved = False

    def _shutdown_requested(self) -> bool:
        return self._shutdown_event is not None and self._shutdown_event.is_set()

    @property
    def _card(self) -> _StartupCard:
        """The ``worker_starting`` card for this manager's container.

        Shared across every ``WorkerManager`` built for the same container:
        the card is a property of the startup, not of whichever instance
        opened it. See :class:`_StartupCard`.
        """
        return _startup_card(self._container_name)

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
        # Nothing is starting up any more, whatever state the container is in:
        # retire the card rather than let it time out into a startup failure
        # for a worker the operator stopped. Above the short-circuit below,
        # because a container that died mid-startup lands in ``exited`` with
        # its card still open — exactly the case that must not be skipped.
        # Both calls are idempotent and cost nothing when no card is open.
        self._reset_readiness_tracking()
        self._close_starting_alerts()
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

    def container_started_after_worker_writes(self, inspect: dict[str, Any] | None) -> bool:
        """Whether the running container already loaded every current worker.

        ``load_workers`` imports the workers directory once, at process start,
        so a container that came up *after* the newest change in that
        directory is already running the current graph. Restarting it then
        buys nothing and discards a boot already in progress.

        Deliberately conservative: every uncertain input returns False, which
        is the unconditional-restart behaviour this check was added to narrow.
        That covers a failed inspect, a missing or unparseable ``StartedAt``,
        an unreadable workers directory, and timestamps that compare equal —
        ext4 stamps from a coarse clock, so two events inside one tick carry
        the same mtime and cannot be assumed ordered.

        It also returns False on the *deactivate* path without having to know
        about it: unlinking a ``wf_*.py`` bumps the directory mtime past
        ``StartedAt``. That restart is load-bearing rather than merely eager,
        because the unlinked module stays imported in the running container
        and would otherwise keep executing the workflow the user just
        deactivated.
        """
        if inspect is None:
            return False
        state = inspect.get("State")
        if not isinstance(state, dict):
            return False
        started_ns = parse_docker_timestamp_ns(str(state.get("StartedAt") or ""))
        if started_ns is None:
            return False
        newest_ns = _newest_worker_write_ns(self._workers_dir())
        if newest_ns is None:
            return False
        return started_ns > newest_ns

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
            # Same path inside the container as on the host, so one value
            # describes both ends.
            "CYBERWAVE_WORKER_STATUS_PATH": str(self._status_path()),
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

        for key in ("ZENOH_CONNECT", "ZENOH_MODE"):
            value = get_runtime_env_var(key)
            if value:
                env[key] = value

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

        args = [
            "-v",
            f"{config_dir}:/app/.cyberwave",
            "-v",
            f"{workers_dir}:/app/workers:ro",
            "-v",
            f"{models_dir}:/app/models",
        ]

        # The worker writes startup milestones here for the readiness
        # watcher. If the directory can't be prepared the worker reports
        # nothing and the watcher falls back to "running means started".
        status_dir = self._status_dir()
        if status_dir is not None:
            args += ["-v", f"{status_dir}:{status_dir}"]
        return args

    def _status_dir(self) -> Path | None:
        """Prepare and return the host dir holding the worker status file.

        Prefers tmpfs, falling back to the config directory when that isn't
        writable — edge-core does not always run as root, and reporting
        nothing is worse than a handful of writes to disk. ``None`` when
        neither works.
        """
        if self._status_dir_resolved:
            return self._status_dir_cache

        candidates = [default_status_dir(self._config_dir), self._config_dir / "run"]
        for candidate in dict.fromkeys(candidates):  # de-dup, order preserved
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                _ensure_dir_writable_by_container_user(candidate)
            except OSError:
                logger.debug("Could not prepare worker status dir %s", candidate, exc_info=True)
                continue
            self._status_dir_cache = candidate
            self._status_dir_resolved = True
            return candidate

        logger.warning(
            "No writable worker status directory; worker startup phases "
            "will not be reported for %s",
            self._container_name,
        )
        # Deliberately not cached: ``/run`` can be unavailable very early in
        # boot, and a transient failure there must not disable phase reporting
        # for the rest of the process.
        return None

    def _status_path(self) -> Path:
        """Full path to the worker status file on the host.

        Resolves through :meth:`_status_dir` so the path handed to the
        container is always the one actually mounted.
        """
        status_dir = self._status_dir() or default_status_dir(self._config_dir)
        return status_dir / WORKER_STATUS_FILENAME

    def _build_network_args(self) -> list[str]:
        """Build network args; mirrors driver container networking.

        An explicit ``network`` wins on every platform: a caller that names one
        knows where its data producer lives, which the platform default cannot.
        """
        if self._network:
            return ["--network", self._network]
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

        For mutable tags (``latest``, ``dev``, ``staging``, ``nightly``,
        ``edge`` and any of those with arch/runtime suffixes like
        ``dev-gpu``) we always issue ``docker pull`` so a developer who
        just pushed a new image gets the new digest on the next worker
        recycle. For immutable tags (e.g. ``v1.2.3``, sha256 digests, and
        the locally built ``:local``) we keep the previous fast-path of
        skipping the pull when the image is already present locally.

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
        from .driver_logs import _run_docker_pull  # noqa: PLC0415

        try:
            _run_docker_pull(image, timeout=timeout)
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

    def _pull_worker_image_with_progress_once(
        self,
        image: str,
        timeout: int = 600,
        *,
        alert_ctxs: list[Any] | None = None,
        is_final_attempt: bool = True,
    ) -> bool:
        """Single attempt to pull *image*. Reuses *alert_ctxs* across retries
        if given; *is_final_attempt* gates critical-escalation vs. just annotating."""
        if not self._alert_twin_uuids:
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

        owns_alert_ctxs = alert_ctxs is None
        if alert_ctxs is None:
            alert_ctxs = []
            for twin_uuid in self._alert_twin_uuids:
                ctx = WorkerStartingAlertContext(
                    twin_uuid=twin_uuid,
                    image=image,
                    container_name=self._container_name,
                )
                ctx.create()
                alert_ctxs.append(ctx)

        def _fail(error_str: str, *, phase: str) -> bool:
            # Only retire a card this call created.  An adopted card belongs to
            # ``_run_container``, which may still fall back to another image —
            # killing it here would blank the rest of the startup.
            if is_final_attempt and owns_alert_ctxs:
                for ctx in alert_ctxs:
                    ctx.mark_failed_and_resolve(error_str, phase=phase)
            else:
                for ctx in alert_ctxs:
                    ctx.update_metadata({"phase": phase, "last_error": error_str[:500]}, force=True)
            return False

        contexts = [
            _PullDeliveryContext(
                twin_uuid=twin_uuid,
                container_name=self._container_name,
                driver_alert_ctx=alert_ctxs[i],
            )
            for i, twin_uuid in enumerate(self._alert_twin_uuids)
        ]

        try:
            _pull_docker_image_with_progress_multi(
                image,
                contexts=contexts,
                token=self._token,
                timeout=timeout,
            )
            logger.info("Successfully pulled %s", image)
            if owns_alert_ctxs:
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
                if owns_alert_ctxs:
                    for ctx in alert_ctxs:
                        ctx.resolve()
                return True
            error_str = (getattr(exc, "stderr", None) or str(exc))[:500]
            logger.error("Failed to pull worker image %s: %s", image, error_str)
            phase = "pull_timeout" if isinstance(exc, subprocess.TimeoutExpired) else "pull_failed"
            return _fail(error_str, phase=phase)
        except OSError as exc:
            if has_local:
                logger.warning(
                    "Pull failed for %s (OS error); using local copy. error=%s",
                    image,
                    exc,
                )
                if owns_alert_ctxs:
                    for ctx in alert_ctxs:
                        ctx.resolve()
                return True
            logger.error("Failed to pull worker image %s: %s", image, exc)
            return _fail(str(exc)[:500], phase="pull_oserror")
        except Exception as exc:
            # Catch-all so an unexpected error doesn't leave the alert active forever.
            logger.exception("Unexpected error pulling worker image %s", image)
            return _fail(str(exc)[:500], phase="pull_unexpected")

    def _pull_worker_image_with_progress(self, image: str, timeout: int = 600) -> bool:
        """Pull *image* with progress reporting to the alert audience.

        Creates a ``worker_starting`` alert on each twin in the alert
        audience for the duration of the pull so the workbench can show a bar
        (byte counts, percent) in analogy with the driver image pull.

        Concurrent callers for the same image wait for the in-flight pull
        instead of racing a second ``docker pull`` and tripping a premature
        ``worker_start_failure`` alert.

        Falls back to the plain subprocess pull
        (:meth:`_ensure_image_pulled`) when the manager has no alert audience
        — nobody is waiting on a run, so there is no bar to draw.  Fallback
        semantics for a failed pull with a local copy present are identical to
        those of :meth:`_ensure_image_pulled`.
        """
        if self._shutdown_requested():
            return False
        if not _begin_worker_image_pull(image):
            waited = _wait_for_worker_image_pull(image, timeout=float(timeout) + 30.0)
            if waited is True or _docker_image_present(image):
                return True
            if waited is False:
                return False
            if not _begin_worker_image_pull(image):
                return _docker_image_present(image)

        max_attempts = max(
            1,
            int(os.getenv("CYBERWAVE_WORKER_IMAGE_PULL_RETRIES", "3")),
        )
        pull_succeeded = False
        # Reused across retries so a later-attempt success doesn't leave a stray critical alert.
        # When ``_run_container`` has already opened the startup alerts, adopt those rather than
        # raising a second set — and leave them open, because the card has to survive past the
        # pull to cover the container start.
        owns_alerts = not self._card.ctxs
        alert_ctxs: list[Any] = list(self._card.ctxs)
        if owns_alerts and self._alert_twin_uuids:
            from .utils import WorkerStartingAlertContext  # noqa: PLC0415

            for twin_uuid in self._alert_twin_uuids:
                ctx = WorkerStartingAlertContext(
                    twin_uuid=twin_uuid,
                    image=image,
                    container_name=self._container_name,
                )
                ctx.create()
                alert_ctxs.append(ctx)
        try:
            for attempt in range(1, max_attempts + 1):
                if self._shutdown_requested():
                    return False
                if attempt > 1:
                    backoff = min(30.0, 2.0 ** (attempt - 1))
                    logger.info(
                        "Retrying worker image pull for %s (attempt %d/%d, backoff=%.1fs)",
                        image,
                        attempt,
                        max_attempts,
                        backoff,
                    )
                    if self._shutdown_event is not None:
                        if self._shutdown_event.wait(backoff):
                            return False
                    else:
                        time.sleep(backoff)
                if self._pull_worker_image_with_progress_once(
                    image,
                    timeout=timeout,
                    alert_ctxs=alert_ctxs or None,
                    is_final_attempt=attempt == max_attempts,
                ):
                    pull_succeeded = True
                    if owns_alerts:
                        for ctx in alert_ctxs:
                            ctx.resolve()
                    return True
            return False
        finally:
            _finish_worker_image_pull(
                image,
                pull_succeeded or _docker_image_present(image),
            )

    def _open_starting_alerts(self, image: str) -> None:
        """Raise one ``worker_starting`` alert per alert-audience twin.

        Opened *before* the pull and held until the worker inside the
        container reports itself ready, so the user is told the worker is
        coming up for the whole download → install → start → load sequence
        rather than only while bytes are moving. A locally-present image skips
        the pull entirely, and that is precisely the case where the startup
        was previously silent.

        Anything still held from an earlier startup is retired first:
        returning early on a non-empty list meant one startup that leaked
        its card (an exception escaping the probe, say) suppressed the card
        for every later startup in the process — a progress bar that works
        once after an edge-core restart and never again. That matters more
        now that a card outlives ``_run_container``, waiting on readiness.
        """
        card = self._card
        if card.ctxs:
            logger.debug(
                "Retiring %d worker_starting alert(s) left open by a previous startup",
                len(card.ctxs),
            )
            self._close_starting_alerts()
        if not self._alert_twin_uuids:
            return
        try:
            from .utils import WorkerStartingAlertContext  # noqa: PLC0415
        except Exception:
            logger.debug("worker_starting alert context unavailable", exc_info=True)
            return
        for twin_uuid in self._alert_twin_uuids:
            try:
                ctx = WorkerStartingAlertContext(
                    twin_uuid=twin_uuid,
                    image=image,
                    container_name=self._container_name,
                )
                ctx.create()
                card.add_ctx(ctx)
            except Exception:
                logger.debug(
                    "Could not open worker_starting alert for twin %s",
                    twin_uuid,
                    exc_info=True,
                )

    @staticmethod
    def _patch_alert_ctxs(ctxs: list[Any], patch: dict[str, Any]) -> None:
        """Push a phase/metadata patch to *ctxs*."""
        for ctx in ctxs:
            try:
                ctx.update_metadata(patch, force=True)
            except Exception:
                logger.debug("Could not update worker_starting alert", exc_info=True)

    @staticmethod
    def _resolve_alert_ctxs(ctxs: list[Any], *, failure: str = "", phase: str = "") -> None:
        """Resolve *ctxs*, optionally marking them failed first."""
        for ctx in ctxs:
            try:
                if failure:
                    ctx.mark_failed_and_resolve(failure, phase=phase or "container_start_failed")
                else:
                    ctx.resolve()
            except Exception:
                logger.debug("Could not close worker_starting alert", exc_info=True)

    def _update_starting_alerts(self, patch: dict[str, Any]) -> None:
        """Push a phase/metadata patch to every alert the card still holds."""
        self._patch_alert_ctxs(self._card.ctxs, patch)

    def _close_starting_alerts(self, *, failure: str = "", phase: str = "") -> None:
        """Resolve the card's alerts, optionally failing them first."""
        self._resolve_alert_ctxs(self._card.take_ctxs(), failure=failure, phase=phase)

    def _reset_readiness_tracking(self) -> None:
        """Forget everything learned about the startup currently being watched.

        Called wherever the card this state describes goes away or is about
        to be replaced, so a new startup never inherits the previous one's
        phase, keepalive deadline, container sightings or uptime baseline.
        """
        self._card.reset_tracking()

    def _readiness_elapsed_seconds(self, inspect: dict[str, Any] | None) -> float:
        """How long this container has been up, in seconds.

        Docker's ``StartedAt`` is the primary source — it is the only one that
        survives an edge-core restart, which is the whole reason the readiness
        budget is measured from it. But it is read against *our* wall clock,
        and a device without an RTC steps that clock the moment NTP lands,
        which on the warm-image path is squarely inside worker startup. A
        forward step of more than the ready timeout would fail a healthy
        worker.

        So the first sighting also records a monotonic baseline, and every
        later reading takes the smaller of the two. A wall-clock step can then
        delay a timeout — the safe direction — but never invent one. When
        Docker declines to answer, the projection is all we have.
        """
        card = self._card
        wall = _container_uptime_seconds(inspect)
        now = now_monotonic()
        if card.uptime_baseline_at is None:
            if wall is None:
                return 0.0
            card.uptime_baseline = wall
            card.uptime_baseline_at = now
            return wall
        projected = card.uptime_baseline + max(0.0, now - card.uptime_baseline_at)
        if wall is None:
            return projected
        return min(wall, projected)

    def poll_startup_readiness(self) -> None:
        """Advance the open startup alert one step. Called once per reconcile tick.

        No-ops unless a startup card is open, so the cost on a steady-state
        edge is one attribute check. While one *is* open this reads the
        worker's status file and either patches the card with the phase the
        worker reports, or closes it — on ``ready``, on the container dying,
        or on the readiness timeout.
        """
        card = self._card
        if not card.ctxs:
            return

        inspect = docker_inspect(self._container_name)
        if inspect is None:
            if not self._record_inspect_miss():
                return
        else:
            card.container_seen = True
            card.inspect_misses = 0

        outcome = evaluate_readiness(
            read_status(self._status_path()),
            container_status=_inspect_status(inspect),
            elapsed_seconds=self._readiness_elapsed_seconds(inspect),
            ready_timeout_seconds=_readiness_timeout_seconds(),
            # ``--restart unless-stopped`` rewrites ``StartedAt`` on every
            # restart, so a crash loop resets the elapsed clock and the
            # timeout can never fire. The restart itself is the evidence.
            restart_count=_container_restart_count(inspect),
        )

        now = now_monotonic()
        # Re-send an unchanged patch on the keepalive interval, not never:
        # the backend drops a ``worker_starting`` alert that has gone too
        # long without a write, and ``loading_models`` publishes exactly
        # once however long the weights take. The comparison still holds
        # the steady-state cost to one call per interval, not one per tick.
        if outcome.patch and (
            outcome.patch != card.last_patch
            or (now - card.last_patch_at) >= _READINESS_KEEPALIVE_SECONDS
        ):
            card.last_patch = outcome.patch
            card.last_patch_at = now
            logger.info(
                "Worker container %s startup phase: %s",
                self._container_name,
                outcome.patch.get("phase"),
            )
            self._update_starting_alerts(outcome.patch)

        if outcome.close:
            if outcome.failure:
                logger.error(
                    "Worker container %s startup failed: %s",
                    self._container_name,
                    outcome.failure,
                )
            else:
                logger.info("Worker container %s is ready", self._container_name)
            self._reset_readiness_tracking()
            self._close_starting_alerts(failure=outcome.failure, phase=outcome.phase)

    def _record_inspect_miss(self) -> bool:
        """Count one unanswered ``docker inspect``; is the container now gone?

        ``docker_inspect`` returns ``None`` for a container that does not
        exist *and* for a daemon that would not answer — a transient error,
        or its own 10 s timeout on a device busy pulling a multi-GB image.
        The card is opened before the pull, so "no container" is the normal
        state for minutes; failing the startup on it would report a crash on
        a device that is working. ``_run_container``'s own probe loop
        re-samples rather than concluding, and this does the same: the
        container counts as gone only once it has been seen and has then
        stayed unobservable for :data:`_READINESS_INSPECT_MISS_LIMIT` ticks.
        """
        card = self._card
        card.inspect_misses += 1
        if not card.container_seen:
            return False
        return card.inspect_misses >= _READINESS_INSPECT_MISS_LIMIT

    def is_worker_ready(self) -> bool:
        """Whether the worker process has reported itself ready.

        Wired as the ``WorkerHealthMonitor`` readiness probe, and surfaced by
        ``cyberwave worker status``. Absence of a record reads as ready *once
        the startup card is closed*: an older worker image never writes one,
        and before this probe existed the monitor treated every running
        container as ready.

        While a card is open the same absence means the opposite — the status
        file was cleared for this startup and the worker has not reached its
        first milestone. Reporting ``ready`` there labelled the gap between
        ``clear_status()`` and ``booting`` as a usable worker. The card is
        itself what resolves the ambiguity: ``evaluate_readiness`` closes it
        after the grace window when no record ever arrives, which is precisely
        the old-image case.
        """
        record = read_status(self._status_path())
        if record is None:
            return not self._card.ctxs
        return record.get("phase") == READY_PHASE

    def _send_startup_failure_alert(
        self, detail: str = "", *, phase: str = "container_start_failed"
    ) -> None:
        """Best-effort alert to all linked twins when worker startup fails.

        *phase* keys the plain-language copy in ``_WORKER_FAILED_DESCRIPTIONS``,
        so callers pass the phase that actually failed — a pull that never
        produced an image should not read "could not start your worker".
        """
        # Always retire the in-progress card first: whatever the suppression
        # rules below decide about the failure alert, leaving a spinner up
        # forever is never the right answer.
        self._close_starting_alerts(
            failure=detail or "worker startup failed",
            phase=phase,
        )
        if not self._alert_twin_uuids:
            return
        if is_worker_image_pull_in_progress():
            logger.info(
                "Suppressing worker_start_failure alert while worker image pull is in progress: %s",
                detail or "(no detail)",
            )
            return
        try:
            from .startup import _send_worker_start_failure_alerts  # noqa: PLC0415

            _send_worker_start_failure_alerts(
                twin_uuids=self._alert_twin_uuids,
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

        # A restart reuses the card the first startup opened
        # (``_open_starting_alerts`` no-ops while one is live), so without
        # this the sighting of the *previous* container survives the
        # ``docker rm`` and the re-pull below — and a poll landing in that
        # window would count missed inspects against a container this method
        # has already removed. Everything from here to ``docker run`` is time
        # in which there is legitimately nothing to inspect.
        self._reset_readiness_tracking()

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

        # Accelerator precedence: Jetson > discrete NVIDIA > Hailo > CPU.
        #
        # Jetson is checked first because on a Jetson host both ``is_jetson()``
        # and ``docker_has_nvidia_runtime()`` return true — the L4T install
        # ships the nvidia-container-runtime — but the amd64 ``-gpu`` blob
        # does not run on Tegra (different libcuda ABI, wrong PTX arch), so
        # we must pick the ``-jetson`` sibling here — but only on a JetPack 7
        # host; older L4T falls through to the multi-arch CPU image.
        #
        # On Jetson JP7 (L4T r38+, CUDA 13.x), the nvidia-container-runtime uses
        # CSV mode to inject GPU device files and the libcuda.so.1 driver.  The
        # injection is triggered by NVIDIA_VISIBLE_DEVICES=all; ``--gpus all`` in
        # Docker CLI sets that env var, but passing it explicitly via ``-e`` is
        # more reliable across different runtime and Docker version combinations.
        # NVIDIA_DRIVER_CAPABILITIES=all is also required so the CUDA runtime
        # libraries bundled in the image can communicate with the host driver.
        gpu_args: list[str] = []
        non_gpu_image: str | None = None
        if is_jetson():
            # The floor speaks for OUR ``-jetson`` image only. A
            # ``CYBERWAVE_WORKER_IMAGE`` pinned elsewhere is unknowable from
            # here and had the runtime before the floor existed — keep it.
            is_our_worker_image = image.startswith(f"{_WORKER_IMAGE_BASE}:")
            if jetson_accelerator_supported() or not is_our_worker_image:
                gpu_args = [
                    "--runtime",
                    "nvidia",
                    "-e",
                    "NVIDIA_VISIBLE_DEVICES=all",
                    "-e",
                    "NVIDIA_DRIVER_CAPABILITIES=all",
                ]
                rewritten = _apply_jetson_image_tag(image)
                if rewritten != image:
                    non_gpu_image = image
                    image = rewritten
                    logger.info("Jetson host detected; using Jetson image %s", image)
                else:
                    logger.info(
                        "Jetson host detected; adding NVIDIA runtime env to worker container %s",
                        image,
                    )
            else:
                # Pre-JetPack-7 driver: the ``-jetson`` wheels cannot load
                # against it and every matmul dies in cuBLAS (CYB-3841). The
                # plain tag is multi-arch, so arm64 CPU inference works.
                # Demote a pinned ``-jetson`` tag too, so the warning names
                # the image the container actually starts from.
                image = _strip_jetson_image_tag(image)
                logger.warning(
                    "Jetson host detected but L4T r%s is older than r%d (JetPack 7); "
                    "the -jetson worker image needs a newer CUDA driver than this "
                    "host provides. Starting %s on CPU instead. Set "
                    "CYBERWAVE_JETSON_ALLOW_UNSUPPORTED_L4T=1 to use the GPU anyway.",
                    jetson_l4t_major(),
                    MIN_JETSON_L4T_MAJOR,
                    image,
                )
        elif docker_has_nvidia_runtime():
            gpu_args = ["--gpus", "all"]
            if (
                image.startswith("cyberwaveos/edge-ml-worker:")
                and not image.endswith("-gpu")
                and not image.endswith("-jetson")
            ):
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

        # Everything but the image, which the pull fallbacks below may still
        # change. The argv is rendered from a DockerRunSpec once the image is
        # final, so the spec-hash label covers exactly what ``docker run`` gets.
        run_options = (
            "--detach",
            "--restart",
            "unless-stopped",
            *build_log_args(),
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
        )

        logger.info("Starting worker container %s from image %s", self._container_name, image)

        self._open_starting_alerts(image)

        if not self._pull_worker_image_with_progress(image):
            if self._shutdown_requested():
                self._close_starting_alerts()
                return False
            if non_gpu_image:
                # Covers both the ``-gpu`` (discrete NVIDIA) and ``-jetson``
                # (Tegra) rewrites: same fallback shape — drop the accelerator
                # suffix and re-pull the base tag so the worker still comes
                # up, albeit on CPU.
                logger.warning(
                    "Accelerator image %s unavailable; falling back to %s",
                    image,
                    non_gpu_image,
                )
                image = non_gpu_image
                self._update_starting_alerts({"image": image, "phase": "pull_started"})
                if not self._pull_worker_image_with_progress(image):
                    self._send_startup_failure_alert(
                        f"image {image} unavailable and no local copy", phase="pull_failed"
                    )
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
                self._update_starting_alerts({"image": image, "phase": "pull_started"})
                if not self._pull_worker_image_with_progress(image):
                    self._send_startup_failure_alert(
                        f"image {image} unavailable and no local copy", phase="pull_failed"
                    )
                    return False
            else:
                self._send_startup_failure_alert(
                    f"image {image} unavailable and no local copy", phase="pull_failed"
                )
                return False

        if self._shutdown_requested():
            self._close_starting_alerts()
            return False

        self._update_starting_alerts({"phase": "starting_container", "image": image})

        cmd = DockerRunSpec(
            options=run_options, image=image, image_id=docker_image_id(image)
        ).run_argv(service=WORKER_SERVICE, extra_labels=self._labels)

        # A leftover ``ready`` from the previous container would resolve the
        # alert on the next readiness poll. Reset again here because the pull
        # above can have taken minutes, during which a poll may have recorded
        # a phase from the container this run is replacing.
        self._reset_readiness_tracking()
        clear_status(self._status_path())

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
                    self._send_startup_failure_alert("docker rm timed out during conflict recovery")
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
                # Not the end of startup: the process inside still has to
                # load and warm up its models. Resolving here — which this
                # code used to do — told the pending-reason copy and the
                # stuck-run sweeper that the device had finished starting
                # while its slowest phase had not begun. The card stays open
                # and ``poll_startup_readiness`` closes it on a later tick.
                self._update_starting_alerts({"phase": "container_running"})
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

    for key in ("ZENOH_CONNECT", "ZENOH_MODE"):
        value = get_runtime_env_var(key)
        if value:
            env[key] = value

    return env
