"""Wait for a driver container's dependencies to become healthy.

`depends_on: {condition: service_healthy}` in the image-declared topology means
a dependent must not start until its dependency's healthcheck passes. Ordering
alone is not enough: a ROS node that comes up before the one it subscribes to
misses discovery and then sits idle looking healthy.

WHY THIS IS ITS OWN MODULE
--------------------------
Two orchestrators launch the same declared graph and both need this wait: Edge
Core on a robot, and cyberwave-sim's `DeclaredTopologyContainers` on a cloud
node. One implementation, because a second copy drifts: without the shared
budget below, a plant that never becomes healthy costs 4x180s on the Go2 graph
instead of 180s.

Inside `cyberwave_edge_core/` rather than beside it because edge-core is mirrored
out by `git subtree split --prefix=cyberwave-edge-core`; a sibling package would
fall off the mirror. No intra-package imports, so cyberwave-sim can import it
through a `sys.path` entry the same way it already imports `driver_selection`.

THE INVARIANT, which is load-bearing
------------------------------------
**The wait may delay a start. It may never prevent one.** The return value is a
report; no caller branches on it. Skipping a service is unrecoverable: it is
never created, so its health snapshot reads `removed`, which
`reconcile_driver_revival` ignores by design and nothing else retries. Docker's
`unhealthy` is not terminal either -- the healthcheck keeps running and a later
pass can flip the container back -- so treating it as fatal reads a transient
state as permanent.

Starting a dependent late, or against a sick dependency, is recoverable: ROS
discovery retries, the restart policy revives a crash, and the restart-loop
reconciler stops a genuinely broken container. Not starting it at all guarantees
the twin never works.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# The Go2 plant's own healthcheck allows ~130s (start_period 10s + 24 retries x
# 5s interval), so the default leaves headroom for a cold ROS graph on a Jetson
# without hanging boot forever.
DEFAULT_TIMEOUT_SECONDS = float(
    os.getenv("CYBERWAVE_DRIVER_DEPENDENCY_HEALTH_TIMEOUT_SECONDS", "180")
)
DEFAULT_POLL_SECONDS = float(os.getenv("CYBERWAVE_DRIVER_DEPENDENCY_HEALTH_POLL_SECONDS", "2"))

# ``(the container exists, its health status)``.
Probe = Callable[[str], "tuple[bool, str]"]


def container_health_status(container_name: str) -> tuple[bool, str]:
    """``(the container exists, its health status)`` for one container.

    ``--format`` rather than a full inspect because this is polled every couple
    of seconds for minutes at a time, and the full payload is 20-30 KB of JSON
    parsed to read one string.

    The template yields ``""`` for a container that declares no healthcheck,
    which is what lets the caller tell that apart from a container that is not
    there (non-zero exit) without parsing anything. A bare
    ``{{.State.Health.Status}}`` cannot: ``Health`` is nil when no healthcheck is
    declared, so the template errors and the two cases become one.
    """
    if not shutil.which("docker"):
        return False, ""
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{if .State.Health}}{{.State.Health.Status}}{{end}}",
                container_name,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False, ""
    return True, (result.stdout or "").strip().lower()


def wait_for_dependency_health(
    container_name: str,
    *,
    probe: Probe = container_health_status,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    watchdog: Optional[Any] = None,
    heartbeat_extend_seconds: float = 30.0,
    log: Optional[logging.Logger] = None,
) -> bool:
    """Wait for *container_name* to become ready. True if it is.

    True means "as ready as it can be": healthy, or nothing to satisfy (no
    healthcheck declared, or no container to inspect). False means the budget
    expired with the dependency still ``starting`` or ``unhealthy``. Either way
    the caller starts the dependent -- see the module docstring.

    **The wait must heartbeat, or it kills the process it runs in.** On a robot
    this is called from the serial launch loop, which runs inline inside the
    runtime loop -- and that loop's only ``watchdog.ping()`` is at the end of the
    cycle. The systemd unit sets ``WatchdogSec=60`` against a ~15s ping cadence,
    and ``start_pinging()`` deliberately spawns no pinger thread: the cadence
    *is* the loop. So a dependency that stays ``starting`` silently held the loop
    for up to ``timeout_seconds`` (default 180s, three times the deadline) with
    nothing pinging, systemd ``SIGABRT``ed edge-core mid-launch, and the restart
    hit the same sick plant again -- a boot loop triggered by exactly the slow
    plant this gate exists for. ``extend_timeout`` covers the boot-time half
    (``TimeoutStartSec``). A cloud workload passes no watchdog: it has no such
    deadline, and this is why the parameter is optional rather than assumed.
    """
    log = log or logger
    deadline = time.monotonic() + max(timeout_seconds, 0.0)

    while True:
        exists, status = probe(container_name)
        if not exists:
            log.warning(
                "Dependency %s not inspectable; starting dependents without waiting",
                container_name,
            )
            return True

        if not status:
            log.debug(
                "Dependency %s declares no healthcheck; treating as started",
                container_name,
            )
            return True

        if status == "healthy":
            return True

        if time.monotonic() >= deadline:
            # `unhealthy` gets an error rather than a warning because it is the
            # one status that will not fix itself by being waited on longer, and
            # it is the likeliest root cause when a whole twin comes up mute.
            emit = log.error if status == "unhealthy" else log.warning
            emit(
                "Dependency %s still %s after %.0fs; starting dependents anyway",
                container_name,
                status,
                timeout_seconds,
            )
            return False

        if watchdog is not None:
            # Never let a watchdog problem be the reason a driver does not
            # start -- same posture as the runtime loop's own ping.
            try:
                watchdog.ping()
                watchdog.extend_timeout(heartbeat_extend_seconds)
                watchdog.notify_status(
                    f"Waiting for {container_name} to become healthy ({status})"
                )
            except Exception:
                log.debug("Watchdog heartbeat failed during health wait", exc_info=True)

        time.sleep(poll_seconds)


class HealthGate:
    """One launch pass's worth of waiting, with the budget shared per dependency.

    Owning the memo here rather than at each call site is the point: both
    orchestrators need it and one of them had already forgotten it.

    Four Go2 services declare ``plant: service_healthy``. Waiting per dependent
    spends the budget once each -- four times over for one twin -- and Edge
    Core's launch loop is serial across every twin on the edge, so one dead robot
    would delay every healthy twin queued behind it. After the first wait the
    answer cannot usefully change within a pass: healthy stays healthy, and a
    dependency that burned the budget will burn it again.

    Construct one per launch pass, not per service, or the memo is a no-op.
    """

    def __init__(
        self,
        *,
        probe: Probe = container_health_status,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        watchdog: Optional[Any] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self._probe = probe
        self._timeout_seconds = timeout_seconds
        self._poll_seconds = poll_seconds
        self._watchdog = watchdog
        self._log = log or logger
        self._resolved: dict[str, bool] = {}

    def wait_once(self, container_name: str) -> bool:
        """Wait for *container_name*, at most once for the life of this gate."""
        if container_name in self._resolved:
            return self._resolved[container_name]
        ready = wait_for_dependency_health(
            container_name,
            probe=self._probe,
            timeout_seconds=self._timeout_seconds,
            poll_seconds=self._poll_seconds,
            watchdog=self._watchdog,
            log=self._log,
        )
        self._resolved[container_name] = ready
        return ready
