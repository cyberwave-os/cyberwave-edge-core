"""Driver container log streaming and MQTT forwarding.

Extracted from startup.py — handles the pipeline that follows ``docker logs``
for each driver container, forwards lines to the service logger (stderr), and
publishes them to the MQTT ``driverlog`` topic.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, Optional, Union

from .utils import DriverStartingAlertContext, WorkerStartingAlertContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level state (moved from startup.py)
# ---------------------------------------------------------------------------

# Track active log streaming threads per container to avoid duplicates.
_CONTAINER_LOG_THREADS: dict[str, threading.Thread] = {}

# Track when log streaming last ended per container so reattach uses --since.
# Process-lifetime only, so a restart has no anchor to resume from.
_CONTAINER_LOG_LAST_SEEN: dict[str, str] = {}

# Backlog to replay when there is no --since anchor. Unbounded, `docker logs -f`
# streams the container's entire history, and every line is both printed to
# stderr and published to MQTT as a TwinDriverLog row. 0 restores that.
DRIVER_LOG_COLD_START_TAIL_ENV_VAR = "CYBERWAVE_DRIVER_LOG_COLD_START_TAIL"
DEFAULT_DRIVER_LOG_COLD_START_TAIL = 500


def _cold_start_tail_lines() -> int:
    """Backlog line count for a follow with no ``--since`` anchor."""
    raw = (os.getenv(DRIVER_LOG_COLD_START_TAIL_ENV_VAR) or "").strip()
    if not raw:
        return DEFAULT_DRIVER_LOG_COLD_START_TAIL
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_DRIVER_LOG_COLD_START_TAIL
    return max(0, value)


# Containers for which a driver_runtime_error alert has already been sent
# during the current log-follow session, to avoid alert spam.
_CONTAINER_RUNTIME_ERROR_ALERTED: set[str] = set()

# Pattern that catches Python RuntimeError tracebacks in driver logs.
_RUNTIME_ERROR_PATTERN = re.compile(r"RuntimeError:\s*(.+)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconcile_driver_log_streams() -> int:
    """Ensure active driver containers have an attached log-forwarding thread."""
    from .startup import _CONTAINER_TWIN_MAP, _list_running_driver_containers, load_token

    running_containers = _list_running_driver_containers()
    running_set = set(running_containers)

    stale = [
        name
        for name, thread in _CONTAINER_LOG_THREADS.items()
        if not thread.is_alive() and name not in running_set
    ]
    for name in stale:
        _CONTAINER_LOG_THREADS.pop(name, None)

    attached = 0
    token: Optional[str] = None
    for container_name in running_containers:
        thread = _CONTAINER_LOG_THREADS.get(container_name)
        if thread and thread.is_alive():
            attached += 1
            continue
        if token is None:
            token = load_token()
        twin_uuid = _CONTAINER_TWIN_MAP.get(container_name)
        _stream_container_logs(container_name, twin_uuid=twin_uuid, token=token)
        thread = _CONTAINER_LOG_THREADS.get(container_name)
        if thread and thread.is_alive():
            attached += 1
    return attached


def _stream_container_logs(
    container_name: str,
    *,
    twin_uuid: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    """Stream container logs into this service logger in the background."""
    existing = _CONTAINER_LOG_THREADS.get(container_name)
    if existing and existing.is_alive():
        return

    thread = threading.Thread(
        target=_follow_container_logs,
        args=(container_name,),
        kwargs={"twin_uuid": twin_uuid, "token": token},
        name=f"docker-logs-{container_name}",
        daemon=True,
    )
    _CONTAINER_LOG_THREADS[container_name] = thread
    thread.start()


# ---------------------------------------------------------------------------
# MQTT publish helpers
# ---------------------------------------------------------------------------


def _parse_log_level(message: str) -> str:
    """Best-effort extraction of log level from a driver log line."""
    upper = message[:80].upper()
    for level in ("ERROR", "CRITICAL", "WARNING", "WARN", "DEBUG", "INFO"):
        if level in upper:
            return "WARNING" if level == "WARN" else level
    return "INFO"


def _build_driver_log_payload(
    message: str,
    container_name: str,
    *,
    driver_image: str | None = None,
) -> dict[str, Any]:
    """Build the MQTT payload for a forwarded driver log line."""
    from .startup import CYBERWAVE_SDK_VERSION, EDGE_CORE_VERSION

    payload: dict[str, Any] = {
        "type": "driver_log",
        "message": message,
        "level": _parse_log_level(message),
        "container_name": container_name,
        "source": "edge",
        "timestamp": time.time(),
        "edge_core_version": EDGE_CORE_VERSION,
    }
    if CYBERWAVE_SDK_VERSION:
        payload["sdk_version"] = CYBERWAVE_SDK_VERSION
    if driver_image:
        payload["driver_image"] = driver_image
    return payload


def _resolve_driver_log_publish_context(
    *,
    twin_uuid: Optional[str],
    token: Optional[str],
) -> tuple[Optional[Any], Optional[str]]:
    """Create the MQTT publish context used for driver log forwarding."""
    from .startup import _get_shared_mqtt_client

    if not twin_uuid or not token:
        return None, None

    mqtt_client = _get_shared_mqtt_client(token)
    if not mqtt_client:
        return None, None

    prefix = mqtt_client.mqtt.topic_prefix
    mqtt_topic = f"{prefix}cyberwave/twin/{twin_uuid}/driverlog"
    return mqtt_client, mqtt_topic


def _publish_driver_log_message(
    message: str,
    container_name: str,
    *,
    mqtt_client: Optional[Any] = None,
    mqtt_topic: Optional[str] = None,
    driver_image: str | None = None,
) -> None:
    """Publish a driver log line to MQTT when a publish context is available."""
    if not mqtt_client or not mqtt_topic:
        return

    try:
        mqtt_client.mqtt.publish(
            mqtt_topic,
            _build_driver_log_payload(
                message,
                container_name,
                driver_image=driver_image,
            ),
        )
    except Exception:
        logger.debug(
            "Failed to publish driver log to MQTT for %s",
            container_name,
            exc_info=True,
        )


def _log_and_publish_driver_message(
    message: str,
    container_name: str,
    *,
    mqtt_client: Optional[Any] = None,
    mqtt_topic: Optional[str] = None,
    driver_image: str | None = None,
) -> None:
    """Mirror a driver log line to stderr and publish via MQTT.

    The container already emits fully formatted log lines (timestamp, level,
    module).  Writing them directly to stderr avoids the double-timestamp
    problem that occurs when ``logger.info()`` wraps the line with
    edge-core's own formatter.
    """
    print(message, file=sys.stderr)
    _publish_driver_log_message(
        message,
        container_name,
        mqtt_client=mqtt_client,
        mqtt_topic=mqtt_topic,
        driver_image=driver_image,
    )


# ---------------------------------------------------------------------------
# Network interface facts -> driver_log fan-out
# ---------------------------------------------------------------------------

# Last-published (name, ipv4_address) signature *per twin*, so the periodic
# host-facts keepalive doesn't republish the same interface set every 30 s --
# only when it actually changes (e.g. the device joins a different WiFi
# network).  Keyed by twin UUID rather than held as one process-wide value
# because the set of bound twins grows over an edge's lifetime: a twin that
# gets bound after the first publish has never seen the line, so "already
# published" has to be a per-twin fact.  An absent key means "never published
# to this twin", which also covers the edge that had no bound twins at all on
# its first tick.
#
# Process-lifetime only: an edge-core restart re-publishes one baseline line
# per twin even when nothing changed, which is an acceptable amount of noise.
_LAST_NETWORK_INTERFACES_SIGNATURES: dict[str, tuple[tuple[str, str], ...]] = {}

# Guards the dict above -- the host-facts keepalive hands the fan-out to a
# worker thread (see :func:`publish_network_facts_log_async`), so reads and
# writes are no longer confined to a single thread.
_NETWORK_FACTS_STATE_LOCK = threading.Lock()

# Held for the duration of one fan-out attempt so a slow MQTT connect can't
# pile up workers: the keepalive ticks every 30 s, while a connect against an
# unreachable broker can burn ~50 s (3 attempts x CONNACK timeout + backoff).
# Acquired non-blocking -- if an attempt is already in flight this tick is
# simply skipped, and the next one retries because nothing was recorded.
_NETWORK_FACTS_PUBLISH_LOCK = threading.Lock()


def _addressed_network_interfaces(
    network_interfaces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only interfaces that actually have an IPv4 address bound.

    An interface with no address is nothing anyone can SSH to, so it has no
    place in the log line -- and, more importantly, no place in the dedup
    signature: address-less interfaces come and go (``docker0``/``veth*`` as
    driver containers cycle, ``usb0``/``rndis0`` as cables are plugged) and
    would otherwise re-fire the whole fan-out on churn that tells an operator
    nothing.  The Edge details row in the frontend filters on the same
    condition; the Edge Devices meta line instead renders only the interface
    named by ``host_facts.primary_interface_name`` and keeps the rest on
    hover.  All three are about the same thing -- don't put unreachable
    interfaces in front of an operator -- reached by different means.
    """
    return [nic for nic in network_interfaces if nic.get("ipv4_address")]


def _network_interfaces_signature(
    network_interfaces: list[dict[str, Any]],
) -> tuple[tuple[str, str], ...]:
    """Order-independent identity of an interface set, for change detection."""
    return tuple(
        sorted(
            (str(nic.get("name") or ""), str(nic.get("ipv4_address"))) for nic in network_interfaces
        )
    )


def _format_network_interfaces_message(network_interfaces: list[dict[str, Any]]) -> str:
    if not network_interfaces:
        return "Network interfaces: none detected"
    parts = [
        f"{nic.get('name')}={nic.get('ipv4_address') or 'no IPv4'} "
        f"({'up' if nic.get('is_up') else 'down'})"
        for nic in network_interfaces
    ]
    return "Network interfaces: " + ", ".join(parts)


def publish_network_facts_log(
    twin_uuids: Iterable[str],
    network_interfaces: list[dict[str, Any]],
    *,
    token: Optional[str],
) -> None:
    """Publish a ``driver_log`` line to every bound twin when interfaces change.

    Surfaces the device's current LAN-facing IP(s) in the twin's Logs tab so
    a support engineer can find/SSH to a device whose IP changed (e.g. after
    reconnecting to a different WiFi network) without scanning the network.
    Deduplicated per twin via :data:`_LAST_NETWORK_INTERFACES_SIGNATURES` so
    this fires on an actual change, not on every host-facts keepalive tick.

    A twin's signature is recorded only after its message has actually been
    handed to a connected broker.  Recording up-front would mean a broker
    that happens to be unreachable on the tick where the IP changed loses
    that line *permanently* -- the publish path swallows failures, so the
    next tick would see an unchanged signature and skip.  Because nothing is
    recorded on failure, every subsequent tick retries until it lands.

    Prefer :func:`publish_network_facts_log_async` from anything on a
    latency-sensitive thread: this function can block for as long as an MQTT
    connect takes.
    """
    twin_uuid_list = list(twin_uuids)
    if not twin_uuid_list:
        return

    addressed = _addressed_network_interfaces(network_interfaces)
    signature = _network_interfaces_signature(addressed)

    with _NETWORK_FACTS_STATE_LOCK:
        pending = [
            twin_uuid
            for twin_uuid in twin_uuid_list
            if _LAST_NETWORK_INTERFACES_SIGNATURES.get(twin_uuid) != signature
        ]
    if not pending:
        return

    message = _format_network_interfaces_message(addressed)
    # One stderr line regardless of fan-out width: the service log doesn't
    # need the same sentence once per bound twin.
    print(message, file=sys.stderr)

    for twin_uuid in pending:
        mqtt_client, mqtt_topic = _resolve_driver_log_publish_context(
            twin_uuid=twin_uuid,
            token=token,
        )
        if not mqtt_client or not mqtt_topic:
            # No usable broker right now -- leave this twin unrecorded so the
            # next keepalive tick tries again.
            continue
        # ``MQTTClient.publish`` drops the message with only a warning when the
        # socket has gone away, so a disconnected client must not count as a
        # publish either.  Default True keeps this tolerant of client objects
        # that don't expose the flag.
        if not getattr(mqtt_client.mqtt, "connected", True):
            continue

        _publish_driver_log_message(
            message,
            "network",
            mqtt_client=mqtt_client,
            mqtt_topic=mqtt_topic,
        )
        with _NETWORK_FACTS_STATE_LOCK:
            _LAST_NETWORK_INTERFACES_SIGNATURES[twin_uuid] = signature


def publish_network_facts_log_async(
    twin_uuids: Iterable[str],
    network_interfaces: list[dict[str, Any]],
    *,
    token: Optional[str],
) -> Optional[threading.Thread]:
    """Run :func:`publish_network_facts_log` on a short-lived daemon thread.

    The caller is the host-facts REST keepalive, whose cadence is what powers
    the Online/Idle/Offline pill (the backend's standby window is only 3x the
    30 s period).  Establishing an MQTT connection inline would let an
    unreachable broker stall that thread for ~50 s and eat the entire margin
    on a liveness signal that is supposed to be independent of MQTT -- so the
    fan-out gets its own thread and the keepalive never waits on it.

    Returns the worker thread, or ``None`` when the tick was skipped because
    an attempt is already in flight.
    """
    if not _NETWORK_FACTS_PUBLISH_LOCK.acquire(blocking=False):
        logger.debug("Network facts log publish already in flight; skipping tick")
        return None

    twin_uuid_list = list(twin_uuids)
    interfaces = list(network_interfaces)

    def _run() -> None:
        try:
            publish_network_facts_log(twin_uuid_list, interfaces, token=token)
        except Exception:
            logger.debug("Network facts log publish failed", exc_info=True)
        finally:
            _NETWORK_FACTS_PUBLISH_LOCK.release()

    thread = threading.Thread(
        target=_run,
        name="cyberwave-edge-core.network-facts-log",
        daemon=True,
    )
    thread.start()
    return thread


def _send_runtime_error_alert(
    twin_uuid: str,
    container_name: str,
    error_message: str,
) -> None:
    """Send a ``driver_runtime_error`` alert when a driver logs a RuntimeError.

    Best-effort: failures are logged and never block log forwarding. Each
    container only triggers one alert per log-follow session to prevent
    spamming the twin's alert feed with repeated messages from restart
    loops.
    """
    if container_name in _CONTAINER_RUNTIME_ERROR_ALERTED:
        return
    _CONTAINER_RUNTIME_ERROR_ALERTED.add(container_name)

    try:
        from cyberwave._error_metadata import redact

        from .startup import _send_alert_for_twin

        _send_alert_for_twin(
            twin_uuid,
            "Driver runtime error",
            "Your robot's driver ran into a problem and may not work correctly.",
            "driver_runtime_error",
            severity="error",
            metadata={
                "error_code": "RuntimeError",
                "technical_detail": f"RuntimeError: {redact(error_message)[:500]}",
            },
        )
        logger.info(
            "Sent driver_runtime_error alert for container %s (twin %s)",
            container_name,
            twin_uuid[:8],
        )
    except Exception as exc:
        logger.debug(
            "Could not send driver_runtime_error alert for %s: %s",
            container_name,
            exc,
        )


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------


def _iter_stream_messages(stream: Any) -> Any:
    """Yield messages delimited by newlines or carriage returns."""
    buffer: list[str] = []
    while True:
        chunk = stream.read(1)
        if chunk == "":
            break
        if chunk in {"\r", "\n"}:
            message = "".join(buffer).strip()
            if message:
                yield message
            buffer.clear()
            continue
        buffer.append(chunk)

    message = "".join(buffer).strip()
    if message:
        yield message


# ---------------------------------------------------------------------------
# Docker pull progress
# ---------------------------------------------------------------------------

# ``Alert.metadata['phase']`` values for ``driver_starting``. Failure
# phases (``pull_spawn_failed`` / ``pull_timed_out`` / ``pull_exit_error``)
# are written inline at their ``raise`` sites and not constants here.
_PULL_PHASE_STARTED = "pull_started"
_PULL_PHASE_DOWNLOADING = "downloading"
_PULL_PHASE_INSTALLING = "installing"
_PULL_PHASE_COMPLETE = "pull_complete"

#: A pull only moves forward. Docker interleaves layers, so the raw states
#: read "nothing is downloading" (all in-flight layers verifying) moments
#: before the next batch starts — without this the phase, and the stage the
#: card names, flip back and forth mid-pull.
_PULL_PHASE_RANK: dict[str, int] = {
    _PULL_PHASE_STARTED: 0,
    _PULL_PHASE_DOWNLOADING: 1,
    _PULL_PHASE_INSTALLING: 2,
    _PULL_PHASE_COMPLETE: 3,
}

#: Ceiling before the pull is confirmed complete, so neither rounding nor a
#: fully-cached prefix can show "100 %" early.
_INCOMPLETE_PERCENT_CEILING = 99

#: Layer states in which the layer's bytes are all on disk.
_LAYER_FETCHED_STATES: frozenset[str] = frozenset(
    {"downloaded", "verifying", "extracting", "complete"}
)


class _EngineAPIUnavailableError(RuntimeError):
    """SDK missing or daemon unreachable; dispatcher falls back to subprocess."""


# ``docker pull`` prints SI-scaled byte counts (1 kB = 1000 B); both
# ``kB`` and ``KB`` show up depending on docker version.
_BYTE_UNITS: dict[str, int] = {
    "B": 1,
    "kB": 10**3,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
}

# Layer line shape: ``<id>: <rest>``.  ``id`` is usually a 12+ hex sha
# but we accept any non-whitespace prefix because the first real line
# every pull emits is ``<tag>: Pulling from <repo>`` — and unit-test
# fixtures use friendly aliases like ``layer-1``.  Top-level lines
# (the ``<tag>: …`` intro and ``Digest:``) are filtered out by
# requiring ``rest`` to match a known per-layer keyword below.
_LAYER_LINE_RE = re.compile(r"^([^\s:]+):\s+(.*)$")

# ``CURR UNIT / TOTAL UNIT`` slice of a docker progress bar.
_BYTES_RE = re.compile(r"([\d.]+)\s*([kKMGT]?B)\s*/\s*([\d.]+)\s*([kKMGT]?B)")

# Per-layer keywords, longest-prefix first so ``download complete`` beats
# ``downloading`` in matching.  Lines whose ``rest`` does not start with
# any of these are top-level / unknown and never register a phantom layer.
_LAYER_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("download complete", "downloaded"),
    ("downloading", "downloading"),
    ("extracting", "extracting"),
    ("pull complete", "complete"),
    ("already exists", "complete"),
    ("verifying checksum", "verifying"),
    ("pulling fs layer", "pending"),
    ("waiting", "waiting"),
)

# Drives ``layers_active`` for the subprocess path: docker CLI suppresses
# byte progress on a non-TTY pipe, so the formatter would otherwise plateau
# on ``starting`` until the very end of a multi-GB pull.
_LAYER_ACTIVE_STATES: frozenset[str] = frozenset(
    {"downloading", "verifying", "downloaded", "extracting", "complete"}
)


def _format_bytes(n: int) -> str:
    """Render a byte count as ``"745 MB"`` / ``"1.55 GB"``."""
    if n < 1000:
        return f"{n} B"
    for unit, divisor in (("kB", 10**3), ("MB", 10**6), ("GB", 10**9), ("TB", 10**12)):
        if abs(n) < divisor * 1000:
            scaled = n / divisor
            if scaled >= 100:
                return f"{scaled:.0f} {unit}"
            if scaled >= 10:
                return f"{scaled:.1f} {unit}"
            return f"{scaled:.2f} {unit}"
    return f"{n / 10**12:.2f} TB"


@dataclass
class _DockerPullProgress:
    """Byte-aggregated progress snapshot for one ``docker pull``."""

    image: str
    downloaded_bytes: int = 0
    total_bytes: int = 0
    layers_total: int = 0
    layers_complete: int = 0
    layers_active: int = 0
    phase: str = _PULL_PHASE_STARTED
    last_line: str = ""
    #: Layers fetched so far, the one in flight counted as its byte
    #: fraction. Drives the percentage: a byte ratio can't, because docker
    #: only declares a layer's size when it starts fetching it, so the
    #: denominator is unknown for most of the pull (CYB-3854).
    layers_fetched: float = 0.0
    #: True while some layer has not declared its size, i.e. ``total_bytes``
    #: is a partial sum rather than the image's size.
    total_bytes_partial: bool = False

    def percent(self) -> int:
        """Fetched layers over announced layers, 0–99 until docker says done.

        Monotone for a fixed layer set — states only advance and byte counts
        only grow — so no ratchet is needed on top. The one thing that can
        lower it is docker announcing more layers, which genuinely means
        there is more to do than the last reading implied; pinning the old,
        higher number there would be a lie rather than a smoothing.
        """
        if self.phase == _PULL_PHASE_COMPLETE:
            return 100
        if self.layers_total <= 0:
            return 0
        # Short of the stream's terminator the layer list can still grow, so
        # "every layer we know about is fetched" is not "done" — reserve 100.
        pct = 100.0 * self.layers_fetched / self.layers_total
        return max(0, min(_INCOMPLETE_PERCENT_CEILING, int(round(pct))))

    def format_summary(self) -> str:
        if self.phase == _PULL_PHASE_COMPLETE:
            return f"{self.image} pull complete"
        if self.phase == _PULL_PHASE_INSTALLING:
            if self.layers_total > 0:
                return (
                    f"{self.image} installing ({self.layers_complete}/{self.layers_total} layers)"
                )
            return f"{self.image} installing"
        if self.total_bytes > 0 and not self.total_bytes_partial:
            return (
                f"{self.image} {_format_bytes(self.downloaded_bytes)} of "
                f"{_format_bytes(self.total_bytes)} ({self.percent()}%)"
            )
        if self.total_bytes > 0:
            # Image size not known yet — report what has landed, not a ratio.
            return (
                f"{self.image} {_format_bytes(self.downloaded_bytes)} downloaded, "
                f"{self.layers_fetched:.1f}/{self.layers_total} layers ({self.percent()}%)"
            )
        if self.layers_total > 0:
            return f"{self.image} pulling ({self.layers_active}/{self.layers_total} layers)"
        return f"{self.image} starting"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "phase": self.phase,
            "progress_percent": self.percent(),
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "downloaded_human": (
                _format_bytes(self.downloaded_bytes) if self.total_bytes > 0 else None
            ),
            # Withheld until every layer is sized: the UI renders "X of Y"
            # verbatim, and a partial Y reads as a wrong number.
            "total_human": (
                _format_bytes(self.total_bytes)
                if self.total_bytes > 0 and not self.total_bytes_partial
                else None
            ),
            "total_bytes_partial": self.total_bytes_partial,
            "layers_fetched": round(self.layers_fetched, 2),
            "layers_total": self.layers_total,
            "layers_complete": self.layers_complete,
            "layers_active": self.layers_active,
            "last_docker_pull_line": self.last_line[:500],
        }


class _DockerPullTracker:
    """Mutates a :class:`_DockerPullProgress` as docker-pull lines arrive.

    ``feed(line)`` returns ``True`` when the externally-visible summary
    (phase / integer percent / layer counts) changed.
    """

    def __init__(self, image: str) -> None:
        self.progress = _DockerPullProgress(image=image)
        self._layer_state: dict[str, str] = {}
        self._layer_downloaded: dict[str, int] = {}
        self._layer_total: dict[str, int] = {}
        self._signature: tuple[Any, ...] = ()
        # Only the stream's terminator (``Status: …``) or ``mark_finished``
        # prove a pull is over: docker announces cached layers as ``Already
        # exists`` before saying anything about the ones it still has to fetch.
        self._stream_finished = False

    def feed(self, line: str) -> bool:
        message = line.strip()
        if not message:
            return False
        self.progress.last_line = message[:500]

        if message.startswith("Status:"):
            self._finalize_layer_bytes()
            self._recompute(force_phase=_PULL_PHASE_COMPLETE)
            return self._refresh_signature()

        match = _LAYER_LINE_RE.match(message)
        if not match:
            return self._refresh_signature()

        layer_id, rest = match.group(1), match.group(2)
        state = _match_layer_keyword(rest)
        if state is None:
            # Top-level line that happens to have ``<word>: …`` shape
            # (``latest: Pulling from cyberwaveos/foo``, ``Digest: …``).
            # Never register a phantom layer.
            return False

        self._apply_layer_event(layer_id, state, rest)
        return self._refresh_signature()

    def mark_finished(self) -> None:
        self._finalize_layer_bytes()
        self._recompute(force_phase=_PULL_PHASE_COMPLETE)

    def _finalize_layer_bytes(self) -> None:
        """Close the books: every layer is fetched and capped at its
        declared total, so the percentage reads exactly 100 %."""
        self._stream_finished = True
        for layer_id, total in self._layer_total.items():
            self._layer_downloaded[layer_id] = total
        for layer_id in self._layer_state:
            self._layer_state[layer_id] = "complete"

    def _apply_layer_event(self, layer_id: str, state: str, rest: str) -> None:
        self._layer_state[layer_id] = state
        if state == "downloading":
            bar = _BYTES_RE.search(rest)
            if bar is not None:
                curr, total = _decode_bytes(bar)
                if total > 0:
                    self._layer_total[layer_id] = total
                self._layer_downloaded[layer_id] = curr
        elif state in ("downloaded", "extracting", "complete"):
            # The layer's full download has landed locally by now —
            # credit the full bytes even if the last bar was partial.
            self._layer_downloaded[layer_id] = self._layer_total.get(
                layer_id,
                self._layer_downloaded.get(layer_id, 0),
            )
        # "pending" / "waiting" / "verifying" → no byte delta
        self._recompute()

    def feed_event(self, evt: Any) -> bool:
        """Engine-API analogue of :meth:`feed`. Returns True on visible change."""
        if not isinstance(evt, dict):
            return False
        status = (evt.get("status") or "").strip()
        if not status:
            return False
        layer_id_raw = evt.get("id")
        layer_id = str(layer_id_raw).strip() if layer_id_raw is not None else None
        detail = evt.get("progressDetail") or {}

        if layer_id and status:
            self.progress.last_line = f"{layer_id}: {status}"[:500]
        else:
            self.progress.last_line = status[:500]

        if status.startswith("Status:"):
            self._finalize_layer_bytes()
            self._recompute(force_phase=_PULL_PHASE_COMPLETE)
            return self._refresh_signature()

        # Top-level events with no layer id (``Digest:``, bare ``Pulling from``).
        if not layer_id:
            return False

        state = _match_layer_keyword(status)
        if state is None:
            return False

        self._layer_state[layer_id] = state
        if state == "downloading":
            cur = _coerce_int(detail.get("current"))
            tot = _coerce_int(detail.get("total"))
            if tot > 0:
                self._layer_total[layer_id] = tot
            if cur > 0:
                self._layer_downloaded[layer_id] = cur
        elif state in ("downloaded", "extracting", "complete"):
            # The layer's full download has landed by now, so credit its
            # compressed total — the same rule ``_apply_layer_event`` uses on
            # the subprocess path.  ``extracting`` must NOT feed the byte
            # counters: docker reports extraction in *uncompressed* bytes
            # restarting from zero, so honouring it would swap a compressed
            # total for a larger uncompressed one mid-pull and walk the
            # percentage backwards.
            self._layer_downloaded[layer_id] = self._layer_total.get(
                layer_id,
                self._layer_downloaded.get(layer_id, 0),
            )
        self._recompute()
        return self._refresh_signature()

    def _unsized_layer_count(self) -> int:
        """Announced layers that will contribute bytes we can't see yet.

        A layer declares its size on its first ``Downloading`` event. Cached
        layers are excluded: they land as ``complete`` without a size because
        they transfer nothing.
        """
        return sum(
            1
            for layer_id, state in self._layer_state.items()
            if layer_id not in self._layer_total and state != "complete"
        )

    def _fetched_layer_count(self) -> float:
        """Layers whose bytes are on disk, the in-flight one counted as its
        byte fraction so the number moves between layer boundaries."""
        fetched = 0.0
        for layer_id, state in self._layer_state.items():
            if state in _LAYER_FETCHED_STATES:
                fetched += 1.0
                continue
            total = self._layer_total.get(layer_id, 0)
            if state == "downloading" and total > 0:
                got = self._layer_downloaded.get(layer_id, 0)
                fetched += max(0.0, min(1.0, got / total))
        return fetched

    def _resolve_phase(self) -> str:
        states = set(self._layer_state.values())
        if not states:
            return _PULL_PHASE_STARTED
        if states & {"downloading", "waiting", "pending"}:
            return _PULL_PHASE_DOWNLOADING
        if states <= {"complete"}:
            # Every layer seen so far is done — the end of the pull only if
            # the stream said so, otherwise this is the opening burst of
            # ``Already exists`` on a re-pull.
            return _PULL_PHASE_COMPLETE if self._stream_finished else _PULL_PHASE_DOWNLOADING
        return _PULL_PHASE_INSTALLING

    def _recompute(self, *, force_phase: Optional[str] = None) -> None:
        self.progress.downloaded_bytes = sum(self._layer_downloaded.values())
        self.progress.total_bytes = sum(self._layer_total.values())
        self.progress.total_bytes_partial = (
            not self._stream_finished and self._unsized_layer_count() > 0
        )
        self.progress.layers_fetched = self._fetched_layer_count()
        self.progress.layers_total = len(self._layer_state)
        self.progress.layers_complete = sum(
            1 for s in self._layer_state.values() if s == "complete"
        )
        self.progress.layers_active = sum(
            1 for s in self._layer_state.values() if s in _LAYER_ACTIVE_STATES
        )

        if force_phase is not None:
            self.progress.phase = force_phase
        else:
            resolved = self._resolve_phase()
            if _PULL_PHASE_RANK.get(resolved, 0) >= _PULL_PHASE_RANK.get(self.progress.phase, 0):
                self.progress.phase = resolved

    def _refresh_signature(self) -> bool:
        sig = (
            self.progress.phase,
            self.progress.percent(),
            self.progress.layers_total,
            self.progress.layers_complete,
            self.progress.layers_active,
        )
        changed = sig != self._signature
        self._signature = sig
        return changed


def _match_layer_keyword(rest: str) -> Optional[str]:
    lowered = rest.lower()
    for prefix, state in _LAYER_KEYWORDS:
        if lowered.startswith(prefix):
            return state
    return None


def _decode_bytes(match: "re.Match[str]") -> tuple[int, int]:
    try:
        curr = float(match.group(1)) * _BYTE_UNITS.get(match.group(2), 1)
        total = float(match.group(3)) * _BYTE_UNITS.get(match.group(4), 1)
        return int(curr), int(total)
    except (ValueError, KeyError):
        return 0, 0


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _format_pull_event_for_log(evt: Any) -> str:
    """Render an engine-API event as a CLI-shaped line for the driverlog feed.

    Error events are filtered out by the caller before reaching this
    function (they raise ``CalledProcessError`` instead).
    """
    if not isinstance(evt, dict):
        return ""
    status = (evt.get("status") or "").strip()
    if not status:
        return ""
    layer_id = evt.get("id")
    detail = evt.get("progressDetail") or {}
    cur = _coerce_int(detail.get("current"))
    tot = _coerce_int(detail.get("total"))
    bar = f" {_format_bytes(cur)} / {_format_bytes(tot)}" if cur and tot else ""
    return f"{layer_id}: {status}{bar}" if layer_id else status


# ---------------------------------------------------------------------------
# Pull delivery contexts + streaming worker
# ---------------------------------------------------------------------------


@dataclass
class _PullDeliveryContext:
    """Per-twin delivery target for one shared ``docker pull``."""

    twin_uuid: str
    container_name: str
    driver_alert_ctx: Optional[Union[DriverStartingAlertContext, WorkerStartingAlertContext]] = None
    mqtt_client: Optional[Any] = None
    mqtt_topic: Optional[str] = None


def _resolve_pull_delivery_contexts(
    contexts: Iterable[_PullDeliveryContext],
    *,
    token: Optional[str],
) -> list[_PullDeliveryContext]:
    resolved: list[_PullDeliveryContext] = []
    for ctx in contexts:
        if ctx.mqtt_client is None or ctx.mqtt_topic is None:
            mqtt_client, mqtt_topic = _resolve_driver_log_publish_context(
                twin_uuid=ctx.twin_uuid,
                token=token,
            )
            ctx = _PullDeliveryContext(
                twin_uuid=ctx.twin_uuid,
                container_name=ctx.container_name,
                driver_alert_ctx=ctx.driver_alert_ctx,
                mqtt_client=ctx.mqtt_client or mqtt_client,
                mqtt_topic=ctx.mqtt_topic or mqtt_topic,
            )
        resolved.append(ctx)
    return resolved


def _publish_pull_line_to_mqtt(
    contexts: Iterable[_PullDeliveryContext],
    message: str,
    *,
    image: str,
) -> None:
    """Fan a docker-pull line to each twin's ``driverlog`` MQTT topic (MQTT only, no stderr)."""
    for ctx in contexts:
        _publish_driver_log_message(
            message,
            ctx.container_name,
            mqtt_client=ctx.mqtt_client,
            mqtt_topic=ctx.mqtt_topic,
            driver_image=image,
        )


def _broadcast_pull_event(
    contexts: Iterable[_PullDeliveryContext],
    message: str,
    *,
    image: str,
) -> None:
    """Mirror a low-rate phase-boundary event to stderr + MQTT."""
    print(message, file=sys.stderr)
    _publish_pull_line_to_mqtt(contexts, message, image=image)


def _apply_progress_to_alerts(
    contexts: Iterable[_PullDeliveryContext],
    progress: _DockerPullProgress,
    *,
    force: bool = False,
) -> None:
    patch = progress.to_metadata()
    for ctx in contexts:
        if ctx.driver_alert_ctx is None:
            continue
        ctx.driver_alert_ctx.update_metadata(patch, force=force)


#: How a registry phrases a rejected login, whichever path reports it:
#: docker-py's ``401 Client Error ... Unauthorized (...)``, the CLI's
#: ``unauthorized: authentication required``, or an in-stream error event.
_REGISTRY_AUTH_ERROR_RE = re.compile(r"unauthorized|incorrect username or password", re.IGNORECASE)


def _is_registry_auth_error(message: Optional[str]) -> bool:
    """True when the registry rejected the credentials the pull sent.

    Docker attaches the host's stored ``docker login`` to every pull and never
    falls back to anonymous when the registry rejects it, so a stale or revoked
    login fails pulls of public images too (CYB-4348).
    """
    return message is not None and _REGISTRY_AUTH_ERROR_RE.search(message) is not None


@contextlib.contextmanager
def _docker_cli_env(*, anonymous: bool) -> Iterator[Optional[dict[str, str]]]:
    """``env`` for a ``docker`` CLI call; *anonymous* hides the stored login.

    Points ``DOCKER_CONFIG`` at an empty directory, so the CLI finds no
    ``auths`` / ``credsStore`` and pulls unauthenticated. ``None`` (inherit
    the environment unchanged) when not *anonymous*.
    """
    if not anonymous:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="cw-docker-anon-") as config_dir:
        yield {**os.environ, "DOCKER_CONFIG": config_dir}


def _pull_with_anonymous_retry(image: str, attempt: Callable[[bool], None]) -> None:
    """Run ``attempt(False)``; if the registry rejects the stored login, ``attempt(True)``.

    The anonymous retry succeeds for any public image, turning a stale host
    ``docker login`` from an outage into a warning. If it fails too (private
    image, registry down) the original credential error is re-raised: that's
    the one the operator can act on.
    """
    try:
        attempt(False)
        return
    except subprocess.CalledProcessError as exc:
        if not _is_registry_auth_error(exc.stderr):
            raise
        auth_exc = exc

    logger.warning(
        "Registry refused the pull of %s as unauthorized; retrying without stored docker "
        "credentials (%s)",
        image,
        auth_exc.stderr,
    )
    try:
        attempt(True)
    except subprocess.CalledProcessError as retry_exc:
        logger.info("Anonymous retry of %s failed too: %s", image, retry_exc.stderr)
        raise auth_exc
    logger.warning(
        "Pulled %s anonymously: the host's stored docker login is stale or revoked. "
        "Run `docker logout` as the user running edge-core to clear it.",
        image,
    )


def _drive_pull(
    image: str,
    *,
    tracker: _DockerPullTracker,
    contexts_list: list[_PullDeliveryContext],
    timeout: int,
    on_progress: Optional[Callable[[_DockerPullProgress], None]],
    anonymous: bool = False,
) -> None:
    """Engine API first; ``docker`` CLI subprocess when the SDK or socket is unavailable."""
    try:
        _drive_pull_via_engine_api(
            image,
            tracker=tracker,
            contexts_list=contexts_list,
            timeout=timeout,
            on_progress=on_progress,
            anonymous=anonymous,
        )
    except _EngineAPIUnavailableError as exc:
        logger.info(
            "Falling back to docker CLI subprocess for pull of %s (%s)",
            image,
            exc,
        )
        _drive_pull_via_subprocess(
            image,
            tracker=tracker,
            contexts_list=contexts_list,
            timeout=timeout,
            on_progress=on_progress,
            anonymous=anonymous,
        )


def _pull_docker_image_with_progress_multi(
    image: str,
    *,
    contexts: Iterable[_PullDeliveryContext],
    token: Optional[str] = None,
    timeout: int = 600,
    on_progress: Optional[Callable[[_DockerPullProgress], None]] = None,
) -> _DockerPullProgress:
    """Pull *image* once and fan progress out to every twin in *contexts*.

    Per-line MQTT publishes go to each twin's ``driverlog`` topic; byte
    progress lands on each twin's ``driver_starting`` alert metadata.
    The journal only gets the start/finish/failure event lines.

    Engine API is the primary path (per-layer byte deltas regardless
    of TTY); subprocess is the fallback when the docker SDK is missing
    or the daemon socket is unreachable.
    """
    contexts_list = _resolve_pull_delivery_contexts(contexts, token=token)
    tracker = _DockerPullTracker(image)

    logger.info("Pulling docker image: %s", image)
    _apply_progress_to_alerts(contexts_list, tracker.progress, force=True)
    _broadcast_pull_event(
        contexts_list,
        f"docker pull started for image {image}",
        image=image,
    )

    def _attempt(anonymous: bool) -> None:
        nonlocal tracker
        if anonymous:
            # A fresh pull: the rejected attempt's error line must not linger
            # as ``last_docker_pull_line``.
            tracker = _DockerPullTracker(image)
            _broadcast_pull_event(
                contexts_list,
                f"docker pull: registry refused {image} as unauthorized; "
                "retrying without stored credentials",
                image=image,
            )
        _drive_pull(
            image,
            tracker=tracker,
            contexts_list=contexts_list,
            timeout=timeout,
            on_progress=on_progress,
            anonymous=anonymous,
        )

    try:
        _pull_with_anonymous_retry(image, _attempt)
    except subprocess.TimeoutExpired:
        for ctx in contexts_list:
            if ctx.driver_alert_ctx:
                ctx.driver_alert_ctx.update_metadata(
                    {
                        "phase": "pull_timed_out",
                        "last_message": f"docker pull timed out for {image}",
                    },
                    force=True,
                )
        _broadcast_pull_event(
            contexts_list,
            f"docker pull timed out for image {image}",
            image=image,
        )
        raise
    except subprocess.CalledProcessError as exc:
        error_output = (
            (exc.stderr or "").strip()
            or tracker.progress.last_line
            or f"docker pull exited with code {exc.returncode}"
        )
        for ctx in contexts_list:
            if ctx.driver_alert_ctx:
                ctx.driver_alert_ctx.update_metadata(
                    {
                        "phase": "pull_exit_error",
                        "last_error": str(error_output)[:500],
                    },
                    force=True,
                )
        _broadcast_pull_event(
            contexts_list,
            f"docker pull failed for image {image}: {error_output}",
            image=image,
        )
        raise
    except OSError as exc:
        for ctx in contexts_list:
            if ctx.driver_alert_ctx:
                ctx.driver_alert_ctx.update_metadata(
                    {"phase": "pull_spawn_failed", "last_error": str(exc)[:500]},
                    force=True,
                )
        _broadcast_pull_event(
            contexts_list,
            f"docker pull failed for image {image}: {exc}",
            image=image,
        )
        raise

    tracker.mark_finished()
    final = tracker.progress
    # Frontend's post-pull gate keys off ``pull_stream_finished``.
    finished_patch = {
        **final.to_metadata(),
        "phase": "pull_stream_finished",
        "last_message": f"docker pull completed for {image}",
    }
    for ctx in contexts_list:
        if ctx.driver_alert_ctx:
            ctx.driver_alert_ctx.update_metadata(finished_patch, force=True)
    if on_progress is not None:
        try:
            on_progress(final)
        except Exception:
            logger.debug(
                "on_progress callback raised for image %s",
                image,
                exc_info=True,
            )
    _broadcast_pull_event(
        contexts_list,
        f"docker pull completed for image {image}",
        image=image,
    )
    return final


def _drive_pull_via_engine_api(
    image: str,
    *,
    tracker: _DockerPullTracker,
    contexts_list: list[_PullDeliveryContext],
    timeout: int,
    on_progress: Optional[Callable[[_DockerPullProgress], None]],
    anonymous: bool = False,
) -> None:
    """Stream the pull via the Docker Engine HTTP API.

    Maps docker-py errors to the ``subprocess.*`` types the dispatcher
    already handles; raises :class:`_EngineAPIUnavailableError` only
    when the SDK can't be loaded at all.
    """
    try:
        import docker  # type: ignore[import-untyped]
        from docker.errors import APIError, DockerException  # type: ignore[import-untyped]
        from docker.utils import parse_repository_tag  # type: ignore[import-untyped]
    except ImportError as exc:
        raise _EngineAPIUnavailableError(f"docker python SDK not installed: {exc}") from exc

    # Socket-level stall watchdog: per-HTTP-read timeout the SDK applies
    # to the streaming pull. The application-level per-event deadline in
    # the loop below is the higher-level watchdog.
    try:
        client = docker.from_env(timeout=timeout)
    except DockerException as exc:
        raise _EngineAPIUnavailableError(f"docker daemon not reachable: {exc}") from exc

    repo, tag = parse_repository_tag(image)

    try:
        stream = client.api.pull(
            repo,
            tag=tag,
            stream=True,
            decode=True,
            # ``None`` lets docker-py attach the stored login from the caller's
            # config.json; ``{}`` sends an empty X-Registry-Auth (anonymous).
            auth_config={} if anonymous else None,
        )
    except APIError as exc:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["docker", "pull", image],
            stderr=str(exc),
        ) from exc
    except DockerException as exc:
        raise OSError(f"docker engine API pull failed for {image}: {exc}") from exc

    # ``timeout`` is a per-event stall watchdog, not a wall-clock cap on
    # the whole pull: a multi-GB image on a slow link can legitimately
    # take an hour, but no single read should ever stall for ``timeout``
    # seconds. Reset ``deadline`` whenever the engine yields anything.
    deadline = time.monotonic() + timeout
    try:
        for evt in stream:
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(
                    cmd=["docker", "pull", image],
                    timeout=timeout,
                )
            deadline = time.monotonic() + timeout

            if isinstance(evt, dict) and "error" in evt:
                err_msg = str(evt.get("error") or "unknown docker pull error")
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=["docker", "pull", image],
                    stderr=err_msg,
                )

            # Throttle every fan-out (MQTT line, alert metadata,
            # on_progress callback) to ``feed_event``'s signature so a
            # high-cadence engine-API stream doesn't hammer subscribers
            # on sub-percent byte deltas.
            if not tracker.feed_event(evt):
                continue

            line = _format_pull_event_for_log(evt)
            if line:
                _publish_pull_line_to_mqtt(
                    contexts_list,
                    f"docker pull: {line}",
                    image=image,
                )
            _apply_progress_to_alerts(contexts_list, tracker.progress)
            if on_progress is not None:
                try:
                    on_progress(tracker.progress)
                except Exception:
                    logger.debug(
                        "on_progress callback raised for image %s",
                        image,
                        exc_info=True,
                    )
    except APIError as exc:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["docker", "pull", image],
            stderr=str(exc),
        ) from exc
    except DockerException as exc:
        raise OSError(f"docker engine API stream failed for {image}: {exc}") from exc


def _drive_pull_via_subprocess(
    image: str,
    *,
    tracker: _DockerPullTracker,
    contexts_list: list[_PullDeliveryContext],
    timeout: int,
    on_progress: Optional[Callable[[_DockerPullProgress], None]],
    anonymous: bool = False,
) -> None:
    """Fallback: ``docker pull`` subprocess + stdout parser.

    Docker's CLI suppresses byte progress on a non-TTY pipe, so this
    path only surfaces layer-count progress via ``layers_active``.
    """
    # The temp DOCKER_CONFIG must outlive the process: the CLI reads it after exec.
    with _docker_cli_env(anonymous=anonymous) as env:
        process = subprocess.Popen(
            ["docker", "pull", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        last_message: Optional[str] = None
        try:
            if process.stdout:
                for message in _iter_stream_messages(process.stdout):
                    if message == last_message:
                        continue
                    last_message = message
                    _publish_pull_line_to_mqtt(
                        contexts_list,
                        f"docker pull: {message}",
                        image=image,
                    )
                    if tracker.feed(message):
                        _apply_progress_to_alerts(contexts_list, tracker.progress)
                        if on_progress is not None:
                            try:
                                on_progress(tracker.progress)
                            except Exception:
                                logger.debug(
                                    "on_progress callback raised for image %s",
                                    image,
                                    exc_info=True,
                                )

            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            raise

    if return_code != 0:
        error_output = tracker.progress.last_line or f"docker pull exited with code {return_code}"
        raise subprocess.CalledProcessError(
            return_code,
            ["docker", "pull", image],
            stderr=error_output,
        )


def _run_docker_pull(image: str, *, timeout: int) -> None:
    """``docker pull`` *image* without progress fan-out, with the anonymous retry."""

    def _attempt(anonymous: bool) -> None:
        with _docker_cli_env(anonymous=anonymous) as env:
            subprocess.run(
                ["docker", "pull", image],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

    _pull_with_anonymous_retry(image, _attempt)


def _pull_docker_image_with_progress(
    image: str,
    *,
    container_name: str,
    twin_uuid: str,
    token: str,
    timeout: int = 600,
    driver_alert_ctx: Optional[
        Union[DriverStartingAlertContext, WorkerStartingAlertContext]
    ] = None,
) -> None:
    """Backward-compat single-twin shim around :func:`_pull_docker_image_with_progress_multi`."""
    _pull_docker_image_with_progress_multi(
        image,
        contexts=[
            _PullDeliveryContext(
                twin_uuid=twin_uuid,
                container_name=container_name,
                driver_alert_ctx=driver_alert_ctx,
            ),
        ],
        token=token,
        timeout=timeout,
    )


def _follow_container_logs(
    container_name: str,
    *,
    twin_uuid: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    """Follow ``docker logs -f`` and forward lines to the service logger.

    When *twin_uuid* and *token* are provided, each log line is also
    published to the backend via MQTT as a ``driver_log`` event.
    """
    from .startup import _inspect_driver_container, _resolve_container_driver_image

    if not shutil.which("docker"):
        logger.warning("Cannot stream logs: Docker is not installed")
        return

    logger.info("Forwarding logs for container %s to service logs", container_name)
    received_lines = 0

    mqtt_client: Optional[Any] = None
    mqtt_topic: Optional[str] = None
    driver_image: Optional[str] = None
    mqtt_client, mqtt_topic = _resolve_driver_log_publish_context(
        twin_uuid=twin_uuid,
        token=token,
    )
    if mqtt_topic:
        logger.info("Driver logs for %s will be published to %s", container_name, mqtt_topic)
        driver_image = _resolve_container_driver_image(_inspect_driver_container(container_name))

    cmd = ["docker", "logs", "-f"]
    since_ts = _CONTAINER_LOG_LAST_SEEN.get(container_name)
    if since_ts:
        cmd += ["--since", since_ts]
    else:
        # Cold start: bound the backlog rather than replay everything.
        tail_lines = _cold_start_tail_lines()
        if tail_lines:
            cmd += ["--tail", str(tail_lines)]
    cmd.append(container_name)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        logger.warning("Failed to start docker log streaming for %s: %s", container_name, exc)
        return

    try:
        if not process.stdout:
            logger.warning("No stdout stream when following logs for %s", container_name)
            return

        for line in process.stdout:
            message = line.rstrip()
            if message:
                received_lines += 1
                _log_and_publish_driver_message(
                    message,
                    container_name,
                    mqtt_client=mqtt_client,
                    mqtt_topic=mqtt_topic,
                    driver_image=driver_image,
                )

                if twin_uuid and container_name not in _CONTAINER_RUNTIME_ERROR_ALERTED:
                    m = _RUNTIME_ERROR_PATTERN.search(message)
                    if m:
                        _send_runtime_error_alert(
                            twin_uuid,
                            container_name,
                            m.group(1).strip(),
                        )
    except Exception as exc:
        logger.warning("Error while streaming logs for %s: %s", container_name, exc)
    finally:
        _CONTAINER_LOG_LAST_SEEN[container_name] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        _CONTAINER_RUNTIME_ERROR_ALERTED.discard(container_name)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        logger.info(
            "Stopped forwarding logs for container %s (lines_received=%d)",
            container_name,
            received_lines,
        )
