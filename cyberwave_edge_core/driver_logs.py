"""Driver container log streaming and MQTT forwarding.

Extracted from startup.py — handles the pipeline that follows ``docker logs``
for each driver container, forwards lines to the service logger (stderr), and
publishes them to the MQTT ``driverlog`` topic.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from .utils import DriverStartingAlertContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level state (moved from startup.py)
# ---------------------------------------------------------------------------

# Track active log streaming threads per container to avoid duplicates.
_CONTAINER_LOG_THREADS: dict[str, threading.Thread] = {}

# Track when log streaming last ended per container so reattach uses --since.
_CONTAINER_LOG_LAST_SEEN: dict[str, str] = {}

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
        from .startup import _send_alert_for_twin

        _send_alert_for_twin(
            twin_uuid,
            "Driver runtime error",
            (
                f"Container '{container_name}' reported a RuntimeError: "
                f"{error_message[:500]}"
            ),
            "driver_runtime_error",
            severity="error",
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

# ``docker pull`` prints SI-scaled byte counts (1 kB = 1000 B); both
# ``kB`` and ``KB`` show up depending on docker version.
_BYTE_UNITS: dict[str, int] = {
    "B": 1, "kB": 10**3, "KB": 10**3,
    "MB": 10**6, "GB": 10**9, "TB": 10**12,
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
    phase: str = _PULL_PHASE_STARTED
    last_line: str = ""

    def percent(self) -> int:
        if self.total_bytes <= 0:
            return 100 if self.phase == _PULL_PHASE_COMPLETE else 0
        pct = 100.0 * self.downloaded_bytes / self.total_bytes
        return max(0, min(100, int(round(pct))))

    def format_summary(self) -> str:
        if self.phase == _PULL_PHASE_COMPLETE:
            return f"{self.image} pull complete"
        if self.phase == _PULL_PHASE_INSTALLING:
            if self.layers_total > 0:
                return (
                    f"{self.image} installing "
                    f"({self.layers_complete}/{self.layers_total} layers)"
                )
            return f"{self.image} installing"
        if self.total_bytes > 0:
            return (
                f"{self.image} {_format_bytes(self.downloaded_bytes)} of "
                f"{_format_bytes(self.total_bytes)} ({self.percent()}%)"
            )
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
            "total_human": (
                _format_bytes(self.total_bytes) if self.total_bytes > 0 else None
            ),
            "layers_total": self.layers_total,
            "layers_complete": self.layers_complete,
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

    def feed(self, line: str) -> bool:
        message = line.strip()
        if not message:
            return False
        self.progress.last_line = message[:500]

        if message.startswith("Status:"):
            # Cap each known layer at its declared total so percent reads 100 %.
            for layer_id, total in self._layer_total.items():
                self._layer_downloaded[layer_id] = total
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
        for layer_id, total in self._layer_total.items():
            self._layer_downloaded[layer_id] = total
        self._recompute(force_phase=_PULL_PHASE_COMPLETE)

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

    def _recompute(self, *, force_phase: Optional[str] = None) -> None:
        self.progress.downloaded_bytes = sum(self._layer_downloaded.values())
        self.progress.total_bytes = sum(self._layer_total.values())
        self.progress.layers_total = len(self._layer_state)
        self.progress.layers_complete = sum(
            1 for s in self._layer_state.values() if s == "complete"
        )

        if force_phase is not None:
            self.progress.phase = force_phase
            return

        states = set(self._layer_state.values())
        if not states:
            self.progress.phase = _PULL_PHASE_STARTED
        elif states <= {"complete"}:
            self.progress.phase = _PULL_PHASE_COMPLETE
        elif states & {"downloading", "waiting", "pending"}:
            self.progress.phase = _PULL_PHASE_DOWNLOADING
        else:
            self.progress.phase = _PULL_PHASE_INSTALLING

    def _refresh_signature(self) -> bool:
        sig = (
            self.progress.phase,
            self.progress.percent(),
            self.progress.layers_total,
            self.progress.layers_complete,
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


# ---------------------------------------------------------------------------
# Pull delivery contexts + streaming worker
# ---------------------------------------------------------------------------


@dataclass
class _PullDeliveryContext:
    """Per-twin delivery target for one shared ``docker pull``."""

    twin_uuid: str
    container_name: str
    driver_alert_ctx: Optional[DriverStartingAlertContext] = None
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

    try:
        process = subprocess.Popen(
            ["docker", "pull", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
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

    if return_code != 0:
        error_output = (
            tracker.progress.last_line
            or f"docker pull exited with code {return_code}"
        )
        for ctx in contexts_list:
            if ctx.driver_alert_ctx:
                ctx.driver_alert_ctx.update_metadata(
                    {
                        "phase": "pull_exit_error",
                        "last_error": error_output[:500],
                    },
                    force=True,
                )
        _broadcast_pull_event(
            contexts_list,
            f"docker pull failed for image {image}: {error_output}",
            image=image,
        )
        raise subprocess.CalledProcessError(
            return_code,
            ["docker", "pull", image],
            stderr=error_output,
        )

    tracker.mark_finished()
    final = tracker.progress
    # Preserve the historical ``pull_stream_finished`` sentinel so the
    # frontend's existing "post-pull" gate keeps working.
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


def _pull_docker_image_with_progress(
    image: str,
    *,
    container_name: str,
    twin_uuid: str,
    token: str,
    timeout: int = 600,
    driver_alert_ctx: Optional[DriverStartingAlertContext] = None,
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
    debug_log_stream = logger.isEnabledFor(logging.DEBUG)
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
        driver_image = _resolve_container_driver_image(
            _inspect_driver_container(container_name)
        )

    cmd = ["docker", "logs", "-f"]
    since_ts = _CONTAINER_LOG_LAST_SEEN.get(container_name)
    if since_ts:
        cmd += ["--since", since_ts]
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

                if debug_log_stream:
                    logger.debug(
                        "Container log line received (container=%s, line=%d, chars=%d)",
                        container_name,
                        received_lines,
                        len(message),
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
