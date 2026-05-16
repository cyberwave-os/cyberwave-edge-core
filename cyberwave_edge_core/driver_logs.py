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
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

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
    from .startup import _list_running_driver_containers, _CONTAINER_TWIN_MAP, load_token

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
    from .startup import EDGE_CORE_VERSION, CYBERWAVE_SDK_VERSION

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


def _pull_docker_image_with_progress(
    image: str,
    *,
    container_name: str,
    twin_uuid: str,
    token: str,
    timeout: int = 600,
    driver_alert_ctx: Optional[DriverStartingAlertContext] = None,
) -> None:
    """Pull a driver image while streaming progress to local logs and MQTT."""
    mqtt_client, mqtt_topic = _resolve_driver_log_publish_context(
        twin_uuid=twin_uuid,
        token=token,
    )
    logger.info("Pulling docker image: %s", image)
    if driver_alert_ctx:
        driver_alert_ctx.update_metadata(
            {"phase": "pulling", "last_message": f"docker pull started for {image}"},
            force=True,
        )
    _log_and_publish_driver_message(
        f"docker pull started for image {image}",
        container_name,
        mqtt_client=mqtt_client,
        mqtt_topic=mqtt_topic,
        driver_image=image,
    )

    try:
        process = subprocess.Popen(
            ["docker", "pull", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        if driver_alert_ctx:
            driver_alert_ctx.update_metadata(
                {"phase": "pull_spawn_failed", "last_error": str(exc)[:500]},
                force=True,
            )
        _log_and_publish_driver_message(
            f"docker pull failed for image {image}: {exc}",
            container_name,
            mqtt_client=mqtt_client,
            mqtt_topic=mqtt_topic,
            driver_image=image,
        )
        raise

    recent_messages: deque[str] = deque(maxlen=20)
    last_message: Optional[str] = None
    try:
        if process.stdout:
            for message in _iter_stream_messages(process.stdout):
                if message == last_message:
                    continue
                last_message = message
                recent_messages.append(message)
                if driver_alert_ctx:
                    driver_alert_ctx.update_metadata(
                        {
                            "phase": "downloading",
                            "last_docker_pull_line": message[:500],
                            "recent_pull_lines": list(recent_messages)[-5:],
                        },
                    )
                _log_and_publish_driver_message(
                    f"docker pull: {message}",
                    container_name,
                    mqtt_client=mqtt_client,
                    mqtt_topic=mqtt_topic,
                    driver_image=image,
                )

        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        if driver_alert_ctx:
            driver_alert_ctx.update_metadata(
                {
                    "phase": "pull_timed_out",
                    "last_message": f"docker pull timed out for {image}",
                },
                force=True,
            )
        _log_and_publish_driver_message(
            f"docker pull timed out for image {image}",
            container_name,
            mqtt_client=mqtt_client,
            mqtt_topic=mqtt_topic,
            driver_image=image,
        )
        raise

    if return_code != 0:
        error_output = "\n".join(recent_messages) or f"docker pull exited with code {return_code}"
        if driver_alert_ctx:
            driver_alert_ctx.update_metadata(
                {
                    "phase": "pull_exit_error",
                    "last_error": error_output[:500],
                    "recent_pull_lines": list(recent_messages)[-5:],
                },
                force=True,
            )
        _log_and_publish_driver_message(
            f"docker pull failed for image {image}: {error_output}",
            container_name,
            mqtt_client=mqtt_client,
            mqtt_topic=mqtt_topic,
            driver_image=image,
        )
        raise subprocess.CalledProcessError(
            return_code,
            ["docker", "pull", image],
            stderr=error_output,
        )

    if driver_alert_ctx:
        driver_alert_ctx.update_metadata(
            {
                "phase": "pull_stream_finished",
                "last_message": f"docker pull completed for {image}",
            },
            force=True,
        )
    _log_and_publish_driver_message(
        f"docker pull completed for image {image}",
        container_name,
        mqtt_client=mqtt_client,
        mqtt_topic=mqtt_topic,
        driver_image=image,
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
