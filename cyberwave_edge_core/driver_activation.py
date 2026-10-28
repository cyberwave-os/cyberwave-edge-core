"""Switch a twin's drivers to a staged release now, and go back if they do not come up.

The backend sends ``activate_driver_update`` once the operator picks "recreate now". By then
it has written the release's images into the twin's ``drivers`` metadata (``pins.activate``),
so the edge does not resolve anything new: it recreates this twin's containers through the
same path a driver start uses (``fetch_and_run_twin_drivers``, without pulling, since staging
already did) and checks they stay up.

The steps, and why each is there:

1. The update must be staged (a record from ``driver_updates``) with its images still on disk,
   and the twin must not run a local ``driver_overrides`` image: an override runs what it
   names, and reporting ``active`` would confirm a release that is not running.
2. The twin is claimed (``begin_driver_activation``) so revival, the settings reconciler and
   an edge-core restart leave its containers alone. If the edge is busy restarting, nothing is
   reported and the command is released, so the backend's next delivery tries again.
3. What runs now is written down (``previous.json``) so the image cleanup keeps it for a day.
4. Every container of the twin is removed (including services the new release no longer has)
   and the twin's drivers are started from the new metadata.
5. They must all stay running for a settle period, without restarting.
6. If not, the twin is recreated from the ``drivers`` block snapshotted at staging time, and the
   failure is reported. The rollback is the edge's; whether the backend then puts the old release
   back in the metadata is the backend's (see ``deployments.report``).

Each failure carries a code the backend stores on the deployment.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from . import driver_update_state as state
from .driver_updates import DOCKER_UNAVAILABLE, INVALID_COMMAND, UNKNOWN, StatusReporter

logger = logging.getLogger(__name__)

ACTIVATE_COMMAND = "activate_driver_update"

NOT_STAGED = "not_staged"
STAGED_IMAGES_MISSING = "staged_images_missing"
LOCALLY_OVERRIDDEN = "locally_overridden"
START_FAILED = "start_failed"
UNHEALTHY = "unhealthy_after_start"

SETTLE_SECONDS_ENV = "CYBERWAVE_DRIVER_ACTIVATION_SETTLE_SECONDS"
DEFAULT_SETTLE_SECONDS = 30.0
POLL_SECONDS = 2.0
#: How long what ran before an update is kept from the image cleanup, for a later rollback.
PREVIOUS_KEEP_SECONDS = 24 * 60 * 60


class ActivationError(Exception):
    """Activation failed for a reason the backend should record as *code*."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class Busy(Exception):
    """The edge is restarting or already activating this twin; try again on the next delivery."""


@dataclass(frozen=True)
class ActivateRequest:
    request_id: str
    twin_uuid: str
    package: str
    target_version: str


def parse_request(payload: Mapping[str, Any]) -> ActivateRequest:
    request_id = str(payload.get("request_id") or "").strip()
    twin_uuid = str(payload.get("twin_uuid") or "").strip()
    if not state.is_valid_request_id(request_id):
        raise ActivationError(INVALID_COMMAND, f"not a usable request_id: {request_id!r}")
    if not twin_uuid:
        raise ActivationError(INVALID_COMMAND, "The activate command names no twin")
    return ActivateRequest(
        request_id=request_id,
        twin_uuid=twin_uuid,
        package=str(payload.get("package") or ""),
        target_version=str(payload.get("target_version") or ""),
    )


# --- Docker ------------------------------------------------------------------------


def twin_containers(twin_uuid: str) -> list[str]:
    """Every driver container of the twin that exists, running or not."""
    from . import startup
    from .docker_helpers import docker_available, driver_container_name

    names = {name for name, owner in startup._CONTAINER_TWIN_MAP.items() if owner == twin_uuid}
    if docker_available():
        try:
            listed = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--all",
                    "--format",
                    "{{.Names}}",
                    "--filter",
                    f"name={driver_container_name(twin_uuid)}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.split()
        except (subprocess.TimeoutExpired, OSError):
            listed = []
        names.update(name for name in listed if startup._container_is_twins(name, twin_uuid))
    return sorted(names)


def _image_of(inspect: Optional[dict[str, Any]]) -> str:
    config = (inspect or {}).get("Config")
    return str(config.get("Image") or "") if isinstance(config, dict) else ""


def running_images(containers: list[str]) -> dict[str, str]:
    """``container name -> image reference`` for the containers that can be inspected."""
    from .docker_helpers import docker_inspect

    return {
        name: image for name in containers if (image := _image_of(docker_inspect(name))) != ""
    }


def remove_containers(containers: list[str]) -> None:
    """Remove the containers, and forget them so revival does not count them as down."""
    from . import startup
    from .docker_helpers import docker_rm

    for name in containers:
        docker_rm(name)
        startup._CONTAINER_TWIN_MAP.pop(name, None)
        startup._DRIVER_HEALTH_PREVIOUS.pop(name, None)


def settle_seconds() -> float:
    try:
        return max(float(os.environ.get(SETTLE_SECONDS_ENV, "") or DEFAULT_SETTLE_SECONDS), 0.0)
    except ValueError:
        return DEFAULT_SETTLE_SECONDS


def wait_until_settled(
    containers: list[str],
    *,
    seconds: float,
    sleep: Any = time.sleep,
    clock: Any = time.monotonic,
) -> Optional[str]:
    """None when every container stayed running for *seconds*; else what went wrong.

    A container that is not running, is restarting, or has restarted since the first look fails
    the check at once; there is no point waiting out the rest of the period.
    """
    from .docker_helpers import container_status_of, docker_inspect

    def restarts(inspect: Optional[dict[str, Any]]) -> int:
        try:
            return int((inspect or {}).get("RestartCount") or 0)
        except (TypeError, ValueError):
            return 0

    first = {name: restarts(docker_inspect(name)) for name in containers}
    deadline = clock() + seconds
    while True:
        for name in containers:
            inspect = docker_inspect(name)
            status = container_status_of(inspect)
            if status != "running":
                return f"{name} is {status}"
            if (inspect or {}).get("State", {}).get("Restarting"):
                return f"{name} is restarting"
            if restarts(inspect) > first[name]:
                return f"{name} restarted"
        if clock() >= deadline:
            return None
        sleep(POLL_SECONDS)


# --- The recreate ------------------------------------------------------------------


def _recreate(
    twin_uuid: str, *, drivers_override: Optional[Mapping[str, Any]] = None
) -> list[dict[str, Any]]:
    """Start the twin's drivers from its metadata (or *drivers_override*), pulling nothing."""
    from . import startup

    token = startup.load_token()
    fingerprint = startup.load_saved_fingerprint()
    environment_uuid = startup.load_environment_uuid()
    if not (token and fingerprint and environment_uuid):
        raise ActivationError(UNKNOWN, "This edge has no credentials to start drivers with")
    return startup.fetch_and_run_twin_drivers(
        token,
        environment_uuid,
        fingerprint,
        pull_images=False,
        only_twin_uuids={twin_uuid},
        drivers_override={twin_uuid: dict(drivers_override)} if drivers_override else None,
    )


def _started(results: list[dict[str, Any]], twin_uuid: str) -> tuple[list[str], list[str]]:
    """``(container names that started, ones that did not)`` among this twin's results."""
    from .docker_helpers import driver_container_name

    up: list[str] = []
    down: list[str] = []
    for result in results:
        if result.get("twin_uuid") != twin_uuid:
            continue
        name = driver_container_name(twin_uuid, result.get("service_name"))
        (up if result.get("success") else down).append(name)
    return up, down


def _bring_up(twin_uuid: str, **kwargs: Any) -> list[str]:
    """Recreate the twin's drivers and wait for them to settle; returns their names.

    Raises :class:`ActivationError` with ``START_FAILED`` or ``UNHEALTHY``.
    """
    results = _recreate(twin_uuid, **kwargs)
    up, down = _started(results, twin_uuid)
    if down or not up:
        raise ActivationError(
            START_FAILED,
            "These containers did not start: " + (", ".join(down) if down else "none were started"),
        )
    problem = wait_until_settled(up, seconds=settle_seconds())
    if problem:
        raise ActivationError(UNHEALTHY, f"The new drivers did not stay up: {problem}")
    return up


def _staged_images_present(record: Mapping[str, Any]) -> Optional[str]:
    """The first staged image no longer on this host, or None when all are here."""
    from .driver_updates import ManifestImage, _is_verified

    for image in record.get("images") or []:
        if not isinstance(image, dict) or not image.get("image"):
            continue
        entry = ManifestImage(
            image=str(image["image"]), digest=image.get("digest"), service=image.get("service")
        )
        if not _is_verified(entry):
            return entry.image
    return None


def _is_running(name: str) -> bool:
    from .docker_helpers import container_status_of, docker_inspect

    return container_status_of(docker_inspect(name)) == "running"


def _already_running(record: Mapping[str, Any], twin_uuid: str) -> Optional[dict[str, str]]:
    """The running images when the twin already runs everything the update staged.

    That is the state after a switch whose ``active`` report did not get through, and the
    backend sends the command again; recreating the containers a second time would only
    interrupt a robot that is fine.
    """
    wanted = {str(image["image"]) for image in record.get("images") or [] if image.get("image")}
    if not wanted:
        return None
    running = {
        name: image
        for name, image in running_images(twin_containers(twin_uuid)).items()
        if _is_running(name)
    }
    return running if wanted <= set(running.values()) else None


def _staged_images_text(record: Mapping[str, Any]) -> str:
    """``image@digest`` for each staged image, for the log."""
    parts = []
    for image in record.get("images") or []:
        if not isinstance(image, dict) or not image.get("image"):
            continue
        digest = str(image.get("digest") or "")
        parts.append(f"{image['image']}@{digest[:19]}" if digest else str(image["image"]))
    return ", ".join(parts) or "no images"


def _running_text(images: Mapping[str, str]) -> str:
    return ", ".join(f"{name}={image}" for name, image in sorted(images.items())) or "nothing"


def activate(request: ActivateRequest) -> dict[str, str]:
    """Switch the twin to the staged release; returns the images now running.

    Raises :class:`Busy` (nothing was done), or :class:`ActivationError` after rolling back as
    far as it could.
    """
    from . import startup
    from .docker_helpers import docker_available

    record = state.read_staged_record(request.request_id)
    if record is None:
        raise ActivationError(NOT_STAGED, "This update was never staged on this edge")
    if record.get("twin_uuid") not in (None, "", request.twin_uuid):
        raise ActivationError(INVALID_COMMAND, "The staged update belongs to another twin")
    if not docker_available():
        raise ActivationError(DOCKER_UNAVAILABLE, "Docker is not available on this edge")
    already = _already_running(record, request.twin_uuid)
    if already is not None:
        logger.info(
            "Activating %s: twin %s already runs the staged images (%s)",
            request.request_id,
            request.twin_uuid,
            _running_text(already),
        )
        return already
    if request.twin_uuid in startup.load_driver_overrides():
        raise ActivationError(
            LOCALLY_OVERRIDDEN,
            "This twin runs a local driver override, which an update must not replace",
        )
    missing = _staged_images_present(record)
    if missing:
        raise ActivationError(STAGED_IMAGES_MISSING, f"The staged image {missing} is gone")

    if not startup.begin_driver_activation(request.twin_uuid):
        raise Busy(request.twin_uuid)
    logger.info(
        "Activating %s: switching twin %s to %s %s: %s",
        request.request_id,
        request.twin_uuid,
        record.get("package") or request.package,
        record.get("target_version") or request.target_version,
        _staged_images_text(record),
    )
    try:
        return _switch(request, record)
    finally:
        startup.end_driver_activation(request.twin_uuid)


def _switch(request: ActivateRequest, record: Mapping[str, Any]) -> dict[str, str]:
    from . import startup

    containers = twin_containers(request.twin_uuid)
    before = running_images(containers)
    logger.info(
        "Activating %s: twin %s runs %s now",
        request.request_id,
        request.twin_uuid,
        _running_text(before),
    )
    if before:
        state.record_previous_set(
            request.twin_uuid,
            request.request_id,
            list(before.values()),
            keep_seconds=PREVIOUS_KEEP_SECONDS,
        )

    remove_containers(containers)
    try:
        up = _bring_up(request.twin_uuid)
    except ActivationError as failure:
        raise _roll_back(request, record, failure, containers) from failure
    except Exception as exc:
        raise _roll_back(
            request, record, ActivationError(UNKNOWN, f"{type(exc).__name__}: {exc}"), containers
        ) from exc

    # The backend wrote the new release into the metadata before it sent the command; the
    # settings reconciler must see the recreated containers as the baseline, not as a change.
    startup._rebaseline_driver_metadata(request.twin_uuid)
    now_running = running_images(up)
    logger.info(
        "Activated %s: twin %s now runs %s",
        request.request_id,
        request.twin_uuid,
        _running_text(now_running),
    )
    return now_running


def _hand_back_to_revival(twin_uuid: str, names: list[str]) -> None:
    """Let driver revival bring *names* back when nothing runs after a failed rollback.

    ``remove_containers`` made the edge forget them, so revival would never look for them.
    Remembered as running and now missing, they are recreated from the twin's metadata once
    the grace period has passed, by which time the backend has put the running release back.
    """
    from . import startup

    for name in names:
        startup._CONTAINER_TWIN_MAP.setdefault(name, twin_uuid)
        startup._DRIVER_HEALTH_PREVIOUS.setdefault(name, "running")


def _roll_back(
    request: ActivateRequest,
    record: Mapping[str, Any],
    failure: ActivationError,
    before: list[str],
) -> ActivationError:
    """Start the previous release again after *failure*; returns the error to report.

    *before* names the containers the twin had, for the case where nothing can be started.
    """
    from . import startup

    previous = record.get("previous_drivers")
    remove_containers(twin_containers(request.twin_uuid))
    if not isinstance(previous, dict) or not previous:
        state.delete_staged_record(request.request_id)
        _hand_back_to_revival(request.twin_uuid, before)
        return ActivationError(
            failure.code,
            f"{failure} There was no earlier configuration to go back to, so the twin's "
            "drivers are down.",
        )
    try:
        _bring_up(request.twin_uuid, drivers_override=previous)
    except Exception as exc:
        logger.exception("Rolling back twin %s failed", request.twin_uuid)
        state.delete_staged_record(request.request_id)
        _hand_back_to_revival(request.twin_uuid, before)
        return ActivationError(
            failure.code, f"{failure} Going back to the previous release failed too: {exc}"
        )
    startup._rebaseline_after_rollback(request.twin_uuid, previous)
    state.delete_staged_record(request.request_id)
    logger.warning(
        "Rolled back %s: twin %s runs %s again after: %s",
        request.request_id,
        request.twin_uuid,
        _running_text(running_images(twin_containers(request.twin_uuid))),
        failure,
    )
    return ActivationError(failure.code, f"{failure} The previous release is running again.")


# --- The command -------------------------------------------------------------------

_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    """Its own worker, so a switch never waits behind a long download."""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="driver-activate")
        return _EXECUTOR


def _run(request: ActivateRequest) -> None:
    from . import startup

    reporter = StatusReporter(request.request_id)
    delivered = False
    try:
        observed = activate(request)
        delivered = reporter.send("active", observed=observed)
        if delivered:
            # Kept until now so that a report that did not get through can be sent again.
            state.delete_staged_record(request.request_id)
    except Busy:
        logger.info("Not activating %s now: the edge is busy; the next delivery retries", request)
        startup.forget_edge_command(ACTIVATE_COMMAND, request.request_id)
        return
    except ActivationError as exc:
        logger.warning("Activating %s failed (%s): %s", request.request_id, exc.code, exc)
        delivered = reporter.send("failed", error_code=exc.code, error=str(exc))
    except Exception as exc:
        logger.exception("Activating %s failed unexpectedly", request.request_id)
        delivered = reporter.send("failed", error_code=UNKNOWN, error=f"{type(exc).__name__}: {exc}")
    if not delivered:
        # The backend still thinks this is pending and will send it again; after a switch the
        # staged record is still here, so the repeat only reports that it is running.
        startup.forget_edge_command(ACTIVATE_COMMAND, request.request_id)


def _reject(request_id: str, error: ActivationError) -> None:
    StatusReporter(request_id).send("failed", error_code=error.code, error=str(error))


def handle_activate_command(payload: Mapping[str, Any]) -> None:
    """The ``activate_driver_update`` handler: validate, then switch on the worker thread."""
    try:
        request = parse_request(payload)
    except ActivationError as exc:
        logger.warning("Ignoring an unusable %s: %s", ACTIVATE_COMMAND, exc)
        request_id = str(payload.get("request_id") or "").strip()
        if not state.is_valid_request_id(request_id):
            return
        _executor().submit(_reject, request_id, exc)
        return
    _executor().submit(_run, request)
