"""Staged driver updates, edge side: download a release now, switch to it later.

The backend (``services/driver_releases/deployments.py``) sends ``stage_driver_update``
with the release's manifest, over MQTT and in every ``/edges/discover`` response until
this edge reports ``staged``. Staging pulls the images the manifest names for this host,
checks each against the digest CI recorded, writes a *staged record* and reports back. It
changes nothing that runs: containers, twin metadata and the images in use are untouched.

Which images: a manifest lists, per **profile** (``default`` or ``linux-aarch64-jetson``,
the keys of a twin's ``drivers`` block), the images of each **variant** (``robot`` or
``sim``). The edge already answers both questions when it starts a driver, through
``_platform_driver_keys`` and ``_resolve_compose_variant``; staging asks the same
functions, so it stages what this host would start.

A command is delivered more than once, so staging is idempotent: a request that already
has a staged record only re-reports ``staged``. If the final report cannot be sent, the
command is released from the dedupe window, so the next delivery tries again.

Every edge advertises ``driver_staging_v1``; whether the backend stages for a given twin is
its own call (the deploy policy of the twin's pin).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from . import driver_update_state as state
from .docker_helpers import docker_available, docker_data_root

logger = logging.getLogger(__name__)

STAGE_COMMAND = "stage_driver_update"
CAPABILITY = "driver_staging_v1"


# Error codes the backend stores on a failed deployment.
INVALID_COMMAND = "invalid_command"
DOCKER_UNAVAILABLE = "docker_unavailable"
NO_MATCHING_PROFILE = "no_matching_profile"
INSUFFICIENT_DISK = "insufficient_disk"
PULL_FAILED = "pull_failed"
DIGEST_MISMATCH = "digest_mismatch"
UNKNOWN = "unknown"

#: The manifest carries no image sizes, so staging needs this much free space in
#: Docker's data root before it pulls. ROS driver images run to several GB each.
MIN_FREE_GB_ENV = "CYBERWAVE_DRIVER_STAGE_MIN_FREE_GB"
DEFAULT_MIN_FREE_GB = 8.0
#: Per image. The backend waits an hour between progress reports before it fails a
#: deployment, and a pull that is still moving keeps reporting.
PULL_TIMEOUT_SECONDS = 3600
PROGRESS_INTERVAL_SECONDS = 10.0
PULL_WORKERS = 2
_GIB = 1024**3


class StageError(Exception):
    """Staging failed for a reason the backend should record as *code*."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ManifestImage:
    image: str
    #: ``sha256:...`` from the manifest; None for an image the manifest names without one.
    digest: Optional[str] = None
    service: Optional[str] = None


@dataclass(frozen=True)
class StageRequest:
    request_id: str
    twin_uuid: str
    package: str
    target_version: str
    flavor: str
    manifest: Mapping[str, Any]


def parse_request(payload: Mapping[str, Any]) -> StageRequest:
    """The command as a request; raises ``StageError(INVALID_COMMAND)`` when it is unusable."""
    request_id = str(payload.get("request_id") or "").strip()
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("profiles"), Mapping):
        raise StageError(INVALID_COMMAND, "The stage command carries no manifest profiles")
    if not state.is_valid_request_id(request_id):
        raise StageError(INVALID_COMMAND, f"not a usable request_id: {request_id!r}")
    return StageRequest(
        request_id=request_id,
        twin_uuid=str(payload.get("twin_uuid") or ""),
        package=str(payload.get("package") or ""),
        target_version=str(payload.get("target_version") or ""),
        flavor=str(payload.get("flavor") or ""),
        manifest=manifest,
    )


def select_images(manifest: Mapping[str, Any]) -> tuple[str, str, list[ManifestImage]]:
    """``(profile, variant, images)`` this host stages from *manifest*.

    The profile is the first of this host's profile keys the manifest has (a Jetson gets
    ``linux-aarch64-jetson`` where a release publishes one), else ``default``. The variant
    is the one the edge starts drivers with. The images are the profile's primary image
    and every service image of that variant, each service with its digest.
    """
    from .driver_selection import _platform_driver_keys, _resolve_compose_variant

    profiles = manifest.get("profiles") or {}
    profile_name = next((key for key in _platform_driver_keys() if key in profiles), None)
    if profile_name is None and "default" in profiles:
        profile_name = "default"
    if profile_name is None:
        raise StageError(
            NO_MATCHING_PROFILE,
            f"The release has no profile for this host (it has: {', '.join(sorted(profiles))})",
        )
    profile = profiles[profile_name]
    variant = _resolve_compose_variant(None)[0]
    services = (profile.get("variants") or {}).get(variant)
    if not services:
        raise StageError(
            NO_MATCHING_PROFILE,
            f"The release's {profile_name!r} profile has no {variant!r} variant",
        )

    images: dict[str, ManifestImage] = {}
    primary = profile.get("primary")
    if primary:
        images[str(primary)] = ManifestImage(image=str(primary))
    for service in services:
        image = str(service["image"])
        images[image] = ManifestImage(
            image=image, digest=service.get("digest"), service=service.get("service")
        )
    return profile_name, variant, list(images.values())


# --- Docker ------------------------------------------------------------------------


def _local_digests(image: str) -> Optional[list[str]]:
    """The repo digests of *image* on this host; None when the image is not here."""
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return None
    try:
        return [str(digest) for digest in json.loads(result.stdout) or []]
    except ValueError:
        return []


def _is_verified(image: ManifestImage) -> bool:
    """On disk, and (when the manifest gives a digest) at exactly that digest."""
    digests = _local_digests(image.image)
    if digests is None:
        return False
    if image.digest is None:
        return True
    return any(digest.endswith(f"@{image.digest}") for digest in digests)


def _min_free_bytes() -> float:
    try:
        gigabytes = float(os.environ.get(MIN_FREE_GB_ENV, "") or DEFAULT_MIN_FREE_GB)
    except ValueError:
        gigabytes = DEFAULT_MIN_FREE_GB
    return gigabytes * _GIB


def ensure_free_space() -> None:
    """Make room before a pull: collect unused images if short, fail if still short."""
    from .image_gc import gc_driver_images

    root = docker_data_root()
    needed = _min_free_bytes()
    if shutil.disk_usage(root).free >= needed:
        return
    logger.info("Little free space under %s; collecting unused images before staging", root)
    gc_driver_images(aggressive=True)
    free = shutil.disk_usage(root).free
    if free < needed:
        raise StageError(
            INSUFFICIENT_DISK,
            f"{free / _GIB:.1f} GiB free under {root}, need {needed / _GIB:.1f} GiB to stage",
        )


# --- Reporting ---------------------------------------------------------------------


class StatusReporter:
    """``POST /api/v1/edges/{uuid}/driver-deployments/{request_id}/status``.

    ``send`` returns True when the backend has the report (or no longer wants it: a 404
    means the deployment was superseded, so retrying would never succeed) and False when
    it could not be delivered.
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.superseded = False
        self._edge_uuid: Optional[str] = None
        self._last_progress = 0.0

    def send(
        self,
        phase: str,
        *,
        progress: Optional[Mapping[str, int]] = None,
        error_code: Optional[str] = None,
        error: Optional[str] = None,
        observed: Optional[Mapping[str, str]] = None,
    ) -> bool:
        from . import startup

        body: dict[str, Any] = {"phase": phase}
        if progress is not None:
            body["progress"] = dict(progress)
        if observed is not None:
            body["observed"] = dict(observed)
        if error_code is not None:
            body["error_code"] = error_code
        if error is not None:
            body["error"] = error
        try:
            token = startup.load_token()
            if not token:
                return False
            client = startup._build_cyberwave_client(token)
            if self._edge_uuid is None:
                self._edge_uuid = startup._resolve_edge_uuid(token)
            if not self._edge_uuid:
                return False
            api = client.api.api_client
            param = api.param_serialize(
                method="POST",
                resource_path="/api/v1/edges/{uuid}/driver-deployments/{request_id}/status",
                path_params={"uuid": self._edge_uuid, "request_id": self.request_id},
                body=body,
                auth_settings=["CustomTokenAuthentication"],
            )
            response = api.call_api(*param)
            response.read()
            api.response_deserialize(response_data=response, response_types_map={"200": "object"})
            return True
        except Exception as exc:
            if getattr(exc, "status", None) == 404:
                logger.info("Deployment %s is gone on the backend; dropping it", self.request_id)
                self.superseded = True
                return True
            logger.warning("Could not report %s for %s: %s", phase, self.request_id, exc)
            return False

    def progress(
        self, done: int, total: int, bytes_done: int, bytes_total: int, *, force: bool = False
    ) -> None:
        """A ``staging`` report, at most every few seconds unless *force*; progress also keeps
        the backend's timeout away."""
        now = time.monotonic()
        if not force and now - self._last_progress < PROGRESS_INTERVAL_SECONDS:
            return
        self._last_progress = now
        self.send(
            "staging",
            progress={
                "images_done": done,
                "images_total": total,
                "bytes_done": bytes_done,
                "bytes_total": bytes_total,
            },
        )


# --- Staging -----------------------------------------------------------------------


def _pull_images(images: list[ManifestImage], reporter: StatusReporter, total: int) -> None:
    """Pull *images* (at most ``PULL_WORKERS`` at a time), reporting progress as bytes arrive."""
    from .driver_logs import _pull_docker_image_with_progress_multi
    from .startup import load_token

    token = load_token()
    lock = threading.Lock()
    seen: dict[str, tuple[int, int]] = {}
    finished = total - len(images)

    def report(force: bool = False) -> None:
        with lock:
            bytes_done = sum(done for done, _ in seen.values())
            bytes_total = sum(size for _, size in seen.values())
            done_images = finished
        reporter.progress(done_images, total, bytes_done, bytes_total, force=force)

    def pull(image: ManifestImage) -> None:
        nonlocal finished
        if reporter.superseded:
            return

        def on_progress(progress: Any) -> None:
            with lock:
                seen[image.image] = (progress.downloaded_bytes, progress.total_bytes)
            report()

        try:
            _pull_docker_image_with_progress_multi(
                image.image,
                contexts=[],
                token=token,
                timeout=PULL_TIMEOUT_SECONDS,
                on_progress=on_progress,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
            raise StageError(PULL_FAILED, f"Could not pull {image.image}: {exc}") from exc
        state.record_pull(image.image)
        with lock:
            finished += 1
        report(force=True)

    with ThreadPoolExecutor(max_workers=min(PULL_WORKERS, max(len(images), 1))) as pool:
        futures = [pool.submit(pull, image) for image in images]
        try:
            # In completion order, so the first failure is seen when it happens and not
            # only once the images ahead of it in the list have finished.
            for future in as_completed(futures):
                future.result()
        except BaseException:
            # One failed pull fails the staging: do not start the images still queued.
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def _describe(images: list[ManifestImage]) -> str:
    """``image@digest`` for each, for the log: which build, not just which tag."""
    return ", ".join(
        f"{image.image}@{image.digest[:19]}" if image.digest else image.image for image in images
    )


def _running_drivers_block(twin_uuid: str) -> Optional[dict[str, Any]]:
    """The ``drivers`` block of the twin's JSON on this edge, as of its last sync."""
    from . import startup

    data = startup._read_twin_json(twin_uuid) if twin_uuid else None
    metadata = data.get("metadata") if isinstance(data, dict) else None
    drivers = metadata.get("drivers") if isinstance(metadata, dict) else None
    return dict(drivers) if isinstance(drivers, dict) and drivers else None


def stage(request: StageRequest, reporter: StatusReporter) -> None:
    """Pull and verify the release's images, write the staged record, report ``staged``.

    Raises :class:`StageError`; the caller reports it as ``failed``.
    """
    if state.read_staged_record(request.request_id) is not None:
        logger.info("Driver update %s is already staged; reporting it again", request.request_id)
        return

    if not docker_available():
        raise StageError(DOCKER_UNAVAILABLE, "Docker is not available on this edge")
    profile, variant, images = select_images(request.manifest)
    total = len(images)
    logger.info(
        "Staging %s: %s %s for twin %s, profile %s, variant %s: %s",
        request.request_id,
        request.package,
        request.target_version,
        request.twin_uuid,
        profile,
        variant,
        _describe(images),
    )
    reporter.send(
        "staging",
        progress={"images_done": 0, "images_total": total, "bytes_done": 0, "bytes_total": 0},
    )

    # No staged record names these images until the end of this function, so hold them
    # against the periodic image cleanup: those already here as well as those to be pulled.
    if not all([state.record_pull(image.image) for image in images]):
        raise StageError(
            UNKNOWN, "Could not record the release's images as in use, so they are not safe to stage"
        )
    missing = [image for image in images if not _is_verified(image)]
    logger.info(
        "Staging %s: %d of %d image(s) already here, pulling %s",
        request.request_id,
        total - len(missing),
        total,
        _describe(missing) or "nothing",
    )
    if missing:
        ensure_free_space()
        _pull_images(missing, reporter, total)
    if reporter.superseded:
        raise StageError(UNKNOWN, "The deployment was replaced while it was being staged")

    for image in images:
        state.record_pull(image.image)
        if not _is_verified(image):
            raise StageError(
                DIGEST_MISMATCH,
                f"{image.image} is missing or is not the build the release names "
                f"({image.digest or 'no digest'})",
            )

    record = {
        "request_id": request.request_id,
        # What the twin is configured to run now, so a failed activation can start it
        # again: by then the backend has already written the new release over it.
        "previous_drivers": _running_drivers_block(request.twin_uuid),
        "twin_uuid": request.twin_uuid,
        "package": request.package,
        "target_version": request.target_version,
        "flavor": request.flavor,
        "profile": profile,
        "variant": variant,
        "placeholders": dict(request.manifest.get("placeholders") or {}),
        "images": [
            {"service": image.service, "image": image.image, "digest": image.digest}
            for image in images
        ],
        "staged_at": time.time(),
    }
    state.write_staged_record(record)
    logger.info(
        "Staged %s: %d image(s) verified and held for twin %s",
        request.request_id,
        total,
        request.twin_uuid,
    )
    # A newer deployment for the same twin and package replaces the older ones, which the
    # backend no longer tracks: let go of their images too.
    state.delete_superseded_records(record)


# --- The command -------------------------------------------------------------------

_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    """One worker: pulls are heavy, so deployments are staged one at a time."""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="driver-stage")
        return _EXECUTOR


def _run(request: StageRequest) -> None:
    from . import startup

    reporter = StatusReporter(request.request_id)
    delivered = False
    try:
        stage(request, reporter)
        delivered = reporter.send("staged") if not reporter.superseded else True
    except StageError as exc:
        logger.warning("Staging %s failed (%s): %s", request.request_id, exc.code, exc)
        delivered = reporter.send("failed", error_code=exc.code, error=str(exc))
    except Exception as exc:
        logger.exception("Staging %s failed unexpectedly", request.request_id)
        delivered = reporter.send("failed", error_code=UNKNOWN, error=f"{type(exc).__name__}: {exc}")
    if not delivered:
        # The backend still thinks this is pending and will send it again; let it through.
        startup.forget_edge_command(STAGE_COMMAND, request.request_id)


def _reject(request_id: str, error: StageError) -> None:
    StatusReporter(request_id).send("failed", error_code=error.code, error=str(error))


def handle_stage_command(payload: Mapping[str, Any]) -> None:
    """The ``stage_driver_update`` handler: validate, then stage on the worker thread."""
    try:
        request = parse_request(payload)
    except StageError as exc:
        logger.warning("Ignoring an unusable %s: %s", STAGE_COMMAND, exc)
        request_id = str(payload.get("request_id") or "").strip()
        if not state.is_valid_request_id(request_id):
            return  # nothing to report it against
        # Say so, or the backend keeps the deployment pending until it times out.
        _executor().submit(_reject, request_id, exc)
        return
    _executor().submit(_run, request)
