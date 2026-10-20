"""Shared helpers for edge-core (kept small to avoid growing ``startup.py``)."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from cyberwave import Cyberwave

from .docker_helpers import driver_container_name

DRIVER_STARTING_ALERT_TYPE = "driver_starting"
WORKER_STARTING_ALERT_TYPE = "worker_starting"
DRIVER_STARTING_ALERT_METADATA_UPDATE_INTERVAL_SECONDS = max(
    0.5,
    float(os.getenv("CYBERWAVE_DRIVER_STARTING_ALERT_UPDATE_INTERVAL_SECONDS", "2")),
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config-directory files
# ---------------------------------------------------------------------------

#: Edge-owned JSON files in the config directory: never twin JSON objects, never
#: removed or re-downloaded as one.
PROTECTED_CONFIG_JSON_FILES: frozenset[str] = frozenset(
    {"credentials.json", "fingerprint.json", "environment.json", "edge.json"}
)


def is_driver_twin_json_file(path: Path) -> bool:
    """Return True when *path* is a managed twin JSON object file (``<twin_uuid>.json``)."""
    if not path.is_file() or path.name in PROTECTED_CONFIG_JSON_FILES:
        return False

    try:
        uuid.UUID(path.stem)
        return True
    except ValueError:
        return False


def atomic_write_text(path: Path, content: str, *, mode: int = 0o600) -> None:
    """Atomically replace *path* with *content*, with permissions *mode*.

    Uses a sibling temp file, ``fsync`` and ``os.replace`` so readers never see
    a half-written file. On POSIX the file is ``chmod``-ed to *mode* **before**
    the rename, so it is never visible with looser permissions. The parent
    directory is created when missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if os.name != "nt":
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Startup progress helpers
# ---------------------------------------------------------------------------

#: Phases where edge-core is actively transferring image layers from the
#: registry.  ``installing`` is its own stage.
_STARTUP_PULLING_PHASES: frozenset[str] = frozenset({"pull_started", "pulling", "downloading"})


#: Band edges for :func:`_compute_overall_progress_percent`, in percent.
#: Sized against measured wall-clock on a multi-GB pull (CYB-3854): the
#: extraction tail runs at least as long as the download it follows, so the
#: two get comparable shares.  Container start is the short phase, but it is
#: reported in discrete steps rather than continuously, so it keeps enough
#: room for those steps to read as progress.
#:
#: The final band belongs to the worker process itself.  ``container_running``
#: used to return 100, putting the bar at "done" as the slowest phase began.
_DOWNLOAD_BAND_END = 40
_INSTALL_BAND_END = 70
_CONTAINER_BAND_END = 80

#: Progress *within the worker-startup stage* for each phase the worker
#: reports.  Mirrors the ``PHASE_*`` constants in the SDK's
#: ``cyberwave.workers.status``.  ``warming_up`` is interpolated from model
#: counts when the worker publishes them.
_WORKER_STAGE_PERCENTS: dict[str, int] = {
    "booting": 10,
    "connecting_mqtt": 25,
    "loading_models": 40,
    "subscribed": 65,
    "warming_up": 75,
    "ready": 100,
}
_WARM_UP_STAGE_START = 75


def _compute_stage_progress_percent(
    phase: str,
    download_percent: float,
    *,
    layers_complete: int = 0,
    layers_total: int = 0,
    models_ready: int = 0,
    models_total: int = 0,
) -> int | None:
    """Progress *within the current startup stage*, 0–100.

    Startup runs four stages — download, install, container start, and the
    worker's own boot inside the started container — and this gives each its
    own 0–100. The alert's phase copy names the stage on screen
    ("Downloading worker image", "Installing worker image", "Starting worker
    container"), so a stage that reads 100 % is one the user can check against
    what they are being told.

    This is published as ``stage_progress_percent``. It is deliberately *not*
    what the card's bar reads — see :func:`_compute_overall_progress_percent`.

    Returns ``None`` for phases with nothing meaningful to report (e.g.
    ``container_probe_unconfirmed``) so callers omit the field rather than
    publishing a stale value.
    """
    if phase == "installing":
        # Extraction is layer-serial and outlasts the download on a large
        # image; layer completion is the only signal that moves here.
        if layers_total > 0:
            fraction = max(0.0, min(1.0, layers_complete / layers_total))
            return round(fraction * 100)
        return 50
    if phase in _STARTUP_PULLING_PHASES or phase == "":
        return round(max(0.0, min(100.0, download_percent)))
    if phase in ("pull_complete", "pull_skipped", "pull_stream_finished"):
        return 100
    # Container start reports discrete steps rather than a continuous signal.
    if phase == "starting_container":
        return 33
    if phase == "container_created":
        return 66
    if phase == "container_running":
        return 100

    worker_stage = _WORKER_STAGE_PERCENTS.get(phase)
    if worker_stage is None:
        return None
    if phase == "warming_up" and models_total > 0:
        # Per-model completion, for the same reason ``installing`` tracks
        # layers: warm-up is serial and slow, and a number that stands still
        # through it reads as a hang.
        fraction = max(0.0, min(1.0, models_ready / models_total))
        span = 100 - _WARM_UP_STAGE_START
        # Capped below 100: only ``ready`` claims completion, and it is
        # published after the replay that follows the last model.
        return _WARM_UP_STAGE_START + min(span - 1, round(fraction * span))
    return worker_stage


def _compute_overall_progress_percent(
    phase: str,
    download_percent: float,
    *,
    layers_complete: int = 0,
    layers_total: int = 0,
    models_ready: int = 0,
    models_total: int = 0,
) -> int | None:
    """Lifecycle-wide 0–100, monotone across download → install → start.

    The workbench renders one bar with no stage segmentation and no ``key``
    between phases, so a stage that restarts its own 0–100 animates as a
    *rewind*: the fill slid from 83 % back to empty at the download → install
    hand-off, and again from full to a third when the container came up
    (CYB-3867).  Banding the per-stage number keeps the bar honest about the
    only question it is asked — is this moving? — while the phase copy above
    it still names the stage, and ``stage_progress_percent`` carries the
    per-stage figure for a stage-aware bar to use once one exists.

    The bands:

    * **Download** (0–40 %) — layer-byte progress from docker pull.
    * **Install**  (40–70 %) — decompression, driven by layer completion.
    * **Container** (70–80 %) — create → start → probe → running.
    * **Worker**   (80–100 %) — boot → connect → load models → warm up → ready.

    Sized by how long each stage actually takes, not by how many steps it has:
    extraction outlasts the bytes, and the worker's own startup outlasts the
    container start by orders of magnitude on a cold device.

    Returns ``None`` on the same phases :func:`_compute_stage_progress_percent`
    does.
    """
    stage = _compute_stage_progress_percent(
        phase,
        download_percent,
        layers_complete=layers_complete,
        layers_total=layers_total,
        models_ready=models_ready,
        models_total=models_total,
    )
    if stage is None:
        return None
    if phase == "installing":
        span = _INSTALL_BAND_END - _DOWNLOAD_BAND_END
        return _DOWNLOAD_BAND_END + round(stage * span / 100)
    if phase in _STARTUP_PULLING_PHASES or phase == "":
        return round(stage * _DOWNLOAD_BAND_END / 100)
    if phase in ("pull_complete", "pull_skipped", "pull_stream_finished"):
        return _INSTALL_BAND_END
    if phase in _WORKER_STAGE_PERCENTS:
        # The container is already up; this stage is the worker inside it.
        span = 100 - _CONTAINER_BAND_END
        overall = _CONTAINER_BAND_END + round(stage * span / 100)
        # Only ``ready`` claims completion: the band is narrow enough that
        # rounding the last warm-up step would otherwise land on 100.
        return overall if phase == "ready" else min(99, overall)
    return _INSTALL_BAND_END + round(
        stage * (_CONTAINER_BAND_END - _INSTALL_BAND_END) / 100
    )


def _coerce_layer_count(value: Any) -> int:
    """Read a non-negative count (layers, models) out of alert metadata.
    Anything else reads as 0, degrading the bar rather than raising."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _inject_progress_percentages(merged: dict[str, Any]) -> None:
    """Derive both startup percentages from *merged* alert metadata, in place.

    ``overall_progress_percent`` is what the card's bar reads;
    ``stage_progress_percent`` is the same startup expressed per stage.
    Either is omitted when the phase has nothing meaningful to report, so a
    stale value is never republished.
    """
    phase = str(merged.get("phase") or "")
    download_pct = merged.get("progress_percent")
    download = float(download_pct) if isinstance(download_pct, (int, float)) else 0.0
    complete = _coerce_layer_count(merged.get("layers_complete"))
    total = _coerce_layer_count(merged.get("layers_total"))

    models_ready = _coerce_layer_count(merged.get("models_ready"))
    models_total = _coerce_layer_count(merged.get("models_total"))

    overall = _compute_overall_progress_percent(
        phase,
        download,
        layers_complete=complete,
        layers_total=total,
        models_ready=models_ready,
        models_total=models_total,
    )
    if overall is not None:
        merged["overall_progress_percent"] = overall
    stage = _compute_stage_progress_percent(
        phase,
        download,
        layers_complete=complete,
        layers_total=total,
        models_ready=models_ready,
        models_total=models_total,
    )
    if stage is not None:
        merged["stage_progress_percent"] = stage


# ---------------------------------------------------------------------------
# User-friendly failure descriptions
# ---------------------------------------------------------------------------

_DRIVER_FAILED_DESCRIPTIONS: dict[str, str] = {
    # Container removal / creation / start failures (b/CYB-2331 and legacy)
    "container_remove_failed": "Could not restart your robot. Please try again.",
    "docker_create_failed": "Could not start your robot. Please try again.",
    "docker_run_failed": "Could not start your robot. Please try again.",
    "docker_start_failed": "Could not start your robot. Please try again.",
    "docker_create_timeout": "Your robot took too long to start. Please try again.",
    "docker_run_timeout": "Your robot took too long to start. Please try again.",
    "container_startup_timeout": "Your robot took too long to start. Please try again.",
    "container_unhealthy": "Your robot stopped unexpectedly. Please try again.",
    # Image pull failures
    "image_missing": "Robot software not found. Please contact support.",
    "pull_failed": "Could not download the robot software. Check your connection and try again.",
    "pull_failed_using_local": "Could not download the robot software. Check your connection and try again.",
    "pull_exit_error": "Could not download the robot software. Check your connection and try again.",
    "pull_timeout": "Download timed out. Check your connection and try again.",
    "pull_timeout_using_local": "Download timed out. Check your connection and try again.",
    "pull_timed_out": "Download timed out. Check your connection and try again.",
    "pull_spawn_failed": "Could not start the download. Please try again.",
    "pull_oserror": "Could not start the download. Please try again.",
    "pull_oserror_using_local": "Could not start the download. Please try again.",
    # macOS-specific
    "macos_bridge_failed": "Network setup failed. Please try again.",
    "pull_unexpected": "An unexpected error occurred. Please contact support.",
}

_WORKER_FAILED_DESCRIPTIONS: dict[str, str] = {
    "container_remove_failed": "Could not restart your worker. Please try again.",
    "docker_create_failed": "Could not start your worker. Please try again.",
    "docker_run_failed": "Could not start your worker. Please try again.",
    "docker_start_failed": "Could not start your worker. Please try again.",
    "docker_create_timeout": "Your worker took too long to start. Please try again.",
    "docker_run_timeout": "Your worker took too long to start. Please try again.",
    "container_startup_timeout": "Your worker took too long to start. Please try again.",
    "container_unhealthy": "Your worker stopped unexpectedly. Please try again.",
    "container_start_failed": "Could not start your worker. Please try again.",
    "image_missing": "Worker software not found. Please contact support.",
    "pull_failed": "Could not download the worker software. Check your connection and try again.",
    "pull_failed_using_local": "Could not download the worker software. Check your connection and try again.",
    "pull_exit_error": "Could not download the worker software. Check your connection and try again.",
    "pull_timeout": "Download timed out. Check your connection and try again.",
    "pull_timeout_using_local": "Download timed out. Check your connection and try again.",
    "pull_timed_out": "Download timed out. Check your connection and try again.",
    "pull_spawn_failed": "Could not start the download. Please try again.",
    "pull_oserror": "Could not start the download. Please try again.",
    "pull_oserror_using_local": "Could not start the download. Please try again.",
    "pull_unexpected": "An unexpected error occurred. Please contact support.",
    # From ``poll_startup_readiness``: the container started but the worker
    # inside never became usable. Distinct copy because the remedy differs —
    # the container is up and its logs are worth reading.
    "worker_ready_timeout": (
        "Your worker started but took too long to load its models. Please try again."
    ),
    "worker_startup_exited": "Your worker stopped while starting up. Please try again.",
}


class DriverStartingAlertContext:
    """Create/update/resolve a ``driver_starting`` twin alert around docker pull and pre-run.

    Resolves API base URL and API key like ``startup._send_alert_for_twin`` (via lazy imports).

    All API calls are best-effort; failures are logged and never block driver startup.
    """

    def __init__(
        self,
        *,
        twin_uuid: str,
        image: str,
        service_name: Optional[str] = None,
        throttle_seconds: Optional[float] = None,
    ) -> None:
        self.twin_uuid = twin_uuid
        self.image = image
        self.service_name = service_name.strip() if isinstance(service_name, str) else None
        self.container_name = driver_container_name(twin_uuid, self.service_name)
        self.throttle_seconds = (
            throttle_seconds
            if throttle_seconds is not None
            else DRIVER_STARTING_ALERT_METADATA_UPDATE_INTERVAL_SECONDS
        )
        self._alert: Any = None
        self._last_update_ts = 0.0

    def create(self) -> None:
        """POST a new active alert (``force=True`` avoids dedupe collapsing repeats)."""
        try:
            # Lazy import: ``startup`` imports this module; same resolution as
            # ``_send_alert_for_twin``.
            from .startup import DEFAULT_API_URL, get_runtime_env_var, load_token

            base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
            token = load_token()
            client = Cyberwave(base_url=base_url, api_key=token)
            twin = client.twin(twin_id=self.twin_uuid)
            initial_metadata: dict[str, Any] = {
                "phase": "pull_started",
                "image": self.image,
                "container_name": self.container_name,
                "service_name": self.service_name,
                "started_at": time.time(),
                "overall_progress_percent": 0,
                "stage_progress_percent": 0,
            }
            service_suffix = f" ({self.service_name})" if self.service_name else ""
            description = "Your robot is starting up. This may take a few minutes."
            self._alert = twin.alerts.create(
                name=f"Driver starting{service_suffix}",
                description=description,
                severity="info",
                alert_type=DRIVER_STARTING_ALERT_TYPE,
                source_type="edge",
                metadata=initial_metadata,
                force=True,
            )
        except Exception as exc:
            logger.warning(
                "Could not create driver_starting alert for twin %s: %s",
                self.twin_uuid,
                exc,
            )
            self._alert = None

    def update_metadata(self, patch: dict[str, Any], *, force: bool = False) -> None:
        """Merge *patch* into alert metadata, optionally throttled.

        Automatically injects ``overall_progress_percent`` (0–100, the
        lifecycle-wide number the bar reads) and ``stage_progress_percent``
        (the same startup expressed per stage), both derived from the merged
        ``phase`` and ``progress_percent`` values.
        """
        if not self._alert:
            return
        now = time.time()
        if not force and (now - self._last_update_ts) < self.throttle_seconds:
            return
        self._last_update_ts = now
        try:
            prev = self._alert.metadata if isinstance(self._alert.metadata, dict) else {}
            merged = {**prev, **patch}
            _inject_progress_percentages(merged)
            self._alert = self._alert.update(metadata=merged)
        except Exception as exc:
            logger.warning(
                "Could not update driver_starting alert for twin %s: %s",
                self.twin_uuid,
                exc,
                exc_info=True,
            )

    def resolve(self) -> None:
        if not self._alert:
            return
        try:
            self._alert.resolve()
        except Exception as exc:
            logger.warning(
                "Could not resolve driver_starting alert for twin %s: %s",
                self.twin_uuid,
                exc,
            )

    def mark_failed_and_resolve(self, description: str, *, phase: str = "pull_failed") -> None:
        """Annotate the alert with failure context, then resolve it.

        The "Driver starting" alert tracks an in-flight startup attempt and
        should not stay active after that attempt finishes — failure or
        success.  Higher-level driver-startup failures are surfaced via
        separate ``driver_start_failure`` alerts created by the caller, so
        leaving this alert active would be redundant and confusing
        (especially after an explicit ``restart edge-core`` request).

        The user-facing *description* is replaced with plain-language copy
        from ``_DRIVER_FAILED_DESCRIPTIONS`` keyed on *phase*; the original
        technical *description* is preserved in ``metadata.technical_detail``.
        """
        if not self._alert:
            return
        try:
            from cyberwave._error_metadata import redact

            # Fall back to the *redacted* detail, never the raw one: an
            # unmapped phase must not put unscrubbed daemon output on screen.
            redacted = redact(description)
            user_description = _DRIVER_FAILED_DESCRIPTIONS.get(phase, redacted)
            prev = self._alert.metadata if isinstance(self._alert.metadata, dict) else {}
            merged = {
                **prev,
                "phase": phase,
                "failed": True,
                "failed_at": time.time(),
                "technical_detail": redacted,
            }
            self._alert = self._alert.update(
                description=user_description,
                metadata=merged,
                severity="critical",
            )
        except Exception as exc:
            logger.warning(
                "Could not annotate driver_starting alert failure for twin %s: %s",
                self.twin_uuid,
                exc,
                exc_info=True,
            )
        self.resolve()

    @staticmethod
    def resolve_active_for_twin(twin_uuid: str) -> int:
        """Resolve any active ``driver_starting`` alerts for *twin_uuid*.

        Used to clear orphans left behind by interrupted driver-startup
        attempts (for example, before an ``edge-core`` restart wipes the
        in-process alert context).  Best-effort: failures are logged and
        never raised.  Returns the number of alerts resolved.

        A ``404 Not Found`` from the backend means the twin (and therefore
        any of its alerts) is already gone — typically because it was
        deleted on the backend while the edge still had stale local state
        referencing it.  In that case there is nothing to clear, so we log
        a single ``INFO`` line instead of a debug traceback.
        """
        return resolve_active_alerts_for_twin(twin_uuid, DRIVER_STARTING_ALERT_TYPE)


class WorkerStartingAlertContext:
    """Create/update/resolve a ``worker_starting`` twin alert around a worker image pull.

    Mirrors :class:`DriverStartingAlertContext` — same interface, same
    best-effort semantics — but uses the ``worker_starting`` alert type and
    worker-specific description copy, and deduplicates on create where the
    driver card forces a fresh row. See :meth:`create`.

    All API calls are best-effort; failures are logged and never block worker
    startup.
    """

    def __init__(
        self,
        *,
        twin_uuid: str,
        image: str,
        container_name: str,
        throttle_seconds: Optional[float] = None,
    ) -> None:
        self.twin_uuid = twin_uuid
        self.image = image
        self.container_name = container_name
        self.throttle_seconds = (
            throttle_seconds
            if throttle_seconds is not None
            else DRIVER_STARTING_ALERT_METADATA_UPDATE_INTERVAL_SECONDS
        )
        self._alert: Any = None
        self._last_update_ts: float = 0.0

    def create(self) -> None:
        """POST an active alert, deduped onto any card a past startup left open.

        Deliberately *not* ``force=True``. A card is only ever resolved by the
        process that opened it, and every close is best-effort, so an edge-core
        that is killed — or whose resolve call simply fails — strands one row
        per twin with no owner left to close it. Forcing a fresh row each time
        meant a device that restarts its worker often accrued a new stranded
        card per restart. Deduping instead re-uses the abandoned row, because
        the backend matches on ``status=ACTIVE`` only: a card that resolved
        cleanly is no longer a candidate and the next startup gets a new row,
        while one left hanging is adopted and its ``dedupe_count`` bumped.
        """
        try:
            from .startup import DEFAULT_API_URL, get_runtime_env_var, load_token

            base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
            token = load_token()
            client = Cyberwave(base_url=base_url, api_key=token)
            twin = client.twin(twin_id=self.twin_uuid)
            baseline = {
                "phase": "pull_started",
                "image": self.image,
                "container_name": self.container_name,
                "started_at": time.time(),
                "overall_progress_percent": 0,
                "stage_progress_percent": 0,
            }
            self._alert = twin.alerts.create(
                name="Worker starting",
                description="Your worker is starting up. This may take a few minutes.",
                severity="info",
                alert_type=WORKER_STARTING_ALERT_TYPE,
                source_type="edge",
                metadata=baseline,
            )
            self._reset_adopted_card(baseline)
        except Exception as exc:
            logger.warning(
                "Could not create worker_starting alert for twin %s: %s",
                self.twin_uuid,
                exc,
            )
            self._alert = None

    def _reset_adopted_card(self, baseline: dict[str, Any]) -> None:
        """Overwrite a re-used card's metadata with *baseline*.

        The backend's dedupe branch returns the existing row as-is: it bumps
        ``dedupe_count`` and ``last_seen_at`` but never applies the posted
        metadata, and deliberately leaves ``updated_at`` alone so a 10 fps
        publisher cannot thrash it. Both matter to an adopted card. It would
        otherwise report the abandoned startup's phase and byte counts, and
        ``updated_at`` is the field the backend's staleness window reads to
        decide whether a device is still coming up — so an adopted row would
        look dead on arrival and let the execution sweeper fail a run that is
        genuinely starting. One plain write fixes both.
        """
        alert = self._alert
        if alert is None:
            return
        current = alert.metadata if isinstance(alert.metadata, dict) else {}
        if current.get("started_at") == baseline["started_at"]:
            return  # Fresh row — the POST already stored this baseline.
        try:
            self._alert = alert.update(metadata=baseline)
        except Exception as exc:
            logger.warning(
                "Could not re-arm adopted worker_starting alert for twin %s: %s",
                self.twin_uuid,
                exc,
            )

    def update_metadata(self, patch: dict[str, Any], *, force: bool = False) -> None:
        """Merge *patch* into alert metadata, optionally throttled.

        Automatically injects ``overall_progress_percent`` and
        ``stage_progress_percent`` derived from the merged ``phase`` and
        ``progress_percent`` values.
        """
        if self._alert is None:
            return
        now = time.time()
        if not force and (now - self._last_update_ts) < self.throttle_seconds:
            return
        self._last_update_ts = now
        try:
            prev = self._alert.metadata if isinstance(self._alert.metadata, dict) else {}
            merged = {**prev, **patch}
            _inject_progress_percentages(merged)
            self._alert = self._alert.update(metadata=merged)
        except Exception as exc:
            logger.warning(
                "Could not update worker_starting alert for twin %s: %s",
                self.twin_uuid,
                exc,
                exc_info=True,
            )

    def resolve(self) -> None:
        """Resolve the alert (pull finished / container up)."""
        if self._alert is None:
            return
        try:
            self._alert.resolve()
        except Exception as exc:
            logger.warning(
                "Could not resolve worker_starting alert for twin %s: %s",
                self.twin_uuid,
                exc,
            )
        self._alert = None

    def mark_failed_and_resolve(self, description: str, *, phase: str = "pull_failed") -> None:
        """Annotate the alert with failure context, then resolve it.

        Replaces the technical *description* with plain-language copy from
        ``_WORKER_FAILED_DESCRIPTIONS``; the original is kept in
        ``metadata.technical_detail``.
        """
        if self._alert is None:
            return
        try:
            from cyberwave._error_metadata import redact

            # Fall back to the *redacted* detail, never the raw one: an
            # unmapped phase must not put unscrubbed daemon output on screen.
            redacted = redact(description)
            user_description = _WORKER_FAILED_DESCRIPTIONS.get(phase, redacted)
            prev = self._alert.metadata if isinstance(self._alert.metadata, dict) else {}
            merged = {
                **prev,
                "phase": phase,
                "failed": True,
                "failed_at": time.time(),
                "technical_detail": redacted,
            }
            self._alert = self._alert.update(
                description=user_description,
                metadata=merged,
                severity="critical",
            )
        except Exception as exc:
            logger.warning(
                "Could not annotate worker_starting alert failure for twin %s: %s",
                self.twin_uuid,
                exc,
                exc_info=True,
            )
        self.resolve()

    @staticmethod
    def resolve_active_for_twin(twin_uuid: str) -> int:
        """Resolve any active ``worker_starting`` alerts for *twin_uuid*.

        Used to clear orphans left behind by interrupted worker image pulls
        (for example, before an ``edge-core`` restart wipes the in-process
        alert context).  Best-effort: failures are logged and never raised.
        Returns the number of alerts resolved.

        A ``404 Not Found`` from the backend means the twin is already gone —
        nothing to clear; logs a single INFO line instead of a traceback.
        """
        return resolve_active_alerts_for_twin(twin_uuid, WORKER_STARTING_ALERT_TYPE)



def _is_not_found_error(exc: BaseException) -> bool:
    """Return True if *exc* represents an HTTP 404 from the Cyberwave API.

    Robust to both the wrapped ``CyberwaveAPIError`` (which exposes
    ``status_code``) and the underlying ``ApiException`` family from the
    generated REST client (which uses ``status``).
    """
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if value is None:
            continue
        try:
            if int(value) == 404:
                return True
        except (TypeError, ValueError):
            continue
    return False


def resolve_active_alerts_for_twin(twin_uuid: str, alert_type: str, *, strict: bool = False) -> int:
    """Resolve active *alert_type* alerts for *twin_uuid*; best-effort, returns count resolved.

    Shared by the alert-context classes and by callers clearing a persistent
    failure alert (e.g. ``driver_start_failure``) once a later pull/start
    attempt succeeds, since the retry/revival loops never do so themselves.
    With *strict*, a failed list or resolve raises so the caller can retry.
    """
    from .startup import DEFAULT_API_URL, get_runtime_env_var, load_token

    token = load_token()
    if not token:
        return 0

    base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL

    resolved = 0
    failed = False
    try:
        client = Cyberwave(base_url=base_url, api_key=token)
        twin = client.twin(twin_id=twin_uuid)
        for alert in twin.alerts.list(status="active", limit=100):
            if getattr(alert, "alert_type", None) != alert_type:
                continue
            try:
                alert.resolve()
                resolved += 1
            except Exception as exc:
                if _is_not_found_error(exc):
                    # Alert was concurrently deleted/resolved; nothing to do.
                    continue
                failed = True
                logger.warning(
                    "Could not resolve stale %s alert %s for twin %s: %s",
                    alert_type,
                    getattr(alert, "uuid", "<unknown>"),
                    twin_uuid,
                    exc,
                )
    except Exception as exc:
        if _is_not_found_error(exc):
            logger.info(
                "Skipping %s alert cleanup for twin %s: "
                "twin no longer exists on backend (404)",
                alert_type,
                twin_uuid,
            )
            return 0
        failed = True
        logger.debug(
            "Could not list %s alerts for twin %s: %s",
            alert_type,
            twin_uuid,
            exc,
            exc_info=True,
        )
    if strict and failed:
        raise RuntimeError(f"Could not resolve {alert_type} alerts for twin {twin_uuid}")
    return resolved


# ---------------------------------------------------------------------------
# Edge-core restart alert lifecycle
# ---------------------------------------------------------------------------

#: Phase strings recorded in ``Alert.metadata['phase']`` for the
#: ``edge_core_restart`` lifecycle alert.  Mirrors the constants in
#: ``src.app.api.edge_nodes`` on the backend — kept in lock-step via the
#: integration tests, not via a shared module (the edge SDK must not
#: import backend code).
EDGE_CORE_RESTART_PHASE_IN_PROGRESS = "in_progress"
EDGE_CORE_RESTART_PHASE_COMPLETED = "completed"
EDGE_CORE_RESTART_PHASE_FAILED = "failed"


class EdgeCoreRestartAlertContext:
    """Patch the backend-created ``edge_core_restart`` alert through its
    lifecycle (``in_progress`` → ``completed`` / ``failed``).

    Unlike :class:`DriverStartingAlertContext`, this class does **not**
    create the alert — the backend ``POST /api/v1/edges/{uuid}/restart-
    core`` endpoint does that and ships the UUID inside the MQTT command
    payload.  This context is purely an updater: given that UUID, it
    patches ``metadata.phase`` (preserving the existing metadata) and
    optionally resolves the alert on terminal phases.

    All HTTP calls are best-effort.  Failure to update the alert never
    blocks the actual restart — telemetry is nice, but a successful
    restart whose alert is stuck in ``in_progress`` is strictly better
    than a refused restart because the SDK round-trip flaked.  The
    reaper on the backend will time out the alert eventually anyway.

    Instantiated with ``alert_uuid=None`` it degrades to a no-op, which
    is the right behaviour when the restart was triggered by something
    other than the backend (CLI directly publishing MQTT, dev shell,
    test harness) and there is no alert to update.
    """

    def __init__(self, *, alert_uuid: Optional[str]) -> None:
        self.alert_uuid = alert_uuid or None

    def _fetch_alert(self) -> Any:
        """Build the SDK client and re-fetch the alert (best-effort).

        Re-fetching on every transition is wasteful in theory but the
        restart loop runs maybe twice per cycle (in_progress + final
        phase) and re-reads catch any concurrent backend mutation —
        including the reaper having flipped us to ``timed_out`` while
        the restart was running.  Returning a fresh ``Alert`` instance
        also lets the caller compose ``.update().resolve()`` without
        carrying the SDK wrapper across method boundaries.

        The ``startup`` import is lazy because ``startup`` imports this
        module — same circular-resolution pattern as
        :class:`DriverStartingAlertContext`.
        """
        from cyberwave.alerts import Alert, _get_alert

        from .startup import DEFAULT_API_URL, get_runtime_env_var, load_token

        base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
        token = load_token()
        if not token:
            return None
        client = Cyberwave(base_url=base_url, api_key=token)
        data = _get_alert(client, self.alert_uuid)
        return Alert(client, data)

    def transition(
        self,
        phase: str,
        *,
        resolve: bool,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Set ``metadata.phase = phase`` and optionally resolve the alert.

        ``extra_metadata`` is shallow-merged on top of the existing
        metadata so the audit trail keeps growing rather than being
        overwritten by each transition.
        """
        if not self.alert_uuid:
            return
        try:
            alert = self._fetch_alert()
            if alert is None:
                return
            metadata = alert.metadata if isinstance(alert.metadata, dict) else {}
            merged = {**metadata, "phase": phase}
            if extra_metadata:
                merged.update(extra_metadata)
            alert.update(metadata=merged)
            if resolve:
                alert.resolve()
        except Exception as exc:
            if _is_not_found_error(exc):
                # Alert was removed (operator deleted it, or the reaper
                # raced us and concluded ``timed_out``).  Nothing to do.
                logger.debug(
                    "edge_core_restart alert %s no longer exists; skipping %s transition",
                    self.alert_uuid,
                    phase,
                )
                return
            logger.warning(
                "Could not transition edge_core_restart alert %s to %s: %s",
                self.alert_uuid,
                phase,
                exc,
            )
