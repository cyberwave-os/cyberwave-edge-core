"""Shared helpers for edge-core (kept small to avoid growing ``startup.py``)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from cyberwave import Cyberwave

DRIVER_STARTING_ALERT_TYPE = "driver_starting"
DRIVER_STARTING_ALERT_METADATA_UPDATE_INTERVAL_SECONDS = max(
    0.5,
    float(os.getenv("CYBERWAVE_DRIVER_STARTING_ALERT_UPDATE_INTERVAL_SECONDS", "2")),
)

logger = logging.getLogger(__name__)


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
        if self.service_name:
            self.container_name = f"cyberwave-driver-{twin_uuid[:8]}-{self.service_name}"
        else:
            self.container_name = f"cyberwave-driver-{twin_uuid[:8]}"
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
            }
            service_suffix = f" ({self.service_name})" if self.service_name else ""
            if self.service_name:
                description = (
                    f"Downloading driver image {self.image} for service '{self.service_name}' "
                    f"on the {self.twin_uuid} twin on the attached edge."
                )
            else:
                description = (
                    f"Downloading driver image {self.image} for the {self.twin_uuid} twin "
                    "on the attached edge."
                )
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
        """Merge *patch* into alert metadata, optionally throttled."""
        if not self._alert:
            return
        now = time.time()
        if not force and (now - self._last_update_ts) < self.throttle_seconds:
            return
        self._last_update_ts = now
        try:
            prev = self._alert.metadata if isinstance(self._alert.metadata, dict) else {}
            merged = {**prev, **patch}
            self._alert = self._alert.update(metadata=merged)
        except Exception as exc:
            logger.debug(
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
        """
        if not self._alert:
            return
        try:
            prev = self._alert.metadata if isinstance(self._alert.metadata, dict) else {}
            merged = {
                **prev,
                "phase": phase,
                "failed": True,
                "failed_at": time.time(),
            }
            self._alert = self._alert.update(
                description=description,
                metadata=merged,
                severity="warning",
            )
        except Exception as exc:
            logger.debug(
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
        from .startup import DEFAULT_API_URL, get_runtime_env_var, load_token

        token = load_token()
        if not token:
            return 0

        base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL

        resolved = 0
        try:
            client = Cyberwave(base_url=base_url, api_key=token)
            twin = client.twin(twin_id=twin_uuid)
            for alert in twin.alerts.list(status="active", limit=100):
                if getattr(alert, "alert_type", None) != DRIVER_STARTING_ALERT_TYPE:
                    continue
                try:
                    alert.resolve()
                    resolved += 1
                except Exception as exc:
                    if _is_not_found_error(exc):
                        # Alert was concurrently deleted/resolved; nothing to do.
                        continue
                    logger.warning(
                        "Could not resolve stale driver_starting alert %s for twin %s: %s",
                        getattr(alert, "uuid", "<unknown>"),
                        twin_uuid,
                        exc,
                    )
        except Exception as exc:
            if _is_not_found_error(exc):
                logger.info(
                    "Skipping driver_starting alert cleanup for twin %s: "
                    "twin no longer exists on backend (404)",
                    twin_uuid,
                )
                return 0
            logger.debug(
                "Could not list driver_starting alerts for twin %s: %s",
                twin_uuid,
                exc,
                exc_info=True,
            )
        return resolved


WORKER_STARTING_ALERT_TYPE = "worker_starting"


class WorkerStartingAlertContext:
    """Create/update/resolve a ``worker_starting`` twin alert around worker image pull.

    Analogous to :class:`DriverStartingAlertContext` but for the shared ML
    worker container.  The key difference is that the caller supplies an
    explicit *container_name* (the worker name is derived from the environment
    UUID, not the twin UUID).

    All API calls are best-effort; failures are logged and never block startup.
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
        """POST a new active ``worker_starting`` alert."""
        try:
            from .startup import DEFAULT_API_URL, get_runtime_env_var, load_token

            base_url = get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL
            token = load_token()
            client = Cyberwave(base_url=base_url, api_key=token)
            twin = client.twin(twin_id=self.twin_uuid)
            self._alert = twin.alerts.create(
                name="Worker starting",
                description=(
                    f"Downloading ML worker image {self.image} for the edge attached to "
                    f"twin {self.twin_uuid}."
                ),
                severity="info",
                alert_type=WORKER_STARTING_ALERT_TYPE,
                source_type="edge",
                metadata={
                    "phase": "pull_started",
                    "image": self.image,
                    "container_name": self.container_name,
                    "started_at": time.time(),
                },
                force=True,
            )
        except Exception as exc:
            logger.warning(
                "Could not create worker_starting alert for twin %s: %s",
                self.twin_uuid,
                exc,
            )
            self._alert = None

    def update_metadata(self, patch: dict[str, Any], *, force: bool = False) -> None:
        """Best-effort metadata patch (e.g. pull progress bytes), throttled.

        Mirrors :meth:`DriverStartingAlertContext.update_metadata` so this
        class is a drop-in duck-type replacement when used by ``driver_logs``.
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
            self._alert = self._alert.update(metadata=merged)
        except Exception as exc:
            logger.debug(
                "Could not update worker_starting alert for twin %s: %s",
                self.twin_uuid,
                exc,
            )

    def resolve(self) -> None:
        """Resolve the alert (pull finished / container up)."""
        if self._alert is None:
            return
        try:
            self._alert.resolve()
        except Exception as exc:
            logger.debug(
                "Could not resolve worker_starting alert for twin %s: %s",
                self.twin_uuid,
                exc,
            )
        self._alert = None

    def mark_failed_and_resolve(self, description: str) -> None:
        """Mark the alert as failed and resolve it."""
        if self._alert is None:
            return
        try:
            prev = self._alert.metadata if isinstance(self._alert.metadata, dict) else {}
            self._alert = self._alert.update(
                description=description,
                metadata={
                    **prev,
                    "phase": "pull_failed",
                    "failed": True,
                    "failed_at": time.time(),
                },
                severity="warning",
            )
        except Exception as exc:
            logger.debug(
                "Could not annotate worker_starting alert failure for twin %s: %s",
                self.twin_uuid,
                exc,
            )
        self.resolve()


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
