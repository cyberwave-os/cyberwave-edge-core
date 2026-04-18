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
        throttle_seconds: Optional[float] = None,
    ) -> None:
        self.twin_uuid = twin_uuid
        self.image = image
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
                "started_at": time.time(),
            }
            self._alert = twin.alerts.create(
                name="Driver starting",
                description=f"Downloading driver image {self.image} for the {self.twin_uuid} twin on the attached edge.",  # noqa: E501
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
                    logger.warning(
                        "Could not resolve stale driver_starting alert %s for twin %s: %s",
                        getattr(alert, "uuid", "<unknown>"),
                        twin_uuid,
                        exc,
                    )
        except Exception as exc:
            logger.debug(
                "Could not list driver_starting alerts for twin %s: %s",
                twin_uuid,
                exc,
                exc_info=True,
            )
        return resolved
