"""EdgeSyncClient — pull generated worker files from the backend edge-sync endpoint.

Responsibilities:
- Fetch the ``GET /api/v1/workflows/edge-sync/{twin_uuid}`` payload.
- Diff the returned ``wf_*.py`` files against what is on disk under
  ``{CONFIG_DIR}/workers/``.
- Write new/updated worker files atomically (write to a temp file, then
  rename).
- Remove ``wf_*.py`` files that are no longer in the sync payload (i.e.
  workflows that were deactivated).
- Content-identical writes are no-ops — the file mtime is not updated so the
  WorkerWatcher (CYB-1546) does not trigger a spurious restart.

Design notes:
- ``EdgeSyncClient.sync()`` writes/removes files and returns a
  :class:`EdgeSyncResult` summary.  It never starts or stops the worker
  container directly.
- The WorkerWatcher detects mtime changes on the next reconcile cycle and
  handles the full restart sequence.
- This module has no dependency on Django; it only needs the ``cyberwave``
  Python SDK for HTTP access.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EdgeSyncResult:
    """Summary of a single sync operation."""

    twin_uuid: str
    written: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        return (
            f"EdgeSyncResult(twin={self.twin_uuid}, "
            f"written={len(self.written)}, "
            f"removed={len(self.removed)}, "
            f"unchanged={len(self.unchanged)}, "
            f"errors={len(self.errors)})"
        )


# Prefix that identifies workflow-generated worker files on disk.
_WF_PREFIX = "wf_"


class EdgeSyncClient:
    """Sync generated worker files from the backend for a specific twin.

    Parameters
    ----------
    workers_dir:
        Path to the local workers directory (e.g. ``/etc/cyberwave/workers``).
        Created if it does not exist.
    base_url:
        Cyberwave API base URL.
    token:
        API auth token.
    """

    def __init__(
        self,
        workers_dir: str | Path,
        base_url: str,
        token: str,
    ) -> None:
        self._workers_dir = Path(workers_dir)
        self._base_url = base_url.rstrip("/")
        self._token = token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self, twin_uuid: str) -> EdgeSyncResult:
        """Fetch the edge-sync payload and reconcile worker files on disk.

        Returns an :class:`EdgeSyncResult` describing what changed.
        """
        result = EdgeSyncResult(twin_uuid=twin_uuid)

        try:
            payload = self._fetch_sync_payload(twin_uuid)
        except Exception as exc:
            msg = f"Failed to fetch edge-sync payload for twin {twin_uuid}: {exc}"
            logger.warning(msg)
            result.errors.append(msg)
            return result

        workflows: list[dict[str, Any]] = payload.get("workflows", [])

        # Build index of expected generated workers: filename → source.
        expected: dict[str, str] = {}
        for wf in workflows:
            filename: str | None = wf.get("worker_filename")
            source: str | None = wf.get("worker_source")
            if filename and source and filename.startswith(_WF_PREFIX):
                expected[filename] = source

        self._workers_dir.mkdir(parents=True, exist_ok=True)

        # Write new/updated files.
        for filename, source in expected.items():
            dest = self._workers_dir / filename
            if dest.exists():
                try:
                    existing = dest.read_text(encoding="utf-8")
                except OSError:
                    existing = None
                if existing == source:
                    result.unchanged.append(filename)
                    continue

            try:
                self._atomic_write(dest, source)
                result.written.append(filename)
                logger.info("Wrote generated worker: %s", filename)
            except OSError as exc:
                msg = f"Failed to write {filename}: {exc}"
                logger.error(msg)
                result.errors.append(msg)

        # Remove stale wf_*.py files that are no longer expected.
        for existing_file in sorted(self._workers_dir.glob(f"{_WF_PREFIX}*.py")):
            if existing_file.name not in expected:
                try:
                    existing_file.unlink()
                    result.removed.append(existing_file.name)
                    logger.info("Removed stale worker: %s", existing_file.name)
                except OSError as exc:
                    msg = f"Failed to remove {existing_file.name}: {exc}"
                    logger.error(msg)
                    result.errors.append(msg)

        logger.info(
            "Edge sync complete for twin %s: %s",
            twin_uuid,
            result,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_sync_payload(self, twin_uuid: str) -> dict[str, Any]:
        """Call the backend edge-sync endpoint and return the parsed payload."""
        try:
            import requests

            url = f"{self._base_url}/api/v1/workflows/edge-sync/{twin_uuid}"
            headers = {"Authorization": f"Bearer {self._token}"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except ImportError:
            pass

        # Fallback: use urllib (stdlib) when requests is not available.
        import json
        import urllib.error
        import urllib.request

        url = f"{self._base_url}/api/v1/workflows/edge-sync/{twin_uuid}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self._token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"HTTP {exc.code} from edge-sync endpoint: {exc.reason}"
            ) from exc

    @staticmethod
    def _atomic_write(dest: Path, content: str) -> None:
        """Write *content* to *dest* atomically using a sibling temp file."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(dest.parent),
            prefix=".tmp_",
            suffix=dest.suffix,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
