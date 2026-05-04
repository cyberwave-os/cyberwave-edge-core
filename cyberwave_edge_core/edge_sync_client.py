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
        previously_missing: set[str] | None = None,
    ) -> None:
        self._workers_dir = Path(workers_dir)
        self._base_url = base_url.rstrip("/")
        self._token = token
        # Two-strikes cleanup state — see :meth:`_cleanup_stale`. The
        # caller owns the set so the strike count survives across
        # separate ``sync_all`` invocations even when a fresh
        # ``EdgeSyncClient`` is constructed each time (which is how
        # ``cyberwave_edge_core.startup._sync_workers_for_twins``
        # uses this class). Defaults to a fresh empty set for one-shot
        # use (tests, manual ``cleanup_stale`` calls) so files get one
        # strike of grace before being removed.
        self._previously_missing: set[str] = (
            previously_missing if previously_missing is not None else set()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self, twin_uuid: str) -> EdgeSyncResult:
        """Fetch the edge-sync payload and write worker files for one twin.

        This only writes/updates files — it does **not** delete stale files.
        Use :meth:`sync_all` when syncing multiple twins to get correct
        cleanup, or call :meth:`cleanup_stale` after all per-twin syncs.

        Returns an :class:`EdgeSyncResult` describing what changed.
        """
        result = EdgeSyncResult(twin_uuid=twin_uuid)
        self._sync_one(twin_uuid, result)
        return result

    def sync_all(self, twin_uuids: list[str]) -> list[EdgeSyncResult]:
        """Sync worker files for all *twin_uuids* and remove stale files.

        This is the preferred entry-point when multiple twins share the same
        ``workers_dir``.  Files are only deleted after **every** twin's
        payload has been fetched, preventing twin A's sync from removing
        twin B's workers.
        """
        all_expected: dict[str, str] = {}
        results: list[EdgeSyncResult] = []

        for twin_uuid in twin_uuids:
            result = EdgeSyncResult(twin_uuid=twin_uuid)
            expected = self._sync_one(twin_uuid, result)
            all_expected.update(expected)
            results.append(result)

        # Remove stale wf_*.py files not claimed by any twin.
        cleanup_result = EdgeSyncResult(twin_uuid="__cleanup__")
        self._cleanup_stale(all_expected, cleanup_result)
        if cleanup_result.removed or cleanup_result.errors:
            for r in results:
                r.removed.extend(cleanup_result.removed)
                r.errors.extend(cleanup_result.errors)
                break  # attribute removals to the first twin result

        return results

    def cleanup_stale(self, expected_filenames: set[str]) -> list[str]:
        """Remove wf_*.py files not in *expected_filenames*. Returns removed names."""
        result = EdgeSyncResult(twin_uuid="__cleanup__")
        self._cleanup_stale(
            {f: "" for f in expected_filenames},
            result,
        )
        return result.removed

    # ------------------------------------------------------------------
    # Internal sync helpers
    # ------------------------------------------------------------------

    def _sync_one(
        self, twin_uuid: str, result: EdgeSyncResult
    ) -> dict[str, str]:
        """Fetch payload for *twin_uuid*, write files, return expected map."""
        try:
            payload = self._fetch_sync_payload(twin_uuid)
        except Exception as exc:
            msg = f"Failed to fetch edge-sync payload for twin {twin_uuid}: {exc}"
            logger.warning(msg)
            result.errors.append(msg)
            return {}

        workflows: list[dict[str, Any]] = payload.get("workflows", [])

        expected: dict[str, str] = {}
        for wf in workflows:
            filename: str | None = wf.get("worker_filename")
            source: str | None = wf.get("worker_source")
            if filename and source and filename.startswith(_WF_PREFIX):
                expected[filename] = source

        self._workers_dir.mkdir(parents=True, exist_ok=True)

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

        logger.info("Edge sync complete for twin %s: %s", twin_uuid, result)
        return expected

    def _cleanup_stale(
        self, expected: dict[str, str], result: EdgeSyncResult
    ) -> None:
        """Remove wf_*.py files that haven't been claimed by the cloud
        for two consecutive syncs.

        **Two-strikes rule.** A file present on disk but missing from
        the latest sync payload is *marked* on the first such sync
        (logged at WARNING) and only deleted on the second consecutive
        sync where it's still missing. Any sync that re-claims the
        file resets its strike count.

        Why: the cloud's ``/workflows/edge-sync`` endpoint silently
        drops workflows that don't compile cleanly to edge — which can
        happen for a single tick while the operator is saving an
        intermediate workflow state in the editor (e.g. a connection
        is briefly broken, ``run_on_edge`` flickers, ``twin_uuid`` is
        empty between saves). A fail-fast cleanup would unlink every
        local worker on that single bad sync, leaving the edge worker
        with nothing to load until the next ~5-minute sync rewrites
        the file. The two-strikes rule keeps a single bad response
        from triggering wholesale deletion, at the cost of a one-cycle
        delay (~5 min in production) before deactivated workflows are
        actually removed from disk — an acceptable trade.

        Strike state lives on ``self._previously_missing`` and is
        owned by the caller (see :meth:`__init__`). After an edge-core
        process restart the set is empty by design — every existing
        ``wf_*.py`` gets one fresh strike of grace, matching the
        "be conservative on cold start" intuition.
        """
        self._workers_dir.mkdir(parents=True, exist_ok=True)
        expected_filenames = set(expected.keys())

        # Files reclaimed by this sync get their strike count reset.
        self._previously_missing.difference_update(expected_filenames)

        for existing_file in sorted(self._workers_dir.glob(f"{_WF_PREFIX}*.py")):
            name = existing_file.name
            if name in expected_filenames:
                continue

            if name not in self._previously_missing:
                # First strike — keep on disk and give the cloud one
                # more sync to claim it. If this is a transient blip
                # (editor save mid-flight, etc.) the next sync will
                # re-claim the file and the strike resets.
                self._previously_missing.add(name)
                logger.warning(
                    "Worker %s missing from edge-sync response; "
                    "keeping for one more sync (transient-state guard)",
                    name,
                )
                continue

            # Second consecutive miss — the cloud has consistently
            # said this worker should not exist. Safe to delete.
            try:
                existing_file.unlink()
                self._previously_missing.discard(name)
                result.removed.append(name)
                logger.info(
                    "Removed stale worker: %s (missing from 2 consecutive syncs)",
                    name,
                )
            except OSError as exc:
                msg = f"Failed to remove {name}: {exc}"
                logger.error(msg)
                result.errors.append(msg)

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
            os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
