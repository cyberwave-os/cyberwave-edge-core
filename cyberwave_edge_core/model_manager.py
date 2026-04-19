"""Model Manager — download, cache, and sync ML model weights for Edge Core.

Responsibilities:

* Resolve required models from worker files (regex scan for ``cw.models.load(...)``).
* Reconcile any pre-staged weights on disk into the local manifest (air-gap
  friendly: an operator can drop ``~/.cyberwave/models/{model_id}/weights.pt``
  on a USB stick and Edge Core will pick it up).
* Download model artifacts from the Cyberwave catalog API and store in a local
  cache directory. The default cache lives under the resolved edge config
  directory (``~/.cyberwave/models/`` on every platform; overridable via
  ``CYBERWAVE_EDGE_CONFIG_DIR``).
* Validate checksums and reuse existing artifacts on cache hit.
* Expose helper for listing and evicting cached models.

Resolution order for ``ensure_model(model_id)``
-----------------------------------------------

1. Reconcile any pre-staged weight directory at ``cache_dir/{model_id}/`` into
   the manifest, computing a checksum if no sidecar is present.
2. If the local file is intact (checksum matches what we last wrote), probe
   the catalog (best-effort, non-fatal) for a newer checksum.

   * If the catalog is unreachable (air-gap, transient network failure) →
     return the cached file as-is.
   * If the catalog checksum matches the local file → return the cached file.
   * If the catalog checksum differs → fall through to download.

3. Download the model. Sources are tried in priority order:

   #. **Cyberwave-hosted signed URL** from ``GET /api/v1/mlmodels/{uuid}/weights``
      — present when we have uploaded a checkpoint (e.g. an internally trained
      or mirrored model) to our private GCS bucket. This is the preferred
      source because it is authenticated and served from infrastructure we
      control.
   #. **Upstream weights URL** from the catalog entry (``download_url`` /
      ``metadata.download_url`` / ``metadata.artifact_url``). Used for public
      community checkpoints (e.g. official Ultralytics releases) that we did
      not mirror.

   The first source that yields a checksum-verified download wins.

4. If every download source fails *and* the local file is intact, return the
   stale-but-intact cached file with a warning. This is the air-gap and
   "intermittent network" fail-soft path.

5. Otherwise raise ``RuntimeError``.

The cache directory is bind-mounted read-only into the worker container at
``/app/models/``. The SDK's ``cw.models.load()`` resolves paths from that
mount — Edge Core is responsible for pre-populating it before starting the
container.

This module is Edge Core-internal and intentionally has **no** user-facing
CLI surface of its own. The ``WorkerManager`` and the startup bootstrap
call into it.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / tunables
# ---------------------------------------------------------------------------

#: Maximum number of download retry attempts on transient failures.
MAX_DOWNLOAD_RETRIES = 3

#: Seconds to wait between download retry attempts (exponential base).
DOWNLOAD_RETRY_BASE_DELAY = 2.0

#: Chunk size for streaming HTTP downloads (bytes).
DOWNLOAD_CHUNK_SIZE = 64 * 1024  # 64 KiB

#: API endpoint for fetching a model catalog entry.
ML_MODELS_ENDPOINT = "/api/v1/mlmodels"

#: HTTP timeout (seconds) for the *best-effort* catalog refresh probe used by
#: the warm-cache fast path. Kept short so that an unreachable backend does
#: not stall worker startup; the in-cache file is used on timeout.
CATALOG_PROBE_TIMEOUT = 5.0

#: HTTP timeout (seconds) for the *strict* catalog fetch used immediately
#: before a download. Longer because we are about to download anyway.
CATALOG_FETCH_TIMEOUT = 30.0

#: Name of the per-model metadata sidecar inside a model sub-directory.
MODEL_METADATA_FILENAME = "metadata.json"

#: Name of the global cache manifest that indexes all cached models.
MANIFEST_FILENAME = "manifest.json"

#: Sidecar ``downloaded_from`` value for files that came from the Cyberwave
#: signed-URL endpoint (``/mlmodels/{uuid}/weights``).
SOURCE_KIND_ARTIFACT = "artifact_url"

#: Sidecar ``downloaded_from`` value for files that came from the upstream
#: ``download_url`` in the catalog entry.
SOURCE_KIND_UPSTREAM = "download_url"

#: Sidecar ``downloaded_from`` value for files that an operator pre-staged on
#: disk (no Edge Core download involved). Set during disk reconciliation.
SOURCE_KIND_PRESTAGED = "prestaged"

#: Per-model lock filename (``cache_dir/{model_id}/.lock``). Held with
#: :func:`fcntl.flock` for the duration of :meth:`ModelManager.ensure_model`
#: so that two callers (e.g. worker startup racing the watcher loop) cannot
#: clobber each other's downloads.
MODEL_LOCK_FILENAME = ".lock"

#: Regex that matches ``cw.models.load("model-id")`` or
#: ``cw.models.load('model-id')`` calls in worker Python source files.
_CW_MODELS_LOAD_RE = re.compile(
    r"""cw\.models\.load\s*\(\s*['"]([^'"]+)['"]\s*""",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CachedModel:
    """Describes a single model artifact stored in the local cache.

    Fields
    ------
    source_url:
        For artifact downloads this stores the *resolver endpoint*
        (``{base_url}/api/v1/mlmodels/{uuid}/weights``), not the
        time-limited signed URL we actually fetched. The signed URL is
        useless for re-fetching and would mislead anyone reading the
        manifest. For upstream downloads it is the public ``download_url``.
        ``None`` for pre-staged files.
    pinned:
        When ``True``, Edge Core treats this entry as operator-curated
        truth: catalog refresh probes are skipped and download paths
        will not overwrite the file. Pin is meaningful only for
        ``downloaded_from == prestaged`` sidecars; opt in by
        hand-writing ``"pinned": true`` in ``metadata.json``.
    mtime_ns:
        File modification time (nanoseconds since epoch) recorded the
        last time we hashed or wrote this file. Used by
        :meth:`ModelManager._cache_is_intact` as a cheap-gate to skip
        re-hashing when ``stat()`` says nothing changed. ``None`` for
        legacy entries; on first verification it is back-filled.
    """

    model_id: str
    local_path: str
    size_bytes: int
    downloaded_at: str
    source_url: Optional[str] = None
    checksum_sha256: Optional[str] = None
    runtime: Optional[str] = None
    pinned: bool = False
    mtime_ns: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CachedModel":
        mtime_raw = data.get("mtime_ns")
        try:
            mtime_ns = int(mtime_raw) if mtime_raw is not None else None
        except (TypeError, ValueError):
            mtime_ns = None
        return cls(
            model_id=data["model_id"],
            local_path=data["local_path"],
            size_bytes=int(data.get("size_bytes", 0)),
            downloaded_at=data.get("downloaded_at", ""),
            source_url=data.get("source_url"),
            checksum_sha256=data.get("checksum_sha256"),
            runtime=data.get("runtime"),
            pinned=bool(data.get("pinned", False)),
            mtime_ns=mtime_ns,
        )


@dataclass
class _Manifest:
    """In-memory representation of ``manifest.json``."""

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "_Manifest":
        if not path.exists():
            return cls()
        try:
            with open(path) as fh:
                raw = json.load(fh)
            if not isinstance(raw, dict):
                logger.warning("manifest.json at %s is not a dict — resetting", path)
                return cls()
            return cls(entries=raw)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read manifest.json at %s: %s", path, exc)
            return cls()

    def save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w") as fh:
                json.dump(self.entries, fh, indent=2)
            tmp_path.replace(path)
        except OSError as exc:
            logger.warning("Failed to write manifest.json to %s: %s", path, exc)

    def get(self, model_id: str) -> Optional[CachedModel]:
        entry = self.entries.get(model_id)
        if entry is None:
            return None
        try:
            return CachedModel.from_dict(entry)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Corrupt manifest entry for '%s': %s", model_id, exc)
            return None

    def set(self, model: CachedModel) -> None:
        self.entries[model.model_id] = model.to_dict()

    def remove(self, model_id: str) -> bool:
        if model_id in self.entries:
            del self.entries[model_id]
            return True
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ModelManager:
    """Download, cache, and manage ML model weights for the edge worker.

    Parameters
    ----------
    cache_dir:
        Root directory for the model cache. Edge Core points this at
        ``CONFIG_DIR / "models"`` (``~/.cyberwave/models/`` by default,
        overridable via ``CYBERWAVE_EDGE_CONFIG_DIR``).
    api_token:
        Cyberwave API token used for catalog API calls.
    base_url:
        Cyberwave REST API base URL (e.g. ``https://api.cyberwave.com``).
    """

    def __init__(
        self,
        cache_dir: Path,
        api_token: str,
        base_url: str,
    ) -> None:
        self._cache_dir = cache_dir
        self._api_token = api_token
        self._base_url = base_url.rstrip("/")
        self._manifest_path = cache_dir / MANIFEST_FILENAME
        self._manifest = _Manifest.load(self._manifest_path)

    # ------------------------------------------------------------------
    # Core public methods
    # ------------------------------------------------------------------

    def ensure_model(self, model_id: str) -> Path:
        """Return the local path for *model_id*, downloading if necessary.

        See the module docstring for the full resolution order. In short:
        prefer a checksum-verified local file, refresh from the Cyberwave
        signed-URL endpoint or the upstream catalog URL when needed, and
        fall back to a stale-but-intact local file if every download
        attempt fails (air-gap and intermittent-network friendly).

        Raises
        ------
        RuntimeError
            If no source can be resolved and no usable cached file exists.
        ValueError
            If *model_id* is empty or whitespace.
        """
        model_id = model_id.strip()
        if not model_id:
            raise ValueError("model_id must be a non-empty string")

        with self._per_model_lock(model_id):
            # Reload manifest under the lock: another writer may have updated
            # it since our last read (worker startup racing the watcher loop).
            self._manifest = _Manifest.load(self._manifest_path)

            self._reconcile_disk_for(model_id)

            cached = self._manifest.get(model_id)
            cached_path: Optional[Path] = Path(cached.local_path) if cached else None
            cached_intact = self._cache_is_intact(cached, cached_path)
            # _cache_is_intact may have restamped a re-staged prestaged file
            # or back-filled mtime_ns; re-read so downstream sees the fresh
            # state.
            if cached_intact and cached is not None:
                cached = self._manifest.get(model_id) or cached

            if cached_intact and cached is not None and cached.pinned:
                # Operator-pinned: never probe the catalog, never download.
                # The on-disk file is the source of truth.
                assert cached_path is not None
                logger.debug(
                    "Pinned cache hit for model '%s' at %s — skipping catalog refresh",
                    model_id,
                    cached_path,
                )
                return cached_path

            if (
                cached_intact
                and cached is not None
                and not self._catalog_indicates_refresh(model_id, cached)
            ):
                assert cached_path is not None
                logger.debug("Cache hit for model '%s' at %s", model_id, cached_path)
                return cached_path

            try:
                return self._download_model(model_id)
            except Exception as exc:
                if cached_intact and cached_path is not None:
                    logger.warning(
                        "Download failed for model '%s' (%s); falling back to cached "
                        "weights at %s — checksum may not match the latest catalog entry",
                        model_id,
                        exc,
                        cached_path,
                    )
                    return cached_path
                raise

    def ensure_models(self, model_ids: list[str]) -> dict[str, Path]:
        """Batch version of :meth:`ensure_model`.

        Returns a mapping of ``model_id → local_path`` for every model that
        could be resolved. Models that fail to download are logged as
        warnings and omitted from the result rather than raising.
        """
        results: dict[str, Path] = {}
        for model_id in model_ids:
            try:
                results[model_id] = self.ensure_model(model_id)
            except Exception as exc:
                logger.warning(
                    "Failed to ensure model '%s': %s — worker will handle gracefully",
                    model_id,
                    exc,
                )
        return results

    def list_cached_models(self) -> list[CachedModel]:
        """Return all models currently in the local cache."""
        self._manifest = _Manifest.load(self._manifest_path)
        models: list[CachedModel] = []
        for model_id, entry in self._manifest.entries.items():
            try:
                models.append(CachedModel.from_dict(entry))
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Skipping corrupt manifest entry for '%s': %s", model_id, exc)
        return models

    def evict_model(self, model_id: str) -> bool:
        """Remove *model_id* from the local cache.

        Deletes the cached file (or sub-directory) and removes the manifest
        entry. Returns ``True`` if something was evicted.
        """
        cached = self._manifest.get(model_id)
        if cached is None:
            logger.debug("evict_model: '%s' not in manifest", model_id)
            return False

        local_path = Path(cached.local_path)
        removed_file = False
        if local_path.is_dir():
            try:
                shutil.rmtree(local_path)
                removed_file = True
                logger.info("Evicted model dir '%s' at %s", model_id, local_path)
            except OSError as exc:
                logger.warning("Failed to remove model dir %s: %s", local_path, exc)
        elif local_path.exists():
            try:
                local_path.unlink()
                removed_file = True
                logger.info("Evicted model file '%s' at %s", model_id, local_path)
            except OSError as exc:
                logger.warning("Failed to remove model file %s: %s", local_path, exc)

        self._manifest.remove(model_id)
        self._manifest.save(self._manifest_path)
        return removed_file

    def cache_size_bytes(self) -> int:
        """Return the total size of all files in the cache directory (bytes)."""
        if not self._cache_dir.exists():
            return 0
        total = 0
        for p in self._cache_dir.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    # ------------------------------------------------------------------
    # Model requirements discovery
    # ------------------------------------------------------------------

    @staticmethod
    def scan_worker_model_ids(workers_dir: Path) -> list[str]:
        """Scan ``*.py`` files in *workers_dir* for ``cw.models.load(...)`` calls.

        Returns a deduplicated list of model IDs referenced in worker files,
        preserving first-seen order.
        """
        model_ids: list[str] = []
        seen: set[str] = set()

        py_files = sorted(workers_dir.glob("*.py")) if workers_dir.is_dir() else []
        for py_file in py_files:
            try:
                source = py_file.read_text(errors="replace")
            except OSError as exc:
                logger.debug("Cannot read worker file %s: %s", py_file, exc)
                continue
            for match in _CW_MODELS_LOAD_RE.finditer(source):
                mid = match.group(1).strip()
                if mid and mid not in seen:
                    model_ids.append(mid)
                    seen.add(mid)
                    logger.debug("Found model reference '%s' in %s", mid, py_file.name)

        return model_ids

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------

    @contextmanager
    def _per_model_lock(self, model_id: str) -> Iterator[None]:
        """Hold an exclusive flock on ``cache_dir/{model_id}/.lock``.

        Without this, two ``ensure_model`` calls for the same model — for
        example a worker container restart racing Edge Core's worker
        watcher loop — could both stream into the same destination via
        their own ``.dl_*`` temp files and either clobber each other on
        ``replace()`` or interleave manifest writes.

        ``fcntl.flock`` is advisory and process-scoped: the OS releases
        the lock automatically when the process exits, so a crash during
        download leaves no stale lock to clean up.
        """
        lock_dir = self._cache_dir / model_id
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / MODEL_LOCK_FILENAME
        # Open in "a+" so we never truncate the file (other processes may
        # already have a handle to it) and we get write permission for
        # flock on every platform we care about.
        lock_fh = open(lock_path, "a+")
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            finally:
                lock_fh.close()

    # ------------------------------------------------------------------
    # Cache integrity helpers
    # ------------------------------------------------------------------

    def _cache_is_intact(self, cached: Optional[CachedModel], cached_path: Optional[Path]) -> bool:
        """Return True if the on-disk file matches what we last wrote.

        Resolution order:

        * No manifest entry, or file missing → not intact (cold cache).
        * Manifest has no stored checksum → treat as intact (legacy
          entries / pre-staged without sidecar).
        * **mtime fast-path**: when the file's ``(size, mtime_ns)`` match
          the manifest, skip SHA-256 entirely and report intact. This is
          the steady-state cost on warm cache (one ``stat()`` per model
          per call). A back-fill happens on first miss for legacy
          entries that lack ``mtime_ns``.
        * Otherwise compute SHA-256 and compare to the manifest:

          - Match → report intact and back-fill ``mtime_ns`` so the next
            call hits the fast-path.
          - Mismatch on a ``downloaded_from == prestaged`` sidecar →
            operator dropped a new build into place; restamp manifest
            and report intact.
          - Mismatch on a downloaded artifact → corrupt; report not
            intact and let the caller try to re-download.
        """
        if cached is None or cached_path is None or not cached_path.exists():
            return False
        if not cached.checksum_sha256:
            return True

        try:
            stat = cached_path.stat()
        except OSError:
            return False

        if (
            cached.mtime_ns is not None
            and stat.st_mtime_ns == cached.mtime_ns
            and stat.st_size == cached.size_bytes
        ):
            return True

        actual = _sha256_file(cached_path)
        if actual == cached.checksum_sha256:
            self._backfill_mtime(cached, stat.st_mtime_ns, stat.st_size)
            return True
        if self._restamp_if_prestaged(cached, cached_path, actual, stat):
            return True
        logger.warning(
            "Local file for model '%s' is corrupt (expected sha256=%s…, got %s…) "
            "— will attempt to re-download",
            cached.model_id,
            cached.checksum_sha256[:12],
            actual[:12],
        )
        return False

    def _backfill_mtime(self, cached: CachedModel, mtime_ns: int, size_bytes: int) -> None:
        """Persist ``mtime_ns`` (and refreshed size) onto a manifest entry.

        Used after a successful SHA-256 verification so that subsequent
        calls hit the cheap-stat fast path. We avoid touching the on-disk
        file's mtime here — that would defeat the point.
        """
        if cached.mtime_ns == mtime_ns and cached.size_bytes == size_bytes:
            return
        refreshed = CachedModel(
            model_id=cached.model_id,
            local_path=cached.local_path,
            size_bytes=size_bytes,
            downloaded_at=cached.downloaded_at,
            source_url=cached.source_url,
            checksum_sha256=cached.checksum_sha256,
            runtime=cached.runtime,
            pinned=cached.pinned,
            mtime_ns=mtime_ns,
        )
        self._manifest.set(refreshed)
        self._manifest.save(self._manifest_path)

    def _restamp_if_prestaged(
        self,
        cached: CachedModel,
        cached_path: Path,
        actual_checksum: str,
        stat: "os.stat_result",
    ) -> bool:
        """Refresh the manifest from disk when *cached_path* is operator-staged.

        Returns True when a restamp happened. The check is gated on the
        sidecar's ``downloaded_from`` field so that bit-rot in downloaded
        artifacts still triggers a re-download rather than being silently
        accepted as the new truth.

        ``pinned`` is preserved from the existing sidecar — restamping is
        an in-place update, not a re-pin.
        """
        sidecar_path = cached_path.parent / MODEL_METADATA_FILENAME
        if not sidecar_path.exists():
            return False
        try:
            with open(sidecar_path) as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return False
        if not isinstance(meta, dict) or meta.get("downloaded_from") != SOURCE_KIND_PRESTAGED:
            return False

        new_at = _utc_iso_now()
        logger.info(
            "Operator-staged model '%s' changed on disk (sha256 %s… → %s…); re-stamping manifest",
            cached.model_id,
            (cached.checksum_sha256 or "?")[:12],
            actual_checksum[:12],
        )

        meta["checksum_sha256"] = actual_checksum
        meta["size_bytes"] = stat.st_size
        meta["downloaded_at"] = new_at
        _write_json_safe(sidecar_path, meta)

        # Re-stat after the sidecar write to capture the file's mtime as
        # it is now (the sidecar write doesn't touch the weight file but
        # the operator's own write does).
        try:
            fresh_mtime = cached_path.stat().st_mtime_ns
        except OSError:
            fresh_mtime = stat.st_mtime_ns

        pinned = bool(meta.get("pinned", cached.pinned))
        self._manifest.set(
            CachedModel(
                model_id=cached.model_id,
                local_path=cached.local_path,
                size_bytes=stat.st_size,
                downloaded_at=new_at,
                source_url=cached.source_url,
                checksum_sha256=actual_checksum,
                runtime=cached.runtime,
                pinned=pinned,
                mtime_ns=fresh_mtime,
            )
        )
        self._manifest.save(self._manifest_path)
        return True

    def _catalog_indicates_refresh(self, model_id: str, cached: CachedModel) -> bool:
        """Best-effort probe: does the catalog have a newer checksum?

        Returns ``False`` (no refresh) when the catalog is unreachable, when
        the catalog entry has no checksum, or when the catalog checksum
        matches the cached file. This is the air-gap / fail-soft default.

        Returns ``True`` only when we successfully reached the catalog and
        the published checksum disagrees with the cached file.
        """
        catalog_entry = self._fetch_catalog_entry_safe(model_id)
        if catalog_entry is None:
            return False

        expected_checksum = _extract_checksum(catalog_entry)
        if not expected_checksum:
            return False

        if expected_checksum == cached.checksum_sha256:
            return False

        logger.info(
            "Catalog has new checksum for model '%s' (cached=%s…, catalog=%s…) — will refresh",
            model_id,
            (cached.checksum_sha256 or "?")[:12],
            expected_checksum[:12],
        )
        return True

    # ------------------------------------------------------------------
    # Disk reconciliation (pre-staged / air-gapped support)
    # ------------------------------------------------------------------

    def _reconcile_disk_for(self, model_id: str) -> None:
        """Rebuild the manifest entry for *model_id* from on-disk state.

        Idempotent. Used for two scenarios:

        * Air-gapped operator copied weights into ``cache_dir/{model_id}/``
          on a USB stick, with or without a sidecar ``metadata.json``.
        * Manifest was deleted but per-model sidecars remain.

        If a sidecar exists it is honored; otherwise a single weight file
        in the directory is auto-detected and a sidecar is written so that
        subsequent runs are deterministic.
        """
        if self._manifest.get(model_id) is not None:
            return

        model_dir = self._cache_dir / model_id
        if not model_dir.is_dir():
            return

        sidecar_path = model_dir / MODEL_METADATA_FILENAME
        meta: dict[str, Any] = {}
        if sidecar_path.exists():
            try:
                with open(sidecar_path) as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    meta = raw
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Cannot read sidecar at %s: %s", sidecar_path, exc)

        weight_path = self._resolve_prestaged_weight_file(model_dir, meta)
        if weight_path is None:
            return

        # Always recompute the on-disk SHA-256 during reconcile, even when
        # the sidecar stores one. Otherwise an operator who overwrites the
        # file *after* writing the sidecar leaves the manifest in an
        # inconsistent state — stale checksum paired with the fresh mtime
        # — that subsequently passes the mtime fast-path in
        # _cache_is_intact and serves the new bytes under the old hash.
        on_disk_checksum = _sha256_file(weight_path)
        sidecar_checksum = meta.get("checksum_sha256")
        if isinstance(sidecar_checksum, str) and sidecar_checksum.strip():
            sidecar_checksum = sidecar_checksum.strip()
            if sidecar_checksum != on_disk_checksum:
                logger.warning(
                    "Pre-staged model '%s' sidecar checksum %s… does not match "
                    "on-disk %s… — using on-disk hash (sidecar is stale or wrong)",
                    model_id,
                    sidecar_checksum[:12],
                    on_disk_checksum[:12],
                )
        else:
            logger.info(
                "Pre-staged model '%s' has no sidecar checksum; computed sha256=%s… from disk",
                model_id,
                on_disk_checksum[:12],
            )
        checksum = on_disk_checksum

        runtime = meta.get("runtime") if isinstance(meta.get("runtime"), str) else None
        if not runtime:
            runtime = _runtime_from_extension(weight_path.suffix)
            if runtime:
                logger.info(
                    "Inferred runtime '%s' for pre-staged model '%s' from extension '%s'",
                    runtime,
                    model_id,
                    weight_path.suffix,
                )
        downloaded_at = meta.get("downloaded_at") or _utc_iso_now()
        source_url = meta.get("source_url") or meta.get("download_url") or meta.get("upstream_url")
        pinned = bool(meta.get("pinned", False))

        weight_stat = weight_path.stat()
        cached = CachedModel(
            model_id=model_id,
            local_path=str(weight_path),
            size_bytes=weight_stat.st_size,
            downloaded_at=downloaded_at,
            source_url=source_url if isinstance(source_url, str) else None,
            checksum_sha256=checksum,
            runtime=runtime,
            pinned=pinned,
            mtime_ns=weight_stat.st_mtime_ns,
        )
        self._manifest.set(cached)
        self._manifest.save(self._manifest_path)

        if not sidecar_path.exists():
            _write_json_safe(
                sidecar_path,
                {
                    "model_id": model_id,
                    "runtime": runtime,
                    "filename": weight_path.name,
                    "checksum_sha256": checksum,
                    "size_bytes": cached.size_bytes,
                    "downloaded_at": downloaded_at,
                    "downloaded_from": SOURCE_KIND_PRESTAGED,
                    "source_url": None,
                    "upstream_url": None,
                    "pinned": False,
                },
            )

        logger.info(
            "Reconciled pre-staged model '%s' from %s (sha256=%s…%s)",
            model_id,
            weight_path,
            checksum[:12],
            ", pinned" if pinned else "",
        )

    @staticmethod
    def _resolve_prestaged_weight_file(model_dir: Path, meta: dict[str, Any]) -> Optional[Path]:
        """Find the weight file in *model_dir*, using sidecar metadata if available."""
        filename = meta.get("filename")
        if isinstance(filename, str) and filename.strip():
            candidate = model_dir / filename.strip()
            if candidate.exists() and candidate.is_file():
                return candidate

        candidates = sorted(
            p
            for p in model_dir.iterdir()
            if p.is_file() and p.name != MODEL_METADATA_FILENAME and not p.name.startswith(".")
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logger.warning(
                "Cannot reconcile pre-staged model in %s: %d candidate weight files "
                "(set 'filename' in metadata.json to disambiguate)",
                model_dir,
                len(candidates),
            )
        return None

    # ------------------------------------------------------------------
    # Download path
    # ------------------------------------------------------------------

    def _download_model(self, model_id: str) -> Path:
        """Fetch model metadata from the API, download weights, cache them.

        Tries sources in priority order:

        #. Cyberwave-hosted signed URL (``GET /mlmodels/{uuid}/weights``).
        #. Upstream weights URL from the catalog entry.

        The first source that yields a checksum-verified file wins.
        """
        catalog_entry = self._fetch_catalog_entry(model_id)

        upstream_url = _extract_download_url(catalog_entry, model_id)
        artifact_url = self._fetch_artifact_url_safe(catalog_entry)

        # Each source carries three URLs:
        #   kind:        which catalog field it came from
        #   fetch_url:   the URL we actually GET (may be a time-limited
        #                signed URL that expires)
        #   source_id:   the URL we persist as a stable identifier of the
        #                source. For artifact downloads this is the
        #                resolver endpoint (/mlmodels/{uuid}/weights), not
        #                the signed URL — the signed URL is useless for
        #                re-fetching and would mislead anyone reading the
        #                manifest later.
        sources: list[tuple[str, str, str]] = []
        if artifact_url:
            artifact_endpoint = self._artifact_endpoint(catalog_entry) or artifact_url
            sources.append((SOURCE_KIND_ARTIFACT, artifact_url, artifact_endpoint))
        if upstream_url:
            sources.append((SOURCE_KIND_UPSTREAM, upstream_url, upstream_url))

        if not sources:
            raise RuntimeError(
                f"No download sources available for model '{model_id}': the backend "
                f"reports no checkpoint at /mlmodels/{{uuid}}/weights and the catalog "
                f"entry has no upstream download_url. "
                f"Catalog entry keys: {sorted(catalog_entry.keys())!r}"
            )

        expected_checksum = _extract_checksum(catalog_entry)
        runtime = _extract_runtime(catalog_entry)
        # Use the first source's URL only to derive a filename when the
        # catalog does not specify one — the filename is independent of
        # which mirror we ultimately download from.
        filename = _derive_filename(model_id, catalog_entry, sources[0][1])

        model_dir = self._cache_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        dest_path = model_dir / filename

        last_exc: Optional[Exception] = None
        for source_kind, fetch_url, source_id in sources:
            logger.info(
                "Downloading model '%s' from %s (%s) → %s",
                model_id,
                source_kind,
                _redact_url(fetch_url),
                dest_path,
            )
            try:
                self._download_with_retries(fetch_url, dest_path)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Download from %s failed for model '%s': %s — trying next source",
                    source_kind,
                    model_id,
                    exc,
                )
                continue

            actual_checksum = _sha256_file(dest_path)
            if expected_checksum and actual_checksum != expected_checksum:
                dest_path.unlink(missing_ok=True)
                last_exc = RuntimeError(
                    f"Checksum mismatch for model '{model_id}' from {source_kind}: "
                    f"expected {expected_checksum}, got {actual_checksum}. "
                    f"Downloaded file removed."
                )
                logger.warning("%s", last_exc)
                # If this was the only/last source, surface the error to the
                # caller (which may then fall back to the stale cached file).
                if (source_kind, fetch_url, source_id) == sources[-1]:
                    raise last_exc
                continue

            self._record_successful_download(
                model_id=model_id,
                dest_path=dest_path,
                checksum=actual_checksum,
                runtime=runtime,
                source_kind=source_kind,
                source_url=source_id,
                upstream_url=upstream_url,
                filename=filename,
            )
            return dest_path

        raise RuntimeError(
            f"All download sources failed for model '{model_id}': {last_exc}"
        ) from last_exc

    def _artifact_endpoint(self, catalog_entry: dict[str, Any]) -> Optional[str]:
        """Return the persistent ``/mlmodels/{uuid}/weights`` endpoint URL.

        This is the URL we record as ``source_url`` for artifact
        downloads (vs. the time-limited signed URL we actually fetched).
        Anyone reading the manifest can re-issue this exact request to
        refresh credentials and download again.
        """
        uuid = catalog_entry.get("uuid")
        if not isinstance(uuid, str) or not _looks_like_uuid(uuid):
            return None
        return f"{self._base_url}{ML_MODELS_ENDPOINT}/{uuid}/weights"

    def _record_successful_download(
        self,
        *,
        model_id: str,
        dest_path: Path,
        checksum: str,
        runtime: Optional[str],
        source_kind: str,
        source_url: str,
        upstream_url: Optional[str],
        filename: str,
    ) -> None:
        """Persist sidecar + manifest entry for a freshly downloaded model.

        ``source_url`` here is the stable identifier of the source (the
        resolver endpoint for artifact downloads, the public URL for
        upstream downloads), not the time-limited signed URL we fetched.
        See :meth:`_download_model` for the rationale.

        Downloaded artifacts are *never* pinned: pinning is an explicit
        operator action recorded in a hand-written prestaged sidecar.
        """
        stat = dest_path.stat()
        downloaded_at = _utc_iso_now()

        _write_json_safe(
            dest_path.parent / MODEL_METADATA_FILENAME,
            {
                "model_id": model_id,
                "runtime": runtime,
                "filename": filename,
                "checksum_sha256": checksum,
                "size_bytes": stat.st_size,
                "downloaded_at": downloaded_at,
                "downloaded_from": source_kind,
                "source_url": source_url,
                "upstream_url": upstream_url,
                "pinned": False,
            },
        )

        # Re-stat after the sidecar write to capture the weight file's
        # mtime as it is now (sidecar write is atomic-replace and does
        # not touch the weight file, but be defensive).
        try:
            mtime_ns = dest_path.stat().st_mtime_ns
        except OSError:
            mtime_ns = stat.st_mtime_ns

        cached = CachedModel(
            model_id=model_id,
            local_path=str(dest_path),
            size_bytes=stat.st_size,
            downloaded_at=downloaded_at,
            source_url=source_url,
            checksum_sha256=checksum,
            runtime=runtime,
            pinned=False,
            mtime_ns=mtime_ns,
        )
        self._manifest.set(cached)
        self._manifest.save(self._manifest_path)
        logger.info(
            "Model '%s' cached at %s (%d bytes, sha256=%s…, source=%s)",
            model_id,
            dest_path,
            stat.st_size,
            checksum[:12],
            source_kind,
        )

    # ------------------------------------------------------------------
    # Catalog + artifact endpoint clients
    # ------------------------------------------------------------------

    def _fetch_catalog_entry(self, model_id: str) -> dict[str, Any]:
        """Call the Cyberwave catalog API and return the model metadata dict.

        Strict variant: raises ``RuntimeError`` if the catalog cannot be
        reached or the response cannot be parsed.

        If *model_id* looks like a UUID, fetch directly via
        ``GET /api/v1/mlmodels/{uuid}``. Otherwise fall back to the list
        endpoint filtered by ``model_external_id`` (e.g. ``yolov8n.pt``).
        """
        import httpx

        headers = {"Authorization": f"Bearer {self._api_token}"}

        if _looks_like_uuid(model_id):
            url = f"{self._base_url}{ML_MODELS_ENDPOINT}/{model_id}"
            try:
                resp = httpx.get(url, headers=headers, timeout=CATALOG_FETCH_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Unexpected catalog response type: {type(data)}")
                return data
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to fetch catalog entry for model '{model_id}' from {url}: {exc}"
                ) from exc

        url = f"{self._base_url}{ML_MODELS_ENDPOINT}?model_external_id={model_id}"
        try:
            resp = httpx.get(url, headers=headers, timeout=CATALOG_FETCH_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                if not data:
                    raise RuntimeError(f"No model found with model_external_id='{model_id}'")
                first = data[0]
                if not isinstance(first, dict):
                    raise RuntimeError(f"Unexpected catalog list element type: {type(first)}")
                return first
            if isinstance(data, dict):
                return data
            raise RuntimeError(f"Unexpected catalog response type: {type(data)}")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch catalog entry for model '{model_id}' from {url}: {exc}"
            ) from exc

    def _fetch_catalog_entry_safe(self, model_id: str) -> Optional[dict[str, Any]]:
        """Best-effort variant of :meth:`_fetch_catalog_entry` for the warm-cache probe.

        Returns ``None`` on any failure (network error, non-200, parse error).
        Uses a short timeout so an unreachable backend does not stall worker
        startup.
        """
        try:
            import httpx
        except ImportError:
            return None

        headers = {"Authorization": f"Bearer {self._api_token}"}

        if _looks_like_uuid(model_id):
            url = f"{self._base_url}{ML_MODELS_ENDPOINT}/{model_id}"
        else:
            url = f"{self._base_url}{ML_MODELS_ENDPOINT}?model_external_id={model_id}"

        try:
            resp = httpx.get(url, headers=headers, timeout=CATALOG_PROBE_TIMEOUT)
            if resp.status_code != 200:
                logger.debug(
                    "Catalog probe for '%s' returned %d — skipping refresh",
                    model_id,
                    resp.status_code,
                )
                return None
            data = resp.json()
        except Exception as exc:
            logger.debug("Catalog probe for '%s' failed: %s — skipping refresh", model_id, exc)
            return None

        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
        return None

    def _fetch_artifact_url_safe(self, catalog_entry: dict[str, Any]) -> Optional[str]:
        """Return a Cyberwave-hosted signed weights URL, or ``None``.

        Calls ``GET /mlmodels/{uuid}/weights`` to obtain a signed URL for a
        checkpoint we have uploaded to our private GCS bucket. Returns
        ``None`` when:

        * the catalog entry has no UUID (legacy / non-DB entry),
        * the backend does not host this checkpoint (HTTP 404), or
        * the request fails for any other reason.

        This method never raises — failures fall through to the upstream
        ``download_url`` source.
        """
        uuid = catalog_entry.get("uuid")
        if not isinstance(uuid, str) or not _looks_like_uuid(uuid):
            return None

        try:
            import httpx
        except ImportError:
            return None

        url = f"{self._base_url}{ML_MODELS_ENDPOINT}/{uuid}/weights"
        headers = {"Authorization": f"Bearer {self._api_token}"}
        try:
            resp = httpx.get(url, headers=headers, timeout=CATALOG_FETCH_TIMEOUT)
            if resp.status_code == 404:
                logger.debug("Backend does not host a checkpoint mirror for model uuid=%s", uuid)
                return None
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("Could not fetch signed artifact URL from %s: %s", url, exc)
            return None

        if not isinstance(data, dict):
            return None
        signed = data.get("signed_url") or data.get("url")
        if isinstance(signed, str) and signed.strip():
            return signed.strip()
        return None

    def _download_with_retries(self, url: str, dest: Path) -> None:
        """Download *url* to *dest* with retry and exponential back-off.

        Auth failures (HTTP 401/403) short-circuit the retry loop: an
        expired GCS signed URL or a missing Bearer token will not become
        valid by waiting, and the caller may have a different source to
        try.
        """

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                self._stream_download(url, dest)
                return
            except Exception as exc:
                last_exc = exc
                if _is_auth_failure(exc):
                    logger.warning(
                        "Authentication failed for %s (%s) — not retrying",
                        _redact_url(url),
                        exc,
                    )
                    raise RuntimeError(
                        f"Authentication failed for {_redact_url(url)}: {exc}"
                    ) from exc
                if attempt < MAX_DOWNLOAD_RETRIES:
                    delay = DOWNLOAD_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Download attempt %d/%d failed for %s: %s — retrying in %.0fs",
                        attempt,
                        MAX_DOWNLOAD_RETRIES,
                        _redact_url(url),
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"Download failed after {MAX_DOWNLOAD_RETRIES} attempts for "
            f"{_redact_url(url)}: {last_exc}"
        ) from last_exc

    def _stream_download(self, url: str, dest: Path) -> None:
        """Stream *url* directly to *dest* via a temp file.

        The Authorization header is only attached when *url* points at the
        Cyberwave backend; signed GCS URLs reject extra Authorization
        headers, and upstream public mirrors do not need our token.
        """
        import httpx

        headers: dict[str, str] = {}
        if url.startswith(self._base_url):
            headers["Authorization"] = f"Bearer {self._api_token}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=dest.parent, prefix=".dl_")
        tmp_path = Path(tmp_path_str)
        try:
            with (
                os.fdopen(tmp_fd, "wb") as fh,
                httpx.stream(
                    "GET", url, headers=headers, timeout=300.0, follow_redirects=True
                ) as resp,
            ):
                resp.raise_for_status()
                for chunk in resp.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    fh.write(chunk)
            tmp_path.replace(dest)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            if exc.errno == errno.ENOSPC:
                logger.error(
                    "Disk full while downloading model from %s — free space in %s and retry",
                    _redact_url(url),
                    dest.parent,
                )
            raise
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _looks_like_uuid(value: str) -> bool:
    """Return True if *value* matches the canonical UUID format."""
    return bool(_UUID_RE.match(value))


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_iso_now() -> str:
    """Return current UTC time as ISO-8601 string (no timezone suffix)."""
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_safe(path: Path, data: Any) -> None:
    """Atomically write *data* as JSON to *path*."""
    tmp_path = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w") as fh:
            json.dump(data, fh, indent=2)
        tmp_path.replace(path)
    except OSError as exc:
        logger.debug("Failed to write %s: %s", path, exc)
        tmp_path.unlink(missing_ok=True)


def _redact_url(url: str) -> str:
    """Strip query strings from a URL for logging (signed URLs leak tokens)."""
    return url.split("?", 1)[0]


def _is_auth_failure(exc: BaseException) -> bool:
    """Return True if *exc* represents an HTTP 401/403 from httpx.

    Used to short-circuit the download retry loop: auth failures will not
    become valid by waiting, and the caller can fall through to the next
    source (e.g. expired signed URL → upstream public URL).
    """
    try:
        import httpx
    except ImportError:
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (401, 403)
    return False


#: Map weight-file extensions to SDK runtime names. Mirrors the SDK's
#: ``ModelManager._detect_runtime_from_extension`` so that pre-staged files
#: get the same default the SDK would pick.
_RUNTIME_BY_EXTENSION: dict[str, str] = {
    ".pt": "ultralytics",
    ".pth": "torch",
    ".onnx": "onnxruntime",
    ".tflite": "tflite",
    ".engine": "tensorrt",
    ".trt": "tensorrt",
    ".xml": "opencv",
}


def _runtime_from_extension(suffix: str) -> Optional[str]:
    """Return the SDK runtime name implied by a weight file's extension."""
    return _RUNTIME_BY_EXTENSION.get(suffix.lower())


def _extract_download_url(entry: dict[str, Any], model_id: str) -> Optional[str]:
    """Resolve the *upstream* weights URL from a catalog API response dict.

    Tries several known key locations in order:

    1. ``entry["download_url"]``
    2. ``entry["metadata"]["download_url"]``
    3. ``entry["metadata"]["upstream_weights_url"]`` (preferred new name)
    4. ``entry["metadata"]["artifact_url"]`` (legacy alias)
    """
    url: Optional[str] = None
    url = url or (entry.get("download_url") if isinstance(entry.get("download_url"), str) else None)
    metadata = entry.get("metadata") or {}
    if isinstance(metadata, dict):
        url = url or (
            metadata.get("download_url") if isinstance(metadata.get("download_url"), str) else None
        )
        url = url or (
            metadata.get("upstream_weights_url")
            if isinstance(metadata.get("upstream_weights_url"), str)
            else None
        )
        url = url or (
            metadata.get("artifact_url") if isinstance(metadata.get("artifact_url"), str) else None
        )
    return url if url and url.strip() else None


def _extract_checksum(entry: dict[str, Any]) -> Optional[str]:
    """Extract SHA-256 checksum from a catalog response."""
    for key in ("checksum_sha256", "sha256", "checksum"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = entry.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("checksum_sha256", "sha256", "checksum"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_runtime(entry: dict[str, Any]) -> Optional[str]:
    """Extract runtime hint from a catalog response."""
    for key in ("runtime", "framework"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = entry.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("runtime", "framework"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _derive_filename(model_id: str, entry: dict[str, Any], download_url: str) -> str:
    """Derive a local filename for the model weight file.

    Uses (in order of preference):

    1. ``entry["filename"]`` or ``entry["metadata"]["filename"]``
    2. Last path component of *download_url* (stripped of query string)
    3. ``{model_id}.pt`` as fallback
    """
    explicit = entry.get("filename")
    if not isinstance(explicit, str) or not explicit.strip():
        metadata = entry.get("metadata") or {}
        if isinstance(metadata, dict):
            explicit = metadata.get("filename")

    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    url_path = download_url.split("?")[0]
    basename = url_path.rstrip("/").split("/")[-1]
    if basename and "." in basename:
        return basename

    return f"{model_id}.pt"


# Module-level convenience alias so callers can use scan_worker_model_ids()
# without going through the class.
scan_worker_model_ids = ModelManager.scan_worker_model_ids
