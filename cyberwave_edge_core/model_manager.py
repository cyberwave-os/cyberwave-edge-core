"""Model Manager — download, cache, and sync ML model weights for Edge Core.

Implements the model lifecycle described in CYB-1561:

* Resolve required models from worker files (regex scan for ``cw.models.load(...)``).
* Download model artifacts from the Cyberwave catalog API and store in a local
  cache directory (``~/.cyberwave/models/`` on macOS, ``/etc/cyberwave/models/``
  on Linux by default).
* Validate checksums and reuse existing artifacts on cache hit.
* Expose helper for listing and evicting cached models.

The cache directory is bind-mounted read-only into the worker container at
``/app/models/``.  The SDK's ``cw.models.load()`` resolves paths from that
mount — Edge Core is responsible for pre-populating it before starting the
container.

This module is Edge Core-internal and intentionally has **no** user-facing
CLI surface of its own.  The ``WorkerManager`` (CYB-1546) calls into it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

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
ML_MODELS_ENDPOINT = "/api/v1/ml-models"

#: Name of the per-model metadata sidecar inside a model sub-directory.
MODEL_METADATA_FILENAME = "metadata.json"

#: Name of the global cache manifest that indexes all cached models.
MANIFEST_FILENAME = "manifest.json"

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
    """Describes a single model artifact stored in the local cache."""

    model_id: str
    local_path: str
    size_bytes: int
    downloaded_at: str
    source_url: Optional[str] = None
    checksum_sha256: Optional[str] = None
    runtime: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CachedModel":
        return cls(
            model_id=data["model_id"],
            local_path=data["local_path"],
            size_bytes=int(data.get("size_bytes", 0)),
            downloaded_at=data.get("downloaded_at", ""),
            source_url=data.get("source_url"),
            checksum_sha256=data.get("checksum_sha256"),
            runtime=data.get("runtime"),
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
        Root directory for the model cache.  Defaults to the same directory
        that startup.py uses as ``CONFIG_DIR`` / ``models/``.
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

        Steps:
        1. Check the manifest for an existing entry.
        2. If entry exists, verify the file is present and the checksum
           matches (if a checksum is stored).  Return path on hit.
        3. On cache miss or checksum mismatch, download from the catalog
           API, verify, update manifest, and return path.

        Raises
        ------
        RuntimeError
            If the model cannot be resolved from the catalog API or if the
            download fails after all retries.
        """
        model_id = model_id.strip()
        if not model_id:
            raise ValueError("model_id must be a non-empty string")

        cached = self._manifest.get(model_id)
        if cached is not None:
            local_path = Path(cached.local_path)
            if local_path.exists():
                if cached.checksum_sha256:
                    actual = _sha256_file(local_path)
                    if actual == cached.checksum_sha256:
                        logger.debug("Cache hit for model '%s' at %s", model_id, local_path)
                        return local_path
                    logger.warning(
                        "Checksum mismatch for model '%s' (expected %s, got %s) — re-downloading",
                        model_id,
                        cached.checksum_sha256,
                        actual,
                    )
                else:
                    logger.debug(
                        "Cache hit for model '%s' (no checksum) at %s", model_id, local_path
                    )
                    return local_path
            else:
                logger.debug(
                    "Manifest entry for '%s' references missing file %s — re-downloading",
                    model_id,
                    local_path,
                )

        return self._download_model(model_id)

    def ensure_models(self, model_ids: list[str]) -> dict[str, Path]:
        """Batch version of :meth:`ensure_model`.

        Returns a mapping of ``model_id → local_path`` for every model that
        could be resolved.  Models that fail to download are logged as
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
        entry.  Returns ``True`` if something was evicted.
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_model(self, model_id: str) -> Path:
        """Fetch model metadata from the API, download weights, cache them."""
        catalog_entry = self._fetch_catalog_entry(model_id)
        download_url: Optional[str] = _extract_download_url(catalog_entry, model_id)
        if not download_url:
            raise RuntimeError(
                f"No download URL found in catalog entry for model '{model_id}'. "
                f"Catalog entry: {catalog_entry!r}"
            )

        expected_checksum: Optional[str] = _extract_checksum(catalog_entry)
        runtime: Optional[str] = _extract_runtime(catalog_entry)
        filename = _derive_filename(model_id, catalog_entry, download_url)

        model_dir = self._cache_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        dest_path = model_dir / filename

        logger.info(
            "Downloading model '%s' → %s",
            model_id,
            dest_path,
        )
        self._download_with_retries(download_url, dest_path)

        actual_checksum = _sha256_file(dest_path)
        if expected_checksum and actual_checksum != expected_checksum:
            dest_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for model '{model_id}': "
                f"expected {expected_checksum}, got {actual_checksum}. "
                f"Downloaded file removed."
            )

        # Write per-model sidecar metadata
        metadata: dict[str, Any] = {
            "model_id": model_id,
            "runtime": runtime,
            "download_url": download_url,
            "checksum_sha256": actual_checksum,
            "size_bytes": dest_path.stat().st_size,
        }
        _write_json_safe(model_dir / MODEL_METADATA_FILENAME, metadata)

        cached = CachedModel(
            model_id=model_id,
            local_path=str(dest_path),
            size_bytes=dest_path.stat().st_size,
            downloaded_at=_utc_iso_now(),
            source_url=download_url,
            checksum_sha256=actual_checksum,
            runtime=runtime,
        )
        self._manifest.set(cached)
        self._manifest.save(self._manifest_path)
        logger.info(
            "Model '%s' cached at %s (%d bytes, sha256=%s…)",
            model_id,
            dest_path,
            cached.size_bytes,
            actual_checksum[:12] if actual_checksum else "?",
        )
        return dest_path

    def _fetch_catalog_entry(self, model_id: str) -> dict[str, Any]:
        """Call the Cyberwave catalog API and return the model metadata dict.

        If *model_id* looks like a UUID, fetch directly via
        ``GET /api/v1/ml-models/{uuid}``.  Otherwise fall back to the list
        endpoint filtered by ``model_external_id`` (e.g. ``yolov8n.pt``).
        """
        import httpx

        headers = {"Authorization": f"Token {self._api_token}"}

        if _looks_like_uuid(model_id):
            url = f"{self._base_url}{ML_MODELS_ENDPOINT}/{model_id}"
            try:
                resp = httpx.get(url, headers=headers, timeout=30.0)
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
            resp = httpx.get(url, headers=headers, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                if not data:
                    raise RuntimeError(f"No model found with model_external_id='{model_id}'")
                return data[0]
            if isinstance(data, dict):
                return data
            raise RuntimeError(f"Unexpected catalog response type: {type(data)}")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch catalog entry for model '{model_id}' from {url}: {exc}"
            ) from exc

    def _download_with_retries(self, url: str, dest: Path) -> None:
        """Download *url* to *dest* with retry and exponential back-off."""

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                self._stream_download(url, dest)
                return
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_DOWNLOAD_RETRIES:
                    delay = DOWNLOAD_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Download attempt %d/%d failed for %s: %s — retrying in %.0fs",
                        attempt,
                        MAX_DOWNLOAD_RETRIES,
                        url,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"Download failed after {MAX_DOWNLOAD_RETRIES} attempts for {url}: {last_exc}"
        ) from last_exc

    def _stream_download(self, url: str, dest: Path) -> None:
        """Stream *url* directly to *dest* via a temp file."""
        import httpx

        headers = {"Authorization": f"Token {self._api_token}"}
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


def _extract_download_url(entry: dict[str, Any], model_id: str) -> Optional[str]:
    """Resolve download URL from a catalog API response dict.

    Tries several known key locations in order:
    1. ``entry["download_url"]``
    2. ``entry["metadata"]["download_url"]``
    3. ``entry["metadata"]["artifact_url"]``
    """
    url: Optional[str] = None
    url = url or (entry.get("download_url") if isinstance(entry.get("download_url"), str) else None)
    metadata = entry.get("metadata") or {}
    if isinstance(metadata, dict):
        url = url or (
            metadata.get("download_url") if isinstance(metadata.get("download_url"), str) else None
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
    # Explicit filename from catalog
    explicit = entry.get("filename")
    if not isinstance(explicit, str) or not explicit.strip():
        metadata = entry.get("metadata") or {}
        if isinstance(metadata, dict):
            explicit = metadata.get("filename")

    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    # Extract from URL path
    url_path = download_url.split("?")[0]
    basename = url_path.rstrip("/").split("/")[-1]
    if basename and "." in basename:
        return basename

    return f"{model_id}.pt"


# Module-level convenience alias so callers can use scan_worker_model_ids()
# without going through the class.
scan_worker_model_ids = ModelManager.scan_worker_model_ids
