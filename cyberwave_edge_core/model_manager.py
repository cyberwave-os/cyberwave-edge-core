"""Model weight download, caching, and eviction for Cyberwave Edge Core.

Models are pre-downloaded before the worker container starts.  The cache
prevents duplicate downloads across restarts and across workers.

Cache layout::

    {config_dir}/models/
    ├── manifest.json          # {model_id: {path, size, downloaded_at, checksum}}
    ├── yolov8n/
    │   ├── yolov8n.pt         # Weight file
    │   └── metadata.json
    └── custom/
        └── {uuid}/
            └── ...
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
_MODEL_LOAD_PATTERN = re.compile(
    r"""(?:cw\.models\.load|models\.load)\s*\(\s*["']([^"']+)["']""",
    re.MULTILINE,
)
_CYBERWAVE_YML_MODELS_PATTERN = re.compile(
    r"^models\s*:\s*$", re.MULTILINE
)


@dataclass
class CachedModel:
    """Metadata entry for one cached model."""

    model_id: str
    path: str
    size_bytes: int
    downloaded_at: float
    checksum: str
    source_url: Optional[str] = None

    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass
class ModelManagerConfig:
    cache_dir: Path
    api_token: str = ""
    base_url: str = "https://api.cyberwave.com"


class ModelManager:
    """Download, cache, and evict ML model weights.

    The manager is intentionally designed to be *best-effort* at startup:
    a failed download does not block driver or worker container startup.
    Workers are responsible for graceful degradation when a model is absent.
    """

    def __init__(self, cache_dir: Path, api_token: str = "", base_url: str = "") -> None:
        self._cache_dir = cache_dir
        self._api_token = api_token
        self._base_url = base_url or "https://api.cyberwave.com"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ensure_model(self, model_id: str) -> Optional[Path]:
        """Return local path to *model_id*, downloading it if necessary.

        Returns None only when the model cannot be obtained and is not cached.
        """
        cached = self._load_manifest_entry(model_id)
        if cached:
            model_path = Path(cached.path)
            if model_path.exists() and self._verify_checksum(model_path, cached.checksum):
                logger.debug("Model %s: cache hit (%s)", model_id, model_path)
                return model_path
            logger.info(
                "Model %s: cache entry corrupt or missing — re-downloading", model_id
            )

        return self._download_model(model_id)

    def ensure_models(self, model_ids: list[str]) -> dict[str, Optional[Path]]:
        """Ensure all models in *model_ids*, returning a mapping of id → path."""
        results: dict[str, Optional[Path]] = {}
        for model_id in model_ids:
            results[model_id] = self.ensure_model(model_id)
        return results

    def list_cached_models(self) -> list[CachedModel]:
        """Return all entries in the local model cache manifest."""
        manifest = self._load_manifest()
        return list(manifest.values())

    def evict_model(self, model_id: str) -> bool:
        """Remove *model_id* from the cache.  Returns True if it was present."""
        manifest = self._load_manifest()
        if model_id not in manifest:
            return False

        entry = manifest.pop(model_id)
        model_path = Path(entry.path)
        try:
            if model_path.is_file():
                model_path.unlink()
            elif model_path.is_dir():
                shutil.rmtree(model_path, ignore_errors=True)
            model_dir = model_path.parent
            if model_dir.exists() and not any(model_dir.iterdir()):
                model_dir.rmdir()
        except OSError as exc:
            logger.warning("Failed to remove model files for %s: %s", model_id, exc)

        self._save_manifest(manifest)
        logger.info("Evicted model %s from cache", model_id)
        return True

    def cache_size_bytes(self) -> int:
        """Return the total byte size of all cached model files."""
        total = 0
        for entry in self.list_cached_models():
            path = Path(entry.path)
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total

    # ------------------------------------------------------------------
    # Worker file scanning
    # ------------------------------------------------------------------

    @staticmethod
    def scan_worker_model_requirements(workers_dir: Path) -> list[str]:
        """Scan *.py files in *workers_dir* for ``cw.models.load("...")`` calls.

        Also reads ``cyberwave.yml`` if present.  Returns a deduplicated list
        of model IDs.
        """
        model_ids: list[str] = []

        if workers_dir.exists():
            for py_file in sorted(workers_dir.glob("*.py")):
                try:
                    content = py_file.read_text(encoding="utf-8", errors="replace")
                    for match in _MODEL_LOAD_PATTERN.finditer(content):
                        model_id = match.group(1).strip()
                        if model_id:
                            model_ids.append(model_id)
                except OSError as exc:
                    logger.warning("Could not read worker file %s: %s", py_file, exc)

            yml_file = workers_dir / "cyberwave.yml"
            if yml_file.exists():
                model_ids.extend(_parse_cyberwave_yml_models(yml_file))

        seen: dict[str, None] = {}
        for mid in model_ids:
            seen[mid] = None
        return list(seen.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_dir(self, model_id: str) -> Path:
        safe_id = model_id.replace("/", "_").replace("..", "_")
        return self._cache_dir / safe_id

    def _download_model(self, model_id: str) -> Optional[Path]:
        """Fetch model metadata from backend and download weights."""
        if not self._api_token:
            logger.warning(
                "Cannot download model %s: no API token configured", model_id
            )
            return None

        try:
            download_url = self._fetch_model_download_url(model_id)
        except Exception as exc:
            logger.warning("Failed to fetch download URL for model %s: %s", model_id, exc)
            return None

        if not download_url:
            logger.warning("No download URL returned for model %s", model_id)
            return None

        model_dir = self._model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        filename = download_url.split("/")[-1].split("?")[0] or f"{model_id}.bin"
        dest_path = model_dir / filename

        logger.info("Downloading model %s to %s", model_id, dest_path)
        try:
            downloaded_path = self._download_file(download_url, dest_path)
        except Exception as exc:
            logger.warning("Failed to download model %s: %s", model_id, exc)
            return None

        checksum = self._compute_checksum(downloaded_path)
        entry = CachedModel(
            model_id=model_id,
            path=str(downloaded_path),
            size_bytes=downloaded_path.stat().st_size,
            downloaded_at=time.time(),
            checksum=checksum,
            source_url=download_url,
        )
        manifest = self._load_manifest()
        manifest[model_id] = entry
        self._save_manifest(manifest)

        logger.info(
            "Model %s cached at %s (%.1f MB)",
            model_id,
            downloaded_path,
            entry.size_mb(),
        )
        return downloaded_path

    def _fetch_model_download_url(self, model_id: str) -> Optional[str]:
        """Call the backend API to get the download URL for *model_id*."""
        import httpx

        url = f"{self._base_url.rstrip('/')}/api/v1/ml-models/{model_id}"
        headers = {"Authorization": f"Token {self._api_token}"}

        response = httpx.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict):
            metadata = data.get("metadata") or {}
            if isinstance(metadata, dict):
                return metadata.get("download_url") or data.get("download_url")
            return data.get("download_url")
        return None

    def _download_file(self, url: str, dest: Path) -> Path:
        """Download *url* to *dest* atomically via a temp file."""
        import httpx

        temp_dir = dest.parent
        with tempfile.NamedTemporaryFile(
            dir=temp_dir,
            prefix=dest.name,
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            with httpx.stream("GET", url, timeout=300, follow_redirects=True) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=65536):
                    tmp.write(chunk)

        tmp_path.replace(dest)
        return dest

    def _compute_checksum(self, path: Path) -> str:
        """Return SHA-256 hex digest of *path*."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _verify_checksum(self, path: Path, expected: str) -> bool:
        if not expected:
            return True
        try:
            return self._compute_checksum(path) == expected
        except OSError:
            return False

    def _manifest_path(self) -> Path:
        return self._cache_dir / MANIFEST_FILENAME

    def _load_manifest(self) -> dict[str, CachedModel]:
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            return {}
        try:
            with open(manifest_path) as f:
                raw: dict = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read model manifest %s: %s", manifest_path, exc)
            return {}

        result: dict[str, CachedModel] = {}
        for model_id, entry_data in raw.items():
            if not isinstance(entry_data, dict):
                continue
            try:
                result[model_id] = CachedModel(
                    model_id=model_id,
                    path=entry_data.get("path", ""),
                    size_bytes=int(entry_data.get("size_bytes", 0)),
                    downloaded_at=float(entry_data.get("downloaded_at", 0)),
                    checksum=entry_data.get("checksum", ""),
                    source_url=entry_data.get("source_url"),
                )
            except (TypeError, ValueError) as exc:
                logger.debug("Skipping malformed manifest entry %s: %s", model_id, exc)
        return result

    def _load_manifest_entry(self, model_id: str) -> Optional[CachedModel]:
        return self._load_manifest().get(model_id)

    def _save_manifest(self, manifest: dict[str, CachedModel]) -> None:
        manifest_path = self._manifest_path()
        raw = {mid: asdict(entry) for mid, entry in manifest.items()}
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                dir=self._cache_dir,
                prefix="manifest.",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                json.dump(raw, tmp, indent=2)
                tmp_path = Path(tmp.name)
            tmp_path.replace(manifest_path)
        except OSError as exc:
            logger.warning("Failed to save model manifest: %s", exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_cyberwave_yml_models(yml_file: Path) -> list[str]:
    """Extract model IDs from the ``models:`` list in ``cyberwave.yml``.

    Uses a simple line-based parser to avoid adding PyYAML as a hard
    dependency.  Recognises::

        models:
          - yolov8n
          - background-subtraction
    """
    model_ids: list[str] = []
    try:
        content = yml_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return model_ids

    in_models_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "models:":
            in_models_block = True
            continue
        if in_models_block:
            if stripped.startswith("- "):
                model_id = stripped[2:].strip().strip("\"'")
                if model_id:
                    model_ids.append(model_id)
            elif stripped and not stripped.startswith("#"):
                in_models_block = False
    return model_ids
