"""Tests for cyberwave_edge_core.model_manager.

Covers:
- Cache hit (warm cache): no download triggered.
- Cache miss: download triggered and manifest updated.
- Checksum mismatch: re-download triggered.
- Corrupted/missing file after manifest entry: re-download.
- Manifest persistence across instances.
- Batch ensure_models: failures isolated per model.
- evict_model: file removed, manifest entry removed.
- cache_size_bytes: sums file sizes in cache dir.
- scan_worker_model_ids: regex scan for cw.models.load() calls.
- _extract_download_url: key lookup priority.
- _extract_checksum / _extract_runtime: catalog key variants.
- _derive_filename: catalog key / URL path / fallback.
- Private helpers: _sha256_file, _utc_iso_now.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Optional
from unittest.mock import patch

import pytest

from cyberwave_edge_core import model_manager as _model_manager_mod
from cyberwave_edge_core.model_manager import (
    MODEL_METADATA_FILENAME,
    SOURCE_KIND_ARTIFACT,
    SOURCE_KIND_PRESTAGED,
    SOURCE_KIND_RUNTIME_MANAGED,
    SOURCE_KIND_UPSTREAM,
    CachedModel,
    ModelManager,
    _derive_filename,
    _extract_checksum,
    _extract_download_url,
    _extract_runtime,
    _Manifest,
    _redact_url,
    _sha256_file,
    _utc_iso_now,
    scan_worker_model_ids,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_network_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the warm-cache catalog refresh probe a no-op by default.

    Tests that exercise the refresh path explicitly opt in by
    monkeypatching ``_fetch_catalog_entry_safe`` (and/or
    ``_fetch_artifact_url_safe``) themselves.
    """
    monkeypatch.setattr(
        ModelManager,
        "_fetch_catalog_entry_safe",
        lambda self, model_id: None,
    )
    monkeypatch.setattr(
        ModelManager,
        "_fetch_artifact_url_safe",
        lambda self, catalog_entry: None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    tmp_path: Path, token: str = "tok", base_url: str = "https://api.test"
) -> ModelManager:
    return ModelManager(cache_dir=tmp_path, api_token=token, base_url=base_url)


def _write_fake_model(cache_dir: Path, model_id: str, content: bytes = b"weights") -> Path:
    """Write a fake model file and update the manifest."""
    model_dir = cache_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / f"{model_id}.pt"
    dest.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()
    manifest_path = cache_dir / "manifest.json"
    manifest = _Manifest.load(manifest_path)
    manifest.set(
        CachedModel(
            model_id=model_id,
            local_path=str(dest),
            size_bytes=len(content),
            downloaded_at="2026-01-01T00:00:00Z",
            source_url="https://example.com/model.pt",
            checksum_sha256=checksum,
            runtime="ultralytics",
        )
    )
    manifest.save(manifest_path)
    return dest


# ---------------------------------------------------------------------------
# Warm-cache tests
# ---------------------------------------------------------------------------


def test_ensure_model_cache_hit_returns_path_without_download(tmp_path: Path) -> None:
    dest = _write_fake_model(tmp_path, "yolov8n")
    manager = _make_manager(tmp_path)

    err = AssertionError("should not download")
    with patch.object(manager, "_download_model", side_effect=err):
        result = manager.ensure_model("yolov8n")

    assert result == dest


def test_ensure_model_cache_hit_no_checksum_skips_hash(tmp_path: Path) -> None:
    model_dir = tmp_path / "yolov8n"
    model_dir.mkdir()
    dest = model_dir / "yolov8n.pt"
    dest.write_bytes(b"data")

    manifest = _Manifest()
    manifest.set(
        CachedModel(
            model_id="yolov8n",
            local_path=str(dest),
            size_bytes=4,
            downloaded_at="2026-01-01T00:00:00Z",
            checksum_sha256=None,
        )
    )
    manifest.save(tmp_path / "manifest.json")

    manager = _make_manager(tmp_path)
    err = AssertionError("should not download")
    with patch.object(manager, "_download_model", side_effect=err):
        result = manager.ensure_model("yolov8n")

    assert result == dest


# ---------------------------------------------------------------------------
# Cold-cache / download tests
# ---------------------------------------------------------------------------


def _make_fake_download(
    tmp_path: Path,
    model_id: str,
    content: bytes = b"fake-weights",
    checksum: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Build a fake catalog entry dict and the expected checksum."""
    actual_checksum = hashlib.sha256(content).hexdigest()
    expected_checksum = checksum if checksum is not None else actual_checksum
    catalog_entry: dict[str, Any] = {
        "download_url": "https://dl.example.com/yolov8n.pt",
        "checksum_sha256": expected_checksum,
        "runtime": "ultralytics",
        "filename": f"{model_id}.pt",
    }
    return catalog_entry, actual_checksum


def test_ensure_model_cache_miss_triggers_download(tmp_path: Path) -> None:
    model_id = "yolov8n"
    content = b"fake weights"
    catalog_entry, checksum = _make_fake_download(tmp_path, model_id, content)
    manager = _make_manager(tmp_path)

    with (
        patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry),
        patch.object(manager, "_stream_download") as mock_dl,
    ):

        def _fake_stream(url: str, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

        mock_dl.side_effect = _fake_stream
        result = manager.ensure_model(model_id)

    assert result.exists()
    assert result.read_bytes() == content

    manifest = _Manifest.load(tmp_path / "manifest.json")
    cached = manifest.get(model_id)
    assert cached is not None
    assert cached.checksum_sha256 == checksum
    assert cached.runtime == "ultralytics"


def test_ensure_model_manifest_persists_across_instances(tmp_path: Path) -> None:
    model_id = "detector"
    content = b"weights-v1"
    catalog_entry, checksum = _make_fake_download(tmp_path, model_id, content)

    manager1 = _make_manager(tmp_path)
    with (
        patch.object(manager1, "_fetch_catalog_entry", return_value=catalog_entry),
        patch.object(manager1, "_stream_download") as mock_dl,
    ):

        def _fake_stream(url: str, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

        mock_dl.side_effect = _fake_stream
        path1 = manager1.ensure_model(model_id)

    manager2 = _make_manager(tmp_path)
    err = AssertionError("should not re-download")
    with patch.object(manager2, "_download_model", side_effect=err):
        path2 = manager2.ensure_model(model_id)

    assert path1 == path2


# ---------------------------------------------------------------------------
# Checksum mismatch / corrupt cache tests
# ---------------------------------------------------------------------------


def test_ensure_model_checksum_mismatch_triggers_redownload(tmp_path: Path) -> None:
    model_id = "yolov8n"
    content = b"good weights"
    model_dir = tmp_path / model_id
    model_dir.mkdir()
    corrupt_dest = model_dir / f"{model_id}.pt"
    corrupt_dest.write_bytes(b"corrupt data")

    wrong_checksum = hashlib.sha256(b"different").hexdigest()
    manifest = _Manifest()
    manifest.set(
        CachedModel(
            model_id=model_id,
            local_path=str(corrupt_dest),
            size_bytes=11,
            downloaded_at="2026-01-01T00:00:00Z",
            checksum_sha256=wrong_checksum,
        )
    )
    manifest.save(tmp_path / "manifest.json")

    manager = _make_manager(tmp_path)
    download_called: list[str] = []

    def _fake_download(mid: str) -> Path:
        download_called.append(mid)
        corrupt_dest.write_bytes(content)
        return corrupt_dest

    with patch.object(manager, "_download_model", side_effect=_fake_download):
        manager.ensure_model(model_id)

    assert download_called == [model_id]


def test_ensure_model_missing_file_triggers_redownload(tmp_path: Path) -> None:
    model_id = "detector"
    manifest = _Manifest()
    missing_path = tmp_path / model_id / f"{model_id}.pt"
    manifest.set(
        CachedModel(
            model_id=model_id,
            local_path=str(missing_path),
            size_bytes=0,
            downloaded_at="2026-01-01T00:00:00Z",
        )
    )
    manifest.save(tmp_path / "manifest.json")

    download_called: list[str] = []
    manager = _make_manager(tmp_path)

    def _fake_download(mid: str) -> Path:
        download_called.append(mid)
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_path.write_bytes(b"new weights")
        return missing_path

    with patch.object(manager, "_download_model", side_effect=_fake_download):
        manager.ensure_model(model_id)

    assert download_called == [model_id]


# ---------------------------------------------------------------------------
# ensure_models (batch)
# ---------------------------------------------------------------------------


def test_ensure_models_isolates_per_model_failures(tmp_path: Path) -> None:
    good_dest = _write_fake_model(tmp_path, "yolov8n", b"good")
    manager = _make_manager(tmp_path)

    def _download_raises(model_id: str) -> Path:
        raise RuntimeError("network error")

    with patch.object(manager, "_download_model", side_effect=_download_raises):
        result = manager.ensure_models(["yolov8n", "nonexistent"])

    assert "yolov8n" in result
    assert result["yolov8n"] == good_dest
    assert "nonexistent" not in result


def test_ensure_models_empty_list(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    assert manager.ensure_models([]) == {}
    assert manager.last_ensure_failures == {}


def test_ensure_models_tracks_failures(tmp_path: Path) -> None:
    """last_ensure_failures should contain the model IDs that failed and their error messages."""
    _write_fake_model(tmp_path, "yolov8n", b"good")
    manager = _make_manager(tmp_path)

    def _download_raises(model_id: str) -> Path:
        raise RuntimeError("network error")

    with patch.object(manager, "_download_model", side_effect=_download_raises):
        manager.ensure_models(["yolov8n", "nonexistent"])

    assert "nonexistent" in manager.last_ensure_failures
    assert "network error" in manager.last_ensure_failures["nonexistent"]
    assert "yolov8n" not in manager.last_ensure_failures


def test_ensure_models_clears_previous_failures(tmp_path: Path) -> None:
    """Each ensure_models call should reset last_ensure_failures."""
    _write_fake_model(tmp_path, "yolov8n", b"good")
    manager = _make_manager(tmp_path)
    manager.last_ensure_failures = {"old_model": "old error"}
    manager.ensure_models(["yolov8n"])
    assert manager.last_ensure_failures == {}


# ---------------------------------------------------------------------------
# evict_model
# ---------------------------------------------------------------------------


def test_evict_model_removes_file_and_manifest_entry(tmp_path: Path) -> None:
    dest = _write_fake_model(tmp_path, "yolov8n")
    manager = _make_manager(tmp_path)

    removed = manager.evict_model("yolov8n")
    assert removed is True
    assert not dest.exists()
    manifest = _Manifest.load(tmp_path / "manifest.json")
    assert manifest.get("yolov8n") is None


def test_evict_model_nonexistent_returns_false(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    assert manager.evict_model("nonexistent") is False


def test_evict_model_removes_orphan_directory_without_manifest_entry(
    tmp_path: Path,
) -> None:
    """``evict_model`` must clean up a wedged ``cache_dir/{id}/`` directory
    even when there is no manifest entry — this is the recovery path
    for hosts that hit the ``IsADirectoryError`` wedge described in
    ``_download_runtime_managed`` / ``_download_model``."""
    model_id = "yoloe-26m-seg.pt"
    orphan = tmp_path / model_id
    orphan.mkdir()
    (orphan / "metadata.json").write_text("{}")

    manager = _make_manager(tmp_path)
    # Defeat the startup auto-sweep so we exercise the evict_model path
    # explicitly: re-create the orphan after __init__ ran.
    orphan.mkdir(exist_ok=True)
    (orphan / "metadata.json").write_text("{}")

    removed = manager.evict_model(model_id)
    assert removed is True
    assert not orphan.exists()


def test_evict_model_explicit_call_overrides_conservative_sweep(
    tmp_path: Path,
) -> None:
    """An explicit ``evict_model`` is a stronger operator-intent signal
    than the conservative startup sweep. The sweep skips dirs with
    non-cruft content, but ``evict_model`` deletes them — if someone
    called evict, they want the cache slot cleared.

    Companion to ``test_prune_orphan_staging_dirs_at_init_preserves_operator_files``,
    which exercises the conservative path."""
    model_id = "yoloe-26m-seg.pt"
    orphan = tmp_path / model_id
    orphan.mkdir()
    (orphan / "metadata.json").write_text("{}")
    (orphan / "operator-weights.pt").write_bytes(b"hand-staged")

    manager = _make_manager(tmp_path)
    # The startup sweep does NOT touch this directory because of the
    # operator file — verify the sweep was conservative.
    assert orphan.exists()
    assert (orphan / "operator-weights.pt").exists()

    removed = manager.evict_model(model_id)
    assert removed is True
    assert not orphan.exists()


# ---------------------------------------------------------------------------
# Startup orphan-directory sweep
# ---------------------------------------------------------------------------


def test_prune_orphan_staging_dirs_at_init_removes_empty_dir(
    tmp_path: Path,
) -> None:
    """``ModelManager.__init__`` self-heals hosts that came up with an
    empty ``cache_dir/{id}/`` directory left over from a previously
    failed download — without it the SDK would crash on
    ``torch.load(<dir>)`` on every restart."""
    orphan = tmp_path / "yoloe-26m-seg.pt"
    orphan.mkdir()

    _make_manager(tmp_path)
    assert not orphan.exists()


def test_prune_orphan_staging_dirs_at_init_removes_metadata_only_dir(
    tmp_path: Path,
) -> None:
    """Dir containing only a metadata sidecar counts as orphan cruft."""
    orphan = tmp_path / "yoloe-26m-seg.pt"
    orphan.mkdir()
    (orphan / MODEL_METADATA_FILENAME).write_text('{"model_id": "yoloe-26m-seg.pt"}')

    _make_manager(tmp_path)
    assert not orphan.exists()


def test_prune_orphan_staging_dirs_at_init_removes_partial_download(
    tmp_path: Path,
) -> None:
    """Dir containing only a ``.dl_*.part`` partial download is orphan cruft."""
    orphan = tmp_path / "yoloe-26m-seg.pt"
    orphan.mkdir()
    (orphan / ".dl_abc123.part").write_bytes(b"partial")

    _make_manager(tmp_path)
    assert not orphan.exists()


def test_prune_orphan_staging_dirs_at_init_preserves_operator_files(
    tmp_path: Path,
) -> None:
    """Operator-staged content (any unexpected file) blocks the sweep —
    a human's hand-staged work always wins over the self-heal."""
    operator_dir = tmp_path / "yoloe-26m-seg.pt"
    operator_dir.mkdir()
    (operator_dir / "operator-weights.pt").write_bytes(b"hand-staged")

    _make_manager(tmp_path)
    assert operator_dir.exists()
    assert (operator_dir / "operator-weights.pt").exists()


def test_prune_orphan_staging_dirs_at_init_preserves_manifest_tracked_dir(
    tmp_path: Path,
) -> None:
    """A directory tracked in the manifest must never be considered
    orphan, even if it transiently looks empty (e.g. between an
    operator delete and a re-stage)."""
    model_id = "tracked-model"
    tracked_dir = tmp_path / model_id
    tracked_dir.mkdir()
    # Pre-populate the manifest pointing at this directory.
    manifest = _Manifest()
    manifest.set(
        CachedModel(
            model_id=model_id,
            local_path=str(tracked_dir / "weights.pt"),
            size_bytes=0,
            downloaded_at="2026-01-01T00:00:00Z",
        )
    )
    manifest.save(tmp_path / "manifest.json")

    _make_manager(tmp_path)
    assert tracked_dir.exists()


def test_evict_model_directory(tmp_path: Path) -> None:
    model_id = "big-model"
    model_dir = tmp_path / model_id
    model_dir.mkdir()
    (model_dir / "big-model.pt").write_bytes(b"a")
    (model_dir / "metadata.json").write_text("{}")

    manifest = _Manifest()
    manifest.set(
        CachedModel(
            model_id=model_id,
            local_path=str(model_dir),
            size_bytes=1,
            downloaded_at="2026-01-01T00:00:00Z",
        )
    )
    manifest.save(tmp_path / "manifest.json")

    manager = _make_manager(tmp_path)
    removed = manager.evict_model(model_id)
    assert removed is True
    assert not model_dir.exists()


# ---------------------------------------------------------------------------
# list_cached_models / cache_size_bytes
# ---------------------------------------------------------------------------


def test_list_cached_models_returns_all_entries(tmp_path: Path) -> None:
    _write_fake_model(tmp_path, "yolov8n")
    _write_fake_model(tmp_path, "detector")
    manager = _make_manager(tmp_path)
    models = manager.list_cached_models()
    ids = {m.model_id for m in models}
    assert ids == {"yolov8n", "detector"}


def test_list_cached_models_empty_cache(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    assert manager.list_cached_models() == []


def test_cache_size_bytes_sums_files(tmp_path: Path) -> None:
    _write_fake_model(tmp_path, "m1", b"abc")
    _write_fake_model(tmp_path, "m2", b"de")
    manager = _make_manager(tmp_path)
    size = manager.cache_size_bytes()
    assert size >= 5  # at least the weight bytes


def test_cache_size_bytes_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    manager = ModelManager(cache_dir=empty, api_token="t", base_url="https://x")
    assert manager.cache_size_bytes() == 0


# ---------------------------------------------------------------------------
# scan_worker_model_ids
# ---------------------------------------------------------------------------


def test_scan_worker_model_ids_finds_single_model(tmp_path: Path) -> None:
    workers_dir = tmp_path / "workers"
    workers_dir.mkdir()
    (workers_dir / "detect.py").write_text('result = cw.models.load("yolov8n")\n')
    ids = scan_worker_model_ids(workers_dir)
    assert ids == ["yolov8n"]


def test_scan_worker_model_ids_finds_multiple_across_files(tmp_path: Path) -> None:
    workers_dir = tmp_path / "workers"
    workers_dir.mkdir()
    (workers_dir / "a.py").write_text('cw.models.load("yolov8n")\n')
    (workers_dir / "b.py").write_text('m = cw.models.load("background-subtraction")\n')
    ids = scan_worker_model_ids(workers_dir)
    assert set(ids) == {"yolov8n", "background-subtraction"}


def test_scan_worker_model_ids_deduplicates(tmp_path: Path) -> None:
    workers_dir = tmp_path / "workers"
    workers_dir.mkdir()
    (workers_dir / "a.py").write_text('cw.models.load("yolov8n")\ncw.models.load("yolov8n")\n')
    ids = scan_worker_model_ids(workers_dir)
    assert ids.count("yolov8n") == 1


def test_scan_worker_model_ids_ignores_non_py_files(tmp_path: Path) -> None:
    workers_dir = tmp_path / "workers"
    workers_dir.mkdir()
    (workers_dir / "readme.txt").write_text('cw.models.load("should-be-ignored")\n')
    (workers_dir / "script.py").write_text('cw.models.load("found")\n')
    ids = scan_worker_model_ids(workers_dir)
    assert ids == ["found"]


def test_scan_worker_model_ids_missing_dir(tmp_path: Path) -> None:
    ids = scan_worker_model_ids(tmp_path / "nonexistent")
    assert ids == []


def test_scan_worker_model_ids_empty_dir(tmp_path: Path) -> None:
    workers_dir = tmp_path / "workers"
    workers_dir.mkdir()
    assert scan_worker_model_ids(workers_dir) == []


def test_scan_worker_model_ids_single_quotes(tmp_path: Path) -> None:
    workers_dir = tmp_path / "workers"
    workers_dir.mkdir()
    (workers_dir / "w.py").write_text("cw.models.load('my-model')\n")
    assert scan_worker_model_ids(workers_dir) == ["my-model"]


# ---------------------------------------------------------------------------
# _extract_download_url
# ---------------------------------------------------------------------------


def test_extract_download_url_top_level(tmp_path: Path) -> None:
    entry = {"download_url": "https://dl.example.com/model.pt"}
    assert _extract_download_url(entry, "m") == "https://dl.example.com/model.pt"


def test_extract_download_url_metadata_download_url(tmp_path: Path) -> None:
    entry = {"metadata": {"download_url": "https://meta.example.com/model.pt"}}
    assert _extract_download_url(entry, "m") == "https://meta.example.com/model.pt"


def test_extract_download_url_metadata_artifact_url(tmp_path: Path) -> None:
    entry = {"metadata": {"artifact_url": "https://artifact.example.com/model.pt"}}
    assert _extract_download_url(entry, "m") == "https://artifact.example.com/model.pt"


def test_extract_download_url_top_level_takes_priority_over_metadata(tmp_path: Path) -> None:
    entry = {
        "download_url": "https://top.example.com/model.pt",
        "metadata": {"download_url": "https://meta.example.com/model.pt"},
    }
    assert _extract_download_url(entry, "m") == "https://top.example.com/model.pt"


def test_extract_download_url_missing_returns_none(tmp_path: Path) -> None:
    assert _extract_download_url({}, "m") is None


# ---------------------------------------------------------------------------
# _extract_checksum
# ---------------------------------------------------------------------------


def test_extract_checksum_top_level(tmp_path: Path) -> None:
    entry = {"checksum_sha256": "abc123"}
    assert _extract_checksum(entry) == "abc123"


def test_extract_checksum_sha256_alias(tmp_path: None) -> None:
    entry = {"sha256": "def456"}
    assert _extract_checksum(entry) == "def456"


def test_extract_checksum_from_metadata(tmp_path: None) -> None:
    entry = {"metadata": {"checksum_sha256": "ghi789"}}
    assert _extract_checksum(entry) == "ghi789"


def test_extract_checksum_missing_returns_none(tmp_path: None) -> None:
    assert _extract_checksum({}) is None


# ---------------------------------------------------------------------------
# _extract_runtime
# ---------------------------------------------------------------------------


def test_extract_runtime_top_level(tmp_path: None) -> None:
    assert _extract_runtime({"runtime": "ultralytics"}) == "ultralytics"


def test_extract_runtime_framework_fallback(tmp_path: None) -> None:
    assert _extract_runtime({"framework": "onnxruntime"}) == "onnxruntime"


def test_extract_runtime_from_metadata(tmp_path: None) -> None:
    assert _extract_runtime({"metadata": {"runtime": "tflite"}}) == "tflite"


def test_extract_runtime_missing_returns_none(tmp_path: None) -> None:
    assert _extract_runtime({}) is None


# ---------------------------------------------------------------------------
# _derive_filename
# ---------------------------------------------------------------------------


def test_derive_filename_from_catalog_key(tmp_path: None) -> None:
    assert _derive_filename("m", {"filename": "weights.pt"}, "https://x.com/x.pt") == "weights.pt"


def test_derive_filename_from_metadata_key(tmp_path: None) -> None:
    entry = {"metadata": {"filename": "det.onnx"}}
    assert _derive_filename("m", entry, "https://x.com/x.pt") == "det.onnx"


def test_derive_filename_from_url(tmp_path: None) -> None:
    assert _derive_filename("m", {}, "https://dl.example.com/yolov8n.pt") == "yolov8n.pt"


def test_derive_filename_from_url_strips_query_string(tmp_path: None) -> None:
    assert _derive_filename("m", {}, "https://s3.example.com/yolov8n.pt?token=abc") == "yolov8n.pt"


def test_derive_filename_fallback_to_model_id(tmp_path: None) -> None:
    assert _derive_filename("my-model", {}, "https://example.com/path/") == "my-model.pt"


# ---------------------------------------------------------------------------
# _sha256_file
# ---------------------------------------------------------------------------


def test_sha256_file(tmp_path: Path) -> None:
    content = b"hello world"
    f = tmp_path / "test.bin"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert _sha256_file(f) == expected


# ---------------------------------------------------------------------------
# _utc_iso_now
# ---------------------------------------------------------------------------


def test_utc_iso_now_format() -> None:
    ts = _utc_iso_now()
    assert ts.endswith("Z")
    assert "T" in ts
    assert len(ts) == 20


# ---------------------------------------------------------------------------
# _Manifest
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = _Manifest()
    model = CachedModel(
        model_id="yolov8n",
        local_path="/app/models/yolov8n/yolov8n.pt",
        size_bytes=12_345_678,
        downloaded_at="2026-01-01T00:00:00Z",
        source_url="https://example.com/model.pt",
        checksum_sha256="abc",
        runtime="ultralytics",
    )
    manifest.set(model)
    path = tmp_path / "manifest.json"
    manifest.save(path)

    loaded = _Manifest.load(path)
    got = loaded.get("yolov8n")
    assert got is not None
    assert got.model_id == "yolov8n"
    assert got.size_bytes == 12_345_678
    assert got.runtime == "ultralytics"


def test_manifest_remove(tmp_path: Path) -> None:
    manifest = _Manifest()
    manifest.set(CachedModel("m", "/p", 0, "2026-01-01T00:00:00Z"))
    assert manifest.remove("m") is True
    assert manifest.get("m") is None
    assert manifest.remove("m") is False


def test_manifest_load_missing_file(tmp_path: Path) -> None:
    m = _Manifest.load(tmp_path / "no_file.json")
    assert m.entries == {}


def test_manifest_load_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.json"
    bad.write_text("not-json{{")
    m = _Manifest.load(bad)
    assert m.entries == {}


# ---------------------------------------------------------------------------
# Checksum mismatch during _download_model raises RuntimeError
# ---------------------------------------------------------------------------


def test_download_model_checksum_mismatch_raises(tmp_path: Path) -> None:
    model_id = "bad-model"
    good_content = b"real weights"
    bad_checksum = hashlib.sha256(b"different").hexdigest()

    catalog_entry: dict[str, Any] = {
        "download_url": "https://dl.example.com/bad-model.pt",
        "checksum_sha256": bad_checksum,
        "filename": "bad-model.pt",
    }
    manager = _make_manager(tmp_path)

    def _fake_stream(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(good_content)

    with (
        patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry),
        patch.object(manager, "_stream_download", side_effect=_fake_stream),
    ):
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            manager.ensure_model(model_id)

    dest = tmp_path / model_id / "bad-model.pt"
    assert not dest.exists()


# ---------------------------------------------------------------------------
# ensure_model with empty/whitespace model_id raises ValueError
# ---------------------------------------------------------------------------


def test_ensure_model_empty_id_raises(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        manager.ensure_model("")

    with pytest.raises(ValueError, match="non-empty"):
        manager.ensure_model("   ")


# ---------------------------------------------------------------------------
# Source priority: signed Cyberwave URL preferred over upstream download_url
# ---------------------------------------------------------------------------


def test_download_prefers_artifact_url_over_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both a signed /weights URL and an upstream download_url are
    available, Edge Core hits the signed URL first."""
    model_id = "yolov8n"
    content = b"signed weights"
    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "download_url": "https://upstream.example.com/yolov8n.pt",
        "checksum_sha256": hashlib.sha256(content).hexdigest(),
        "filename": f"{model_id}.pt",
        "runtime": "ultralytics",
    }
    artifact_url = "https://signed.googleapis.com/yolov8n.pt?Signature=xyz"
    monkeypatch.setattr(ModelManager, "_fetch_artifact_url_safe", lambda self, entry: artifact_url)

    manager = _make_manager(tmp_path)
    hits: list[str] = []

    def _fake_stream(url: str, dest: Path) -> None:
        hits.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

    with (
        patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry),
        patch.object(manager, "_stream_download", side_effect=_fake_stream),
    ):
        manager.ensure_model(model_id)

    assert hits == [artifact_url]
    sidecar = tmp_path / model_id / MODEL_METADATA_FILENAME
    meta = json.loads(sidecar.read_text())
    assert meta["downloaded_from"] == SOURCE_KIND_ARTIFACT
    # The signed URL we fetched expires within minutes; persisting it
    # would mislead anyone reading the manifest later. Use the catalog
    # entry to refresh.
    assert meta["source_url"] is None
    assert meta["upstream_url"] == catalog_entry["download_url"]


def test_download_falls_back_to_upstream_when_artifact_url_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the signed URL download fails (e.g. expired token), Edge Core
    transparently retries against the upstream download_url."""
    model_id = "yolov8n"
    content = b"upstream weights"
    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "download_url": "https://upstream.example.com/yolov8n.pt",
        "checksum_sha256": hashlib.sha256(content).hexdigest(),
        "filename": f"{model_id}.pt",
    }
    artifact_url = "https://signed.googleapis.com/expired"
    monkeypatch.setattr(ModelManager, "_fetch_artifact_url_safe", lambda self, entry: artifact_url)

    manager = _make_manager(tmp_path)
    hits: list[str] = []

    def _fake_stream(url: str, dest: Path) -> None:
        hits.append(url)
        if url == artifact_url:
            raise RuntimeError("403 expired")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

    # Disable retries so the test is fast.
    monkeypatch.setattr("cyberwave_edge_core.model_manager.MAX_DOWNLOAD_RETRIES", 1)

    with (
        patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry),
        patch.object(manager, "_stream_download", side_effect=_fake_stream),
    ):
        manager.ensure_model(model_id)

    assert hits == [artifact_url, catalog_entry["download_url"]]
    sidecar = tmp_path / model_id / MODEL_METADATA_FILENAME
    meta = json.loads(sidecar.read_text())
    assert meta["downloaded_from"] == SOURCE_KIND_UPSTREAM
    assert meta["source_url"] == catalog_entry["download_url"]


def test_download_raises_when_no_sources_available(tmp_path: Path) -> None:
    """A catalog entry with neither signed URL nor download_url is fatal."""
    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "name": "broken-entry",
    }
    manager = _make_manager(tmp_path)
    with patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry):
        with pytest.raises(RuntimeError, match="No download sources"):
            manager.ensure_model("broken")


# ---------------------------------------------------------------------------
# Runtime-managed fallback: catalog has no URL but edge_runtime ships one
# ---------------------------------------------------------------------------


def test_download_runtime_managed_succeeds_when_no_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An Ultralytics entry with no download URL falls back to the runtime
    downloader and records the result as SOURCE_KIND_RUNTIME_MANAGED."""
    model_id = "yoloe-26s-seg"
    content = b"fake yoloe weights"

    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "name": "YOLOE-26 Small",
        "edge_runtime": "ultralytics",
        "model_external_id": "yoloe-26s-seg.pt",
    }

    def _fake_downloader(filename: str, dest_dir: Path) -> Path:
        assert filename == "yoloe-26s-seg.pt"
        out = dest_dir / filename
        out.write_bytes(content)
        return out

    monkeypatch.setitem(
        _model_manager_mod._RUNTIME_SELF_DOWNLOADERS,
        "ultralytics",
        _fake_downloader,
    )

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry):
        result = manager.ensure_model(model_id)

    expected = tmp_path / model_id / "yoloe-26s-seg.pt"
    assert result == expected
    assert result.read_bytes() == content

    sidecar = tmp_path / model_id / MODEL_METADATA_FILENAME
    meta = json.loads(sidecar.read_text())
    assert meta["downloaded_from"] == SOURCE_KIND_RUNTIME_MANAGED
    assert meta["runtime"] == "ultralytics"
    assert meta["filename"] == "yoloe-26s-seg.pt"
    assert meta["source_url"] is None
    assert meta["upstream_url"] is None
    assert meta["checksum_sha256"] == hashlib.sha256(content).hexdigest()


def test_download_runtime_managed_faster_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """faster-whisper STT entries resolve via catalog model_external_id."""
    model_id = "tiny.en"
    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "edge_runtime": "faster_whisper",
        "model_external_id": "tiny.en",
        "metadata": {
            "faster_whisper_model_id": "tiny.en",
            "edge_runtime": "faster_whisper",
        },
    }

    class _FakeWhisperModel:
        def __init__(self, fw_id: str, **kwargs: Any) -> None:
            self.fw_id = fw_id
            self.kwargs = kwargs

    fake_module = ModuleType("faster_whisper")
    fake_module.WhisperModel = _FakeWhisperModel

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        fake_module,
    )

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry):
        result = manager.ensure_model(model_id)

    marker = tmp_path / model_id / ".faster_whisper_ready"
    assert result == marker
    assert marker.read_text(encoding="utf-8") == "tiny.en"

    sidecar = tmp_path / model_id / MODEL_METADATA_FILENAME
    meta = json.loads(sidecar.read_text())
    assert meta["downloaded_from"] == SOURCE_KIND_RUNTIME_MANAGED
    assert meta["runtime"] == "faster_whisper"


def test_download_runtime_managed_skipped_for_unsupported_runtime(
    tmp_path: Path,
) -> None:
    """An entry with no URL and no self-downloading runtime still errors —
    the runtime-managed fallback only fires for known runtimes."""
    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "edge_runtime": "tflite",  # not in _RUNTIME_SELF_DOWNLOADERS
        "model_external_id": "model.tflite",
    }
    manager = _make_manager(tmp_path)
    with patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry):
        with pytest.raises(RuntimeError, match="No download sources"):
            manager.ensure_model("unsupported-runtime")


def test_download_runtime_managed_failure_wraps_with_workarounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the runtime downloader raises, the RuntimeError tells the
    operator how to recover (drop file / upload to backend / set
    download_url)."""
    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "edge_runtime": "ultralytics",
        "model_external_id": "yoloe-26s-seg.pt",
    }

    def _fake_downloader(filename: str, dest_dir: Path) -> Path:
        raise RuntimeError("ultralytics not installed")

    monkeypatch.setitem(
        _model_manager_mod._RUNTIME_SELF_DOWNLOADERS,
        "ultralytics",
        _fake_downloader,
    )

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry):
        with pytest.raises(RuntimeError) as excinfo:
            manager.ensure_model("yoloe-26s-seg")

    msg = str(excinfo.value)
    assert "Runtime-managed download" in msg
    assert "ultralytics not installed" in msg
    # Operator workarounds must be discoverable from the error text alone.
    assert "drop the weight file" in msg
    assert "/api/v1/mlmodels/" in msg
    assert "metadata.download_url" in msg


def test_download_runtime_managed_failure_prunes_orphan_staging_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the ``IsADirectoryError`` wedge: a failed
    runtime-managed download must NOT leave an empty staging directory
    behind, otherwise the SDK's path resolver in the worker container
    would later route the directory into ``torch.load`` and crash on
    every restart."""
    model_id = "yoloe-26s-seg"
    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "edge_runtime": "ultralytics",
        "model_external_id": "yoloe-26s-seg.pt",
    }

    def _fake_downloader(filename: str, dest_dir: Path) -> Path:
        # Touch a temp partial-download file the way the real
        # ``_stream_download`` would, then bail out — simulates an
        # interrupted Ultralytics hub fetch.
        (dest_dir / ".dl_xyz.part").write_bytes(b"partial")
        raise RuntimeError("ultralytics hub returned 404 for yoloe-26s-seg.pt")

    monkeypatch.setitem(
        _model_manager_mod._RUNTIME_SELF_DOWNLOADERS,
        "ultralytics",
        _fake_downloader,
    )

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry):
        with pytest.raises(RuntimeError, match="Runtime-managed download"):
            manager.ensure_model(model_id)

    # The empty / cruft-only staging dir must be gone after the failure.
    assert not (tmp_path / model_id).exists(), (
        "orphan staging directory must be pruned after a failed download "
        "so the SDK's path resolver does not later mistake it for a model file"
    )


def test_download_runtime_managed_failure_preserves_dir_with_operator_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Self-heal must NOT delete operator-staged content. If the staging
    directory contains a non-cruft file (e.g. an operator-dropped
    README + a partial weight file), the prune is skipped — human
    always wins.

    Two non-cruft files are used so that
    :meth:`_resolve_prestaged_weight_file` returns ``None`` (it requires
    *exactly one* candidate to claim a directory as pre-staged) and the
    flow reaches the download path, where the prune logic gets
    exercised.
    """
    model_id = "yoloe-26s-seg"
    model_dir = tmp_path / model_id
    model_dir.mkdir()
    (model_dir / "operator-weights.pt").write_bytes(b"hand-staged")
    (model_dir / "operator-notes.txt").write_text("staged by ops on 2026-05-19")

    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "edge_runtime": "ultralytics",
        "model_external_id": "yoloe-26s-seg.pt",
    }

    def _fake_downloader(filename: str, dest_dir: Path) -> Path:
        raise RuntimeError("network down")

    monkeypatch.setitem(
        _model_manager_mod._RUNTIME_SELF_DOWNLOADERS,
        "ultralytics",
        _fake_downloader,
    )

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry):
        with pytest.raises(RuntimeError, match="Runtime-managed download"):
            manager.ensure_model(model_id)

    assert model_dir.exists()
    assert (model_dir / "operator-weights.pt").read_bytes() == b"hand-staged"
    assert (model_dir / "operator-notes.txt").exists()


def test_download_model_checksum_mismatch_prunes_orphan_staging_dir(
    tmp_path: Path,
) -> None:
    """A signed-URL / upstream-URL download that fails checksum
    verification must also leave the cache in a clean state."""
    model_id = "bad-model"
    good_content = b"real weights"
    bad_checksum = hashlib.sha256(b"different").hexdigest()

    catalog_entry: dict[str, Any] = {
        "download_url": "https://dl.example.com/bad-model.pt",
        "checksum_sha256": bad_checksum,
        "filename": "bad-model.pt",
    }

    def _fake_stream(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(good_content)

    manager = _make_manager(tmp_path)
    with (
        patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry),
        patch.object(manager, "_stream_download", side_effect=_fake_stream),
    ):
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            manager.ensure_model(model_id)

    # The weight file is unlinked by the checksum-mismatch branch and the
    # now-empty staging directory must also be removed by the self-heal
    # at the end of ``_download_model``.
    assert not (tmp_path / model_id).exists()


def test_download_runtime_managed_warm_cache_skips_downloader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a runtime-managed download succeeds, subsequent ensure_model
    calls must hit the warm cache and not re-invoke the runtime."""
    model_id = "yoloe-26s-seg"
    content = b"fake yoloe weights"
    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "edge_runtime": "ultralytics",
        "model_external_id": "yoloe-26s-seg.pt",
    }

    call_count = {"n": 0}

    def _fake_downloader(filename: str, dest_dir: Path) -> Path:
        call_count["n"] += 1
        out = dest_dir / filename
        out.write_bytes(content)
        return out

    monkeypatch.setitem(
        _model_manager_mod._RUNTIME_SELF_DOWNLOADERS,
        "ultralytics",
        _fake_downloader,
    )

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry):
        manager.ensure_model(model_id)
    assert call_count["n"] == 1

    # Second instance picks up the manifest from disk and serves from cache.
    manager2 = _make_manager(tmp_path)
    with patch.object(
        manager2, "_download_model", side_effect=AssertionError("must not re-download")
    ):
        path2 = manager2.ensure_model(model_id)
    assert path2 == tmp_path / model_id / "yoloe-26s-seg.pt"
    assert call_count["n"] == 1


def test_extract_runtime_reads_edge_runtime() -> None:
    """The seed catalog uses ``edge_runtime``; the resolver must honor it
    as a synonym for ``runtime`` so the runtime-managed downloader can
    fire on entries authored before this change."""
    assert _extract_runtime({"edge_runtime": "ultralytics"}) == "ultralytics"
    assert _extract_runtime({"metadata": {"edge_package": "ultralytics"}}) == "ultralytics"
    # Explicit ``runtime`` still wins when both are set.
    assert (
        _extract_runtime({"runtime": "onnxruntime", "edge_runtime": "ultralytics"})
        == "onnxruntime"
    )


def test_derive_filename_reads_model_external_id() -> None:
    """When no explicit filename and no URL are available (runtime-managed
    case), the catalog's ``model_external_id`` is used as the filename."""
    entry = {"model_external_id": "yoloe-26s-seg.pt"}
    assert _derive_filename("yoloe-26s-seg", entry, "") == "yoloe-26s-seg.pt"
    # Explicit filename still wins.
    entry_explicit = {
        "filename": "custom.pt",
        "model_external_id": "yoloe-26s-seg.pt",
    }
    assert _derive_filename("yoloe-26s-seg", entry_explicit, "") == "custom.pt"
    # An external id without an extension must not be mistaken for a filename.
    entry_no_ext = {"model_external_id": "openvla-foo-v1"}
    assert _derive_filename("m", entry_no_ext, "") == "m.pt"


# ---------------------------------------------------------------------------
# Air-gap / fail-soft: cached file used when download fails
# ---------------------------------------------------------------------------


def test_ensure_model_falls_back_to_cached_when_download_fails(
    tmp_path: Path,
) -> None:
    """If a model is intact in the cache but the network is unreachable,
    ensure_model returns the cached path instead of raising."""
    model_id = "yolov8n"
    dest = _write_fake_model(tmp_path, model_id, b"local weights")
    manager = _make_manager(tmp_path)

    with patch.object(
        manager,
        "_download_model",
        side_effect=RuntimeError("network unreachable"),
    ):
        # Force the warm-cache fast path to attempt a refresh by signaling
        # that the catalog has a different checksum.
        with patch.object(manager, "_catalog_indicates_refresh", return_value=True):
            result = manager.ensure_model(model_id)

    assert result == dest
    assert result.read_bytes() == b"local weights"


def test_ensure_model_no_cache_no_network_raises(tmp_path: Path) -> None:
    """Cold cache + unreachable network → RuntimeError. There is no
    fail-soft for a model that has never been downloaded."""
    manager = _make_manager(tmp_path)
    with patch.object(
        manager,
        "_download_model",
        side_effect=RuntimeError("offline"),
    ):
        with pytest.raises(RuntimeError, match="offline"):
            manager.ensure_model("never-seen")


# ---------------------------------------------------------------------------
# Catalog refresh probe: triggers re-download when checksum drifts
# ---------------------------------------------------------------------------


def test_warm_cache_with_matching_catalog_checksum_skips_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm cache + catalog reachable + checksums match → no download."""
    model_id = "yolov8n"
    dest = _write_fake_model(tmp_path, model_id, b"weights")
    cached_checksum = hashlib.sha256(b"weights").hexdigest()

    monkeypatch.setattr(
        ModelManager,
        "_fetch_catalog_entry_safe",
        lambda self, mid: {"checksum_sha256": cached_checksum},
    )
    manager = _make_manager(tmp_path)

    with patch.object(manager, "_download_model", side_effect=AssertionError("must not download")):
        result = manager.ensure_model(model_id)
    assert result == dest


def test_warm_cache_with_drifted_catalog_checksum_triggers_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm cache + catalog reachable + checksums differ → re-download."""
    model_id = "yolov8n"
    _write_fake_model(tmp_path, model_id, b"old weights")
    new_content = b"new weights"
    new_checksum = hashlib.sha256(new_content).hexdigest()

    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "download_url": "https://upstream.example.com/yolov8n.pt",
        "checksum_sha256": new_checksum,
        "filename": f"{model_id}.pt",
    }
    monkeypatch.setattr(
        ModelManager,
        "_fetch_catalog_entry_safe",
        lambda self, mid: catalog_entry,
    )

    manager = _make_manager(tmp_path)
    download_called: list[str] = []

    def _fake_stream(url: str, dest: Path) -> None:
        download_called.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(new_content)

    with (
        patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry),
        patch.object(manager, "_stream_download", side_effect=_fake_stream),
    ):
        result = manager.ensure_model(model_id)

    assert download_called == [catalog_entry["download_url"]]
    assert result.read_bytes() == new_content
    cached = _Manifest.load(tmp_path / "manifest.json").get(model_id)
    assert cached is not None
    assert cached.checksum_sha256 == new_checksum


def test_warm_cache_unreachable_catalog_uses_cached(
    tmp_path: Path,
) -> None:
    """Warm cache + catalog unreachable (probe returns None) → cached path,
    no download attempted. This is the air-gap default."""
    model_id = "yolov8n"
    dest = _write_fake_model(tmp_path, model_id, b"weights")
    manager = _make_manager(tmp_path)

    # The autouse fixture already sets _fetch_catalog_entry_safe → None.
    with patch.object(manager, "_download_model", side_effect=AssertionError("must not download")):
        result = manager.ensure_model(model_id)
    assert result == dest


# ---------------------------------------------------------------------------
# Disk reconciliation: pre-staged files (air-gapped operator workflow)
# ---------------------------------------------------------------------------


def test_reconcile_picks_up_prestaged_weight_file_without_sidecar(
    tmp_path: Path,
) -> None:
    """An operator drops ``cache_dir/{model_id}/weights.pt`` from a USB
    stick. ensure_model picks it up, computes a checksum, and writes a
    sidecar so the file is treated as a normal cache hit thereafter."""
    model_id = "yolov8n"
    model_dir = tmp_path / model_id
    model_dir.mkdir(parents=True)
    weight_path = model_dir / "weights.pt"
    weight_path.write_bytes(b"prestaged-bytes")

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_download_model", side_effect=AssertionError("must not download")):
        result = manager.ensure_model(model_id)

    assert result == weight_path
    sidecar = model_dir / MODEL_METADATA_FILENAME
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["downloaded_from"] == SOURCE_KIND_PRESTAGED
    assert meta["filename"] == "weights.pt"
    assert meta["checksum_sha256"] == hashlib.sha256(b"prestaged-bytes").hexdigest()


def test_reconcile_honors_existing_sidecar(tmp_path: Path) -> None:
    """When a sidecar already exists, reconcile uses its filename and
    checksum verbatim (the operator may have populated those fields by
    hand to enable corruption detection)."""
    model_id = "detector"
    model_dir = tmp_path / model_id
    model_dir.mkdir(parents=True)
    weight_path = model_dir / "model.onnx"
    content = b"onnx-bytes"
    weight_path.write_bytes(content)
    sidecar_checksum = hashlib.sha256(content).hexdigest()
    (model_dir / MODEL_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "model_id": model_id,
                "filename": "model.onnx",
                "runtime": "onnxruntime",
                "checksum_sha256": sidecar_checksum,
            }
        )
    )

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_download_model", side_effect=AssertionError("must not download")):
        result = manager.ensure_model(model_id)

    assert result == weight_path
    cached = _Manifest.load(tmp_path / "manifest.json").get(model_id)
    assert cached is not None
    assert cached.runtime == "onnxruntime"
    assert cached.checksum_sha256 == sidecar_checksum


def test_reconcile_skips_dir_with_multiple_unidentified_weights(
    tmp_path: Path,
) -> None:
    """Two weight files in the dir without a sidecar → cannot disambiguate;
    reconcile is a no-op and ensure_model attempts to download instead."""
    model_id = "ambiguous"
    model_dir = tmp_path / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "a.pt").write_bytes(b"a")
    (model_dir / "b.pt").write_bytes(b"b")

    manager = _make_manager(tmp_path)
    download_called: list[str] = []

    def _fake_download(mid: str) -> Path:
        download_called.append(mid)
        raise RuntimeError("network down for test")

    with patch.object(manager, "_download_model", side_effect=_fake_download):
        with pytest.raises(RuntimeError, match="network down"):
            manager.ensure_model(model_id)

    assert download_called == [model_id]
    cached = _Manifest.load(tmp_path / "manifest.json").get(model_id)
    assert cached is None


def test_reconcile_ignores_dotfiles_and_temp_downloads(tmp_path: Path) -> None:
    """In-flight ``.dl_*`` temp files and other dotfiles must not count
    as candidate weights during reconciliation."""
    model_id = "yolov8n"
    model_dir = tmp_path / model_id
    model_dir.mkdir(parents=True)
    weight_path = model_dir / "yolov8n.pt"
    weight_path.write_bytes(b"weights")
    (model_dir / ".dl_partial").write_bytes(b"junk")
    (model_dir / ".DS_Store").write_bytes(b"junk")

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_download_model", side_effect=AssertionError("must not download")):
        result = manager.ensure_model(model_id)

    assert result == weight_path


# ---------------------------------------------------------------------------
# _extract_download_url: new upstream_weights_url alias
# ---------------------------------------------------------------------------


def test_extract_download_url_metadata_upstream_weights_url(tmp_path: Path) -> None:
    entry = {"metadata": {"upstream_weights_url": "https://up.example.com/yolov8n.pt"}}
    assert _extract_download_url(entry, "m") == "https://up.example.com/yolov8n.pt"


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def test_redact_url_strips_query_string() -> None:
    assert (
        _redact_url("https://signed.googleapis.com/x.pt?Signature=secret&Key=foo")
        == "https://signed.googleapis.com/x.pt"
    )
    assert _redact_url("https://example.com/x.pt") == "https://example.com/x.pt"


# ---------------------------------------------------------------------------
# Re-staging: operator overwrites a pre-staged file in place
# ---------------------------------------------------------------------------


def _write_prestaged_model(cache_dir: Path, model_id: str, content: bytes) -> Path:
    """Pre-stage a model on disk and reconcile it into the manifest.

    Returns the path to the weight file. The sidecar is written by
    ensure_model's reconciliation step.
    """
    model_dir = cache_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / f"{model_id}.pt"
    dest.write_bytes(content)
    manager = ModelManager(cache_dir=cache_dir, api_token="t", base_url="https://api.test")
    with patch.object(manager, "_download_model", side_effect=AssertionError("must not download")):
        manager.ensure_model(model_id)
    return dest


def test_restaging_in_place_offline_does_not_brick_worker(tmp_path: Path) -> None:
    """Operator overwrites a prestaged file with a new build. On the next
    ensure_model call, even with the network down, the worker should
    successfully resolve the new file (not raise 'corrupt' + try to
    download)."""
    model_id = "yolov8n"
    weight_path = _write_prestaged_model(tmp_path, model_id, b"v1-weights")

    new_content = b"v2-weights-with-different-size-XXXXXX"
    weight_path.write_bytes(new_content)

    manager = _make_manager(tmp_path)
    with patch.object(
        manager,
        "_download_model",
        side_effect=AssertionError("must not download for prestaged updates"),
    ):
        result = manager.ensure_model(model_id)

    assert result == weight_path
    assert result.read_bytes() == new_content
    cached = _Manifest.load(tmp_path / "manifest.json").get(model_id)
    assert cached is not None
    assert cached.checksum_sha256 == hashlib.sha256(new_content).hexdigest()
    assert cached.size_bytes == len(new_content)
    sidecar = json.loads((weight_path.parent / MODEL_METADATA_FILENAME).read_text())
    assert sidecar["downloaded_from"] == SOURCE_KIND_PRESTAGED
    assert sidecar["checksum_sha256"] == cached.checksum_sha256


def test_restaging_in_place_handles_same_size_overwrite(tmp_path: Path) -> None:
    """Same-size overwrites are unusual but possible (e.g. quantized
    re-export of the same architecture). The restamp path keys off the
    SHA-256 mismatch reported by ``_cache_is_intact``, not file size, so
    these are handled too."""
    model_id = "yolov8n"
    original = b"weights-of-fixed-size"
    replacement = b"different-of-same-len"
    assert len(original) == len(replacement)

    weight_path = _write_prestaged_model(tmp_path, model_id, original)
    weight_path.write_bytes(replacement)

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_download_model", side_effect=AssertionError("must not download")):
        result = manager.ensure_model(model_id)

    assert result == weight_path
    cached_after = _Manifest.load(tmp_path / "manifest.json").get(model_id)
    assert cached_after is not None
    assert cached_after.checksum_sha256 == hashlib.sha256(replacement).hexdigest()


def test_unchanged_prestaged_file_does_not_restamp(tmp_path: Path) -> None:
    """Steady-state: when nothing has changed on disk, ensure_model must
    not rewrite the manifest. Pointless writes risk pulling in noisy
    fsync churn on flash media in field deployments."""
    model_id = "yolov8n"
    weight_path = _write_prestaged_model(tmp_path, model_id, b"unchanged-weights")
    manifest_path = tmp_path / "manifest.json"
    mtime_before = manifest_path.stat().st_mtime_ns

    time.sleep(0.01)  # ensure mtime would change if rewritten

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_download_model", side_effect=AssertionError("must not download")):
        result = manager.ensure_model(model_id)

    assert result == weight_path
    assert manifest_path.stat().st_mtime_ns == mtime_before


def test_restaging_does_not_apply_to_downloaded_files(tmp_path: Path) -> None:
    """Files marked ``downloaded_from: download_url`` keep the
    corruption-detection semantics. Bit-rot or a half-written download
    should still trigger a re-download attempt rather than being silently
    accepted."""
    model_id = "yolov8n"
    dest = _write_fake_model(tmp_path, model_id, b"original-weights")
    sidecar_path = dest.parent / MODEL_METADATA_FILENAME
    sidecar_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "filename": dest.name,
                "checksum_sha256": hashlib.sha256(b"original-weights").hexdigest(),
                "size_bytes": len(b"original-weights"),
                "downloaded_from": SOURCE_KIND_UPSTREAM,
            }
        )
    )

    dest.write_bytes(b"corrupted-bytes-XXXXXX")  # different size + content

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_download_model", side_effect=RuntimeError("network down")):
        with pytest.raises(RuntimeError, match="network down"):
            manager.ensure_model(model_id)


# ---------------------------------------------------------------------------
# Auth-failure short-circuit in download retry loop
# ---------------------------------------------------------------------------


def test_download_retries_short_circuit_on_403(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expired GCS signed URL returns 403; retrying the same URL is
    pointless. ``_download_with_retries`` should bail immediately so the
    caller can fall through to the next source."""
    import httpx

    manager = _make_manager(tmp_path)
    attempts: list[int] = []

    def _fake_stream(url: str, dest: Path) -> None:
        attempts.append(len(attempts) + 1)
        request = httpx.Request("GET", url)
        response = httpx.Response(403, request=request)
        raise httpx.HTTPStatusError("forbidden", request=request, response=response)

    monkeypatch.setattr(manager, "_stream_download", _fake_stream)
    monkeypatch.setattr("cyberwave_edge_core.model_manager.MAX_DOWNLOAD_RETRIES", 5)

    with pytest.raises(RuntimeError, match="Authentication failed"):
        manager._download_with_retries("https://signed/x.pt", tmp_path / "x.pt")
    assert attempts == [1]


def test_download_retries_continue_on_500(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Server-side errors (5xx) and network errors are still retried."""
    import httpx

    manager = _make_manager(tmp_path)
    attempts: list[int] = []

    def _fake_stream(url: str, dest: Path) -> None:
        attempts.append(len(attempts) + 1)
        request = httpx.Request("GET", url)
        response = httpx.Response(500, request=request)
        raise httpx.HTTPStatusError("server error", request=request, response=response)

    monkeypatch.setattr(manager, "_stream_download", _fake_stream)
    monkeypatch.setattr("cyberwave_edge_core.model_manager.MAX_DOWNLOAD_RETRIES", 3)
    monkeypatch.setattr("cyberwave_edge_core.model_manager.DOWNLOAD_RETRY_BASE_DELAY", 0.0)

    with pytest.raises(RuntimeError, match="Download failed after 3 attempts"):
        manager._download_with_retries("https://upstream/x.pt", tmp_path / "x.pt")
    assert attempts == [1, 2, 3]


def test_signed_url_403_falls_through_to_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a 403 from the signed URL must not block the upstream
    fallback. Combined with the retry short-circuit, this means an
    expired signed URL costs a single HTTP attempt before we move on."""
    import httpx

    model_id = "yolov8n"
    content = b"upstream weights"
    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "download_url": "https://upstream.example.com/yolov8n.pt",
        "checksum_sha256": hashlib.sha256(content).hexdigest(),
        "filename": f"{model_id}.pt",
    }
    artifact_url = "https://signed.googleapis.com/expired"
    monkeypatch.setattr(ModelManager, "_fetch_artifact_url_safe", lambda self, entry: artifact_url)

    manager = _make_manager(tmp_path)
    attempts: list[str] = []

    def _fake_stream(url: str, dest: Path) -> None:
        attempts.append(url)
        if url == artifact_url:
            request = httpx.Request("GET", url)
            response = httpx.Response(403, request=request)
            raise httpx.HTTPStatusError("forbidden", request=request, response=response)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

    with (
        patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry),
        patch.object(manager, "_stream_download", side_effect=_fake_stream),
    ):
        manager.ensure_model(model_id)

    assert attempts == [artifact_url, catalog_entry["download_url"]]


# ---------------------------------------------------------------------------
# Runtime inference for pre-staged files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected_runtime",
    [
        ("model.pt", "ultralytics"),
        ("model.onnx", "onnxruntime"),
        ("model.engine", "tensorrt"),
        ("model.tflite", "tflite"),
        ("model.pth", "torch"),
        ("model.xml", "opencv"),
        ("model.hef", "hailo"),
    ],
)
def test_reconcile_infers_runtime_from_extension(
    tmp_path: Path, filename: str, expected_runtime: str
) -> None:
    """When a pre-staged file has no sidecar, the reconciler infers a
    sensible default runtime from the file extension. Without this, the
    SDK falls back to its own filename heuristics, which only recognise
    well-known model_ids like 'yolov8n'."""
    model_id = "my-custom-model"
    model_dir = tmp_path / model_id
    model_dir.mkdir(parents=True)
    weight_path = model_dir / filename
    weight_path.write_bytes(b"opaque-bytes")

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_download_model", side_effect=AssertionError("must not download")):
        manager.ensure_model(model_id)

    cached = _Manifest.load(tmp_path / "manifest.json").get(model_id)
    assert cached is not None
    assert cached.runtime == expected_runtime
    sidecar = json.loads((model_dir / MODEL_METADATA_FILENAME).read_text())
    assert sidecar["runtime"] == expected_runtime


def test_reconcile_unknown_extension_leaves_runtime_none(tmp_path: Path) -> None:
    """Unknown extensions (e.g. .gguf) get runtime=None. The SDK is then
    free to apply its own heuristics or to error out at load time."""
    model_id = "exotic"
    model_dir = tmp_path / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "model.gguf").write_bytes(b"opaque-bytes")

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_download_model", side_effect=AssertionError("must not download")):
        manager.ensure_model(model_id)

    cached = _Manifest.load(tmp_path / "manifest.json").get(model_id)
    assert cached is not None
    assert cached.runtime is None


def test_reconcile_honors_explicit_sidecar_runtime_over_extension(
    tmp_path: Path,
) -> None:
    """An explicit ``runtime`` in the sidecar always wins over the
    extension-based inference (e.g. a .pt file actually loaded by the
    'torch' runtime, not 'ultralytics')."""
    model_id = "yolov8n"
    model_dir = tmp_path / model_id
    model_dir.mkdir(parents=True)
    weight_path = model_dir / "yolov8n.pt"
    weight_path.write_bytes(b"weights")
    (model_dir / MODEL_METADATA_FILENAME).write_text(
        json.dumps({"model_id": model_id, "filename": "yolov8n.pt", "runtime": "torch"})
    )

    manager = _make_manager(tmp_path)
    with patch.object(manager, "_download_model", side_effect=AssertionError("must not download")):
        manager.ensure_model(model_id)

    cached = _Manifest.load(tmp_path / "manifest.json").get(model_id)
    assert cached is not None
    assert cached.runtime == "torch"


# ---------------------------------------------------------------------------
# Pre-staged files are operator-curated truth: never auto-overwritten
# ---------------------------------------------------------------------------


def test_prestaged_file_skips_catalog_probe_and_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator-staged file is never auto-overwritten by Edge Core,
    even when the catalog publishes a different checksum. To force a
    re-download the operator must evict the model directory."""
    model_id = "yolov8n"
    weight_path = _write_prestaged_model(tmp_path, model_id, b"the-blessed-build")
    blessed_sha = hashlib.sha256(b"the-blessed-build").hexdigest()

    catalog_probe_calls: list[str] = []

    def _spy_probe(self: ModelManager, mid: str) -> Optional[dict[str, Any]]:
        catalog_probe_calls.append(mid)
        return {"checksum_sha256": "deadbeef" * 8}

    monkeypatch.setattr(ModelManager, "_fetch_catalog_entry_safe", _spy_probe)

    manager = _make_manager(tmp_path)
    with patch.object(
        manager,
        "_download_model",
        side_effect=AssertionError("prestaged: must not download"),
    ):
        result = manager.ensure_model(model_id)

    assert result == weight_path
    assert result.read_bytes() == b"the-blessed-build"
    assert catalog_probe_calls == [], "prestaged cache must not probe the catalog"
    cached = _Manifest.load(tmp_path / "manifest.json").get(model_id)
    assert cached is not None
    assert cached.downloaded_from == SOURCE_KIND_PRESTAGED
    assert cached.checksum_sha256 == blessed_sha


def test_downloaded_file_round_trips_downloaded_from_through_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``downloaded_from`` survives a manifest serialise/deserialise.

    This matters because the prestaged-skip-catalog logic above keys off
    of ``cached.downloaded_from`` after a fresh process load.
    """
    model_id = "yolov8n"
    content = b"upstream-weights"
    catalog_entry: dict[str, Any] = {
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "download_url": "https://upstream.example.com/yolov8n.pt",
        "checksum_sha256": hashlib.sha256(content).hexdigest(),
        "filename": f"{model_id}.pt",
    }

    def _fake_stream(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

    manager_a = _make_manager(tmp_path)
    with (
        patch.object(manager_a, "_fetch_catalog_entry", return_value=catalog_entry),
        patch.object(manager_a, "_stream_download", side_effect=_fake_stream),
    ):
        manager_a.ensure_model(model_id)

    raw = json.loads((tmp_path / "manifest.json").read_text())
    assert raw[model_id]["downloaded_from"] == SOURCE_KIND_UPSTREAM
    assert raw[model_id]["source_url"] == catalog_entry["download_url"]

    manager_b = _make_manager(tmp_path)
    cached = manager_b._manifest.get(model_id)
    assert cached is not None
    assert cached.downloaded_from == SOURCE_KIND_UPSTREAM
