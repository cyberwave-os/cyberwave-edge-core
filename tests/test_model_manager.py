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
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cyberwave_edge_core.model_manager import (
    CachedModel,
    ModelManager,
    _derive_filename,
    _extract_checksum,
    _extract_download_url,
    _extract_runtime,
    _Manifest,
    _sha256_file,
    _utc_iso_now,
    scan_worker_model_ids,
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
