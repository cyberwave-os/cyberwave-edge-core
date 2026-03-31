"""Unit tests for ModelManager (model_manager.py).

Covers:
- Cache hit returns path without download
- Cache miss triggers download path
- Manifest persistence across instances
- Checksum mismatch triggers re-download
- Eviction removes manifest entry and file
- Worker file scanning for model requirements
- cyberwave.yml model parsing
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cyberwave_edge_core.model_manager import ModelManager, _parse_cyberwave_yml_models


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture
def manager(cache_dir: Path) -> ModelManager:
    return ModelManager(cache_dir=cache_dir, api_token="test-token", base_url="http://localhost")


class TestModelManagerCacheHit:
    def test_returns_path_without_download_when_cached(
        self, manager: ModelManager, cache_dir: Path
    ) -> None:
        model_id = "yolov8n"
        model_dir = cache_dir / model_id
        model_dir.mkdir(parents=True)
        weight_file = model_dir / "yolov8n.pt"
        weight_file.write_bytes(b"fake-weights")

        checksum = manager._compute_checksum(weight_file)
        manifest = {
            model_id: {
                "model_id": model_id,
                "path": str(weight_file),
                "size_bytes": weight_file.stat().st_size,
                "downloaded_at": time.time(),
                "checksum": checksum,
                "source_url": None,
            }
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))

        download_calls: list = []
        with patch.object(manager, "_download_model", side_effect=download_calls.append):
            result = manager.ensure_model(model_id)

        assert result == weight_file
        assert not download_calls

    def test_returns_none_and_attempts_download_on_cache_miss(
        self, manager: ModelManager
    ) -> None:
        with patch.object(manager, "_download_model", return_value=None) as mock_dl:
            result = manager.ensure_model("nonexistent-model")

        mock_dl.assert_called_once_with("nonexistent-model")
        assert result is None

    def test_redownloads_when_checksum_mismatch(
        self, manager: ModelManager, cache_dir: Path
    ) -> None:
        model_id = "corrupted"
        model_dir = cache_dir / model_id
        model_dir.mkdir(parents=True)
        weight_file = model_dir / "weights.pt"
        weight_file.write_bytes(b"corrupted-data")

        manifest = {
            model_id: {
                "model_id": model_id,
                "path": str(weight_file),
                "size_bytes": weight_file.stat().st_size,
                "downloaded_at": time.time(),
                "checksum": "0000000000000000000000000000000000000000000000000000000000000000",
                "source_url": None,
            }
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))

        with patch.object(manager, "_download_model", return_value=None) as mock_dl:
            manager.ensure_model(model_id)

        mock_dl.assert_called_once_with(model_id)


class TestModelManagerManifestPersistence:
    def test_manifest_persists_across_instances(
        self, cache_dir: Path
    ) -> None:
        model_id = "persisted-model"
        model_dir = cache_dir / model_id
        model_dir.mkdir(parents=True)
        weight_file = model_dir / "weights.pt"
        weight_file.write_bytes(b"real-weights")

        m1 = ModelManager(cache_dir=cache_dir)
        checksum = m1._compute_checksum(weight_file)
        from cyberwave_edge_core.model_manager import CachedModel

        entry = CachedModel(
            model_id=model_id,
            path=str(weight_file),
            size_bytes=weight_file.stat().st_size,
            downloaded_at=time.time(),
            checksum=checksum,
        )
        manifest = {model_id: entry}
        m1._save_manifest(manifest)

        m2 = ModelManager(cache_dir=cache_dir)
        loaded = m2._load_manifest_entry(model_id)
        assert loaded is not None
        assert loaded.model_id == model_id
        assert loaded.path == str(weight_file)
        assert loaded.checksum == checksum


class TestModelManagerEviction:
    def test_evict_removes_manifest_entry(
        self, manager: ModelManager, cache_dir: Path
    ) -> None:
        model_id = "to-evict"
        model_dir = cache_dir / model_id
        model_dir.mkdir(parents=True)
        weight_file = model_dir / "weights.pt"
        weight_file.write_bytes(b"data")

        from cyberwave_edge_core.model_manager import CachedModel

        manifest = {
            model_id: CachedModel(
                model_id=model_id,
                path=str(weight_file),
                size_bytes=4,
                downloaded_at=time.time(),
                checksum="abc",
            )
        }
        manager._save_manifest(manifest)

        result = manager.evict_model(model_id)
        assert result is True
        assert not weight_file.exists()
        assert manager._load_manifest_entry(model_id) is None

    def test_evict_returns_false_when_not_cached(self, manager: ModelManager) -> None:
        result = manager.evict_model("nonexistent")
        assert result is False


class TestModelManagerCacheSize:
    def test_cache_size_sums_file_sizes(
        self, manager: ModelManager, cache_dir: Path
    ) -> None:
        model_id = "size-test"
        model_dir = cache_dir / model_id
        model_dir.mkdir(parents=True)
        weight_file = model_dir / "weights.pt"
        weight_file.write_bytes(b"0" * 1024)

        from cyberwave_edge_core.model_manager import CachedModel

        manager._save_manifest(
            {
                model_id: CachedModel(
                    model_id=model_id,
                    path=str(weight_file),
                    size_bytes=1024,
                    downloaded_at=time.time(),
                    checksum=manager._compute_checksum(weight_file),
                )
            }
        )
        assert manager.cache_size_bytes() == 1024


class TestWorkerFileScan:
    def test_detects_cw_models_load(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        (workers_dir / "detect.py").write_text(
            'import cw\nmodel = cw.models.load("yolov8n")\n'
        )
        model_ids = ModelManager.scan_worker_model_requirements(workers_dir)
        assert "yolov8n" in model_ids

    def test_detects_models_load_shorthand(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        (workers_dir / "worker.py").write_text(
            "from cyberwave import models\nm = models.load('bg-sub')\n"
        )
        model_ids = ModelManager.scan_worker_model_requirements(workers_dir)
        assert "bg-sub" in model_ids

    def test_deduplicates_model_ids(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        (workers_dir / "a.py").write_text('cw.models.load("yolov8n")\n')
        (workers_dir / "b.py").write_text('cw.models.load("yolov8n")\n')
        model_ids = ModelManager.scan_worker_model_requirements(workers_dir)
        assert model_ids.count("yolov8n") == 1

    def test_returns_empty_when_no_workers(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "empty_workers"
        model_ids = ModelManager.scan_worker_model_requirements(workers_dir)
        assert model_ids == []

    def test_reads_cyberwave_yml(self, tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        (workers_dir / "cyberwave.yml").write_text(
            "models:\n  - yolov8n\n  - background-sub\n"
        )
        model_ids = ModelManager.scan_worker_model_requirements(workers_dir)
        assert "yolov8n" in model_ids
        assert "background-sub" in model_ids


class TestParseCyberwaveYml:
    def test_parses_simple_list(self, tmp_path: Path) -> None:
        yml = tmp_path / "cyberwave.yml"
        yml.write_text("models:\n  - modelA\n  - modelB\n")
        assert _parse_cyberwave_yml_models(yml) == ["modelA", "modelB"]

    def test_stops_at_next_section(self, tmp_path: Path) -> None:
        yml = tmp_path / "cyberwave.yml"
        yml.write_text("models:\n  - modelA\nother:\n  - modelB\n")
        result = _parse_cyberwave_yml_models(yml)
        assert "modelA" in result
        assert "modelB" not in result

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        yml = tmp_path / "does_not_exist.yml"
        assert _parse_cyberwave_yml_models(yml) == []
