from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cyberwave_edge_core._version as version_module


def test_get_version_prefers_build_override(monkeypatch):
    monkeypatch.setattr(version_module, "BUILD_VERSION", "0.0.42.9")
    monkeypatch.setattr(version_module, "metadata_version", lambda _name: "0.0.42")

    assert version_module.get_version() == "0.0.42.9"


def test_get_version_falls_back_to_package_metadata(monkeypatch):
    monkeypatch.setattr(version_module, "BUILD_VERSION", None)
    monkeypatch.setattr(version_module, "metadata_version", lambda _name: "0.0.42")

    assert version_module.get_version() == "0.0.42"


def test_get_version_falls_back_to_static_version_on_metadata_error(monkeypatch):
    monkeypatch.setattr(version_module, "BUILD_VERSION", None)

    def _raise(_name: str) -> str:
        raise version_module.PackageNotFoundError

    monkeypatch.setattr(version_module, "metadata_version", _raise)

    assert version_module.get_version() == version_module.STATIC_VERSION
