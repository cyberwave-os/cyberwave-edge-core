"""Unit tests for cyberwave_edge_core.hailo_preflight.

Covers:
* ``normalize_arch`` — every wire format we've seen across HailoRT versions.
* ``host_hailo_device_present`` — Linux vs non-Linux short-circuit.
* ``host_hailo_arch`` — hailortcli missing, non-zero, timeout, parse miss,
  and the happy path for both Hailo-8 and Hailo-8L.
* ``_is_hailo_model`` — edge_runtime, filename, model_external_id, and
  metadata.edge_model_path signals (each sufficient on its own).
* ``preflight_hailo_arch`` — every silent-skip branch (non-Hailo,
  no device, no hw_arch, no probe result) and the one raising branch.
* End-to-end wiring through ``ModelManager._download_model``: a HEF
  catalog entry with the wrong arch short-circuits before any bytes
  hit the wire.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cyberwave_edge_core import hailo_preflight as hp
from cyberwave_edge_core.hailo_preflight import (
    HailoArchMismatchError,
    _is_hailo_model,
    _suggest_sibling_slug,
    host_hailo_arch,
    host_hailo_device_present,
    normalize_arch,
    preflight_hailo_arch,
)
from cyberwave_edge_core.model_manager import ModelManager

# ---------------------------------------------------------------------------
# normalize_arch
# ---------------------------------------------------------------------------


class TestNormalizeArch:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("HAILO8", "hailo8"),
            ("Hailo-8", "hailo8"),
            ("hailo8", "hailo8"),
            ("HAILO_ARCH_HAILO8", "hailo8"),
            ("HAILO_ARCH_HAILO_8L", "hailo8l"),
            ("HAILO8L", "hailo8l"),
            ("Hailo-8L", "hailo8l"),
            ("  HAILO8  ", "hailo8"),
            ("", ""),
        ],
    )
    def test_canonical_forms(self, raw: str, expected: str) -> None:
        assert normalize_arch(raw) == expected


# ---------------------------------------------------------------------------
# host_hailo_device_present
# ---------------------------------------------------------------------------


class TestHostHailoDevicePresent:
    def test_false_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp.platform, "system", lambda: "Darwin")
        # Even if the path exists somehow on a Mac, we never report True.
        monkeypatch.setattr(Path, "exists", lambda self: True)
        assert host_hailo_device_present() is False

    def test_true_when_path_exists_on_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp.platform, "system", lambda: "Linux")
        monkeypatch.setattr(Path, "exists", lambda self: True)
        assert host_hailo_device_present() is True

    def test_false_when_path_missing_on_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp.platform, "system", lambda: "Linux")
        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert host_hailo_device_present() is False


# ---------------------------------------------------------------------------
# host_hailo_arch
# ---------------------------------------------------------------------------


# Real ``hailortcli fw-control identify`` output from a Pi 5 + AI HAT+
# running HailoRT 4.23.0. Kept verbatim so the parser is exercised
# against the exact line layout HailoRT actually ships, not a synthetic
# approximation that could drift over time.
_HAILORTCLI_HAILO8_STDOUT = """\
Executing on device: 0000:01:00.0
Identifying board
Control Protocol Version: 2
Firmware Version: 4.23.0 (release,app,extended context switch buffer)
Logger Version: 0
Board Name: Hailo-8
Device Architecture: HAILO8
Serial Number: HLDDLBB242802044
Part Number: HM218B1C2FAE
Product Name: HAILO-8 AI ACCELERATOR
"""

_HAILORTCLI_HAILO8L_STDOUT = _HAILORTCLI_HAILO8_STDOUT.replace("HAILO8", "HAILO8L").replace(
    "Hailo-8", "Hailo-8L"
)


def _make_completed(stdout: str = "", returncode: int = 0, stderr: str = "") -> MagicMock:
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.stdout = stdout
    m.stderr = stderr
    m.returncode = returncode
    return m


class TestHostHailoArch:
    def test_returns_none_when_hailortcli_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp.shutil, "which", lambda name: None)
        assert host_hailo_arch() is None

    def test_returns_normalized_arch_for_hailo8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp.shutil, "which", lambda name: "/usr/bin/hailortcli")
        monkeypatch.setattr(
            hp.subprocess,
            "run",
            lambda *a, **kw: _make_completed(stdout=_HAILORTCLI_HAILO8_STDOUT),
        )
        assert host_hailo_arch() == "hailo8"

    def test_returns_normalized_arch_for_hailo8l(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp.shutil, "which", lambda name: "/usr/bin/hailortcli")
        monkeypatch.setattr(
            hp.subprocess,
            "run",
            lambda *a, **kw: _make_completed(stdout=_HAILORTCLI_HAILO8L_STDOUT),
        )
        assert host_hailo_arch() == "hailo8l"

    def test_returns_none_when_hailortcli_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp.shutil, "which", lambda name: "/usr/bin/hailortcli")
        monkeypatch.setattr(
            hp.subprocess,
            "run",
            lambda *a, **kw: _make_completed(returncode=1, stderr="no device"),
        )
        assert host_hailo_arch() is None

    def test_returns_none_when_arch_line_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp.shutil, "which", lambda name: "/usr/bin/hailortcli")
        monkeypatch.setattr(
            hp.subprocess,
            "run",
            lambda *a, **kw: _make_completed(stdout="something else entirely\n"),
        )
        assert host_hailo_arch() is None

    def test_returns_none_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp.shutil, "which", lambda name: "/usr/bin/hailortcli")

        def _raise_timeout(*args: Any, **kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="hailortcli", timeout=3.0)

        monkeypatch.setattr(hp.subprocess, "run", _raise_timeout)
        assert host_hailo_arch() is None

    def test_returns_none_on_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp.shutil, "which", lambda name: "/usr/bin/hailortcli")

        def _raise_oserror(*args: Any, **kwargs: Any) -> None:
            raise OSError("permission denied")

        monkeypatch.setattr(hp.subprocess, "run", _raise_oserror)
        assert host_hailo_arch() is None


# ---------------------------------------------------------------------------
# _is_hailo_model
# ---------------------------------------------------------------------------


class TestIsHailoModel:
    def test_true_for_edge_runtime_hailo_top_level(self) -> None:
        assert _is_hailo_model({"edge_runtime": "hailo"}) is True

    def test_true_for_edge_runtime_hailo_under_metadata(self) -> None:
        assert _is_hailo_model({"metadata": {"edge_runtime": "hailo"}}) is True

    def test_true_for_hef_filename(self) -> None:
        assert _is_hailo_model({"filename": "yolov8s.hef"}) is True

    def test_true_for_hef_model_external_id(self) -> None:
        assert _is_hailo_model({"model_external_id": "yolov8s.hef"}) is True

    def test_true_for_hef_edge_model_path(self) -> None:
        assert _is_hailo_model({"metadata": {"edge_model_path": "yolov8s.hef"}}) is True

    def test_false_for_pt_entry(self) -> None:
        assert _is_hailo_model({"edge_runtime": "ultralytics", "filename": "yolov8s.pt"}) is False

    def test_false_for_empty_entry(self) -> None:
        assert _is_hailo_model({}) is False

    def test_case_insensitive_runtime(self) -> None:
        assert _is_hailo_model({"edge_runtime": "Hailo"}) is True


# ---------------------------------------------------------------------------
# preflight_hailo_arch
# ---------------------------------------------------------------------------


def _hailo_entry(*, hw_arch: str = "hailo8", slug: str = "yolov8s.hef") -> dict[str, Any]:
    return {
        "edge_runtime": "hailo",
        "filename": slug,
        "metadata": {"hw_arch": hw_arch, "edge_model_path": slug},
    }


class TestPreflightHailoArch:
    def test_no_op_for_non_hailo_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even with a present device + probe ready, a non-Hailo entry
        # must never reach the probe (would slow every download on
        # Hailo-equipped hosts).
        called = {"probe": False, "device_check": False}

        def _track_device() -> bool:
            called["device_check"] = True
            return True

        def _track_probe() -> str:
            called["probe"] = True
            return "hailo8"

        monkeypatch.setattr(hp, "host_hailo_device_present", _track_device)
        monkeypatch.setattr(hp, "host_hailo_arch", _track_probe)

        preflight_hailo_arch({"edge_runtime": "ultralytics", "filename": "yolov8s.pt"}, "yolov8s")

        assert called == {"probe": False, "device_check": False}

    def test_no_op_when_no_hailo_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp, "host_hailo_device_present", lambda: False)
        # Probe must not be called when there's no device — there's
        # nothing to compare against.
        monkeypatch.setattr(hp, "host_hailo_arch", lambda: pytest.fail("must not be called"))
        preflight_hailo_arch(_hailo_entry(), "yolov8s_h8")  # no exception

    def test_no_op_when_catalog_missing_hw_arch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp, "host_hailo_device_present", lambda: True)
        monkeypatch.setattr(hp, "host_hailo_arch", lambda: pytest.fail("must not be called"))
        entry = {"edge_runtime": "hailo", "filename": "custom.hef"}  # no hw_arch
        preflight_hailo_arch(entry, "custom")  # no exception

    def test_warns_when_probe_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(hp, "host_hailo_device_present", lambda: True)
        monkeypatch.setattr(hp, "host_hailo_arch", lambda: None)
        with caplog.at_level("WARNING"):
            preflight_hailo_arch(_hailo_entry(), "yolov8s_h8")  # no exception
        assert any("could not determine" in record.message for record in caplog.records)

    def test_passes_when_arch_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp, "host_hailo_device_present", lambda: True)
        monkeypatch.setattr(hp, "host_hailo_arch", lambda: "hailo8")
        preflight_hailo_arch(_hailo_entry(hw_arch="hailo8"), "yolov8s_h8")

    def test_raises_on_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp, "host_hailo_device_present", lambda: True)
        monkeypatch.setattr(hp, "host_hailo_arch", lambda: "hailo8l")
        with pytest.raises(HailoArchMismatchError) as excinfo:
            preflight_hailo_arch(_hailo_entry(hw_arch="hailo8"), "yolov8s_h8")
        msg = str(excinfo.value)
        assert "hailo8" in msg and "hailo8l" in msg
        # Sibling hint derived from the slug suffix.
        assert "yolov8s_h8l" in msg

    def test_raises_without_sibling_hint_for_irregular_slug(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hp, "host_hailo_device_present", lambda: True)
        monkeypatch.setattr(hp, "host_hailo_arch", lambda: "hailo8")
        with pytest.raises(HailoArchMismatchError) as excinfo:
            # A user-uploaded slug that doesn't follow the _h8/_h8l
            # convention — error message should still be useful.
            preflight_hailo_arch(
                _hailo_entry(hw_arch="hailo8l", slug="custom-detector.hef"),
                "custom-detector",
            )
        msg = str(excinfo.value)
        assert "hailo8" in msg and "hailo8l" in msg

    def test_honors_explicit_sibling_slug_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hp, "host_hailo_device_present", lambda: True)
        monkeypatch.setattr(hp, "host_hailo_arch", lambda: "hailo8l")
        with pytest.raises(HailoArchMismatchError) as excinfo:
            preflight_hailo_arch(
                _hailo_entry(hw_arch="hailo8"),
                "yolov8s_h8",
                sibling_slug_hint="something-explicit",
            )
        assert "something-explicit" in str(excinfo.value)


# ---------------------------------------------------------------------------
# _suggest_sibling_slug
# ---------------------------------------------------------------------------


class TestSuggestSiblingSlug:
    @pytest.mark.parametrize(
        "model_id,device_arch,expected",
        [
            ("yolov8s_h8", "hailo8l", "yolov8s_h8l"),
            ("yolov8s_h8l", "hailo8", "yolov8s_h8"),
            ("yolov6n_h8", "hailo8l", "yolov6n_h8l"),
            # No suffix → no useful suggestion.
            ("custom-model", "hailo8", ""),
            # Wrong-direction suggestion would be silly.
            ("yolov8s_h8", "hailo8", ""),
        ],
    )
    def test_suffix_flip(self, model_id: str, device_arch: str, expected: str) -> None:
        assert _suggest_sibling_slug(model_id, device_arch) == expected


# ---------------------------------------------------------------------------
# End-to-end wiring through ModelManager._download_model
# ---------------------------------------------------------------------------


class TestPreflightWiring:
    """Confirm the preflight runs inside _download_model.

    We don't exercise the full download path here — that's covered by
    test_model_manager.py — but we do prove that a mismatched HEF
    catalog entry short-circuits before any HTTP fetch is attempted.
    """

    def test_mismatch_short_circuits_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hp, "host_hailo_device_present", lambda: True)
        monkeypatch.setattr(hp, "host_hailo_arch", lambda: "hailo8l")

        manager = ModelManager(cache_dir=tmp_path, api_token="tok", base_url="https://api.test")

        catalog_entry = {
            "edge_runtime": "hailo",
            "filename": "yolov8s.hef",
            "metadata": {"hw_arch": "hailo8", "edge_model_path": "yolov8s.hef"},
        }

        with (
            patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry),
            patch.object(
                manager,
                "_download_with_retries",
                side_effect=AssertionError("must not download"),
            ),
            pytest.raises(HailoArchMismatchError),
        ):
            manager._download_model("yolov8s_h8")

    def test_match_proceeds_to_download_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Probe matches the catalog arch — Gate 3 must let the call
        # through and any downstream RuntimeError must come from the
        # download path, not from the preflight.
        monkeypatch.setattr(hp, "host_hailo_device_present", lambda: True)
        monkeypatch.setattr(hp, "host_hailo_arch", lambda: "hailo8")

        manager = ModelManager(cache_dir=tmp_path, api_token="tok", base_url="https://api.test")

        catalog_entry = {
            "edge_runtime": "hailo",
            "filename": "yolov8s.hef",
            "metadata": {"hw_arch": "hailo8", "edge_model_path": "yolov8s.hef"},
            # No download_url, no signed URL endpoint — the manager will
            # raise its own "no sources" error AFTER the preflight passes.
        }

        with (
            patch.object(manager, "_fetch_catalog_entry", return_value=catalog_entry),
            patch.object(manager, "_fetch_artifact_url_safe", return_value=None),
        ):
            with pytest.raises(RuntimeError) as excinfo:
                manager._download_model("yolov8s_h8")
            # The error must come from the download-source-resolution
            # branch, not from the preflight.
            assert "No download sources" in str(excinfo.value)
