"""Tests for the worker sync orchestration in startup.py.

Covers:
  1. reconcile_worker_sync()  — early returns, exception handling, aggregation
  2. _sync_workers_for_twins() — EdgeSyncClient wiring, per-twin error isolation
  3. run_startup_checks() step 7 — sync runs when linked twins exist
  4. load_selected_twin_uuids() / _resolve_worker_sync_twin_uuids()
     — environment.json is authoritative; fingerprint discovery is only
     a fallback for installs that predate the ``twin_uuids`` field.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import cyberwave_edge_core.startup as startup
from cyberwave_edge_core.edge_sync_client import EdgeSyncResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZERO_SUMMARY = {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}


def _result(
    twin_uuid: str = "twin-1",
    *,
    written: list[str] | None = None,
    removed: list[str] | None = None,
    unchanged: list[str] | None = None,
    errors: list[str] | None = None,
) -> EdgeSyncResult:
    return EdgeSyncResult(
        twin_uuid=twin_uuid,
        written=written or [],
        removed=removed or [],
        unchanged=unchanged or [],
        errors=errors or [],
    )


# ===========================================================================
# 1. reconcile_worker_sync()
# ===========================================================================


class TestReconcileWorkerSync:
    """Unit tests for reconcile_worker_sync() orchestration logic."""

    def test_returns_zeros_when_no_token(self, monkeypatch):
        monkeypatch.setattr(startup, "load_token", lambda: None)
        assert startup.reconcile_worker_sync() == _ZERO_SUMMARY

    def test_returns_zeros_when_no_environment_uuid(self, monkeypatch):
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: None)
        assert startup.reconcile_worker_sync() == _ZERO_SUMMARY

    def test_returns_zeros_when_no_fingerprint(self, monkeypatch):
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: None)
        assert startup.reconcile_worker_sync() == _ZERO_SUMMARY

    def test_returns_error_when_list_twins_raises(self, monkeypatch):
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")
        monkeypatch.setattr(
            startup,
            "_resolve_worker_sync_twin_uuids",
            MagicMock(side_effect=RuntimeError("API down")),
        )
        result = startup.reconcile_worker_sync()
        assert result == {"written": 0, "removed": 0, "unchanged": 0, "errors": 1}

    def test_returns_zeros_when_no_twins_linked(self, monkeypatch):
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")
        monkeypatch.setattr(
            startup,
            "_resolve_worker_sync_twin_uuids",
            lambda *a: [],
        )
        assert startup.reconcile_worker_sync() == _ZERO_SUMMARY

    def test_aggregates_per_twin_results(self, monkeypatch, tmp_path):
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")
        monkeypatch.setattr(
            startup,
            "_resolve_worker_sync_twin_uuids",
            lambda *a: ["twin-a", "twin-b"],
        )
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(
            startup,
            "_sync_workers_for_twins",
            lambda **kw: {
                "twin-a": {"written": 2, "removed": 1, "unchanged": 3, "errors": 0},
                "twin-b": {"written": 0, "removed": 0, "unchanged": 1, "errors": 1},
            },
        )

        result = startup.reconcile_worker_sync()
        assert result == {"written": 2, "removed": 1, "unchanged": 4, "errors": 1}


# ===========================================================================
# 2. _sync_workers_for_twins()
# ===========================================================================


class TestSyncWorkersForTwins:
    """Unit tests for _sync_workers_for_twins() EdgeSyncClient integration."""

    def test_syncs_each_twin_and_returns_summary(self, monkeypatch, tmp_path):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

        fake_client = MagicMock()
        fake_client.sync_all.return_value = [
            _result("twin-a", written=["wf_aaa.py"]),
            _result("twin-b", unchanged=["wf_bbb.py"]),
        ]

        with patch(
            "cyberwave_edge_core.edge_sync_client.EdgeSyncClient",
            return_value=fake_client,
        ):
            summary = startup._sync_workers_for_twins(
                token="tok",
                twin_uuids=["twin-a", "twin-b"],
                base_url="http://localhost:8000",
            )

        assert summary["twin-a"] == {"written": 1, "removed": 0, "unchanged": 0, "errors": 0}
        assert summary["twin-b"] == {"written": 0, "removed": 0, "unchanged": 1, "errors": 0}
        fake_client.sync_all.assert_called_once_with(["twin-a", "twin-b"])

    def test_isolates_per_twin_exception(self, monkeypatch, tmp_path):
        """If sync_all() raises, all twins get error entries."""
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

        fake_client = MagicMock()
        fake_client.sync_all.side_effect = RuntimeError("network timeout")

        with patch(
            "cyberwave_edge_core.edge_sync_client.EdgeSyncClient",
            return_value=fake_client,
        ):
            summary = startup._sync_workers_for_twins(
                token="tok",
                twin_uuids=["twin-a", "twin-b"],
                base_url="http://localhost:8000",
            )

        assert summary["twin-a"]["errors"] == 1
        assert summary["twin-b"]["errors"] == 1

    def test_uses_config_dir_workers_subdirectory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

        constructed_args: list[dict] = []

        class FakeEdgeSyncClient:
            def __init__(self, *, workers_dir, base_url, token, previously_missing=None):
                constructed_args.append(
                    {
                        "workers_dir": workers_dir,
                        "base_url": base_url,
                        "token": token,
                        "previously_missing": previously_missing,
                    }
                )

            def sync_all(self, twin_uuids):
                return [_result(t) for t in twin_uuids]

        with patch(
            "cyberwave_edge_core.edge_sync_client.EdgeSyncClient",
            FakeEdgeSyncClient,
        ):
            startup._sync_workers_for_twins(
                token="tok",
                twin_uuids=["twin-1"],
                base_url="https://api.cyberwave.com",
            )

        assert len(constructed_args) == 1
        assert constructed_args[0]["workers_dir"] == tmp_path / "workers"
        assert constructed_args[0]["base_url"] == "https://api.cyberwave.com"
        assert constructed_args[0]["token"] == "tok"
        # Persistent module-level set is passed in so two-strikes
        # cleanup state survives across periodic-sync cycles.
        assert constructed_args[0]["previously_missing"] is startup._WORKER_SYNC_PREVIOUSLY_MISSING


# ===========================================================================
# 3. run_startup_checks() — step 7 worker sync
# ===========================================================================


class TestStartupWorkerSyncStep:
    """Verify that run_startup_checks() invokes worker sync for linked twins."""

    def _patch_prerequisites(
        self,
        monkeypatch,
        *,
        linked_twins: list[str] | None = None,
        selected_twins: list[str] | None = None,
    ):
        """Stub everything before step 7 so startup reaches the sync step.

        ``linked_twins`` drives the fingerprint-based discovery used for the
        bootstrap health publisher (drivers path).  ``selected_twins`` drives
        the operator-picked list that workflow sync should actually use;
        defaults to ``linked_twins`` when not specified so existing tests
        keep working unchanged.
        """
        monkeypatch.setattr(startup, "load_token", lambda: "token-123")
        monkeypatch.setattr(startup, "validate_token", lambda token: True)
        monkeypatch.setattr(startup, "check_mqtt_connection", lambda token: True)
        monkeypatch.setattr(startup, "register_edge", lambda token: True)
        monkeypatch.setattr(
            startup,
            "load_environment_uuid",
            lambda retries=0, retry_delay_seconds=0.2: "env-uuid",
        )
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")
        monkeypatch.setattr(
            startup,
            "_list_linked_twin_uuids_for_fingerprint",
            lambda *a: linked_twins or [],
        )
        effective_selected = selected_twins if selected_twins is not None else (linked_twins or [])
        monkeypatch.setattr(
            startup,
            "_resolve_worker_sync_twin_uuids",
            lambda *a: list(effective_selected),
        )
        monkeypatch.setattr(
            startup,
            "_start_bootstrap_edge_health_publisher",
            lambda *a, **kw: True,
        )
        monkeypatch.setattr(
            startup,
            "fetch_and_run_twin_drivers",
            lambda *a, **kw: [],
        )
        monkeypatch.setattr(
            startup,
            "_get_zenoh_config",
            lambda: MagicMock(is_zenoh=False, router_enabled=False),
        )
        monkeypatch.setattr(
            startup,
            "log_zenoh_diagnostics",
            lambda cfg: MagicMock(mode="disabled", warnings=[]),
        )

    def test_worker_sync_called_with_linked_twins(self, monkeypatch):
        self._patch_prerequisites(
            monkeypatch,
            linked_twins=["twin-aaa", "twin-bbb"],
        )
        sync_calls: list[dict] = []
        monkeypatch.setattr(
            startup,
            "_sync_workers_for_twins",
            lambda **kw: (
                sync_calls.append(kw) or {"twin-aaa": _ZERO_SUMMARY, "twin-bbb": _ZERO_SUMMARY}
            ),
        )

        result = startup.run_startup_checks()

        assert result is True
        assert len(sync_calls) == 1
        assert sync_calls[0]["twin_uuids"] == ["twin-aaa", "twin-bbb"]
        assert sync_calls[0]["token"] == "token-123"

    def test_worker_sync_skipped_when_no_linked_twins(self, monkeypatch):
        self._patch_prerequisites(monkeypatch, linked_twins=[])
        sync_calls: list[dict] = []
        monkeypatch.setattr(
            startup,
            "_sync_workers_for_twins",
            lambda **kw: sync_calls.append(kw) or {},
        )

        result = startup.run_startup_checks()

        assert result is True
        assert sync_calls == []

    def test_worker_sync_uses_selected_twin_list_not_fingerprint(self, monkeypatch):
        """Regression for CYB-XXX: ``environment.json`` is authoritative.

        When the operator selected only ``twin-aaa`` at install time but the
        environment still has a stale ``twin-bbb`` carrying this edge's
        fingerprint, worker sync must scope to the selected list only.
        Otherwise every edge pulls workflows for every twin whose metadata
        happens to match the fingerprint.
        """
        self._patch_prerequisites(
            monkeypatch,
            linked_twins=["twin-aaa", "twin-bbb"],
            selected_twins=["twin-aaa"],
        )
        sync_calls: list[dict] = []
        monkeypatch.setattr(
            startup,
            "_sync_workers_for_twins",
            lambda **kw: sync_calls.append(kw) or {"twin-aaa": _ZERO_SUMMARY},
        )

        result = startup.run_startup_checks()

        assert result is True
        assert len(sync_calls) == 1
        assert sync_calls[0]["twin_uuids"] == ["twin-aaa"]


# ===========================================================================
# 4. load_selected_twin_uuids() / _resolve_worker_sync_twin_uuids()
# ===========================================================================


class TestLoadSelectedTwinUuids:
    """environment.json drives the operator-selected twin list for worker sync."""

    @pytest.fixture
    def env_file(self, tmp_path, monkeypatch):
        path = tmp_path / "environment.json"
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", path)
        return path

    def test_returns_none_when_field_missing_or_unreadable(self, env_file):
        # File missing — legacy install predating the field.
        assert startup.load_selected_twin_uuids() is None
        # File present but no twin_uuids key — same "fall back" signal.
        env_file.write_text(json.dumps({"uuid": "env-1"}), encoding="utf-8")
        assert startup.load_selected_twin_uuids() is None
        # Corrupt JSON — don't silently "fall back to syncing everything".
        env_file.write_text("{not valid json", encoding="utf-8")
        assert startup.load_selected_twin_uuids() is None

    def test_empty_list_is_an_explicit_selection(self, env_file):
        """An empty list means 'nothing selected', not 'fall back to fingerprint'."""
        env_file.write_text(json.dumps({"uuid": "env-1", "twin_uuids": []}), encoding="utf-8")
        assert startup.load_selected_twin_uuids() == []

    def test_returns_cleaned_selection(self, env_file):
        """Whitespace/empties/dupes are sanitised; order preserved."""
        env_file.write_text(
            json.dumps({"twin_uuids": ["twin-a", " twin-b ", "twin-a", "", None]}),
            encoding="utf-8",
        )
        assert startup.load_selected_twin_uuids() == ["twin-a", "twin-b"]


class TestResolveWorkerSyncTwinUuids:
    """Selection wins; fingerprint discovery is only a legacy-install fallback."""

    def test_selected_list_wins_over_fingerprint_discovery(self, monkeypatch):
        monkeypatch.setattr(startup, "load_selected_twin_uuids", lambda: ["twin-picked"])
        monkeypatch.setattr(
            startup,
            "_list_linked_twin_uuids_for_fingerprint",
            lambda *a: ["twin-from-metadata"],
        )
        assert startup._resolve_worker_sync_twin_uuids("tok", "env-uuid", "fp") == ["twin-picked"]

    def test_explicit_empty_selection_blocks_fallback(self, monkeypatch):
        """Empty selection means "user picked nothing" — don't resurrect the
        metadata-discovered list, which is exactly the bug we're fixing."""
        monkeypatch.setattr(startup, "load_selected_twin_uuids", lambda: [])
        monkeypatch.setattr(
            startup,
            "_list_linked_twin_uuids_for_fingerprint",
            lambda *a: ["twin-from-metadata"],
        )
        assert startup._resolve_worker_sync_twin_uuids("tok", "env-uuid", "fp") == []

    def test_falls_back_to_fingerprint_when_field_absent(self, monkeypatch):
        """Backward compat for edges installed before ``twin_uuids`` existed."""
        monkeypatch.setattr(startup, "load_selected_twin_uuids", lambda: None)
        monkeypatch.setattr(
            startup,
            "_list_linked_twin_uuids_for_fingerprint",
            lambda token, env, fp: ["twin-legacy"],
        )
        assert startup._resolve_worker_sync_twin_uuids("tok", "env-uuid", "fp") == ["twin-legacy"]
