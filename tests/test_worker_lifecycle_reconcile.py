"""Tests for ``reconcile_worker_lifecycle`` (CYB-1766).

The lifecycle reconcile runs once per worker-sync cycle in the runtime
loop. Its job is to bring ``WorkerManager`` in line with the current
state of ``~/.cyberwave/workers/``:

* Files present (active workflows synced) → start the container.
* No files (no active workflows) → stop the container.
* Sync reported errors → leave the worker alone (don't churn on
  transient API failures).

We only mock the parts that hit Docker / the network; the start/stop
decision logic is exercised with the real branching.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import cyberwave_edge_core.startup as startup


@pytest.fixture
def stub_worker_manager_module(monkeypatch):
    """Patch ``worker_manager`` so the lazy import inside the helper is cheap.

    Returns the ``WorkerManager`` class mock so tests can inspect calls.
    """
    fake_instance = MagicMock(name="WorkerManagerInstance")
    fake_class = MagicMock(name="WorkerManagerClass", return_value=fake_instance)

    import cyberwave_edge_core.worker_manager as worker_manager

    monkeypatch.setattr(worker_manager, "WorkerManager", fake_class)
    monkeypatch.setattr(
        worker_manager, "resolve_worker_image", lambda: "cyberwaveos/edge-ml-worker:test"
    )
    return fake_class, fake_instance


@pytest.fixture
def configured_edge(monkeypatch, tmp_path: Path):
    """Wire up the minimum runtime state for the helper to proceed."""
    monkeypatch.setattr(startup, "load_token", lambda: "tok")
    monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
    monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")
    monkeypatch.setattr(
        startup,
        "_resolve_worker_sync_twin_uuids",
        lambda *a, **kw: ["twin-a", "twin-b"],
    )
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(startup, "load_worker_resource_limits", lambda: None)
    return tmp_path


class TestReconcileWorkerLifecycle:
    def test_errors_in_sync_summary_skip_start_and_stop(
        self, configured_edge, stub_worker_manager_module
    ):
        """Transient API failures must not churn a healthy worker."""
        fake_class, fake_instance = stub_worker_manager_module

        startup.reconcile_worker_lifecycle(
            {"written": 0, "removed": 0, "unchanged": 0, "errors": 1}
        )

        fake_class.assert_not_called()
        fake_instance.start.assert_not_called()
        fake_instance.stop.assert_not_called()

    def test_no_files_calls_stop(
        self, configured_edge, stub_worker_manager_module
    ):
        """Empty workers/ → tear the container down."""
        fake_class, fake_instance = stub_worker_manager_module
        # workers/ does not exist → has_files == False
        assert not (configured_edge / "workers").exists()

        startup.reconcile_worker_lifecycle(
            {"written": 0, "removed": 1, "unchanged": 0, "errors": 0}
        )

        fake_class.assert_called_once()
        fake_instance.stop.assert_called_once()
        fake_instance.start.assert_not_called()

    def test_files_present_calls_start(
        self, configured_edge, stub_worker_manager_module
    ):
        """A wf_*.py file present → bring the container up."""
        fake_class, fake_instance = stub_worker_manager_module
        workers_dir = configured_edge / "workers"
        workers_dir.mkdir()
        (workers_dir / "wf_demo.py").write_text("# generated worker\n")

        startup.reconcile_worker_lifecycle(
            {"written": 1, "removed": 0, "unchanged": 0, "errors": 0}
        )

        fake_class.assert_called_once()
        fake_instance.start.assert_called_once()
        fake_instance.stop.assert_not_called()

    def test_missing_token_is_noop(
        self, configured_edge, stub_worker_manager_module, monkeypatch
    ):
        """Without auth we can't construct a manager; bail out cleanly."""
        fake_class, fake_instance = stub_worker_manager_module
        monkeypatch.setattr(startup, "load_token", lambda: None)

        startup.reconcile_worker_lifecycle(
            {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}
        )

        fake_class.assert_not_called()
        fake_instance.start.assert_not_called()
        fake_instance.stop.assert_not_called()

    def test_missing_environment_is_noop(
        self, configured_edge, stub_worker_manager_module, monkeypatch
    ):
        fake_class, fake_instance = stub_worker_manager_module
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: None)

        startup.reconcile_worker_lifecycle(
            {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}
        )

        fake_class.assert_not_called()
        fake_instance.start.assert_not_called()
        fake_instance.stop.assert_not_called()

    def test_missing_fingerprint_is_noop(
        self, configured_edge, stub_worker_manager_module, monkeypatch
    ):
        """Match reconcile_worker_sync's caution: bail when fingerprint is unavailable."""
        fake_class, fake_instance = stub_worker_manager_module
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: None)

        startup.reconcile_worker_lifecycle(
            {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}
        )

        fake_class.assert_not_called()
        fake_instance.start.assert_not_called()
        fake_instance.stop.assert_not_called()

    def test_twin_resolution_failure_is_noop(
        self, configured_edge, stub_worker_manager_module, monkeypatch
    ):
        """If we can't resolve twins, leave the worker untouched."""
        fake_class, fake_instance = stub_worker_manager_module
        monkeypatch.setattr(
            startup,
            "_resolve_worker_sync_twin_uuids",
            MagicMock(side_effect=RuntimeError("API down")),
        )

        startup.reconcile_worker_lifecycle(
            {"written": 0, "removed": 0, "unchanged": 0, "errors": 0}
        )

        fake_class.assert_not_called()
        fake_instance.start.assert_not_called()
        fake_instance.stop.assert_not_called()
