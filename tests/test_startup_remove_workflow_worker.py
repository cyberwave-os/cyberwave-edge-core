"""Tests for the surgical ``remove_workflow_worker`` MQTT command and
the lifecycle-reconcile chaining in ``_run_immediate_worker_sync``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import cyberwave_edge_core.startup as startup


@pytest.fixture(autouse=True)
def _clear_module_state():
    startup._HANDLED_TWIN_COMMAND_REQUEST_IDS.clear()
    startup._WORKER_SYNC_PREVIOUSLY_MISSING.clear()
    yield
    startup._HANDLED_TWIN_COMMAND_REQUEST_IDS.clear()
    startup._WORKER_SYNC_PREVIOUSLY_MISSING.clear()


@pytest.fixture
def workers_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    workers = tmp_path / "workers"
    workers.mkdir(parents=True, exist_ok=True)
    return workers


def _payload(
    *,
    workflow_uuid: str = "11112222333344445555666677778888",
    worker_filenames: list[str] | None = None,
    request_id: str = "req-aaaa",
    command: str = "remove_workflow_worker",
) -> dict:
    if worker_filenames is None:
        worker_filenames = [
            f"wf_{workflow_uuid[:12]}.py",
            f"wf_{workflow_uuid[:8]}.py",
        ]
    return {
        "command": command,
        "request_id": request_id,
        "requested_at": "2026-05-21T11:00:00+00:00",
        "requested_by_uuid": "user-xyz",
        "workflow_uuid": workflow_uuid,
        "worker_filenames": worker_filenames,
    }


class TestRunRemoveWorkflowWorker:
    def test_unlinks_named_file_and_calls_lifecycle(self, workers_dir, monkeypatch):
        target = workers_dir / "wf_aaaaaaaaaaaa.py"
        target.write_text("# stale worker\n", encoding="utf-8")
        reconcile = MagicMock()
        monkeypatch.setattr(startup, "reconcile_worker_lifecycle", reconcile)

        startup._run_remove_workflow_worker(
            _payload(worker_filenames=["wf_aaaaaaaaaaaa.py"])
        )

        assert not target.exists()
        reconcile.assert_called_once()
        summary = reconcile.call_args.args[0]
        assert summary["removed"] == 1
        assert summary["errors"] == 0

    def test_idempotent_when_file_already_absent(self, workers_dir, monkeypatch):
        """Lifecycle reconcile must still run so the container can be
        stopped if other workflows already cleaned up."""
        reconcile = MagicMock()
        monkeypatch.setattr(startup, "reconcile_worker_lifecycle", reconcile)

        startup._run_remove_workflow_worker(
            _payload(worker_filenames=["wf_bbbbbbbbbbbb.py"])
        )

        reconcile.assert_called_once()
        summary = reconcile.call_args.args[0]
        assert summary["removed"] == 0

    def test_unlinks_both_candidate_filenames(self, workers_dir, monkeypatch):
        """Backend ships both 12-hex and 8-hex candidates because two
        compilers exist; either may have landed on disk."""
        twelve = workers_dir / "wf_111122223333.py"
        eight = workers_dir / "wf_11112222.py"
        twelve.write_text("# 12\n", encoding="utf-8")
        eight.write_text("# 8\n", encoding="utf-8")
        monkeypatch.setattr(startup, "reconcile_worker_lifecycle", MagicMock())

        startup._run_remove_workflow_worker(
            _payload(worker_filenames=["wf_111122223333.py", "wf_11112222.py"])
        )

        assert not twelve.exists()
        assert not eight.exists()

    def test_rejects_non_wf_prefix_filenames(self, workers_dir, monkeypatch):
        sibling = workers_dir / "important_config.py"
        sibling.write_text("# do not touch\n", encoding="utf-8")
        monkeypatch.setattr(startup, "reconcile_worker_lifecycle", MagicMock())

        startup._run_remove_workflow_worker(
            _payload(worker_filenames=["important_config.py"])
        )

        assert sibling.exists()

    @pytest.mark.parametrize(
        "bad_filename",
        [
            "../etc/passwd",
            "wf_AAAA.py",
            "wf_aaaaaaaaaaaa.txt",
            "",
        ],
    )
    def test_rejects_path_traversal_and_malformed_names(
        self, workers_dir, monkeypatch, bad_filename
    ):
        sentinel = workers_dir / "wf_cccccccccccc.py"
        sentinel.write_text("# survivor\n", encoding="utf-8")
        monkeypatch.setattr(startup, "reconcile_worker_lifecycle", MagicMock())

        startup._run_remove_workflow_worker(
            _payload(worker_filenames=[bad_filename])
        )

        assert sentinel.exists()

    def test_non_list_worker_filenames_is_rejected(self, workers_dir, monkeypatch):
        sibling = workers_dir / "wf_dddddddddddd.py"
        sibling.write_text("# do not touch\n", encoding="utf-8")
        reconcile = MagicMock()
        monkeypatch.setattr(startup, "reconcile_worker_lifecycle", reconcile)

        startup._run_remove_workflow_worker(
            {
                "command": "remove_workflow_worker",
                "request_id": "req-x",
                "worker_filenames": "wf_dddddddddddd.py",
            }
        )

        assert sibling.exists()
        reconcile.assert_not_called()

    def test_discards_filename_from_two_strikes_set(self, workers_dir, monkeypatch):
        """Surgical removal must clear the bulk-sync two-strikes
        record, otherwise the next periodic tick warns about a file
        we just intentionally removed."""
        startup._WORKER_SYNC_PREVIOUSLY_MISSING.add("wf_abababababab.py")
        target = workers_dir / "wf_abababababab.py"
        target.write_text("# x\n", encoding="utf-8")
        monkeypatch.setattr(startup, "reconcile_worker_lifecycle", MagicMock())

        startup._run_remove_workflow_worker(
            _payload(worker_filenames=["wf_abababababab.py"])
        )

        assert "wf_abababababab.py" not in startup._WORKER_SYNC_PREVIOUSLY_MISSING

    def test_lifecycle_reconcile_failure_is_swallowed(self, workers_dir, monkeypatch):
        target = workers_dir / "wf_cdcdcdcdcdcd.py"
        target.write_text("# x\n", encoding="utf-8")
        monkeypatch.setattr(
            startup,
            "reconcile_worker_lifecycle",
            MagicMock(side_effect=RuntimeError("docker daemon unreachable")),
        )

        startup._run_remove_workflow_worker(
            _payload(worker_filenames=["wf_cdcdcdcdcdcd.py"])
        )
        assert not target.exists()


class TestHandleTwinCommandDispatch:
    def test_dispatches_remove_command_to_thread_target(self, monkeypatch):
        captured: dict = {}

        class FakeThread:
            def __init__(self, *, target, args=(), name="", daemon=False):
                captured["target"] = target
                captured["args"] = args

            def start(self):
                captured["started"] = True

        monkeypatch.setattr(startup.threading, "Thread", FakeThread)

        payload = _payload()
        startup._handle_twin_command_message(payload)

        assert captured["target"] is startup._run_remove_workflow_worker
        assert captured["args"] == (payload,)
        assert captured["started"] is True

    def test_dedupes_on_request_id(self, monkeypatch):
        """Duplicate MQTT delivery with the same ``request_id`` must
        not spawn a second thread."""
        thread_calls: list[dict] = []

        class FakeThread:
            def __init__(self, *, target, args=(), name="", daemon=False):
                thread_calls.append({"target": target})

            def start(self):
                pass

        monkeypatch.setattr(startup.threading, "Thread", FakeThread)

        payload = _payload(request_id="dup-1")
        startup._handle_twin_command_message(payload)
        startup._handle_twin_command_message(payload)

        assert len(thread_calls) == 1

    def test_sync_workflows_command_still_routes_to_immediate_sync(self, monkeypatch):
        captured: dict = {}

        class FakeThread:
            def __init__(self, *, target, args=(), name="", daemon=False):
                captured["target"] = target

            def start(self):
                captured["started"] = True

        monkeypatch.setattr(startup.threading, "Thread", FakeThread)

        startup._handle_twin_command_message(
            {
                "command": "sync_workflows",
                "request_id": "x",
                "requested_at": "...",
            }
        )

        assert captured["target"] is startup._run_immediate_worker_sync
        assert captured["started"] is True


class TestImmediateWorkerSyncChainsLifecycle:
    """``_run_immediate_worker_sync`` chains
    :func:`reconcile_worker_lifecycle` after a clean sync so the
    worker container starts/stops *now* rather than on the periodic
    tick."""

    def test_chains_lifecycle_after_clean_sync(self, monkeypatch):
        clean_summary = {"written": 1, "removed": 0, "unchanged": 0, "errors": 0}
        monkeypatch.setattr(startup, "reconcile_worker_sync", lambda: clean_summary)
        reconcile_lifecycle = MagicMock()
        monkeypatch.setattr(startup, "reconcile_worker_lifecycle", reconcile_lifecycle)

        startup._run_immediate_worker_sync()

        reconcile_lifecycle.assert_called_once_with(clean_summary)

    def test_skips_lifecycle_when_sync_reported_errors(self, monkeypatch):
        error_summary = {"written": 0, "removed": 0, "unchanged": 0, "errors": 1}
        monkeypatch.setattr(startup, "reconcile_worker_sync", lambda: error_summary)
        reconcile_lifecycle = MagicMock()
        monkeypatch.setattr(startup, "reconcile_worker_lifecycle", reconcile_lifecycle)

        startup._run_immediate_worker_sync()

        reconcile_lifecycle.assert_not_called()

    def test_skips_lifecycle_when_sync_raises(self, monkeypatch):
        monkeypatch.setattr(
            startup,
            "reconcile_worker_sync",
            MagicMock(side_effect=RuntimeError("backend unreachable")),
        )
        reconcile_lifecycle = MagicMock()
        monkeypatch.setattr(startup, "reconcile_worker_lifecycle", reconcile_lifecycle)

        startup._run_immediate_worker_sync()
        reconcile_lifecycle.assert_not_called()

    def test_lifecycle_failure_does_not_propagate(self, monkeypatch):
        clean_summary = {"written": 0, "removed": 1, "unchanged": 0, "errors": 0}
        monkeypatch.setattr(startup, "reconcile_worker_sync", lambda: clean_summary)
        monkeypatch.setattr(
            startup,
            "reconcile_worker_lifecycle",
            MagicMock(side_effect=RuntimeError("docker down")),
        )

        startup._run_immediate_worker_sync()
