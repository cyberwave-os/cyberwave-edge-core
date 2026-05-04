"""Unit tests for EdgeSyncClient."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cyberwave_edge_core.edge_sync_client import EdgeSyncClient, EdgeSyncResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_payload(
    twin_uuid: str = "twin-1",
    *,
    workflows: list[dict] | None = None,
) -> dict:
    if workflows is None:
        workflows = []
    return {"twin_uuid": twin_uuid, "workflows": workflows}


def _wf_entry(filename: str, source: str) -> dict:
    return {
        "workflow_uuid": "wf-uuid",
        "workflow_name": "Test WF",
        "worker_filename": filename,
        "worker_source": source,
        "model_requirements": [],
    }


def _make_client(tmp_path: Path) -> EdgeSyncClient:
    return EdgeSyncClient(
        workers_dir=tmp_path / "workers",
        base_url="http://localhost:8000",
        token="test-token",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEdgeSyncClientSync:
    def test_writes_new_worker_file(self, tmp_path):
        client = _make_client(tmp_path)
        source = "# generated\n@cw.on_frame('twin-1')\ndef h(f, c): pass\n"
        payload = _make_payload(workflows=[_wf_entry("wf_abc.py", source)])

        with patch.object(client, "_fetch_sync_payload", return_value=payload):
            result = client.sync("twin-1")

        assert result.written == ["wf_abc.py"]
        assert result.errors == []
        written_file = tmp_path / "workers" / "wf_abc.py"
        assert written_file.exists()
        assert written_file.read_text() == source

    def test_no_write_when_content_identical(self, tmp_path):
        client = _make_client(tmp_path)
        source = "# same content\n"
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir(parents=True)
        (workers_dir / "wf_abc.py").write_text(source)

        original_mtime = (workers_dir / "wf_abc.py").stat().st_mtime

        payload = _make_payload(workflows=[_wf_entry("wf_abc.py", source)])

        with patch.object(client, "_fetch_sync_payload", return_value=payload):
            result = client.sync("twin-1")

        assert result.unchanged == ["wf_abc.py"]
        assert result.written == []
        # mtime must not change (no spurious restart trigger)
        assert (workers_dir / "wf_abc.py").stat().st_mtime == original_mtime

    def test_overwrites_changed_content(self, tmp_path):
        client = _make_client(tmp_path)
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir(parents=True)
        (workers_dir / "wf_abc.py").write_text("# old\n")

        new_source = "# new content\n"
        payload = _make_payload(workflows=[_wf_entry("wf_abc.py", new_source)])

        with patch.object(client, "_fetch_sync_payload", return_value=payload):
            result = client.sync("twin-1")

        assert result.written == ["wf_abc.py"]
        assert result.unchanged == []
        assert (workers_dir / "wf_abc.py").read_text() == new_source

    def test_removes_stale_wf_file_after_two_consecutive_misses(self, tmp_path):
        """Two-strikes rule: a stale file is removed only on the
        SECOND consecutive sync where it's missing from the payload.
        First sync = warning + keep; second sync = remove."""
        client = _make_client(tmp_path)
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir(parents=True)
        stale = workers_dir / "wf_old.py"
        stale.write_text("# stale\n")

        # Payload has no entry for wf_old.py — cleanup only via sync_all
        payload = _make_payload(workflows=[])

        with patch.object(client, "_fetch_sync_payload", return_value=payload):
            first_results = client.sync_all(["twin-1"])
            # First sync: file should be kept (first strike).
            assert all("wf_old.py" not in r.removed for r in first_results)
            assert stale.exists(), "first strike must keep file on disk"

            second_results = client.sync_all(["twin-1"])

        assert any("wf_old.py" in r.removed for r in second_results)
        assert not stale.exists()

    def test_does_not_remove_custom_workers(self, tmp_path):
        """Files without wf_ prefix must not be touched."""
        client = _make_client(tmp_path)
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir(parents=True)
        custom = workers_dir / "my_worker.py"
        custom.write_text("# custom\n")

        payload = _make_payload(workflows=[])

        with patch.object(client, "_fetch_sync_payload", return_value=payload):
            result = client.sync("twin-1")

        assert result.removed == []
        assert custom.exists()

    def test_creates_workers_dir_if_missing(self, tmp_path):
        client = _make_client(tmp_path)
        workers_dir = tmp_path / "workers"
        assert not workers_dir.exists()

        source = "# new\n"
        payload = _make_payload(workflows=[_wf_entry("wf_new.py", source)])

        with patch.object(client, "_fetch_sync_payload", return_value=payload):
            result = client.sync("twin-1")

        assert workers_dir.exists()
        assert result.written == ["wf_new.py"]

    def test_ignores_entries_without_source(self, tmp_path):
        """Entries missing worker_source are silently skipped."""
        client = _make_client(tmp_path)
        payload = _make_payload(
            workflows=[
                {
                    "workflow_uuid": "wf-uuid",
                    "worker_filename": "wf_abc.py",
                    "worker_source": None,
                }
            ]
        )

        with patch.object(client, "_fetch_sync_payload", return_value=payload):
            result = client.sync("twin-1")

        assert result.written == []
        assert result.errors == []

    def test_ignores_entries_without_wf_prefix(self, tmp_path):
        """Entries with filenames not starting with wf_ are skipped."""
        client = _make_client(tmp_path)
        payload = _make_payload(
            workflows=[
                {
                    "workflow_uuid": "wf-uuid",
                    "worker_filename": "custom.py",
                    "worker_source": "# source\n",
                }
            ]
        )

        with patch.object(client, "_fetch_sync_payload", return_value=payload):
            result = client.sync("twin-1")

        workers_dir = tmp_path / "workers"
        assert not (workers_dir / "custom.py").exists()

    def test_network_failure_returns_error(self, tmp_path):
        client = _make_client(tmp_path)

        with patch.object(
            client, "_fetch_sync_payload", side_effect=RuntimeError("network error")
        ):
            result = client.sync("twin-1")

        assert not result.ok
        assert len(result.errors) == 1
        assert "network error" in result.errors[0]

    def test_write_failure_reported_in_errors(self, tmp_path):
        """An OS error during file write is captured in result.errors, not raised."""
        client = _make_client(tmp_path)
        source = "# source\n"
        payload = _make_payload(workflows=[_wf_entry("wf_fail.py", source)])

        with (
            patch.object(client, "_fetch_sync_payload", return_value=payload),
            patch.object(
                EdgeSyncClient,
                "_atomic_write",
                side_effect=OSError("disk full"),
            ),
        ):
            result = client.sync("twin-1")

        assert len(result.errors) == 1
        assert "disk full" in result.errors[0]
        assert result.written == []

    def test_multiple_twins_writes_in_expected_dirs(self, tmp_path):
        """Sync for different twin UUIDs writes to the same shared workers dir."""
        client = _make_client(tmp_path)
        source_a = "# twin-a\n"
        source_b = "# twin-b\n"

        def fake_fetch(twin_uuid):
            if twin_uuid == "twin-a":
                return _make_payload(
                    twin_uuid="twin-a",
                    workflows=[_wf_entry("wf_aaa.py", source_a)],
                )
            return _make_payload(
                twin_uuid="twin-b",
                workflows=[_wf_entry("wf_bbb.py", source_b)],
            )

        with patch.object(client, "_fetch_sync_payload", side_effect=fake_fetch):
            result_a = client.sync("twin-a")
            result_b = client.sync("twin-b")

        assert result_a.written == ["wf_aaa.py"]
        assert result_b.written == ["wf_bbb.py"]

    def test_result_str(self):
        result = EdgeSyncResult(
            twin_uuid="t1",
            written=["wf_a.py"],
            removed=["wf_b.py"],
            unchanged=["wf_c.py"],
            errors=[],
        )
        s = str(result)
        assert "t1" in s
        assert "written=1" in s
        assert "removed=1" in s

    def test_result_ok_true_when_no_errors(self):
        result = EdgeSyncResult(twin_uuid="t1")
        assert result.ok

    def test_result_ok_false_when_errors(self):
        result = EdgeSyncResult(twin_uuid="t1", errors=["some error"])
        assert not result.ok


class TestEdgeSyncClientTwoStrikesCleanup:
    """Two-strikes rule for stale-file cleanup.

    Regression coverage for the case where the cloud's edge-sync
    response is briefly empty (e.g. while the operator is saving an
    intermediate workflow state in the editor) and a single bad sync
    used to wipe every local ``wf_*.py`` file.
    """

    def test_transient_empty_response_does_not_delete(self, tmp_path):
        """Sync 1 has the workflow, sync 2 is transiently empty,
        sync 3 has it again. The file must survive the dip."""
        previously_missing: set[str] = set()
        client = EdgeSyncClient(
            workers_dir=tmp_path / "workers",
            base_url="http://localhost:8000",
            token="test-token",
            previously_missing=previously_missing,
        )
        source = "# generated\n"
        full_payload = _make_payload(workflows=[_wf_entry("wf_x.py", source)])
        empty_payload = _make_payload(workflows=[])
        worker_path = tmp_path / "workers" / "wf_x.py"

        with patch.object(client, "_fetch_sync_payload", return_value=full_payload):
            client.sync_all(["twin-1"])
        assert worker_path.exists()

        with patch.object(client, "_fetch_sync_payload", return_value=empty_payload):
            results = client.sync_all(["twin-1"])
        assert worker_path.exists(), "first strike must not delete the file"
        assert all("wf_x.py" not in r.removed for r in results)
        assert "wf_x.py" in previously_missing, "file should be flagged for next-sync removal"

        with patch.object(client, "_fetch_sync_payload", return_value=full_payload):
            client.sync_all(["twin-1"])
        assert worker_path.exists()
        assert "wf_x.py" not in previously_missing, "strike count should reset on re-claim"

    def test_two_consecutive_empty_responses_delete(self, tmp_path):
        """Two empty syncs in a row: the file is removed on the second."""
        previously_missing: set[str] = set()
        client = EdgeSyncClient(
            workers_dir=tmp_path / "workers",
            base_url="http://localhost:8000",
            token="test-token",
            previously_missing=previously_missing,
        )
        full_payload = _make_payload(
            workflows=[_wf_entry("wf_x.py", "# source\n")]
        )
        empty_payload = _make_payload(workflows=[])
        worker_path = tmp_path / "workers" / "wf_x.py"

        with patch.object(client, "_fetch_sync_payload", return_value=full_payload):
            client.sync_all(["twin-1"])
        assert worker_path.exists()

        with patch.object(client, "_fetch_sync_payload", return_value=empty_payload):
            first_empty = client.sync_all(["twin-1"])
            assert worker_path.exists()
            assert all("wf_x.py" not in r.removed for r in first_empty)

            second_empty = client.sync_all(["twin-1"])

        assert not worker_path.exists()
        assert any("wf_x.py" in r.removed for r in second_empty)
        assert "wf_x.py" not in previously_missing, "strike state should clear on delete"

    def test_file_recovering_clears_strike(self, tmp_path):
        """Strike count must reset when the cloud re-claims the file,
        so a later transient miss starts fresh from strike 1 instead
        of getting deleted immediately."""
        previously_missing: set[str] = set()
        client = EdgeSyncClient(
            workers_dir=tmp_path / "workers",
            base_url="http://localhost:8000",
            token="test-token",
            previously_missing=previously_missing,
        )
        full_payload = _make_payload(
            workflows=[_wf_entry("wf_x.py", "# source\n")]
        )
        empty_payload = _make_payload(workflows=[])
        worker_path = tmp_path / "workers" / "wf_x.py"

        with patch.object(client, "_fetch_sync_payload", return_value=full_payload):
            client.sync_all(["twin-1"])
        assert worker_path.exists()

        with patch.object(client, "_fetch_sync_payload", return_value=empty_payload):
            client.sync_all(["twin-1"])
        assert worker_path.exists()
        assert "wf_x.py" in previously_missing

        with patch.object(client, "_fetch_sync_payload", return_value=full_payload):
            client.sync_all(["twin-1"])
        assert "wf_x.py" not in previously_missing

        with patch.object(client, "_fetch_sync_payload", return_value=empty_payload):
            results = client.sync_all(["twin-1"])
        assert worker_path.exists(), (
            "after a recovery, the next miss must be treated as a fresh first strike"
        )
        assert all("wf_x.py" not in r.removed for r in results)


class TestAtomicWrite:
    def test_writes_content_to_dest(self, tmp_path):
        dest = tmp_path / "output.py"
        EdgeSyncClient._atomic_write(dest, "# content\n")
        assert dest.read_text() == "# content\n"

    def test_no_temp_file_left_after_success(self, tmp_path):
        dest = tmp_path / "output.py"
        EdgeSyncClient._atomic_write(dest, "# content\n")
        tmp_files = list(tmp_path.glob(".tmp_*"))
        assert tmp_files == []

    def test_creates_parent_dirs(self, tmp_path):
        dest = tmp_path / "subdir" / "output.py"
        EdgeSyncClient._atomic_write(dest, "# content\n")
        assert dest.exists()
