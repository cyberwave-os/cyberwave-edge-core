"""What the edge remembers about staged updates: small files, read defensively."""

from __future__ import annotations

import json
from typing import Any

import pytest

from cyberwave_edge_core import driver_update_state as state

NOW = 1_800_000_000.0
HOUR = 3600


@pytest.fixture(autouse=True)
def _dir(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setattr(state, "updates_dir", lambda: tmp_path / "driver-updates")


class TestStagedRecords:
    def test_a_record_round_trips(self):
        record = {"request_id": "req-1", "images": [{"image": "a:1"}], "package": "go2"}

        state.write_staged_record(record)

        assert state.read_staged_record("req-1") == record

    def test_no_record_reads_as_none(self):
        assert state.read_staged_record("req-1") is None

    def test_a_corrupt_record_reads_as_none_not_an_error(self, tmp_path):
        state.write_staged_record({"request_id": "req-1"})
        (tmp_path / "driver-updates" / "req-1.json").write_text("{not json")

        assert state.read_staged_record("req-1") is None

    @pytest.mark.parametrize("request_id", ["", "..", "../x", "a/b", ".hidden", "pulls.json", "pulls", "previous"])
    def test_an_id_that_is_not_a_plain_name_never_becomes_a_path(self, request_id):
        with pytest.raises(ValueError):
            state.staged_record_path(request_id)

    def test_the_files_that_are_not_records_are_not_read_as_records(self, tmp_path):
        state.write_staged_record({"request_id": "req-1", "images": [{"image": "a:1"}]})
        state.record_pull("b:1", now=NOW)
        (tmp_path / "driver-updates" / state.PREVIOUS_FILE).write_text("{}")

        assert [record["request_id"] for record in state.read_staged_records()] == ["req-1"]

    def test_the_images_of_every_staged_record_are_listed(self):
        state.write_staged_record({"request_id": "r1", "images": [{"image": "a:1"}]})
        state.write_staged_record(
            {"request_id": "r2", "images": [{"image": "b:1"}, {"image": "a:1"}, {"service": "x"}]}
        )

        assert state.staged_images() == {"a:1", "b:1"}

    def test_no_directory_means_nothing_staged(self):
        assert state.read_staged_records() == []
        assert state.staged_images() == set()


class TestRecentPulls:
    def test_a_pull_is_recent_for_two_hours(self):
        state.record_pull("a:1", now=NOW - HOUR)

        assert state.recent_pulls(now=NOW) == {"a:1"}
        assert state.recent_pulls(now=NOW + 2 * HOUR) == set()

    def test_old_pulls_are_forgotten_when_a_new_one_is_written(self, tmp_path):
        state.record_pull("old:1", now=NOW - 5 * HOUR)
        state.record_pull("new:1", now=NOW)

        stored = json.loads((tmp_path / "driver-updates" / state.PULLS_FILE).read_text())

        assert list(stored) == ["new:1"]

    def test_recording_a_pull_never_raises(self, monkeypatch):
        monkeypatch.setattr(state, "_write_json", lambda *a: (_ for _ in ()).throw(OSError("full")))

        state.record_pull("a:1")

    def test_a_garbled_file_reads_as_nothing_pulled(self, tmp_path):
        directory = tmp_path / "driver-updates"
        directory.mkdir()
        (directory / state.PULLS_FILE).write_text("[1, 2")

        assert state.recent_pulls(now=NOW) == set()


class TestPreviousSets:
    def _write(self, tmp_path: Any, data: Any) -> None:
        directory = tmp_path / "driver-updates"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / state.PREVIOUS_FILE).write_text(json.dumps(data))

    def test_a_set_is_kept_until_its_time(self, tmp_path):
        self._write(
            tmp_path,
            {
                "t1": {"images": ["a:1", "b:1"], "keep_until": NOW + HOUR},
                "t2": {"images": ["c:1"], "keep_until": NOW - HOUR},
            },
        )

        assert state.previous_image_sets(now=NOW) == {"a:1", "b:1"}

    @pytest.mark.parametrize("data", [None, [], {"t": "x"}, {"t": {"images": ["a:1"]}}])
    def test_malformed_entries_keep_nothing(self, tmp_path, data):
        self._write(tmp_path, data)

        assert state.previous_image_sets(now=NOW) == set()


class TestForgetting:
    def test_a_deleted_record_stops_protecting_its_images(self):
        state.write_staged_record({"request_id": "req-1", "images": [{"image": "a:1"}]})

        state.delete_staged_record("req-1")
        state.delete_staged_record("req-1")  # already gone is fine

        assert state.staged_images() == set()

    def test_an_unusable_id_is_left_alone(self):
        state.delete_staged_record("../x")

    def test_records_without_a_twin_or_package_are_never_superseded(self):
        state.write_staged_record({"request_id": "a", "images": []})
        state.write_staged_record({"request_id": "b", "images": []})
        state.write_staged_record({"request_id": "c", "twin_uuid": "t1", "images": []})
        state.write_staged_record({"request_id": "d", "twin_uuid": "t1", "images": []})

        for request_id in "abcd":
            state.delete_superseded_records(state.read_staged_record(request_id))

        assert [r["request_id"] for r in state.read_staged_records()] == ["a", "b", "c", "d"]

    def test_an_unreadable_record_is_reported(self, tmp_path):
        assert state.staged_records_unreadable() is False
        state.write_staged_record({"request_id": "req-1", "images": []})
        assert state.staged_records_unreadable() is False

        (tmp_path / "driver-updates" / "req-2.json").write_text("{broken")

        assert state.staged_records_unreadable() is True

    def test_only_records_of_the_same_twin_and_package_are_superseded(self):
        for request_id, twin, package in [
            ("old", "t1", "p1"),
            ("same-twin-other-package", "t1", "p2"),
            ("other-twin", "t2", "p1"),
            ("new", "t1", "p1"),
        ]:
            state.write_staged_record(
                {"request_id": request_id, "twin_uuid": twin, "package": package, "images": []}
            )

        state.delete_superseded_records(state.read_staged_record("new"))

        assert [r["request_id"] for r in state.read_staged_records()] == [
            "new",
            "other-twin",
            "same-twin-other-package",
        ]


class TestWaiting:
    def test_a_record_waits_until_its_images_run(self):
        state.write_staged_record(
            {"request_id": "req-1", "twin_uuid": "t1", "images": [{"image": "a:1"}, {"image": "b:1"}]}
        )

        assert state.twin_has_staged_update("t1")
        assert state.twin_has_staged_update("t1", running=["a:1"])
        assert not state.twin_has_staged_update("t1", running=["b:1", "a:1", "c:1"])

    def test_another_twins_record_does_not_wait_here(self):
        state.write_staged_record({"request_id": "req-1", "twin_uuid": "t2", "images": []})

        assert not state.twin_has_staged_update("t1")

    def test_a_record_naming_no_images_always_waits(self):
        state.write_staged_record({"request_id": "req-1", "twin_uuid": "t1", "images": []})

        assert state.twin_has_staged_update("t1", running=["a:1"])
