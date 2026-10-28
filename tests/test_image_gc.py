"""The image garbage collector: it removes what nothing needs, and only that.

Docker is replaced by a table of images and containers. The point of these tests is the
keep-set: an image a staged update will start must survive a cleanup that would have
removed it under the old ``docker image prune --all --filter until=2h``.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any

import pytest

from cyberwave_edge_core import driver_update_state as state
from cyberwave_edge_core import image_gc

HOUR = 3600
NOW = 1_800_000_000.0


def _created(age_hours: float) -> str:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(NOW - age_hours * HOUR))
    return f"{stamp}.123456789Z"


class FakeDocker:
    """`docker` as the collector uses it: images, containers, and what was removed."""

    def __init__(self) -> None:
        self.images: dict[str, dict[str, Any]] = {}
        self.containers: dict[str, str] = {}
        self.removed: list[list[str]] = []
        self.fail_rmi: set[str] = set()

    def add(
        self, image_id: str, *, age_hours: float = 24, tags: list[str] | None = None,
        digests: list[str] | None = None,
    ) -> None:
        self.images[image_id] = {
            "created": _created(age_hours),
            "tags": tags or [],
            "digests": digests or [],
        }

    def run(self, *args: str) -> str:
        if args[:2] == ("image", "ls"):
            return "\n".join(self.images) + "\n"
        if args[:2] == ("image", "inspect"):
            ids = [arg for arg in args if arg in self.images]
            return "".join(
                f"{i}\t{self.images[i]['created']}\t{json.dumps(self.images[i]['tags'])}"
                f"\t{json.dumps(self.images[i]['digests'])}\n"
                for i in ids
            )
        if args[0] == "ps":
            return "\n".join(self.containers) + "\n"
        if args[0] == "inspect":
            return "".join(f"{self.containers[c]}\n" for c in args if c in self.containers)
        if args[0] == "rmi":
            if any(target in self.fail_rmi for target in args[1:]):
                raise subprocess.CalledProcessError(1, "docker rmi", stderr="in use")
            self.removed.append(list(args[1:]))
            return ""
        raise AssertionError(f"unexpected docker call: {args}")


@pytest.fixture
def docker(monkeypatch: Any, tmp_path: Any) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr(image_gc, "docker_available", lambda: True)
    monkeypatch.setattr(image_gc, "_docker", fake.run)
    monkeypatch.setattr(state, "updates_dir", lambda: tmp_path / "driver-updates")
    return fake


def _gc(**kwargs: Any) -> bool:
    return image_gc.gc_driver_images(now=NOW, **kwargs)


class TestWhatIsRemoved:
    def test_an_old_unused_image_is_removed_by_its_tag(self, docker):
        docker.add("sha256:old", tags=["drv:1.0.0"])

        assert _gc() is True

        assert docker.removed == [["drv:1.0.0"]]

    def test_an_untagged_image_is_removed_by_id(self, docker):
        docker.add("sha256:dangling")

        _gc()

        assert docker.removed == [["sha256:dangling"]]

    def test_every_tag_of_an_image_goes_together(self, docker):
        docker.add("sha256:two", tags=["drv:1.0.0", "drv:humble"])

        _gc()

        assert docker.removed == [["drv:1.0.0", "drv:humble"]]

    def test_the_oldest_goes_first(self, docker):
        docker.add("sha256:newer", age_hours=5, tags=["b:1"])
        docker.add("sha256:oldest", age_hours=90, tags=["a:1"])
        docker.add("sha256:older", age_hours=30, tags=["c:1"])

        _gc()

        assert docker.removed == [["a:1"], ["c:1"], ["b:1"]]

    def test_a_failed_removal_does_not_stop_the_rest(self, docker):
        docker.add("sha256:stuck", age_hours=50, tags=["stuck:1"])
        docker.add("sha256:fine", age_hours=10, tags=["fine:1"])
        docker.fail_rmi.add("stuck:1")

        assert _gc() is True

        assert docker.removed == [["fine:1"]]


class TestWhatIsKept:
    def test_an_image_a_container_uses_is_kept_even_when_stopped(self, docker):
        docker.add("sha256:used", tags=["drv:1.0.0"])
        docker.containers["c1"] = "sha256:used"

        _gc()

        assert docker.removed == []

    def test_an_image_created_in_the_last_two_hours_is_kept(self, docker):
        """The old `until=2h` guard: a fresh local build no container has started yet."""
        docker.add("sha256:fresh", age_hours=1.5, tags=["local:dev"])

        _gc()

        assert docker.removed == []

    def test_a_staged_image_is_kept_however_old_it_was_built(self, docker, tmp_path):
        """The reason this collector exists: pulled now, built last month, started in a week."""
        docker.add("sha256:staged", age_hours=24 * 30, tags=["drv:2.0.0"])
        state.write_staged_record(
            {"request_id": "req-1", "images": [{"image": "drv:2.0.0", "digest": "sha256:abc"}]}
        )

        _gc()

        assert docker.removed == []

    def test_a_staged_image_named_by_digest_is_kept(self, docker):
        docker.add("sha256:staged", digests=["repo/drv@sha256:" + "a" * 64], tags=["drv:2.0.0"])
        state.write_staged_record(
            {"request_id": "req-1", "images": [{"image": "repo/drv@sha256:" + "a" * 64}]}
        )

        _gc()

        assert docker.removed == []

    def test_an_image_pulled_in_the_last_two_hours_is_kept(self, docker):
        docker.add("sha256:justpulled", age_hours=500, tags=["drv:1.1.0"])
        state.record_pull("drv:1.1.0", now=NOW - 0.5 * HOUR)

        _gc()

        assert docker.removed == []

    def test_a_pull_older_than_two_hours_no_longer_protects(self, docker):
        docker.add("sha256:stale", age_hours=500, tags=["drv:1.1.0"])
        state.record_pull("drv:1.1.0", now=NOW - 3 * HOUR)

        _gc()

        assert docker.removed == [["drv:1.1.0"]]

    def test_a_staged_image_shared_with_another_tag_keeps_the_whole_image(self, docker):
        docker.add("sha256:shared", tags=["drv:2.0.0", "drv:humble"])
        state.write_staged_record({"request_id": "req-1", "images": [{"image": "drv:2.0.0"}]})

        _gc()

        assert docker.removed == []


class TestReferencesAreComparedAsDockerListsThem:
    @pytest.mark.parametrize(
        ("staged", "listed"),
        [
            ("docker.io/org/drv:2.0.0", "org/drv:2.0.0"),
            ("index.docker.io/org/drv:2.0.0", "org/drv:2.0.0"),
            ("docker.io/library/drv:2.0.0", "drv:2.0.0"),
            ("org/drv", "org/drv:latest"),
            ("docker.io/org/drv", "org/drv:latest"),
            ("ghcr.io/org/drv:2.0.0", "ghcr.io/org/drv:2.0.0"),
        ],
    )
    def test_a_staged_image_is_kept_whatever_way_the_manifest_spells_it(
        self, docker, staged, listed
    ):
        docker.add("sha256:staged", age_hours=24 * 30, tags=[listed])
        state.write_staged_record({"request_id": "req-1", "images": [{"image": staged}]})

        _gc()

        assert docker.removed == []

    def test_a_staged_image_is_kept_by_its_digest_even_when_its_tag_has_moved(self, docker):
        digest = "sha256:" + "d" * 64
        docker.add("sha256:staged", age_hours=24 * 30, tags=["drv:renamed"], digests=[f"org/drv@{digest}"])
        state.write_staged_record(
            {"request_id": "req-1", "images": [{"image": "org/drv:2.0.0", "digest": digest}]}
        )

        _gc()

        assert docker.removed == []

    def test_a_different_registry_is_a_different_image(self, docker):
        docker.add("sha256:other", age_hours=24 * 30, tags=["ghcr.io/org/drv:2.0.0"])
        state.write_staged_record({"request_id": "req-1", "images": [{"image": "org/drv:2.0.0"}]})

        _gc()

        assert docker.removed == [["ghcr.io/org/drv:2.0.0"]]


class TestRollbackSets:
    def _previous(self, tmp_path: Any, keep_until: float) -> None:
        directory = tmp_path / "driver-updates"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / state.PREVIOUS_FILE).write_text(
            json.dumps({"twin-1": {"images": ["drv:0.9.0"], "keep_until": keep_until}})
        )

    def test_a_rollback_set_inside_its_period_is_kept(self, docker, tmp_path):
        docker.add("sha256:prev", tags=["drv:0.9.0"])
        self._previous(tmp_path, NOW + HOUR)

        _gc()

        assert docker.removed == []

    def test_a_rollback_set_past_its_period_is_released(self, docker, tmp_path):
        docker.add("sha256:prev", tags=["drv:0.9.0"])
        self._previous(tmp_path, NOW - HOUR)

        _gc()

        assert docker.removed == [["drv:0.9.0"]]

    def test_a_short_disk_releases_the_rollback_set_early_but_never_a_staged_image(
        self, docker, tmp_path
    ):
        docker.add("sha256:prev", age_hours=48, tags=["drv:0.9.0"])
        docker.add("sha256:staged", age_hours=48, tags=["drv:2.0.0"])
        self._previous(tmp_path, NOW + 20 * HOUR)
        state.write_staged_record({"request_id": "req-1", "images": [{"image": "drv:2.0.0"}]})

        _gc(aggressive=True)

        assert docker.removed == [["drv:0.9.0"]]


class TestWhenItCannotRun:
    def test_no_docker_is_not_an_error_but_does_nothing(self, monkeypatch):
        monkeypatch.setattr(image_gc, "docker_available", lambda: False)

        assert _gc() is False

    @pytest.mark.parametrize(
        "error",
        [
            subprocess.CalledProcessError(1, "docker"),
            subprocess.TimeoutExpired("docker", 120),
            OSError("no docker"),
        ],
    )
    def test_a_failing_listing_reports_false_and_removes_nothing(self, monkeypatch, docker, error):
        docker.add("sha256:old", tags=["drv:1.0.0"])

        def boom(*_args: str) -> str:
            raise error

        monkeypatch.setattr(image_gc, "_docker", boom)

        assert _gc() is False
        assert docker.removed == []

    def test_an_unreadable_staged_record_stops_the_cleanup_rather_than_guessing(
        self, docker, tmp_path
    ):
        docker.add("sha256:old", age_hours=24 * 30, tags=["drv:1.0.0"])
        directory = tmp_path / "driver-updates"
        directory.mkdir()
        (directory / "req-1.json").write_text("{not json")

        assert _gc() is False
        assert docker.removed == []

    def test_an_empty_host_is_fine(self, docker):
        assert _gc() is True
        assert docker.removed == []


def test_the_periodic_cleanup_entry_point_runs_this_collector(monkeypatch):
    from cyberwave_edge_core import docker_helpers

    calls: list[bool] = []
    monkeypatch.setattr(image_gc, "gc_driver_images", lambda **kw: calls.append(True) or True)

    assert docker_helpers.docker_prune_unused_images() is True
    assert calls == [True]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-10-26T15:00:00.123456789Z", 1793026800.0),
        ("2026-10-26T15:00:00Z", 1793026800.0),
        ("", 0.0),
        ("not a time", 0.0),
    ],
)
def test_docker_timestamps_are_read_as_utc(text, expected):
    assert image_gc._parse_created(text) == expected
