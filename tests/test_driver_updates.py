"""Staging a driver update: pull and verify a release's images, change nothing that runs.

Docker, the registry and the backend are replaced. What is pinned here is the contract
with the backend (`deployments.py`): which images a host stages, what it reports and when,
and that a command delivered twice, or a report that did not get through, does the right
thing.
"""

from __future__ import annotations

import subprocess
import sys
import time
import types
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from cyberwave_edge_core import driver_update_state as state
from cyberwave_edge_core import driver_updates as du
from cyberwave_edge_core import startup

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64

MANIFEST = {
    "placeholders": {"CW_CHANNEL_TAG": "humble-0.2.0"},
    "profiles": {
        "default": {
            "primary": "cyberwaveos/go2:humble-0.2.0",
            "variants": {
                "robot": [
                    {"service": "plant", "image": "cyberwaveos/go2:humble-0.2.0", "digest": DIGEST_A},
                    {"service": "nav", "image": "cyberwaveos/nav2:humble-0.2.0", "digest": DIGEST_B},
                ],
                "sim": [
                    {"service": "plant", "image": "cyberwaveos/go2-sim:humble-0.2.0", "digest": DIGEST_A},
                ],
            },
        },
        "linux-aarch64-jetson": {
            "primary": "cyberwaveos/go2:jetson-humble-0.2.0",
            "variants": {
                "robot": [
                    {
                        "service": "plant",
                        "image": "cyberwaveos/go2:jetson-humble-0.2.0",
                        "digest": DIGEST_A,
                    }
                ]
            },
        },
    },
}


def _payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "command": du.STAGE_COMMAND,
        "request_id": "req-1",
        "twin_uuid": "twin-1",
        "package": "go2-ros2-driver",
        "target_version": "0.2.0",
        "flavor": "humble",
        "manifest": MANIFEST,
    }
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setattr(state, "updates_dir", lambda: tmp_path / "driver-updates")
    monkeypatch.setattr("cyberwave_edge_core.driver_selection._platform_driver_keys", lambda: [])
    monkeypatch.delenv("CYBERWAVE_DRIVER_COMPOSE_VARIANT", raising=False)
    monkeypatch.setattr(startup, "_HANDLED_EDGE_COMMAND_REQUEST_IDS", deque(maxlen=1024))


class Reports:
    """Stands in for `StatusReporter`: records what the edge told the backend."""

    def __init__(self, deliver: bool = True) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.deliver = deliver
        self.superseded = False

    def send(self, phase: str, **fields: Any) -> bool:
        self.sent.append((phase, fields))
        return self.deliver

    def progress(self, *args: Any, **kwargs: Any) -> None:
        self.sent.append(("progress", {"args": args}))

    def phases(self) -> list[str]:
        return [phase for phase, _ in self.sent]


class TestParseRequest:
    def test_a_complete_command_becomes_a_request(self):
        request = du.parse_request(_payload())

        assert request.request_id == "req-1"
        assert request.package == "go2-ros2-driver"
        assert request.target_version == "0.2.0"
        assert request.manifest is MANIFEST

    @pytest.mark.parametrize("manifest", [None, "text", {}, {"profiles": []}])
    def test_a_command_without_manifest_profiles_is_invalid(self, manifest):
        with pytest.raises(du.StageError) as raised:
            du.parse_request(_payload(manifest=manifest))

        assert raised.value.code == du.INVALID_COMMAND

    @pytest.mark.parametrize(
        "request_id", ["", "../../etc/passwd", "a/b", "pulls.json", "pulls", "previous", ".hidden"]
    )
    def test_a_request_id_that_is_not_a_plain_name_is_invalid(self, request_id):
        """It becomes a file name under the config directory."""
        with pytest.raises(du.StageError) as raised:
            du.parse_request(_payload(request_id=request_id))

        assert raised.value.code == du.INVALID_COMMAND


class TestSelectImages:
    def test_a_plain_host_stages_the_default_profiles_robot_variant(self):
        profile, variant, images = du.select_images(MANIFEST)

        assert (profile, variant) == ("default", "robot")
        assert [(image.image, image.digest) for image in images] == [
            ("cyberwaveos/go2:humble-0.2.0", DIGEST_A),
            ("cyberwaveos/nav2:humble-0.2.0", DIGEST_B),
        ]

    def test_the_primary_image_is_staged_once_even_when_a_service_runs_it(self):
        _profile, _variant, images = du.select_images(MANIFEST)

        assert [image.image for image in images].count("cyberwaveos/go2:humble-0.2.0") == 1

    def test_a_jetson_gets_its_own_profile(self, monkeypatch):
        monkeypatch.setattr(
            "cyberwave_edge_core.driver_selection._platform_driver_keys",
            lambda: ["linux-aarch64-jetson", "linux-aarch64", "linux"],
        )

        profile, _variant, images = du.select_images(MANIFEST)

        assert profile == "linux-aarch64-jetson"
        assert [image.image for image in images] == ["cyberwaveos/go2:jetson-humble-0.2.0"]

    def test_a_jetson_falls_back_to_default_when_the_release_has_no_jetson_profile(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "cyberwave_edge_core.driver_selection._platform_driver_keys",
            lambda: ["linux-aarch64-jetson"],
        )
        manifest = {"profiles": {"default": MANIFEST["profiles"]["default"]}}

        profile, _variant, _images = du.select_images(manifest)

        assert profile == "default"

    def test_the_variant_follows_what_this_edge_starts_drivers_with(self, monkeypatch):
        monkeypatch.setenv("CYBERWAVE_DRIVER_COMPOSE_VARIANT", "sim")

        _profile, variant, images = du.select_images(MANIFEST)

        assert variant == "sim"
        assert [image.image for image in images] == [
            "cyberwaveos/go2:humble-0.2.0",
            "cyberwaveos/go2-sim:humble-0.2.0",
        ]

    def test_a_release_with_no_profile_for_this_host_fails_with_a_code(self):
        manifest = {"profiles": {"linux-aarch64-jetson": MANIFEST["profiles"]["linux-aarch64-jetson"]}}

        with pytest.raises(du.StageError) as raised:
            du.select_images(manifest)

        assert raised.value.code == du.NO_MATCHING_PROFILE

    def test_a_profile_without_this_variant_fails_with_a_code(self, monkeypatch):
        monkeypatch.setenv("CYBERWAVE_DRIVER_COMPOSE_VARIANT", "cloud")

        with pytest.raises(du.StageError) as raised:
            du.select_images(MANIFEST)

        assert raised.value.code == du.NO_MATCHING_PROFILE


class TestStage:
    @pytest.fixture
    def host(self, monkeypatch):
        """A host where nothing is pulled yet; `pull` makes the images appear."""
        host = SimpleNamespace(local={}, pulled=[], free_space_checks=0, held_during_pull=set())

        def local_digests(image: str):
            return host.local.get(image)

        def pull(images, reporter, total):
            host.held_during_pull = state.recent_pulls()
            host.pulled.extend(image.image for image in images)
            for image in images:
                host.local[image.image] = [f"{image.image.split(':')[0]}@{image.digest}"]

        def ensure_free_space():
            host.free_space_checks += 1

        monkeypatch.setattr(du, "docker_available", lambda: True)
        monkeypatch.setattr(du, "_local_digests", local_digests)
        monkeypatch.setattr(du, "_pull_images", pull)
        monkeypatch.setattr(du, "ensure_free_space", ensure_free_space)
        return host

    def _stage(self, reporter: Reports | None = None, **overrides: Any) -> Reports:
        reporter = reporter or Reports()
        du.stage(du.parse_request(_payload(**overrides)), reporter)
        return reporter

    def test_it_pulls_the_missing_images_and_writes_the_staged_record(self, host):
        reporter = self._stage()

        assert host.pulled == ["cyberwaveos/go2:humble-0.2.0", "cyberwaveos/nav2:humble-0.2.0"]
        record = state.read_staged_record("req-1")
        assert record["package"] == "go2-ros2-driver"
        assert record["target_version"] == "0.2.0"
        assert record["profile"] == "default" and record["variant"] == "robot"
        assert record["placeholders"] == {"CW_CHANNEL_TAG": "humble-0.2.0"}
        assert [image["image"] for image in record["images"]] == host.pulled
        assert reporter.phases()[0] == "staging"

    def test_it_snapshots_what_the_twin_is_configured_to_run_for_a_later_rollback(
        self, host, monkeypatch
    ):
        running = {"default": {"docker_image": "cyberwaveos/go2:humble-0.1.0"}}
        monkeypatch.setattr(
            startup, "_read_twin_json", lambda twin_uuid: {"metadata": {"drivers": running}}
        )

        self._stage()

        assert state.read_staged_record("req-1")["previous_drivers"] == running

    def test_without_a_twin_json_the_snapshot_is_empty_not_an_error(self, host, monkeypatch):
        monkeypatch.setattr(startup, "_read_twin_json", lambda twin_uuid: None)

        self._stage()

        assert state.read_staged_record("req-1")["previous_drivers"] is None

    def test_the_log_says_which_images_are_staged_and_which_were_pulled(self, host, caplog):
        host.local["cyberwaveos/go2:humble-0.2.0"] = [f"cyberwaveos/go2@{DIGEST_A}"]

        with caplog.at_level("INFO", logger="cyberwave_edge_core.driver_updates"):
            self._stage()

        text = caplog.text
        assert "Staging req-1: go2-ros2-driver 0.2.0 for twin twin-1, profile default, variant robot" in text
        assert f"cyberwaveos/go2:humble-0.2.0@{DIGEST_A[:19]}" in text
        assert f"cyberwaveos/nav2:humble-0.2.0@{DIGEST_B[:19]}" in text
        assert "1 of 2 image(s) already here, pulling cyberwaveos/nav2:humble-0.2.0@" in text
        assert "Staged req-1: 2 image(s) verified and held for twin twin-1" in text

    def test_images_already_here_at_the_right_digest_are_not_pulled_again(self, host):
        host.local["cyberwaveos/go2:humble-0.2.0"] = [f"cyberwaveos/go2@{DIGEST_A}"]
        host.local["cyberwaveos/nav2:humble-0.2.0"] = [f"cyberwaveos/nav2@{DIGEST_B}"]

        self._stage()

        assert host.pulled == []
        assert host.free_space_checks == 0
        assert state.read_staged_record("req-1") is not None

    def test_an_image_here_at_the_wrong_digest_is_pulled_again(self, host):
        host.local["cyberwaveos/go2:humble-0.2.0"] = ["cyberwaveos/go2@sha256:" + "f" * 64]
        host.local["cyberwaveos/nav2:humble-0.2.0"] = [f"cyberwaveos/nav2@{DIGEST_B}"]

        self._stage()

        assert host.pulled == ["cyberwaveos/go2:humble-0.2.0"]

    def test_staging_stops_when_the_hold_on_its_images_cannot_be_written(self, host, monkeypatch):
        monkeypatch.setattr(state, "record_pull", lambda *a, **k: False)

        with pytest.raises(du.StageError) as raised:
            self._stage()

        assert raised.value.code == du.UNKNOWN
        assert host.pulled == [] and host.free_space_checks == 0

    def test_images_already_here_are_held_against_the_cleanup_while_others_download(self, host):
        host.local["cyberwaveos/go2:humble-0.2.0"] = [f"cyberwaveos/go2@{DIGEST_A}"]

        self._stage()

        assert host.pulled == ["cyberwaveos/nav2:humble-0.2.0"]
        assert "cyberwaveos/go2:humble-0.2.0" in host.held_during_pull

    def test_an_image_that_vanishes_while_staging_fails_the_staging(self, host, monkeypatch):
        host.local["cyberwaveos/go2:humble-0.2.0"] = [f"cyberwaveos/go2@{DIGEST_A}"]
        host.local["cyberwaveos/nav2:humble-0.2.0"] = [f"cyberwaveos/nav2@{DIGEST_B}"]
        original = du._is_verified
        calls = {"n": 0}

        def vanishing(image):
            calls["n"] += 1
            # The first sweep sees it; by the final check the cleanup has removed it.
            return original(image) if calls["n"] <= 2 else False

        monkeypatch.setattr(du, "_is_verified", vanishing)

        with pytest.raises(du.StageError) as raised:
            self._stage()

        assert raised.value.code == du.DIGEST_MISMATCH
        assert state.read_staged_record("req-1") is None

    def test_a_newer_staging_replaces_the_older_one_of_the_same_twin_and_package(self, host):
        self._stage(request_id="old")
        self._stage(request_id="other-twin", twin_uuid="twin-2")
        self._stage(request_id="other-package", package="nav2-driver")

        self._stage(request_id="new")

        assert state.read_staged_record("old") is None
        assert state.read_staged_record("new") is not None
        assert state.read_staged_record("other-twin") is not None
        assert state.read_staged_record("other-package") is not None

    def test_free_space_is_checked_before_a_pull(self, host):
        self._stage()

        assert host.free_space_checks == 1

    def test_a_pulled_image_that_is_not_the_named_build_fails_and_stages_nothing(
        self, host, monkeypatch
    ):
        def pull_wrong_build(images, reporter, total):
            for image in images:
                host.local[image.image] = [f"{image.image.split(':')[0]}@sha256:" + "0" * 64]

        monkeypatch.setattr(du, "_pull_images", pull_wrong_build)

        with pytest.raises(du.StageError) as raised:
            self._stage()

        assert raised.value.code == du.DIGEST_MISMATCH
        assert state.read_staged_record("req-1") is None

    def test_an_already_staged_request_pulls_nothing(self, host):
        self._stage()
        host.pulled.clear()

        self._stage()

        assert host.pulled == []

    def test_without_docker_it_fails_with_a_code(self, host, monkeypatch):
        monkeypatch.setattr(du, "docker_available", lambda: False)

        with pytest.raises(du.StageError) as raised:
            self._stage()

        assert raised.value.code == du.DOCKER_UNAVAILABLE

    def test_it_changes_nothing_that_runs(self, host, monkeypatch):
        """Staging only pulls and writes under driver-updates/: no container, no twin."""
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: pytest.fail("staging must not run docker itself")
        )

        self._stage()


class TestFreeSpace:
    def test_enough_space_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(du, "docker_data_root", lambda: "/var/lib/docker")
        monkeypatch.setattr(du.shutil, "disk_usage", lambda _p: SimpleNamespace(free=50 * 1024**3))
        collected: list[bool] = []
        monkeypatch.setattr(
            "cyberwave_edge_core.image_gc.gc_driver_images", lambda **kw: collected.append(True)
        )

        du.ensure_free_space()

        assert collected == []

    def test_short_space_collects_unused_images_and_proceeds_if_that_was_enough(
        self, monkeypatch
    ):
        free = iter([1 * 1024**3, 20 * 1024**3])
        monkeypatch.setattr(du, "docker_data_root", lambda: "/var/lib/docker")
        monkeypatch.setattr(du.shutil, "disk_usage", lambda _p: SimpleNamespace(free=next(free)))
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "cyberwave_edge_core.image_gc.gc_driver_images", lambda **kw: calls.append(kw)
        )

        du.ensure_free_space()

        assert calls == [{"aggressive": True, "keep": set()}]

    def test_short_space_keeps_the_image_of_a_flap_stopped_driver(self, monkeypatch):
        free = iter([1 * 1024**3, 20 * 1024**3])
        monkeypatch.setattr(du, "docker_data_root", lambda: "/var/lib/docker")
        monkeypatch.setattr(du.shutil, "disk_usage", lambda _p: SimpleNamespace(free=next(free)))
        monkeypatch.setitem(
            startup._FLAP_STOPPED,
            "cyberwave-driver-gone",
            startup._FlapStop(backoff=120, due_at=0, image="sha256:flap"),
        )
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "cyberwave_edge_core.image_gc.gc_driver_images", lambda **kw: calls.append(kw)
        )

        du.ensure_free_space()

        assert calls == [{"aggressive": True, "keep": {"sha256:flap"}}]

    def test_still_short_after_collecting_fails_with_insufficient_disk(self, monkeypatch):
        monkeypatch.setattr(du, "docker_data_root", lambda: "/var/lib/docker")
        monkeypatch.setattr(du.shutil, "disk_usage", lambda _p: SimpleNamespace(free=1 * 1024**3))
        monkeypatch.setattr("cyberwave_edge_core.image_gc.gc_driver_images", lambda **kw: True)

        with pytest.raises(du.StageError) as raised:
            du.ensure_free_space()

        assert raised.value.code == du.INSUFFICIENT_DISK
        assert "GiB free" in str(raised.value)

    def test_the_floor_can_be_set_by_environment(self, monkeypatch):
        monkeypatch.setenv(du.MIN_FREE_GB_ENV, "2")
        monkeypatch.setattr(du, "docker_data_root", lambda: "/var/lib/docker")
        monkeypatch.setattr(du.shutil, "disk_usage", lambda _p: SimpleNamespace(free=3 * 1024**3))

        du.ensure_free_space()


class TestRun:
    """The worker: what it reports, and when it lets the backend's next delivery through."""

    def _request(self) -> du.StageRequest:
        return du.parse_request(_payload())

    def _remembered(self) -> bool:
        return startup._edge_command_key(du.STAGE_COMMAND, "req-1") in (
            startup._HANDLED_EDGE_COMMAND_REQUEST_IDS
        )

    @pytest.fixture
    def reporter(self, monkeypatch) -> Reports:
        reporter = Reports()
        monkeypatch.setattr(du, "StatusReporter", lambda request_id: reporter)
        startup._HANDLED_EDGE_COMMAND_REQUEST_IDS.append(
            startup._edge_command_key(du.STAGE_COMMAND, "req-1")
        )
        return reporter

    def test_success_reports_staged_and_keeps_the_command_handled(self, reporter, monkeypatch):
        monkeypatch.setattr(du, "stage", lambda request, reporter: None)

        du._run(self._request())

        assert reporter.phases() == ["staged"]
        assert self._remembered()

    def test_a_stage_error_reports_failed_with_its_code(self, reporter, monkeypatch):
        def fail(request, reporter):
            raise du.StageError(du.INSUFFICIENT_DISK, "1.0 GiB free")

        monkeypatch.setattr(du, "stage", fail)

        du._run(self._request())

        assert reporter.sent == [
            ("failed", {"error_code": du.INSUFFICIENT_DISK, "error": "1.0 GiB free"})
        ]
        assert self._remembered()

    def test_an_unexpected_error_reports_failed_unknown(self, reporter, monkeypatch):
        def explode(request, reporter):
            raise RuntimeError("boom")

        monkeypatch.setattr(du, "stage", explode)

        du._run(self._request())

        (phase, fields), = reporter.sent
        assert phase == "failed" and fields["error_code"] == du.UNKNOWN
        assert "boom" in fields["error"]

    def test_a_report_that_did_not_get_through_lets_the_next_delivery_retry(
        self, reporter, monkeypatch
    ):
        reporter.deliver = False
        monkeypatch.setattr(du, "stage", lambda request, reporter: None)

        du._run(self._request())

        assert not self._remembered()

    def test_a_deployment_the_backend_dropped_is_not_retried(self, reporter, monkeypatch):
        def supersede(request, reporter_):
            reporter.superseded = True

        reporter.deliver = False
        monkeypatch.setattr(du, "stage", supersede)

        du._run(self._request())

        assert self._remembered()


class TestHandleCommand:
    @pytest.fixture
    def inline(self, monkeypatch):
        """Run submitted work now, and record it."""
        jobs: list[tuple[Any, ...]] = []

        class _Inline:
            def submit(self, fn, *args):
                jobs.append((fn.__name__, *args))

        monkeypatch.setattr(du, "_executor", lambda: _Inline())
        return jobs

    def test_a_valid_command_is_handed_to_the_worker_and_returns(self, inline):
        du.handle_stage_command(_payload())

        assert [job[0] for job in inline] == ["_run"]

    def test_a_command_with_an_id_but_no_manifest_is_reported_failed(self, inline):
        du.handle_stage_command(_payload(manifest=None))

        name, request_id, error = inline[0]
        assert (name, request_id) == ("_reject", "req-1")
        assert error.code == du.INVALID_COMMAND

    def test_a_command_with_no_usable_id_is_dropped_quietly(self, inline):
        du.handle_stage_command(_payload(request_id="", manifest=None))

        assert inline == []

    def test_the_command_is_registered_with_the_edge(self):
        assert startup._EDGE_COMMAND_HANDLERS[du.STAGE_COMMAND] is du.handle_stage_command


class TestStatusReporter:
    @pytest.fixture
    def api(self, monkeypatch):
        calls: list[dict[str, Any]] = []

        class _Api:
            def param_serialize(self, **kwargs):
                calls.append(kwargs)
                return ("args",)

            def call_api(self, *_args):
                return SimpleNamespace(read=lambda: None)

            def response_deserialize(self, **_kwargs):
                if self.error:
                    raise self.error
                return SimpleNamespace(data={})

            error: Exception | None = None

        api = _Api()
        client = SimpleNamespace(api=SimpleNamespace(api_client=api))
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "_build_cyberwave_client", lambda token: client)
        monkeypatch.setattr(startup, "_resolve_edge_uuid", lambda token: "edge-1")
        api.calls = calls
        return api

    def test_it_posts_the_phase_to_the_deployments_status_endpoint(self, api):
        reporter = du.StatusReporter("req-1")

        assert reporter.send("staged") is True

        (call,) = api.calls
        assert call["method"] == "POST"
        assert call["resource_path"] == (
            "/api/v1/edges/{uuid}/driver-deployments/{request_id}/status"
        )
        assert call["path_params"] == {"uuid": "edge-1", "request_id": "req-1"}
        assert call["body"] == {"phase": "staged"}

    def test_a_failure_carries_its_code_and_message(self, api):
        du.StatusReporter("req-1").send("failed", error_code="digest_mismatch", error="nope")

        assert api.calls[0]["body"] == {
            "phase": "failed",
            "error_code": "digest_mismatch",
            "error": "nope",
        }

    def test_what_runs_after_a_switch_is_sent_as_observed(self, api):
        du.StatusReporter("req-1").send("active", observed={"cyberwave-driver-1": "drv:2.0.0"})

        assert api.calls[0]["body"] == {
            "phase": "active",
            "observed": {"cyberwave-driver-1": "drv:2.0.0"},
        }

    def test_progress_is_sent_in_the_shape_the_backend_stores(self, api):
        du.StatusReporter("req-1").progress(1, 3, 100, 900)

        assert api.calls[0]["body"] == {
            "phase": "staging",
            "progress": {
                "images_done": 1,
                "images_total": 3,
                "bytes_done": 100,
                "bytes_total": 900,
            },
        }

    def test_progress_is_rate_limited_unless_forced(self, api):
        reporter = du.StatusReporter("req-1")

        reporter.progress(0, 3, 1, 9)
        reporter.progress(0, 3, 2, 9)
        reporter.progress(1, 3, 9, 9, force=True)

        assert len(api.calls) == 2

    def test_an_unknown_deployment_is_dropped_not_retried(self, api):
        api.error = type("NotFound", (Exception,), {"status": 404})()
        reporter = du.StatusReporter("req-1")

        assert reporter.send("staged") is True
        assert reporter.superseded is True

    def test_any_other_failure_is_reported_as_undelivered(self, api):
        api.error = RuntimeError("backend down")
        reporter = du.StatusReporter("req-1")

        assert reporter.send("staged") is False
        assert reporter.superseded is False

    def test_no_token_is_undelivered(self, api, monkeypatch):
        monkeypatch.setattr(startup, "load_token", lambda: None)

        assert du.StatusReporter("req-1").send("staged") is False


class TestPullImages:
    @pytest.fixture
    def pulls(self, monkeypatch):
        seen = SimpleNamespace(images=[], fail=set())

        def fake_pull(image, *, contexts, token, timeout, on_progress):
            seen.images.append((image, timeout))
            if image in seen.fail:
                raise subprocess.CalledProcessError(1, "docker pull")
            on_progress(SimpleNamespace(downloaded_bytes=50, total_bytes=100))
            return SimpleNamespace()

        seen.fake_pull = fake_pull
        monkeypatch.setattr(
            "cyberwave_edge_core.driver_logs._pull_docker_image_with_progress_multi", fake_pull
        )
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        # One at a time, so the progress reports arrive in a known order.
        monkeypatch.setattr(du, "PULL_WORKERS", 1)
        return seen

    def test_every_image_is_pulled_recorded_and_counted(self, pulls):
        reporter = Reports()
        images = [du.ManifestImage("a:1.0.0", DIGEST_A), du.ManifestImage("b:1.0.0", DIGEST_B)]

        du._pull_images(images, reporter, total=2)

        assert sorted(image for image, _ in pulls.images) == ["a:1.0.0", "b:1.0.0"]
        assert {"a:1.0.0", "b:1.0.0"} <= state.recent_pulls()
        assert reporter.sent[-1][1]["args"][:2] == (2, 2)

    def test_images_already_here_count_as_done(self, pulls):
        reporter = Reports()

        du._pull_images([du.ManifestImage("b:1.0.0", DIGEST_B)], reporter, total=3)

        assert reporter.sent[-1][1]["args"][:2] == (3, 3)

    def test_a_pull_gets_the_long_timeout(self, pulls):
        du._pull_images([du.ManifestImage("a:1.0.0")], Reports(), total=1)

        assert pulls.images == [("a:1.0.0", du.PULL_TIMEOUT_SECONDS)]

    def test_a_failed_pull_fails_with_a_code_naming_the_image(self, pulls):
        pulls.fail.add("a:1.0.0")

        with pytest.raises(du.StageError) as raised:
            du._pull_images([du.ManifestImage("a:1.0.0")], Reports(), total=1)

        assert raised.value.code == du.PULL_FAILED
        assert "a:1.0.0" in str(raised.value)

    def test_a_failed_pull_stops_the_images_still_queued(self, pulls):
        pulls.fail.add("a:1.0.0")
        images = [du.ManifestImage(f"{name}:1.0.0") for name in "abcdef"]

        with pytest.raises(du.StageError):
            du._pull_images(images, Reports(), total=6)

        assert [image for image, _ in pulls.images] == ["a:1.0.0"]

    def test_a_failure_behind_a_slow_pull_is_seen_when_it_happens(self, pulls, monkeypatch):
        """Not only once the images ahead of it in the list have finished."""
        pulls.fail.add("b:1.0.0")
        original = pulls.fake_pull

        def fake_pull(image, **kwargs):
            if image == "a:1.0.0":
                time.sleep(0.4)  # still downloading when b fails
            return original(image, **kwargs)

        monkeypatch.setattr(
            "cyberwave_edge_core.driver_logs._pull_docker_image_with_progress_multi", fake_pull
        )
        monkeypatch.setattr(du, "PULL_WORKERS", 2)
        images = [du.ManifestImage(f"{name}:1.0.0") for name in "abcde"]

        with pytest.raises(du.StageError):
            du._pull_images(images, Reports(), total=5)

        # One more may have been picked up in the instant before the failure was handled.
        assert {image for image, _ in pulls.images} <= {"a:1.0.0", "b:1.0.0", "c:1.0.0"}

    def test_nothing_more_is_pulled_once_the_deployment_is_superseded(self, pulls):
        reporter = Reports()
        reporter.superseded = True

        du._pull_images([du.ManifestImage("a:1.0.0")], reporter, total=1)

        assert pulls.images == []
