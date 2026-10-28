"""The rest of what activation changes: who stays out of its way, and the options it needs.

* the claim that keeps revival, the settings reconciler and a restart from interleaving;
* `fetch_and_run_twin_drivers(only_twin_uuids=..., drivers_override=...)`, which recreate a whole
  twin and roll one back;
* pinned image tags, which are not pulled again when the build is already here;
* the capability, which every edge advertises.
"""

from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import cyberwave_edge_core.startup as startup
from cyberwave_edge_core import driver_updates
from cyberwave_edge_core.docker_helpers import driver_container_name

TWIN = "11111111-2222-4333-8444-555555555555"
OTHER = "99999999-8888-7777-6666-555555555555"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: Any) -> None:
    monkeypatch.setattr(startup, "_ACTIVATING_TWINS", set())
    monkeypatch.setattr(startup, "_LAUNCHING_TWINS", {})
    monkeypatch.setattr(startup, "_edge_core_restart_in_progress", lambda: False)
    monkeypatch.setattr(startup, "_CAMERA_DRIFT_RESTART_IN_PROGRESS", False)


class TestTheClaim:
    def test_a_twin_can_be_claimed_once(self):
        assert startup.begin_driver_activation(TWIN) is True
        assert startup.begin_driver_activation(TWIN) is False

    def test_another_twin_is_not_blocked(self):
        startup.begin_driver_activation(TWIN)

        assert startup.begin_driver_activation(OTHER) is True

    def test_ending_it_releases_the_twin(self):
        startup.begin_driver_activation(TWIN)
        startup.end_driver_activation(TWIN)

        assert startup.driver_activation_in_progress(TWIN) is False
        assert startup.begin_driver_activation(TWIN) is True

    def test_ending_a_twin_that_was_not_claimed_is_harmless(self):
        startup.end_driver_activation(TWIN)

    def test_it_is_refused_while_edge_core_restarts(self, monkeypatch):
        monkeypatch.setattr(startup, "_edge_core_restart_in_progress", lambda: True)

        assert startup.begin_driver_activation(TWIN) is False
        assert startup.driver_activation_in_progress(TWIN) is False

    def test_it_is_refused_during_a_camera_drift_restart(self, monkeypatch):
        monkeypatch.setattr(startup, "_CAMERA_DRIFT_RESTART_IN_PROGRESS", True)

        assert startup.begin_driver_activation(TWIN) is False

    def test_any_twin_or_one_twin_can_be_asked_about(self):
        assert startup.driver_activation_in_progress() is False
        startup.begin_driver_activation(TWIN)

        assert startup.driver_activation_in_progress() is True
        assert startup.driver_activation_in_progress(TWIN) is True
        assert startup.driver_activation_in_progress(OTHER) is False


class TestTheLaunchClaim:
    """The claim holds both ways: the runtime loop's launches also keep an activation out."""

    def test_an_activation_is_refused_while_the_loop_launches_the_twin(self):
        assert startup.begin_loop_launch([TWIN]) is True

        assert startup.begin_driver_activation(TWIN) is False
        assert startup.begin_driver_activation(OTHER) is True

    def test_it_is_refused_to_the_loop_while_the_twin_is_activated(self):
        startup.begin_driver_activation(TWIN)

        assert startup.begin_loop_launch([OTHER, TWIN]) is False
        assert startup.begin_loop_launch([OTHER]) is True

    def test_two_launches_of_one_twin_release_it_only_when_both_end(self):
        startup.begin_loop_launch([TWIN])
        startup.begin_loop_launch([TWIN])

        startup.end_loop_launch([TWIN])
        assert startup.begin_driver_activation(TWIN) is False

        startup.end_loop_launch([TWIN])
        assert startup.begin_driver_activation(TWIN) is True

    def test_ending_a_launch_that_was_not_claimed_is_harmless(self):
        startup.end_loop_launch([TWIN, ""])


class TestWaiting:
    def test_nothing_to_wait_for_returns_at_once(self):
        assert startup.wait_for_driver_activations(0.0) is True

    def test_it_times_out_while_an_activation_runs(self):
        startup.begin_driver_activation(TWIN)

        assert startup.wait_for_driver_activations(0.05) is False

    def test_it_returns_as_soon_as_the_activation_ends(self):
        startup.begin_driver_activation(TWIN)
        threading.Timer(0.05, startup.end_driver_activation, args=(TWIN,)).start()

        assert startup.wait_for_driver_activations(5.0) is True

    def test_an_edge_core_restart_waits_for_it_before_doing_anything(self, monkeypatch):
        order: list[str] = []
        monkeypatch.setattr(startup, "_EDGE_RESTART_IN_PROGRESS", False)
        monkeypatch.setattr(
            startup, "wait_for_driver_activations", lambda timeout: order.append("wait") or True
        )
        monkeypatch.setattr(startup, "load_token", lambda: order.append("token") or None)

        startup._run_edge_core_restart_worker("req-1", None)

        assert order == ["wait", "token"]

    def test_a_camera_drift_restart_waits_for_it_too(self, monkeypatch):
        order: list[str] = []
        monkeypatch.setattr(startup, "_EDGE_RESTART_IN_PROGRESS", False)
        monkeypatch.setattr(
            startup, "wait_for_driver_activations", lambda timeout: order.append("wait") or True
        )
        monkeypatch.setattr(
            startup, "_perform_edge_core_restart", lambda token: order.append("restart")
        )

        startup._run_camera_config_drift_restart_worker("tok")

        assert order == ["wait", "restart"]

    def test_a_skipped_camera_drift_restart_does_not_leave_activations_refused(self, monkeypatch):
        """The flag it set must not outlive a restart it declined to run."""
        monkeypatch.setattr(startup, "_EDGE_RESTART_IN_PROGRESS", True)
        monkeypatch.setattr(startup, "_CAMERA_DRIFT_RESTART_IN_PROGRESS", True)
        monkeypatch.setattr(startup, "_edge_core_restart_in_progress", lambda: False)

        startup._run_camera_config_drift_restart_worker("tok")

        assert startup._CAMERA_DRIFT_RESTART_IN_PROGRESS is False
        assert startup.begin_driver_activation(TWIN) is True

    def test_a_restart_goes_ahead_if_the_activation_never_ends(self, monkeypatch, caplog):
        order: list[str] = []
        monkeypatch.setattr(startup, "_EDGE_RESTART_IN_PROGRESS", False)
        monkeypatch.setattr(startup, "wait_for_driver_activations", lambda timeout: False)
        monkeypatch.setattr(startup, "load_token", lambda: order.append("token") or None)

        startup._run_edge_core_restart_worker("req-1", None)

        assert order == ["token"]
        assert "activation still running" in caplog.text


class TestRevivalLeavesAnActivatingTwinAlone:
    CONTAINER = driver_container_name(TWIN, "driver")

    @pytest.fixture(autouse=True)
    def _state(self, monkeypatch):
        monkeypatch.setattr(startup, "_DRIVER_HEALTH_PREVIOUS", {self.CONTAINER: "down"})
        monkeypatch.setattr(startup, "_DRIVER_HEALTH_STALE", False)
        monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {self.CONTAINER: TWIN})
        monkeypatch.setattr(startup, "_CONTAINER_RESTART_POLICY", {})
        monkeypatch.setattr(startup, "_LAST_REVIVAL_ATTEMPT_AT", None)
        monkeypatch.setattr(startup, "_REMOVED_SINCE", {})
        monkeypatch.setattr(
            "cyberwave_edge_core.docker_helpers.docker_inspect", lambda _name: None
        )
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_saved_fingerprint", lambda: "fp")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda **_: "env")

    def test_a_down_driver_is_revived_as_usual(self, monkeypatch):
        fetch = MagicMock(return_value=[])
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)

        startup.reconcile_driver_revival()

        fetch.assert_called_once()

    def test_not_while_its_twin_is_being_activated(self, monkeypatch):
        fetch = MagicMock(return_value=[])
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)
        startup.begin_driver_activation(TWIN)

        summary = startup.reconcile_driver_revival()

        fetch.assert_not_called()
        assert summary["revived_attempted"] == 0

    def test_an_activation_cannot_start_while_revival_launches_the_twin(self, monkeypatch):
        seen = {}

        def fetch(*args, **kwargs):
            seen["activation_allowed"] = startup.begin_driver_activation(TWIN)
            return []

        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)

        startup.reconcile_driver_revival()

        assert seen == {"activation_allowed": False}
        assert startup.begin_driver_activation(TWIN) is True  # released afterwards

    def test_a_failing_revival_still_releases_the_twin(self, monkeypatch):
        monkeypatch.setattr(
            startup, "fetch_and_run_twin_drivers", MagicMock(side_effect=RuntimeError("boom"))
        )

        startup.reconcile_driver_revival()

        assert startup.begin_driver_activation(TWIN) is True

    def test_revival_skips_a_twin_that_is_being_activated_even_if_its_map_entry_vanished(
        self, monkeypatch
    ):
        startup._CONTAINER_TWIN_MAP.clear()
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", MagicMock(return_value=[]))

        startup.reconcile_driver_revival()  # an orphan is skipped, and nothing raises

    def test_another_twins_down_driver_is_still_revived(self, monkeypatch):
        other = driver_container_name(OTHER, "driver")
        startup._DRIVER_HEALTH_PREVIOUS[other] = "down"
        startup._CONTAINER_TWIN_MAP[other] = OTHER
        fetch = MagicMock(return_value=[])
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)
        startup.begin_driver_activation(TWIN)

        startup.reconcile_driver_revival()

        assert fetch.call_args.kwargs["only_container_names"] == {other}


class _FakeAlertContext:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def create(self) -> None:
        pass

    def update_metadata(self, *args: Any, **kwargs: Any) -> None:
        pass

    def resolve(self) -> None:
        pass

    def mark_failed_and_resolve(self, *args: Any, **kwargs: Any) -> None:
        pass


def _twin(uuid: str, name: str, services: list[str], image: str = "humble") -> SimpleNamespace:
    return SimpleNamespace(
        uuid=uuid,
        name=name,
        asset_uuid=f"asset-{name}",
        asset_id=f"asset-{name}",
        metadata={
            "edge_fingerprint": "edge-fp",
            "drivers": {
                "default": {
                    "services": [
                        {"name": svc, "image": f"cyberwaveos/{name}-{svc}:{image}"}
                        for svc in services
                    ]
                }
            },
        },
    )


class TestRecreatingOneWholeTwin:
    @pytest.fixture
    def edge(self, monkeypatch: Any) -> SimpleNamespace:
        twins = [_twin(TWIN, "go2", ["driver", "bridges"]), _twin(OTHER, "arm", ["driver"])]
        seen = SimpleNamespace(launched=[], images=[])
        client = SimpleNamespace(
            twins=SimpleNamespace(list=lambda environment_id: twins, get_raw=lambda uuid: {}),
            assets=SimpleNamespace(get=lambda asset_uuid: SimpleNamespace(metadata={})),
        )

        def run(image: str, params: list[str], **kwargs: Any) -> bool:
            seen.launched.append(driver_container_name(kwargs["twin_uuid"], kwargs["service_name"]))
            seen.images.append(image)
            return True

        monkeypatch.setattr(startup, "Cyberwave", lambda base_url, api_key: client)
        monkeypatch.setattr(startup, "DriverStartingAlertContext", _FakeAlertContext)
        monkeypatch.setattr(startup, "_maybe_rewrite_jetson_tag", lambda image, _="": image)
        monkeypatch.setattr(startup, "_check_and_alert_sensors_devices", lambda *a, **k: None)
        monkeypatch.setattr(startup, "write_or_update_twin_json_file", lambda *a, **k: True)
        monkeypatch.setattr(startup, "_clear_stale_driver_starting_alerts", lambda *a, **k: 0)
        monkeypatch.setattr(
            startup, "_pull_driver_images_parallel", lambda images, **k: {i: True for i in images}
        )
        monkeypatch.setattr(startup, "_run_docker_image", run)
        monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda _image: True)
        return seen

    def test_only_the_named_twin_is_started(self, edge):
        startup.fetch_and_run_twin_drivers("tok", "env", "edge-fp", only_twin_uuids={TWIN})

        assert sorted(edge.launched) == sorted(
            [driver_container_name(TWIN, "driver"), driver_container_name(TWIN, "bridges")]
        )

    def test_every_service_the_twin_now_declares_is_started_even_a_new_one(self, edge):
        """`only_container_names` could not start a service that was not running before."""
        startup.fetch_and_run_twin_drivers("tok", "env", "edge-fp", only_twin_uuids={TWIN})

        assert driver_container_name(TWIN, "bridges") in edge.launched

    def test_without_the_option_every_twin_is_started(self, edge):
        startup.fetch_and_run_twin_drivers("tok", "env", "edge-fp")

        assert driver_container_name(OTHER, "driver") in edge.launched

    def test_a_twin_that_is_not_linked_is_simply_not_started(self, edge):
        results = startup.fetch_and_run_twin_drivers(
            "tok", "env", "edge-fp", only_twin_uuids={"ffffffff-0000-0000-0000-000000000000"}
        )

        assert results == [] and edge.launched == []

    def test_a_drivers_override_replaces_the_twins_metadata_for_that_run(self, edge):
        previous = {"default": {"services": [{"name": "driver", "image": "cyberwaveos/go2:old"}]}}

        startup.fetch_and_run_twin_drivers(
            "tok",
            "env",
            "edge-fp",
            only_twin_uuids={TWIN},
            drivers_override={TWIN: previous},
        )

        assert edge.images == ["cyberwaveos/go2:old"]
        assert edge.launched == [driver_container_name(TWIN, "driver")]

    def test_an_override_for_one_twin_leaves_the_others_on_their_own_metadata(self, edge):
        startup.fetch_and_run_twin_drivers(
            "tok",
            "env",
            "edge-fp",
            drivers_override={TWIN: {"default": {"services": [{"name": "d", "image": "x:1"}]}}},
        )

        assert "cyberwaveos/arm-driver:humble" in edge.images
        assert "x:1" in edge.images


class TestPinnedTags:
    @pytest.mark.parametrize(
        "image",
        [
            "cyberwaveos/go2:humble-0.2.0",
            "cyberwaveos/go2:0.3.1-realsense",
            "cyberwaveos/go2:humble-dev-sha-a1b2c3d4e5f6",
            "cyberwaveos/go2:sha-a1b2c3d",
            "cyberwaveos/go2@sha256:" + "a" * 64,
        ],
    )
    def test_a_tag_that_names_one_build_is_pinned(self, image):
        assert startup._image_is_pinned(image) is True

    @pytest.mark.parametrize(
        "image",
        [
            "cyberwaveos/go2",
            "cyberwaveos/go2:latest",
            "cyberwaveos/go2:humble",
            "cyberwaveos/go2:dev-0.2.0",
            "cyberwaveos/go2:dev-sha-a1b2c3d",
            "cyberwaveos/go2:staging",
            "cyberwaveos/go2:jetson-humble",
        ],
    )
    def test_a_tag_that_can_move_is_not(self, image):
        assert startup._image_is_pinned(image) is False

    @pytest.fixture
    def pulls(self, monkeypatch):
        pulled: list[str] = []
        monkeypatch.setattr(startup, "_use_local_build", lambda image: False)
        monkeypatch.setattr(startup, "shutdown_event", SimpleNamespace(is_set=lambda: False))
        monkeypatch.setattr(
            startup,
            "_pull_docker_image_with_progress_multi",
            lambda image, **kwargs: pulled.append(image),
        )
        return pulled

    def test_a_pinned_image_that_is_already_here_is_not_pulled(self, pulls, monkeypatch):
        monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda _image: True)

        results = startup._pull_driver_images_parallel(["cyberwaveos/go2:humble-0.2.0"])

        assert results == {"cyberwaveos/go2:humble-0.2.0": True}
        assert pulls == []

    def test_a_pinned_image_that_is_not_here_is_pulled(self, pulls, monkeypatch):
        monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda _image: False)

        startup._pull_driver_images_parallel(["cyberwaveos/go2:humble-0.2.0"])

        assert pulls == ["cyberwaveos/go2:humble-0.2.0"]

    def test_a_moving_tag_is_pulled_even_when_it_is_here(self, pulls, monkeypatch):
        monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda _image: True)

        startup._pull_driver_images_parallel(["cyberwaveos/go2:humble"])

        assert pulls == ["cyberwaveos/go2:humble"]

    def test_only_the_pinned_ones_are_skipped_from_a_mixed_list(self, pulls, monkeypatch):
        monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda _image: True)

        results = startup._pull_driver_images_parallel(
            ["cyberwaveos/go2:humble-0.2.0", "cyberwaveos/nav2:humble"]
        )

        assert pulls == ["cyberwaveos/nav2:humble"]
        assert results == {
            "cyberwaveos/go2:humble-0.2.0": True,
            "cyberwaveos/nav2:humble": True,
        }


class TestTheCapability:
    def _discover(self, monkeypatch) -> dict[str, Any]:
        sent: dict[str, Any] = {}
        host_metrics = types.ModuleType("cyberwave.edge.host_metrics")
        host_metrics.read_host_facts = lambda: SimpleNamespace(to_dict=dict)
        monkeypatch.setitem(sys.modules, "cyberwave.edge.host_metrics", host_metrics)
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp-test")

        def discover(**kwargs: Any) -> dict[str, Any]:
            sent.update(kwargs)
            return {"twins": []}

        client = SimpleNamespace(edges=SimpleNamespace(discover=discover))
        monkeypatch.setattr(startup, "Cyberwave", lambda *a, **k: client)
        assert startup._post_host_facts_once("tok") is True
        return sent["host_facts"]

    def test_every_edge_advertises_it(self, monkeypatch):
        assert self._discover(monkeypatch)["capabilities"] == [driver_updates.CAPABILITY]

    def test_the_backend_reads_it_as_the_staging_capability(self):
        """The string the backend checks (`deployments.CAPABILITY`)."""
        assert driver_updates.CAPABILITY == "driver_staging_v1"
