"""The twin driver resolver, on its own.

`resolve_twin_driver_specs` is what `fetch_and_run_twin_drivers` used to do inline for
each twin: decide which containers the twin runs. These tests call it directly, with no
Docker and no backend, which is what lets a caller resolve a release that is not the one
running (staging reads the graph of an update, then pulls its images).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from cyberwave_edge_core import driver_selection as ds
from cyberwave_edge_core import startup

TWIN_UUID = "abcd1234-0000-0000-0000-000000000000"
TWIN = SimpleNamespace(name="go2-1")


@pytest.fixture(autouse=True)
def _no_host_effects(monkeypatch: Any) -> None:
    monkeypatch.setattr(startup, "_resolve_driver_image_tag", lambda image: image)
    monkeypatch.setattr(startup, "_maybe_rewrite_jetson_tag", lambda image, _name="": image)
    monkeypatch.setattr(startup, "load_driver_overrides", lambda: {})
    monkeypatch.setattr(startup, "DriverStartingAlertContext", MagicMock())
    monkeypatch.setattr(startup, "_clear_stale_driver_starting_alerts", lambda *a, **kw: 0)


def _resolve(drivers: dict[str, Any] | None = None, **kwargs: Any) -> list[Any]:
    return startup.resolve_twin_driver_specs(
        TWIN,
        TWIN_UUID,
        drivers if drivers is not None else {"default": {"docker_image": "drv:1.0.0"}},
        child_registry_ids=set(),
        child_camera_twin_uuids=["cam-1"],
        macos_bridge_candidates=["bridge-1"],
        token=None,
        **kwargs,
    )


class TestSingleContainer:
    def test_a_plain_docker_image_is_one_spec_carrying_the_callers_context(self):
        [spec] = _resolve()

        assert spec.twin is TWIN
        assert spec.twin_uuid == TWIN_UUID
        assert spec.driver_image == "drv:1.0.0"
        assert spec.service_name is None
        assert spec.child_camera_twin_uuids == ["cam-1"]
        assert spec.macos_bridge_candidates == ["bridge-1"]

    def test_a_local_override_replaces_the_image_and_drops_its_params(self, monkeypatch):
        monkeypatch.setattr(startup, "load_driver_overrides", lambda: {TWIN_UUID: "mine:dev"})
        drivers = {"default": {"docker_image": "drv:1.0.0", "params": ["--fast"]}}

        [spec] = _resolve(drivers)

        assert spec.driver_image == "mine:dev"
        assert spec.driver_params == []

    def test_no_image_is_an_error_the_caller_alerts_on(self, monkeypatch):
        monkeypatch.setattr(startup, "_get_driver_services", lambda *_a, **_k: None)
        monkeypatch.setattr(startup, "_image_declared_services_for_twin", lambda *_a, **_k: None)
        monkeypatch.setattr(
            startup, "_get_best_driver_image_and_params", lambda *_a, **_k: ("", [], False, "all")
        )

        with pytest.raises(startup.DriverResolutionError) as raised:
            _resolve()

        assert raised.value.kind == startup.DriverResolutionError.NO_DRIVER_IMAGE

    def test_a_driver_block_without_an_image_is_an_unusable_topology(self):
        with pytest.raises(startup.DriverResolutionError) as raised:
            _resolve({"default": {}})

        assert raised.value.kind == startup.DriverResolutionError.UNUSABLE_TOPOLOGY


class TestMultiContainer:
    def test_each_declared_service_becomes_a_spec_with_its_own_fields(self, monkeypatch):
        plant = ds._ServiceSpec(
            image="drv:plant",
            name="plant",
            env={"A": "service"},
            params=["--p"],
            restart="always",
            stop_timeout_s=9,
        )
        bridge = ds._ServiceSpec(
            image="drv:bridge", name="bridge", depends_on={"plant": "service_healthy"}
        )
        monkeypatch.setattr(
            startup,
            "_get_driver_services",
            lambda *_a, **_k: ([plant, bridge], {"A": "shared", "B": "shared"}, ["--shared"]),
        )

        specs = _resolve()

        assert [spec.service_name for spec in specs] == ["plant", "bridge"]
        assert specs[0].service_env == {"A": "service", "B": "shared"}
        assert specs[0].driver_params == ["--shared", "--p"]
        assert specs[0].restart == "always"
        assert specs[0].stop_timeout_s == 9
        assert specs[1].depends_on == {"plant": "service_healthy"}
        assert all(spec.child_camera_twin_uuids == ["cam-1"] for spec in specs)

    def test_metadata_services_win_over_the_image_declared_graph(self, monkeypatch):
        declared = MagicMock()
        monkeypatch.setattr(
            startup,
            "_get_driver_services",
            lambda *_a, **_k: ([ds._ServiceSpec(image="drv:meta", name="meta")], {}, []),
        )
        monkeypatch.setattr(startup, "_image_declared_services_for_twin", declared)

        [spec] = _resolve()

        assert spec.driver_image == "drv:meta"
        declared.assert_not_called()

    @pytest.mark.parametrize("error", [ValueError("bad services"), RuntimeError("bad graph")])
    def test_an_unusable_topology_is_an_error_the_caller_alerts_on(self, monkeypatch, error):
        def _boom(*_a, **_k):
            raise error

        monkeypatch.setattr(startup, "_get_driver_services", _boom)

        with pytest.raises(startup.DriverResolutionError) as raised:
            _resolve()

        assert raised.value.kind == startup.DriverResolutionError.UNUSABLE_TOPOLOGY
        assert str(error) in str(raised.value)


class TestAlerts:
    @pytest.mark.parametrize(
        ("kind", "title"),
        [
            (startup.DriverResolutionError.UNUSABLE_TOPOLOGY, "Unusable declared topology"),
            (
                startup.DriverResolutionError.NO_DRIVER_IMAGE,
                "No driver_docker_image in asset metadata",
            ),
        ],
    )
    def test_each_kind_raises_its_own_driver_start_failure_alert(self, monkeypatch, kind, title):
        sent = MagicMock()
        monkeypatch.setattr(startup, "_send_alert_for_twin", sent)

        startup._alert_unresolved_driver(
            TWIN, TWIN_UUID, startup.DriverResolutionError(kind, "why")
        )

        args, kwargs = sent.call_args
        assert args[0] == TWIN_UUID
        assert args[1] == title
        assert args[3] == "driver_start_failure"
        assert kwargs["severity"] == "error"

    def test_a_failing_alert_does_not_stop_the_other_twins(self, monkeypatch):
        monkeypatch.setattr(
            startup, "_send_alert_for_twin", MagicMock(side_effect=RuntimeError("backend down"))
        )

        startup._alert_unresolved_driver(
            TWIN,
            TWIN_UUID,
            startup.DriverResolutionError(startup.DriverResolutionError.NO_DRIVER_IMAGE),
        )


class TestTopologyRead:
    """`_image_declared_services_for_twin`: which image the graph is read from."""

    @pytest.fixture
    def graph(self, monkeypatch: Any) -> dict[str, Any]:
        seen: dict[str, Any] = {"pulls": [], "read": []}
        specs = [ds._ServiceSpec(image="drv:plant", name="plant")]

        def _pull(images: list[str], **_kw: Any) -> dict[str, bool]:
            seen["pulls"].append(tuple(images))
            return {image: seen["pullable"](image) for image in images}

        def _read(image: str, **kwargs: Any) -> Any:
            seen["read"].append((image, kwargs))
            return specs, {}, []

        seen["pullable"] = lambda image: True
        monkeypatch.setattr(startup, "_pull_driver_images_parallel", _pull)
        monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda _i: False)
        monkeypatch.setattr(startup, "get_image_declared_services", _read)
        return seen

    def _declared(self, **kwargs: Any) -> Any:
        return startup._image_declared_services_for_twin(
            {"default": {"docker_image": "drv:humble-0.1.0"}},
            TWIN,
            TWIN_UUID,
            child_registry_ids=set(),
            token=None,
            **kwargs,
        )

    def test_off_jetson_the_one_image_is_pulled_and_read(self, graph):
        assert self._declared() is not None

        assert graph["pulls"] == [("drv:humble-0.1.0",)]
        assert graph["read"][0][0] == "drv:humble-0.1.0"

    def test_a_missing_jetson_tag_falls_back_to_the_original_tag_for_the_graph(
        self, monkeypatch, graph
    ):
        """Not every driver publishes `jetson-` tags (Kinova, MiR250). The twin used to fall
        back to a single container and lose every sibling the graph declares."""
        monkeypatch.setattr(
            startup, "_maybe_rewrite_jetson_tag", lambda image, _n="": image.replace(":", ":jetson-")
        )
        graph["pullable"] = lambda image: "jetson-" not in image

        assert self._declared() is not None

        assert graph["pulls"] == [("drv:jetson-humble-0.1.0",), ("drv:humble-0.1.0",)]
        assert graph["read"][0][0] == "drv:humble-0.1.0"

    def test_a_present_jetson_tag_is_used_and_the_original_never_pulled(self, monkeypatch, graph):
        monkeypatch.setattr(
            startup, "_maybe_rewrite_jetson_tag", lambda image, _n="": image.replace(":", ":jetson-")
        )

        assert self._declared() is not None

        assert graph["pulls"] == [("drv:jetson-humble-0.1.0",)]
        assert graph["read"][0][0] == "drv:jetson-humble-0.1.0"

    def test_when_no_tag_can_be_pulled_there_is_no_graph(self, monkeypatch, graph):
        monkeypatch.setattr(
            startup, "_maybe_rewrite_jetson_tag", lambda image, _n="": image.replace(":", ":jetson-")
        )
        graph["pullable"] = lambda image: False

        assert self._declared() is None

        assert graph["read"] == []
