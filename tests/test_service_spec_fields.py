"""Tests for the optional inline ``services[]`` lifecycle fields and the restart floor.

Covers:
  1. ``_get_driver_services`` parses and validates the lifecycle fields
  2. A twin that sets none of them reaches ``_run_docker_image`` unchanged,
     and gets the byte-identical ``docker run`` every driver always got
  3. The fields that change the argv (restart, stop timeout, privileged,
     restart-floor label, host env file)
  4. ``_restart_floor_decision`` and the launcher's refuse/adopt branches
  5. Revival skips ``restart: "no"`` and restart-floor containers, but brings
     back a ``restart: "no"`` one whose relaunch the floor refused
  6. Restart teardown stops restart-floor containers gracefully, and the
     periodic prune keeps them

No docker: subprocess and the docker helpers are faked throughout.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import cyberwave_edge_core.docker_helpers as docker_helpers
import cyberwave_edge_core.driver_selection as driver_selection
import cyberwave_edge_core.startup as startup
from cyberwave_edge_core.driver_launcher import _restart_floor_decision
from tests.driver_subprocess_fakes import find_driver_create_cmd
from tests.test_multi_service_driver import _patch_driver_container_launch

TWIN_UUID = "aabbccdd-1234-5678-9012-abcdef012345"
IMAGE = "cyberwaveos/kinova-gen3-driver:humble"
FLOOR_LABEL = docker_helpers.MIN_RESTART_INTERVAL_LABEL
DRIVER_CONTAINER = docker_helpers.driver_container_name(TWIN_UUID, "driver")


def _services(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"default": {"services": list(entries)}}


def _one_service(**fields: Any) -> dict[str, Any]:
    return _services({"name": "driver", "image": IMAGE, **fields})


def _fixed_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
    monkeypatch.setattr(driver_selection.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(driver_selection, "is_jetson", lambda: False)


def _iso(ns: int) -> str:
    """A ``docker inspect`` style UTC timestamp for *ns* since the epoch."""
    import datetime as _dt

    whole = _dt.datetime.fromtimestamp(ns // 1_000_000_000, tz=_dt.timezone.utc)
    return whole.strftime("%Y-%m-%dT%H:%M:%S") + f".{ns % 1_000_000_000:09d}Z"


def _run(**kwargs: Any) -> bool:
    return startup._run_docker_image(
        IMAGE,
        [],
        twin_uuid=TWIN_UUID,
        token="test-token",
        skip_pull=True,
        service_name="driver",
        **kwargs,
    )


def _created(captured: list[list[str]]) -> bool:
    return any(c[:2] == ["docker", "create"] for c in captured)


# ===========================================================================
# 1. Parsing and validation
# ===========================================================================


class TestServiceFieldParsing:
    def test_defaults_are_the_old_launch_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fixed_platform(monkeypatch)
        specs, _, _ = startup._get_driver_services(_one_service())
        spec = specs[0]
        assert (spec.restart, spec.stop_timeout_s, spec.privileged) == ("unless-stopped", 5, True)
        assert spec.min_restart_interval_s is None
        assert spec.host_env_file is None
        assert driver_selection._service_lifecycle_overrides(spec) == {}

    def test_all_fields_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fixed_platform(monkeypatch)
        specs, _, _ = startup._get_driver_services(
            _one_service(
                restart="no",
                stop_timeout_s=60,
                privileged=False,
                min_restart_interval_s=75,
                host_env_file=" kinova/robot.env ",
            )
        )
        assert driver_selection._service_lifecycle_overrides(specs[0]) == {
            "restart": "no",
            "stop_timeout_s": 60,
            "privileged": False,
            "min_restart_interval_s": 75,
            "host_env_file": "kinova/robot.env",
        }

    def test_explicit_nulls_mean_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fixed_platform(monkeypatch)
        specs, _, _ = startup._get_driver_services(
            _one_service(restart=None, min_restart_interval_s=None, host_env_file=None)
        )
        assert driver_selection._service_lifecycle_overrides(specs[0]) == {}

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"restart": "always"}, "restart"),
            ({"restart": 0}, "restart"),
            ({"stop_timeout_s": True}, "stop_timeout_s"),
            ({"stop_timeout_s": -1}, "stop_timeout_s"),
            ({"stop_timeout_s": "30"}, "stop_timeout_s"),
            ({"min_restart_interval_s": "75"}, "min_restart_interval_s"),
            ({"min_restart_interval_s": 7.5}, "min_restart_interval_s"),
            ({"privileged": "false"}, "privileged"),
            ({"host_env_file": "/etc/shadow"}, "host_env_file"),
            ({"host_env_file": "../credentials.json"}, "host_env_file"),
            ({"host_env_file": "kinova/../../x.env"}, "host_env_file"),
            ({"host_env_file": " "}, "host_env_file"),
        ],
    )
    def test_invalid_values_raise(
        self, monkeypatch: pytest.MonkeyPatch, fields: dict[str, Any], message: str
    ) -> None:
        _fixed_platform(monkeypatch)
        with pytest.raises(ValueError, match=message):
            startup._get_driver_services(_one_service(**fields))


# ===========================================================================
# 2./3. The docker run argv
# ===========================================================================


class TestDriverRunArgv:
    def test_default_argv_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured)

        assert _run() is True
        cmd = find_driver_create_cmd(captured)
        # The pre-existing flag block, in its old order.
        assert cmd[:8] == [
            "docker",
            "create",
            "--init",
            "--stop-timeout",
            "5",
            "--restart",
            "unless-stopped",
            "--privileged",
        ]
        assert not any(arg.startswith(f"{FLOOR_LABEL}=") for arg in cmd)
        assert "--env-file" not in cmd

    def test_explicit_defaults_give_the_same_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        implicit: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, implicit)
        _run()
        explicit: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, explicit)
        _run(restart="unless-stopped", stop_timeout_s=5, privileged=True)
        assert find_driver_create_cmd(explicit) == find_driver_create_cmd(implicit)

    def test_lifecycle_fields_reach_the_argv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        (tmp_path / "kinova").mkdir()
        (tmp_path / "kinova" / "robot.env").write_text("KINOVA_IP=192.168.1.10\n")
        # No container before the create (no floor to honour), running after it.
        monkeypatch.setattr(
            docker_helpers,
            "docker_inspect",
            lambda _n: {"State": {"Status": "running"}} if _created(captured) else None,
        )

        assert (
            _run(
                restart="no",
                stop_timeout_s=60,
                privileged=False,
                min_restart_interval_s=75,
                host_env_file="kinova/robot.env",
            )
            is True
        )
        cmd = find_driver_create_cmd(captured)
        assert cmd[:7] == ["docker", "create", "--init", "--stop-timeout", "60", "--restart", "no"]
        assert "--privileged" not in cmd
        assert f"{FLOOR_LABEL}=75" in cmd
        assert cmd[cmd.index(f"{FLOOR_LABEL}=75") - 1] == "--label"
        assert cmd[cmd.index("--env-file") + 1] == str(tmp_path / "kinova" / "robot.env")
        # The robot credentials go through the file, never through `-e`.
        assert "KINOVA_IP=192.168.1.10" not in cmd

    def test_remove_before_create_outlasts_the_stop_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured)
        seen: dict[str, Any] = {}

        def _remove(name: str, **kwargs: Any) -> bool:
            seen.update(kwargs)
            return True

        monkeypatch.setattr("cyberwave_edge_core.docker_launch.remove_existing_container", _remove)
        _run(stop_timeout_s=60)
        assert seen["min_timeout"] == 75

    def test_missing_host_env_file_refuses(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        captured: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

        with caplog.at_level("ERROR"):
            assert _run(host_env_file="kinova/robot.env") is False
        assert "host_env_file_missing" in caplog.text
        assert str(tmp_path / "kinova" / "robot.env") in caplog.text
        assert not _created(captured)


# ===========================================================================
# 4. Restart floor
# ===========================================================================

NOW_NS = 1_790_000_000 * 1_000_000_000


def _inspect(status: str, *, finished_ns: int | None = None) -> dict[str, Any]:
    finished = _iso(finished_ns) if finished_ns is not None else "0001-01-01T00:00:00Z"
    return {"State": {"Status": status, "FinishedAt": finished}}


class TestRestartFloorDecision:
    def test_no_container_proceeds(self) -> None:
        assert _restart_floor_decision(None, min_restart_interval_s=75, now_ns=NOW_NS) == (
            "proceed",
            {},
        )

    @pytest.mark.parametrize("status", ["running", "restarting", "paused"])
    def test_live_container_is_adopted(self, status: str) -> None:
        assert _restart_floor_decision(
            _inspect(status), min_restart_interval_s=75, now_ns=NOW_NS
        ) == ("adopt", {})

    def test_recently_exited_is_too_soon(self) -> None:
        decision, detail = _restart_floor_decision(
            _inspect("exited", finished_ns=NOW_NS - 10_500_000_000),
            min_restart_interval_s=75,
            now_ns=NOW_NS,
        )
        assert decision == "refuse"
        assert detail["reason"] == "restart_too_soon"
        # ceil(75 - 10.5)
        assert detail["retry_after_s"] == 65

    def test_exited_long_enough_ago_proceeds(self) -> None:
        assert _restart_floor_decision(
            _inspect("exited", finished_ns=NOW_NS - 75_000_000_000),
            min_restart_interval_s=75,
            now_ns=NOW_NS,
        ) == ("proceed", {})

    def test_never_ran_proceeds(self) -> None:
        assert _restart_floor_decision(
            _inspect("created"), min_restart_interval_s=75, now_ns=NOW_NS
        ) == ("proceed", {})

    def test_future_finished_at_counts_as_just_now(self) -> None:
        decision, detail = _restart_floor_decision(
            _inspect("exited", finished_ns=NOW_NS + 30_000_000_000),
            min_restart_interval_s=75,
            now_ns=NOW_NS,
        )
        assert decision == "refuse"
        assert detail["retry_after_s"] == 75


class TestLauncherRestartFloor:
    def test_too_soon_refuses_and_keeps_the_twin_mapping(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        captured: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured)
        finished = time.time_ns() - 5_000_000_000
        monkeypatch.setattr(
            docker_helpers, "docker_inspect", lambda _n: _inspect("exited", finished_ns=finished)
        )
        startup._CONTAINER_TWIN_MAP.pop(DRIVER_CONTAINER, None)

        try:
            with caplog.at_level("ERROR"):
                assert _run(min_restart_interval_s=75) is False
            assert "restart_too_soon" in caplog.text
            # Revival skips an unmapped container as an orphan.
            assert startup._CONTAINER_TWIN_MAP.get(DRIVER_CONTAINER) == TWIN_UUID
            assert captured == []
        finally:
            startup._CONTAINER_TWIN_MAP.pop(DRIVER_CONTAINER, None)

    def test_live_container_is_adopted_not_recreated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured)
        monkeypatch.setattr(docker_helpers, "docker_inspect", lambda _n: _inspect("running"))

        def _no_remove(*a: Any, **kw: Any) -> bool:
            raise AssertionError("a live floor container must never be removed")

        monkeypatch.setattr(
            "cyberwave_edge_core.docker_launch.remove_existing_container", _no_remove
        )

        try:
            assert _run(min_restart_interval_s=75) is True
            assert startup._CONTAINER_TWIN_MAP.get(DRIVER_CONTAINER) == TWIN_UUID
            assert captured == []
        finally:
            startup._CONTAINER_TWIN_MAP.pop(DRIVER_CONTAINER, None)

    @pytest.mark.parametrize(("running_image", "warned"), [(IMAGE, False), ("old:1", True)])
    def test_adopting_another_image_is_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        running_image: str,
        warned: bool,
    ) -> None:
        _patch_driver_container_launch(monkeypatch, [])
        live = {**_inspect("running"), "Config": {"Image": running_image}}
        monkeypatch.setattr(docker_helpers, "docker_inspect", lambda _n: live)

        try:
            with caplog.at_level("WARNING"):
                assert _run(min_restart_interval_s=75) is True
            assert (f"runs old:1, not the requested {IMAGE}" in caplog.text) is warned
        finally:
            startup._CONTAINER_TWIN_MAP.pop(DRIVER_CONTAINER, None)


class TestLaunchLoopRestartFloor:
    """A floor refusal in the launch loop is not reported as a hardware failure."""

    @pytest.fixture(autouse=True)
    def _loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fixed_platform(monkeypatch)
        twin = SimpleNamespace(
            uuid=TWIN_UUID,
            name="Gen3",
            asset_uuid="asset-gen3",
            asset_id="asset-gen3",
            metadata={
                "edge_fingerprint": "edge-fp",
                "drivers": _one_service(min_restart_interval_s=75),
            },
        )
        client = SimpleNamespace(
            twins=SimpleNamespace(list=lambda environment_id: [twin], get_raw=lambda _u: {}),
            assets=SimpleNamespace(get=lambda _a: SimpleNamespace(metadata={})),
        )

        class _Alert:
            def __init__(self, **_kw: Any) -> None:
                pass

            def __getattr__(self, _name: str) -> Any:
                return lambda *a, **kw: None

        self.alerts: list[str] = []
        monkeypatch.setattr(startup, "Cyberwave", lambda base_url, api_key: client)
        monkeypatch.setattr(startup, "DriverStartingAlertContext", _Alert)
        monkeypatch.setattr(startup, "_maybe_rewrite_jetson_tag", lambda image, _: image)
        monkeypatch.setattr(startup, "_check_and_alert_sensors_devices", lambda *a, **kw: None)
        monkeypatch.setattr(startup, "write_or_update_twin_json_file", lambda *a, **kw: True)
        monkeypatch.setattr(startup, "_clear_stale_driver_starting_alerts", lambda *a, **kw: 0)
        monkeypatch.setattr(
            startup,
            "_pull_driver_images_parallel",
            lambda images, **kw: {img: True for img in set(images)},
        )
        monkeypatch.setattr(startup, "_run_docker_image", lambda *a, **kw: False)
        monkeypatch.setattr(
            startup, "_send_alert_for_twin", lambda _t, title, *a, **kw: self.alerts.append(title)
        )

    def _finished_ago(self, monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
        inspect = _inspect("exited", finished_ns=time.time_ns() - int(seconds * 1e9))
        monkeypatch.setattr(docker_helpers, "docker_inspect", lambda _n: inspect)

    def test_refused_by_the_floor_sends_no_failure_alert(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._finished_ago(monkeypatch, 5)
        results = startup.fetch_and_run_twin_drivers("test-token", "env-uuid", "edge-fp")
        assert [r["success"] for r in results] == [False]
        assert self.alerts == []

    def test_failure_after_the_floor_passed_still_alerts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._finished_ago(monkeypatch, 120)
        startup.fetch_and_run_twin_drivers("test-token", "env-uuid", "edge-fp")
        assert self.alerts == ["Driver failed to start"]


class TestLaunchLoopInvalidServices:
    """One twin's invalid ``services`` metadata does not stop the other twins."""

    def test_other_twins_still_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fixed_platform(monkeypatch)
        bad_uuid = "11111111-1234-5678-9012-abcdef012345"

        def _twin(uuid: str, drivers: dict[str, Any]) -> SimpleNamespace:
            return SimpleNamespace(
                uuid=uuid,
                name=uuid[:8],
                asset_uuid="asset-gen3",
                asset_id="asset-gen3",
                metadata={"edge_fingerprint": "edge-fp", "drivers": drivers},
            )

        twins = [_twin(bad_uuid, _one_service(restart="always")), _twin(TWIN_UUID, _one_service())]
        client = SimpleNamespace(
            twins=SimpleNamespace(list=lambda environment_id: twins, get_raw=lambda _u: {}),
            assets=SimpleNamespace(get=lambda _a: SimpleNamespace(metadata={})),
        )
        launched: list[str] = []
        alerts: list[tuple[str, str]] = []
        monkeypatch.setattr(startup, "Cyberwave", lambda base_url, api_key: client)
        monkeypatch.setattr(startup, "_maybe_rewrite_jetson_tag", lambda image, _: image)
        monkeypatch.setattr(startup, "_check_and_alert_sensors_devices", lambda *a, **kw: None)
        monkeypatch.setattr(startup, "write_or_update_twin_json_file", lambda *a, **kw: True)
        monkeypatch.setattr(startup, "_clear_stale_driver_starting_alerts", lambda *a, **kw: 0)
        monkeypatch.setattr(
            startup,
            "_pull_driver_images_parallel",
            lambda images, **kw: {img: True for img in set(images)},
        )
        monkeypatch.setattr(
            startup,
            "_run_docker_image",
            lambda *a, twin_uuid, **kw: launched.append(twin_uuid) or True,
        )
        monkeypatch.setattr(
            startup,
            "_send_alert_for_twin",
            lambda uuid, title, *a, **kw: alerts.append((uuid, title)),
        )

        startup.fetch_and_run_twin_drivers("test-token", "env-uuid", "edge-fp")

        assert launched == [TWIN_UUID]
        assert (bad_uuid, "Unusable declared topology") in alerts


# ===========================================================================
# 5. Revival
# ===========================================================================


def _exited(*, restart: str, floor_s: int | None = None, finished_ago_s: float = 3600.0):
    labels = {FLOOR_LABEL: str(floor_s)} if floor_s else {}
    return {
        "State": {
            "Status": "exited",
            "FinishedAt": _iso(time.time_ns() - int(finished_ago_s * 1e9)),
        },
        "HostConfig": {"RestartPolicy": {"Name": restart}},
        "Config": {"Labels": labels},
    }


class TestRevival:
    @pytest.fixture(autouse=True)
    def _state(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        startup._DRIVER_HEALTH_PREVIOUS.clear()
        startup._CONTAINER_TWIN_MAP.clear()
        startup._RESTART_FLOOR_DEFERRED.clear()
        startup._LAST_REVIVAL_ATTEMPT_AT = None
        self.fetch_calls = 0

        def _fetch(*_a: Any, **_kw: Any) -> list[Any]:
            self.fetch_calls += 1
            return []

        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", _fetch)
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_saved_fingerprint", lambda: "fp")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env")
        self.inspects: dict[str, Any] = {}
        monkeypatch.setattr(docker_helpers, "docker_inspect", lambda n: self.inspects.get(n))
        yield
        startup._DRIVER_HEALTH_PREVIOUS.clear()
        startup._CONTAINER_TWIN_MAP.clear()
        startup._RESTART_FLOOR_DEFERRED.clear()
        startup._LAST_REVIVAL_ATTEMPT_AT = None

    def _down(self, name: str, inspect: dict[str, Any]) -> None:
        startup._DRIVER_HEALTH_PREVIOUS[name] = "down"
        startup._CONTAINER_TWIN_MAP[name] = TWIN_UUID
        self.inspects[name] = inspect

    def test_exited_restart_no_service_stays_down(self) -> None:
        self._down(DRIVER_CONTAINER, _exited(restart="no", floor_s=75))
        summary = startup.reconcile_driver_revival()
        assert self.fetch_calls == 0
        assert summary["skipped_restart_no"] == 1
        # No attempt made, so no debounce armed against the next real one.
        assert startup._LAST_REVIVAL_ATTEMPT_AT is None

    def test_restart_no_container_refused_on_its_floor_comes_back_after_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An edge-core restart: teardown stopped the container just now, and
        # the relaunch right after it is refused on the floor.
        _patch_driver_container_launch(monkeypatch, [])
        monkeypatch.setattr(docker_helpers, "docker_inspect", lambda n: self.inspects.get(n))
        self.inspects[DRIVER_CONTAINER] = _exited(restart="no", floor_s=75, finished_ago_s=0)
        assert _run(min_restart_interval_s=75, restart="no") is False
        startup._DRIVER_HEALTH_PREVIOUS[DRIVER_CONTAINER] = "down"

        summary = startup.reconcile_driver_revival()
        assert self.fetch_calls == 0
        assert summary["skipped_restart_floor"] == 1

        self.inspects[DRIVER_CONTAINER] = _exited(restart="no", floor_s=75, finished_ago_s=120)
        summary = startup.reconcile_driver_revival()
        assert self.fetch_calls == 1
        assert summary["revived_attempted"] == 1

    def test_relaunch_past_the_floor_ends_the_restart_no_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_driver_container_launch(monkeypatch, [])
        monkeypatch.setattr(docker_helpers, "docker_inspect", lambda n: self.inspects.get(n))
        startup._RESTART_FLOOR_DEFERRED.add(DRIVER_CONTAINER)
        self.inspects[DRIVER_CONTAINER] = _exited(restart="no", floor_s=75, finished_ago_s=120)
        _run(min_restart_interval_s=75, restart="no")
        assert DRIVER_CONTAINER not in startup._RESTART_FLOOR_DEFERRED

        # Exiting on its own afterwards is what restart=no is about.
        self._down(DRIVER_CONTAINER, _exited(restart="no", floor_s=75, finished_ago_s=120))
        summary = startup.reconcile_driver_revival()
        assert self.fetch_calls == 0
        assert summary["skipped_restart_no"] == 1

    def test_floor_container_waits_out_its_floor(self) -> None:
        self._down(
            DRIVER_CONTAINER, _exited(restart="unless-stopped", floor_s=75, finished_ago_s=5)
        )
        summary = startup.reconcile_driver_revival()
        assert self.fetch_calls == 0
        assert summary["skipped_restart_floor"] == 1

    def test_floor_container_revived_once_the_floor_passed(self) -> None:
        self._down(
            DRIVER_CONTAINER, _exited(restart="unless-stopped", floor_s=75, finished_ago_s=120)
        )
        summary = startup.reconcile_driver_revival()
        assert self.fetch_calls == 1
        assert summary["revived_attempted"] == 1

    def test_uninspectable_container_is_still_revived(self) -> None:
        startup._DRIVER_HEALTH_PREVIOUS[DRIVER_CONTAINER] = "down"
        startup._CONTAINER_TWIN_MAP[DRIVER_CONTAINER] = TWIN_UUID
        summary = startup.reconcile_driver_revival()
        assert self.fetch_calls == 1
        assert summary["revived_attempted"] == 1


# ===========================================================================
# 6. Restart teardown and periodic prune
# ===========================================================================


class TestStopAndPrune:
    @pytest.fixture(autouse=True)
    def _docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.commands: list[tuple[list[str], dict[str, Any]]] = []
        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [DRIVER_CONTAINER, "cyberwave-driver-00000000"],
        )
        monkeypatch.setattr(startup.shutil, "which", lambda _n: "/usr/bin/docker")

        def _run_cmd(cmd: list[str], **kwargs: Any) -> Any:
            self.commands.append((cmd, kwargs))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _run_cmd)
        monkeypatch.setattr(
            docker_helpers, "docker_inspect", lambda n: {"Config": {"StopTimeout": 60}}
        )

    def test_floor_container_is_stopped_gracefully_and_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            docker_helpers, "docker_names_with_label", lambda _l: {DRIVER_CONTAINER}
        )
        assert startup._stop_and_prune_driver_containers() == (
            ["cyberwave-driver-00000000"],
            [DRIVER_CONTAINER],
        )
        by_target = {cmd[-1]: (cmd, kw) for cmd, kw in self.commands if cmd[1] in {"stop", "rm"}}
        stop_cmd, stop_kwargs = by_target[DRIVER_CONTAINER]
        # No -t: docker waits the container's own --stop-timeout.
        assert stop_cmd == ["docker", "stop", DRIVER_CONTAINER]
        assert stop_kwargs["timeout"] == 90
        assert by_target["cyberwave-driver-00000000"][0] == [
            "docker",
            "rm",
            "-f",
            "-v",
            "cyberwave-driver-00000000",
        ]
        prune = [cmd for cmd, _ in self.commands if cmd[1:3] == ["container", "prune"]]
        assert prune == [
            ["docker", "container", "prune", "--force", "--filter", f"label!={FLOOR_LABEL}"]
        ]

    def test_unknown_labels_stop_everything_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docker_helpers, "docker_names_with_label", lambda _l: None)
        startup._stop_and_prune_driver_containers()
        assert not any(cmd[:3] == ["docker", "rm", "-f"] for cmd, _ in self.commands)
        stops = [cmd[-1] for cmd, _ in self.commands if cmd[:2] == ["docker", "stop"]]
        assert sorted(stops) == sorted([DRIVER_CONTAINER, "cyberwave-driver-00000000"])


class TestPeriodicPruneKeepsFloorContainers:
    @pytest.fixture(autouse=True)
    def _docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.removed: list[str] = []
        monkeypatch.setattr(docker_helpers, "docker_available", lambda: True)
        monkeypatch.setattr(
            docker_helpers,
            "docker_ps_by_prefix",
            lambda _p, include_stopped: ["cyberwave-a", "cyberwave-b"] if include_stopped else [],
        )
        monkeypatch.setattr(
            docker_helpers, "docker_rm", lambda name, **kw: self.removed.append(name) or True
        )

    def test_labelled_container_is_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(docker_helpers, "docker_names_with_label", lambda _l: {"cyberwave-a"})
        docker_helpers.docker_prune_stopped_cyberwave_containers(keep_label=FLOOR_LABEL)
        assert self.removed == ["cyberwave-b"]

    def test_unlistable_labels_prune_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(docker_helpers, "docker_names_with_label", lambda _l: None)
        assert docker_helpers.docker_prune_stopped_cyberwave_containers(keep_label=FLOOR_LABEL) == 0
        assert self.removed == []
