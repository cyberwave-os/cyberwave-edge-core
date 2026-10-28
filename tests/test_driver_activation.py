"""Activating a staged driver update: switch the twin's drivers, or go back.

Docker, the launcher and the backend are replaced. What is pinned here is the decision
logic: when the edge recreates a twin's containers and when it refuses, what it reports for
each outcome, and that a failed switch leaves the previous release running.
"""

from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from cyberwave_edge_core import driver_activation as da
from cyberwave_edge_core import driver_update_state as state
from cyberwave_edge_core import startup

TWIN = "11111111-2222-4333-8444-555555555555"
OTHER_TWIN = "99999999-2222-4333-8444-555555555555"
CONTAINER = "cyberwave-driver-11111111-driver"
OLD_IMAGE = "cyberwaveos/go2:humble-0.1.0"
NEW_IMAGE = "cyberwaveos/go2:humble-0.2.0"
PREVIOUS_DRIVERS = {"default": {"docker_image": OLD_IMAGE}}


def _payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "command": da.ACTIVATE_COMMAND,
        "request_id": "req-1",
        "twin_uuid": TWIN,
        "package": "go2-ros2-driver",
        "target_version": "0.2.0",
    }
    payload.update(overrides)
    return payload


def _stage(**overrides: Any) -> None:
    record = {
        "request_id": "req-1",
        "twin_uuid": TWIN,
        "package": "go2-ros2-driver",
        "images": [{"service": "driver", "image": NEW_IMAGE, "digest": "sha256:" + "a" * 64}],
        "previous_drivers": PREVIOUS_DRIVERS,
    }
    record.update(overrides)
    state.write_staged_record(record)


@pytest.fixture
def host(monkeypatch: Any, tmp_path: Any) -> SimpleNamespace:
    """A host with the old release running and the new one staged."""
    host = SimpleNamespace(
        running={CONTAINER: OLD_IMAGE},
        events=[],
        recreate_calls=[],
        recreate_results=[],  # one entry per call: a result list, or an Exception
        settled=[],  # one entry per wait: None, or the problem
        overrides={},
        staged_here=True,
        rebaselined=[],
        rolled_back=[],
        running_up=True,  # whether the containers in `running` are actually up
    )

    def recreate(twin_uuid, *, drivers_override=None):
        host.events.append("recreate")
        host.recreate_calls.append(drivers_override)
        outcome = host.recreate_results.pop(0) if host.recreate_results else None
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is not None:
            return outcome
        return [{"twin_uuid": twin_uuid, "service_name": "driver", "success": True}]

    def remove(containers):
        host.events.append("remove")
        host.running = {}

    def wait(containers, *, seconds):
        host.events.append("settle")
        problem = host.settled.pop(0) if host.settled else None
        if problem is None:
            host.running = {CONTAINER: NEW_IMAGE if len(host.recreate_calls) == 1 else OLD_IMAGE}
        return problem

    monkeypatch.setattr(state, "updates_dir", lambda: tmp_path / "driver-updates")
    monkeypatch.setattr(da, "twin_containers", lambda twin_uuid: sorted(host.running))
    monkeypatch.setattr(da, "running_images", lambda containers: dict(host.running))
    monkeypatch.setattr(da, "remove_containers", remove)
    monkeypatch.setattr(da, "_is_running", lambda name: name in host.running and host.running_up)
    monkeypatch.setattr(da, "_recreate", recreate)
    monkeypatch.setattr(da, "wait_until_settled", wait)
    monkeypatch.setattr(da, "settle_seconds", lambda: 0.0)
    monkeypatch.setattr(da, "_staged_images_present", lambda record: None if host.staged_here else NEW_IMAGE)
    monkeypatch.setattr("cyberwave_edge_core.docker_helpers.docker_available", lambda: True)
    monkeypatch.setattr(startup, "load_driver_overrides", lambda: host.overrides)
    monkeypatch.setattr(
        startup, "_rebaseline_driver_metadata", lambda twin_uuid: host.rebaselined.append(twin_uuid)
    )
    monkeypatch.setattr(
        startup,
        "_rebaseline_after_rollback",
        lambda twin_uuid, previous: host.rolled_back.append((twin_uuid, previous)),
    )
    monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {})
    monkeypatch.setattr(startup, "_DRIVER_HEALTH_PREVIOUS", {})
    monkeypatch.setattr(startup, "_edge_core_restart_in_progress", lambda: False)
    monkeypatch.setattr(startup, "_HANDLED_EDGE_COMMAND_REQUEST_IDS", deque(maxlen=1024))
    monkeypatch.setattr(startup, "_ACTIVATING_TWINS", set())
    _stage()
    return host


def _activate(**overrides: Any) -> dict[str, str]:
    return da.activate(da.parse_request(_payload(**overrides)))


class TestParse:
    def test_a_complete_command_becomes_a_request(self):
        request = da.parse_request(_payload())

        assert (request.request_id, request.twin_uuid) == ("req-1", TWIN)
        assert request.target_version == "0.2.0"

    @pytest.mark.parametrize("request_id", ["", "../x", "a/b", "pulls.json"])
    def test_a_request_id_that_is_not_a_plain_name_is_invalid(self, request_id):
        with pytest.raises(da.ActivationError) as raised:
            da.parse_request(_payload(request_id=request_id))

        assert raised.value.code == da.INVALID_COMMAND

    def test_a_command_without_a_twin_is_invalid(self):
        with pytest.raises(da.ActivationError) as raised:
            da.parse_request(_payload(twin_uuid=""))

        assert raised.value.code == da.INVALID_COMMAND


class TestSwitch:
    def test_the_twins_containers_are_removed_then_started_from_the_new_metadata(self, host):
        observed = _activate()

        assert host.events == ["remove", "recreate", "settle"]
        assert host.recreate_calls == [None]
        assert observed == {CONTAINER: NEW_IMAGE}

    def test_what_ran_before_is_kept_for_a_day_so_it_can_come_back(self, host):
        _activate()

        stored = state._read_json(state.updates_dir() / state.PREVIOUS_FILE)
        assert stored[TWIN]["images"] == [OLD_IMAGE]
        assert stored[TWIN]["request_id"] == "req-1"
        assert state.previous_image_sets() == {OLD_IMAGE}
        # Still kept a day from now, not just "some time from the epoch".
        soon = time.time() + da.PREVIOUS_KEEP_SECONDS - 60
        assert state.previous_image_sets(now=soon) == {OLD_IMAGE}
        assert state.previous_image_sets(now=soon + 120) == set()

    def test_the_settings_reconciler_is_told_the_new_containers_are_the_baseline(self, host):
        _activate()

        assert host.rebaselined == [TWIN]

    def test_the_staged_record_is_kept_until_the_report_is_delivered(self, host):
        _activate()

        assert state.read_staged_record("req-1") is not None

    def test_the_twin_is_released_afterwards(self, host):
        _activate()

        assert startup.driver_activation_in_progress(TWIN) is False

    def test_the_twin_is_claimed_while_it_switches(self, host, monkeypatch):
        seen: list[bool] = []
        original = da._recreate
        monkeypatch.setattr(
            da,
            "_recreate",
            lambda twin, **kw: seen.append(startup.driver_activation_in_progress(TWIN))
            or original(twin, **kw),
        )

        _activate()

        assert seen == [True]


class TestTheLog:
    def test_a_switch_logs_what_it_goes_to_and_what_runs_afterwards(self, host, caplog):
        _stage(images=[{"service": "driver", "image": NEW_IMAGE, "digest": "sha256:" + "a" * 64}])

        with caplog.at_level("INFO", logger="cyberwave_edge_core.driver_activation"):
            _activate()

        text = caplog.text
        assert f"switching twin {TWIN} to go2-ros2-driver 0.2.0: {NEW_IMAGE}@sha256:" in text
        assert f"twin {TWIN} runs {CONTAINER}={OLD_IMAGE} now" in text
        assert f"twin {TWIN} now runs {CONTAINER}={NEW_IMAGE}" in text

    def test_a_twin_already_on_the_images_says_so(self, host, caplog):
        host.running = {CONTAINER: NEW_IMAGE}

        with caplog.at_level("INFO", logger="cyberwave_edge_core.driver_activation"):
            _activate()

        assert f"twin {TWIN} already runs the staged images ({CONTAINER}={NEW_IMAGE})" in caplog.text

    def test_a_rollback_logs_what_runs_again(self, host, caplog):
        host.settled = ["x is exited"]

        with caplog.at_level("INFO", logger="cyberwave_edge_core.driver_activation"):
            with pytest.raises(da.ActivationError):
                _activate()

        assert f"Rolled back req-1: twin {TWIN} runs {CONTAINER}=" in caplog.text


class TestRefusals:
    def test_an_update_that_was_never_staged_fails_with_a_code(self, host):
        state.staged_record_path("req-1").unlink()

        with pytest.raises(da.ActivationError) as raised:
            _activate()

        assert raised.value.code == da.NOT_STAGED
        assert host.events == []

    def test_a_twin_with_a_local_override_is_never_switched(self, host):
        """It runs what the override names; reporting `active` would confirm a release that is
        not running."""
        host.overrides = {TWIN: "mine:dev"}

        with pytest.raises(da.ActivationError) as raised:
            _activate()

        assert raised.value.code == da.LOCALLY_OVERRIDDEN
        assert host.events == []

    def test_an_override_on_another_twin_does_not_matter(self, host):
        host.overrides = {OTHER_TWIN: "mine:dev"}

        _activate()

        assert "recreate" in host.events

    def test_a_staged_image_that_is_gone_fails_before_anything_is_removed(self, host):
        host.staged_here = False

        with pytest.raises(da.ActivationError) as raised:
            _activate()

        assert raised.value.code == da.STAGED_IMAGES_MISSING
        assert NEW_IMAGE in str(raised.value)
        assert host.events == []

    def test_a_record_for_another_twin_is_invalid(self, host):
        _stage(twin_uuid=OTHER_TWIN)

        with pytest.raises(da.ActivationError) as raised:
            _activate()

        assert raised.value.code == da.INVALID_COMMAND

    def test_no_docker_fails_with_a_code(self, host, monkeypatch):
        monkeypatch.setattr("cyberwave_edge_core.docker_helpers.docker_available", lambda: False)

        with pytest.raises(da.ActivationError) as raised:
            _activate()

        assert raised.value.code == da.DOCKER_UNAVAILABLE

    def test_a_busy_edge_does_nothing_and_says_nothing(self, host, monkeypatch):
        monkeypatch.setattr(startup, "_edge_core_restart_in_progress", lambda: True)

        with pytest.raises(da.Busy):
            _activate()

        assert host.events == []

    def test_a_twin_already_being_activated_is_refused(self, host):
        assert startup.begin_driver_activation(TWIN)

        with pytest.raises(da.Busy):
            _activate()

    def test_a_failed_check_releases_nothing_it_did_not_claim(self, host):
        assert startup.begin_driver_activation(TWIN)
        with pytest.raises(da.Busy):
            _activate()

        assert startup.driver_activation_in_progress(TWIN) is True


class TestAlreadyRunning:
    def test_a_twin_already_on_the_staged_images_is_not_recreated(self, host):
        """The state after a switch whose `active` report did not get through."""
        host.running = {CONTAINER: NEW_IMAGE}

        observed = _activate()

        assert observed == {CONTAINER: NEW_IMAGE}
        assert host.events == []

    def test_stopped_containers_on_the_staged_images_are_not_running_it(self, host):
        host.running = {CONTAINER: NEW_IMAGE}
        host.running_up = False

        _activate()

        assert "recreate" in host.events

    def test_one_staged_image_missing_from_what_runs_means_it_is_recreated(self, host):
        _stage(
            images=[
                {"service": "driver", "image": NEW_IMAGE},
                {"service": "nav", "image": "cyberwaveos/nav2:humble-0.2.0"},
            ]
        )
        host.running = {CONTAINER: NEW_IMAGE}

        _activate()

        assert "recreate" in host.events


class TestRollback:
    def test_a_container_that_does_not_start_puts_the_previous_release_back(self, host):
        host.recreate_results = [
            [{"twin_uuid": TWIN, "service_name": "driver", "success": False}],
            None,
        ]

        with pytest.raises(da.ActivationError) as raised:
            _activate()

        assert raised.value.code == da.START_FAILED
        assert "previous release is running again" in str(raised.value)
        assert host.recreate_calls == [None, PREVIOUS_DRIVERS]
        assert host.events.count("remove") == 2
        assert host.rolled_back == [(TWIN, PREVIOUS_DRIVERS)]

    def test_drivers_that_do_not_stay_up_put_the_previous_release_back(self, host):
        host.settled = ["cyberwave-driver-11111111-driver restarted"]

        with pytest.raises(da.ActivationError) as raised:
            _activate()

        assert raised.value.code == da.UNHEALTHY
        assert "restarted" in str(raised.value)
        assert host.recreate_calls == [None, PREVIOUS_DRIVERS]

    def test_an_error_while_starting_is_rolled_back_too(self, host):
        host.recreate_results = [RuntimeError("docker exploded"), None]

        with pytest.raises(da.ActivationError) as raised:
            _activate()

        assert raised.value.code == da.UNKNOWN
        assert "docker exploded" in str(raised.value)
        assert host.recreate_calls == [None, PREVIOUS_DRIVERS]

    def test_nothing_started_counts_as_failed(self, host):
        host.recreate_results = [[], None]

        with pytest.raises(da.ActivationError) as raised:
            _activate()

        assert raised.value.code == da.START_FAILED

    def test_without_an_earlier_configuration_it_says_the_drivers_are_down(self, host):
        _stage(previous_drivers=None)
        host.recreate_results = [[{"twin_uuid": TWIN, "service_name": "driver", "success": False}]]

        with pytest.raises(da.ActivationError) as raised:
            _activate()

        assert raised.value.code == da.START_FAILED
        assert "no earlier configuration" in str(raised.value)
        assert len(host.recreate_calls) == 1

    def test_a_rollback_that_fails_too_says_so(self, host):
        host.recreate_results = [
            [{"twin_uuid": TWIN, "service_name": "driver", "success": False}],
            RuntimeError("nothing starts"),
        ]

        with pytest.raises(da.ActivationError) as raised:
            _activate()

        assert raised.value.code == da.START_FAILED
        assert "failed too" in str(raised.value) and "nothing starts" in str(raised.value)

    def test_a_failed_rollback_leaves_the_containers_to_driver_revival(self, host):
        host.recreate_results = [
            [{"twin_uuid": TWIN, "service_name": "driver", "success": False}],
            RuntimeError("nothing starts"),
        ]

        with pytest.raises(da.ActivationError):
            _activate()

        assert startup._CONTAINER_TWIN_MAP == {CONTAINER: TWIN}
        assert startup._DRIVER_HEALTH_PREVIOUS == {CONTAINER: "running"}

    def test_no_earlier_configuration_also_leaves_them_to_driver_revival(self, host):
        _stage(previous_drivers=None)
        host.recreate_results = [[{"twin_uuid": TWIN, "service_name": "driver", "success": False}]]

        with pytest.raises(da.ActivationError):
            _activate()

        assert startup._CONTAINER_TWIN_MAP == {CONTAINER: TWIN}

    def test_a_rollback_that_works_hands_nothing_back(self, host):
        host.settled = ["x is exited"]

        with pytest.raises(da.ActivationError):
            _activate()

        assert startup._DRIVER_HEALTH_PREVIOUS == {}

    def test_the_failed_update_is_forgotten_so_its_images_can_be_collected(self, host):
        host.settled = ["x is exited"]

        with pytest.raises(da.ActivationError):
            _activate()

        assert state.read_staged_record("req-1") is None

    def test_the_twin_is_released_after_a_failure(self, host):
        host.settled = ["x is exited"]

        with pytest.raises(da.ActivationError):
            _activate()

        assert startup.driver_activation_in_progress(TWIN) is False


class _Reports:
    def __init__(self, deliver: bool = True) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.deliver = deliver

    def send(self, phase: str, **fields: Any) -> bool:
        self.sent.append((phase, fields))
        return self.deliver


class TestRun:
    @pytest.fixture
    def reports(self, host, monkeypatch) -> _Reports:
        reports = _Reports()
        monkeypatch.setattr(da, "StatusReporter", lambda request_id: reports)
        startup._HANDLED_EDGE_COMMAND_REQUEST_IDS.append(
            startup._edge_command_key(da.ACTIVATE_COMMAND, "req-1")
        )
        return reports

    def _remembered(self) -> bool:
        return startup._edge_command_key(da.ACTIVATE_COMMAND, "req-1") in (
            startup._HANDLED_EDGE_COMMAND_REQUEST_IDS
        )

    def test_a_switch_reports_active_with_what_runs_and_forgets_the_staged_record(
        self, host, reports
    ):
        da._run(da.parse_request(_payload()))

        assert reports.sent == [("active", {"observed": {CONTAINER: NEW_IMAGE}})]
        assert state.read_staged_record("req-1") is None
        assert self._remembered()

    def test_a_failed_switch_reports_failed_with_its_code_and_message(self, host, reports):
        host.settled = ["x is exited"]

        da._run(da.parse_request(_payload()))

        (phase, fields), = reports.sent
        assert phase == "failed"
        assert fields["error_code"] == da.UNHEALTHY
        assert "x is exited" in fields["error"]

    def test_an_unexpected_error_reports_failed_unknown(self, host, reports, monkeypatch):
        monkeypatch.setattr(da, "activate", lambda request: (_ for _ in ()).throw(KeyError("boom")))

        da._run(da.parse_request(_payload()))

        (phase, fields), = reports.sent
        assert (phase, fields["error_code"]) == ("failed", da.UNKNOWN)

    def test_a_busy_edge_reports_nothing_and_lets_the_next_delivery_through(self, host, reports):
        assert startup.begin_driver_activation(TWIN)

        da._run(da.parse_request(_payload()))

        assert reports.sent == []
        assert not self._remembered()

    def test_a_report_that_did_not_get_through_lets_the_next_delivery_retry(self, host, reports):
        reports.deliver = False

        da._run(da.parse_request(_payload()))

        assert not self._remembered()
        assert state.read_staged_record("req-1") is not None

    def test_the_retry_after_an_undelivered_report_just_reports_active(self, host, reports):
        reports.deliver = False
        da._run(da.parse_request(_payload()))
        host.events.clear()
        reports.deliver = True
        reports.sent.clear()

        da._run(da.parse_request(_payload()))

        assert host.events == []
        assert reports.sent == [("active", {"observed": {CONTAINER: NEW_IMAGE}})]


class TestHandle:
    @pytest.fixture
    def jobs(self, monkeypatch):
        submitted: list[tuple[Any, ...]] = []

        class _Inline:
            def submit(self, fn, *args):
                submitted.append((fn.__name__, *args))

        monkeypatch.setattr(da, "_executor", lambda: _Inline())
        return submitted

    def test_a_valid_command_is_handed_to_the_worker(self, jobs):
        da.handle_activate_command(_payload())

        assert [job[0] for job in jobs] == ["_run"]

    def test_an_invalid_command_with_an_id_is_reported_failed(self, jobs):
        da.handle_activate_command(_payload(twin_uuid=""))

        name, request_id, error = jobs[0]
        assert (name, request_id) == ("_reject", "req-1")
        assert error.code == da.INVALID_COMMAND

    def test_an_invalid_command_without_a_usable_id_is_dropped(self, jobs):
        da.handle_activate_command(_payload(request_id="../x"))

        assert jobs == []

    def test_the_command_is_registered_with_the_edge(self):
        assert startup._EDGE_COMMAND_HANDLERS[da.ACTIVATE_COMMAND] is da.handle_activate_command


class TestWaitUntilSettled:
    @pytest.fixture
    def docker(self, monkeypatch):
        docker = SimpleNamespace(frames=[], seen=0)

        def inspect(name):
            frame = docker.frames[min(docker.seen, len(docker.frames) - 1)]
            docker.seen += 1
            return frame.get(name)

        monkeypatch.setattr("cyberwave_edge_core.docker_helpers.docker_inspect", inspect)
        return docker

    def _clock(self):
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        return (lambda: now[0]), sleep

    @staticmethod
    def _up(restart_count: int = 0, restarting: bool = False) -> dict[str, Any]:
        return {
            "State": {"Status": "running", "Restarting": restarting},
            "RestartCount": restart_count,
        }

    def test_containers_that_stay_up_settle(self, docker):
        docker.frames = [{"a": self._up(), "b": self._up()}]
        clock, sleep = self._clock()

        assert da.wait_until_settled(["a", "b"], seconds=10, sleep=sleep, clock=clock) is None

    def test_a_container_that_exits_fails_at_once(self, docker):
        docker.frames = [
            {"a": self._up()},
            {"a": self._up()},
            {"a": {"State": {"Status": "exited"}, "RestartCount": 0}},
        ]
        clock, sleep = self._clock()

        problem = da.wait_until_settled(["a"], seconds=60, sleep=sleep, clock=clock)

        assert problem == "a is exited"
        assert clock() < 60

    def test_a_container_that_is_restarting_fails(self, docker):
        docker.frames = [{"a": self._up()}, {"a": self._up(restarting=True)}]
        clock, sleep = self._clock()

        assert da.wait_until_settled(["a"], seconds=10, sleep=sleep, clock=clock) == (
            "a is restarting"
        )

    def test_a_container_that_restarted_since_the_first_look_fails(self, docker):
        docker.frames = [{"a": self._up(restart_count=0)}, {"a": self._up(restart_count=1)}]
        clock, sleep = self._clock()

        assert da.wait_until_settled(["a"], seconds=10, sleep=sleep, clock=clock) == (
            "a restarted"
        )

    def test_a_container_that_vanished_fails(self, docker):
        docker.frames = [{"a": self._up()}, {}]
        clock, sleep = self._clock()

        assert da.wait_until_settled(["a"], seconds=10, sleep=sleep, clock=clock) == "a is none"

    def test_a_zero_second_check_looks_once(self, docker):
        docker.frames = [{"a": self._up()}]
        clock, sleep = self._clock()

        assert da.wait_until_settled(["a"], seconds=0, sleep=sleep, clock=clock) is None
        assert clock() == 0.0


def test_the_settle_period_can_be_set_by_environment(monkeypatch):
    monkeypatch.setenv(da.SETTLE_SECONDS_ENV, "5")
    assert da.settle_seconds() == 5.0
    monkeypatch.setenv(da.SETTLE_SECONDS_ENV, "not a number")
    assert da.settle_seconds() == da.DEFAULT_SETTLE_SECONDS
    monkeypatch.delenv(da.SETTLE_SECONDS_ENV)
    assert da.settle_seconds() == da.DEFAULT_SETTLE_SECONDS
