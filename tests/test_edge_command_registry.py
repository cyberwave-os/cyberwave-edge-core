"""Edge commands arrive over MQTT and in the ``/edges/discover`` response.

Both routes share one registry and one ``request_id`` dedupe, so the backend can
send a command twice (a hint over MQTT, the contract through discover) and the
edge acts once. A command nobody registered is ignored, which is what lets a newer
backend talk to an older edge.
"""

from __future__ import annotations

import sys
import types
from collections import deque

import pytest

from cyberwave_edge_core import startup


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(startup, "_EDGE_COMMAND_HANDLERS", dict(startup._EDGE_COMMAND_HANDLERS))
    monkeypatch.setattr(startup, "_HANDLED_EDGE_COMMAND_REQUEST_IDS", deque(maxlen=1024))


def _command(request_id: str = "req-1", command: str = "do_thing", **extra) -> dict:
    return {"command": command, "request_id": request_id, **extra}


def _discover(monkeypatch, response: dict) -> bool:
    """Run one keepalive upload against a backend that answers ``response``."""
    host_metrics = types.ModuleType("cyberwave.edge.host_metrics")
    host_metrics.read_host_facts = lambda: types.SimpleNamespace(to_dict=dict)
    monkeypatch.setitem(sys.modules, "cyberwave.edge.host_metrics", host_metrics)
    monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp-test")
    client = types.SimpleNamespace(
        edges=types.SimpleNamespace(discover=lambda **kwargs: response)
    )
    monkeypatch.setattr(startup, "Cyberwave", lambda *args, **kwargs: client)
    return startup._post_host_facts_once("test-token")


def _pending(*commands: dict) -> dict:
    return {"twins": [], "driver_deployments": list(commands)}


def test_a_registered_command_runs_its_handler_with_the_payload():
    seen: list[dict] = []
    startup.register_edge_command("do_thing", seen.append)

    assert startup.handle_edge_command(_command(command="  Do_Thing ", twin_uuid="t-1")) is True

    assert seen == [_command(command="  Do_Thing ", twin_uuid="t-1")]


def test_a_command_registered_in_mixed_case_still_matches():
    seen: list[dict] = []
    startup.register_edge_command("  Do_Thing ", seen.append)

    assert startup.handle_edge_command(_command(command="do_thing")) is True

    assert len(seen) == 1


def test_concurrent_deliveries_of_one_request_run_it_once():
    """MQTT and the keepalive thread can deliver the same request at the same moment."""
    import threading

    runs: list[str] = []
    startup.register_edge_command("do_thing", lambda payload: runs.append(payload["request_id"]))

    class _SlowDeque(startup.deque):
        """Widen the check-then-record window so a missing lock shows up every time."""

        def __contains__(self, item):
            found = super().__contains__(item)
            threading.Event().wait(0.01)
            return found

    startup._HANDLED_EDGE_COMMAND_REQUEST_IDS = _SlowDeque(maxlen=1024)
    barrier = threading.Barrier(4)

    def deliver():
        barrier.wait()
        startup.handle_edge_command(_command("same"))

    threads = [threading.Thread(target=deliver) for _ in range(4)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]

    assert runs == ["same"]


def test_stage_and_activate_of_one_deployment_both_run_though_they_share_a_request_id():
    """The backend labels a deployment's two commands with the same request_id."""
    ran: list[str] = []
    startup.register_edge_command("stage_thing", lambda payload: ran.append("stage"))
    startup.register_edge_command("activate_thing", lambda payload: ran.append("activate"))

    assert startup.handle_edge_command(_command("deployment-1", command="stage_thing")) is True
    assert startup.handle_edge_command(_command("deployment-1", command="activate_thing")) is True

    assert ran == ["stage", "activate"]


def test_each_of_the_two_commands_is_still_deduplicated_on_its_own():
    ran: list[str] = []
    startup.register_edge_command("stage_thing", lambda payload: ran.append("stage"))
    startup.register_edge_command("activate_thing", lambda payload: ran.append("activate"))

    for command in ("stage_thing", "activate_thing", "stage_thing", "activate_thing"):
        startup.handle_edge_command(_command("deployment-1", command=command))

    assert ran == ["stage", "activate"]


def test_an_unknown_command_is_ignored_and_not_remembered():
    seen: list[dict] = []

    assert startup.handle_edge_command(_command(command="from_the_future")) is False

    # An edge that later learns the command must still act on this request.
    startup.register_edge_command("from_the_future", seen.append)
    assert startup.handle_edge_command(_command(command="from_the_future")) is True
    assert len(seen) == 1


def test_a_payload_that_is_not_a_mapping_is_ignored():
    assert startup.handle_edge_command("restart_edge_core") is False
    assert startup.handle_edge_command(None) is False


def test_the_built_in_commands_are_registered():
    assert startup.EDGE_COMMAND_RESTART in startup._EDGE_COMMAND_HANDLERS
    for command in startup.remote_access.COMMANDS:
        assert command in startup._EDGE_COMMAND_HANDLERS


def test_the_dedupe_is_shared_by_mqtt_and_discover(monkeypatch):
    seen: list[dict] = []
    startup.register_edge_command("do_thing", seen.append)

    startup._handle_edge_command_message("edges/e/command", _command("both"))
    _discover(monkeypatch, _pending(_command("both")))

    assert len(seen) == 1


class TestDiscover:
    def test_the_pending_commands_reach_their_handlers(self, monkeypatch):
        seen: list[dict] = []
        startup.register_edge_command("do_thing", seen.append)

        assert _discover(monkeypatch, _pending(_command("a"), _command("b"))) is True

        assert [command["request_id"] for command in seen] == ["a", "b"]

    def test_the_same_pending_command_on_every_tick_runs_once(self, monkeypatch):
        seen: list[dict] = []
        startup.register_edge_command("do_thing", seen.append)

        for _ in range(3):
            _discover(monkeypatch, _pending(_command("a")))

        assert len(seen) == 1

    def test_a_failing_handler_does_not_fail_the_keepalive_or_the_next_command(
        self, monkeypatch
    ):
        seen: list[dict] = []

        def boom(payload):
            raise RuntimeError("handler exploded")

        startup.register_edge_command("explodes", boom)
        startup.register_edge_command("do_thing", seen.append)

        pending = _pending(_command("a", command="explodes"), _command("b"))
        assert _discover(monkeypatch, pending) is True

        assert [command["request_id"] for command in seen] == ["b"]

    def test_a_command_whose_handler_failed_is_tried_again_on_the_next_tick(self, monkeypatch):
        attempts: list[str] = []

        def flaky(payload):
            attempts.append(payload["request_id"])
            if len(attempts) == 1:
                raise RuntimeError("not yet")

        startup.register_edge_command("do_thing", flaky)

        _discover(monkeypatch, _pending(_command("a")))
        _discover(monkeypatch, _pending(_command("a")))
        _discover(monkeypatch, _pending(_command("a")))

        assert attempts == ["a", "a"]

    def test_a_failing_command_logs_its_name_and_id_not_its_payload(self, monkeypatch, caplog):
        def boom(payload):
            raise RuntimeError("handler exploded")

        startup.register_edge_command("explodes", boom)
        manifest = {"images": ["registry.example/driver:1.2.3"]}

        _discover(monkeypatch, _pending(_command("a", command="explodes", manifest=manifest)))

        failure = next(record for record in caplog.records if "from /discover failed" in record.message)
        assert "explodes" in failure.message and "request_id=a" in failure.message
        assert "registry.example" not in failure.message

    @pytest.mark.parametrize(
        "response",
        [
            {"twins": []},  # a backend that predates the field
            {"twins": [], "driver_deployments": None},
            {"twins": [], "driver_deployments": []},
        ],
    )
    def test_nothing_pending_changes_nothing(self, monkeypatch, response):
        seen: list[dict] = []
        startup.register_edge_command("do_thing", seen.append)

        assert _discover(monkeypatch, response) is True

        assert seen == []
