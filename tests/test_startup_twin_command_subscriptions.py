"""Tests for ``ensure_twin_command_subscriptions()`` in startup.py.

Covers the operator-triggered ``sync_workflows`` MQTT handshake (CYB-1766
follow-up). The function must:

* Reconcile the subscription set every call rather than latching after
  the first success — twins paired *after* edge-core started should be
  picked up automatically and twins that get unpaired must be dropped.
* Stay cheap (no MQTT client touched) when the desired set already
  matches what's subscribed.
* Survive missing token / environment / fingerprint / API failures
  without crashing the runtime loop.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import cyberwave_edge_core.startup as startup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_subscription_state():
    """Reset the module-level subscription set between tests."""
    startup._SUBSCRIBED_TWIN_COMMAND_UUIDS.clear()
    yield
    startup._SUBSCRIBED_TWIN_COMMAND_UUIDS.clear()


def _make_mqtt_client(prefix: str = "") -> MagicMock:
    client = MagicMock()
    client.mqtt.topic_prefix = prefix
    return client


def _patch_happy_path_deps(
    monkeypatch: pytest.MonkeyPatch,
    twin_uuids: list[str],
    *,
    mqtt_client: MagicMock | None = None,
) -> MagicMock:
    """Stub the four lookups used by ``ensure_twin_command_subscriptions``."""
    monkeypatch.setattr(startup, "load_token", lambda: "tok")
    monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
    monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")
    monkeypatch.setattr(
        startup,
        "_resolve_worker_sync_twin_uuids",
        lambda *_args, **_kwargs: list(twin_uuids),
    )
    client = mqtt_client or _make_mqtt_client()
    monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda _token: client)
    return client


# ===========================================================================
# Early-return / safety paths
# ===========================================================================


class TestEarlyReturns:
    def test_returns_false_when_no_token(self, monkeypatch):
        monkeypatch.setattr(startup, "load_token", lambda: None)
        assert startup.ensure_twin_command_subscriptions() is False

    def test_returns_false_when_no_environment_uuid(self, monkeypatch):
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: None)
        assert startup.ensure_twin_command_subscriptions() is False

    def test_returns_false_when_no_fingerprint(self, monkeypatch):
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: None)
        assert startup.ensure_twin_command_subscriptions() is False

    def test_returns_false_when_resolve_raises(self, monkeypatch):
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")
        monkeypatch.setattr(
            startup,
            "_resolve_worker_sync_twin_uuids",
            MagicMock(side_effect=RuntimeError("API down")),
        )
        assert startup.ensure_twin_command_subscriptions() is False
        assert startup._SUBSCRIBED_TWIN_COMMAND_UUIDS == set()

    def test_returns_true_when_no_twins_and_nothing_subscribed(self, monkeypatch):
        """No-op reconcile must succeed without touching the MQTT client."""
        client = MagicMock()
        _patch_happy_path_deps(monkeypatch, [], mqtt_client=client)
        # Force a failure if the MQTT client is unexpectedly created/used
        monkeypatch.setattr(
            startup,
            "_get_shared_mqtt_client",
            MagicMock(side_effect=AssertionError("MQTT client must not be created")),
        )
        assert startup.ensure_twin_command_subscriptions() is True
        assert startup._SUBSCRIBED_TWIN_COMMAND_UUIDS == set()


# ===========================================================================
# Happy path / diff-based reconciliation
# ===========================================================================


class TestSubscriptionReconciliation:
    def test_initial_call_subscribes_to_all_linked_twins(self, monkeypatch):
        client = _patch_happy_path_deps(monkeypatch, ["twin-a", "twin-b"])

        assert startup.ensure_twin_command_subscriptions() is True

        topics = [
            call.args[0] for call in client.mqtt.subscribe.call_args_list
        ]
        assert sorted(topics) == [
            "cyberwave/twin/twin-a/command",
            "cyberwave/twin/twin-b/command",
        ]
        assert client.mqtt.unsubscribe.call_count == 0
        assert startup._SUBSCRIBED_TWIN_COMMAND_UUIDS == {"twin-a", "twin-b"}

    def test_subsequent_call_with_same_set_is_a_noop(self, monkeypatch):
        client = _patch_happy_path_deps(monkeypatch, ["twin-a"])
        startup.ensure_twin_command_subscriptions()
        client.mqtt.subscribe.reset_mock()
        client.mqtt.unsubscribe.reset_mock()

        assert startup.ensure_twin_command_subscriptions() is True

        # Crucially: no extra subscribe / unsubscribe calls when nothing
        # changed. This is what prevents subscription churn each loop tick.
        client.mqtt.subscribe.assert_not_called()
        client.mqtt.unsubscribe.assert_not_called()

    def test_newly_paired_twin_is_subscribed_on_next_reconcile(self, monkeypatch):
        """A twin paired *after* edge-core started must be picked up."""
        client = _make_mqtt_client()

        twin_state = ["twin-a"]
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")
        monkeypatch.setattr(
            startup,
            "_resolve_worker_sync_twin_uuids",
            lambda *_a, **_kw: list(twin_state),
        )
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda _t: client)

        # First reconcile subscribes only to twin-a.
        startup.ensure_twin_command_subscriptions()
        assert startup._SUBSCRIBED_TWIN_COMMAND_UUIDS == {"twin-a"}
        client.mqtt.subscribe.reset_mock()

        # Operator pairs twin-b in the dashboard → API now returns both.
        twin_state.append("twin-b")
        startup.ensure_twin_command_subscriptions()

        # Only the *new* twin gets a subscribe call (no churn for twin-a).
        topics = [c.args[0] for c in client.mqtt.subscribe.call_args_list]
        assert topics == ["cyberwave/twin/twin-b/command"]
        client.mqtt.unsubscribe.assert_not_called()
        assert startup._SUBSCRIBED_TWIN_COMMAND_UUIDS == {"twin-a", "twin-b"}

    def test_unpaired_twin_is_unsubscribed_on_next_reconcile(self, monkeypatch):
        client = _make_mqtt_client()
        twin_state = ["twin-a", "twin-b"]
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")
        monkeypatch.setattr(
            startup,
            "_resolve_worker_sync_twin_uuids",
            lambda *_a, **_kw: list(twin_state),
        )
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda _t: client)

        startup.ensure_twin_command_subscriptions()
        client.mqtt.subscribe.reset_mock()

        # Operator unpairs twin-a → API now returns only twin-b.
        twin_state.remove("twin-a")
        startup.ensure_twin_command_subscriptions()

        client.mqtt.subscribe.assert_not_called()
        unsubscribed = [
            c.args[0] for c in client.mqtt.unsubscribe.call_args_list
        ]
        assert unsubscribed == ["cyberwave/twin/twin-a/command"]
        assert startup._SUBSCRIBED_TWIN_COMMAND_UUIDS == {"twin-b"}

    def test_subscribe_failure_does_not_mark_topic_as_subscribed(self, monkeypatch):
        """If the broker rejects a subscribe, retry on the next reconcile."""
        client = _make_mqtt_client()
        client.mqtt.subscribe.side_effect = RuntimeError("broker offline")
        _patch_happy_path_deps(monkeypatch, ["twin-a"], mqtt_client=client)

        # Reconcile reports success (other twins might have subscribed),
        # but the failing UUID stays out of the tracked set so the next
        # reconcile retries.
        result = startup.ensure_twin_command_subscriptions()
        assert result is True
        assert startup._SUBSCRIBED_TWIN_COMMAND_UUIDS == set()

        client.mqtt.subscribe.reset_mock()
        client.mqtt.subscribe.side_effect = None
        startup.ensure_twin_command_subscriptions()
        assert client.mqtt.subscribe.call_count == 1
        assert startup._SUBSCRIBED_TWIN_COMMAND_UUIDS == {"twin-a"}

    def test_topic_prefix_is_applied(self, monkeypatch):
        client = _make_mqtt_client(prefix="dev/")
        _patch_happy_path_deps(monkeypatch, ["twin-a"], mqtt_client=client)

        startup.ensure_twin_command_subscriptions()
        topics = [c.args[0] for c in client.mqtt.subscribe.call_args_list]
        assert topics == ["dev/cyberwave/twin/twin-a/command"]

    def test_handler_wired_to_subscribe(self, monkeypatch):
        """The handler installed must be the twin-command dispatcher."""
        client = _patch_happy_path_deps(monkeypatch, ["twin-a"])
        startup.ensure_twin_command_subscriptions()
        # Each subscribe call is (topic, handler).
        _, handler = client.mqtt.subscribe.call_args_list[0].args
        assert handler is startup._handle_twin_command_message
