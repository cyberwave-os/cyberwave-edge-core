"""Tests for publish_network_facts_log (network interface change -> driver_log)."""

from __future__ import annotations

import pytest

from cyberwave_edge_core import driver_logs, startup


class _FakeMQTT:
    def __init__(self, connected: bool = True) -> None:
        self.topic_prefix = "cw/"
        self.connected = connected
        self.published: list[tuple[str, dict]] = []

    def publish(self, topic, payload):  # type: ignore[no-untyped-def]
        # Mirrors MQTTClient.publish: a disconnected socket drops the message
        # with a warning rather than raising.
        if not self.connected:
            return
        self.published.append((topic, payload))


class _FakeClient:
    def __init__(self, connected: bool = True) -> None:
        self.mqtt = _FakeMQTT(connected=connected)


NICS_A = [{"name": "eth0", "ipv4_address": "192.168.1.42", "is_up": True}]
NICS_B = [{"name": "wlan0", "ipv4_address": "10.0.0.77", "is_up": True}]


def _twins_published_to(client: _FakeClient) -> set[str]:
    return {topic.split("/")[3] for topic, _payload in client.mqtt.published}


@pytest.fixture(autouse=True)
def _reset_dedup_state():
    """The dedup signatures are module state; isolate tests from each other."""
    driver_logs._LAST_NETWORK_INTERFACES_SIGNATURES.clear()
    yield
    driver_logs._LAST_NETWORK_INTERFACES_SIGNATURES.clear()


def test_publishes_to_every_bound_twin(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeClient()
    monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

    driver_logs.publish_network_facts_log(["twin-1", "twin-2"], NICS_A, token="tok")

    assert len(fake_client.mqtt.published) == 2
    topics = {topic for topic, _payload in fake_client.mqtt.published}
    assert topics == {"cw/cyberwave/twin/twin-1/driverlog", "cw/cyberwave/twin/twin-2/driverlog"}
    _, payload = fake_client.mqtt.published[0]
    assert payload["type"] == "driver_log"
    assert "eth0=192.168.1.42 (up)" in payload["message"]
    assert payload["container_name"] == "network"
    assert payload["source"] == "edge"


def test_does_not_republish_when_interfaces_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()
    monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

    driver_logs.publish_network_facts_log(["twin-1"], NICS_A, token="tok")
    driver_logs.publish_network_facts_log(["twin-1"], NICS_A, token="tok")

    assert len(fake_client.mqtt.published) == 1


def test_republishes_when_interfaces_change(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeClient()
    monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

    driver_logs.publish_network_facts_log(["twin-1"], NICS_A, token="tok")
    driver_logs.publish_network_facts_log(["twin-1"], NICS_B, token="tok")

    assert len(fake_client.mqtt.published) == 2
    _, second_payload = fake_client.mqtt.published[1]
    assert "wlan0=10.0.0.77 (up)" in second_payload["message"]


def test_no_bound_twins_does_not_consume_the_baseline_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An edge with zero bound twins must not silently swallow the first
    log line once a twin later gets bound with the same interface set."""
    fake_client = _FakeClient()
    monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

    driver_logs.publish_network_facts_log([], NICS_A, token="tok")
    assert fake_client.mqtt.published == []

    driver_logs.publish_network_facts_log(["twin-1"], NICS_A, token="tok")
    assert len(fake_client.mqtt.published) == 1


def test_twin_bound_later_receives_the_baseline_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dedup is per twin: a twin bound after the first publish has never seen
    the line, so an unchanged interface set must still reach it once."""
    fake_client = _FakeClient()
    monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

    driver_logs.publish_network_facts_log(["twin-1"], NICS_A, token="tok")
    driver_logs.publish_network_facts_log(["twin-1", "twin-2"], NICS_A, token="tok")

    assert _twins_published_to(fake_client) == {"twin-1", "twin-2"}
    # ...and twin-1 is not spammed a second time with the same interfaces.
    assert len(fake_client.mqtt.published) == 2


def test_no_network_interfaces_publishes_none_detected_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()
    monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

    driver_logs.publish_network_facts_log(["twin-1"], [], token="tok")

    assert len(fake_client.mqtt.published) == 1
    _, payload = fake_client.mqtt.published[0]
    assert payload["message"] == "Network interfaces: none detected"


class TestFailedPublishIsRetried:
    """A broker that is unreachable on the tick where the IP changed must not
    lose that line permanently -- nothing is recorded unless it was sent."""

    def test_retries_after_no_mqtt_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: None)
        driver_logs.publish_network_facts_log(["twin-1"], NICS_A, token="tok")

        fake_client = _FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)
        driver_logs.publish_network_facts_log(["twin-1"], NICS_A, token="tok")

        assert len(fake_client.mqtt.published) == 1

    def test_retries_after_disconnected_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        disconnected = _FakeClient(connected=False)
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: disconnected)
        driver_logs.publish_network_facts_log(["twin-1"], NICS_A, token="tok")
        assert disconnected.mqtt.published == []

        fake_client = _FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)
        driver_logs.publish_network_facts_log(["twin-1"], NICS_A, token="tok")

        assert len(fake_client.mqtt.published) == 1

    def test_partial_fan_out_only_records_delivered_twins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """twin-1 lands, twin-2's resolve fails -> only twin-2 is retried."""
        fake_client = _FakeClient()
        resolved: list[str] = []

        def _flaky(token):  # type: ignore[no-untyped-def]
            resolved.append(token)
            return fake_client if len(resolved) == 1 else None

        monkeypatch.setattr(startup, "_get_shared_mqtt_client", _flaky)
        driver_logs.publish_network_facts_log(["twin-1", "twin-2"], NICS_A, token="tok")
        assert _twins_published_to(fake_client) == {"twin-1"}

        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)
        driver_logs.publish_network_facts_log(["twin-1", "twin-2"], NICS_A, token="tok")

        assert _twins_published_to(fake_client) == {"twin-1", "twin-2"}
        assert len(fake_client.mqtt.published) == 2  # twin-1 not re-sent


class TestAddressLessInterfacesAreIgnored:
    """Interfaces with no bound IPv4 are nothing to SSH to, and they churn
    (docker0/veth as containers cycle, usb0 as cables move). Keeping them out
    of the signature stops that churn re-firing the fan-out, and keeps the log
    line consistent with the frontend, which filters the same way."""

    def test_address_less_interface_churn_does_not_republish(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = _FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

        driver_logs.publish_network_facts_log(
            ["twin-1"],
            NICS_A + [{"name": "veth1a2b", "ipv4_address": None, "is_up": True}],
            token="tok",
        )
        driver_logs.publish_network_facts_log(
            ["twin-1"],
            NICS_A + [{"name": "veth9z8y", "ipv4_address": None, "is_up": True}],
            token="tok",
        )

        assert len(fake_client.mqtt.published) == 1

    def test_address_less_interfaces_are_absent_from_the_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = _FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

        driver_logs.publish_network_facts_log(
            ["twin-1"],
            NICS_A + [{"name": "wlan0", "ipv4_address": None, "is_up": False}],
            token="tok",
        )

        _, payload = fake_client.mqtt.published[0]
        assert payload["message"] == "Network interfaces: eth0=192.168.1.42 (up)"

    def test_all_address_less_reports_none_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_client = _FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

        driver_logs.publish_network_facts_log(
            ["twin-1"],
            [{"name": "wlan0", "ipv4_address": None, "is_up": False}],
            token="tok",
        )

        _, payload = fake_client.mqtt.published[0]
        assert payload["message"] == "Network interfaces: none detected"


class TestAsyncFanOut:
    """The caller is the REST keepalive thread, whose cadence powers the
    liveness pill -- it must never wait on an MQTT connect."""

    def test_does_not_block_the_caller(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import threading
        import time

        release = threading.Event()
        fake_client = _FakeClient()

        def _slow_connect(token):  # type: ignore[no-untyped-def]
            release.wait(timeout=5)
            return fake_client

        monkeypatch.setattr(startup, "_get_shared_mqtt_client", _slow_connect)

        started = time.monotonic()
        thread = driver_logs.publish_network_facts_log_async(["twin-1"], NICS_A, token="tok")
        elapsed = time.monotonic() - started

        assert elapsed < 0.5, "keepalive thread waited on the MQTT connect"
        release.set()
        assert thread is not None
        thread.join(timeout=5)
        assert len(fake_client.mqtt.published) == 1

    def test_skips_tick_when_an_attempt_is_already_in_flight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A connect can outlast the 30 s keepalive period; workers must not
        pile up while the broker is unreachable."""
        import threading

        release = threading.Event()
        fake_client = _FakeClient()

        def _slow_connect(token):  # type: ignore[no-untyped-def]
            release.wait(timeout=5)
            return fake_client

        monkeypatch.setattr(startup, "_get_shared_mqtt_client", _slow_connect)

        first = driver_logs.publish_network_facts_log_async(["twin-1"], NICS_A, token="tok")
        second = driver_logs.publish_network_facts_log_async(["twin-1"], NICS_A, token="tok")

        assert first is not None
        assert second is None, "second tick spawned a concurrent worker"

        release.set()
        first.join(timeout=5)

        # The lock is released, so a later tick can run again.
        third = driver_logs.publish_network_facts_log_async(["twin-1"], NICS_B, token="tok")
        assert third is not None
        third.join(timeout=5)
        assert len(fake_client.mqtt.published) == 2
