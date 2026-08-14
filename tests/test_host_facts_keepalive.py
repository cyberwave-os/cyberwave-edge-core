"""Tests for the REST host-facts keepalive that powers ``Edge.last_seen_at``.

Kept deliberately small: the keepalive is one daemon thread + one
``_post_host_facts_once`` callable, and the contract is "the thread
calls the function on a period, idempotently". Anything beyond the
basic two assertions (it spawns + it ticks) is testing Python's
``threading.Event`` rather than our code.
"""

from __future__ import annotations

import threading
import time

import pytest

from cyberwave_edge_core import startup


@pytest.fixture(autouse=True)
def _reset_keepalive_singleton():
    """Stop and clear the module-level thread between tests.

    The keepalive uses module globals so a stray thread from one test
    would otherwise leak into the next and break the "single thread"
    invariant we're trying to assert.
    """
    yield
    startup._stop_host_facts_keepalive()


def test_start_keepalive_spawns_thread_that_ticks_repeatedly(monkeypatch):
    """Successive ticks call ``_post_host_facts_once``; the thread is a daemon.

    With ``fire_immediately=False`` we still get at least two ticks within
    two periods, proving the periodic loop is alive without conflating it
    with the bootstrap tick (covered separately below).
    """
    calls: list[float] = []
    call_event = threading.Event()

    def fake_post(token: str, watchdog=None) -> bool:
        calls.append(time.monotonic())
        if len(calls) >= 2:
            call_event.set()
        return True

    monkeypatch.setattr(startup, "_post_host_facts_once", fake_post)

    # 50 ms period keeps the test under ~150 ms even on slow CI; the
    # production period (30 s) is the same code path with a different
    # constant, so we don't pay for it here.
    assert startup._start_host_facts_keepalive(
        "test-token", period_seconds=0.05, fire_immediately=False
    )

    assert call_event.wait(timeout=2.0), (
        f"keepalive thread did not tick twice within 2 s "
        f"(got {len(calls)} calls)"
    )

    thread = startup._HOST_FACTS_KEEPALIVE_THREAD
    assert thread is not None and thread.is_alive(), "thread must still be running"
    assert thread.daemon, "thread must be daemon so process exit doesn't block"


def test_start_keepalive_is_idempotent(monkeypatch):
    """Repeat calls don't spawn duplicate threads."""
    monkeypatch.setattr(
        startup, "_post_host_facts_once", lambda token, watchdog=None: True
    )

    assert startup._start_host_facts_keepalive("test-token", period_seconds=0.5)
    first = startup._HOST_FACTS_KEEPALIVE_THREAD

    assert startup._start_host_facts_keepalive("test-token", period_seconds=0.5)
    second = startup._HOST_FACTS_KEEPALIVE_THREAD

    assert first is second, (
        "second call must reuse the running thread, not start a parallel one"
    )


def test_upload_on_startup_is_non_blocking_and_posts_from_thread(monkeypatch):
    """Boot path must not POST on the caller's stack.

    This is the contract that lets systemd ``Type=notify`` services
    signal ``READY=1`` quickly: ``run_startup_checks`` must return even
    when the backend is unreachable / slow. We assert that:

    1. ``_upload_host_facts_on_startup`` returns immediately (no I/O on
       the caller's stack — verified by patching ``_post_host_facts_once``
       to record which thread invoked it).
    2. The keepalive thread is the one that performs the bootstrap POST.
    """
    calling_threads: list[int] = []
    bootstrap_done = threading.Event()

    def fake_post(token: str, watchdog=None) -> bool:
        calling_threads.append(threading.get_ident())
        bootstrap_done.set()
        return True

    monkeypatch.setattr(startup, "_post_host_facts_once", fake_post)

    main_thread_id = threading.get_ident()
    result = startup._upload_host_facts_on_startup("test-token")

    assert result is True, "boot path returns True synchronously regardless of POST result"

    assert bootstrap_done.wait(timeout=2.0), (
        "bootstrap tick did not run within 2 s — keepalive thread is broken"
    )

    assert len(calling_threads) >= 1
    assert main_thread_id not in calling_threads, (
        "bootstrap POST must NOT execute on the caller's stack — systemd "
        "Type=notify relies on run_startup_checks returning before any "
        "network I/O completes"
    )

    thread = startup._HOST_FACTS_KEEPALIVE_THREAD
    assert thread is not None and thread.is_alive(), (
        "boot path must also schedule the periodic keepalive"
    )


class _FakeWatchdog:
    """Stands in for ``ProcessWatchdog``; only ``active_layers()`` is used."""

    def __init__(self, layers=None, raises=False):
        self._layers = layers if layers is not None else []
        self._raises = raises

    def active_layers(self):
        if self._raises:
            raise RuntimeError("watchdog probe exploded")
        return list(self._layers)


def _post_with(monkeypatch, watchdog, facts=None):
    """Run ``_post_host_facts_once`` against fakes; return the uploaded facts."""
    import sys
    import types

    fake_module = types.ModuleType("cyberwave.edge.host_metrics")

    class _Facts:
        def to_dict(self):
            return dict(facts if facts is not None else {"platform": "Linux"})

    fake_module.read_host_facts = lambda: _Facts()
    monkeypatch.setitem(sys.modules, "cyberwave.edge.host_metrics", fake_module)
    monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp-test")

    sent: dict = {}

    class _Edges:
        def discover(self, **kwargs):
            sent.update(kwargs)
            return {"twins": []}

    class _Client:
        def __init__(self, *a, **kw):
            self.edges = _Edges()

    monkeypatch.setattr(startup, "Cyberwave", _Client)

    startup._post_host_facts_once("test-token", watchdog=watchdog)
    return sent.get("host_facts", {})


def test_watchdog_layers_ride_the_rest_upload(monkeypatch):
    """Layers reach the dashboard over REST, not the ``edge_health`` heartbeat.

    The heartbeat publisher stops at the first driver start, so a static
    capability published only there disappears on exactly the edges that are
    doing work. ``read_host_facts()`` cannot supply this itself -- it probes
    ``/dev/watchdog`` and so knows the device exists, not what is pinging it.
    """
    facts = _post_with(monkeypatch, _FakeWatchdog(["systemd", "hardware"]))

    assert facts["watchdog_layers"] == ["systemd", "hardware"]


def test_watchdog_layers_omitted_when_no_layer_is_enabled(monkeypatch):
    """Absent, not ``[]``: "no layers enabled" and "edge predates this upload"
    are different states, and only the missing key can express the second --
    which is what lets the dashboard fall back to ``has_hardware_watchdog``."""
    facts = _post_with(monkeypatch, _FakeWatchdog([]))

    assert "watchdog_layers" not in facts


def test_watchdog_probe_failure_does_not_lose_the_upload(monkeypatch):
    """A broken watchdog must not cost us ``last_seen_at``.

    The POST is also the keepalive that holds the edge out of "Offline", so
    letting ``active_layers()`` propagate would trade one cosmetic field for
    the row's whole liveness signal.
    """
    facts = _post_with(monkeypatch, _FakeWatchdog(raises=True))

    assert "watchdog_layers" not in facts
    assert facts["platform"] == "Linux"


def test_no_watchdog_leaves_host_facts_untouched(monkeypatch):
    """One-shot CLI flows pass no watchdog and must still upload."""
    facts = _post_with(monkeypatch, None)

    assert "watchdog_layers" not in facts
    assert facts["platform"] == "Linux"
