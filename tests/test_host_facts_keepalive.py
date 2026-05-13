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
    """Successive ticks call ``_post_host_facts_once``; the thread is a daemon."""
    calls: list[float] = []
    call_event = threading.Event()

    def fake_post(token: str) -> bool:
        calls.append(time.monotonic())
        # Two ticks is enough to prove "this fires on a period, not just once".
        # We don't tighten the count further because doing so would make the
        # test flaky on slow CI without any extra coverage.
        if len(calls) >= 2:
            call_event.set()
        return True

    monkeypatch.setattr(startup, "_post_host_facts_once", fake_post)

    # 50 ms period keeps the test under ~150 ms even on slow CI; the
    # production period (30 s) is the same code path with a different
    # constant, so we don't pay for it here.
    assert startup._start_host_facts_keepalive("test-token", period_seconds=0.05)

    assert call_event.wait(timeout=2.0), (
        f"keepalive thread did not tick twice within 2 s "
        f"(got {len(calls)} calls)"
    )

    thread = startup._HOST_FACTS_KEEPALIVE_THREAD
    assert thread is not None and thread.is_alive(), "thread must still be running"
    assert thread.daemon, "thread must be daemon so process exit doesn't block"


def test_start_keepalive_is_idempotent(monkeypatch):
    """Repeat calls don't spawn duplicate threads."""
    monkeypatch.setattr(startup, "_post_host_facts_once", lambda token: True)

    assert startup._start_host_facts_keepalive("test-token", period_seconds=0.5)
    first = startup._HOST_FACTS_KEEPALIVE_THREAD

    assert startup._start_host_facts_keepalive("test-token", period_seconds=0.5)
    second = startup._HOST_FACTS_KEEPALIVE_THREAD

    assert first is second, (
        "second call must reuse the running thread, not start a parallel one"
    )


def test_upload_on_startup_posts_once_and_starts_keepalive(monkeypatch):
    """Boot path = one foreground POST + background keepalive scheduled."""
    posts: list[str] = []

    def fake_post(token: str) -> bool:
        posts.append(token)
        return True

    monkeypatch.setattr(startup, "_post_host_facts_once", fake_post)

    result = startup._upload_host_facts_on_startup("test-token")

    assert result is True
    assert posts == ["test-token"], "boot path posts exactly once in the foreground"

    thread = startup._HOST_FACTS_KEEPALIVE_THREAD
    assert thread is not None and thread.is_alive(), (
        "boot path must also schedule the periodic keepalive"
    )
