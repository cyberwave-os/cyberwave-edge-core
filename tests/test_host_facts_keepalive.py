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

    def fake_post(token: str) -> bool:
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
    monkeypatch.setattr(startup, "_post_host_facts_once", lambda token: True)

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

    def fake_post(token: str) -> bool:
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
