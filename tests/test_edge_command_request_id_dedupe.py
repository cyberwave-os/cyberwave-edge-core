"""Regression tests for bounded ``request_id`` dedupe on edge restart commands.

PR #2563 replaced the unbounded ``set`` with ``deque(maxlen=1024)`` so a
long-lived edge cannot grow restart dedupe memory without bound. These tests
guard the MQTT handler contract: duplicates are dropped, and IDs evicted from
the FIFO can be accepted again.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import cyberwave_edge_core.startup as startup


def _install_handler_stubs(monkeypatch):
    monkeypatch.setattr(startup, "load_token", lambda: "tok")
    monkeypatch.setattr(startup, "_perform_edge_core_restart", MagicMock())
    startup._HANDLED_EDGE_COMMAND_REQUEST_IDS.clear()
    startup._EDGE_RESTART_IN_PROGRESS = False


def _count_restart_threads(monkeypatch) -> list[dict]:
    thread_calls: list[dict] = []

    class FakeThread:
        def __init__(self, *, target, args=(), name="", daemon=False):
            thread_calls.append({"target": target, "args": args})

        def start(self):
            pass

    monkeypatch.setattr(startup.threading, "Thread", FakeThread)
    return thread_calls


def test_edge_restart_dedupes_duplicate_request_id(monkeypatch):
    _install_handler_stubs(monkeypatch)
    thread_calls = _count_restart_threads(monkeypatch)

    payload = {
        "command": "restart_edge_core",
        "request_id": "edge-restart-dup-1",
    }
    startup._handle_edge_command_message("edges/abc/command", payload)
    startup._handle_edge_command_message("edges/abc/command", payload)

    assert len(thread_calls) == 1
    assert thread_calls[0]["args"] == ("edge-restart-dup-1", None)


def test_edge_restart_request_id_fifo_evicts_oldest_after_maxlen(monkeypatch):
    """After 1024 unique IDs, the oldest entry must fall out so a replay is accepted."""
    _install_handler_stubs(monkeypatch)
    thread_calls = _count_restart_threads(monkeypatch)

    maxlen = startup._HANDLED_EDGE_COMMAND_REQUEST_IDS.maxlen
    assert maxlen == 1024

    first_id = "edge-restart-evict-me"
    startup._handle_edge_command_message(
        "edges/abc/command",
        {"command": "restart_edge_core", "request_id": first_id},
    )
    assert len(thread_calls) == 1

    for index in range(maxlen):
        startup._handle_edge_command_message(
            "edges/abc/command",
            {
                "command": "restart_edge_core",
                "request_id": f"edge-restart-fill-{index:04d}",
            },
        )

    assert first_id not in startup._HANDLED_EDGE_COMMAND_REQUEST_IDS
    count_after_fill = len(thread_calls)

    startup._handle_edge_command_message(
        "edges/abc/command",
        {"command": "restart_edge_core", "request_id": first_id},
    )
    assert len(thread_calls) == count_after_fill + 1
    assert thread_calls[-1]["args"][0] == first_id
