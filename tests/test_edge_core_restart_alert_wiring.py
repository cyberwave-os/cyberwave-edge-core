"""Wiring tests for the ``edge_core_restart`` lifecycle alert.

These exercise the seam between the MQTT command handler
(:func:`startup._handle_edge_command_message`) and the background
worker (:func:`startup._run_edge_core_restart_worker`) — specifically:

* ``alert_uuid`` is extracted from the incoming MQTT payload and
  threaded through to the worker.
* On a clean restart the alert transitions to ``completed`` and is
  resolved.
* On a failed restart the alert transitions to ``failed`` (with the
  exception text in ``metadata.error``) and is resolved.
* Restart commands without an ``alert_uuid`` (CLI-direct publish, dev
  shell, smoke tests) continue to work — the worker degrades to a
  no-op for alert telemetry.

These tests deliberately stub out :func:`_perform_edge_core_restart`
itself so they stay focused on the alert wiring and do not have to
mock Docker, MQTT, or the worker container.  ``_perform_edge_core_restart``
has its own coverage in ``test_startup_core.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import cyberwave_edge_core.startup as startup
import cyberwave_edge_core.utils as utils


def _install_restart_stubs(monkeypatch, *, perform_side_effect=None):
    """Stub ``load_token`` + ``_perform_edge_core_restart`` so the
    worker thread runs synchronously without touching the real restart
    machinery.  Returns the ``perform`` mock so the test can assert on
    arguments."""
    monkeypatch.setattr(startup, "load_token", lambda: "tok")
    perform = MagicMock(side_effect=perform_side_effect)
    monkeypatch.setattr(startup, "_perform_edge_core_restart", perform)
    # Reset coalescing state so each test starts from a clean slate.
    startup._HANDLED_EDGE_COMMAND_REQUEST_IDS.clear()
    startup._EDGE_RESTART_IN_PROGRESS = False
    return perform


def _patch_restart_alert(monkeypatch):
    """Replace ``EdgeCoreRestartAlertContext`` with a mock instance so
    we can assert on the exact sequence of phase transitions without
    going through the SDK."""
    fake_ctx = MagicMock(spec=utils.EdgeCoreRestartAlertContext)
    fake_ctx.alert_uuid = None
    monkeypatch.setattr(startup, "EdgeCoreRestartAlertContext", MagicMock(return_value=fake_ctx))
    return fake_ctx


def test_handle_edge_command_extracts_alert_uuid_and_threads_it_through(monkeypatch):
    perform = _install_restart_stubs(monkeypatch)
    _patch_restart_alert(monkeypatch)

    captured: dict = {}
    real_thread = startup.threading.Thread

    def _capture_thread(*, target, args, name, daemon):  # type: ignore[no-untyped-def]
        captured["target"] = target
        captured["args"] = args
        # Run inline so we can synchronously assert on the worker's
        # behaviour without juggling timing.
        return real_thread(target=lambda: target(*args), name=name, daemon=daemon)

    monkeypatch.setattr(startup.threading, "Thread", _capture_thread)

    startup._handle_edge_command_message(
        "edges/abc/command",
        {
            "command": "restart_edge_core",
            "request_id": "req-123",
            "alert_uuid": "alert-xyz",
        },
    )

    # The handler must have spawned a worker carrying both ids.
    assert captured["args"] == ("req-123", "alert-xyz")
    # Constructor was called with the same alert_uuid we received.
    startup.EdgeCoreRestartAlertContext.assert_called_with(alert_uuid="alert-xyz")
    # Restart was actually invoked.
    perform.assert_called_once()


def test_worker_transitions_completed_on_success(monkeypatch):
    _install_restart_stubs(monkeypatch)
    fake_ctx = _patch_restart_alert(monkeypatch)

    startup._run_edge_core_restart_worker("req-1", "alert-1")

    # Terminal transition must be ``completed`` with ``resolve=True``;
    # no ``failed`` transition should have happened.
    phases = [call.args[0] for call in fake_ctx.transition.call_args_list]
    assert "completed" in phases
    assert "failed" not in phases
    completed_call = next(c for c in fake_ctx.transition.call_args_list if c.args[0] == "completed")
    assert completed_call.kwargs["resolve"] is True
    assert "completed_at" in completed_call.kwargs["extra_metadata"]


def test_worker_transitions_failed_on_exception(monkeypatch):
    _install_restart_stubs(monkeypatch, perform_side_effect=RuntimeError("driver pull failed"))
    fake_ctx = _patch_restart_alert(monkeypatch)

    startup._run_edge_core_restart_worker("req-2", "alert-2")

    phases = [call.args[0] for call in fake_ctx.transition.call_args_list]
    assert "failed" in phases
    assert "completed" not in phases
    failed_call = next(c for c in fake_ctx.transition.call_args_list if c.args[0] == "failed")
    assert failed_call.kwargs["resolve"] is True
    extra = failed_call.kwargs["extra_metadata"]
    assert "driver pull failed" in extra["error"]
    assert "failed_at" in extra


def test_worker_no_alert_uuid_still_runs_restart(monkeypatch):
    """Restart commands published directly (CLI, smoke test) carry no
    ``alert_uuid``; the worker must still run the restart and the
    alert context must not raise (it degrades to a no-op)."""
    perform = _install_restart_stubs(monkeypatch)
    fake_ctx = _patch_restart_alert(monkeypatch)

    startup._run_edge_core_restart_worker("req-3", None)

    perform.assert_called_once()
    # ``transition`` is still called on the context — but the underlying
    # context with ``alert_uuid=None`` short-circuits.  The wiring test
    # owns making sure we *attempt* the transition; the no-op behaviour
    # itself is covered in ``test_utils.test_restart_alert_transition_no_uuid_is_noop``.
    assert fake_ctx.transition.called


def test_handler_ignores_unknown_command(monkeypatch):
    """Sanity: a non-restart command on the edge topic must not spawn
    a worker or touch the alert context."""
    perform = _install_restart_stubs(monkeypatch)
    fake_ctx_cls = MagicMock()
    monkeypatch.setattr(startup, "EdgeCoreRestartAlertContext", fake_ctx_cls)

    startup._handle_edge_command_message(
        "edges/abc/command",
        {"command": "some_future_command", "request_id": "req-4"},
    )

    perform.assert_not_called()
    fake_ctx_cls.assert_not_called()


def test_worker_marks_concurrent_drop_as_failed(monkeypatch):
    """A second restart that arrives while one is already in progress
    must not orphan its alert in ``requested`` (where it would block
    the workbench banner for the full 5-min reaper window).  Instead
    the worker transitions the dropped alert to ``failed`` with
    ``metadata.reason='concurrent_restart'`` and resolves it.
    """
    perform = _install_restart_stubs(monkeypatch)
    fake_ctx = _patch_restart_alert(monkeypatch)

    # Simulate "another restart is already running" by setting the
    # in-process flag before invoking the worker.
    startup._EDGE_RESTART_IN_PROGRESS = True
    try:
        startup._run_edge_core_restart_worker("req-dup", "alert-dup")
    finally:
        startup._EDGE_RESTART_IN_PROGRESS = False

    # The real restart must NOT have been triggered (the original is
    # still running), and the dropped alert must be in a terminal
    # ``failed`` state with the concurrency reason recorded.
    perform.assert_not_called()
    fake_ctx.transition.assert_called_once()
    call = fake_ctx.transition.call_args
    assert call.args[0] == "failed"
    assert call.kwargs["resolve"] is True
    assert call.kwargs["extra_metadata"]["reason"] == "concurrent_restart"
    assert "failed_at" in call.kwargs["extra_metadata"]


def test_worker_timestamps_are_iso_strings(monkeypatch):
    """All ``Alert.metadata`` timestamps written by the worker must be
    ISO-8601 strings, not unix-epoch floats — the backend writes ISO
    strings for ``requested_at`` and ``timed_out_at``, and mixing the
    two shapes in one metadata bag forces every downstream reader to
    handle both.  This test guards against regression to ``time.time()``.
    """
    _install_restart_stubs(monkeypatch)
    fake_ctx = _patch_restart_alert(monkeypatch)

    startup._run_edge_core_restart_worker("req-iso", "alert-iso")

    completed_call = next(c for c in fake_ctx.transition.call_args_list if c.args[0] == "completed")
    completed_at = completed_call.kwargs["extra_metadata"]["completed_at"]
    assert isinstance(completed_at, str)
    # ISO-8601 looks like ``2026-05-17T20:00:00.123456+00:00`` — assert
    # on the cheap structural markers rather than parsing it back, so
    # the test does not pin a specific microsecond format.
    assert "T" in completed_at
    assert completed_at.endswith("+00:00") or completed_at.endswith("Z")
