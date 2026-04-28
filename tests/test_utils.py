"""Tests for ``cyberwave_edge_core.utils``."""

from __future__ import annotations

import logging

import cyberwave_edge_core.startup as startup
import cyberwave_edge_core.utils as utils


def test_resolve_active_for_twin_404_logs_info_without_traceback(monkeypatch, caplog):
    """A deleted twin (404 from the backend) must produce a single INFO line,
    not a debug traceback, and return 0."""
    monkeypatch.setattr(startup, "load_token", lambda: "test-token")
    monkeypatch.setattr(startup, "get_runtime_env_var", lambda *_a, **_kw: "https://api.test")

    class _Client:
        def __init__(self, *_a, **_kw): ...
        def twin(self, *, twin_id):  # noqa: ARG002
            err = type("ApiErr", (Exception,), {"status_code": 404})("not found")
            raise err

    monkeypatch.setattr(utils, "Cyberwave", _Client)

    with caplog.at_level(logging.DEBUG, logger=utils.logger.name):
        resolved = utils.DriverStartingAlertContext.resolve_active_for_twin("dead-twin")

    assert resolved == 0
    assert any(
        r.levelno == logging.INFO and "twin no longer exists" in r.getMessage()
        for r in caplog.records
    )
    assert not any(r.exc_info for r in caplog.records)