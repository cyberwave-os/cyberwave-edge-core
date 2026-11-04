"""Fixtures shared by the edge-core unit tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_worker_startup_cards() -> object:
    """Drop every ``worker_starting`` card between tests.

    The card is module-level state keyed by container name (see
    ``worker_manager._StartupCard``) precisely so that the manager which opens
    it need not be the one that closes it. The flip side is that it outlives a
    ``WorkerManager``, and several test modules build managers for the same
    environment uuid — and so the same container name. Without this, a card
    left open by one test is adopted by the next.
    """
    from cyberwave_edge_core import worker_manager  # noqa: PLC0415

    worker_manager._STARTUP_CARDS.clear()
    yield
    worker_manager._STARTUP_CARDS.clear()


@pytest.fixture(autouse=True)
def _legacy_driver_tokens_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep launch tests offline: scoped driver tokens call the backend.

    Tests of the scoped path (test_driver_credentials.py) set the mode back.
    """
    from cyberwave_edge_core import driver_credentials  # noqa: PLC0415

    monkeypatch.delenv(driver_credentials.MODE_ENV, raising=False)
    monkeypatch.setattr(driver_credentials, "DEFAULT_MODE", driver_credentials.MODE_LEGACY)


@pytest.fixture(autouse=True)
def _closed_cloud_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with the backend reachable and no degraded or flap state.

    The breaker is module-level: a test whose fake backend raises a network
    error would otherwise pause the cloud steps of the tests after it.
    """
    from cyberwave_edge_core import startup  # noqa: PLC0415

    monkeypatch.setattr(startup, "_cloud_paused_until", 0.0)
    monkeypatch.setattr(startup, "_cloud_backoff_seconds", 0.0)
    monkeypatch.setattr(startup, "_cloud_unreachable", False)
    monkeypatch.setattr(startup, "_DEGRADED_STARTUP_PENDING", False)
    monkeypatch.setattr(startup, "_last_degraded_retry_at", None)
    monkeypatch.setattr(startup, "_shared_mqtt_was_connected", None)
    monkeypatch.setattr(startup, "_FLAP_STOPPED", {})
