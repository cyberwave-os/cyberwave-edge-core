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
