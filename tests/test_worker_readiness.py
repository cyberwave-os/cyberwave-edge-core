"""Unit tests for worker readiness evaluation (worker_readiness.py).

``running`` is not the same fact as "can accept a run". These tests pin the
four ways that wait can end — ready, container death, timeout, and an old
worker image that never reports — because each closes the ``worker_starting``
alert differently, and two backend consumers read that alert to decide what to
tell the user and whether to fail their run.

``evaluate_readiness`` is a pure function, so none of this needs threads,
sleeps or wall-clock timing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cyberwave_edge_core.worker_readiness import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    STATUS_FILE_GRACE_SECONDS,
    alert_patch,
    clear_status,
    evaluate_readiness,
    read_status,
)


def record(phase: str, **extra: Any) -> dict[str, Any]:
    return {"version": 1, "phase": phase, **extra}


def test_resolves_without_failure_once_the_worker_reports_ready() -> None:
    outcome = evaluate_readiness(
        record("ready"), container_status="running", elapsed_seconds=30.0
    )

    assert outcome.close is True
    # No failure: a worker that came up is not a startup failure.
    assert outcome.failure == ""
    assert outcome.patch["phase"] == "ready"


@pytest.mark.parametrize(
    "phase",
    ["booting", "connecting_mqtt", "loading_models", "subscribed", "warming_up"],
)
def test_keeps_the_card_open_through_every_startup_phase(phase: str) -> None:
    outcome = evaluate_readiness(
        record(phase), container_status="running", elapsed_seconds=30.0
    )

    assert outcome.close is False
    assert outcome.patch["phase"] == phase


def test_forwards_model_counts_for_the_progress_bar() -> None:
    outcome = evaluate_readiness(
        record("warming_up", models_ready=2, models_total=5),
        container_status="running",
        elapsed_seconds=30.0,
    )

    assert outcome.patch["models_ready"] == 2
    assert outcome.patch["models_total"] == 5


@pytest.mark.parametrize("status", ["exited", "dead", "none"])
def test_fails_the_card_when_the_container_dies_mid_startup(status: str) -> None:
    outcome = evaluate_readiness(
        record("loading_models"), container_status=status, elapsed_seconds=30.0
    )

    assert outcome.close is True
    assert outcome.phase == "worker_startup_exited"
    # The phase reached before dying is the actionable part of the message.
    assert "loading_models" in outcome.failure


def test_a_dead_container_fails_even_before_the_worker_reported_anything() -> None:
    outcome = evaluate_readiness(None, container_status="exited", elapsed_seconds=5.0)

    assert outcome.close is True
    assert outcome.phase == "worker_startup_exited"
    assert "unknown" in outcome.failure


def test_fails_the_card_when_readiness_never_arrives() -> None:
    outcome = evaluate_readiness(
        record("warming_up"),
        container_status="running",
        elapsed_seconds=DEFAULT_READY_TIMEOUT_SECONDS + 1,
    )

    assert outcome.close is True
    assert outcome.phase == "worker_ready_timeout"
    assert "warming_up" in outcome.failure


def test_a_crash_looping_container_fails_the_card() -> None:
    """``restarting`` is not a dead state and every restart rewrites
    ``StartedAt``, so a crash loop clears both rules above it: the container
    never looks dead and elapsed never reaches the timeout. Docker's own
    restart tally is the evidence that survives."""
    outcome = evaluate_readiness(
        record("loading_models"),
        container_status="restarting",
        elapsed_seconds=3.0,
        restart_count=2,
    )

    assert outcome.close is True
    assert outcome.phase == "worker_startup_exited"
    assert "restarted 2x" in outcome.failure
    assert "loading_models" in outcome.failure


def test_a_crash_loop_before_the_worker_reported_anything_still_fails() -> None:
    outcome = evaluate_readiness(
        None, container_status="restarting", elapsed_seconds=3.0, restart_count=1
    )

    assert outcome.close is True
    assert outcome.phase == "worker_startup_exited"
    assert "unknown" in outcome.failure


def test_a_worker_that_came_up_after_one_restart_is_not_called_failed() -> None:
    """A container that OOMed once and succeeded on the retry is a worker that
    is ready, not a startup failure — so the ready check runs first."""
    outcome = evaluate_readiness(
        record("ready"), container_status="running", elapsed_seconds=45.0, restart_count=1
    )

    assert outcome.close is True
    assert outcome.failure == ""
    assert outcome.patch["phase"] == "ready"


def test_old_worker_image_resolves_instead_of_spinning_forever() -> None:
    """A worker image predating status reporting writes no file, and must fall
    back to "running means started" rather than hold the card to the timeout."""
    outcome = evaluate_readiness(
        None,
        container_status="running",
        elapsed_seconds=STATUS_FILE_GRACE_SECONDS + 1,
    )

    assert outcome.close is True
    assert outcome.failure == ""


def test_waits_out_the_grace_window_before_assuming_an_old_image() -> None:
    """A merely-slow interpreter start must not be mistaken for an old image."""
    outcome = evaluate_readiness(
        None,
        container_status="running",
        elapsed_seconds=STATUS_FILE_GRACE_SECONDS - 1,
    )

    assert outcome.close is False
    assert outcome.patch == {}


def test_a_worker_that_reports_late_is_not_held_against_the_grace_window() -> None:
    """Once any record exists the grace window is irrelevant — only the much
    longer readiness timeout applies."""
    outcome = evaluate_readiness(
        record("loading_models"),
        container_status="running",
        elapsed_seconds=STATUS_FILE_GRACE_SECONDS + 300,
    )

    assert outcome.close is False


@pytest.mark.parametrize(
    "content",
    ["", "not json", "[]", '{"no_phase": 1}', '{"phase": ""}'],
)
def test_read_status_tolerates_junk(tmp_path: Path, content: str) -> None:
    """A torn read must not raise: the next tick is seconds away."""
    status_path = tmp_path / "worker-status.json"
    status_path.write_text(content, encoding="utf-8")
    assert read_status(status_path) is None


def test_read_status_missing_file_is_none(tmp_path: Path) -> None:
    assert read_status(tmp_path / "absent.json") is None


def test_read_status_returns_the_record(tmp_path: Path) -> None:
    status_path = tmp_path / "worker-status.json"
    status_path.write_text(json.dumps(record("ready")), encoding="utf-8")
    assert read_status(status_path) == {"version": 1, "phase": "ready"}


def test_clear_status_is_idempotent(tmp_path: Path) -> None:
    status_path = tmp_path / "worker-status.json"
    status_path.write_text(json.dumps(record("ready")), encoding="utf-8")
    clear_status(status_path)
    assert not status_path.exists()
    clear_status(status_path)  # must not raise


def test_alert_patch_forwards_only_consumer_visible_fields() -> None:
    patch = alert_patch(
        {
            "phase": "warming_up",
            "models_ready": 2,
            "models_total": 5,
            "detail": "yolov8n",
            "pid": 41,
            "boot_id": "abc",
        }
    )
    assert patch == {
        "phase": "warming_up",
        "models_ready": 2,
        "models_total": 5,
        "detail": "yolov8n",
    }


def test_alert_patch_drops_malformed_counts() -> None:
    patch = alert_patch(
        {"phase": "warming_up", "models_ready": "two", "models_total": -1}
    )
    assert patch == {"phase": "warming_up"}
