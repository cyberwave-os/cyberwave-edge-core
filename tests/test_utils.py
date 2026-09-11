"""Tests for ``cyberwave_edge_core.utils``."""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Any
from unittest.mock import MagicMock


def _force_real_cyberwave_sdk() -> Any:
    """Purge any bare-stub ``cyberwave`` module installed by earlier
    tests (e.g. ``test_twin_file_sync.py`` replaces it with a
    ``types.ModuleType`` to avoid loading the real SDK) and re-import
    the on-disk SDK so ``cyberwave.alerts`` is reachable here.

    A ``types.ModuleType`` has no ``__file__``; the real package
    always does — so that is the cheapest signal to tell them apart.
    """
    for name in list(sys.modules):
        if name == "cyberwave" or name.startswith("cyberwave."):
            module = sys.modules[name]
            if not getattr(module, "__file__", None):
                del sys.modules[name]
    return importlib.import_module("cyberwave.alerts")


sdk_alerts = _force_real_cyberwave_sdk()

import cyberwave_edge_core.startup as startup  # noqa: E402
import cyberwave_edge_core.utils as utils  # noqa: E402


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


# ---------------------------------------------------------------------------
# EdgeCoreRestartAlertContext
# ---------------------------------------------------------------------------


def _install_restart_alert_stubs(
    monkeypatch, *, existing_metadata: dict | None = None
) -> tuple[MagicMock, MagicMock]:
    """Wire up the SDK indirections that
    :class:`EdgeCoreRestartAlertContext` reaches through (``load_token``,
    ``get_runtime_env_var``, ``Cyberwave``, ``_get_alert``) and return the
    ``mock_alert_class`` (which yields one alert instance whose
    ``update``/``resolve`` we assert on) plus the ``mock_get_alert``.
    """
    monkeypatch.setattr(startup, "load_token", lambda: "tok")
    monkeypatch.setattr(startup, "get_runtime_env_var", lambda *_a, **_kw: "https://api.test")
    monkeypatch.setattr(utils, "Cyberwave", lambda *_a, **_kw: object())

    mock_get_alert = MagicMock(
        return_value={"uuid": "alert-1", "metadata": existing_metadata or {}}
    )
    monkeypatch.setattr(sdk_alerts, "_get_alert", mock_get_alert)

    mock_alert_instance = MagicMock()
    mock_alert_instance.metadata = existing_metadata or {}
    mock_alert_class = MagicMock(return_value=mock_alert_instance)
    monkeypatch.setattr(sdk_alerts, "Alert", mock_alert_class)

    return mock_alert_class, mock_get_alert


def test_restart_alert_transition_no_uuid_is_noop(monkeypatch):
    """``alert_uuid=None`` is the contract for CLI-driven restarts that
    bypass the backend.  The context must not touch the SDK at all."""
    sentinel = MagicMock()
    monkeypatch.setattr(sdk_alerts, "_get_alert", sentinel)

    ctx = utils.EdgeCoreRestartAlertContext(alert_uuid=None)
    ctx.transition("in_progress", resolve=False)
    ctx.transition("completed", resolve=True)

    assert sentinel.call_count == 0


def test_restart_alert_transition_merges_metadata_and_resolves(monkeypatch):
    """``transition(..., resolve=True)`` must (1) merge ``phase`` and
    ``extra_metadata`` on top of existing metadata so the audit trail
    grows rather than gets clobbered, and (2) call ``resolve()`` on
    terminal phases."""
    mock_alert_class, _ = _install_restart_alert_stubs(
        monkeypatch,
        existing_metadata={
            "phase": "in_progress",
            "request_id": "req-1",
            "in_progress_at": 100.0,
        },
    )

    ctx = utils.EdgeCoreRestartAlertContext(alert_uuid="alert-1")
    ctx.transition("completed", resolve=True, extra_metadata={"completed_at": 200.0})

    mock_alert = mock_alert_class.return_value
    mock_alert.update.assert_called_once()
    update_kwargs = mock_alert.update.call_args.kwargs
    assert update_kwargs["metadata"] == {
        "phase": "completed",
        "request_id": "req-1",
        "in_progress_at": 100.0,
        "completed_at": 200.0,
    }
    mock_alert.resolve.assert_called_once()


def test_restart_alert_transition_in_progress_does_not_resolve(monkeypatch):
    """Non-terminal transitions must update metadata but leave the alert
    in ``active`` status — the reaper relies on this to detect stuck
    restarts."""
    mock_alert_class, _ = _install_restart_alert_stubs(
        monkeypatch, existing_metadata={"phase": "requested", "request_id": "req-1"}
    )

    ctx = utils.EdgeCoreRestartAlertContext(alert_uuid="alert-1")
    ctx.transition("in_progress", resolve=False)

    mock_alert = mock_alert_class.return_value
    mock_alert.update.assert_called_once()
    mock_alert.resolve.assert_not_called()


def test_restart_alert_transition_swallows_404(monkeypatch, caplog):
    """A 404 (operator deleted the alert, or reaper got there first)
    must log at debug and return silently — never raise into the
    restart loop."""
    monkeypatch.setattr(startup, "load_token", lambda: "tok")
    monkeypatch.setattr(startup, "get_runtime_env_var", lambda *_a, **_kw: "https://api.test")
    monkeypatch.setattr(utils, "Cyberwave", lambda *_a, **_kw: object())

    def _raise_404(*_a, **_kw) -> Any:
        raise type("ApiErr", (Exception,), {"status_code": 404})("gone")

    monkeypatch.setattr(sdk_alerts, "_get_alert", _raise_404)

    ctx = utils.EdgeCoreRestartAlertContext(alert_uuid="alert-1")
    with caplog.at_level(logging.DEBUG, logger=utils.logger.name):
        ctx.transition("completed", resolve=True)

    # Debug line should mention the alert is gone; nothing should have
    # propagated up.
    assert any("no longer exists" in r.getMessage() for r in caplog.records)


def test_restart_alert_transition_swallows_generic_errors(monkeypatch, caplog):
    """Non-404 errors must be logged at warning and swallowed.  A failed
    telemetry update can never block the actual restart."""
    monkeypatch.setattr(startup, "load_token", lambda: "tok")
    monkeypatch.setattr(startup, "get_runtime_env_var", lambda *_a, **_kw: "https://api.test")
    monkeypatch.setattr(utils, "Cyberwave", lambda *_a, **_kw: object())

    def _raise_boom(*_a, **_kw) -> Any:
        raise RuntimeError("backend unhappy")

    monkeypatch.setattr(sdk_alerts, "_get_alert", _raise_boom)

    ctx = utils.EdgeCoreRestartAlertContext(alert_uuid="alert-1")
    with caplog.at_level(logging.WARNING, logger=utils.logger.name):
        ctx.transition("completed", resolve=True)

    assert any(
        r.levelno == logging.WARNING
        and "Could not transition edge_core_restart alert" in r.getMessage()
        for r in caplog.records
    )


class TestStageProgressPercent:
    """The per-stage 0–100 published as ``stage_progress_percent``."""

    def test_download_stage_tracks_the_pull(self):
        assert utils._compute_stage_progress_percent("downloading", 0.0) == 0
        assert utils._compute_stage_progress_percent("downloading", 43.0) == 43
        assert utils._compute_stage_progress_percent("downloading", 100.0) == 100

    def test_install_stage_tracks_layer_completion(self):
        """Extraction is layer-serial and slow; byte progress is pinned at
        100 % throughout it, so the stage must move on layer counts."""
        early = utils._compute_stage_progress_percent(
            "installing", 100.0, layers_complete=0, layers_total=14
        )
        mid = utils._compute_stage_progress_percent(
            "installing", 100.0, layers_complete=7, layers_total=14
        )
        done = utils._compute_stage_progress_percent(
            "installing", 100.0, layers_complete=14, layers_total=14
        )
        assert (early, mid, done) == (0, 50, 100)

    def test_install_stage_falls_back_when_counts_are_unknown(self):
        assert utils._compute_stage_progress_percent("installing", 100.0) == 50

    def test_install_stage_is_clamped(self):
        """Junk counts must not push the stage past 100."""
        assert (
            utils._compute_stage_progress_percent(
                "installing", 100.0, layers_complete=99, layers_total=14
            )
            == 100
        )

    def test_each_stage_ends_at_100(self):
        """A stage that finishes says so, so the number can be checked
        against the stage the alert copy names on screen."""
        assert utils._compute_stage_progress_percent("downloading", 100.0) == 100
        assert utils._compute_stage_progress_percent("pull_complete", 100.0) == 100
        assert utils._compute_stage_progress_percent("pull_stream_finished", 100.0) == 100
        assert (
            utils._compute_stage_progress_percent(
                "installing", 100.0, layers_complete=14, layers_total=14
            )
            == 100
        )
        assert utils._compute_stage_progress_percent("container_running", 0.0) == 100

    def test_container_stage_steps_forward(self):
        started = utils._compute_stage_progress_percent("starting_container", 0.0)
        created = utils._compute_stage_progress_percent("container_created", 0.0)
        running = utils._compute_stage_progress_percent("container_running", 0.0)
        assert 0 < started < created < running == 100

    def test_unknown_phase_publishes_nothing(self):
        assert utils._compute_stage_progress_percent("container_probe_unconfirmed", 0.0) is None


class TestOverallProgressPercent:
    """The lifecycle-wide 0–100 the UI's progress bar reads."""

    def test_never_walks_backwards_across_a_whole_startup(self):
        """CYB-3867: the workbench renders one bar with no stage
        segmentation and no ``key`` between phases, so a stage that restarts
        its own 0–100 animates as a rewind — the fill slid from 83 % back to
        empty when the last layer finished downloading and extraction had
        not started."""
        steps: list[int] = []
        for percent in (0.0, 43.0, 83.0, 100.0):
            steps.append(utils._compute_overall_progress_percent("downloading", percent))
        for complete in range(15):
            steps.append(
                utils._compute_overall_progress_percent(
                    "installing", 100.0, layers_complete=complete, layers_total=14
                )
            )
        for phase in (
            "pull_complete",
            "starting_container",
            "container_created",
            "container_running",
        ):
            steps.append(
                utils._compute_overall_progress_percent(
                    phase, 100.0, layers_complete=14, layers_total=14
                )
            )

        assert all(b >= a for a, b in zip(steps, steps[1:])), steps
        assert steps[0] == 0
        assert steps[-1] == 100

    def test_download_band_tracks_the_pull(self):
        assert utils._compute_overall_progress_percent("downloading", 0.0) == 0
        assert (
            utils._compute_overall_progress_percent("downloading", 100.0)
            == utils._DOWNLOAD_BAND_END
        )

    def test_install_band_tracks_layer_completion(self):
        early = utils._compute_overall_progress_percent(
            "installing", 100.0, layers_complete=0, layers_total=14
        )
        done = utils._compute_overall_progress_percent(
            "installing", 100.0, layers_complete=14, layers_total=14
        )
        assert (early, done) == (utils._DOWNLOAD_BAND_END, utils._INSTALL_BAND_END)

    def test_every_layer_completion_moves_the_bar(self):
        """CYB-3854: a 5-point install band left completions invisible
        through the longest part of the wait."""
        steps = [
            utils._compute_overall_progress_percent(
                "installing", 100.0, layers_complete=n, layers_total=14
            )
            for n in range(15)
        ]
        assert all(b - a >= 1 for a, b in zip(steps, steps[1:]))

    def test_unknown_phase_publishes_nothing(self):
        assert utils._compute_overall_progress_percent("container_probe_unconfirmed", 0.0) is None


class TestInjectProgressPercentages:
    """Both numbers land on the alert metadata, from one merged patch."""

    def test_publishes_both_percentages(self):
        merged = {
            "phase": "installing",
            "progress_percent": 99,
            "layers_complete": 7,
            "layers_total": 14,
        }
        utils._inject_progress_percentages(merged)
        assert merged["stage_progress_percent"] == 50
        assert merged["overall_progress_percent"] == 63

    def test_omits_both_on_a_phase_with_nothing_to_report(self):
        merged = {"phase": "container_probe_unconfirmed"}
        utils._inject_progress_percentages(merged)
        assert "stage_progress_percent" not in merged
        assert "overall_progress_percent" not in merged


class TestCoerceLayerCount:
    def test_reads_a_plain_count(self):
        assert utils._coerce_layer_count(14) == 14

    def test_rejects_junk_from_unvalidated_metadata(self):
        for junk in (None, "14", -1, 3.5, True):
            assert utils._coerce_layer_count(junk) == 0


class TestFailureCopyIsUserFacing:
    """Every phase a startup can fail on needs plain-language copy.

    ``mark_failed_and_resolve`` falls back to the raw technical detail for an
    unmapped phase, which puts docker daemon output on the user's card.
    """

    def test_every_phase_the_worker_fails_on_is_mapped(self):
        # Phases ``WorkerManager`` can close a ``worker_starting`` card with.
        for phase in (
            "container_start_failed",
            "pull_failed",
            "pull_timeout",
            "pull_oserror",
            "pull_unexpected",
        ):
            assert phase in utils._WORKER_FAILED_DESCRIPTIONS, phase

    def test_copy_never_leaks_the_technical_detail(self):
        for phase, copy in utils._WORKER_FAILED_DESCRIPTIONS.items():
            assert copy.endswith("."), phase
            assert "docker" not in copy.lower(), phase
