"""Edge Core worker lifecycle integration tests against real Docker (CYB-2139).

Tier 1 coverage: exercises ``WorkerManager``, ``reconcile_worker_lifecycle``,
``WorkerWatcher``, ``_handle_twin_command_message``, and
``_run_remove_workflow_worker`` against the real local Docker daemon
using a session-built stub image. No backend, no MQTT broker, no
``cyberwaveos/edge-ml-worker`` pull.

Each phase from the ticket is a separate ``test_phase_NN_*`` function so a
failure report identifies exactly which invariant regressed. Every test is
decorated ``@requires_docker`` and skipped (not failed) when Docker is
unavailable, mirroring the gating in ``tests/test_systemd_notify_integration.py``.

What the unit suite already covers (and this module deliberately does not
duplicate): env-var resolution, image tag mutability matrix, GPU/Hailo
arg construction, monitor circuit-breaker arithmetic, watcher hash
algorithm. See ``test_worker_manager.py``, ``test_worker_health.py``,
``test_worker_watcher.py``, ``test_worker_lifecycle_reconcile.py``,
``test_startup_remove_workflow_worker.py``.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, List
from unittest.mock import MagicMock

import pytest

import cyberwave_edge_core.driver_logs as driver_logs_module
import cyberwave_edge_core.startup as startup
import cyberwave_edge_core.worker_manager as worker_manager_module
from cyberwave_edge_core.docker_helpers import (
    docker_container_status,
    docker_inspect,
)
from cyberwave_edge_core._clock import FakeMonotonicClock, set_now_monotonic
from cyberwave_edge_core.worker_manager import WorkerManager
from cyberwave_edge_core.worker_watcher import WorkerWatcher

from .conftest import (
    STUB_IMAGE_IMMUTABLE,
    STUB_IMAGE_IT,
    STUB_IMAGE_MUTABLE,
    requires_docker,
    wait_until,
)

# All tests in this module carry the ``docker`` marker so CI / operators
# can select or deselect the suite explicitly (``pytest -m docker`` /
# ``-m "not docker"``). Per-test ``@requires_docker`` is the runtime skip
# gate for hosts where the daemon is unreachable — the marker alone is
# not enough; it does not imply ``skipif``.
pytestmark = pytest.mark.docker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reconcile_summary(*, written: int = 0, removed: int = 0) -> dict[str, int]:
    """Build the ``sync_summary`` dict that ``reconcile_worker_lifecycle`` expects."""
    return {
        "written": written,
        "removed": removed,
        "unchanged": 0,
        "errors": 0,
    }


def _write_worker_file(workers_dir: Path, name: str, body: str = "pass\n") -> Path:
    """Write a worker stub file and return its path.

    ``name`` should follow the ``wf_<hex>.py`` shape so the
    ``remove_workflow_worker`` regex accepts it in Phase 5/9.
    """
    target = workers_dir / name
    target.write_text(body, encoding="utf-8")
    return target


def _make_watcher(
    workers_dir: Path,
    mgr: WorkerManager,
    *,
    cooldown_seconds: float = 0.0,
) -> WorkerWatcher:
    """Create a ``WorkerWatcher`` wired to *mgr* with a no-op model manager.

    The integration suite never exercises ``ModelManager`` end-to-end —
    its behaviour is covered by ``test_worker_watcher.py`` — so a
    ``MagicMock`` returning an empty model list is sufficient.
    """
    model_manager = MagicMock()
    model_manager.scan_worker_model_ids.return_value = []
    return WorkerWatcher(
        workers_dir=workers_dir,
        worker_manager=mgr,
        model_manager=model_manager,
        min_restart_interval_seconds=cooldown_seconds,
    )


def _container_id(name: str) -> str | None:
    """Return the container's Docker ``Id``, or None when missing."""
    data = docker_inspect(name)
    if data is None:
        return None
    cid = data.get("Id")
    return str(cid) if cid else None


def _worker_filename_for_workflow(workflow_uuid: str) -> str:
    """Return the 12-hex ``wf_*.py`` name the backend publishes on deactivate.

    Matches ``_workflow_worker_filename_candidates`` in
    ``cyberwave-backend/src/app/api/workflows.py`` (first candidate).
    """
    hex_only = workflow_uuid.replace("-", "")
    return f"wf_{hex_only[:12]}.py"


def _simulate_workflow_activation(
    workers_dir: Path,
    workflow_uuid: str,
    *,
    body: str = "pass\n",
) -> Path:
    """Simulate activation: backend sync wrote ``wf_*.py``, then lifecycle reconcile.

    In production the file lands via ``reconcile_worker_sync`` (periodic or
    immediate after ``sync_workflows`` MQTT). Edge Core then calls
    ``reconcile_worker_lifecycle`` with ``written >= 1`` to start the worker.
    """
    path = _write_worker_file(
        workers_dir,
        _worker_filename_for_workflow(workflow_uuid),
        body,
    )
    startup.reconcile_worker_lifecycle(_reconcile_summary(written=1))
    return path


def _simulate_workflow_deactivation(
    workflow_uuid: str,
    *,
    request_id: str,
    worker_filenames: list[str] | None = None,
) -> None:
    """Simulate UI deactivate: backend publishes ``remove_workflow_worker`` MQTT.

    Uses the same dispatcher entry point as production
    (:func:`startup._handle_twin_command_message`).
    """
    filenames = worker_filenames or [_worker_filename_for_workflow(workflow_uuid)]
    startup._handle_twin_command_message(
        {
            "command": "remove_workflow_worker",
            "request_id": request_id,
            "workflow_uuid": workflow_uuid,
            "worker_filenames": filenames,
        }
    )


# ---------------------------------------------------------------------------
# Phase 1 — idle start
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_01_idle_start_no_files_no_pull_no_container(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty ``workers/`` → reconcile must not start, must not pull, must not create a container.

    Three independent observations pin the contract:

    * No ``WorkerManager.start()`` invocation (the only path that could
      pull or create a container).
    * No ``_pull_worker_image_with_progress`` invocation (defence in depth in case
      anyone reorders the call inside ``WorkerManager.start``).
    * No container present in Docker after the reconcile.
    """
    start_calls: List[None] = []
    original_start = WorkerManager.start

    def spy_start(self: WorkerManager) -> bool:
        start_calls.append(None)
        return original_start(self)

    monkeypatch.setattr(WorkerManager, "start", spy_start)

    pull_calls: List[str] = []
    original_pull = WorkerManager._pull_worker_image_with_progress

    def spy_pull(self: WorkerManager, image: str, timeout: int = 600) -> bool:
        pull_calls.append(image)
        return original_pull(self, image, timeout)

    monkeypatch.setattr(
        WorkerManager,
        "_pull_worker_image_with_progress",
        spy_pull,
    )

    startup.reconcile_worker_lifecycle(_reconcile_summary(removed=1))

    assert start_calls == [], (
        "reconcile called WorkerManager.start() with an empty workers dir; "
        f"calls: {len(start_calls)}"
    )
    assert pull_calls == [], (
        "reconcile pulled an image with an empty workers dir; "
        f"unexpected calls: {pull_calls}"
    )
    assert docker_container_status(worker_container_name) == "none", (
        "reconcile created a container even though no wf_*.py files exist"
    )


# ---------------------------------------------------------------------------
# Phase 2 — cold start with files
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_02_cold_start_with_files_reaches_running(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
) -> None:
    """A ``wf_*.py`` file + reconcile → container reaches ``running``."""
    workers_dir = configured_edge / "workers"
    _write_worker_file(workers_dir, "wf_demo.py")

    startup.reconcile_worker_lifecycle(_reconcile_summary(written=1))

    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description=f"container {worker_container_name} to reach running",
    )

    data = docker_inspect(worker_container_name)
    assert data is not None
    assert data.get("Name", "").lstrip("/") == worker_container_name


# ---------------------------------------------------------------------------
# Phase 3 — add second worker mid-run, restart in place
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_03_add_second_worker_restart_preserves_name(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
    make_worker_manager: Callable[..., WorkerManager],
) -> None:
    """Adding a second worker file → watcher restarts in place; name preserved, Id changes."""
    workers_dir = configured_edge / "workers"
    _write_worker_file(workers_dir, "wf_aaaaaaaaaaaa.py")

    mgr = make_worker_manager()
    assert mgr.start() is True
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="initial cold start to reach running",
    )
    initial_id = _container_id(worker_container_name)
    assert initial_id is not None

    watcher = _make_watcher(workers_dir, mgr, cooldown_seconds=0.0)
    watcher.reconcile_worker_files()

    time.sleep(0.05)  # ensure mtime granularity sees the new file
    _write_worker_file(workers_dir, "wf_bbbbbbbbbbbb.py")

    restarted = watcher.reconcile_worker_files()
    assert restarted is True, "watcher did not restart after second worker file appeared"

    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="container to reach running after watcher restart",
    )
    new_id = _container_id(worker_container_name)
    assert new_id is not None
    assert new_id != initial_id, (
        "container Id did not change across restart; "
        "WorkerManager.restart() must docker-rm and re-create"
    )

    data = docker_inspect(worker_container_name)
    assert data is not None
    assert data.get("Name", "").lstrip("/") == worker_container_name, (
        "container name changed across restart"
    )


# ---------------------------------------------------------------------------
# Phase 4 — remove all workers stops (not removes) the container
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_04_remove_all_workers_stops_but_does_not_rm(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
) -> None:
    """Emptying workers/ → reconcile stops via ``docker stop`` (container persists in ``exited``)."""
    workers_dir = configured_edge / "workers"
    wf = _write_worker_file(workers_dir, "wf_demo.py")

    startup.reconcile_worker_lifecycle(_reconcile_summary(written=1))
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="initial start to reach running",
    )

    wf.unlink()
    startup.reconcile_worker_lifecycle(_reconcile_summary(removed=1))

    wait_until(
        lambda: docker_container_status(worker_container_name) == "exited",
        timeout_s=15,
        description="container to transition to exited (not removed)",
    )

    data = docker_inspect(worker_container_name)
    assert data is not None, (
        "container should persist in 'exited' state for diagnostics; "
        "docker_inspect returned None which means it was removed"
    )


# ---------------------------------------------------------------------------
# Phase 5 — surgical remove via _run_remove_workflow_worker
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_05_surgical_remove_via_run_remove_workflow_worker(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
) -> None:
    """Unlinking one of two ``wf_*.py`` files → siblings keep the container running."""
    workers_dir = configured_edge / "workers"
    a = _write_worker_file(workers_dir, "wf_aaaaaaaaaaaa.py")
    b = _write_worker_file(workers_dir, "wf_bbbbbbbbbbbb.py")

    startup.reconcile_worker_lifecycle(_reconcile_summary(written=2))
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="initial start with two workers",
    )

    startup._run_remove_workflow_worker(
        {
            "command": "remove_workflow_worker",
            "request_id": "phase-5-req",
            "workflow_uuid": "a" * 32,
            "worker_filenames": ["wf_aaaaaaaaaaaa.py"],
        }
    )

    assert not a.exists(), "named worker file should be unlinked"
    assert b.exists(), "sibling worker file must remain"
    assert docker_container_status(worker_container_name) == "running", (
        "container must stay running while at least one wf_*.py file is present"
    )


# ---------------------------------------------------------------------------
# Phase 6 — symmetric restart after edge-core restart
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_06_symmetric_restart_after_edge_core_restart(
    configured_edge: Path,
    worker_env_uuid: str,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
    make_worker_manager: Callable[..., WorkerManager],
) -> None:
    """A fresh ``WorkerManager`` against an existing running container is a no-op start."""
    workers_dir = configured_edge / "workers"
    _write_worker_file(workers_dir, "wf_demo.py")

    mgr_a = make_worker_manager()
    assert mgr_a.start() is True
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="instance #1 to start the container",
    )
    initial_id = _container_id(worker_container_name)

    # Simulate process restart: discard mgr_a, construct a fresh manager
    # with the same environment_uuid (same container name) and call start().
    del mgr_a
    mgr_b = make_worker_manager()
    assert mgr_b.container_name == worker_container_name
    assert mgr_b.start() is True, "post-restart start should succeed against running container"

    # Container was already running → start() short-circuits without docker_rm/docker_run.
    assert docker_container_status(worker_container_name) == "running"
    assert _container_id(worker_container_name) == initial_id, (
        "post-restart start re-created the container; the running-short-circuit is broken"
    )


# ---------------------------------------------------------------------------
# Phase 7 — restart cooldown within the configured window
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_07_restart_cooldown_within_window(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
    make_worker_manager: Callable[..., WorkerManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rapid file changes inside the cool-down window must collapse to one restart.

    The cool-down clock is driven by a ``FakeMonotonicClock`` so Docker restart
    latency on slow CI runners does not consume the cooldown window before the
    burst loop runs.  Real Docker restarts still happen — only the
    ``now_monotonic()`` source used by ``WorkerWatcher`` is decoupled from wall
    time.
    """
    workers_dir = configured_edge / "workers"
    _write_worker_file(workers_dir, "wf_aaaaaaaaaaaa.py")

    mgr = make_worker_manager()
    assert mgr.start() is True
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="initial start to reach running",
    )

    fake_clock = FakeMonotonicClock(start=1000.0)
    set_now_monotonic(fake_clock)

    restart_calls: list[str] = []
    original_restart = mgr.restart

    def spy_restart(*, reason: str = "requested") -> bool:
        restart_calls.append(reason)
        return original_restart(reason=reason)

    monkeypatch.setattr(mgr, "restart", spy_restart)

    cooldown = 30.0  # large value; clock is fake so this costs nothing
    watcher = _make_watcher(workers_dir, mgr, cooldown_seconds=cooldown)
    assert watcher.reconcile_worker_files() is False  # baseline

    fake_clock.advance(0.05)
    _write_worker_file(workers_dir, "wf_bbbbbbbbbbbb.py")
    fired_first = watcher.reconcile_worker_files()
    assert fired_first is True, "first restart inside the cool-down window must fire"

    # Burst of changes well inside the cool-down window — all must be deferred.
    for i in range(3):
        fake_clock.advance(0.1)
        _write_worker_file(workers_dir, f"wf_cccccccccc{i:02d}.py")
        assert watcher.reconcile_worker_files() is False, (
            f"restart #{i + 2} fired inside cool-down window"
        )

    assert len(restart_calls) == 1, (
        f"expected exactly one restart inside cool-down window, got {restart_calls}"
    )

    # Advance the fake clock past the cool-down; the pending restart must fire.
    fake_clock.advance(cooldown + 0.5)
    fired_again = watcher.reconcile_worker_files()
    assert fired_again is True, "deferred restart must fire after cool-down expires"
    assert len(restart_calls) == 2, restart_calls

    set_now_monotonic(None)  # restore real monotonic clock for subsequent tests


# ---------------------------------------------------------------------------
# Phase 8 — image pull policy: mutable pulls, immutable short-circuits
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_08_image_pull_policy_mutable_vs_immutable(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
    make_worker_manager: Callable[..., WorkerManager],
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Immutable tag (``:v1``) skips ``docker pull``; mutable tag (``:local``) always attempts it.

    The mutable pull fails (the stub image is local-only) and falls back
    to the local copy via ``_pull_worker_image_with_progress``'s exception path; we
    only need to assert the attempt was made, not that it succeeded.

    When ``twin_uuids`` are set (as in the integration-test fixture) the pull
    goes through ``_pull_docker_image_with_progress_multi`` → Docker Engine
    API, not through ``subprocess.run``.  We therefore detect the attempt via
    two independent signals so the assertion is robust to either path:

    1. ``subprocess.run(['docker', 'pull', …])`` — the no-twin-uuid /
       subprocess-fallback path.
    2. ``sys.stderr`` — ``_broadcast_pull_event`` always prints
       ``"docker pull started for image <tag>"`` to stderr at the very
       beginning of every engine-API pull, before any network I/O.
    """
    workers_dir = configured_edge / "workers"
    _write_worker_file(workers_dir, "wf_demo.py")

    pull_invocations: list[str] = []
    original_pull_multi = driver_logs_module._pull_docker_image_with_progress_multi

    def spy_pull_multi(image: str, **kwargs: Any) -> Any:
        pull_invocations.append(image)
        return original_pull_multi(image, **kwargs)

    monkeypatch.setattr(
        driver_logs_module,
        "_pull_docker_image_with_progress_multi",
        spy_pull_multi,
    )

    # --- Immutable tag: pre-pulled by the session fixture's docker tag step.
    capfd.readouterr()  # Reset capture buffer.
    mgr_imm = make_worker_manager(image=STUB_IMAGE_IMMUTABLE)
    assert mgr_imm.start() is True
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="immutable-tag container to reach running",
    )
    immutable_pulls = [c for c in pull_invocations if STUB_IMAGE_IMMUTABLE in c]
    immutable_stderr = capfd.readouterr().err
    assert immutable_pulls == [], (
        f"immutable tag {STUB_IMAGE_IMMUTABLE} triggered docker pull (subprocess); "
        f"got {immutable_pulls}"
    )
    assert f"docker pull started for image {STUB_IMAGE_IMMUTABLE}" not in immutable_stderr, (
        f"immutable tag {STUB_IMAGE_IMMUTABLE} triggered docker pull (engine API); "
        f"stderr: {immutable_stderr[:500]}"
    )

    assert mgr_imm.stop() is True
    wait_until(
        lambda: docker_container_status(worker_container_name) in {"exited", "none"},
        timeout_s=15,
        description="immutable-tag container to stop before mutable run",
    )

    # --- Mutable tag: must attempt a docker pull every start, even when local copy exists.
    capfd.readouterr()  # Reset capture buffer.
    pull_invocations.clear()
    mgr_mut = make_worker_manager(image=STUB_IMAGE_MUTABLE)
    assert mgr_mut.start() is True
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="mutable-tag container to reach running",
    )
    mutable_pulls = [c for c in pull_invocations if STUB_IMAGE_MUTABLE in c]
    mutable_stderr = capfd.readouterr().err
    pull_attempted = len(mutable_pulls) >= 1 or f"docker pull started for image {STUB_IMAGE_MUTABLE}" in mutable_stderr
    assert pull_attempted, (
        f"mutable tag {STUB_IMAGE_MUTABLE} should trigger docker pull; "
        f"recorded subprocess pulls: {pull_invocations}, stderr: {mutable_stderr[:500]}"
    )


# ---------------------------------------------------------------------------
# Phase 9 — MQTT request_id dedupe AND function-level idempotence
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_09_remove_workflow_worker_dedupe_and_idempotence(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both behaviours pinned by the ticket:

    1. Dispatcher dedupe: ``_handle_twin_command_message`` drops second
       delivery with the same ``request_id`` — the worker function is
       invoked only once.
    2. Function-level idempotence: calling ``_run_remove_workflow_worker``
       directly a second time with no remaining files is a safe no-op.
    """
    workers_dir = configured_edge / "workers"
    target = _write_worker_file(workers_dir, "wf_aaaaaaaaaaaa.py")

    startup.reconcile_worker_lifecycle(_reconcile_summary(written=1))
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="container to be running before dedupe test",
    )

    spy_lock = threading.Lock()
    spy_invocations: list[dict] = []
    original_remove = startup._run_remove_workflow_worker

    def spy_remove(payload: dict) -> None:
        with spy_lock:
            spy_invocations.append(payload)
        return original_remove(payload)

    monkeypatch.setattr(startup, "_run_remove_workflow_worker", spy_remove)

    payload = {
        "command": "remove_workflow_worker",
        "request_id": "phase-9-req-dupe",
        "workflow_uuid": "a" * 32,
        "worker_filenames": ["wf_aaaaaaaaaaaa.py"],
    }

    startup._handle_twin_command_message(payload)
    # The spy appends *before* delegating, so observing the file removal
    # — which only happens inside ``original_remove`` — implies the spy
    # has already recorded the invocation. One wait is sufficient.
    wait_until(
        lambda: not target.exists(),
        timeout_s=10,
        description="dispatcher-scheduled remove to unlink wf_aaaaaaaaaaaa.py",
    )
    assert len(spy_invocations) == 1, (
        f"first dispatcher call should invoke worker exactly once, got {len(spy_invocations)}"
    )

    # Container should now be stopped because the worker file is gone.
    # Timer starts at file-unlink, but the actual docker stop is dispatched
    # only after reconcile_worker_lifecycle creates a WorkerManager and calls
    # docker_container_status + docker_stop — each a subprocess round-trip.
    # 30 s gives headroom on loaded CI runners without being unreasonably long.
    wait_until(
        lambda: docker_container_status(worker_container_name) == "exited",
        timeout_s=30,
        description="container to stop after surgical remove of last worker",
    )

    # Second delivery with same request_id → dedupe must drop it.
    startup._handle_twin_command_message(payload)
    time.sleep(0.5)  # give the dispatcher time to spawn a thread if dedupe fails
    assert len(spy_invocations) == 1, (
        f"duplicate request_id was not deduped; spy got {len(spy_invocations)} calls"
    )
    assert "phase-9-req-dupe" in startup._HANDLED_TWIN_COMMAND_REQUEST_IDS

    # Function-level idempotence: calling ``_run_remove_workflow_worker``
    # directly twice with no remaining files must be observably a no-op.
    # We use ``original_remove`` (not the dispatcher) so dedupe can't help
    # and any non-idempotence would be visible.
    direct_payload = {
        "command": "remove_workflow_worker",
        "request_id": "phase-9-req-direct",
        "workflow_uuid": "a" * 32,
        "worker_filenames": ["wf_aaaaaaaaaaaa.py"],
    }
    original_remove(direct_payload)
    original_remove(direct_payload)

    assert not target.exists(), "file should still be absent after idempotent calls"
    assert docker_container_status(worker_container_name) == "exited", (
        "container should remain stopped across idempotent calls; "
        "double-stop or accidental restart would change the state"
    )


# ---------------------------------------------------------------------------
# Phase 10 — health monitor record_stop wiring
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_10_health_monitor_record_stop_wiring(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
    make_worker_manager: Callable[..., WorkerManager],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``WorkerManager.stop()`` → ``monitor.record_stop()`` suppresses spontaneous-exit WARNING.

    The monitor is constructed without an ``expected_running_probe`` so
    ``record_stop`` is the only suppression channel — broken wiring would
    let the next ``check()`` log the crash-loop warning.
    """
    workers_dir = configured_edge / "workers"
    wf = _write_worker_file(workers_dir, "wf_demo.py")

    mgr = make_worker_manager(with_health_monitor=True, expected_running_probe=None)
    monitor = mgr.health_monitor
    assert monitor is not None

    startup._set_monitored_worker_manager(mgr)

    startup.reconcile_worker_lifecycle(_reconcile_summary(written=1))
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="monitored container to reach running",
    )

    # Prime the monitor's last-seen container status; a real watcher tick
    # would do this each reconcile cycle.
    monitor.check(container_status="running")
    assert monitor._last_container_status == "running"

    wf.unlink()
    startup.reconcile_worker_lifecycle(_reconcile_summary(removed=1))
    wait_until(
        lambda: docker_container_status(worker_container_name) == "exited",
        timeout_s=15,
        description="container to reach exited via reconcile",
    )

    # record_stop should have been invoked from WorkerManager.stop() and
    # seeded _last_container_status with "exited" so the next check()
    # cannot detect a running → exited transition.
    assert monitor._last_container_status == "exited", (
        "WorkerManager.stop() did not call monitor.record_stop(); "
        "the running→exited transition will be misreported as a crash"
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="cyberwave_edge_core.worker_health"):
        state = monitor.check(container_status="exited")

    crash_warnings = [
        r for r in caplog.records if "exited spontaneously" in r.getMessage()
    ]
    assert crash_warnings == [], (
        "spontaneous-exit WARNING was emitted after a deliberate stop; "
        f"records: {[r.getMessage() for r in caplog.records]}"
    )
    assert state.container_status == "exited"


# ---------------------------------------------------------------------------
# Phase 11 — health monitor positive crash detection
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_11_spontaneous_exit_emits_warning(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
    make_worker_manager: Callable[..., WorkerManager],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An external stop that bypasses ``WorkerManager.stop()`` MUST log a crash warning.

    Phase 10 pins the negative side (``record_stop`` suppresses the warning
    after a deliberate ``WorkerManager.stop()``). This is the matching
    positive side: stop the container out-of-band (``docker stop`` via
    subprocess, not ``mgr.stop``) so ``record_stop`` is never called, then
    the next ``monitor.check()`` must surface the ``"exited spontaneously"``
    WARNING — that line is what pages on-call and what bug-bash dashboards
    grep for.
    """
    workers_dir = configured_edge / "workers"
    _write_worker_file(workers_dir, "wf_demo.py")

    mgr = make_worker_manager(with_health_monitor=True, expected_running_probe=None)
    monitor = mgr.health_monitor
    assert monitor is not None

    startup._set_monitored_worker_manager(mgr)

    startup.reconcile_worker_lifecycle(_reconcile_summary(written=1))
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="monitored container to reach running",
    )

    monitor.check(container_status="running")
    assert monitor._last_container_status == "running"

    # Out-of-band stop — never touches ``mgr.stop()`` or ``record_stop``.
    # ``docker stop`` honours ``--restart unless-stopped`` (the manual-stop
    # flag), so the container stays exited and the monitor's check() can
    # observe a clean running→exited transition.
    subprocess.run(
        ["docker", "stop", worker_container_name],
        capture_output=True,
        check=True,
        timeout=20,
    )
    wait_until(
        lambda: docker_container_status(worker_container_name) == "exited",
        timeout_s=15,
        description="container to reach exited after external docker stop",
    )

    assert monitor._last_container_status == "running", (
        "monitor's bookkeeping changed without going through record_stop; "
        "external stops must not seed _last_container_status"
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="cyberwave_edge_core.worker_health"):
        # No explicit container_status — let check() query Docker. This is
        # what the periodic health-tick does in production.
        state = monitor.check()

    crash_warnings = [
        r for r in caplog.records if "exited spontaneously" in r.getMessage()
    ]
    assert len(crash_warnings) == 1, (
        "expected exactly one spontaneous-exit WARNING after an out-of-band "
        f"stop; got {len(crash_warnings)}: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    assert state.container_status == "exited"


# ---------------------------------------------------------------------------
# Phase 12 — sync_workflows MQTT command dispatch + dedupe
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_12_sync_workflows_dispatch_and_dedupe(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sync_workflows`` dispatches to ``_run_immediate_worker_sync`` and dedupes by request_id.

    Symmetric to Phase 9 for the other dedupe-eligible command. The real
    ``_run_immediate_worker_sync`` hits the backend via
    ``reconcile_worker_sync`` (and is exercised by unit tests), so the
    spy replaces it entirely — this phase only pins the dispatcher's
    routing + dedupe contract, which is the integration concern.
    """
    workers_dir = configured_edge / "workers"
    _write_worker_file(workers_dir, "wf_demo.py")

    startup.reconcile_worker_lifecycle(_reconcile_summary(written=1))
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="container to be running before sync_workflows dispatch",
    )

    spy_lock = threading.Lock()
    spy_invocations: list[None] = []

    def spy_sync() -> None:
        with spy_lock:
            spy_invocations.append(None)

    monkeypatch.setattr(startup, "_run_immediate_worker_sync", spy_sync)

    payload = {
        "command": "sync_workflows",
        "request_id": "phase-12-req-dupe",
    }

    startup._handle_twin_command_message(payload)
    # Threads spawned by the dispatcher are daemons and have a single
    # statement before the lock; 0.5 s is several orders of magnitude
    # over the spawn+append latency on any realistic host.
    time.sleep(0.5)
    assert len(spy_invocations) == 1, (
        f"first sync_workflows command should dispatch exactly once; "
        f"spy got {len(spy_invocations)} calls"
    )
    assert "phase-12-req-dupe" in startup._HANDLED_TWIN_COMMAND_REQUEST_IDS

    startup._handle_twin_command_message(payload)
    time.sleep(0.5)
    assert len(spy_invocations) == 1, (
        "duplicate sync_workflows request_id was not deduped; "
        f"spy got {len(spy_invocations)} calls"
    )

    # New request_id → must dispatch again (dedupe scoped per request_id,
    # not "one sync_workflows per process").
    fresh_payload = dict(payload, request_id="phase-12-req-fresh")
    startup._handle_twin_command_message(fresh_payload)
    time.sleep(0.5)
    assert len(spy_invocations) == 2, (
        "fresh request_id should dispatch a second time; "
        f"spy got {len(spy_invocations)} calls"
    )


# ---------------------------------------------------------------------------
# Phase 13 — _run_remove_workflow_worker fires watcher hot-reload
# ---------------------------------------------------------------------------


@requires_docker
def test_phase_13_remove_workflow_worker_triggers_hot_reload(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
    make_worker_manager: Callable[..., WorkerManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With an active watcher, surgical remove must call ``watcher.force_restart``.

    Phases 5 and 9 deliberately run without a registered watcher, so the
    ``_hot_reload_running_worker`` branch inside ``_run_remove_workflow_worker``
    is dormant — it returns at the ``watcher is None`` guard. This phase
    registers a real watcher, observes that the branch fires exactly once,
    and verifies the container is re-created (Id changes) so stale modules
    get evicted on the next start.
    """
    workers_dir = configured_edge / "workers"
    a = _write_worker_file(workers_dir, "wf_aaaaaaaaaaaa.py")
    b = _write_worker_file(workers_dir, "wf_bbbbbbbbbbbb.py")

    mgr = make_worker_manager()
    assert mgr.start() is True
    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="container to start with two worker files",
    )
    initial_id = _container_id(worker_container_name)
    assert initial_id is not None

    watcher = _make_watcher(workers_dir, mgr, cooldown_seconds=0.0)
    watcher.reconcile_worker_files()  # seed baseline hash
    startup._set_active_worker_watcher(watcher)

    force_restart_calls: list[str] = []
    original_force_restart = watcher.force_restart

    def spy_force_restart(*, reason: str) -> bool:
        force_restart_calls.append(reason)
        return original_force_restart(reason=reason)

    monkeypatch.setattr(watcher, "force_restart", spy_force_restart)

    startup._run_remove_workflow_worker(
        {
            "command": "remove_workflow_worker",
            "request_id": "phase-13-req",
            "workflow_uuid": "a" * 32,
            "worker_filenames": ["wf_aaaaaaaaaaaa.py"],
        }
    )

    assert not a.exists(), "named worker file should be unlinked"
    assert b.exists(), "sibling worker file must remain"

    assert force_restart_calls == ["remove-workflow-worker"], (
        f"watcher.force_restart should fire exactly once with the dispatch "
        f"reason; got {force_restart_calls}"
    )

    wait_until(
        lambda: docker_container_status(worker_container_name) == "running",
        timeout_s=15,
        description="container to be running again after hot reload",
    )
    new_id = _container_id(worker_container_name)
    assert new_id is not None
    assert new_id != initial_id, (
        "hot reload must docker-rm + docker-run; same container Id means "
        "stale wf_*.py modules are still loaded in the worker process"
    )


# ---------------------------------------------------------------------------
# Phase 14 — user activate / deactivate / reactivate cycle
# ---------------------------------------------------------------------------

# Stable workflow UUID for the whole test (dashed form, as in API/MQTT payloads).
_WORKFLOW_UUID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


@requires_docker
def test_phase_14_user_activate_deactivate_reactivate_cycle(
    configured_edge: Path,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
) -> None:
    """Ten activate → deactivate cycles without backend or broker.

    Maps production behaviour on each iteration:

    * **Activate** — :func:`_simulate_workflow_activation` (sync wrote
      ``wf_<hex>.py`` + ``reconcile_worker_lifecycle(written=1)``).
    * **Deactivate** — :func:`_simulate_workflow_deactivation` (MQTT
      ``remove_workflow_worker`` via the real dispatcher).

    Each deactivate uses a fresh ``request_id`` so dedupe cannot mask a
    broken repeat cycle.
    """
    workers_dir = configured_edge / "workers"
    worker_name = _worker_filename_for_workflow(_WORKFLOW_UUID_A)
    wf_path = workers_dir / worker_name
    cycles = 10

    assert docker_container_status(worker_container_name) == "none", (
        "precondition: no worker container before first activation"
    )

    for i in range(cycles):
        _simulate_workflow_activation(
            workers_dir,
            _WORKFLOW_UUID_A,
            body=f"pass\n# activation cycle {i}\n",
        )
        assert wf_path.name == worker_name
        wait_until(
            lambda: docker_container_status(worker_container_name) == "running",
            timeout_s=15,
            description=f"container running after activation cycle {i}",
        )

        _simulate_workflow_deactivation(
            _WORKFLOW_UUID_A,
            request_id=f"phase-14-deactivate-{i}",
        )
        wait_until(
            lambda: not wf_path.exists(),
            timeout_s=10,
            description=f"worker file removed after deactivate cycle {i}",
        )
        wait_until(
            lambda: docker_container_status(worker_container_name) == "exited",
            timeout_s=15,
            description=f"container exited after deactivate cycle {i}",
        )

    assert not wf_path.exists()
    assert docker_container_status(worker_container_name) == "exited"
