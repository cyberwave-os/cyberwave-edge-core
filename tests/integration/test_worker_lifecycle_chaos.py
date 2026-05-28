"""Stateful property-based worker lifecycle chaos tests (CYB-2140).

Randomized interleavings of the same Edge Core surface exercised by
``test_worker_lifecycle_docker.py`` (Tier 1), with cross-cutting invariants
checked after every rule on a settled Docker state.

Requires Hypothesis (``pip install -e '.[dev]'``) and a reachable Docker daemon.
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Optional
from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

import cyberwave_edge_core.docker_helpers as docker_helpers
import cyberwave_edge_core.startup as startup
import cyberwave_edge_core.worker_manager as worker_manager_module
from cyberwave_edge_core._clock import FakeMonotonicClock, now_monotonic, set_now_monotonic
from cyberwave_edge_core.docker_helpers import docker_container_status, docker_inspect
from cyberwave_edge_core.worker_manager import WorkerManager
from cyberwave_edge_core.worker_watcher import WorkerWatcher

from .conftest import (
    STUB_IMAGE_IT,
    STUB_IMAGE_TAGS,
    requires_docker,
    wait_until,
)

pytestmark = pytest.mark.docker

# PR CI: deterministic, bounded runtime (CYB-2140 acceptance).
_CHAOS_SETTINGS = settings(
    max_examples=20,
    stateful_step_count=30,
    deadline=None,
    derandomize=True,
    # Real Docker + 30 steps per example can exceed default 200 ms/step;
    # suppress rather than flake on slow self-hosted runners.
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
    ],
)

_WORKFLOW_UUIDS = (
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "cccccccc-cccc-cccc-cccc-cccccccccccc",
)
_REQUEST_ID_POOL = ("chaos-req-0", "chaos-req-1", "chaos-req-2")
_WATCHER_COOLDOWN_S = 2.0


def _worker_filename(workflow_uuid: str) -> str:
    hex_only = workflow_uuid.replace("-", "")
    return f"wf_{hex_only[:12]}.py"


def _reconcile_summary(*, written: int = 0, removed: int = 0, errors: int = 0) -> dict[str, int]:
    return {
        "written": written,
        "removed": removed,
        "unchanged": 0,
        "errors": errors,
    }


def _list_worker_files(workers_dir: Path) -> list[str]:
    if not workers_dir.exists():
        return []
    return sorted(p.name for p in workers_dir.glob("wf_*.py") if p.is_file())


@dataclass
class ChaosContext:
    """Per-example shared state wired from pytest before Hypothesis runs."""

    config_dir: Path
    workers_dir: Path
    container_name: str
    env_uuid: str
    fake_clock: FakeMonotonicClock
    mgr: WorkerManager
    watcher: WorkerWatcher
    # docker_rm is called by _run_container (recreate) and restart; tracked to
    # distinguish from bare stop paths (invariant 2).
    docker_rm_calls_during_stop: int = 0
    immutable_pull_count: int = 0
    # Counts of calls to WorkerManager.stop() that found the container running —
    # each such stop must produce exactly one monitor.record_stop() call.
    stop_running_transitions: int = 0
    record_stop_count: int = 0
    last_sync_had_errors: bool = False
    watcher_restart_times: List[float] = field(default_factory=list)
    # Per-request-id dispatch counts for dedupe assertion (invariant 4).
    remove_dispatch_counts: dict[str, int] = field(default_factory=dict)
    _rm_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Set to True while a restart is in progress so the rm spy can
    # distinguish restart-rm from stop-only-rm.
    _in_restart: bool = field(default=False, repr=False)

    def wrap_docker_rm(self, name: str, *, timeout: int = 30) -> bool:
        if name == self.container_name and not self._in_restart:
            # rm fired outside a restart — should never happen on the stop path
            with self._rm_lock:
                self.docker_rm_calls_during_stop += 1
        original = getattr(self, "_original_docker_rm")
        return original(name, timeout=timeout)

    def wrap_subprocess_run(self, cmd: object, *args: object, **kwargs: object) -> object:
        original = getattr(self, "_original_subprocess_run")
        result = original(cmd, *args, **kwargs)
        if (
            isinstance(cmd, (list, tuple))
            and len(cmd) >= 2
            and cmd[0] == "docker"
            and cmd[1] == "pull"
            and any(STUB_IMAGE_IT in str(part) for part in cmd)
            and getattr(result, "returncode", 1) == 0
        ):
            self.immutable_pull_count += 1
        return result


class WorkerLifecycleChaosMachine(RuleBasedStateMachine):
    """Hypothesis state machine driving worker lifecycle under real Docker."""

    _ctx: ClassVar[Optional[ChaosContext]] = None

    worker_uuids = Bundle("workflow_uuid")
    # Populated by run_remove_workflow_worker; consumed by replay_remove_request.
    dispatched_request_ids = Bundle("dispatched_request_id")

    def __init__(self) -> None:
        super().__init__()
        if self._ctx is None:
            raise RuntimeError("ChaosContext not bound; pytest fixture missing")
        self.ctx = self._ctx
        self._last_restart_at_before_tick: Optional[float] = None

    # ------------------------------------------------------------------
    # Settlement (required after every rule; no sleep in invariants)
    # ------------------------------------------------------------------

    def _settle(self) -> None:
        """Drive lifecycle to a quiescent Docker state before invariants run."""
        files = _list_worker_files(self.ctx.workers_dir)
        # Settlement always reconciles physical state. ``last_sync_had_errors``
        # only gates production reconcile after a failed API sync (invariant 1).
        if files:
            startup.reconcile_worker_lifecycle(
                _reconcile_summary(written=max(1, len(files)))
            )
        else:
            startup.reconcile_worker_lifecycle(_reconcile_summary(removed=1))

        if files:
            wait_until(
                lambda: docker_container_status(self.ctx.container_name) == "running",
                timeout_s=15,
                description="chaos settle: container running while worker files exist",
            )
        else:
            wait_until(
                lambda: docker_container_status(self.ctx.container_name)
                in {"exited", "none"},
                timeout_s=15,
                description="chaos settle: container stopped when workers dir empty",
            )

    # ------------------------------------------------------------------
    # Rules (CYB-2140)
    # ------------------------------------------------------------------

    @rule(target=worker_uuids, workflow_uuid=st.sampled_from(_WORKFLOW_UUIDS))
    def add_worker_file(self, workflow_uuid: str) -> str:
        path = self.ctx.workers_dir / _worker_filename(workflow_uuid)
        path.write_text(f"pass\n# chaos add {workflow_uuid}\n", encoding="utf-8")
        self._settle()
        return workflow_uuid

    @rule(workflow_uuid=worker_uuids)
    def remove_worker_file(self, workflow_uuid: str) -> None:
        """Remove a file that was previously added via add_worker_file.

        Drawing from the bundle instead of ``st.sampled_from`` ensures
        Hypothesis only generates removes for UUIDs it actually created,
        making every step exercise a real state transition.
        """
        path = self.ctx.workers_dir / _worker_filename(workflow_uuid)
        if path.exists():
            path.unlink()
        self._settle()

    @rule(
        written=st.integers(0, 2),
        removed=st.integers(0, 2),
        errors=st.sampled_from([0, 1]),
    )
    def reconcile_lifecycle(self, written: int, removed: int, errors: int) -> None:
        self.ctx.last_sync_had_errors = errors > 0
        summary = _reconcile_summary(written=written, removed=removed, errors=errors)
        startup.reconcile_worker_lifecycle(summary)
        self._settle()

    @rule(
        target=dispatched_request_ids,
        request_id=st.sampled_from(_REQUEST_ID_POOL),
        workflow_uuid=st.sampled_from(_WORKFLOW_UUIDS),
    )
    def run_remove_workflow_worker(self, request_id: str, workflow_uuid: str) -> str:
        filename = _worker_filename(workflow_uuid)
        file_was_present = (self.ctx.workers_dir / filename).exists()
        already_seen = request_id in startup._HANDLED_TWIN_COMMAND_REQUEST_IDS

        startup._handle_twin_command_message(
            {
                "command": "remove_workflow_worker",
                "request_id": request_id,
                "workflow_uuid": workflow_uuid,
                "worker_filenames": [filename],
            }
        )

        # Track how many times each request_id was dispatched to the worker
        # function (post-dedupe). If dedupe drops it, count stays unchanged.
        if not already_seen:
            self.ctx.remove_dispatch_counts[request_id] = (
                self.ctx.remove_dispatch_counts.get(request_id, 0) + 1
            )

        # _handle_twin_command_message spawns a daemon thread; wait for the
        # file side-effect to land before calling _settle() so invariants
        # never observe an intermediate state.
        if file_was_present and not already_seen:
            wait_until(
                lambda: not (self.ctx.workers_dir / filename).exists(),
                timeout_s=10,
                description=f"MQTT remove thread to unlink {filename}",
            )

        self._settle()
        return request_id

    @rule(
        request_id=dispatched_request_ids,
        workflow_uuid=st.sampled_from(_WORKFLOW_UUIDS),
    )
    def replay_remove_request(self, request_id: str, workflow_uuid: str) -> None:
        """Re-deliver a request_id that has already been processed.

        Hypothesis draws ``request_id`` from the ``dispatched_request_ids``
        bundle, so this rule only fires after ``run_remove_workflow_worker``
        has succeeded at least once.  Dedupe must silently drop the replay —
        the dispatch count must not increase.
        """
        counts_before = dict(self.ctx.remove_dispatch_counts)
        startup._handle_twin_command_message(
            {
                "command": "remove_workflow_worker",
                "request_id": request_id,
                "workflow_uuid": workflow_uuid,
                "worker_filenames": [_worker_filename(workflow_uuid)],
            }
        )
        # Dispatch counts must be identical — dedupe absorbed the replay.
        assert self.ctx.remove_dispatch_counts == counts_before, (
            f"replay of request_id={request_id!r} incremented dispatch count; "
            "dedupe gate is broken"
        )
        self._settle()

    @rule()
    def worker_manager_restart(self) -> None:
        self.ctx.mgr.restart(reason="chaos-restart")
        self._settle()

    @rule()
    def restart_edge_core_process(self) -> None:
        old = self.ctx.mgr
        self.ctx.mgr = WorkerManager(
            config_dir=self.ctx.config_dir,
            environment_uuid=self.ctx.env_uuid,
            token="it-token",
            twin_uuids=["it-twin"],
            image=STUB_IMAGE_IT,
        )
        monitor = old.health_monitor
        if monitor is not None:
            self.ctx.mgr.set_health_monitor(monitor)
        self.ctx.watcher._worker_manager = self.ctx.mgr  # noqa: SLF001
        startup._set_monitored_worker_manager(self.ctx.mgr)
        if _list_worker_files(self.ctx.workers_dir):
            self.ctx.mgr.start()
        self._settle()

    @rule(advance_s=st.floats(min_value=0.5, max_value=15.0))
    def tick_clock(self, advance_s: float) -> None:
        self.ctx.fake_clock.advance(advance_s)
        # Let deferred watcher restarts observe the advanced clock.
        self.ctx.watcher.reconcile_worker_files()
        self._settle()

    @rule()
    def simulate_external_container_exit(self) -> None:
        if docker_container_status(self.ctx.container_name) == "running":
            subprocess.run(
                ["docker", "stop", self.ctx.container_name],
                capture_output=True,
                timeout=20,
            )
        self._settle()

    # ------------------------------------------------------------------
    # Invariants (CYB-2140)
    # ------------------------------------------------------------------

    @invariant()
    def workers_dir_matches_container_when_sync_clean(self) -> None:
        if self.ctx.last_sync_had_errors:
            return
        files = _list_worker_files(self.ctx.workers_dir)
        status = docker_container_status(self.ctx.container_name)
        if files:
            assert status == "running", (
                f"worker files present but container status={status!r}"
            )
        else:
            assert status in {"exited", "none"}, (
                f"empty workers dir but container status={status!r}"
            )

    @invariant()
    def stop_path_never_docker_rms_worker(self) -> None:
        """``reconcile`` stop issues ``docker stop`` only; ``docker rm`` is the
        restart/recreate path inside ``_run_container``.  Any rm that fires
        outside a restart is a bug — it would destroy diagnostics and violate
        the ticket invariant ``docker_rm against worker container == 0``."""
        assert self.ctx.docker_rm_calls_during_stop == 0, (
            f"docker_rm fired {self.ctx.docker_rm_calls_during_stop} time(s) "
            "outside of a restart; bare stop path must only call docker stop"
        )

    @invariant()
    def container_name_stable(self) -> None:
        data = docker_inspect(self.ctx.container_name)
        if data is None:
            return
        name = data.get("Name", "").lstrip("/")
        assert name == self.ctx.container_name

    @invariant()
    def request_id_dedupe_fires_at_most_once(self) -> None:
        """Each unique request_id must dispatch ``_run_remove_workflow_worker``
        at most once, even when the same payload is delivered multiple times."""
        for rid, count in self.ctx.remove_dispatch_counts.items():
            assert count <= 1, (
                f"request_id={rid!r} dispatched {count} times; dedupe must drop repeats"
            )

    @invariant()
    def immutable_image_pulled_at_most_once(self) -> None:
        # ``:it`` is an immutable basename; with the stub pre-tagged locally,
        # ``_ensure_image_pulled`` must not successfully pull more than once.
        assert self.ctx.immutable_pull_count <= 1, (
            f"local stub image pull succeeded {self.ctx.immutable_pull_count} times"
        )

    @invariant()
    def record_stop_called_on_every_deliberate_stop(self) -> None:
        """Every call to ``WorkerManager.stop()`` that transitions the container
        from ``running`` to ``exited`` must call ``monitor.record_stop()`` exactly
        once.  Broken wiring causes the next ``check()`` to emit a false
        "exited spontaneously" WARNING."""
        assert self.ctx.record_stop_count == self.ctx.stop_running_transitions, (
            f"record_stop called {self.ctx.record_stop_count} time(s) but "
            f"{self.ctx.stop_running_transitions} running→exited transition(s) were observed; "
            "WorkerManager.stop() wiring to monitor.record_stop() is broken"
        )

    @invariant()
    def watcher_restart_cooldown_respected(self) -> None:
        """Consecutive *watcher-initiated* restarts must be at least
        ``_WATCHER_COOLDOWN_S`` apart on the monotonic clock.

        Only records restarts that go through the watcher's own flow
        (reasons ``"worker-files-changed"`` and ``"remove-workflow-worker"``).
        Direct ``WorkerManager.restart`` calls (e.g. ``chaos-restart``) bypass
        the watcher's cooldown gate entirely and are intentionally not tracked
        here — they are not covered by the 2 s invariant.
        """
        times = self.ctx.watcher_restart_times
        if len(times) < 2:
            return
        for prev, curr in zip(times, times[1:]):
            delta = curr - prev
            assert delta >= _WATCHER_COOLDOWN_S - 0.01 or self.ctx.watcher._pending_restart, (  # noqa: SLF001
                f"watcher restarts {delta:.2f}s apart (< {_WATCHER_COOLDOWN_S}s)"
            )


# ---------------------------------------------------------------------------
# Pytest wiring
# ---------------------------------------------------------------------------


def _build_chaos_context(
    configured_edge: Path,
    worker_env_uuid: str,
    worker_container_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> ChaosContext:
    fake_clock = FakeMonotonicClock(start=1000.0)
    set_now_monotonic(fake_clock)

    workers_dir = configured_edge / "workers"
    mgr = WorkerManager(
        config_dir=configured_edge,
        environment_uuid=worker_env_uuid,
        token="it-token",
        twin_uuids=["it-twin"],
        image=STUB_IMAGE_IT,
    )
    for tag in STUB_IMAGE_TAGS:
        inspect = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
            timeout=10,
        )
        if inspect.returncode != 0:
            raise RuntimeError(f"stub image {tag} missing before chaos run")

    from cyberwave_edge_core.worker_health import WorkerHealthMonitor

    monitor = WorkerHealthMonitor(
        container_name=mgr.container_name,
        max_restarts_in_window=10_000,
        restart_window_seconds=3600.0,
    )
    mgr.set_health_monitor(monitor)

    original_record_stop = monitor.record_stop

    ctx_holder: list[ChaosContext] = []

    def tracking_record_stop(*, reason: str) -> None:
        if ctx_holder:
            ctx_holder[0].record_stop_count += 1
        return original_record_stop(reason=reason)

    monkeypatch.setattr(monitor, "record_stop", tracking_record_stop)

    original_stop = WorkerManager.stop

    def tracking_stop(self: WorkerManager, *, reason: str = "requested") -> bool:
        before = docker_container_status(self.container_name)
        ok = original_stop(self, reason=reason)
        after = docker_container_status(self.container_name)
        if ctx_holder and before == "running" and after in {"exited", "none"}:
            ctx_holder[0].stop_running_transitions += 1
        return ok

    monkeypatch.setattr(WorkerManager, "stop", tracking_stop)

    original_restart = WorkerManager.restart

    def tracking_restart(self: WorkerManager, *, reason: str = "requested") -> bool:
        if ctx_holder:
            ctx = ctx_holder[0]
            if reason in {"worker-files-changed", "remove-workflow-worker"}:
                ctx.watcher_restart_times.append(now_monotonic())
            # Flag that we're inside a restart so the docker_rm spy knows
            # this rm is a legitimate recreate, not a bare stop.
            ctx._in_restart = True
        try:
            return original_restart(self, reason=reason)
        finally:
            if ctx_holder:
                ctx_holder[0]._in_restart = False

    monkeypatch.setattr(WorkerManager, "restart", tracking_restart)

    model_manager = MagicMock()
    model_manager.scan_worker_model_ids.return_value = []
    watcher = WorkerWatcher(
        workers_dir=workers_dir,
        worker_manager=mgr,
        model_manager=model_manager,
        min_restart_interval_seconds=_WATCHER_COOLDOWN_S,
    )
    startup._set_active_worker_watcher(watcher)
    startup._set_monitored_worker_manager(mgr)
    startup._HANDLED_TWIN_COMMAND_REQUEST_IDS.clear()

    ctx = ChaosContext(
        config_dir=configured_edge,
        workers_dir=workers_dir,
        container_name=worker_container_name,
        env_uuid=worker_env_uuid,
        fake_clock=fake_clock,
        mgr=mgr,
        watcher=watcher,
    )
    object.__setattr__(ctx, "_original_docker_rm", docker_helpers.docker_rm)
    object.__setattr__(
        ctx, "_original_subprocess_run", worker_manager_module.subprocess.run
    )
    ctx_holder.append(ctx)

    monkeypatch.setattr(docker_helpers, "docker_rm", ctx.wrap_docker_rm)
    monkeypatch.setattr(
        worker_manager_module.subprocess,
        "run",
        ctx.wrap_subprocess_run,
    )

    return ctx


@pytest.fixture
def chaos_context(
    configured_edge: Path,
    worker_env_uuid: str,
    worker_container_name: str,
    stub_worker_image: str,
    docker_cleanup: str,
    monkeypatch: pytest.MonkeyPatch,
) -> ChaosContext:
    ctx = _build_chaos_context(
        configured_edge, worker_env_uuid, worker_container_name, monkeypatch
    )
    yield ctx
    set_now_monotonic(None)
    startup._set_active_worker_watcher(None)
    startup._set_monitored_worker_manager(None)
    startup._HANDLED_TWIN_COMMAND_REQUEST_IDS.clear()


def _list_cyberwave_worker_containers() -> list[str]:
    """Return names of all running/stopped ``cyberwave-worker-*`` containers."""
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "name=cyberwave-worker-",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return [n.strip() for n in result.stdout.splitlines() if n.strip()]


class _WorkerLifecycleChaosTestCase(WorkerLifecycleChaosMachine.TestCase):
    """Hypothesis TestCase with CYB-2140 PR profile settings."""

    settings = _CHAOS_SETTINGS
    # Prevent pytest from collecting this as a standalone unittest test;
    # it is only meant to be driven via test_worker_lifecycle_chaos().
    __test__ = False


@requires_docker
def test_worker_lifecycle_chaos(chaos_context: ChaosContext) -> None:
    """Run the Hypothesis state machine (CYB-2140 PR profile)."""
    containers_before = set(_list_cyberwave_worker_containers())

    WorkerLifecycleChaosMachine._ctx = chaos_context
    try:
        _WorkerLifecycleChaosTestCase().runTest()
    finally:
        WorkerLifecycleChaosMachine._ctx = None

    # Invariant 8: no orphan cyberwave-worker-* containers after the run.
    # The only allowed name is the one docker_cleanup will remove; any others
    # are leftovers the machine failed to clean up.
    containers_after = set(_list_cyberwave_worker_containers())
    new_containers = containers_after - containers_before - {chaos_context.container_name}
    assert not new_containers, (
        f"orphan containers left after chaos run: {new_containers}"
    )
