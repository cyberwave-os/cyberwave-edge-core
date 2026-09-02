"""Waiting on `depends_on: {condition: service_healthy}` (CYB-3469).

Ordering alone was never the contract. A Compose file that says nav2 depends on
plant *being healthy* got only "start plant first", so nav2 came up the moment
plant's container was created — before its ROS graph existed. A node that misses
discovery does not crash; it sits idle looking healthy, which is the failure mode
these files are meant to make impossible.

Two invariants hold throughout, and both exist because getting them wrong is
worse than not gating at all:

* **The wait can delay a start but never prevent one.** The return value says
  whether the dependency reached healthy; the launch loop does not branch on it.
  A skipped service is never created, so its health snapshot reads `removed`, and
  `reconcile_driver_revival` deliberately ignores `removed` containers — removal
  is an operator signal there. Skipping would strand the service until edge-core
  restarts, with nothing to revive it. Docker's `unhealthy` is not terminal
  either: the healthcheck keeps running and a later probe can flip it back.

* **At most one wait per (twin, dependency) per pass.** Four Go2 services declare
  `plant: service_healthy`, and the launch loop is serial across every twin on
  the edge, so an unshared budget would multiply into a long boot stall for a
  single dead robot.
"""

from __future__ import annotations

import inspect
from typing import Any

from cyberwave_edge_core import health_gating, startup
from cyberwave_edge_core.docker_helpers import driver_container_name

TWIN = "abcd1234-0000-0000-0000-000000000000"


def _states(monkeypatch: Any, payloads: list[dict[str, Any] | None]) -> list[str]:
    """Serve *payloads* from successive probes; record what was asked.

    Cases are still written as `docker inspect` payload shapes because that is
    what the three outcomes mean -- absent container, no healthcheck declared,
    a status -- and translated here into what `_container_health_status`
    returns. Keeping the cases in docker's vocabulary is what lets the probe be
    swapped for a cheaper one (a `--format` template instead of parsing 30 KB
    of JSON per poll) without rewriting eleven tests.
    """
    asked: list[str] = []
    queue = list(payloads)

    def _probe(container_name: str) -> tuple[bool, str]:
        asked.append(container_name)
        payload = queue.pop(0) if queue else payloads[-1]
        if payload is None:
            return False, ""
        state = payload.get("State") or {}
        health = state.get("Health")
        if not isinstance(health, dict):
            return True, ""
        return True, str(health.get("Status", "")).lower()

    monkeypatch.setattr(startup, "_container_health_status", _probe)
    monkeypatch.setattr(startup.time, "sleep", lambda _seconds: None)
    return asked


def _health(status: str) -> dict[str, Any]:
    return {"State": {"Status": "running", "Health": {"Status": status}}}


def _wait(**kwargs: Any) -> bool:
    """The wait as Edge Core calls it: its probe, its logger, its name convention.

    `probe` and `log` are read from `startup` HERE rather than bound at import,
    so the monkeypatches above still land after the implementation moved into
    `health_gating` -- the two loggers below are asserted on `startup.logger`,
    and the probe is what `_states` swaps out.
    """
    return health_gating.wait_for_dependency_health(
        driver_container_name(TWIN, "plant"),
        probe=startup._container_health_status,
        log=startup.logger,
        **kwargs,
    )


# --- reaching healthy -------------------------------------------------------


def test_healthy_dependency_returns_immediately(monkeypatch: Any) -> None:
    asked = _states(monkeypatch, [_health("healthy")])
    assert _wait() is True
    assert len(asked) == 1, "a healthy dependency must not be polled twice"


def test_container_name_follows_the_edge_core_convention(monkeypatch: Any) -> None:
    """The name is how the wait finds the dependency, so it is part of the
    contract with driver_launcher's `cyberwave-driver-<twin8>-<service>`."""
    asked = _states(monkeypatch, [_health("healthy")])
    _wait()
    assert asked == ["cyberwave-driver-abcd1234-plant"]


def test_starting_dependency_is_polled_until_healthy(monkeypatch: Any) -> None:
    asked = _states(
        monkeypatch,
        [_health("starting"), _health("starting"), _health("healthy")],
    )
    assert _wait() is True
    assert len(asked) == 3


# --- never stranding a dependent -------------------------------------------


def test_the_launch_loop_never_branches_on_the_result() -> None:
    """The regression test for a stranded service.

    An earlier version returned False for an unhealthy dependency and the launch
    loop skipped the service. Never created means a `removed` health snapshot,
    which `reconcile_driver_revival` ignores by design — so the service stayed
    down until edge-core restarted. The wait must be called for its side effect
    (the delay) and nothing else.
    """
    source = inspect.getsource(startup.fetch_and_run_twin_drivers)
    calls = [line.strip() for line in source.splitlines() if ".wait_once(" in line]
    assert calls, "the launch loop no longer waits on dependency health at all"
    for call in calls:
        assert call.startswith("health_gate.wait_once("), (
            "the launch loop must not branch on the wait's result — a skipped "
            f"service is never revived. Found: {call!r}"
        )


def test_unhealthy_dependency_still_reports_rather_than_blocks(monkeypatch: Any) -> None:
    """False is a report, not a veto — see the test above for the veto check."""
    _states(monkeypatch, [_health("unhealthy")])
    assert _wait(timeout_seconds=0.0, poll_seconds=0.0) is False


def test_unhealthy_is_logged_as_an_error(monkeypatch: Any) -> None:
    """It will not fix itself by being waited on, and it is the likeliest root
    cause when a whole twin comes up mute, so it must not sit at warning level
    alongside the ordinary still-starting timeout."""
    _states(monkeypatch, [_health("unhealthy")])
    levels: list[str] = []
    monkeypatch.setattr(startup.logger, "error", lambda *a, **k: levels.append("error"))
    monkeypatch.setattr(startup.logger, "warning", lambda *a, **k: levels.append("warning"))
    _wait(timeout_seconds=0.0, poll_seconds=0.0)
    assert levels == ["error"]


def test_still_starting_at_timeout_is_a_warning(monkeypatch: Any) -> None:
    """A slow cold ROS graph on a Jetson is ordinary, not an error."""
    _states(monkeypatch, [_health("starting")])
    levels: list[str] = []
    monkeypatch.setattr(startup.logger, "error", lambda *a, **k: levels.append("error"))
    monkeypatch.setattr(startup.logger, "warning", lambda *a, **k: levels.append("warning"))
    _wait(timeout_seconds=0.0, poll_seconds=0.0)
    assert levels == ["warning"]


# --- nothing to wait for ---------------------------------------------------


def test_dependency_without_a_healthcheck_is_treated_as_started(monkeypatch: Any) -> None:
    """No `Health` key means the image declares no healthcheck. Compose treats
    that as started, and blocking would hang boot on nothing."""
    asked = _states(monkeypatch, [{"State": {"Status": "running"}}])
    assert _wait() is True
    assert len(asked) == 1


def test_missing_container_does_not_block(monkeypatch: Any) -> None:
    """The caller's own pull/create failure reporting describes this better than
    a silent two-minute timeout here would."""
    _states(monkeypatch, [None])
    assert _wait() is True


# --- the wait must not kill the process it runs in -------------------------


def test_a_long_wait_heartbeats_the_watchdog(monkeypatch: Any) -> None:
    """The regression test for a boot loop on the robot.

    This runs in the serial launch loop, which `reconcile_driver_revival` calls
    inline inside the runtime loop -- and that loop's only `watchdog.ping()` is
    at the very end of the cycle. With `WatchdogSec=60` against a ~15s cadence
    and no pinger thread (`start_pinging` deliberately spawns none), a default
    180s wait held the loop for three times the deadline with nothing pinging:
    systemd SIGABRTs edge-core mid-launch, the restart meets the same sick
    plant, and the robot boot-loops -- triggered by exactly the slow plant this
    gate exists for. So the poll must ping from inside.
    """
    _states(monkeypatch, [_health("starting"), _health("starting"), _health("healthy")])

    class _Watchdog:
        def __init__(self) -> None:
            self.pings = 0
            self.extended: list[float] = []

        def ping(self) -> None:
            self.pings += 1

        def extend_timeout(self, seconds: float) -> None:
            self.extended.append(seconds)

        def notify_status(self, status: str) -> None:
            pass

    watchdog = _Watchdog()
    assert _wait(poll_seconds=0.0, watchdog=watchdog) is True
    assert watchdog.pings >= 2, (
        "every poll iteration must ping, or a wait longer than WatchdogSec "
        f"kills the process; got {watchdog.pings}"
    )
    assert watchdog.extended, "TimeoutStartSec must be extended during a boot-time wait"


def test_the_launch_loop_passes_its_watchdog_to_the_wait() -> None:
    """The ping above is worthless if the caller never hands the watchdog over.

    Asserted against the source because the launch loop cannot be driven here
    without a backend, and this is a one-token omission that no behavioural
    test in this file would notice.
    """
    source = inspect.getsource(startup.fetch_and_run_twin_drivers)
    assert "_health_gate(" in source, "the launch loop builds no health gate"
    call = source.split("_health_gate(", 1)[1].split(")", 1)[0]
    assert "watchdog" in call, (
        "the launch loop must forward its watchdog when it builds the gate, or "
        f"the wait cannot ping: got {call!r}"
    )


def test_a_watchdog_failure_never_stops_a_driver_starting(monkeypatch: Any) -> None:
    """Same posture as the runtime loop's own ping: a watchdog problem is
    logged, never a reason a driver does not start."""
    _states(monkeypatch, [_health("starting"), _health("healthy")])

    class _Broken:
        def ping(self) -> None:
            raise RuntimeError("no /dev/watchdog")

        def extend_timeout(self, seconds: float) -> None:
            raise RuntimeError("not under systemd")

        def notify_status(self, status: str) -> None:
            raise RuntimeError("no socket")

    assert _wait(poll_seconds=0.0, watchdog=_Broken()) is True


# --- bounded cost ----------------------------------------------------------


def test_timeout_is_bounded_rather_than_infinite(monkeypatch: Any) -> None:
    """A dependency stuck in `starting` must not wedge startup forever."""
    ticks = iter([0.0] + [float(n) for n in range(1, 500)])
    monkeypatch.setattr(startup.time, "monotonic", lambda: next(ticks))
    asked = _states(monkeypatch, [_health("starting")])

    assert _wait(timeout_seconds=5.0, poll_seconds=0.0) is False
    assert len(asked) <= 7, f"polled {len(asked)} times for a 5s budget"


def test_the_budget_is_shared_across_dependents_of_one_dependency() -> None:
    """The regression test for a boot stall, now asserted as behaviour.

    Go2's bridges, nav2, slam and nav-bridge all declare
    `plant: service_healthy`. Waiting per dependent would spend the budget once
    each — four times over for one twin — and the launch loop is serial across
    twins, so a single dead robot would delay every healthy twin behind it.

    This used to be pinned by grepping the launch loop for a memo variable.
    Moving the memo inside `HealthGate` makes it directly observable instead: a
    second wait on the same dependency must not probe again. That also covers
    cyberwave-sim, which reuses the gate and previously had no memo at all.
    """
    probes: list[str] = []

    def _probe(name: str) -> tuple[bool, str]:
        probes.append(name)
        return True, "unhealthy"

    gate = health_gating.HealthGate(probe=_probe, timeout_seconds=0.0, poll_seconds=0.0)
    first = gate.wait_once("cyberwave-driver-abcd1234-plant")
    second = gate.wait_once("cyberwave-driver-abcd1234-plant")

    assert first is False and second is False, "the memoized answer must match the first"
    assert len(probes) == 1, (
        "the second dependent re-probed a dependency that already burned the "
        f"budget; {len(probes)} probes for one dependency"
    )


def test_the_gate_still_waits_on_a_different_dependency() -> None:
    """The memo is per dependency, not a one-shot for the whole pass."""
    probes: list[str] = []

    def _probe(name: str) -> tuple[bool, str]:
        probes.append(name)
        return True, "healthy"

    gate = health_gating.HealthGate(probe=_probe, poll_seconds=0.0)
    gate.wait_once("cyberwave-driver-abcd1234-plant")
    gate.wait_once("cyberwave-driver-abcd1234-slam")
    assert probes == [
        "cyberwave-driver-abcd1234-plant",
        "cyberwave-driver-abcd1234-slam",
    ]


def test_the_launch_loop_gates_through_one_pass_scoped_gate() -> None:
    """A gate built per service would make the memo a no-op.

    Behaviour cannot see this — the loop needs a backend to run — so it is read
    from the source, the same way the watchdog hand-off above is.
    """
    source = inspect.getsource(startup.fetch_and_run_twin_drivers)
    lines = source.splitlines()
    built = [line for line in lines if "_health_gate(" in line]
    assert len(built) == 1, f"expected exactly one gate per pass, found {len(built)}"
    assert ".wait_once(" in source, "the launch loop no longer uses the gate"


# ---------------------------------------------------------------------------
# Who pays the budget. The memo above bounds what ONE dead dependency costs;
# these bound who waits behind it. The launch loop is serial across every twin
# on the edge, so before this a Go2 with a sick plant held every camera on the
# box for the full timeout -- and whether it did depended on the order the
# backend happened to return twins in.
# ---------------------------------------------------------------------------


class _Spec:
    """Stands in for `_DriverSpec` (only the two fields the partition reads)."""

    def __init__(self, twin_uuid: str, service: str, depends_on: dict | None = None):
        self.twin_uuid = twin_uuid
        self.service_name = service
        self.depends_on = depends_on or {}

    def __repr__(self) -> str:  # pragma: no cover - assertion output only
        return f"{self.twin_uuid[:4]}/{self.service_name}"


GO2 = "aaaa1111-0000-0000-0000-000000000000"
CAM = "bbbb2222-0000-0000-0000-000000000000"


def _go2_graph() -> list[_Spec]:
    """The Go2 shape: an ungated plant plus dependents gated on it."""
    return [
        _Spec(GO2, "plant"),
        _Spec(GO2, "nav2", {"plant": "service_healthy"}),
        _Spec(GO2, "slam", {"plant": "service_healthy"}),
        _Spec(GO2, "map-stream-bridge", {"slam": "service_started"}),
    ]


def test_a_gated_twin_does_not_delay_an_ungated_one() -> None:
    specs = [*_go2_graph(), _Spec(CAM, None)]

    ordered = startup._defer_health_gated_twins(specs)

    assert ordered[0].twin_uuid == CAM, (
        "the camera must start before the Go2 whose plant may never go healthy"
    )
    assert [s.twin_uuid for s in ordered[1:]] == [GO2] * 4


def test_the_gated_twin_keeps_its_dependency_order() -> None:
    """Moving a twin must not reshuffle `_order_by_dependencies`' sequence."""
    ordered = startup._defer_health_gated_twins([*_go2_graph(), _Spec(CAM, None)])

    assert [s.service_name for s in ordered if s.twin_uuid == GO2] == [
        "plant",
        "nav2",
        "slam",
        "map-stream-bridge",
    ]


def test_the_whole_twin_moves_not_just_its_gated_services() -> None:
    """Per-spec would hoist `plant` away from the graph it is the root of."""
    ordered = startup._defer_health_gated_twins([*_go2_graph(), _Spec(CAM, None)])

    positions = [i for i, s in enumerate(ordered) if s.twin_uuid == GO2]
    assert positions == list(range(positions[0], positions[0] + len(positions))), (
        "the Go2's services must stay contiguous rather than interleaving"
    )


def test_a_graph_with_no_gates_is_returned_untouched() -> None:
    specs = [_Spec(CAM, None), _Spec(GO2, "plant")]
    assert startup._defer_health_gated_twins(specs) is specs


def test_every_twin_gated_changes_nothing() -> None:
    specs = _go2_graph()
    assert [s.service_name for s in startup._defer_health_gated_twins(specs)] == [
        s.service_name for s in specs
    ]


def test_service_started_alone_does_not_defer_a_twin() -> None:
    """Only `service_healthy` makes the loop wait; ordering costs nothing."""
    specs = [
        _Spec(GO2, "plant"),
        _Spec(GO2, "bridges", {"plant": "service_started"}),
        _Spec(CAM, None),
    ]
    assert startup._defer_health_gated_twins(specs) is specs


def test_no_spec_is_dropped_or_duplicated() -> None:
    specs = [*_go2_graph(), _Spec(CAM, None)]
    ordered = startup._defer_health_gated_twins(specs)
    assert sorted(map(id, ordered)) == sorted(map(id, specs))


def test_the_partition_runs_before_the_index_keyed_structures() -> None:
    """Placement is the correctness condition, not a style choice.

    `alert_by_spec_index` and everything derived from it are keyed by position
    in `driver_specs`. Reordering after they exist silently attaches every
    "Downloading driver image ..." alert to the wrong twin.
    """
    source = inspect.getsource(startup.fetch_and_run_twin_drivers)
    partition = source.index("_defer_health_gated_twins(")
    alerts = source.index("alert_by_spec_index: dict[int,")
    assert partition < alerts, (
        "the gated-twin partition must run before alert_by_spec_index is built"
    )
