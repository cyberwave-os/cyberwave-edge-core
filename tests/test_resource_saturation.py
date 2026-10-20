"""Edge-core alerts every managed twin while a host resource is saturated."""

from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from cyberwave_edge_core import startup
from cyberwave_edge_core.resource_monitor import MemoryInfo, ResourceSnapshot

TWIN_A = "11111111-2222-4333-8444-555555555555"
TWIN_B = "99999999-2222-4333-8444-555555555555"


@pytest.fixture
def edge(monkeypatch: pytest.MonkeyPatch) -> Any:
    state: Any = type("Edge", (), {})()
    state.monitor = SimpleNamespace(
        last_snapshot=None, saturation_percent=95.0, saturation_clear_percent=90.0
    )
    state.clock = 0.0
    state.sent = []
    state.resolved = []
    state.resolve_fails = False
    state.fail_twins = set()
    state.top_calls = []
    monkeypatch.setattr(
        startup,
        "_CONTAINER_TWIN_MAP",
        {"cyberwave-driver-11111111": TWIN_A, "cyberwave-driver-99999999": TWIN_B},
    )
    monkeypatch.setattr(startup, "_SATURATED", {})
    monkeypatch.setattr(startup, "_SATURATION_STREAKS", {})
    monkeypatch.setattr(startup, "_SATURATION_UNRESOLVED", {})
    monkeypatch.setattr(startup, "_SATURATION_LEFT_OPEN", set())
    monkeypatch.setattr(startup, "_LAST_SATURATION_SNAPSHOT", None)
    monkeypatch.setattr(startup.socket, "gethostname", lambda: "edgepi")

    def _top(resource: str) -> str:
        state.top_calls.append(resource)
        return "cyberwave-driver-11111111 61%" if resource != "disk" else ""

    monkeypatch.setattr(startup, "_top_containers", _top)
    state.send_fails = set()

    def _send(twin: str, name: str, desc: str, kind: str, **kw: Any) -> None:
        if twin in state.send_fails:
            raise RuntimeError("backend unreachable")
        state.sent.append({"twin": twin, "name": name, "description": desc, "type": kind, **kw})

    monkeypatch.setattr(startup, "_send_alert_for_twin", _send)

    def _resolve(twin: str, kind: str, *, strict: bool = False) -> int:
        assert strict
        if state.resolve_fails or twin in state.fail_twins:
            raise RuntimeError("backend unreachable")
        state.resolved.append((twin, kind))
        return 1

    monkeypatch.setattr(startup, "resolve_active_alerts_for_twin", _resolve)

    def snapshot(cpu: float | None = 20.0, memory: float | None = 40.0, disk: float | None = 50.0):
        state.clock += 30.0
        state.monitor.last_snapshot = ResourceSnapshot(
            timestamp=state.clock,
            memory=MemoryInfo(total_mb=4096, available_mb=1024, used_percent=memory)
            if memory is not None
            else None,
            cpu_temp=None,
            cpu_percent=cpu,
            disk_percent=disk,
        )
        startup.reconcile_resource_saturation(state.monitor)

    state.snapshot = snapshot
    return state


def _types(sent: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(alert["twin"], alert["type"]) for alert in sent]


def test_two_snapshots_over_the_threshold_alert_every_managed_twin_once(edge: Any) -> None:
    edge.snapshot(cpu=97)
    assert edge.sent == []

    edge.snapshot(cpu=97.04)
    assert _types(edge.sent) == [(TWIN_A, "edge_cpu_saturation"), (TWIN_B, "edge_cpu_saturation")]
    alert = edge.sent[0]
    assert alert["name"] == "Edge CPU saturated"
    assert alert["severity"] == "warning"
    assert alert["metadata"] == {"resource": "cpu", "percent": 97.0, "threshold": 95.0}
    assert "97%" in alert["description"]
    assert "edgepi" in alert["description"]
    assert "cyberwave-driver-11111111 61%" in alert["description"]
    assert edge.top_calls == ["cpu"]

    edge.snapshot(cpu=99)
    edge.snapshot(cpu=98)
    assert len(edge.sent) == 2
    assert edge.top_calls == ["cpu"]


def test_only_a_new_snapshot_counts(edge: Any) -> None:
    edge.snapshot(memory=97)
    for _ in range(3):
        startup.reconcile_resource_saturation(edge.monitor)
    assert edge.sent == []

    edge.snapshot(memory=97)
    assert _types(edge.sent) == [
        (TWIN_A, "edge_memory_saturation"),
        (TWIN_B, "edge_memory_saturation"),
    ]
    assert edge.sent[0]["name"] == "Edge memory saturated"


def test_a_single_spike_or_dip_does_not_flip(edge: Any) -> None:
    edge.snapshot(disk=96)
    edge.snapshot(disk=50)
    edge.snapshot(disk=96)
    assert edge.sent == []

    edge.snapshot(disk=96)
    edge.snapshot(disk=80)
    edge.snapshot(disk=96)
    edge.snapshot(disk=80)
    assert _types(edge.sent) == [(TWIN_A, "edge_disk_saturation"), (TWIN_B, "edge_disk_saturation")]
    assert edge.resolved == []


def test_resolves_after_two_snapshots_below_the_clear_level(edge: Any) -> None:
    edge.snapshot(cpu=97)
    edge.snapshot(cpu=97)
    edge.snapshot(cpu=92)
    edge.snapshot(cpu=92)
    edge.snapshot(cpu=85)
    assert edge.resolved == []

    edge.snapshot(cpu=85)
    assert edge.resolved == [(TWIN_A, "edge_cpu_saturation"), (TWIN_B, "edge_cpu_saturation")]

    edge.snapshot(cpu=97)
    edge.snapshot(cpu=97)
    assert len(edge.sent) == 4


def test_a_failed_resolve_is_retried_on_the_next_snapshot(edge: Any) -> None:
    edge.snapshot(cpu=97)
    edge.snapshot(cpu=97)
    edge.resolve_fails = True
    edge.snapshot(cpu=10)
    edge.snapshot(cpu=10)
    assert edge.resolved == []

    edge.resolve_fails = False
    edge.snapshot(cpu=10)
    assert edge.resolved == [(TWIN_A, "edge_cpu_saturation"), (TWIN_B, "edge_cpu_saturation")]
    edge.snapshot(cpu=10)
    assert len(edge.resolved) == 2


def test_without_managed_twins_it_only_logs(
    edge: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {})
    with caplog.at_level(logging.WARNING, logger=startup.logger.name):
        edge.snapshot(memory=99)
        edge.snapshot(memory=99)

    assert edge.sent == []
    assert "Edge memory saturated: 99% (threshold 95%) for 30 s" in caplog.text


def test_logs_every_60_s_while_saturated(edge: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=startup.logger.name):
        for _ in range(6):
            edge.snapshot(cpu=97)

    lines = [r.getMessage() for r in caplog.records if "Edge CPU saturated" in r.getMessage()]
    assert lines == [
        "Edge CPU saturated: 97% (threshold 95%) for 30 s; top: cyberwave-driver-11111111 61%",
        "Edge CPU saturated: 97% (threshold 95%) for 90 s; top: cyberwave-driver-11111111 61%",
        "Edge CPU saturated: 97% (threshold 95%) for 150 s; top: cyberwave-driver-11111111 61%",
    ]


def test_missing_readings_change_nothing(edge: Any) -> None:
    edge.snapshot(cpu=None, memory=None, disk=None)
    edge.snapshot(cpu=None, memory=None, disk=None)
    edge.monitor.last_snapshot = None
    startup.reconcile_resource_saturation(edge.monitor)

    assert edge.sent == []


def test_top_containers_ranks_docker_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _run(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(cmd)
        assert kwargs["timeout"] > 0
        return SimpleNamespace(
            stdout="a\t5.0%\t70%\nb\t61.2%\t3%\nc\t20%\t10%\nd\t30%\t1%\nbad line\n"
        )

    monkeypatch.setattr(startup.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(startup.subprocess, "run", _run)

    assert startup._top_containers("cpu") == "b 61%, d 30%, c 20%"
    assert startup._top_containers("memory") == "a 70%, c 10%, b 3%"
    assert startup._top_containers("disk") == ""
    assert len(calls) == 2
    assert calls[0][:3] == ["docker", "stats", "--no-stream"]


def test_top_containers_is_empty_when_docker_stats_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _run(cmd: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(startup.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(startup.subprocess, "run", _run)

    assert startup._top_containers("cpu") == ""


def test_outstanding_resolutions_survive_a_new_crossing(
    edge: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    edge.snapshot(cpu=97)
    edge.snapshot(cpu=97)
    edge.fail_twins = {TWIN_B}
    edge.snapshot(cpu=10)
    edge.snapshot(cpu=10)
    assert edge.resolved == [(TWIN_A, "edge_cpu_saturation")]

    monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {"cyberwave-driver-11111111": TWIN_A})
    edge.snapshot(cpu=97)
    edge.snapshot(cpu=97)
    edge.fail_twins = set()
    edge.snapshot(cpu=97)
    assert edge.resolved == [(TWIN_A, "edge_cpu_saturation"), (TWIN_B, "edge_cpu_saturation")]

    edge.snapshot(cpu=10)
    edge.snapshot(cpu=10)
    assert edge.resolved[-1] == (TWIN_A, "edge_cpu_saturation")
    assert len(edge.resolved) == 3


def test_an_undelivered_alert_is_retried_unchanged_while_saturated(edge: Any) -> None:
    edge.send_fails = {TWIN_B}
    edge.snapshot(cpu=97)
    edge.snapshot(cpu=97)
    assert _types(edge.sent) == [(TWIN_A, "edge_cpu_saturation")]

    edge.snapshot(cpu=99)
    assert len(edge.sent) == 1
    edge.send_fails = set()
    edge.snapshot(cpu=99)
    assert _types(edge.sent) == [(TWIN_A, "edge_cpu_saturation"), (TWIN_B, "edge_cpu_saturation")]
    first, retried = edge.sent
    assert {k: v for k, v in retried.items() if k != "twin"} == {
        k: v for k, v in first.items() if k != "twin"
    }

    edge.snapshot(cpu=99)
    assert len(edge.sent) == 2


def test_no_alert_is_retried_once_the_saturation_clears(edge: Any) -> None:
    edge.send_fails = {TWIN_A, TWIN_B}
    edge.snapshot(memory=97)
    edge.snapshot(memory=97)
    edge.snapshot(memory=50)
    edge.send_fails = set()
    edge.snapshot(memory=50)
    edge.snapshot(memory=50)

    assert edge.sent == []


def test_a_restart_resolves_the_alerts_an_earlier_process_left_open(edge: Any) -> None:
    startup.adopt_saturation_alerts_left_open()

    edge.snapshot(cpu=20, memory=40, disk=50)

    expected = {(twin, f"edge_{r}_saturation") for twin in (TWIN_A, TWIN_B) for r in ("cpu", "memory", "disk")}
    assert set(edge.resolved) == expected
    edge.snapshot()
    assert len(edge.resolved) == 6, "resolved once, not on every snapshot"


def test_a_resource_saturated_again_after_a_restart_keeps_its_alert(edge: Any) -> None:
    startup.adopt_saturation_alerts_left_open()

    edge.snapshot(cpu=97)
    edge.snapshot(cpu=97)
    assert (TWIN_A, "edge_cpu_saturation") not in edge.resolved
    assert _types(edge.sent) == [(TWIN_A, "edge_cpu_saturation"), (TWIN_B, "edge_cpu_saturation")]

    edge.snapshot(cpu=10)
    edge.snapshot(cpu=10)
    assert edge.resolved.count((TWIN_A, "edge_cpu_saturation")) == 1


def test_a_reading_between_the_clear_level_and_the_threshold_resolves_nothing_yet(edge: Any) -> None:
    startup.adopt_saturation_alerts_left_open()

    edge.snapshot(cpu=92, memory=40, disk=50)

    assert not any(kind == "edge_cpu_saturation" for _, kind in edge.resolved)
