"""Attaching a container to more than one network (CYB-3469).

`docker create` honours a single ``--network`` before Docker 25, so a container
that has to span two gets the rest connected between create and start. The
simulation plant proxy is the reason: it bridges the cyberwave-sim Zenoh network
and the ROS/DDS network, and that split is what keeps physics off the ROS
segment. Attaching it to only one leaves a container that looks healthy while
being unable to reach either the plant or the autonomy graph, so a failed attach
has to be fatal rather than a warning.
"""

from __future__ import annotations

import subprocess
from typing import Any

from cyberwave_edge_core import docker_launch
from tests.driver_subprocess_fakes import fake_docker_start_popen


def _run_recorder(commands: list[list[str]], *, fail_on: str | None = None) -> Any:
    def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(list(cmd))
        if fail_on is not None and fail_on in cmd:
            raise subprocess.CalledProcessError(1, cmd, stderr="network not found")
        return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")

    return _run


def _launch(
    monkeypatch: Any,
    commands: list[list[str]],
    *,
    extra_networks: tuple[str, ...],
    fail_on: str | None = None,
    removed: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Drive launch_detached_container with docker fully mocked."""
    failures: list[str] = []

    monkeypatch.setattr(docker_launch.subprocess, "run", _run_recorder(commands, fail_on=fail_on))
    monkeypatch.setattr(docker_launch, "_docker_start_popen", fake_docker_start_popen(commands))
    monkeypatch.setattr(
        docker_launch,
        "probe_container_startup",
        lambda name, probe_seconds: docker_launch.ContainerProbeResult(True, "running"),
    )
    monkeypatch.setattr(
        docker_launch.docker_helpers,
        "docker_rm",
        lambda name, **kw: (removed.append(name) if removed is not None else None) or True,
    )

    ok = docker_launch.launch_detached_container(
        container_name="cyberwave-driver-abcd1234-plant",
        run_argv=["docker", "run", "--detach", "--name", "x", "--network", "stacknet", "img:tag"],
        get_runtime_env_var=lambda name, default=None: default,
        on_container_created=lambda: None,
        extra_networks=extra_networks,
        on_running=lambda: None,
        on_failure=lambda message, kind: failures.append(kind),
    )
    return ok, failures


def _connects(commands: list[list[str]]) -> list[list[str]]:
    return [c for c in commands if c[:3] == ["docker", "network", "connect"]]


def test_no_extra_networks_issues_no_connect(monkeypatch: Any) -> None:
    """The robot variant is host-networked; nothing may change for it."""
    commands: list[list[str]] = []
    ok, failures = _launch(monkeypatch, commands, extra_networks=())
    assert ok is True
    assert failures == []
    assert _connects(commands) == []


def test_extra_network_is_connected_after_create(monkeypatch: Any) -> None:
    commands: list[list[str]] = []
    ok, failures = _launch(monkeypatch, commands, extra_networks=("sim-net",))
    assert ok is True
    assert failures == []
    assert _connects(commands) == [
        ["docker", "network", "connect", "sim-net", "cyberwave-driver-abcd1234-plant"]
    ]


def test_connect_happens_between_create_and_start(monkeypatch: Any) -> None:
    """Order matters: connecting a running container would race discovery."""
    commands: list[list[str]] = []
    ok, _ = _launch(monkeypatch, commands, extra_networks=("sim-net",))
    assert ok is True

    kinds = [
        "create" if c[:2] == ["docker", "create"] else "connect"
        for c in commands
        if c[:2] == ["docker", "create"] or c[:3] == ["docker", "network", "connect"]
    ]
    assert kinds == ["create", "connect"], kinds


def test_multiple_extra_networks_are_all_connected(monkeypatch: Any) -> None:
    commands: list[list[str]] = []
    ok, _ = _launch(monkeypatch, commands, extra_networks=("sim-net", "other-net"))
    assert ok is True
    assert [c[3] for c in _connects(commands)] == ["sim-net", "other-net"]


def test_failed_attach_is_fatal_and_reaps_the_container(monkeypatch: Any) -> None:
    """Half-attached is worse than not started: it would look healthy."""
    commands: list[list[str]] = []
    removed: list[str] = []
    ok, failures = _launch(
        monkeypatch, commands, extra_networks=("sim-net",), fail_on="sim-net", removed=removed
    )
    assert ok is False
    assert failures == ["docker_network_attach_failed"]
    assert removed == ["cyberwave-driver-abcd1234-plant"], "the container must not linger"


def test_failed_attach_never_starts_the_container(monkeypatch: Any) -> None:
    commands: list[list[str]] = []
    ok, _ = _launch(monkeypatch, commands, extra_networks=("sim-net",), fail_on="sim-net")
    assert ok is False
    starts = [c for c in commands if c[:2] == ["docker", "start"]]
    assert starts == [], "a container that failed to attach must never be started"
