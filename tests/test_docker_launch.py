"""Unit tests for cyberwave_edge_core.docker_launch."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import cyberwave_edge_core.docker_helpers as docker_helpers
import cyberwave_edge_core.docker_launch as dl

_PROBE_ENV = "CYBERWAVE_DRIVER_STARTUP_PROBE_SECONDS"


def _finished_popen(*_args: Any, **_kwargs: Any) -> MagicMock:
    """A ``docker start`` Popen double whose process has already exited cleanly."""
    return MagicMock(poll=MagicMock(return_value=0))


def _probe_env(value: object) -> Any:
    """Build a ``get_runtime_env_var`` fake returning *value* for the probe window."""

    def _env(name: str, default: object = None) -> object:
        return value if name == _PROBE_ENV else default

    return _env


class TestDockerCreateArgvFromRunArgv:
    def test_strips_detach_flag(self) -> None:
        run_argv = [
            "docker",
            "run",
            "--detach",
            "--init",
            "--name",
            "cyberwave-driver-abc",
            "img:latest",
        ]
        assert dl.docker_create_argv_from_run_argv(run_argv) == [
            "docker",
            "create",
            "--init",
            "--name",
            "cyberwave-driver-abc",
            "img:latest",
        ]

    def test_rejects_non_run_argv(self) -> None:
        with pytest.raises(ValueError, match="Expected docker run"):
            dl.docker_create_argv_from_run_argv(["docker", "create", "img"])


class TestDriverStartupProbeSeconds:
    def test_default_when_unset(self) -> None:
        assert dl.driver_startup_probe_seconds(lambda _n, default=None: None) == 120

    def test_env_override(self) -> None:
        assert dl.driver_startup_probe_seconds(_probe_env("45")) == 45

    def test_invalid_override_uses_default(self) -> None:
        assert dl.driver_startup_probe_seconds(_probe_env("bad")) == 120


def _remove_env(value: object) -> Any:
    """Build a ``get_runtime_env_var`` fake for the removal timeout knob."""

    def _env(name: str, default: object = None) -> object:
        return value if name == "CYBERWAVE_DRIVER_REMOVE_TIMEOUT_SECONDS" else default

    return _env


class TestDriverRemoveTimeoutSeconds:
    def test_default_when_unset(self) -> None:
        assert dl.driver_remove_timeout_seconds(lambda _n, default=None: None) == 60

    def test_env_override(self) -> None:
        assert dl.driver_remove_timeout_seconds(_remove_env("90")) == 90

    def test_invalid_override_uses_default(self) -> None:
        assert dl.driver_remove_timeout_seconds(_remove_env("nope")) == 60


class TestRemoveExistingContainer:
    def _noop_env(self, _name: str, default: Any = None) -> Any:
        return default

    def test_no_container_is_noop(self) -> None:
        with patch.object(docker_helpers, "docker_inspect", return_value=None) as insp:
            with patch.object(docker_helpers, "docker_stop") as stop:
                with patch.object(docker_helpers, "docker_rm") as rm:
                    ok = dl.remove_existing_container(
                        "cyberwave-driver-deadbeef", get_runtime_env_var=self._noop_env
                    )
        assert ok is True
        insp.assert_called_once()
        stop.assert_not_called()
        rm.assert_not_called()

    def test_stops_then_removes_when_present(self) -> None:
        # Present on the first inspect, gone after stop+rm.
        inspect_results = [{"State": {"Status": "running"}}, None]
        calls: list[str] = []
        with patch.object(docker_helpers, "docker_inspect", side_effect=inspect_results):
            with patch.object(
                docker_helpers, "docker_stop", side_effect=lambda n, **kw: calls.append("stop")
            ):
                with patch.object(
                    docker_helpers, "docker_rm", side_effect=lambda n, **kw: calls.append("rm")
                ):
                    ok = dl.remove_existing_container(
                        "cyberwave-driver-deadbeef", get_runtime_env_var=self._noop_env
                    )
        assert ok is True
        assert calls == ["stop", "rm"]

    def test_returns_false_when_container_wont_die(self) -> None:
        # Still present after stop+rm (wedged on a held device).
        with patch.object(
            docker_helpers, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            with patch.object(docker_helpers, "docker_stop"):
                with patch.object(docker_helpers, "docker_rm"):
                    ok = dl.remove_existing_container(
                        "cyberwave-driver-deadbeef", get_runtime_env_var=self._noop_env
                    )
        assert ok is False

    def test_uses_configurable_timeout(self) -> None:
        captured: dict[str, int] = {}

        def _stop(_n: str, *, timeout: int) -> bool:
            captured["stop"] = timeout
            return True

        def _rm(_n: str, *, timeout: int) -> bool:
            captured["rm"] = timeout
            return True

        inspect_results = [{"State": {"Status": "running"}}, None]
        with patch.object(docker_helpers, "docker_inspect", side_effect=inspect_results):
            with patch.object(docker_helpers, "docker_stop", side_effect=_stop):
                with patch.object(docker_helpers, "docker_rm", side_effect=_rm):
                    dl.remove_existing_container(
                        "cyberwave-driver-deadbeef", get_runtime_env_var=_remove_env("90")
                    )
        assert captured == {"stop": 90, "rm": 90}


class TestProbeContainerStartup:
    def test_running_returns_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)
        with patch.object(
            docker_helpers, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            result = dl.probe_container_startup("cyberwave-driver-test", probe_seconds=5)
        assert result.success is True
        assert result.last_status == "running"

    def test_created_waits_until_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)
        with patch.object(
            docker_helpers,
            "docker_inspect",
            return_value={"State": {"Status": "created"}},
        ):
            result = dl.probe_container_startup("cyberwave-driver-test", probe_seconds=3)
        assert result.success is False
        assert result.last_status == "created"

    def test_exited_without_restarts_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)
        with patch.object(
            docker_helpers,
            "docker_inspect",
            return_value={"State": {"Status": "exited", "ExitCode": 1}, "RestartCount": 0},
        ):
            result = dl.probe_container_startup("cyberwave-driver-test", probe_seconds=1)
        assert result.success is False
        assert result.last_status == "exited"

    def test_exited_with_restarts_keeps_waiting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)
        inspect_results = [
            {"State": {"Status": "exited", "ExitCode": 1}, "RestartCount": 1},
            {"State": {"Status": "running"}, "RestartCount": 1},
        ]
        with patch.object(docker_helpers, "docker_inspect", side_effect=inspect_results):
            result = dl.probe_container_startup("cyberwave-driver-test", probe_seconds=5)
        assert result.success is True
        assert result.last_status == "running"

    def test_becomes_running_mid_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)
        inspect_results = [{"State": {"Status": "created"}}] * 2 + [
            {"State": {"Status": "running"}},
        ]
        with patch.object(docker_helpers, "docker_inspect", side_effect=inspect_results):
            result = dl.probe_container_startup("cyberwave-driver-test", probe_seconds=10)
        assert result.success is True


class TestLaunchDetachedContainer:
    _RUN_ARGV = [
        "docker",
        "run",
        "--detach",
        "--name",
        "cyberwave-driver-deadbeef",
        "img:latest",
    ]

    def _noop_env(self, _name: str, default: Any = None) -> Any:
        return default

    def test_happy_path_create_start_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        phases: list[str] = []
        monkeypatch.setattr(dl.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(dl.subprocess, "Popen", _finished_popen)
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)

        with patch.object(
            docker_helpers,
            "docker_inspect",
            return_value={"State": {"Status": "running"}},
        ):
            ok = dl.launch_detached_container(
                container_name="cyberwave-driver-deadbeef",
                run_argv=self._RUN_ARGV,
                get_runtime_env_var=self._noop_env,
                on_container_created=lambda: phases.append("created"),
                on_running=lambda: phases.append("running"),
                on_failure=lambda _m, _p: phases.append("failed"),
                stream_logs=lambda: phases.append("logs"),
            )

        assert ok is True
        assert phases == ["created", "logs", "running"]

    def test_registers_map_before_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Container-created callback must fire immediately after docker create."""
        events: list[str] = []
        create_ran = False

        def _fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            nonlocal create_ran
            if cmd[:2] == ["docker", "create"]:
                create_ran = True
                events.append("docker_create")
            return MagicMock()

        def _on_created() -> None:
            assert create_ran is True
            events.append("callback_created")

        monkeypatch.setattr(dl.subprocess, "run", _fake_run)
        monkeypatch.setattr(dl.subprocess, "Popen", _finished_popen)
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)

        inspect_results = [{"State": {"Status": "created"}}] * 2 + [
            {"State": {"Status": "running"}},
        ]
        with patch.object(docker_helpers, "docker_inspect", side_effect=inspect_results):
            ok = dl.launch_detached_container(
                container_name="cyberwave-driver-deadbeef",
                run_argv=self._RUN_ARGV,
                get_runtime_env_var=self._noop_env,
                on_container_created=_on_created,
                on_running=lambda: events.append("running"),
                on_failure=lambda _m, _p: events.append("failed"),
            )

        assert ok is True
        assert events.index("docker_create") < events.index("callback_created")
        assert "running" in events

    def test_create_failure_invokes_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        failures: list[tuple[str, str]] = []

        def _raise_cpe(*_a: Any, **_kw: Any) -> None:
            raise subprocess.CalledProcessError(1, ["docker", "create"], stderr="boom")

        monkeypatch.setattr(dl.subprocess, "run", _raise_cpe)

        ok = dl.launch_detached_container(
            container_name="cyberwave-driver-deadbeef",
            run_argv=self._RUN_ARGV,
            get_runtime_env_var=self._noop_env,
            on_container_created=lambda: failures.append(("created", "")),
            on_running=lambda: failures.append(("running", "")),
            on_failure=lambda msg, phase: failures.append((phase, msg)),
        )

        assert ok is False
        assert failures[0][0] == "docker_create_failed"

    def test_recovery_start_when_stuck_in_created(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run_cmds: list[list[str]] = []
        popen_calls = 0

        def _fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            run_cmds.append(list(cmd))
            return MagicMock()

        def _fake_popen(cmd: list[str], **kwargs: Any) -> MagicMock:
            nonlocal popen_calls
            popen_calls += 1
            run_cmds.append(list(cmd))
            proc = MagicMock()
            proc.poll.return_value = None if popen_calls == 1 else 0
            return proc

        monkeypatch.setattr(dl.subprocess, "run", _fake_run)
        monkeypatch.setattr(dl.subprocess, "Popen", _fake_popen)
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)

        probe_calls = {"n": 0}

        def _fake_probe(container_name: str, *, probe_seconds: int) -> dl.ContainerProbeResult:
            probe_calls["n"] += 1
            if probe_calls["n"] == 1:
                return dl.ContainerProbeResult(success=False, last_status="created")
            return dl.ContainerProbeResult(success=True, last_status="running")

        rm_calls: list[str] = []
        with patch.object(dl, "probe_container_startup", side_effect=_fake_probe):
            with patch.object(
                docker_helpers,
                "docker_rm",
                side_effect=lambda name: rm_calls.append(name) or True,
            ):
                ok = dl.launch_detached_container(
                    container_name="cyberwave-driver-deadbeef",
                    run_argv=self._RUN_ARGV,
                    get_runtime_env_var=self._noop_env,
                    on_container_created=lambda: None,
                    on_running=lambda: None,
                    on_failure=lambda _m, _p: None,
                )

        assert ok is True
        assert any(cmd[:3] == ["docker", "start", "cyberwave-driver-deadbeef"] for cmd in run_cmds)
        assert rm_calls == []

    def test_happy_path_reaps_start_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The detached ``docker start`` is communicate()'d so it is not leaked."""
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.communicate.return_value = ("", "")
        monkeypatch.setattr(dl.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(dl.subprocess, "Popen", lambda *a, **kw: proc)
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)

        with patch.object(
            docker_helpers,
            "docker_inspect",
            return_value={"State": {"Status": "running"}},
        ):
            ok = dl.launch_detached_container(
                container_name="cyberwave-driver-deadbeef",
                run_argv=self._RUN_ARGV,
                get_runtime_env_var=self._noop_env,
                on_container_created=lambda: None,
                on_running=lambda: None,
                on_failure=lambda _m, _p: None,
            )

        assert ok is True
        proc.communicate.assert_called_once()

    def test_failure_invokes_on_removed_after_rm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """on_removed fires after the orphan is force-removed."""
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.communicate.return_value = ("", "device busy")
        monkeypatch.setattr(dl.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(dl.subprocess, "Popen", lambda *a, **kw: proc)
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)

        events: list[str] = []
        with patch.object(
            dl,
            "probe_container_startup",
            return_value=dl.ContainerProbeResult(success=False, last_status="exited"),
        ):
            with patch.object(
                docker_helpers,
                "docker_rm",
                side_effect=lambda name: events.append(f"rm:{name}") or True,
            ):
                ok = dl.launch_detached_container(
                    container_name="cyberwave-driver-deadbeef",
                    run_argv=self._RUN_ARGV,
                    get_runtime_env_var=self._noop_env,
                    on_container_created=lambda: None,
                    on_running=lambda: None,
                    on_failure=lambda _m, _p: events.append("failed"),
                    on_removed=lambda: events.append("removed"),
                )

        assert ok is False
        assert events == ["rm:cyberwave-driver-deadbeef", "removed", "failed"]

    def test_definitive_failure_removes_orphan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dl.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(dl.subprocess, "Popen", _finished_popen)
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)

        failures: list[str] = []
        rm_calls: list[str] = []

        with patch.object(
            dl,
            "probe_container_startup",
            return_value=dl.ContainerProbeResult(success=False, last_status="exited"),
        ):
            with patch.object(
                docker_helpers,
                "docker_rm",
                side_effect=lambda name: rm_calls.append(name) or True,
            ):
                ok = dl.launch_detached_container(
                    container_name="cyberwave-driver-deadbeef",
                    run_argv=self._RUN_ARGV,
                    get_runtime_env_var=self._noop_env,
                    on_container_created=lambda: None,
                    on_running=lambda: None,
                    on_failure=lambda _msg, phase: failures.append(phase),
                )

        assert ok is False
        assert failures == ["container_unhealthy"]
        assert rm_calls == ["cyberwave-driver-deadbeef"]

    def test_probe_timeout_removes_orphan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dl.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(dl.subprocess, "Popen", _finished_popen)
        monkeypatch.setattr(dl.time, "sleep", lambda _: None)

        failures: list[str] = []
        rm_calls: list[str] = []

        with patch.object(
            dl,
            "probe_container_startup",
            side_effect=[
                dl.ContainerProbeResult(success=False, last_status="created"),
                dl.ContainerProbeResult(success=False, last_status="created"),
            ],
        ):
            with patch.object(
                docker_helpers,
                "docker_rm",
                side_effect=lambda name: rm_calls.append(name) or True,
            ):
                ok = dl.launch_detached_container(
                    container_name="cyberwave-driver-deadbeef",
                    run_argv=self._RUN_ARGV,
                    get_runtime_env_var=self._noop_env,
                    on_container_created=lambda: None,
                    on_running=lambda: None,
                    on_failure=lambda _msg, phase: failures.append(phase),
                )

        assert ok is False
        assert failures == ["container_startup_timeout"]
        assert rm_calls == ["cyberwave-driver-deadbeef"]

    def test_env_var_overrides_probe_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleep_calls: list[float] = []
        monkeypatch.setattr(dl.time, "sleep", lambda s: sleep_calls.append(s))
        monkeypatch.setattr(dl.subprocess, "run", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(dl.subprocess, "Popen", _finished_popen)

        with patch.object(
            docker_helpers,
            "docker_inspect",
            return_value={"State": {"Status": "created"}},
        ):
            dl.probe_container_startup(
                "cyberwave-driver-test",
                probe_seconds=dl.driver_startup_probe_seconds(
                    lambda name, default=None: (
                        "7" if name == "CYBERWAVE_DRIVER_STARTUP_PROBE_SECONDS" else default
                    )
                ),
            )

        assert len(sleep_calls) == 7
