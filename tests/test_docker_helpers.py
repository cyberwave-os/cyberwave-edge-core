"""Unit tests for cyberwave_edge_core.docker_helpers.

Covers:
- build_user_args: returns --user on Linux, empty on macOS
- docker_available: binary present / absent
- docker_rm: success, CalledProcessError treated as success, timeout/OSError
- docker_stop: success, CalledProcessError treated as success, timeout/OSError
- docker_inspect: valid JSON, bad JSON, empty list, non-list, subprocess errors
- docker_image_exists_locally: present, absent, errors
- docker_container_status: running, exited, inspect returns None, bad State type
- docker_ps_by_prefix: returns names, filters by prefix, handles errors,
  include_stopped passes -a flag
- docker_has_nvidia_runtime: nvidia present, absent, empty output, bad JSON, errors
- docker_logs_follow: returns Popen handle, docker unavailable, OSError on Popen
"""
from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import cyberwave_edge_core.docker_helpers as dh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completed(stdout: str = "", returncode: int = 0) -> MagicMock:
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.stdout = stdout
    m.returncode = returncode
    return m


# ---------------------------------------------------------------------------
# build_user_args
# ---------------------------------------------------------------------------


class TestBuildUserArgs:
    def test_returns_user_flag_on_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.platform, "system", lambda: "Linux")
        monkeypatch.setattr(dh.os, "getuid", lambda: 1000)
        monkeypatch.setattr(dh.os, "getgid", lambda: 1000)
        assert dh.build_user_args() == ["--user", "1000:1000"]

    def test_returns_empty_on_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.platform, "system", lambda: "Darwin")
        assert dh.build_user_args() == []

    def test_returns_empty_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.platform, "system", lambda: "Windows")
        assert dh.build_user_args() == []

    def test_uses_actual_uid_gid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.platform, "system", lambda: "Linux")
        monkeypatch.setattr(dh.os, "getuid", lambda: 501)
        monkeypatch.setattr(dh.os, "getgid", lambda: 20)
        assert dh.build_user_args() == ["--user", "501:20"]


# ---------------------------------------------------------------------------
# docker_available
# ---------------------------------------------------------------------------


class TestDockerAvailable:
    def test_returns_true_when_docker_in_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        assert dh.docker_available() is True

    def test_returns_false_when_docker_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: None)
        assert dh.docker_available() is False


# ---------------------------------------------------------------------------
# docker_rm
# ---------------------------------------------------------------------------


class TestDockerRm:
    def test_returns_false_when_docker_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: None)
        assert dh.docker_rm("my_container") is False

    def test_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed())
        assert dh.docker_rm("my_container") is True

    def test_returns_true_on_called_process_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_cpe(*a: Any, **kw: Any) -> None:
            raise subprocess.CalledProcessError(1, "docker")

        monkeypatch.setattr(dh.subprocess, "run", raise_cpe)
        assert dh.docker_rm("my_container") is True

    def test_returns_false_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_timeout(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired("docker", 30)

        monkeypatch.setattr(dh.subprocess, "run", raise_timeout)
        assert dh.docker_rm("my_container") is False

    def test_returns_false_on_os_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(OSError("not found")))
        assert dh.docker_rm("my_container") is False

    def test_passes_container_name_to_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        captured: list[list[str]] = []
        monkeypatch.setattr(
            dh.subprocess, "run", lambda cmd, **kw: captured.append(cmd) or _make_completed()
        )
        dh.docker_rm("target_container")
        assert captured[0] == ["docker", "rm", "-f", "target_container"]


# ---------------------------------------------------------------------------
# docker_stop
# ---------------------------------------------------------------------------


class TestDockerStop:
    def test_returns_false_when_docker_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: None)
        assert dh.docker_stop("my_container") is False

    def test_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed())
        assert dh.docker_stop("my_container") is True

    def test_returns_true_on_called_process_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_cpe(*a: Any, **kw: Any) -> None:
            raise subprocess.CalledProcessError(1, "docker")

        monkeypatch.setattr(dh.subprocess, "run", raise_cpe)
        assert dh.docker_stop("my_container") is True

    def test_returns_false_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_timeout(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired("docker", 30)

        monkeypatch.setattr(dh.subprocess, "run", raise_timeout)
        assert dh.docker_stop("my_container") is False

    def test_passes_container_name_to_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        captured: list[list[str]] = []
        monkeypatch.setattr(
            dh.subprocess, "run", lambda cmd, **kw: captured.append(cmd) or _make_completed()
        )
        dh.docker_stop("target_container")
        assert captured[0] == ["docker", "stop", "target_container"]


# ---------------------------------------------------------------------------
# docker_inspect
# ---------------------------------------------------------------------------


class TestDockerInspect:
    def test_returns_none_when_docker_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: None)
        assert dh.docker_inspect("c") is None

    def test_returns_dict_on_valid_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = [{"Id": "abc", "State": {"Status": "running"}}]
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed(json.dumps(payload))
        )
        result = dh.docker_inspect("my_container")
        assert result == payload[0]

    def test_returns_none_on_called_process_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_cpe(*a: Any, **kw: Any) -> None:
            raise subprocess.CalledProcessError(1, "docker")

        monkeypatch.setattr(dh.subprocess, "run", raise_cpe)
        assert dh.docker_inspect("c") is None

    def test_returns_none_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_timeout(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired("docker", 10)

        monkeypatch.setattr(dh.subprocess, "run", raise_timeout)
        assert dh.docker_inspect("c") is None

    def test_returns_none_on_invalid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed("not-json")
        )
        assert dh.docker_inspect("c") is None

    def test_returns_none_on_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed("[]")
        )
        assert dh.docker_inspect("c") is None

    def test_returns_none_when_first_element_not_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed('["just-a-string"]')
        )
        assert dh.docker_inspect("c") is None

    def test_returns_none_when_output_is_not_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed('{"Id": "abc"}')
        )
        assert dh.docker_inspect("c") is None


# ---------------------------------------------------------------------------
# docker_image_exists_locally
# ---------------------------------------------------------------------------


class TestDockerImageExistsLocally:
    def test_returns_false_when_docker_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: None)
        assert dh.docker_image_exists_locally("myimage:latest") is False

    def test_returns_true_when_image_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed())
        assert dh.docker_image_exists_locally("myimage:latest") is True

    def test_returns_false_on_called_process_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_cpe(*a: Any, **kw: Any) -> None:
            raise subprocess.CalledProcessError(1, "docker")

        monkeypatch.setattr(dh.subprocess, "run", raise_cpe)
        assert dh.docker_image_exists_locally("missing:latest") is False

    def test_returns_false_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_timeout(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired("docker", 10)

        monkeypatch.setattr(dh.subprocess, "run", raise_timeout)
        assert dh.docker_image_exists_locally("img") is False

    def test_passes_image_name_to_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        captured: list[list[str]] = []
        monkeypatch.setattr(
            dh.subprocess, "run", lambda cmd, **kw: captured.append(cmd) or _make_completed()
        )
        dh.docker_image_exists_locally("ghcr.io/org/myimage:v1")
        assert captured[0] == ["docker", "image", "inspect", "ghcr.io/org/myimage:v1"]


# ---------------------------------------------------------------------------
# docker_container_status
# ---------------------------------------------------------------------------


class TestDockerContainerStatus:
    def test_returns_none_when_inspect_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh, "docker_inspect", lambda name: None)
        assert dh.docker_container_status("c") == "none"

    def test_returns_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            dh, "docker_inspect", lambda name: {"State": {"Status": "running"}}
        )
        assert dh.docker_container_status("c") == "running"

    def test_returns_exited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            dh, "docker_inspect", lambda name: {"State": {"Status": "exited"}}
        )
        assert dh.docker_container_status("c") == "exited"

    def test_returns_unknown_when_state_not_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh, "docker_inspect", lambda name: {"State": "bad"})
        assert dh.docker_container_status("c") == "unknown"

    def test_status_is_lowercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            dh, "docker_inspect", lambda name: {"State": {"Status": "Running"}}
        )
        assert dh.docker_container_status("c") == "running"

    def test_returns_unknown_when_status_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh, "docker_inspect", lambda name: {"State": {}})
        assert dh.docker_container_status("c") == "unknown"


# ---------------------------------------------------------------------------
# docker_ps_by_prefix
# ---------------------------------------------------------------------------


class TestDockerPsByPrefix:
    def test_returns_empty_list_when_docker_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: None)
        assert dh.docker_ps_by_prefix("cw_") == []

    def test_returns_matching_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = "cw_worker_1\ncw_worker_2\n"
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed(output)
        )
        result = dh.docker_ps_by_prefix("cw_")
        assert result == ["cw_worker_1", "cw_worker_2"]

    def test_strips_whitespace_from_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = "  cw_worker_1  \n  cw_worker_2  \n"
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed(output)
        )
        result = dh.docker_ps_by_prefix("cw_")
        assert result == ["cw_worker_1", "cw_worker_2"]

    def test_returns_empty_list_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_cpe(*a: Any, **kw: Any) -> None:
            raise subprocess.CalledProcessError(1, "docker")

        monkeypatch.setattr(dh.subprocess, "run", raise_cpe)
        assert dh.docker_ps_by_prefix("cw_") == []

    def test_include_stopped_adds_dash_a_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        captured: list[list[str]] = []
        monkeypatch.setattr(
            dh.subprocess, "run", lambda cmd, **kw: captured.append(cmd) or _make_completed("")
        )
        dh.docker_ps_by_prefix("cw_", include_stopped=True)
        assert "-a" in captured[0]

    def test_no_dash_a_flag_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        captured: list[list[str]] = []
        monkeypatch.setattr(
            dh.subprocess, "run", lambda cmd, **kw: captured.append(cmd) or _make_completed("")
        )
        dh.docker_ps_by_prefix("cw_")
        assert "-a" not in captured[0]

    def test_prefix_used_in_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        captured: list[list[str]] = []
        monkeypatch.setattr(
            dh.subprocess, "run", lambda cmd, **kw: captured.append(cmd) or _make_completed("")
        )
        dh.docker_ps_by_prefix("myprefix_")
        cmd = captured[0]
        filter_idx = cmd.index("--filter")
        assert "myprefix_" in cmd[filter_idx + 1]


# ---------------------------------------------------------------------------
# docker_has_nvidia_runtime
# ---------------------------------------------------------------------------


class TestDockerHasNvidiaRuntime:
    def test_returns_false_when_docker_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: None)
        assert dh.docker_has_nvidia_runtime() is False

    def test_returns_true_when_nvidia_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtimes = {"nvidia": {}, "runc": {}}
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed(json.dumps(runtimes))
        )
        assert dh.docker_has_nvidia_runtime() is True

    def test_returns_false_when_nvidia_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtimes = {"runc": {}}
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed(json.dumps(runtimes))
        )
        assert dh.docker_has_nvidia_runtime() is False

    def test_returns_false_on_empty_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed("")
        )
        assert dh.docker_has_nvidia_runtime() is False

    def test_returns_false_on_invalid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed("not-json")
        )
        assert dh.docker_has_nvidia_runtime() is False

    def test_returns_false_on_called_process_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_cpe(*a: Any, **kw: Any) -> None:
            raise subprocess.CalledProcessError(1, "docker")

        monkeypatch.setattr(dh.subprocess, "run", raise_cpe)
        assert dh.docker_has_nvidia_runtime() is False

    def test_returns_false_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_timeout(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired("docker", 10)

        monkeypatch.setattr(dh.subprocess, "run", raise_timeout)
        assert dh.docker_has_nvidia_runtime() is False

    def test_returns_false_when_output_is_not_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed('["nvidia"]')
        )
        assert dh.docker_has_nvidia_runtime() is False


# ---------------------------------------------------------------------------
# docker_logs_follow
# ---------------------------------------------------------------------------


class TestDockerLogsFollow:
    def test_returns_none_when_docker_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: None)
        assert dh.docker_logs_follow("c") is None

    def test_returns_popen_handle_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        fake_proc = MagicMock(spec=subprocess.Popen)
        monkeypatch.setattr(dh.subprocess, "Popen", lambda *a, **kw: fake_proc)
        result = dh.docker_logs_follow("my_container")
        assert result is fake_proc

    def test_returns_none_on_os_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_os_error(*a: Any, **kw: Any) -> None:
            raise OSError("binary not found")

        monkeypatch.setattr(dh.subprocess, "Popen", raise_os_error)
        assert dh.docker_logs_follow("c") is None

    def test_passes_container_name_and_tail_to_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        captured: list[list[str]] = []

        def fake_popen(cmd: list, **kw: Any) -> MagicMock:
            captured.append(cmd)
            return MagicMock(spec=subprocess.Popen)

        monkeypatch.setattr(dh.subprocess, "Popen", fake_popen)
        dh.docker_logs_follow("target_container")
        assert captured[0] == ["docker", "logs", "-f", "--tail", "50", "target_container"]
