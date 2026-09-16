"""Unit tests for cyberwave_edge_core.docker_helpers.

Covers:
- build_user_args: returns --user on Linux, empty on macOS
- docker_available: binary present / absent
- docker_rm: success, CalledProcessError treated as success, timeout/OSError
- docker_stop: success, CalledProcessError treated as success, timeout/OSError
- docker_inspect: valid JSON, bad JSON, empty list, non-list, subprocess errors
- docker_image_exists_locally: present, absent, errors
- docker_container_status: running, exited, inspect returns None, bad State type
- container_status_of: reads an already-fetched payload
- parse_docker_timestamp_ns: nanosecond fidelity, offsets, unparseable input
- docker_ps_by_prefix: returns names, filters by prefix, handles errors,
  include_stopped passes -a flag
- docker_has_nvidia_runtime: nvidia present, absent, empty output, bad JSON, errors
- docker_prune_stopped_cyberwave_containers: removes stopped containers, skips running
- docker_prune_unused_images: success, failure, docker unavailable
- docker_prune_dangling_volumes: command shape, success, failure paths
- docker_prune_navigation_bundle_volumes: labelled named-volume reclamation
- docker_logs_follow: returns Popen handle, docker unavailable, OSError on Popen
- backing_device_for: longest-mountpoint match, sibling-prefix safety
- is_sd_card_path / is_sd_card_root: detects mmcblk backing device on Linux
- docker_data_root: docker info -> daemon.json -> default, and caching
"""

from __future__ import annotations

import json
import os
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
# group_gid
# ---------------------------------------------------------------------------


class TestGroupGid:
    def test_returns_gid_when_group_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import grp

        monkeypatch.setattr(dh.platform, "system", lambda: "Linux")

        class _FakeGroup:
            gr_gid = 1010

        monkeypatch.setattr(grp, "getgrnam", lambda name: _FakeGroup())
        assert dh.group_gid("hailo") == 1010

    def test_returns_none_when_group_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import grp

        monkeypatch.setattr(dh.platform, "system", lambda: "Linux")

        def _raise(name: str) -> Any:
            raise KeyError(name)

        monkeypatch.setattr(grp, "getgrnam", _raise)
        assert dh.group_gid("hailo") is None

    def test_returns_none_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.platform, "system", lambda: "Darwin")
        assert dh.group_gid("hailo") is None


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
    def test_returns_false_when_docker_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: None)
        assert dh.docker_rm("my_container") is False

    def test_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed())
        assert dh.docker_rm("my_container") is True

    def test_returns_true_on_called_process_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(OSError("not found"))
        )
        assert dh.docker_rm("my_container") is False

    def test_passes_container_name_to_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        captured: list[list[str]] = []
        monkeypatch.setattr(
            dh.subprocess, "run", lambda cmd, **kw: captured.append(cmd) or _make_completed()
        )
        dh.docker_rm("target_container")
        # -v reaps the container's anonymous volumes; named volumes and bind
        # mounts are untouched by it.
        assert captured[0] == ["docker", "rm", "-f", "-v", "target_container"]


# ---------------------------------------------------------------------------
# docker_stop
# ---------------------------------------------------------------------------


class TestDockerStop:
    def test_returns_false_when_docker_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: None)
        assert dh.docker_stop("my_container") is False

    def test_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed())
        assert dh.docker_stop("my_container") is True

    def test_returns_true_on_called_process_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
    def test_returns_none_when_docker_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_returns_none_on_called_process_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed("not-json"))
        assert dh.docker_inspect("c") is None

    def test_returns_none_on_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed("[]"))
        assert dh.docker_inspect("c") is None

    def test_returns_none_when_first_element_not_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            dh.subprocess, "run", lambda *a, **kw: _make_completed('["just-a-string"]')
        )
        assert dh.docker_inspect("c") is None

    def test_returns_none_when_output_is_not_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed('{"Id": "abc"}'))
        assert dh.docker_inspect("c") is None


# ---------------------------------------------------------------------------
# docker_image_exists_locally
# ---------------------------------------------------------------------------


class TestDockerImageExistsLocally:
    def test_returns_false_when_docker_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: None)
        assert dh.docker_image_exists_locally("myimage:latest") is False

    def test_returns_true_when_image_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed())
        assert dh.docker_image_exists_locally("myimage:latest") is True

    def test_returns_false_on_called_process_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
    def test_returns_none_when_inspect_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_inspect", lambda name: None)
        assert dh.docker_container_status("c") == "none"

    def test_returns_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_inspect", lambda name: {"State": {"Status": "running"}})
        assert dh.docker_container_status("c") == "running"

    def test_returns_exited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_inspect", lambda name: {"State": {"Status": "exited"}})
        assert dh.docker_container_status("c") == "exited"

    def test_returns_unknown_when_state_not_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_inspect", lambda name: {"State": "bad"})
        assert dh.docker_container_status("c") == "unknown"

    def test_status_is_lowercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_inspect", lambda name: {"State": {"Status": "Running"}})
        assert dh.docker_container_status("c") == "running"

    def test_returns_unknown_when_status_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed(output))
        result = dh.docker_ps_by_prefix("cw_")
        assert result == ["cw_worker_1", "cw_worker_2"]

    def test_strips_whitespace_from_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = "  cw_worker_1  \n  cw_worker_2  \n"
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed(output))
        result = dh.docker_ps_by_prefix("cw_")
        assert result == ["cw_worker_1", "cw_worker_2"]

    def test_returns_empty_list_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")

        def raise_cpe(*a: Any, **kw: Any) -> None:
            raise subprocess.CalledProcessError(1, "docker")

        monkeypatch.setattr(dh.subprocess, "run", raise_cpe)
        assert dh.docker_ps_by_prefix("cw_") == []

    def test_include_stopped_adds_dash_a_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
    def test_returns_false_when_docker_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed(""))
        assert dh.docker_has_nvidia_runtime() is False

    def test_returns_false_on_invalid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed("not-json"))
        assert dh.docker_has_nvidia_runtime() is False

    def test_returns_false_on_called_process_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_returns_false_when_output_is_not_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed('["nvidia"]'))
        assert dh.docker_has_nvidia_runtime() is False


# ---------------------------------------------------------------------------
# docker_has_nvidia_default_runtime
# ---------------------------------------------------------------------------


class TestDockerHasNvidiaDefaultRuntime:
    def test_returns_true_when_nvidia_is_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        daemon_json = tmp_path / "daemon.json"
        daemon_json.write_text(json.dumps({"default-runtime": "nvidia"}))
        monkeypatch.setattr(dh.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            dh,
            "docker_has_nvidia_default_runtime",
            lambda: json.loads(daemon_json.read_text()).get("default-runtime") == "nvidia",
        )
        assert dh.docker_has_nvidia_default_runtime() is True

    def test_returns_false_when_not_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.platform, "system", lambda: "Darwin")
        assert dh.docker_has_nvidia_default_runtime() is False

    def test_returns_false_when_daemon_json_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.platform, "system", lambda: "Linux")
        original = dh.docker_has_nvidia_default_runtime

        import builtins

        real_open = builtins.open

        def fake_open(path: Any, *a: Any, **kw: Any) -> Any:
            if str(path) == "/etc/docker/daemon.json":
                raise FileNotFoundError("no such file")
            return real_open(path, *a, **kw)

        monkeypatch.setattr(builtins, "open", fake_open)
        assert original() is False

    def test_returns_false_when_different_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        daemon_json = tmp_path / "daemon.json"
        daemon_json.write_text(json.dumps({"default-runtime": "runc"}))
        monkeypatch.setattr(dh.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            dh,
            "docker_has_nvidia_default_runtime",
            lambda: json.loads(daemon_json.read_text()).get("default-runtime") == "nvidia",
        )
        assert dh.docker_has_nvidia_default_runtime() is False


# ---------------------------------------------------------------------------
# docker_logs_follow
# ---------------------------------------------------------------------------


class TestDockerLogsFollow:
    def test_returns_none_when_docker_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
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


# ---------------------------------------------------------------------------
# docker_prune_stopped_cyberwave_containers
# ---------------------------------------------------------------------------


class TestDockerPruneStoppedCyberwaveContainers:
    def test_returns_zero_when_docker_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: False)
        assert dh.docker_prune_stopped_cyberwave_containers() == 0

    def test_returns_zero_when_no_stopped_containers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(
            dh,
            "docker_ps_by_prefix",
            lambda prefix, include_stopped=False: (
                ["cyberwave-driver-abc"] if include_stopped else ["cyberwave-driver-abc"]
            ),
        )
        assert dh.docker_prune_stopped_cyberwave_containers() == 0

    def test_removes_stopped_containers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        all_containers = [
            "cyberwave-driver-abc",
            "cyberwave-worker-def",
            "cyberwave-zenoh-router-ghi",
        ]
        running_containers = ["cyberwave-driver-abc"]

        def fake_ps(prefix: str, include_stopped: bool = False) -> list[str]:
            return all_containers if include_stopped else running_containers

        removed_names: list[str] = []

        def fake_rm(name: str) -> bool:
            removed_names.append(name)
            return True

        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(dh, "docker_ps_by_prefix", fake_ps)
        monkeypatch.setattr(dh, "docker_rm", fake_rm)

        result = dh.docker_prune_stopped_cyberwave_containers()
        assert result == 2
        assert "cyberwave-worker-def" in removed_names
        assert "cyberwave-zenoh-router-ghi" in removed_names
        assert "cyberwave-driver-abc" not in removed_names

    def test_counts_failed_removals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_ps(prefix: str, include_stopped: bool = False) -> list[str]:
            return ["cyberwave-a", "cyberwave-b"] if include_stopped else []

        call_count = 0

        def fake_rm(name: str) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count == 1

        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(dh, "docker_ps_by_prefix", fake_ps)
        monkeypatch.setattr(dh, "docker_rm", fake_rm)

        assert dh.docker_prune_stopped_cyberwave_containers() == 1

    def test_uses_custom_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_prefixes: list[str] = []

        def fake_ps(prefix: str, include_stopped: bool = False) -> list[str]:
            captured_prefixes.append(prefix)
            return []

        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(dh, "docker_ps_by_prefix", fake_ps)

        dh.docker_prune_stopped_cyberwave_containers(prefix="my-prefix")
        assert all(p == "my-prefix" for p in captured_prefixes)


# ---------------------------------------------------------------------------
# docker_prune_unused_images
# ---------------------------------------------------------------------------


class TestDockerPruneUnusedImages:
    def test_returns_false_when_docker_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: False)
        assert dh.docker_prune_unused_images() is False

    def test_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed())
        assert dh.docker_prune_unused_images() is True

    def test_returns_false_on_called_process_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)

        def raise_cpe(*a: Any, **kw: Any) -> None:
            raise subprocess.CalledProcessError(1, "docker")

        monkeypatch.setattr(dh.subprocess, "run", raise_cpe)
        assert dh.docker_prune_unused_images() is False

    def test_returns_false_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)

        def raise_timeout(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired("docker", 300)

        monkeypatch.setattr(dh.subprocess, "run", raise_timeout)
        assert dh.docker_prune_unused_images() is False

    def test_returns_false_on_os_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)

        def raise_os_error(*a: Any, **kw: Any) -> None:
            raise OSError("not found")

        monkeypatch.setattr(dh.subprocess, "run", raise_os_error)
        assert dh.docker_prune_unused_images() is False

    def test_passes_correct_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)
        captured: list[list[str]] = []
        monkeypatch.setattr(
            dh.subprocess, "run", lambda cmd, **kw: captured.append(cmd) or _make_completed()
        )
        dh.docker_prune_unused_images()
        # The until=2h guard keeps freshly-pulled images that no container has
        # had a chance to reference yet.
        assert captured[0] == [
            "docker",
            "image",
            "prune",
            "--all",
            "--force",
            "--filter",
            "until=2h",
        ]


# ---------------------------------------------------------------------------
# docker_prune_dangling_volumes
# ---------------------------------------------------------------------------


class TestDockerPruneDanglingVolumes:
    @pytest.fixture(autouse=True)
    def _modern_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """These cases are about the prune itself; open the version gate."""
        monkeypatch.setattr(dh, "volume_prune_is_anonymous_only", lambda: True)

    def test_returns_false_when_docker_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: False)
        assert dh.docker_prune_dangling_volumes() is False

    def test_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed())
        assert dh.docker_prune_dangling_volumes() is True

    def test_returns_false_on_called_process_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)

        def raise_cpe(*a: Any, **kw: Any) -> None:
            raise subprocess.CalledProcessError(1, "docker")

        monkeypatch.setattr(dh.subprocess, "run", raise_cpe)
        assert dh.docker_prune_dangling_volumes() is False

    def test_returns_false_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)

        def raise_timeout(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired("docker", 120)

        monkeypatch.setattr(dh.subprocess, "run", raise_timeout)
        assert dh.docker_prune_dangling_volumes() is False

    def test_command_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both omissions are load-bearing.

        No ``--all`` keeps operator-created named volumes; no ``--filter``
        because the daemon rejects ``until`` for volume prune.
        """
        monkeypatch.setattr(dh, "docker_available", lambda: True)
        captured: list[list[str]] = []
        monkeypatch.setattr(
            dh.subprocess, "run", lambda cmd, **kw: captured.append(cmd) or _make_completed()
        )
        dh.docker_prune_dangling_volumes()
        assert captured[0] == ["docker", "volume", "prune", "--force"]
        assert "--all" not in captured[0]
        assert "--filter" not in captured[0]


class TestDockerPruneNavigationBundleVolumes:
    def test_removes_only_labelled_unused_volumes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)
        captured: list[list[str]] = []

        def run(command: list[str], **_kwargs: Any) -> MagicMock:
            captured.append(command)
            if command[1:3] == ["volume", "ls"]:
                return _make_completed("bundle-old\nbundle-active\n")
            if command[-1] == "bundle-active":
                raise subprocess.CalledProcessError(1, command)
            return _make_completed()

        monkeypatch.setattr(dh.subprocess, "run", run)

        assert dh.docker_prune_navigation_bundle_volumes() is True
        assert captured[0] == [
            "docker",
            "volume",
            "ls",
            "-q",
            "--filter",
            "label=io.cyberwave.navigation.bundle=1",
        ]
        assert captured[1:] == [
            [
                "docker",
                "volume",
                "inspect",
                "--format",
                '{{ index .Labels "io.cyberwave.navigation.created-at" }}',
                "bundle-old",
            ],
            ["docker", "volume", "rm", "bundle-old"],
            [
                "docker",
                "volume",
                "inspect",
                "--format",
                '{{ index .Labels "io.cyberwave.navigation.created-at" }}',
                "bundle-active",
            ],
            ["docker", "volume", "rm", "bundle-active"],
        ]

    def test_preserves_cached_and_newly_created_volumes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(dh.time, "time", lambda: 1_000.0)
        captured: list[list[str]] = []

        def run(command: list[str], **_kwargs: Any) -> MagicMock:
            captured.append(command)
            if command[1:3] == ["volume", "ls"]:
                return _make_completed("bundle-cached\nbundle-new\n")
            if command[1:3] == ["volume", "inspect"]:
                return _make_completed("900\n")
            return _make_completed()

        monkeypatch.setattr(dh.subprocess, "run", run)

        assert dh.docker_prune_navigation_bundle_volumes(
            protected_volumes={"bundle-cached"}, min_age_seconds=300
        ) is True
        assert not any(command[1:3] == ["volume", "rm"] for command in captured)

    def test_returns_false_when_listing_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(
            dh.subprocess,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("docker", 30)),
        )
        assert dh.docker_prune_navigation_bundle_volumes() is False


# ---------------------------------------------------------------------------
# docker_server_version / volume_prune_is_anonymous_only
# ---------------------------------------------------------------------------


class TestVolumePruneVersionGate:
    """Bare ``docker volume prune`` only spares named volumes on Docker >= 23.0.

    Below that it deletes every unused volume on the host, other workloads'
    named ones included.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> Any:
        dh._docker_server_version = None
        yield
        dh._docker_server_version = None

    def _with_version(self, monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(dh.subprocess, "run", lambda *a, **kw: _make_completed(stdout=stdout))

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("24.0.7", (24, 0)),
            ("23.0.0", (23, 0)),
            ("20.10.24", (20, 10)),
            # Vendor builds tack a suffix on: 20.10.25-0ubuntu1~22.04.1
            ("20.10.25-0ubuntu1~22.04.1", (20, 10)),
            ("27", (27, 0)),
        ],
    )
    def test_parses_version(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: tuple[int, int]
    ) -> None:
        self._with_version(monkeypatch, f"{raw}\n")
        assert dh.docker_server_version() == expected

    @pytest.mark.parametrize("raw", ["", "dev", "not.a.version"])
    def test_unparseable_version_is_none(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        self._with_version(monkeypatch, raw)
        assert dh.docker_server_version() is None

    def test_daemon_down_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)

        def raise_timeout(*_a: Any, **_kw: Any) -> None:
            raise subprocess.TimeoutExpired("docker", 15)

        monkeypatch.setattr(dh.subprocess, "run", raise_timeout)
        assert dh.docker_server_version() is None

    def test_version_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = [0]

        def _run(*_a: Any, **_kw: Any) -> Any:
            calls[0] += 1
            return _make_completed(stdout="24.0.7\n")

        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(dh.subprocess, "run", _run)
        dh.docker_server_version()
        dh.docker_server_version()
        assert calls[0] == 1

    def test_failure_is_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A miss stays retryable: Edge Core can start before dockerd."""
        calls = [0]
        stdout = [""]

        def _run(*_a: Any, **_kw: Any) -> Any:
            calls[0] += 1
            return _make_completed(stdout=stdout[0])

        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(dh.subprocess, "run", _run)

        assert dh.docker_server_version() is None
        assert dh.docker_server_version() is None
        assert calls[0] == 2

        # dockerd finished starting; the next probe settles and sticks.
        stdout[0] = "24.0.7\n"
        assert dh.docker_server_version() == (24, 0)
        assert dh.docker_server_version() == (24, 0)
        assert calls[0] == 3

    @pytest.mark.parametrize(
        "raw,allowed",
        [("24.0.7", True), ("23.0.0", True), ("22.9.9", False), ("20.10.24", False)],
    )
    def test_gate_follows_version(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, allowed: bool
    ) -> None:
        self._with_version(monkeypatch, f"{raw}\n")
        assert dh.volume_prune_is_anonymous_only() is allowed

    def test_unknown_version_is_treated_as_old(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh, "docker_server_version", lambda: None)
        assert dh.volume_prune_is_anonymous_only() is False

    def test_old_docker_never_runs_the_prune(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On an old daemon, no prune is issued.

        Asserted against the prune command rather than "no subprocess at all":
        the skip path logs the version, a cache hit in production but a real
        call here because the gate is stubbed out.
        """
        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(dh, "volume_prune_is_anonymous_only", lambda: False)
        captured: list[list[str]] = []
        monkeypatch.setattr(
            dh.subprocess, "run", lambda cmd, **kw: captured.append(cmd) or _make_completed()
        )
        # True, not False — a deliberate skip is not a failure. False is
        # reserved for "we tried and Docker refused".
        assert dh.docker_prune_dangling_volumes() is True
        assert not any("prune" in cmd for cmd in captured)


# ---------------------------------------------------------------------------
# backing_device_for / is_sd_card_path / docker_data_root
# ---------------------------------------------------------------------------


def _fake_proc_mounts(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, content: str) -> None:
    """Redirect reads of /proc/mounts to a fixture file."""
    mounts = tmp_path / "mounts"
    mounts.write_text(content)
    monkeypatch.setattr(dh.platform, "system", lambda: "Linux")

    import builtins

    real_open = builtins.open

    def fake_open(path: Any, *a: Any, **kw: Any) -> Any:
        if str(path) == "/proc/mounts":
            return real_open(str(mounts), *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)


# A Pi that boots from SD but keeps Docker on an external SSD — the case that
# matching on "/" alone gets wrong in both directions.
#
# Mountpoints go through realpath since the host may symlink a component
# (macOS maps /var -> /private/var); queries below use the raw spelling.
_DOCKER_MOUNTPOINT = os.path.realpath("/var/lib/docker")
_SPLIT_MOUNTS = (
    "/dev/mmcblk0p2 / ext4 rw,relatime 0 0\n"
    f"/dev/sda1 {_DOCKER_MOUNTPOINT} ext4 rw,relatime 0 0\n"
    "tmpfs /tmp tmpfs rw 0 0\n"
)


class TestBackingDeviceFor:
    def test_picks_longest_matching_mountpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_proc_mounts(monkeypatch, tmp_path, _SPLIT_MOUNTS)
        assert dh.backing_device_for("/var/lib/docker/overlay2") == "/dev/sda1"
        assert dh.backing_device_for("/etc/hostname") == "/dev/mmcblk0p2"

    def test_does_not_match_sibling_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """/var/lib/docker must not claim /var/lib/docker-backup."""
        _fake_proc_mounts(monkeypatch, tmp_path, _SPLIT_MOUNTS)
        assert dh.backing_device_for("/var/lib/docker-backup") == "/dev/mmcblk0p2"

    def test_returns_none_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.platform, "system", lambda: "Darwin")
        assert dh.backing_device_for("/") is None


class TestIsSdCardPath:
    def test_distinguishes_docker_root_from_boot_card(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """The wear/pressure decision follows the data root, not the boot device."""
        _fake_proc_mounts(monkeypatch, tmp_path, _SPLIT_MOUNTS)
        assert dh.is_sd_card_path("/") is True
        assert dh.is_sd_card_path("/var/lib/docker") is False


class TestDockerDataRoot:
    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> Any:
        dh._docker_data_root = None
        yield
        dh._docker_data_root = None

    def test_prefers_docker_info(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        root = tmp_path / "dockerroot"
        root.mkdir()
        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(
            dh.subprocess,
            "run",
            lambda *a, **kw: _make_completed(stdout=f"{root}\n"),
        )
        assert dh.docker_data_root() == str(root)

    def test_result_is_cached(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        """docker info is a round-trip; the gate runs every ~15 s."""
        root = tmp_path / "dockerroot"
        root.mkdir()
        calls = [0]

        def _run(*_a: Any, **_kw: Any) -> Any:
            calls[0] += 1
            return _make_completed(stdout=f"{root}\n")

        monkeypatch.setattr(dh, "docker_available", lambda: True)
        monkeypatch.setattr(dh.subprocess, "run", _run)
        dh.docker_data_root()
        dh.docker_data_root()
        assert calls[0] == 1

    def test_falls_back_to_root_when_path_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-existent data root would make disk_usage raise."""
        monkeypatch.setattr(dh, "docker_available", lambda: False)

        import builtins

        real_open = builtins.open

        def fake_open(path: Any, *a: Any, **kw: Any) -> Any:
            if str(path) == "/etc/docker/daemon.json":
                raise OSError("no such file")
            return real_open(path, *a, **kw)

        monkeypatch.setattr(builtins, "open", fake_open)
        monkeypatch.setattr(dh.os.path, "isdir", lambda p: False)
        assert dh.docker_data_root() == "/"

    def test_docker_info_failure_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(dh, "docker_available", lambda: True)

        def raise_timeout(*_a: Any, **_kw: Any) -> None:
            raise subprocess.TimeoutExpired("docker", 15)

        monkeypatch.setattr(dh.subprocess, "run", raise_timeout)
        daemon = tmp_path / "daemon.json"
        daemon.write_text(json.dumps({"data-root": "/mnt/docker"}))

        import builtins

        real_open = builtins.open

        def fake_open(path: Any, *a: Any, **kw: Any) -> Any:
            if str(path) == "/etc/docker/daemon.json":
                return real_open(str(daemon), *a, **kw)
            return real_open(path, *a, **kw)

        monkeypatch.setattr(builtins, "open", fake_open)
        monkeypatch.setattr(dh.os.path, "isdir", lambda p: p == "/mnt/docker")
        assert dh.docker_data_root() == "/mnt/docker"


class TestIsSdCardRoot:
    def test_returns_false_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.platform, "system", lambda: "Darwin")
        assert dh.is_sd_card_root() is False

    def test_returns_true_when_root_on_mmcblk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        mounts = tmp_path / "mounts"
        mounts.write_text("/dev/mmcblk0p2 / ext4 rw,relatime 0 0\ntmpfs /tmp tmpfs rw 0 0\n")
        monkeypatch.setattr(dh.platform, "system", lambda: "Linux")

        import builtins

        real_open = builtins.open

        def fake_open(path: Any, *a: Any, **kw: Any) -> Any:
            if str(path) == "/proc/mounts":
                return real_open(str(mounts), *a, **kw)
            return real_open(path, *a, **kw)

        monkeypatch.setattr(builtins, "open", fake_open)
        assert dh.is_sd_card_root() is True

    def test_returns_false_when_root_on_ssd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        mounts = tmp_path / "mounts"
        mounts.write_text("/dev/sda1 / ext4 rw,relatime 0 0\ntmpfs /tmp tmpfs rw 0 0\n")
        monkeypatch.setattr(dh.platform, "system", lambda: "Linux")

        import builtins

        real_open = builtins.open

        def fake_open(path: Any, *a: Any, **kw: Any) -> Any:
            if str(path) == "/proc/mounts":
                return real_open(str(mounts), *a, **kw)
            return real_open(path, *a, **kw)

        monkeypatch.setattr(builtins, "open", fake_open)
        assert dh.is_sd_card_root() is False

    def test_returns_false_when_proc_mounts_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dh.platform, "system", lambda: "Linux")

        import builtins

        real_open = builtins.open

        def fake_open(path: Any, *a: Any, **kw: Any) -> Any:
            if str(path) == "/proc/mounts":
                raise OSError("No such file")
            return real_open(path, *a, **kw)

        monkeypatch.setattr(builtins, "open", fake_open)
        assert dh.is_sd_card_root() is False

    def test_returns_false_when_no_root_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        mounts = tmp_path / "mounts"
        mounts.write_text("tmpfs /tmp tmpfs rw 0 0\n")
        monkeypatch.setattr(dh.platform, "system", lambda: "Linux")

        import builtins

        real_open = builtins.open

        def fake_open(path: Any, *a: Any, **kw: Any) -> Any:
            if str(path) == "/proc/mounts":
                return real_open(str(mounts), *a, **kw)
            return real_open(path, *a, **kw)

        monkeypatch.setattr(builtins, "open", fake_open)
        assert dh.is_sd_card_root() is False


class TestContainerStatusOf:
    """``docker_container_status`` split so one inspect can answer two questions."""

    def test_reads_status_from_an_existing_payload(self):
        assert dh.container_status_of({"State": {"Status": "Running"}}) == "running"

    def test_none_payload_is_none_status(self):
        assert dh.container_status_of(None) == "none"

    @pytest.mark.parametrize("payload", [{}, {"State": "not-a-dict"}, {"State": []}])
    def test_unusable_state_is_unknown(self, payload):
        assert dh.container_status_of(payload) == "unknown"

    def test_docker_container_status_delegates(self, monkeypatch):
        monkeypatch.setattr(dh, "docker_inspect", lambda _: {"State": {"Status": "exited"}})
        assert dh.docker_container_status("c") == "exited"


class TestParseDockerTimestampNs:
    """Timestamps are compared against ``st_mtime_ns``, so precision is load-bearing.

    A float cannot represent nanoseconds at epoch magnitude and ``datetime``
    truncates to microseconds; either can make one side of that comparison
    look older than it is, which would skip a restart the edge needed.
    """

    def test_keeps_all_nine_fractional_digits(self):
        assert (
            dh.parse_docker_timestamp_ns("2026-09-08T14:22:44.233126554Z") == 1788877364_233126554
        )

    def test_no_fractional_part(self):
        assert dh.parse_docker_timestamp_ns("2026-09-08T14:22:44Z") == 1788877364_000000000

    def test_short_fractional_part_is_padded_not_misread(self):
        assert dh.parse_docker_timestamp_ns("2026-09-08T14:22:44.5Z") == 1788877364_500000000

    def test_more_than_nine_digits_is_truncated(self):
        assert (
            dh.parse_docker_timestamp_ns("2026-09-08T14:22:44.233126554999Z")
            == 1788877364_233126554
        )

    @pytest.mark.parametrize(
        "raw",
        [
            "2026-09-08T15:22:44.5+01:00",
            "2026-09-08T15:22:44.5+0100",
        ],
    )
    def test_non_utc_offsets_are_honoured(self, raw):
        assert dh.parse_docker_timestamp_ns(raw) == 1788877364_500000000

    def test_never_started_placeholder_is_unknown(self):
        # Docker reports this for a container that has never run. It must read
        # as "no answer", never as an ancient timestamp that compares older
        # than every possible file mtime.
        assert dh.parse_docker_timestamp_ns("0001-01-01T00:00:00Z") is None

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "not a timestamp",
            "2026-09-08",
            "2026-09-08T14:22:44",  # no offset at all
            "2026-13-45T99:99:99Z",
            "2026-09-08T14:22:44.233126554X",
        ],
    )
    def test_unparseable_input_is_unknown(self, raw):
        assert dh.parse_docker_timestamp_ns(raw) is None
