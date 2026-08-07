"""Container log-size bounding.

Two halves: the per-container ``--log-opt`` block, and the bounded backlog
replayed when the log follower has no ``--since`` anchor.
"""

from __future__ import annotations

import pytest

from cyberwave_edge_core import driver_logs
from cyberwave_edge_core.docker_args import (
    DEFAULT_DOCKER_LOG_MAX_FILE,
    DEFAULT_DOCKER_LOG_MAX_SIZE,
    build_log_args,
)

_MAX_SIZE_ENV = "CYBERWAVE_DOCKER_LOG_MAX_SIZE"
_MAX_FILE_ENV = "CYBERWAVE_DOCKER_LOG_MAX_FILE"


class TestBuildLogArgs:
    def test_defaults_to_bounded_json_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_MAX_SIZE_ENV, raising=False)
        monkeypatch.delenv(_MAX_FILE_ENV, raising=False)
        assert build_log_args() == [
            "--log-driver",
            "json-file",
            "--log-opt",
            f"max-size={DEFAULT_DOCKER_LOG_MAX_SIZE}",
            "--log-opt",
            f"max-file={DEFAULT_DOCKER_LOG_MAX_FILE}",
        ]

    def test_env_overrides_size_and_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_MAX_SIZE_ENV, "5m")
        monkeypatch.setenv(_MAX_FILE_ENV, "10")
        assert build_log_args() == [
            "--log-driver",
            "json-file",
            "--log-opt",
            "max-size=5m",
            "--log-opt",
            "max-file=10",
        ]

    def test_off_yields_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Escape hatch for a daemon-wide driver (journald, fluentd, ...)."""
        monkeypatch.setenv(_MAX_SIZE_ENV, "OFF")
        assert build_log_args() == []

    def test_blank_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_MAX_SIZE_ENV, "   ")
        monkeypatch.setenv(_MAX_FILE_ENV, "")
        assert f"max-size={DEFAULT_DOCKER_LOG_MAX_SIZE}" in build_log_args()
        assert f"max-file={DEFAULT_DOCKER_LOG_MAX_FILE}" in build_log_args()


class TestOperatorParamsOverride:
    """Driver ``docker_run_params`` win, at the granularity they were set."""

    @pytest.mark.parametrize(
        "params",
        [
            ["--log-driver", "journald"],
            ["--log-driver=journald"],
            ["--privileged", "--log-driver", "none"],
        ],
    )
    def test_explicit_driver_suppresses_everything(self, params: list[str]) -> None:
        """--log-opt is json-file-specific; the daemon rejects it for `none`."""
        assert build_log_args(params) == []

    def test_operator_max_size_kept_and_max_file_still_supplied(self) -> None:
        """A pinned key is left alone; the unset one still gets a bound.

        Any --log-opt used to suppress the whole block, so max-size=1g
        silently dropped the max-file cap.
        """
        args = build_log_args(["--log-opt", "max-size=1g"])
        assert args == [
            "--log-driver",
            "json-file",
            "--log-opt",
            f"max-file={DEFAULT_DOCKER_LOG_MAX_FILE}",
        ]

    def test_equals_form_is_parsed(self) -> None:
        args = build_log_args(["--log-opt=max-file=10"])
        assert "--log-driver" in args
        assert f"max-size={DEFAULT_DOCKER_LOG_MAX_SIZE}" in args
        assert not any(a.startswith("max-file=") for a in args)

    def test_both_keys_pinned_leaves_only_the_driver(self) -> None:
        args = build_log_args(["--log-opt", "max-size=1g", "--log-opt", "max-file=9"])
        assert args == ["--log-driver", "json-file"]

    @pytest.mark.parametrize(
        "params",
        [
            [],
            ["--privileged"],
            ["--device", "/dev/ttyACM0:/dev/ttyACM0"],
            # Must not false-positive on a value that merely contains the flag
            # name, e.g. an env var mentioning it.
            ["-e", "NOTES=--log-driver is unset"],
        ],
    )
    def test_unrelated_params_get_full_defaults(self, params: list[str]) -> None:
        assert build_log_args(params) == build_log_args()

    def test_never_repeats_a_log_opt_key(self) -> None:
        """Docker silently takes the last of a repeated key.

        Emitting one value per key keeps the result independent of ordering.
        """
        combined = ["--log-opt", "max-size=1g"] + build_log_args(["--log-opt", "max-size=1g"])
        keys = [
            combined[i + 1].split("=", 1)[0]
            for i, a in enumerate(combined)
            if a == "--log-opt" and i + 1 < len(combined)
        ]
        assert len(keys) == len(set(keys))


class TestColdStartTailBound:
    def test_defaults_to_bounded_tail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(driver_logs.DRIVER_LOG_COLD_START_TAIL_ENV_VAR, raising=False)
        assert (
            driver_logs._cold_start_tail_lines() == driver_logs.DEFAULT_DRIVER_LOG_COLD_START_TAIL
        )

    @pytest.mark.parametrize("raw,expected", [("50", 50), ("0", 0), ("-5", 0), ("junk", 500)])
    def test_env_parsing(self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: int) -> None:
        monkeypatch.setenv(driver_logs.DRIVER_LOG_COLD_START_TAIL_ENV_VAR, raw)
        assert driver_logs._cold_start_tail_lines() == expected

    def test_cold_start_bounds_backlog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No --since anchor => --tail, so a restart can't replay everything."""
        cmd = self._capture_logs_cmd(monkeypatch, last_seen=None)
        assert "--tail" in cmd
        assert cmd[cmd.index("--tail") + 1] == str(driver_logs.DEFAULT_DRIVER_LOG_COLD_START_TAIL)
        assert "--since" not in cmd

    def test_resume_uses_since_and_no_tail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a resume anchor, --since already bounds the replay."""
        cmd = self._capture_logs_cmd(monkeypatch, last_seen="2026-08-07T12:00:00Z")
        assert cmd[cmd.index("--since") + 1] == "2026-08-07T12:00:00Z"
        assert "--tail" not in cmd

    def test_zero_restores_unbounded_replay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(driver_logs.DRIVER_LOG_COLD_START_TAIL_ENV_VAR, "0")
        cmd = self._capture_logs_cmd(monkeypatch, last_seen=None)
        assert "--tail" not in cmd

    @staticmethod
    def _capture_logs_cmd(monkeypatch: pytest.MonkeyPatch, *, last_seen: str | None) -> list[str]:
        """Run the follower against a fake Popen and return the argv it built."""
        container = "cyberwave-driver-test"
        monkeypatch.setattr(driver_logs.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(
            driver_logs,
            "_resolve_driver_log_publish_context",
            lambda **kwargs: (None, None),
        )
        driver_logs._CONTAINER_LOG_LAST_SEEN.pop(container, None)
        if last_seen is not None:
            driver_logs._CONTAINER_LOG_LAST_SEEN[container] = last_seen

        captured: list[list[str]] = []

        class _Proc:
            stdout: list[str] = []

            def wait(self, timeout: int | None = None) -> int:
                return 0

            def kill(self) -> None:
                return None

        def _fake_popen(cmd: list[str], **_kwargs: object) -> _Proc:
            captured.append(cmd)
            return _Proc()

        monkeypatch.setattr(driver_logs.subprocess, "Popen", _fake_popen)
        try:
            driver_logs._follow_container_logs(container)
        finally:
            driver_logs._CONTAINER_LOG_LAST_SEEN.pop(container, None)
        return captured[0]
