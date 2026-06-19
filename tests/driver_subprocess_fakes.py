"""Shared fakes for driver container launch unit tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


def fake_docker_start_popen(commands: list[list[str]]) -> Any:
    """Record ``docker start`` and return a process handle that looks finished."""

    def _popen(cmd: list[str], **kwargs: Any) -> MagicMock:
        commands.append(list(cmd))
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        return proc

    return _popen


def find_driver_create_cmd(commands: list[list[str]]) -> list[str]:
    """Return the first ``docker create`` argv captured in *commands*."""
    matches = [cmd for cmd in commands if len(cmd) >= 2 and cmd[:2] == ["docker", "create"]]
    if not matches:
        raise AssertionError(f"No docker create command in: {commands}")
    return matches[0]
