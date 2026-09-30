"""Tests for edge-core owning its systemd unit (auto-update plan B1)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

import cyberwave_edge_core.main as main_module
from cyberwave_edge_core import systemd_unit
from cyberwave_edge_core.systemd_unit import EnsureOutcome, ensure_systemd_unit, render_unit

REPO_ROOT = Path(__file__).resolve().parent.parent
POSTINST = REPO_ROOT / "packaging" / "debian" / "postinst"

# The unit the CLI wrote before B1, plus the ExecStartPre line. Any change to
# this snapshot changes what every edge device runs: update it deliberately.
EXPECTED_UNIT = """\
[Unit]
Description=Cyberwave Edge Core Orchestrator
After=network-online.target docker.service
Wants=network-online.target docker.service

[Service]
Type=notify
NotifyAccess=all
ExecStartPre=-/bin/sh -c 'if [ -x /usr/lib/cyberwave-edge-core/update-guard ]; then exec /usr/lib/cyberwave-edge-core/update-guard; fi'
ExecStart=/usr/bin/cyberwave-edge-core
Restart=always
RestartSec=5
WatchdogSec=60
TimeoutStartSec=900
Environment=CYBERWAVE_EDGE_CONFIG_DIR=/home/pi/.cyberwave
OOMScoreAdjust=-800
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cyberwave-edge-core

[Install]
WantedBy=multi-user.target
"""  # noqa: E501

# What ``cyberwave edge install`` wrote before B1 (cyberwave_cli/core.py).
LEGACY_CLI_UNIT = EXPECTED_UNIT.replace(
    "ExecStartPre=" + systemd_unit.UPDATE_GUARD_EXEC_START_PRE + "\n", ""
)


class RecordingRunner:
    def __init__(self, fail_stderr: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_stderr = fail_stderr

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        if self.fail_stderr is not None:
            raise subprocess.CalledProcessError(1, list(cmd), output="", stderr=self.fail_stderr)
        return subprocess.CompletedProcess(list(cmd), 0, "", "")


@pytest.fixture
def systemd_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run-systemd-system"
    path.mkdir()
    return path


def _ensure(unit_path: Path, systemd_dir: Path, runner: RecordingRunner, **kwargs):
    return ensure_systemd_unit(
        unit_path=unit_path, systemd_runtime_dir=systemd_dir, runner=runner, **kwargs
    )


# ---- rendering ---------------------------------------------------------------


def test_rendered_unit_snapshot() -> None:
    rendered = render_unit(
        binary_path="/usr/bin/cyberwave-edge-core", config_dir="/home/pi/.cyberwave"
    )
    assert rendered == EXPECTED_UNIT


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_exec_start_pre_command_succeeds_without_guard(tmp_path: Path) -> None:
    # Run the exact shell snippet systemd will run, pointed at a missing path.
    script = systemd_unit.UPDATE_GUARD_EXEC_START_PRE.split(" -c ", 1)[1].strip("'")
    script = script.replace(systemd_unit.UPDATE_GUARD_PATH, str(tmp_path / "missing-guard"))
    assert subprocess.run(["sh", "-c", script]).returncode == 0

    guard = tmp_path / "guard"
    guard.write_text('#!/bin/sh\necho ran > "$0.out"\nexit 0\n')
    guard.chmod(0o755)
    script = script.replace(str(tmp_path / "missing-guard"), str(guard))
    assert subprocess.run(["sh", "-c", script]).returncode == 0
    assert (tmp_path / "guard.out").read_text().strip() == "ran"


def test_systemd_analyze_verify_accepts_unit(tmp_path: Path) -> None:
    analyze = shutil.which("systemd-analyze")
    if analyze is None:
        pytest.skip("systemd-analyze not available")
    unit = tmp_path / "cyberwave-edge-core.service"
    unit.write_text(
        render_unit(binary_path="/bin/true", config_dir=str(tmp_path)), encoding="utf-8"
    )
    result = subprocess.run(
        [analyze, "verify", "--man=no", str(unit)], capture_output=True, text=True
    )
    # Only fail on problems in our own unit (dependencies like docker.service
    # may be absent on the test host).
    own_errors = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if unit.name in line and "docker.service" not in line
    ]
    assert own_errors == [], own_errors


def test_parse_existing_settings() -> None:
    exec_start, config_dir = systemd_unit.parse_existing_settings(LEGACY_CLI_UNIT)
    assert exec_start == "/usr/bin/cyberwave-edge-core"
    assert config_dir == "/home/pi/.cyberwave"

    quoted = 'ExecStart=/opt/x/bin/edge\nEnvironment="CYBERWAVE_EDGE_CONFIG_DIR=/srv/cw"\n'
    assert systemd_unit.parse_existing_settings(quoted) == ("/opt/x/bin/edge", "/srv/cw")
    assert systemd_unit.parse_existing_settings("") == (None, None)


# ---- ensure_systemd_unit -----------------------------------------------------


def test_no_unit_and_no_install_writes_nothing(tmp_path: Path, systemd_dir: Path) -> None:
    unit_path = tmp_path / "cyberwave-edge-core.service"
    runner = RecordingRunner()

    result = _ensure(unit_path, systemd_dir, runner)

    assert result.outcome is EnsureOutcome.SKIPPED_NO_UNIT
    assert not unit_path.exists()
    assert runner.calls == []


def test_install_creates_unit_and_reloads(tmp_path: Path, systemd_dir: Path) -> None:
    unit_path = tmp_path / "cyberwave-edge-core.service"
    runner = RecordingRunner()

    result = _ensure(
        unit_path,
        systemd_dir,
        runner,
        install=True,
        binary_path="/usr/bin/cyberwave-edge-core",
        config_dir="/home/pi/.cyberwave",
    )

    assert result.outcome is EnsureOutcome.WRITTEN
    assert result.daemon_reloaded is True
    assert unit_path.read_text() == EXPECTED_UNIT
    assert oct(unit_path.stat().st_mode & 0o777) == oct(0o644)
    assert runner.calls == [["systemctl", "daemon-reload"]]


def test_second_run_is_a_no_op(tmp_path: Path, systemd_dir: Path) -> None:
    unit_path = tmp_path / "cyberwave-edge-core.service"
    runner = RecordingRunner()
    kwargs = dict(binary_path="/usr/bin/cyberwave-edge-core", config_dir="/home/pi/.cyberwave")

    _ensure(unit_path, systemd_dir, runner, install=True, **kwargs)
    mtime = unit_path.stat().st_mtime_ns
    runner.calls.clear()

    result = _ensure(unit_path, systemd_dir, runner, install=True, **kwargs)
    assert result.outcome is EnsureOutcome.UNCHANGED
    # Postinst path (no flags) on the same unit is also a no-op.
    result = _ensure(unit_path, systemd_dir, runner)
    assert result.outcome is EnsureOutcome.UNCHANGED

    assert runner.calls == []
    assert unit_path.stat().st_mtime_ns == mtime


def test_existing_legacy_unit_is_upgraded_preserving_parameters(
    tmp_path: Path, systemd_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The postinst runs as root without SUDO_USER: the defaults would point at
    # /root/.cyberwave. The existing unit's values must win.
    monkeypatch.delenv("CYBERWAVE_EDGE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("SUDO_USER", raising=False)
    unit_path = tmp_path / "cyberwave-edge-core.service"
    legacy = LEGACY_CLI_UNIT.replace(
        "ExecStart=/usr/bin/cyberwave-edge-core",
        "ExecStart=/home/pi/.cyberwave-cli/venv-local/bin/cyberwave-edge-core",
    ).replace("/home/pi/.cyberwave\n", "/srv/cyberwave\n")
    unit_path.write_text(legacy)
    runner = RecordingRunner()

    result = _ensure(unit_path, systemd_dir, runner)

    assert result.outcome is EnsureOutcome.WRITTEN
    text = unit_path.read_text()
    assert "ExecStart=/home/pi/.cyberwave-cli/venv-local/bin/cyberwave-edge-core\n" in text
    assert "Environment=CYBERWAVE_EDGE_CONFIG_DIR=/srv/cyberwave\n" in text
    assert "ExecStartPre=" in text
    assert runner.calls == [["systemctl", "daemon-reload"]]


def test_explicit_values_override_existing_unit(tmp_path: Path, systemd_dir: Path) -> None:
    unit_path = tmp_path / "cyberwave-edge-core.service"
    unit_path.write_text(LEGACY_CLI_UNIT)
    runner = RecordingRunner()

    _ensure(unit_path, systemd_dir, runner, install=True, config_dir="/data/cw")

    assert "Environment=CYBERWAVE_EDGE_CONFIG_DIR=/data/cw\n" in unit_path.read_text()


def test_install_without_existing_unit_uses_defaults(
    tmp_path: Path, systemd_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CYBERWAVE_EDGE_CONFIG_DIR", "/env/cw")
    unit_path = tmp_path / "cyberwave-edge-core.service"

    _ensure(unit_path, systemd_dir, RecordingRunner(), install=True)

    text = unit_path.read_text()
    assert "ExecStart=/usr/bin/cyberwave-edge-core\n" in text
    assert "Environment=CYBERWAVE_EDGE_CONFIG_DIR=/env/cw\n" in text


def test_no_daemon_reload_without_systemd(tmp_path: Path) -> None:
    unit_path = tmp_path / "cyberwave-edge-core.service"
    runner = RecordingRunner()

    result = _ensure(unit_path, tmp_path / "missing", runner, install=True, config_dir="/x")

    assert result.outcome is EnsureOutcome.WRITTEN
    assert result.daemon_reloaded is False
    assert unit_path.exists()
    assert runner.calls == []


def test_daemon_reload_failure_is_reported_not_raised(tmp_path: Path, systemd_dir: Path) -> None:
    unit_path = tmp_path / "cyberwave-edge-core.service"
    runner = RecordingRunner(fail_stderr="Access denied\n")

    result = _ensure(unit_path, systemd_dir, runner, install=True, config_dir="/x")

    assert result.outcome is EnsureOutcome.WRITTEN
    assert result.daemon_reload_error is not None
    assert "Access denied" in result.daemon_reload_error
    assert unit_path.exists()


def test_never_restarts_across_all_paths(tmp_path: Path, systemd_dir: Path) -> None:
    unit_path = tmp_path / "cyberwave-edge-core.service"
    runner = RecordingRunner()

    _ensure(unit_path, systemd_dir, runner)  # skipped
    _ensure(unit_path, systemd_dir, runner, install=True, config_dir="/a")  # written
    _ensure(unit_path, systemd_dir, runner)  # unchanged
    _ensure(unit_path, systemd_dir, runner, config_dir="/b")  # rewritten

    assert runner.calls == [["systemctl", "daemon-reload"]] * 2


# ---- hidden CLI command ------------------------------------------------------


def _patch_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, list]:
    unit_path = tmp_path / "cyberwave-edge-core.service"
    systemd_dir = tmp_path / "run-systemd"
    systemd_dir.mkdir()
    calls: list = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    monkeypatch.setattr(systemd_unit, "UNIT_PATH", unit_path)
    monkeypatch.setattr(systemd_unit, "SYSTEMD_RUNTIME_DIR", systemd_dir)
    monkeypatch.setattr(systemd_unit.subprocess, "run", fake_run)
    return unit_path, calls


def test_command_is_hidden_from_help() -> None:
    result = CliRunner().invoke(main_module.cli, ["--help"])
    assert result.exit_code == 0
    assert "__ensure-systemd-unit" not in result.output


def test_command_without_unit_is_a_no_op(monkeypatch, tmp_path) -> None:
    unit_path, calls = _patch_defaults(monkeypatch, tmp_path)

    result = CliRunner().invoke(main_module.cli, ["__ensure-systemd-unit"])

    assert result.exit_code == 0, result.output
    assert "nothing to do" in result.output
    assert not unit_path.exists()
    assert calls == []


def test_command_install_then_rerun(monkeypatch, tmp_path) -> None:
    unit_path, calls = _patch_defaults(monkeypatch, tmp_path)
    args = [
        "__ensure-systemd-unit",
        "--install",
        "--binary-path",
        "/usr/bin/cyberwave-edge-core",
        "--config-dir",
        "/home/pi/.cyberwave",
    ]

    first = CliRunner().invoke(main_module.cli, args)
    second = CliRunner().invoke(main_module.cli, ["__ensure-systemd-unit"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert unit_path.read_text() == EXPECTED_UNIT
    assert "up to date" in second.output
    assert calls == [["systemctl", "daemon-reload"]]


def test_command_reports_write_failure(monkeypatch, tmp_path) -> None:
    _patch_defaults(monkeypatch, tmp_path)

    def boom(**kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(systemd_unit, "ensure_systemd_unit", boom)

    result = CliRunner().invoke(main_module.cli, ["__ensure-systemd-unit", "--install"])

    assert result.exit_code == 1
    assert "denied" in result.output


def test_unknown_subcommand_does_not_start_orchestrator(monkeypatch) -> None:
    # An older binary that lacks __ensure-systemd-unit must fail fast with a
    # usage error (the CLI's fallback cue), not boot the orchestrator.
    import cyberwave_edge_core.watchdog as watchdog_module

    def fail(*_args, **_kwargs):
        raise AssertionError("orchestrator started")

    monkeypatch.setattr(watchdog_module, "protect_edge_core_oom", fail)
    monkeypatch.setattr(watchdog_module, "ProcessWatchdog", fail)

    result = CliRunner().invoke(main_module.cli, ["__no-such-hidden-command"])

    assert result.exit_code == 2
    assert "No such command" in result.output


# ---- postinst ----------------------------------------------------------------


def test_postinst_is_executable_posix_sh() -> None:
    assert POSTINST.exists()
    assert os.access(POSTINST, os.X_OK)
    assert POSTINST.read_text().startswith("#!/bin/sh\n")
    assert subprocess.run(["sh", "-n", str(POSTINST)]).returncode == 0


def _run_postinst(tmp_path: Path, *, systemd: bool, unit: bool, binary_rc: int, arg="configure"):
    """Run the postinst with its absolute paths redirected into ``tmp_path``."""
    root = tmp_path / "root"
    (root / "run/systemd/system").mkdir(parents=True) if systemd else root.mkdir()
    unit_path = root / "etc/systemd/system/cyberwave-edge-core.service"
    if unit:
        unit_path.parent.mkdir(parents=True)
        unit_path.write_text("old")
    log = tmp_path / "calls.log"
    binary = root / "usr/bin/cyberwave-edge-core"
    binary.parent.mkdir(parents=True)
    binary.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexit {binary_rc}\n')
    binary.chmod(0o755)

    script = (
        POSTINST.read_text()
        .replace("/run/systemd/system", str(root / "run/systemd/system"))
        .replace("/etc/systemd/system/", str(root / "etc/systemd/system") + "/")
        .replace("/usr/bin/cyberwave-edge-core", str(binary))
    )
    patched = tmp_path / "postinst"
    patched.write_text(script)
    patched.chmod(0o755)
    result = subprocess.run([str(patched), arg], capture_output=True, text=True)
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


def test_postinst_calls_binary_when_unit_exists(tmp_path: Path) -> None:
    result, calls = _run_postinst(tmp_path, systemd=True, unit=True, binary_rc=0)
    assert result.returncode == 0
    assert calls == ["__ensure-systemd-unit"]


@pytest.mark.parametrize(
    "systemd,unit,arg",
    [
        (False, True, "configure"),  # non-systemd host
        (True, False, "configure"),  # never ran `cyberwave edge install`
        (True, True, "abort-upgrade"),  # not a configure call
    ],
)
def test_postinst_skips(tmp_path: Path, systemd: bool, unit: bool, arg: str) -> None:
    result, calls = _run_postinst(tmp_path, systemd=systemd, unit=unit, binary_rc=0, arg=arg)
    assert result.returncode == 0
    assert calls == []


def test_postinst_never_fails_the_install(tmp_path: Path) -> None:
    result, calls = _run_postinst(tmp_path, systemd=True, unit=True, binary_rc=3)
    assert result.returncode == 0
    assert calls == ["__ensure-systemd-unit"]
    assert "could not refresh" in result.stderr


def test_postinst_never_restarts(tmp_path: Path) -> None:
    text = POSTINST.read_text()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "systemctl" not in code
    assert "restart" not in code


# ---- running systemctl from the frozen binary --------------------------------


def test_host_tool_env_restores_pyinstaller_library_path(monkeypatch) -> None:
    monkeypatch.setattr(systemd_unit.sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEIabc")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/opt/lib")
    env = systemd_unit.host_tool_env()
    assert env["LD_LIBRARY_PATH"] == "/opt/lib"
    assert "LD_LIBRARY_PATH_ORIG" not in env


def test_host_tool_env_drops_bundle_library_path(monkeypatch) -> None:
    monkeypatch.setattr(systemd_unit.sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEIabc")
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    assert "LD_LIBRARY_PATH" not in systemd_unit.host_tool_env()


def test_host_tool_env_is_untouched_when_not_frozen(monkeypatch) -> None:
    monkeypatch.delattr(systemd_unit.sys, "frozen", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/lib")
    assert systemd_unit.host_tool_env()["LD_LIBRARY_PATH"] == "/usr/local/lib"


def test_default_runner_detaches_stdin_and_cleans_env(monkeypatch) -> None:
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(systemd_unit.subprocess, "run", fake_run)
    monkeypatch.setattr(systemd_unit.sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEIabc")
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)

    systemd_unit._default_runner(["systemctl", "daemon-reload"])

    assert seen["stdin"] is subprocess.DEVNULL
    assert "LD_LIBRARY_PATH" not in seen["env"]


def test_postinst_detaches_the_binary_from_the_terminal() -> None:
    calls = [
        line.strip()
        for line in POSTINST.read_text().splitlines()
        if "__ensure-systemd-unit" in line and not line.lstrip().startswith("#")
    ]
    assert calls
    assert all(call.endswith("</dev/null") for call in calls)
