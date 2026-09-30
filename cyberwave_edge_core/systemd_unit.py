"""Own the ``cyberwave-edge-core.service`` systemd unit.

The unit template used to live in the Cyberwave CLI (``cyberwave edge
install``). It now lives here so that the unit ships with the binary it
starts: a package upgrade (``DEBIAN/postinst``) or the CLI calls the hidden
``cyberwave-edge-core __ensure-systemd-unit`` command, which re-renders the
unit from this template.

Rules (edge-core auto-update plan, B1):

* The unit is written only when it already exists, or when ``install=True``
  (``--install``). A package install on a host that never ran
  ``cyberwave edge install`` must not start managing a service.
* It is written only when the rendered content differs, and then
  ``systemctl daemon-reload`` runs. A second run with the same inputs writes
  nothing and reloads nothing.
* It **never** restarts, starts or stops the service. The new unit takes
  effect on the next process start.

Values that the CLI parameterizes (the ``ExecStart`` command and
``CYBERWAVE_EDGE_CONFIG_DIR``) are preserved from the existing unit unless
passed explicitly. This matters for the postinst: it runs as root without
``SUDO_USER``, so the default config dir would resolve to ``/root/.cyberwave``
instead of the directory the operator installed with.

This module deliberately imports nothing heavy, so it adds no cost to the
hidden command beyond starting the ``cyberwave-edge-core`` binary itself.
"""

from __future__ import annotations

import enum
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

UNIT_NAME = "cyberwave-edge-core.service"
UNIT_PATH = Path(f"/etc/systemd/system/{UNIT_NAME}")
DEFAULT_BINARY_PATH = "/usr/bin/cyberwave-edge-core"
UPDATE_GUARD_PATH = "/usr/lib/cyberwave-edge-core/update-guard"
CONFIG_DIR_ENV_VAR = "CYBERWAVE_EDGE_CONFIG_DIR"
SYSTEMD_RUNTIME_DIR = Path("/run/systemd/system")

# The update guard (plan B2) is shipped by a later package version. Until it
# lands the file does not exist, and even afterwards a manually installed
# binary may lack it, so the pre-start step must never block the service:
#
# * The leading ``-`` tells systemd to ignore a failing exit status of the
#   command (systemd.service(5), "ExecStart="). systemd 249 (Ubuntu 22.04)
#   also treats a *missing* executable under ``-`` as success, logging
#   "Executable ... missing, skipping" (src/core/execute.c, exec_child).
# * The ``/bin/sh`` test makes the missing-file case a plain success on every
#   systemd version, including older ones (Ubuntu 20.04 / JetPack 5 ship 245)
#   where we could not confirm the missing-executable behaviour, and keeps
#   the unit clean under ``systemd-analyze verify``.
UPDATE_GUARD_EXEC_START_PRE = (
    f"-/bin/sh -c 'if [ -x {UPDATE_GUARD_PATH} ]; then exec {UPDATE_GUARD_PATH}; fi'"
)

# Keep functionally identical to the template the CLI wrote before B1
# (``cyberwave_cli/core.py``), apart from the ``ExecStartPre=`` line.
UNIT_TEMPLATE = textwrap.dedent("""\
    [Unit]
    Description=Cyberwave Edge Core Orchestrator
    After=network-online.target docker.service
    Wants=network-online.target docker.service

    [Service]
    Type=notify
    NotifyAccess=all
    ExecStartPre={exec_start_pre}
    ExecStart={binary_path}
    Restart=always
    RestartSec=5
    WatchdogSec=60
    TimeoutStartSec=900
    Environment=CYBERWAVE_EDGE_CONFIG_DIR={config_dir}
    OOMScoreAdjust=-800
    StandardOutput=journal
    StandardError=journal
    SyslogIdentifier=cyberwave-edge-core

    [Install]
    WantedBy=multi-user.target
""")

_EXEC_START_RE = re.compile(r"^\s*ExecStart\s*=\s*(\S.*?)\s*$", re.MULTILINE)
_CONFIG_DIR_RE = re.compile(
    r"^\s*Environment\s*=\s*\"?" + CONFIG_DIR_ENV_VAR + r"=(.*?)\"?\s*$", re.MULTILINE
)

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


class EnsureOutcome(enum.Enum):
    """What ``ensure_systemd_unit`` did."""

    SKIPPED_NO_UNIT = "skipped_no_unit"
    UNCHANGED = "unchanged"
    WRITTEN = "written"


@dataclass(frozen=True)
class EnsureResult:
    outcome: EnsureOutcome
    unit_path: Path
    daemon_reloaded: bool = False
    daemon_reload_error: Optional[str] = None


def render_unit(*, binary_path: str, config_dir: str) -> str:
    """Render the unit file for ``binary_path`` and ``config_dir``."""
    return UNIT_TEMPLATE.format(
        exec_start_pre=UPDATE_GUARD_EXEC_START_PRE,
        binary_path=binary_path,
        config_dir=config_dir,
    )


def parse_existing_settings(unit_text: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(exec_start, config_dir)`` from an existing unit, if present.

    Only the first non-empty ``ExecStart=`` counts: the base unit never has
    the reset form (``ExecStart=`` with no value) that drop-ins use.
    """
    return _first_group(_EXEC_START_RE, unit_text), _first_group(_CONFIG_DIR_RE, unit_text)


def _first_group(pattern: re.Pattern[str], text: str) -> Optional[str]:
    match = pattern.search(text)
    return (match.group(1) or None) if match else None


def default_config_dir() -> str:
    """Mirror edge-core's own config-dir resolution (``startup._resolve_config_dir``)."""
    override = os.getenv(CONFIG_DIR_ENV_VAR, "").strip()
    if override:
        return override
    home: Optional[Path] = None
    sudo_user = os.getenv("SUDO_USER", "").strip()
    if sudo_user:
        try:
            import pwd

            pw_dir = pwd.getpwnam(sudo_user).pw_dir
            home = Path(pw_dir) if pw_dir else None
        except Exception:
            home = None
    return str((home or Path.home()) / ".cyberwave")


def host_tool_env() -> dict[str, str]:
    """Environment for host tools such as ``systemctl`` started from this process.

    The frozen (PyInstaller) binary runs with ``LD_LIBRARY_PATH`` pointing at its
    bundled libraries, and a host binary must not load those. PyInstaller keeps
    the caller's value in ``LD_LIBRARY_PATH_ORIG``: restore it, or drop the
    variable when there was none.
    """
    env = dict(os.environ)
    if getattr(sys, "frozen", False):
        original = env.pop("LD_LIBRARY_PATH_ORIG", None)
        if original is None:
            env.pop("LD_LIBRARY_PATH", None)
        else:
            env["LD_LIBRARY_PATH"] = original
    return env


def _default_runner(cmd: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    # No stdin: from a dpkg maintainer script this may run in a background
    # process group, where touching the terminal stops the process.
    return subprocess.run(
        list(cmd),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
        env=host_tool_env(),
    )


def _describe_failure(exc: BaseException) -> str:
    detail = ""
    if isinstance(exc, subprocess.CalledProcessError):
        detail = (exc.stderr or exc.stdout or "").strip()
    summary = str(exc) or type(exc).__name__
    return f"{summary}: {detail}" if detail else summary


def _write_atomically(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(contents)
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def ensure_systemd_unit(
    *,
    install: bool = False,
    binary_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    unit_path: Optional[Path] = None,
    systemd_runtime_dir: Optional[Path] = None,
    runner: Optional[Runner] = None,
) -> EnsureResult:
    """Write the unit if needed and reload systemd. Never restarts the service.

    Raises ``OSError`` (e.g. ``PermissionError``) when the unit cannot be
    read or written.
    """
    unit_path = unit_path or UNIT_PATH
    systemd_runtime_dir = systemd_runtime_dir or SYSTEMD_RUNTIME_DIR
    runner = runner or _default_runner
    existing: Optional[str] = None
    if unit_path.exists():
        existing = unit_path.read_text(encoding="utf-8")
    elif not install:
        return EnsureResult(EnsureOutcome.SKIPPED_NO_UNIT, unit_path)

    existing_exec, existing_config_dir = parse_existing_settings(existing or "")
    rendered = render_unit(
        binary_path=binary_path or existing_exec or DEFAULT_BINARY_PATH,
        config_dir=config_dir or existing_config_dir or default_config_dir(),
    )

    if existing == rendered:
        return EnsureResult(EnsureOutcome.UNCHANGED, unit_path)

    _write_atomically(unit_path, rendered)

    # No systemd (container, chroot, image build): the file is in place for
    # the next boot, nothing to reload.
    if not systemd_runtime_dir.is_dir():
        return EnsureResult(EnsureOutcome.WRITTEN, unit_path)

    try:
        runner(["systemctl", "daemon-reload"])
    except (OSError, subprocess.SubprocessError) as exc:
        return EnsureResult(
            EnsureOutcome.WRITTEN, unit_path, daemon_reload_error=_describe_failure(exc)
        )
    return EnsureResult(EnsureOutcome.WRITTEN, unit_path, daemon_reloaded=True)
