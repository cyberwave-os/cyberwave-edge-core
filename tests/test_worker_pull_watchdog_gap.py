"""The worker image pull does not extend systemd's start timeout (CYB-3338).

Driver image pulls heartbeat ``EXTEND_TIMEOUT_USEC`` while they run, added in
CYB-2049. The worker pull — moved onto the boot path by CYB-2029 — never
receives the watchdog, so a slow multi-GB pull can outlast ``TimeoutStartSec``
and be SIGTERMed before ``READY=1``.

``test_worker_pull_extends_start_timeout`` is ``xfail(strict=True)``: it fails
today, and the moment someone threads the watchdog through it will XPASS and
break the suite, prompting removal of the marker. The positive control beside it
validates the instrument — without it, a zero from the worker path could just
mean the harness is broken.
"""

from __future__ import annotations

import ast
import inspect
import time
from pathlib import Path

import pytest

from cyberwave_edge_core import startup, worker_manager


class _Recorder:
    """Duck-typed stand-in for the watchdog surface the pull path uses."""

    def __init__(self) -> None:
        self.extensions: list[float] = []

    def extend_timeout(self, seconds: float) -> None:
        self.extensions.append(seconds)

    def notify_status(self, status: str) -> None:
        pass


def _call_keywords(fn_name: str, callee: str) -> list[list[str]]:
    """Keyword names passed to *callee* at each call site inside *fn_name*."""
    tree = ast.parse(Path(startup.__file__).read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == fn_name
    )
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == callee:
                out.append([kw.arg for kw in node.keywords])
    return out


def test_driver_pull_call_site_forwards_watchdog() -> None:
    """run_startup_checks hands the watchdog to the driver pull."""
    calls = _call_keywords("run_startup_checks", "fetch_and_run_twin_drivers")
    assert calls, "no call to fetch_and_run_twin_drivers found"
    assert all("watchdog" in kw for kw in calls)


def test_worker_start_runs_inline_on_the_boot_path() -> None:
    """The worker start is not deferred to a thread, so it is pre-READY.

    If this ever fails, the pull moved off the boot path and CYB-3338 is moot.
    """
    calls = _call_keywords("run_startup_checks", "_start_worker_after_drivers")
    assert calls, "no call to _start_worker_after_drivers found"
    assert all("watchdog" not in kw for kw in calls)
    assert "watchdog" not in inspect.signature(
        startup._start_worker_after_drivers
    ).parameters


def test_driver_pull_extends_the_start_timeout() -> None:
    """Positive control: the instrument does detect extensions when they happen."""
    recorder = _Recorder()
    original = startup._pull_docker_image_with_progress_multi
    startup._pull_docker_image_with_progress_multi = (
        lambda image, *, contexts, token, timeout, on_progress: time.sleep(0.4)
    )
    try:
        startup._pull_driver_images_parallel(
            ["fake/image:tag"],
            watchdog=recorder,
            heartbeat_interval_seconds=0.05,
            heartbeat_extend_seconds=30.0,
        )
    finally:
        startup._pull_docker_image_with_progress_multi = original

    assert recorder.extensions, "driver pull emitted no extend_timeout"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CYB-3338: the worker pull path has no watchdog to extend the systemd "
        "start timeout. Remove this marker once it is threaded through."
    ),
)
def test_worker_pull_extends_start_timeout(tmp_path: Path) -> None:
    """The worker pull should heartbeat while it runs, as the driver pull does."""
    from cyberwave_edge_core.watchdog import ProcessWatchdog

    extensions: list[float] = []
    original = ProcessWatchdog.extend_timeout
    # Patched on the class so a call from anywhere in the pull path is caught,
    # including one this test never got the chance to inject an object into.
    ProcessWatchdog.extend_timeout = (  # type: ignore[method-assign]
        lambda self, seconds: extensions.append(seconds)
    )
    manager = worker_manager.WorkerManager(
        config_dir=tmp_path,
        environment_uuid="deadbeef-0000-0000-0000-000000000000",
        token="test-token",
    )
    manager._pull_worker_image_with_progress_once = (  # type: ignore[method-assign]
        lambda image, *, timeout, alert_ctxs, is_final_attempt: (
            time.sleep(0.4),
            True,
        )[1]
    )
    try:
        assert manager._pull_worker_image_with_progress("fake/image:tag") is True
    finally:
        ProcessWatchdog.extend_timeout = original  # type: ignore[method-assign]

    assert extensions, "worker pull emitted no extend_timeout"
