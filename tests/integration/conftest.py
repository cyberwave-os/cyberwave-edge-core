"""Fixtures for the Edge Core worker-lifecycle real-Docker integration suite.

The suite exercises ``WorkerManager``, ``reconcile_worker_lifecycle``,
``WorkerWatcher``, ``_handle_twin_command_message``, and
``_run_remove_workflow_worker`` against the real local Docker daemon
using a session-built stub image. No backend, no MQTT broker, no
``cyberwaveos/edge-ml-worker`` pull.

The fixtures here mirror the per-test isolation patterns from the unit
suite (``test_worker_lifecycle_reconcile.py``, ``test_worker_manager.py``,
``test_startup_remove_workflow_worker.py``) while replacing every
Docker mock with a real subprocess call.

Skip behaviour matches ``tests/test_systemd_notify_integration.py``:
each test is decorated ``@requires_docker``; ``_have_docker()`` is
cached so the daemon probe runs at most once per process.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Iterator

import pytest

import cyberwave_edge_core.startup as startup
from cyberwave_edge_core.worker_health import WorkerHealthMonitor
from cyberwave_edge_core.worker_manager import WorkerManager


_DOCKERFILE = Path(__file__).parent / "stub_worker.Dockerfile"

STUB_IMAGE_IT = "cyberwave-worker-stub:it"
STUB_IMAGE_IMMUTABLE = "cyberwave-worker-stub:v1"
STUB_IMAGE_MUTABLE = "cyberwave-worker-stub:local"
STUB_IMAGE_TAGS = (STUB_IMAGE_IT, STUB_IMAGE_IMMUTABLE, STUB_IMAGE_MUTABLE)


# ---------------------------------------------------------------------------
# Docker availability gating
# ---------------------------------------------------------------------------


_docker_available_cache: bool | None = None


def _have_docker() -> bool:
    """Return True when a reachable Docker daemon is available.

    Cached for the lifetime of the process so collection does not pay the
    cost of repeated ``docker info`` probes. Mirrors the gating helper in
    :mod:`tests.test_systemd_notify_integration`.
    """
    global _docker_available_cache
    if _docker_available_cache is not None:
        return _docker_available_cache

    if shutil.which("docker") is None:
        _docker_available_cache = False
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _docker_available_cache = False
        return False

    _docker_available_cache = result.returncode == 0
    return _docker_available_cache


requires_docker = pytest.mark.skipif(
    not _have_docker(),
    reason="Docker daemon not reachable; integration suite requires local Docker",
)


# ---------------------------------------------------------------------------
# Polling helper (mirrors ``wait_for_event`` from
# ``devops/e2e-driver-tests/verify_recording_pipeline.py``).
# ---------------------------------------------------------------------------


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float,
    poll_interval_s: float = 0.25,
    description: str,
) -> None:
    """Block until *predicate* returns truthy or *timeout_s* elapses.

    On timeout calls :func:`pytest.fail` with *description* so the
    failure message identifies which condition timed out.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception:
            # Predicates may probe Docker; transient errors are tolerated.
            pass
        time.sleep(poll_interval_s)
    pytest.fail(f"Timed out after {timeout_s:.1f}s waiting for: {description}")


# ---------------------------------------------------------------------------
# Stub image build (session-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def stub_worker_image() -> Iterator[str]:
    """Build the stub worker image once per session; remove all tags on teardown.

    Three tags are produced from a single ``docker build`` so Phase 8
    can exercise the mutable vs immutable image-pull policy without
    needing two builds:

    * ``cyberwave-worker-stub:it`` — primary tag used by most tests.
    * ``cyberwave-worker-stub:v1`` — immutable tag (no entry in
      :data:`cyberwave_edge_core.worker_manager._MUTABLE_TAG_BASENAMES`).
    * ``cyberwave-worker-stub:local`` — mutable tag (``local`` is in
      the mutable basenames list).
    """
    if not _have_docker():
        # No Docker → nothing to build; tests will be skipped individually.
        yield STUB_IMAGE_IT
        return

    try:
        build = subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(_DOCKERFILE),
                "-t",
                STUB_IMAGE_IT,
                str(_DOCKERFILE.parent),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.skip(f"Stub worker image build timed out after {exc.timeout:.0f}s")

    if build.returncode != 0:
        pytest.skip(
            "Could not build stub worker image "
            f"(rc={build.returncode}): {build.stderr.strip()}"
        )

    # Retag must succeed: Phase 8 needs both ``:v1`` (immutable) and
    # ``:local`` (mutable) present locally to exercise the pull policy.
    # If a retag silently failed, ``WorkerManager._ensure_image_pulled``
    # would try to fetch the missing tag from docker.io, hanging up to
    # 600 s. Skip the suite with a clear reason instead.
    for extra_tag in (STUB_IMAGE_IMMUTABLE, STUB_IMAGE_MUTABLE):
        tag_result = subprocess.run(
            ["docker", "tag", STUB_IMAGE_IT, extra_tag],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if tag_result.returncode != 0:
            pytest.skip(
                f"Failed to retag stub image as {extra_tag} "
                f"(rc={tag_result.returncode}): {tag_result.stderr.strip()}"
            )

    try:
        yield STUB_IMAGE_IT
    finally:
        for tag in STUB_IMAGE_TAGS:
            try:
                subprocess.run(
                    ["docker", "rmi", "-f", tag],
                    capture_output=True,
                    timeout=15,
                )
            except (subprocess.TimeoutExpired, OSError):
                # Best-effort cleanup — never let session teardown raise.
                pass


# ---------------------------------------------------------------------------
# Per-test environment isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def worker_env_uuid() -> str:
    """Return a fresh environment UUID whose first 8 hex chars are unique.

    The container name is ``cyberwave-worker-{env_uuid[:8]}``; uniqueness
    of the prefix is what isolates parallel/serial tests from each other.
    """
    return uuid.uuid4().hex


@pytest.fixture
def worker_container_name(worker_env_uuid: str) -> str:
    """Return the container name that :class:`WorkerManager` will use."""
    return f"cyberwave-worker-{worker_env_uuid[:8]}"


@pytest.fixture
def docker_cleanup(worker_container_name: str) -> Iterator[str]:
    """Force-remove the per-test container on teardown.

    The cleanup never raises: ``finally`` swallows ``TimeoutExpired`` and
    ``OSError`` so the original test failure (if any) is what bubbles up,
    instead of a confusing teardown exception masking it.
    """
    try:
        yield worker_container_name
    finally:
        try:
            subprocess.run(
                ["docker", "rm", "-f", worker_container_name],
                capture_output=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass


@pytest.fixture
def configured_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_env_uuid: str,
) -> Iterator[Path]:
    """Point :mod:`startup` at a temp config dir and stub the auth lookups.

    Mirrors the ``configured_edge`` fixture pattern in
    ``tests/test_worker_lifecycle_reconcile.py`` so the integration suite
    drives the same reconcile code paths the unit suite covers, just
    with a real ``WorkerManager`` instead of a ``MagicMock`` plugged in.
    """
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(startup, "load_token", lambda: "it-token")
    monkeypatch.setattr(
        startup, "load_environment_uuid", lambda: worker_env_uuid
    )
    monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "it-fp")
    monkeypatch.setattr(
        startup,
        "_resolve_worker_sync_twin_uuids",
        lambda *a, **kw: ["it-twin"],
    )
    monkeypatch.setattr(startup, "load_worker_resource_limits", lambda: None)
    monkeypatch.setattr(
        "cyberwave_edge_core.worker_manager.resolve_worker_image",
        lambda: STUB_IMAGE_IT,
    )

    # macOS-only test override: production code maps port 7447 for Zenoh
    # so the host can reach the worker. The stub never starts Zenoh and
    # the port routinely collides with another local Cyberwave container
    # or process on dev machines. Linux uses ``--network host`` and is
    # unaffected. This override is strictly test scaffolding — it tweaks
    # argv only, no Docker helper is mocked.
    if platform.system() == "Darwin":
        monkeypatch.setattr(
            WorkerManager,
            "_build_network_args",
            lambda self: ["--add-host", "host.docker.internal:host-gateway"],
        )

    workers_dir = tmp_path / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)

    # Keep the startup probe short so tests don't pay 30 s per cold start.
    monkeypatch.setenv("CYBERWAVE_WORKER_STARTUP_PROBE_SECONDS", "5")

    # Reset shared module state so previous tests' wiring never leaks in.
    startup._HANDLED_TWIN_COMMAND_REQUEST_IDS.clear()
    startup._WORKER_SYNC_PREVIOUSLY_MISSING.clear()
    startup._set_monitored_worker_manager(None)
    startup._set_active_worker_watcher(None)

    try:
        yield tmp_path
    finally:
        startup._HANDLED_TWIN_COMMAND_REQUEST_IDS.clear()
        startup._WORKER_SYNC_PREVIOUSLY_MISSING.clear()
        startup._set_monitored_worker_manager(None)
        startup._set_active_worker_watcher(None)


@pytest.fixture
def make_worker_manager(
    configured_edge: Path,
    worker_env_uuid: str,
    stub_worker_image: str,
) -> Callable[..., WorkerManager]:
    """Factory for :class:`WorkerManager` instances bound to the per-test config.

    Tests that need to attach a health monitor or override the image use
    this factory; tests that need the default wiring can rely on the
    ``reconcile_worker_lifecycle`` path constructing its own manager.
    """

    def _factory(
        *,
        image: str = STUB_IMAGE_IT,
        with_health_monitor: bool = False,
        expected_running_probe: Callable[[], bool] | None = None,
    ) -> WorkerManager:
        mgr = WorkerManager(
            config_dir=configured_edge,
            environment_uuid=worker_env_uuid,
            token="it-token",
            twin_uuids=["it-twin"],
            image=image,
        )
        if with_health_monitor:
            monitor = WorkerHealthMonitor(
                container_name=mgr.container_name,
                expected_running_probe=expected_running_probe,
            )
            mgr.set_health_monitor(monitor)
        return mgr

    return _factory
