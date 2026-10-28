"""Staged driver updates against the real local Docker daemon (CYB-4072).

The unit suites mock Docker. This one runs the activation and the image cleanup against real
containers and images, with local stub images in place of a registry: the "release" images are
three tags built here, one of which exits at once. What stays faked is only the call that
starts a twin's drivers from its metadata (``driver_activation._recreate``), which needs a
backend; it is replaced by a ``docker run`` of whatever image the metadata (or the rollback
override) names.

Everything is named per test run (``cyberwave-driver-it-<run>``), and the cleanup is limited to
those images, so a developer's own images are never touched.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import pytest

import cyberwave_edge_core.driver_activation as da
import cyberwave_edge_core.driver_update_state as state
import cyberwave_edge_core.image_gc as image_gc
import cyberwave_edge_core.startup as startup
from cyberwave_edge_core.docker_helpers import (
    docker_inspect,
    driver_container_name,
    parse_docker_timestamp_ns,
)

from .conftest import _have_docker, requires_docker

pytestmark = pytest.mark.docker

_LONG_RUNNING = 'FROM alpine:3.20\nLABEL release="{release}"\nCMD ["sleep", "infinity"]\n'
_EXITS_AT_ONCE = 'FROM alpine:3.20\nLABEL release="{release}"\nCMD ["sh", "-c", "exit 1"]\n'
_THREE_HOURS = 3 * 60 * 60


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=120, check=check
    )


def _build(tag: str, dockerfile: str) -> None:
    result = subprocess.run(
        ["docker", "build", "-t", tag, "-"],
        input=dockerfile,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        message = f"Could not build {tag}: {result.stderr.strip()}"
        # A skip would let CI go green without having run anything.
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)


def _image_exists(tag: str) -> bool:
    return _docker("image", "inspect", tag, check=False).returncode == 0


@pytest.fixture
def releases() -> Iterator[SimpleNamespace]:
    """Three local images: the running release, the one being staged, and a broken one."""
    if not _have_docker():
        pytest.skip("Docker daemon not reachable")
    run = uuid.uuid4().hex[:8]
    repo = f"cyberwave-driver-it-{run}"
    images = SimpleNamespace(
        repo=repo,
        old=f"{repo}:1.0.0",
        new=f"{repo}:2.0.0",
        broken=f"{repo}:3.0.0",
        unrelated=f"{repo}:0.9.0",
    )
    _build(images.old, _LONG_RUNNING.format(release="1.0.0"))
    _build(images.new, _LONG_RUNNING.format(release="2.0.0"))
    _build(images.broken, _EXITS_AT_ONCE.format(release="3.0.0"))
    _build(images.unrelated, _LONG_RUNNING.format(release="0.9.0"))
    yield images
    listed = _docker("image", "ls", repo, "--format", "{{.Repository}}:{{.Tag}}", check=False)
    for tag in listed.stdout.split():
        _docker("rmi", "--force", tag, check=False)


@pytest.fixture
def twin(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, releases: SimpleNamespace) -> Iterator[Any]:
    """A twin with the old release running; the next start uses ``host.live_image``."""
    twin_uuid = str(uuid.uuid4())
    container = driver_container_name(twin_uuid, "driver")
    host = SimpleNamespace(
        uuid=twin_uuid, container=container, live_image=releases.old, starts=[]
    )

    def start(image: str) -> None:
        host.starts.append(image)
        _docker("run", "--detach", "--name", container, image)

    def recreate(twin_uuid_: str, *, drivers_override: Optional[dict] = None) -> list[dict]:
        image = (drivers_override or {}).get("default", {}).get("docker_image") or host.live_image
        start(image)
        return [{"twin_uuid": twin_uuid_, "service_name": "driver", "success": True}]

    monkeypatch.setattr(state, "updates_dir", lambda: tmp_path / "driver-updates")
    monkeypatch.setattr(da, "_recreate", recreate)
    monkeypatch.setattr(startup, "load_driver_overrides", lambda: {})
    monkeypatch.setenv(da.SETTLE_SECONDS_ENV, "2")
    monkeypatch.setattr(startup, "_ACTIVATING_TWINS", set())
    start(releases.old)
    host.starts.clear()
    try:
        yield host
    finally:
        _docker("rm", "--force", container, check=False)


def _stage(twin: Any, image: str, releases: SimpleNamespace, request_id: str = "req-1") -> None:
    state.write_staged_record(
        {
            "request_id": request_id,
            "twin_uuid": twin.uuid,
            "package": "stub-driver",
            "images": [{"service": "driver", "image": image, "digest": None}],
            "previous_drivers": {"default": {"docker_image": releases.old}},
        }
    )


def _activate(twin: Any, request_id: str = "req-1") -> dict[str, str]:
    return da.activate(
        da.parse_request(
            {
                "command": da.ACTIVATE_COMMAND,
                "request_id": request_id,
                "twin_uuid": twin.uuid,
                "package": "stub-driver",
                "target_version": "2.0.0",
            }
        )
    )


def _running_image(twin: Any) -> str:
    return str(((docker_inspect(twin.container) or {}).get("Config") or {}).get("Image") or "")


def _status(twin: Any) -> str:
    return str(((docker_inspect(twin.container) or {}).get("State") or {}).get("Status") or "")


@requires_docker
class TestActivation:
    def test_the_twin_is_recreated_on_the_staged_image(self, twin, releases):
        twin.live_image = releases.new
        _stage(twin, releases.new, releases)

        running = _activate(twin)

        assert running == {twin.container: releases.new}
        assert _running_image(twin) == releases.new
        assert _status(twin) == "running"

    def test_a_second_command_for_a_twin_already_switched_does_not_restart_it(
        self, twin, releases
    ):
        twin.live_image = releases.new
        _stage(twin, releases.new, releases)
        _activate(twin)
        twin.starts.clear()

        _activate(twin)

        assert twin.starts == []

    def test_a_release_that_exits_is_rolled_back_to_the_running_one(self, twin, releases):
        twin.live_image = releases.broken
        _stage(twin, releases.broken, releases)

        with pytest.raises(da.ActivationError) as raised:
            _activate(twin)

        assert raised.value.code == da.UNHEALTHY
        assert "previous release is running again" in str(raised.value)
        assert twin.starts == [releases.broken, releases.old]
        assert _running_image(twin) == releases.old
        assert _status(twin) == "running"

    def test_a_staged_image_that_is_gone_stops_before_anything_is_touched(
        self, twin, releases
    ):
        _stage(twin, f"{releases.repo}:9.9.9", releases)

        with pytest.raises(da.ActivationError) as raised:
            _activate(twin)

        assert raised.value.code == da.STAGED_IMAGES_MISSING
        assert _running_image(twin) == releases.old
        assert twin.starts == []

    def test_an_update_that_was_never_staged_is_refused(self, twin):
        with pytest.raises(da.ActivationError) as raised:
            _activate(twin, request_id="never-staged")

        assert raised.value.code == da.NOT_STAGED


@requires_docker
class TestImageCleanup:
    """``gc_driver_images`` on real images, limited to this run's repository."""

    @pytest.fixture(autouse=True)
    def only_this_runs_images(self, monkeypatch, releases):
        real = image_gc._list_images
        monkeypatch.setattr(
            image_gc,
            "_list_images",
            lambda: [
                image
                for image in real()
                if any(tag.startswith(f"{releases.repo}:") for tag in image.tags)
            ],
        )

    def _collect(self, **kwargs: Any) -> None:
        assert image_gc.gc_driver_images(now=time.time() + _THREE_HOURS, **kwargs)

    def test_a_staged_image_survives_and_an_unused_one_does_not(self, twin, releases):
        _stage(twin, releases.new, releases)

        self._collect()

        assert _image_exists(releases.new)
        assert not _image_exists(releases.unrelated)
        assert _image_exists(releases.old)  # a container runs it

    def test_the_image_a_switch_left_behind_is_kept_for_a_rollback(self, twin, releases):
        twin.live_image = releases.new
        _stage(twin, releases.new, releases)
        _activate(twin)

        self._collect()
        assert _image_exists(releases.old)

        self._collect(aggressive=True)
        assert not _image_exists(releases.old)
        assert _image_exists(releases.new)  # running

    def test_nothing_is_removed_inside_the_grace_period(self, twin, releases):
        created = _docker("image", "inspect", releases.unrelated, "--format", "{{.Created}}")
        built_at = (parse_docker_timestamp_ns(created.stdout.strip()) or 0) / 1e9

        assert image_gc.gc_driver_images(now=built_at + 60)

        assert _image_exists(releases.unrelated)
