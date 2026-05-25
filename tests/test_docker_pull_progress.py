"""Tests for the byte-aggregated ``docker pull`` progress tracker."""

from __future__ import annotations

import io
import subprocess
import time

import pytest

from cyberwave_edge_core import driver_logs, startup
from cyberwave_edge_core.driver_logs import (
    _PULL_PHASE_COMPLETE,
    _PULL_PHASE_DOWNLOADING,
    _PULL_PHASE_INSTALLING,
    _PULL_PHASE_STARTED,
    _DockerPullProgress,
    _DockerPullTracker,
    _EngineAPIUnavailableError,
    _format_bytes,
    _PullDeliveryContext,
)

# Captured pre-fixture so engine-API tests can restore it (the autouse
# fixture below replaces the module-level reference for every test).
_REAL_ENGINE_API_DRIVER = driver_logs._drive_pull_via_engine_api


@pytest.fixture(autouse=True)
def _force_subprocess_fallback(monkeypatch):
    """Default to the subprocess driver; engine-API tests re-patch back."""

    def _refuse(*_args, **_kwargs):
        raise _EngineAPIUnavailableError("engine API disabled by test fixture")

    monkeypatch.setattr(driver_logs, "_drive_pull_via_engine_api", _refuse)


class TestFormatBytes:
    def test_renders_si_units_matching_docker_pull_bars(self):
        assert _format_bytes(0) == "0 B"
        assert _format_bytes(500) == "500 B"
        assert _format_bytes(1_500) == "1.50 kB"
        assert _format_bytes(45_000_000) == "45.0 MB"
        assert _format_bytes(745_000_000) == "745 MB"
        assert _format_bytes(1_550_000_000) == "1.55 GB"


class TestDockerPullTracker:
    def test_initial_state_is_pull_started_with_zero_progress(self):
        tracker = _DockerPullTracker("foo:latest")
        assert tracker.progress.phase == _PULL_PHASE_STARTED
        assert tracker.progress.percent() == 0
        assert tracker.progress.downloaded_bytes == 0
        assert tracker.progress.total_bytes == 0
        assert tracker.progress.layers_total == 0

    def test_top_level_intro_line_does_not_register_a_phantom_layer(self):
        """Regression: ``latest: Pulling from cyberwaveos/foo`` used to
        be treated as a layer named "latest" stuck in pending, inflating
        layers_total and dragging percent down."""
        tracker = _DockerPullTracker("foo:latest")

        tracker.feed("Using default tag: latest")
        tracker.feed("latest: Pulling from cyberwaveos/foo")
        tracker.feed("abc123def456: Pulling fs layer")
        tracker.feed("abc123def456: Pull complete")
        tracker.feed("Status: Downloaded newer image for foo:latest")

        assert tracker.progress.layers_total == 1
        assert tracker.progress.layers_complete == 1
        assert tracker.progress.phase == _PULL_PHASE_COMPLETE
        assert tracker.progress.percent() == 100

    def test_digest_line_does_not_register_a_phantom_layer(self):
        """``Digest: sha256:…`` is a top-level metadata line, not a layer."""
        tracker = _DockerPullTracker("foo:latest")

        tracker.feed("abc: Pulling fs layer")
        tracker.feed("Digest: sha256:abcdef0123456789")
        assert tracker.progress.layers_total == 1

    def test_download_bar_drives_byte_counts(self):
        tracker = _DockerPullTracker("foo:latest")

        tracker.feed("abc: Pulling fs layer")
        tracker.feed("abc: Downloading [>     ] 0B/200MB")
        assert tracker.progress.phase == _PULL_PHASE_DOWNLOADING
        assert tracker.progress.total_bytes == 200_000_000
        assert tracker.progress.downloaded_bytes == 0
        assert tracker.progress.percent() == 0

        tracker.feed("abc: Downloading [=====>] 100MB/200MB")
        assert tracker.progress.downloaded_bytes == 100_000_000
        assert tracker.progress.percent() == 50

        tracker.feed("abc: Downloading [==========>] 200MB/200MB")
        assert tracker.progress.percent() == 100

    def test_already_exists_layer_credits_zero_bytes(self):
        """``Already exists`` means no download happened; the layer
        should count as complete but contribute nothing to total_bytes."""
        tracker = _DockerPullTracker("foo:latest")

        tracker.feed("abc: Already exists")
        tracker.feed("def: Already exists")

        assert tracker.progress.layers_total == 2
        assert tracker.progress.layers_complete == 2
        assert tracker.progress.total_bytes == 0
        assert tracker.progress.phase == _PULL_PHASE_COMPLETE

    def test_phase_walks_downloading_then_installing_then_complete(self):
        tracker = _DockerPullTracker("foo:latest")

        tracker.feed("a: Pulling fs layer")
        tracker.feed("a: Downloading [==>] 50MB/100MB")
        assert tracker.progress.phase == _PULL_PHASE_DOWNLOADING

        tracker.feed("a: Download complete")
        tracker.feed("a: Extracting [==>] 50MB/100MB")
        # Download done but no layer is yet "complete" → installing.
        assert tracker.progress.phase == _PULL_PHASE_INSTALLING

        tracker.feed("a: Pull complete")
        tracker.feed("Status: Downloaded newer image for foo:latest")
        assert tracker.progress.phase == _PULL_PHASE_COMPLETE
        assert tracker.progress.percent() == 100

    def test_aggregate_byte_progress_across_layers(self):
        tracker = _DockerPullTracker("foo:latest")
        # Three layers, 100 MB + 200 MB + 700 MB = 1 GB total.
        tracker.feed("a: Downloading [>] 0B/100MB")
        tracker.feed("b: Downloading [>] 0B/200MB")
        tracker.feed("c: Downloading [>] 0B/700MB")
        # Halfway through `a` and `c`, full `b`.
        tracker.feed("a: Downloading [=====>] 50MB/100MB")
        tracker.feed("b: Download complete")
        tracker.feed("c: Downloading [=====>] 350MB/700MB")

        # 50 + 200 + 350 = 600 MB / 1000 MB = 60 %.
        assert tracker.progress.total_bytes == 1_000_000_000
        assert tracker.progress.downloaded_bytes == 600_000_000
        assert tracker.progress.percent() == 60

    def test_feed_returns_false_when_summary_is_unchanged(self):
        tracker = _DockerPullTracker("foo:latest")
        tracker.feed("a: Pulling fs layer")
        first = tracker.feed("a: Downloading [>] 50MB/100MB")
        # Same percent, same phase, same layer count → no externally
        # visible change, so subsequent identical-tier updates dedupe.
        second = tracker.feed("a: Downloading [>] 50MB/100MB")
        assert first is True
        assert second is False

    def test_extracting_alongside_completed_layers_reads_as_installing(self):
        """Multi-layer pull where N-1 layers are ``complete`` and one is
        still ``Extracting`` must read as ``installing`` — never bounce
        back to ``downloading`` (no pending/waiting/downloading present)
        and never read as ``pull_complete`` (extraction isn't done)."""
        tracker = _DockerPullTracker("foo:latest")

        tracker.feed("a: Pulling fs layer")
        tracker.feed("b: Pulling fs layer")
        tracker.feed("c: Pulling fs layer")
        tracker.feed("a: Pull complete")
        tracker.feed("b: Pull complete")
        tracker.feed("c: Extracting [==>] 50MB/100MB")

        assert tracker.progress.phase == _PULL_PHASE_INSTALLING
        assert tracker.progress.layers_total == 3
        assert tracker.progress.layers_complete == 2

    def test_status_image_up_to_date_forces_pull_complete(self):
        """``Status: Image is up to date for X`` is what cached pulls
        emit instead of ``Status: Downloaded newer image …``; both
        prefixes must collapse the phase to ``pull_complete``."""
        tracker = _DockerPullTracker("foo:latest")
        tracker.feed("Status: Image is up to date for foo:latest")
        assert tracker.progress.phase == _PULL_PHASE_COMPLETE

    def test_piped_cli_shape_drives_layers_active_progress(self):
        """Piped ``docker pull`` emits no byte lines; layers_active must move."""
        tracker = _DockerPullTracker("python:3.12-slim")

        for line in (
            "3.12-slim: Pulling from library/python",
            "5b4d6ff92fc4: Pulling fs layer",
            "4a9dde5cdde1: Pulling fs layer",
            "e113665b194b: Pulling fs layer",
            "07342fe545e6: Pulling fs layer",
        ):
            tracker.feed(line)

        assert tracker.progress.layers_total == 4
        assert tracker.progress.layers_active == 0
        assert tracker.progress.total_bytes == 0
        assert tracker.progress.phase == _PULL_PHASE_DOWNLOADING

        tracker.feed("e113665b194b: Verifying Checksum")
        tracker.feed("e113665b194b: Download complete")
        assert tracker.progress.layers_active == 1
        assert tracker.progress.phase == _PULL_PHASE_DOWNLOADING

        for line in (
            "5b4d6ff92fc4: Verifying Checksum",
            "5b4d6ff92fc4: Download complete",
            "4a9dde5cdde1: Download complete",
            "07342fe545e6: Verifying Checksum",
            "07342fe545e6: Download complete",
        ):
            tracker.feed(line)
        assert tracker.progress.phase == _PULL_PHASE_INSTALLING
        assert tracker.progress.layers_complete == 0
        assert tracker.progress.layers_active == 4

        for line in (
            "5b4d6ff92fc4: Pull complete",
            "4a9dde5cdde1: Pull complete",
            "e113665b194b: Pull complete",
            "07342fe545e6: Pull complete",
            "Status: Downloaded newer image for python:3.12-slim",
        ):
            tracker.feed(line)
        assert tracker.progress.phase == _PULL_PHASE_COMPLETE
        assert tracker.progress.layers_complete == 4

    def test_layers_active_changes_tick_the_signature(self):
        """layers_active must be in the signature so heartbeat ticks on transitions."""
        tracker = _DockerPullTracker("foo:latest")
        first = tracker.feed("a: Pulling fs layer")
        second = tracker.feed("a: Verifying Checksum")
        assert first is True
        assert second is True

    def test_feed_event_handles_engine_api_byte_deltas(self):
        """Engine API ``progressDetail`` populates byte counts."""
        tracker = _DockerPullTracker("alpine:3.18")

        tracker.feed_event(
            {
                "status": "Pulling fs layer",
                "progressDetail": {},
                "id": "44cf07d57ee4",
            }
        )
        tracker.feed_event(
            {
                "status": "Downloading",
                "progressDetail": {"current": 48486, "total": 3418409},
                "id": "44cf07d57ee4",
            }
        )
        assert tracker.progress.phase == _PULL_PHASE_DOWNLOADING
        assert tracker.progress.total_bytes == 3418409
        assert tracker.progress.downloaded_bytes == 48486

        tracker.feed_event(
            {
                "status": "Extracting",
                "progressDetail": {"current": 3418409, "total": 3418409},
                "id": "44cf07d57ee4",
            }
        )
        assert tracker.progress.phase == _PULL_PHASE_INSTALLING
        assert tracker.progress.percent() == 100

        tracker.feed_event(
            {
                "status": "Pull complete",
                "progressDetail": {},
                "id": "44cf07d57ee4",
            }
        )
        tracker.feed_event(
            {
                "status": "Status: Downloaded newer image for alpine:3.18",
            }
        )
        assert tracker.progress.phase == _PULL_PHASE_COMPLETE
        assert tracker.progress.percent() == 100

    def test_feed_event_ignores_non_dict_and_statusless_events(self):
        tracker = _DockerPullTracker("foo:latest")
        assert tracker.feed_event(None) is False
        assert tracker.feed_event("not a dict") is False
        assert tracker.feed_event({}) is False
        assert tracker.feed_event({"status": ""}) is False
        assert tracker.feed_event({"status": "Pulling from library/foo"}) is False
        assert tracker.progress.layers_total == 0


class TestDockerPullProgressFormatSummary:
    def test_renders_bytes_during_download(self):
        progress = _DockerPullProgress(
            image="cyberwaveos/ugv-driver:dev",
            downloaded_bytes=745_000_000,
            total_bytes=1_550_000_000,
            layers_total=12,
            layers_complete=4,
            phase=_PULL_PHASE_DOWNLOADING,
        )
        assert progress.format_summary() == "cyberwaveos/ugv-driver:dev 745 MB of 1.55 GB (48%)"

    def test_renders_layer_count_during_install(self):
        progress = _DockerPullProgress(
            image="cyberwaveos/ugv-driver:dev",
            layers_total=12,
            layers_complete=9,
            phase=_PULL_PHASE_INSTALLING,
        )
        assert progress.format_summary() == "cyberwaveos/ugv-driver:dev installing (9/12 layers)"

    def test_renders_pull_complete(self):
        progress = _DockerPullProgress(
            image="cyberwaveos/ugv-driver:dev",
            phase=_PULL_PHASE_COMPLETE,
        )
        assert progress.format_summary() == "cyberwaveos/ugv-driver:dev pull complete"

    def test_renders_starting_before_any_bytes_flow(self):
        progress = _DockerPullProgress(image="cyberwaveos/ugv-driver:dev")
        assert progress.format_summary() == "cyberwaveos/ugv-driver:dev starting"

    def test_renders_layer_count_when_bytes_unknown(self):
        """Subprocess fallback: surfaces layers_active when bytes are unknown."""
        progress = _DockerPullProgress(
            image="cyberwaveos/ugv-driver:dev",
            layers_total=12,
            layers_active=4,
            phase=_PULL_PHASE_DOWNLOADING,
        )
        assert progress.format_summary() == "cyberwaveos/ugv-driver:dev pulling (4/12 layers)"


class TestPullDockerImageWithProgress:
    """End-to-end: every parsed line publishes to MQTT and updates the alert."""

    class _FakeProcess:
        def __init__(self, output: str, *, returncode: int = 0):
            self.stdout = io.StringIO(output)
            self._returncode = returncode

        def wait(self, timeout=None):  # type: ignore[no-untyped-def]
            return self._returncode

        def kill(self):  # pragma: no cover - safety only
            pass

    class _FakeMQTT:
        def __init__(self):
            self.topic_prefix = "cw/"
            self.published: list[tuple[str, dict]] = []

        def publish(self, topic, payload):  # type: ignore[no-untyped-def]
            self.published.append((topic, payload))

    class _FakeClient:
        def __init__(self):
            self.mqtt = TestPullDockerImageWithProgress._FakeMQTT()

    class _FakeAlertCtx:
        def __init__(self):
            self.metadata_history: list[tuple[dict, bool]] = []

        def update_metadata(self, patch, *, force=False):  # type: ignore[no-untyped-def]
            self.metadata_history.append((dict(patch), force))

    def test_publishes_byte_progress_to_alert_metadata(self, monkeypatch):
        fake_client = self._FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)
        monkeypatch.setattr(
            driver_logs.subprocess,
            "Popen",
            lambda *args, **kwargs: self._FakeProcess(
                "Using default tag: latest\n"
                "latest: Pulling from cyberwaveos/ugv-driver\n"
                "abc: Pulling fs layer\r"
                "abc: Downloading [>] 0B/100MB\r"
                "abc: Downloading [=====>] 50MB/100MB\r"
                "abc: Downloading [==========>] 100MB/100MB\r"
                "abc: Pull complete\n"
                "Status: Downloaded newer image for cyberwaveos/ugv-driver:latest\n"
            ),
        )

        alert_ctx = self._FakeAlertCtx()
        final = driver_logs._pull_docker_image_with_progress_multi(
            "cyberwaveos/ugv-driver:latest",
            contexts=[
                _PullDeliveryContext(
                    twin_uuid="twin-1",
                    container_name="cyberwave-driver-twin1",
                    driver_alert_ctx=alert_ctx,
                ),
            ],
            token="test-token",
            timeout=30,
        )

        # Final snapshot must say the pull finished at 100% of one layer.
        assert final.phase == _PULL_PHASE_COMPLETE
        assert final.percent() == 100
        assert final.layers_total == 1
        assert final.total_bytes == 100_000_000

        # Alert metadata history must include byte counts that grow with
        # the download bar (50 MB seen before 100 MB seen).
        downloaded_seen = [
            patch.get("downloaded_bytes")
            for patch, _ in alert_ctx.metadata_history
            if "downloaded_bytes" in patch
        ]
        assert any(b == 50_000_000 for b in downloaded_seen)
        assert any(b == 100_000_000 for b in downloaded_seen)

        # And the final, force=True patch is the "pull_stream_finished"
        # sentinel the frontend's post-pull gate keys off.
        final_patch, final_force = alert_ctx.metadata_history[-1]
        assert final_force is True
        assert final_patch["phase"] == "pull_stream_finished"

    def test_fan_out_publishes_each_line_to_every_twins_mqtt_topic(self, monkeypatch):
        fake_client = self._FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)
        monkeypatch.setattr(
            driver_logs.subprocess,
            "Popen",
            lambda *args, **kwargs: self._FakeProcess(
                "abc: Pulling fs layer\r"
                "abc: Pull complete\n"
                "Status: Downloaded newer image for foo:latest\n"
            ),
        )

        driver_logs._pull_docker_image_with_progress_multi(
            "foo:latest",
            contexts=[
                _PullDeliveryContext(
                    twin_uuid="twin-a",
                    container_name="cyberwave-driver-twina",
                    driver_alert_ctx=self._FakeAlertCtx(),
                ),
                _PullDeliveryContext(
                    twin_uuid="twin-b",
                    container_name="cyberwave-driver-twinb",
                    driver_alert_ctx=self._FakeAlertCtx(),
                ),
            ],
            token="test-token",
            timeout=30,
        )

        published_topics = {topic for topic, _ in fake_client.mqtt.published}
        assert "cw/cyberwave/twin/twin-a/driverlog" in published_topics
        assert "cw/cyberwave/twin/twin-b/driverlog" in published_topics

    def test_dedupes_consecutive_identical_pull_lines(self, monkeypatch):
        """``docker pull`` replays the same bar via ``\\r`` until a real
        delta lands; the loop's ``last_message`` guard must collapse
        consecutive identical lines into one MQTT publish so the
        driverlog feed isn't dozens of duplicate ``50MB/100MB`` entries.
        """
        fake_client = self._FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)
        monkeypatch.setattr(
            driver_logs.subprocess,
            "Popen",
            lambda *args, **kwargs: self._FakeProcess(
                "abc: Pulling fs layer\r"
                "abc: Downloading [=====>] 50MB/100MB\r"
                "abc: Downloading [=====>] 50MB/100MB\r"
                "abc: Downloading [=====>] 50MB/100MB\r"
                "abc: Pull complete\n"
                "Status: Downloaded newer image for foo:latest\n"
            ),
        )

        driver_logs._pull_docker_image_with_progress_multi(
            "foo:latest",
            contexts=[
                _PullDeliveryContext(
                    twin_uuid="twin-1",
                    container_name="cyberwave-driver-twin1",
                    driver_alert_ctx=self._FakeAlertCtx(),
                ),
            ],
            token="test-token",
            timeout=30,
        )

        matching = [
            payload
            for _topic, payload in fake_client.mqtt.published
            if "50MB/100MB" in str(payload.get("message", ""))
        ]
        # Three identical lines, exactly one publish after dedupe.
        assert len(matching) == 1, [p.get("message") for p in matching]


class TestPullDriverImagesParallelHeartbeat:
    """The boot heartbeat must surface bytes/percent instead of the
    legacy ``Still pulling: <image> ...`` line."""

    def test_heartbeat_summary_includes_byte_progress(self, monkeypatch):
        """The fake worker blocks until the test releases it so the
        heartbeat loop actually reaches its ``else`` branch — the
        previous version of this test ran the worker to completion
        before the heartbeat ever fired, and asserted vacuously.
        """
        import logging
        import threading

        captured_status: list[str] = []
        captured_logs: list[str] = []

        class _Watchdog:
            def extend_timeout(self, _seconds: float) -> None:
                pass

            def notify_status(self, message: str) -> None:
                captured_status.append(message)

        worker_release = threading.Event()
        progress_published = threading.Event()

        def _fake_pull(image, *, contexts, token, timeout, on_progress=None):
            if on_progress is not None:
                on_progress(
                    _DockerPullProgress(
                        image=image,
                        downloaded_bytes=745_000_000,
                        total_bytes=1_550_000_000,
                        layers_total=12,
                        layers_complete=4,
                        phase=_PULL_PHASE_DOWNLOADING,
                    )
                )
            progress_published.set()
            worker_release.wait(timeout=15)
            return _DockerPullProgress(image=image, phase=_PULL_PHASE_COMPLETE)

        monkeypatch.setattr(driver_logs, "_pull_docker_image_with_progress_multi", _fake_pull)
        monkeypatch.setattr(startup, "_pull_docker_image_with_progress_multi", _fake_pull)
        monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda _img: True)

        class _Capture(logging.Handler):
            def emit(self, record):
                captured_logs.append(record.getMessage())

        handler = _Capture(level=logging.INFO)
        startup.logger.addHandler(handler)

        runner = threading.Thread(
            target=lambda: startup._pull_driver_images_parallel(
                ["cyberwaveos/ugv-driver:dev"],
                watchdog=_Watchdog(),
                heartbeat_interval_seconds=0.05,
                heartbeat_extend_seconds=0.1,
            ),
            daemon=True,
        )
        try:
            runner.start()

            # Worker publishes one snapshot then blocks; the heartbeat
            # loop fires every 50 ms and should surface bytes/percent
            # within a second.
            assert progress_published.wait(timeout=5), "worker never published"
            heartbeat_seen = False
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if any("745 MB of 1.55 GB" in s for s in captured_status + captured_logs):
                    heartbeat_seen = True
                    break
                threading.Event().wait(0.02)
        finally:
            worker_release.set()
            runner.join(timeout=10)
            startup.logger.removeHandler(handler)

        assert heartbeat_seen, (captured_status, captured_logs)
        # And the watchdog status line is the phase-aware ``Pulling: …``,
        # not the legacy ``Pulling driver images: …``.
        assert any(s.startswith("Pulling: ") for s in captured_status), (captured_status,)


def _install_fake_docker_sdk(monkeypatch, events):
    """Wire a fake ``docker`` SDK that yields *events* and re-enable the real driver."""
    import sys
    import types

    class _FakeAPI:
        def pull(self, repository, tag=None, stream=False, decode=False):
            assert stream is True and decode is True
            yield from events

    class _FakeDockerClient:
        api = _FakeAPI()

    fake_docker = types.ModuleType("docker")
    fake_docker.from_env = lambda timeout=None: _FakeDockerClient()
    fake_errors = types.ModuleType("docker.errors")

    class _APIError(Exception):  # noqa: N818
        pass

    class _DockerException(Exception):  # noqa: N818
        pass

    fake_errors.APIError = _APIError
    fake_errors.DockerException = _DockerException
    fake_utils = types.ModuleType("docker.utils")
    fake_utils.parse_repository_tag = (
        lambda ref: tuple(ref.rsplit(":", 1)) if ":" in ref else (ref, None)
    )
    fake_docker.errors = fake_errors
    fake_docker.utils = fake_utils

    monkeypatch.setitem(sys.modules, "docker", fake_docker)
    monkeypatch.setitem(sys.modules, "docker.errors", fake_errors)
    monkeypatch.setitem(sys.modules, "docker.utils", fake_utils)
    monkeypatch.setattr(driver_logs, "_drive_pull_via_engine_api", _REAL_ENGINE_API_DRIVER)


class TestPullDockerImageViaEngineAPI:
    """Engine-API path: byte deltas come from the SDK stream, not stdout."""

    class _FakeMQTT:
        def __init__(self):
            self.topic_prefix = "cw/"
            self.published: list[tuple[str, dict]] = []

        def publish(self, topic, payload):  # type: ignore[no-untyped-def]
            self.published.append((topic, payload))

    class _FakeClient:
        def __init__(self):
            self.mqtt = TestPullDockerImageViaEngineAPI._FakeMQTT()

    class _FakeAlertCtx:
        def __init__(self):
            self.metadata_history: list[tuple[dict, bool]] = []

        def update_metadata(self, patch, *, force=False):  # type: ignore[no-untyped-def]
            self.metadata_history.append((dict(patch), force))

    def test_engine_api_byte_deltas_drive_alert_metadata(self, monkeypatch):
        """Engine-API events push byte progress into the alert metadata."""
        events = [
            {"status": "Pulling from library/alpine", "id": "3.18"},
            {"status": "Pulling fs layer", "progressDetail": {}, "id": "44cf07"},
            {
                "status": "Downloading",
                "progressDetail": {"current": 48486, "total": 3418409},
                "id": "44cf07",
            },
            {
                "status": "Downloading",
                "progressDetail": {"current": 1_700_000, "total": 3418409},
                "id": "44cf07",
            },
            {"status": "Verifying Checksum", "progressDetail": {}, "id": "44cf07"},
            {"status": "Download complete", "progressDetail": {}, "id": "44cf07"},
            {
                "status": "Extracting",
                "progressDetail": {"current": 3418409, "total": 3418409},
                "id": "44cf07",
            },
            {"status": "Pull complete", "progressDetail": {}, "id": "44cf07"},
            {"status": "Digest: sha256:abc"},
            {"status": "Status: Downloaded newer image for alpine:3.18"},
        ]
        _install_fake_docker_sdk(monkeypatch, events)
        fake_client = self._FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

        alert_ctx = self._FakeAlertCtx()
        final = driver_logs._pull_docker_image_with_progress_multi(
            "alpine:3.18",
            contexts=[
                _PullDeliveryContext(
                    twin_uuid="twin-1",
                    container_name="cyberwave-driver-twin1",
                    driver_alert_ctx=alert_ctx,
                ),
            ],
            token="test-token",
            timeout=30,
        )

        assert final.phase == _PULL_PHASE_COMPLETE
        assert final.percent() == 100

        downloaded_seen = [
            patch.get("downloaded_bytes")
            for patch, _ in alert_ctx.metadata_history
            if "downloaded_bytes" in patch
        ]
        assert any(0 < b < 3_418_409 for b in downloaded_seen), downloaded_seen
        assert any(b == 3_418_409 for b in downloaded_seen)

        final_patch, final_force = alert_ctx.metadata_history[-1]
        assert final_force is True
        assert final_patch["phase"] == "pull_stream_finished"

    def test_engine_api_unavailable_falls_through_to_subprocess(self, monkeypatch):
        """SDK missing: dispatcher must complete via the subprocess driver."""
        called: dict[str, bool] = {"subprocess": False}

        def _fake_subprocess_driver(image, *, tracker, contexts_list, timeout, on_progress):
            called["subprocess"] = True
            tracker.feed("abc: Pulling fs layer")
            tracker.feed("abc: Pull complete")
            tracker.feed("Status: Downloaded newer image for foo:latest")

        monkeypatch.setattr(driver_logs, "_drive_pull_via_subprocess", _fake_subprocess_driver)

        fake_client = self._FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

        final = driver_logs._pull_docker_image_with_progress_multi(
            "foo:latest",
            contexts=[
                _PullDeliveryContext(
                    twin_uuid="twin-1",
                    container_name="cyberwave-driver-twin1",
                    driver_alert_ctx=self._FakeAlertCtx(),
                ),
            ],
            token="test-token",
            timeout=30,
        )

        assert called["subprocess"] is True
        assert final.phase == _PULL_PHASE_COMPLETE

    def test_engine_api_timeout_is_a_per_event_stall_watchdog(self, monkeypatch):
        """A pull that exceeds ``timeout`` total wall-clock must succeed as
        long as no single event-to-event gap exceeds it. Regression guard
        against treating ``timeout`` as a wall-clock cap on the whole pull
        (5 GB images on slow links legitimately take > 10 min)."""
        # Per-event sleep (0.3 s) stays well under timeout=1; total
        # iteration time (5 events ≈ 1.5 s) deliberately exceeds it.
        per_event_sleep = 0.3
        base_events = [
            {"status": "Pulling from library/alpine", "id": "3.18"},
            {
                "status": "Downloading",
                "progressDetail": {"current": 10_000, "total": 100_000},
                "id": "44cf07",
            },
            {
                "status": "Downloading",
                "progressDetail": {"current": 50_000, "total": 100_000},
                "id": "44cf07",
            },
            {
                "status": "Downloading",
                "progressDetail": {"current": 90_000, "total": 100_000},
                "id": "44cf07",
            },
            {"status": "Status: Downloaded newer image for alpine:3.18"},
        ]

        def _slow_stream():
            for evt in base_events:
                time.sleep(per_event_sleep)
                yield evt

        _install_fake_docker_sdk(monkeypatch, _slow_stream())
        fake_client = self._FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

        started = time.monotonic()
        final = driver_logs._pull_docker_image_with_progress_multi(
            "alpine:3.18",
            contexts=[
                _PullDeliveryContext(
                    twin_uuid="twin-1",
                    container_name="cyberwave-driver-twin1",
                    driver_alert_ctx=self._FakeAlertCtx(),
                ),
            ],
            token="test-token",
            timeout=1,
        )
        elapsed = time.monotonic() - started
        assert final.phase == _PULL_PHASE_COMPLETE
        assert elapsed > 1.0, f"pull finished too fast ({elapsed:.2f} s) — invariant invalid"

    def test_engine_api_throttles_mqtt_to_signature_changes(self, monkeypatch):
        """A flood of byte-delta events at the same integer-percent must
        not produce a publish per event. Regression guard against
        sub-percent MQTT spam on the engine-API path."""
        # 200 events all landing at 50 % (50 000 / 100 000) — these change
        # ``downloaded_bytes`` but not ``percent()`` / phase / layer count.
        events = [{"status": "Pulling from library/alpine", "id": "3.18"}]
        events.append(
            {
                "status": "Downloading",
                "progressDetail": {"current": 50_000, "total": 100_000},
                "id": "44cf07",
            }
        )
        # 200 identical byte-progress events (only ``downloaded_bytes`` is
        # already 50_000, so percent stays at 50). These must collapse to
        # 0 additional publishes.
        for _ in range(200):
            events.append(
                {
                    "status": "Downloading",
                    "progressDetail": {"current": 50_000, "total": 100_000},
                    "id": "44cf07",
                }
            )
        events.append(
            {
                "status": "Downloading",
                "progressDetail": {"current": 100_000, "total": 100_000},
                "id": "44cf07",
            }
        )
        events.append({"status": "Status: Downloaded newer image for alpine:3.18"})

        _install_fake_docker_sdk(monkeypatch, events)
        fake_client = self._FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

        driver_logs._pull_docker_image_with_progress_multi(
            "alpine:3.18",
            contexts=[
                _PullDeliveryContext(
                    twin_uuid="twin-1",
                    container_name="cyberwave-driver-twin1",
                    driver_alert_ctx=self._FakeAlertCtx(),
                ),
            ],
            token="test-token",
            timeout=30,
        )

        pull_publishes = [
            payload
            for _topic, payload in fake_client.mqtt.published
            if str(payload.get("message", "")).startswith("docker pull:")
        ]
        # Strict upper bound: start, 50% jump, 100% jump, status; some
        # framework may insert the started/finished sentinels too. The
        # important invariant is "≪ 200" — pre-fix this would be ~204.
        assert len(pull_publishes) <= 10, (
            len(pull_publishes),
            [p.get("message") for p in pull_publishes],
        )

    def test_engine_api_error_event_raises_called_process_error(self, monkeypatch):
        """An ``{"error": "..."}`` event must bubble up as
        :class:`subprocess.CalledProcessError` so the dispatcher's
        existing failure-handling branch fires."""
        events = [
            {"status": "Pulling from library/foo", "id": "latest"},
            {"error": "manifest unknown for foo:latest"},
        ]
        _install_fake_docker_sdk(monkeypatch, events)
        fake_client = self._FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)

        alert_ctx = self._FakeAlertCtx()
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            driver_logs._pull_docker_image_with_progress_multi(
                "foo:latest",
                contexts=[
                    _PullDeliveryContext(
                        twin_uuid="twin-1",
                        container_name="cyberwave-driver-twin1",
                        driver_alert_ctx=alert_ctx,
                    ),
                ],
                token="test-token",
                timeout=30,
            )
        assert "manifest unknown" in str(excinfo.value.stderr)
        assert any(
            patch.get("phase") == "pull_exit_error" for patch, _ in alert_ctx.metadata_history
        )
