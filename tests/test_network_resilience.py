"""Edge-core across a lost uplink: degraded startup, the cloud breaker, flap revival.

Covers:
  1. What counts as "backend unreachable" and the breaker it opens
  2. Degraded startup on a network error vs a fatal 401 / refused registration,
     and the drivers the last shutdown stopped
  3. The deferred startup retry from the runtime loop
  4. Flap-stopped drivers: one alert per episode, capped revival backoff, reset
  5. Revival arguments (watchdog, no pulls) and the breaker skip
  6. Periodic cleanup keeps flap-stopped containers and their images
  7. Runtime-loop guards: failing steps, watchdog pings, paused cloud steps
  8. The shared MQTT client is reused while it reconnects
"""

from __future__ import annotations

import errno
import socket
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from cyberwave.exceptions import CyberwaveAPIError, CyberwaveConnectionError
from urllib3.exceptions import MaxRetryError, NewConnectionError

import cyberwave_edge_core.docker_helpers as dh
import cyberwave_edge_core.startup as startup
from cyberwave_edge_core import driver_logs


def _never(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("must not be called")


def _api_error(status: int) -> CyberwaveAPIError:
    return CyberwaveAPIError(f"HTTP {status}", status_code=status)


def _wrapped_transport_error() -> CyberwaveAPIError:
    """What the SDK raises for a dead uplink: an API error without a status."""
    try:
        try:
            raise MaxRetryError(None, "/api/v1/workspaces", NewConnectionError(None, "refused"))
        except MaxRetryError as exc:
            raise CyberwaveAPIError("request failed") from exc
    except CyberwaveAPIError as exc:
        return exc


# ===========================================================================
# 1. Unreachable classification and the breaker
# ===========================================================================


class TestCloudUnreachable:
    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionError("refused"),
            socket.gaierror(-3, "Temporary failure in name resolution"),
            TimeoutError("timed out"),
            CyberwaveConnectionError("down"),
            _api_error(502),
            _api_error(503),
            _api_error(429),
            _api_error(0),
            _wrapped_transport_error(),
        ],
    )
    def test_network_errors_and_outages_are_unreachable(self, exc):
        assert startup._is_cloud_unreachable_error(exc) is True

    @pytest.mark.parametrize(
        "exc", [_api_error(401), _api_error(403), _api_error(404), ValueError("bad json")]
    )
    def test_an_answer_from_the_backend_is_not(self, exc):
        assert startup._is_cloud_unreachable_error(exc) is False

    @pytest.mark.parametrize(
        "exc",
        [
            OSError(errno.ENETUNREACH, "Network is unreachable"),
            OSError(errno.EHOSTUNREACH, "No route to host"),
        ],
    )
    def test_a_dead_route_is_unreachable(self, exc):
        assert startup._is_cloud_unreachable_error(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            PermissionError(errno.EACCES, "Permission denied"),
            FileNotFoundError(errno.ENOENT, "No such file: ca-bundle.crt"),
            OSError(errno.EMFILE, "Too many open files"),
        ],
    )
    def test_a_local_os_error_is_not(self, exc):
        assert startup._is_cloud_unreachable_error(exc) is False

    def test_a_network_error_pauses_cloud_steps_with_a_doubling_backoff(self):
        assert startup.note_cloud_failure(ConnectionError("down")) is True
        assert startup.cloud_reconciliation_paused()
        assert startup._cloud_backoff_seconds == startup._CLOUD_BACKOFF_INITIAL_SECONDS

        # A call already in flight when the breaker opened does not extend it.
        assert startup.note_cloud_failure(ConnectionError("down")) is True
        assert startup._cloud_backoff_seconds == startup._CLOUD_BACKOFF_INITIAL_SECONDS

        startup._cloud_paused_until = time.monotonic() - 1
        assert not startup.cloud_reconciliation_paused()
        startup.note_cloud_failure(ConnectionError("still down"))
        assert startup._cloud_backoff_seconds == 2 * startup._CLOUD_BACKOFF_INITIAL_SECONDS

    def test_the_backoff_is_capped(self):
        startup._cloud_backoff_seconds = startup._CLOUD_BACKOFF_MAX_SECONDS
        startup._cloud_paused_until = time.monotonic() - 1
        startup.note_cloud_failure(ConnectionError("down"))
        assert startup._cloud_backoff_seconds == startup._CLOUD_BACKOFF_MAX_SECONDS

    def test_a_success_closes_the_breaker(self):
        startup.note_cloud_failure(ConnectionError("down"))
        startup.note_cloud_success()
        assert not startup.cloud_reconciliation_paused()
        assert startup._cloud_backoff_seconds == 0.0

    def test_a_rejection_leaves_the_breaker_closed(self):
        assert startup.note_cloud_failure(_api_error(401)) is False
        assert not startup.cloud_reconciliation_paused()

    @pytest.mark.parametrize("status", [503, 429, 404])
    def test_an_http_answer_about_one_twin_leaves_the_breaker_closed(self, status):
        startup.note_twin_call_failure(_api_error(status))
        assert not startup.cloud_reconciliation_paused()

    @pytest.mark.parametrize(
        "exc", [ConnectionError("down"), _api_error(0), _wrapped_transport_error()]
    )
    def test_an_unreachable_backend_on_a_twin_call_opens_the_breaker(self, exc):
        startup.note_twin_call_failure(exc)
        assert startup.cloud_reconciliation_paused()

    def test_a_twin_update_failing_with_a_5xx_leaves_the_breaker_closed(self, tmp_path):
        twin_json = tmp_path / "twin.json"
        twin_json.write_text('{"name": "go2"}')
        client = MagicMock()
        client.twins.update.side_effect = _api_error(500)
        client.twins.get_raw.side_effect = _api_error(500)

        assert startup._sync_twin_json_file_with_backend(client, "t", twin_json) is False
        assert startup._pull_twin_json_file_from_backend(client, "t", twin_json) is None

        client.twins.update.assert_called_once()
        client.twins.get_raw.assert_called_once()
        assert not startup.cloud_reconciliation_paused()


class TestMqttReconnectEndsThePause:
    @pytest.fixture
    def mqtt(self, monkeypatch):
        state = SimpleNamespace(connected=False)
        monkeypatch.setattr(startup, "_shared_mqtt_client", SimpleNamespace(mqtt=state))
        return state

    def test_a_reconnect_probes_the_backend_now(self, mqtt):
        startup.note_cloud_failure(ConnectionError("down"))
        startup._end_cloud_pause_on_mqtt_reconnect()
        assert startup.cloud_reconciliation_paused()

        mqtt.connected = True
        startup._end_cloud_pause_on_mqtt_reconnect()

        assert not startup.cloud_reconciliation_paused()
        # A probe that fails again backs off further.
        startup.note_cloud_failure(ConnectionError("still down"))
        assert startup._cloud_backoff_seconds == 2 * startup._CLOUD_BACKOFF_INITIAL_SECONDS

    def test_a_broker_that_stays_up_leaves_the_pause_alone(self, mqtt):
        mqtt.connected = True
        startup._end_cloud_pause_on_mqtt_reconnect()
        startup.note_cloud_failure(ConnectionError("api down"))

        startup._end_cloud_pause_on_mqtt_reconnect()

        assert startup.cloud_reconciliation_paused()


class TestCheckToken:
    def _backend(self, monkeypatch, exc: BaseException | None) -> None:
        def make_client(**_kwargs: Any) -> Any:
            def list_workspaces() -> list[Any]:
                if exc is not None:
                    raise exc
                return []

            return SimpleNamespace(workspaces=SimpleNamespace(list=list_workspaces))

        monkeypatch.setattr(startup, "Cyberwave", make_client)

    def test_valid(self, monkeypatch):
        self._backend(monkeypatch, None)
        assert startup.check_token("tok-123456789012") == startup.TOKEN_VALID
        assert startup.validate_token("tok-123456789012") is True

    def test_a_network_error_is_unreachable(self, monkeypatch):
        self._backend(monkeypatch, _wrapped_transport_error())
        assert startup.check_token("tok-123456789012") == startup.CLOUD_UNREACHABLE
        assert startup.validate_token("tok-123456789012") is False

    def test_a_401_is_rejected(self, monkeypatch):
        self._backend(monkeypatch, _api_error(401))
        assert startup.check_token("tok-123456789012") == startup.TOKEN_REJECTED
        assert not startup.cloud_reconciliation_paused()


# ===========================================================================
# 2. Degraded startup
# ===========================================================================


class TestDegradedStartup:
    @pytest.fixture(autouse=True)
    def _boot(self, monkeypatch, tmp_path):
        monkeypatch.setattr(startup, "_fix_config_dir_ownership", lambda: None)
        monkeypatch.setattr(startup, "_ensure_config_subdirs", lambda: None)
        monkeypatch.setattr(startup, "load_token", lambda: "token-123")
        monkeypatch.setattr(
            startup, "_upload_host_facts_on_startup", lambda token, watchdog=None: False
        )
        # Nothing past the registration step may run in degraded mode.
        monkeypatch.setattr(startup, "load_environment_uuid", _never)
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", _never)

    def test_a_network_error_on_the_token_check_boots_degraded(self, monkeypatch):
        monkeypatch.setattr(startup, "check_token", lambda token: startup.CLOUD_UNREACHABLE)
        monkeypatch.setattr(startup, "check_mqtt_connection", _never)
        monkeypatch.setattr(startup, "register_edge_outcome", _never)

        assert startup.run_startup_checks() is True
        assert startup.degraded_startup_pending()

    def test_a_rejected_token_still_fails(self, monkeypatch):
        monkeypatch.setattr(startup, "check_token", lambda token: startup.TOKEN_REJECTED)
        monkeypatch.setattr(startup, "check_mqtt_connection", _never)

        assert startup.run_startup_checks() is False
        assert not startup.degraded_startup_pending()

    def test_an_unreachable_registration_boots_degraded(self, monkeypatch):
        monkeypatch.setattr(startup, "check_token", lambda token: startup.TOKEN_VALID)
        monkeypatch.setattr(startup, "check_mqtt_connection", lambda token: False)
        monkeypatch.setattr(
            startup, "register_edge_outcome", lambda token: startup.CLOUD_UNREACHABLE
        )

        assert startup.run_startup_checks() is True
        assert startup.degraded_startup_pending()

    def test_a_refused_registration_still_fails(self, monkeypatch):
        monkeypatch.setattr(startup, "check_token", lambda token: startup.TOKEN_VALID)
        monkeypatch.setattr(startup, "check_mqtt_connection", lambda token: True)
        monkeypatch.setattr(
            startup, "register_edge_outcome", lambda token: startup.EDGE_REGISTRATION_FAILED
        )

        assert startup.run_startup_checks() is False
        assert not startup.degraded_startup_pending()

    def test_registration_classifies_a_network_error(self, monkeypatch):
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")

        def make_client(**_kwargs: Any) -> Any:
            def create(**_kw: Any) -> Any:
                raise _wrapped_transport_error()

            return SimpleNamespace(edges=SimpleNamespace(create=create))

        monkeypatch.setattr(startup, "Cyberwave", make_client)
        assert startup.register_edge_outcome("tok") == startup.CLOUD_UNREACHABLE

        startup.note_cloud_success()

        def make_rejecting(**_kwargs: Any) -> Any:
            def create(**_kw: Any) -> Any:
                raise _api_error(403)

            return SimpleNamespace(edges=SimpleNamespace(create=create))

        monkeypatch.setattr(startup, "Cyberwave", make_rejecting)
        assert startup.register_edge_outcome("tok") == startup.EDGE_REGISTRATION_FAILED


class TestDriversStoppedAtShutdown:
    """A restart or reboot while offline must not leave the robot without drivers."""

    @pytest.fixture(autouse=True)
    def _state(self, monkeypatch, tmp_path):
        monkeypatch.setattr(startup, "STOPPED_DRIVERS_FILE", tmp_path / "stopped_drivers.json")
        monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {})
        monkeypatch.setattr(startup.shutil, "which", lambda _name: "/usr/bin/docker")
        self.started: list[str] = []
        self.containers = {"cyberwave-driver-go2": "exited", "cyberwave-driver-cam": "running"}

        def docker(cmd, **_kwargs):
            if cmd[:2] == ["docker", "ps"]:
                names = [
                    n for n, state in self.containers.items() if "-a" in cmd or state == "running"
                ]
                return SimpleNamespace(returncode=0, stdout="\n".join(names), stderr="")
            if cmd[:2] == ["docker", "start"]:
                self.started.append(cmd[2])
                self.containers[cmd[2]] = "running"
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if cmd[:2] == ["docker", "stop"]:
                self.containers[cmd[2]] = "exited"
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(cmd)

        monkeypatch.setattr(startup.subprocess, "run", docker)

    def test_an_offline_boot_restarts_the_drivers_the_shutdown_stopped(self):
        startup._CONTAINER_TWIN_MAP.update(
            {"cyberwave-driver-go2": "twin-go2", "cyberwave-driver-cam": "twin-cam"}
        )
        startup._graceful_shutdown(None)
        assert self.containers["cyberwave-driver-go2"] == "exited"
        startup._CONTAINER_TWIN_MAP.clear()  # a fresh process

        resumed = startup._resume_drivers_stopped_at_shutdown()

        assert sorted(resumed) == ["cyberwave-driver-cam", "cyberwave-driver-go2"]
        assert startup._CONTAINER_TWIN_MAP == {
            "cyberwave-driver-go2": "twin-go2",
            "cyberwave-driver-cam": "twin-cam",
        }

    def test_a_running_driver_is_adopted_without_a_restart(self):
        startup._remember_drivers_stopped_at_shutdown(
            {"cyberwave-driver-go2": "twin-go2", "cyberwave-driver-cam": "twin-cam"}
        )

        assert startup._resume_drivers_stopped_at_shutdown() == ["cyberwave-driver-go2"]
        assert self.started == ["cyberwave-driver-go2"]
        assert startup._CONTAINER_TWIN_MAP["cyberwave-driver-cam"] == "twin-cam"

    def test_a_removed_driver_is_skipped(self):
        startup._remember_drivers_stopped_at_shutdown({"cyberwave-driver-gone": "twin-x"})

        assert startup._resume_drivers_stopped_at_shutdown() == []
        assert self.started == []
        assert startup._CONTAINER_TWIN_MAP == {}

    def test_nothing_recorded_or_garbage_resumes_nothing(self):
        assert startup._resume_drivers_stopped_at_shutdown() == []
        startup.STOPPED_DRIVERS_FILE.write_text("not json")
        assert startup._resume_drivers_stopped_at_shutdown() == []
        assert self.started == []

    def test_a_shutdown_with_no_drivers_clears_the_record(self):
        startup._remember_drivers_stopped_at_shutdown({"cyberwave-driver-go2": "twin-go2"})
        startup._graceful_shutdown(None)
        assert not startup.STOPPED_DRIVERS_FILE.exists()

    def test_a_degraded_boot_that_could_not_read_docker_keeps_the_record(self, monkeypatch):
        startup._remember_drivers_stopped_at_shutdown({"cyberwave-driver-go2": "twin-go2"})
        monkeypatch.setattr(startup, "_DEGRADED_STARTUP_PENDING", True)
        monkeypatch.setattr(
            startup, "_list_driver_containers_or_none", lambda *, include_stopped: None
        )

        assert startup._resume_drivers_stopped_at_shutdown() == []
        startup._graceful_shutdown(None)

        assert startup.STOPPED_DRIVERS_FILE.exists()

    def test_the_degraded_boot_resumes_them(self, monkeypatch):
        monkeypatch.setattr(startup, "_fix_config_dir_ownership", lambda: None)
        monkeypatch.setattr(startup, "_ensure_config_subdirs", lambda: None)
        monkeypatch.setattr(startup, "load_token", lambda: "token-123")
        monkeypatch.setattr(
            startup, "_upload_host_facts_on_startup", lambda token, watchdog=None: False
        )
        monkeypatch.setattr(startup, "check_token", lambda token: startup.CLOUD_UNREACHABLE)
        startup._remember_drivers_stopped_at_shutdown({"cyberwave-driver-go2": "twin-go2"})

        assert startup.run_startup_checks() is True
        assert self.started == ["cyberwave-driver-go2"]
        # Kept until the backend relaunches the drivers.
        assert startup.STOPPED_DRIVERS_FILE.exists()

    def test_the_deferred_startup_drops_the_record(self, monkeypatch):
        startup._remember_drivers_stopped_at_shutdown({"cyberwave-driver-go2": "twin-go2"})
        monkeypatch.setattr(startup, "_DEGRADED_STARTUP_PENDING", True)
        monkeypatch.setattr(startup, "_last_degraded_retry_at", None)
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")
        monkeypatch.setattr(startup, "check_token", lambda token: startup.TOKEN_VALID)
        monkeypatch.setattr(startup, "register_edge_outcome", lambda token: startup.EDGE_REGISTERED)
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", MagicMock(return_value=[]))
        _stub_boot_extras(monkeypatch)

        assert startup.retry_degraded_startup() is True
        assert not startup.STOPPED_DRIVERS_FILE.exists()


# ===========================================================================
# 3. The deferred startup retry
# ===========================================================================


def _stub_boot_extras(monkeypatch: pytest.MonkeyPatch, *, router_enabled: bool = False) -> Any:
    """Stub the heartbeat and Zenoh router steps the deferred startup repeats."""
    extras = SimpleNamespace(
        heartbeat=MagicMock(return_value=True),
        stop_heartbeat=MagicMock(),
        router=MagicMock(return_value=True),
        zenoh_cfg=SimpleNamespace(router_enabled=router_enabled),
    )
    monkeypatch.setattr(
        startup, "_list_linked_twin_uuids_for_fingerprint", lambda *_args: ["twin-1"]
    )
    monkeypatch.setattr(startup, "_start_bootstrap_edge_health_publisher", extras.heartbeat)
    monkeypatch.setattr(startup, "_stop_bootstrap_edge_health_publisher", extras.stop_heartbeat)
    monkeypatch.setattr(startup, "_get_zenoh_config", lambda: extras.zenoh_cfg)
    monkeypatch.setattr(startup, "start_zenoh_router", extras.router)
    return extras


class TestRetryDegradedStartup:
    @pytest.fixture
    def pending(self, monkeypatch):
        monkeypatch.setattr(startup, "_DEGRADED_STARTUP_PENDING", True)
        monkeypatch.setattr(startup, "_last_degraded_retry_at", None)
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "fp")
        _stub_boot_extras(monkeypatch)

    def test_completes_the_skipped_steps_without_pulling(self, monkeypatch, pending):
        monkeypatch.setattr(startup, "check_token", lambda token: startup.TOKEN_VALID)
        monkeypatch.setattr(startup, "register_edge_outcome", lambda token: startup.EDGE_REGISTERED)
        fetch = MagicMock(return_value=[{"success": True}])
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)
        watchdog = object()

        assert startup.retry_degraded_startup(watchdog=watchdog) is True

        fetch.assert_called_once_with("tok", "env-uuid", "fp", watchdog=watchdog, pull_images=False)
        assert not startup.degraded_startup_pending()

    def test_starts_the_heartbeat_and_router_a_normal_boot_starts(self, monkeypatch, pending):
        """Without them an edge whose drivers all fail sends no heartbeat until a restart."""
        extras = _stub_boot_extras(monkeypatch, router_enabled=True)
        monkeypatch.setattr(startup, "check_token", lambda token: startup.TOKEN_VALID)
        monkeypatch.setattr(startup, "register_edge_outcome", lambda token: startup.EDGE_REGISTERED)
        monkeypatch.setattr(
            startup, "fetch_and_run_twin_drivers", MagicMock(return_value=[{"success": False}])
        )
        monitor, watchdog = object(), MagicMock()

        assert startup.retry_degraded_startup(resource_monitor=monitor, watchdog=watchdog) is True

        extras.heartbeat.assert_called_once_with(
            "tok", ["twin-1"], edge_id="fp", resource_monitor=monitor, watchdog=watchdog
        )
        extras.router.assert_called_once_with(extras.zenoh_cfg, "env-uuid")
        # No driver came up, so the bootstrap publisher keeps the edge's heartbeat.
        extras.stop_heartbeat.assert_not_called()

    def test_a_started_driver_takes_over_the_heartbeat(self, monkeypatch, pending):
        extras = _stub_boot_extras(monkeypatch)
        monkeypatch.setattr(startup, "check_token", lambda token: startup.TOKEN_VALID)
        monkeypatch.setattr(startup, "register_edge_outcome", lambda token: startup.EDGE_REGISTERED)
        monkeypatch.setattr(
            startup, "fetch_and_run_twin_drivers", MagicMock(return_value=[{"success": True}])
        )

        assert startup.retry_degraded_startup() is True

        extras.router.assert_not_called()
        extras.stop_heartbeat.assert_called_once_with()

    def test_waits_for_its_interval(self, monkeypatch, pending):
        monkeypatch.setattr(startup, "_last_degraded_retry_at", time.monotonic())
        monkeypatch.setattr(startup, "check_token", _never)

        assert startup.retry_degraded_startup() is False
        assert startup.degraded_startup_pending()

    def test_stays_pending_while_the_backend_is_down(self, monkeypatch, pending):
        monkeypatch.setattr(startup, "check_token", lambda token: startup.TOKEN_VALID)
        monkeypatch.setattr(startup, "register_edge_outcome", lambda token: startup.EDGE_REGISTERED)

        def fetch(*_args: Any, **_kwargs: Any) -> Any:
            raise ConnectionError("down again")

        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)

        assert startup.retry_degraded_startup() is False
        assert startup.degraded_startup_pending()
        assert startup.cloud_reconciliation_paused()

    def test_a_token_rejected_later_does_not_exit(self, monkeypatch, pending):
        monkeypatch.setattr(startup, "check_token", lambda token: startup.TOKEN_REJECTED)
        monkeypatch.setattr(startup, "register_edge_outcome", _never)

        assert startup.retry_degraded_startup() is False
        assert startup.degraded_startup_pending()

    def test_nothing_to_do_without_a_degraded_boot(self, monkeypatch):
        monkeypatch.setattr(startup, "check_token", _never)
        assert startup.retry_degraded_startup() is False


# ===========================================================================
# 4. Flap-stopped drivers
# ===========================================================================

NAME = "cyberwave-driver-aaaaaaaa"
TWIN = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _started(seconds_ago: float) -> str:
    """A docker ``StartedAt`` for a container started *seconds_ago*."""
    stamp = datetime.fromtimestamp(time.time() - seconds_ago, timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class TestFlapEpisodes:
    @pytest.fixture
    def flapping(self, monkeypatch):
        """A driver whose restart count keeps climbing past the threshold."""
        state: dict[str, Any] = {
            "running": False,
            "started": "",
            "restarts": (1, startup.DRIVER_RESTART_LOOP_THRESHOLD + 1),
        }
        alerts: list[str] = []
        stops: list[str] = []
        monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {NAME: TWIN})
        monkeypatch.setattr(startup, "_CONTAINER_RESTART_HISTORY", {})
        monkeypatch.setattr(startup, "_CONTAINER_LAST_RESTART_COUNT", {})
        monkeypatch.setattr(startup, "_list_driver_containers", lambda *, include_stopped: [NAME])
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda name: {
                "Image": "sha256:drv",
                "RestartCount": 9,
                "State": {
                    "Running": state["running"],
                    "Status": "restarting",
                    "StartedAt": state["started"],
                },
            },
        )
        monkeypatch.setattr(
            startup, "_track_container_restarts", lambda name, count: state["restarts"]
        )
        monkeypatch.setattr(
            startup, "_stop_driver_container", lambda name: stops.append(name) or True
        )
        monkeypatch.setattr(
            startup, "_resolve_container_twin_uuid", lambda name, inspect=None: TWIN
        )
        monkeypatch.setattr(
            startup,
            "_send_alert_for_twin",
            lambda twin, title, description, alert_type, **kw: alerts.append(alert_type),
        )
        return SimpleNamespace(state=state, alerts=alerts, stops=stops)

    def test_one_alert_per_episode(self, flapping):
        first = startup.reconcile_driver_restart_failures()
        assert first["stopped"] == 1 and first["alerts_sent"] == 1

        record = startup._FLAP_STOPPED[NAME]
        assert record.backoff == startup._FLAP_REVIVAL_BACKOFF_INITIAL_SECONDS
        assert record.image == "sha256:drv"

        startup._note_flap_revival(NAME, time.monotonic())
        again = startup.reconcile_driver_restart_failures()

        assert again["stopped"] == 1 and again["alerts_sent"] == 0
        assert flapping.alerts == ["driver_restart_loop"]
        assert len(flapping.stops) == 2

    def test_an_episode_whose_first_alert_failed_alerts_on_its_next_stop(
        self, monkeypatch, flapping
    ):
        def offline(*args, **kwargs):
            raise RuntimeError("cloud calls paused")

        monkeypatch.setattr(startup, "_send_alert_for_twin", offline)
        first = startup.reconcile_driver_restart_failures()
        assert first["stopped"] == 1 and first["alerts_sent"] == 0
        twin, message = startup._FLAP_STOPPED[NAME].undelivered_alert
        assert twin == TWIN and "restarted" in message

        monkeypatch.setattr(
            startup,
            "_send_alert_for_twin",
            lambda twin, title, description, alert_type, **kw: flapping.alerts.append(
                alert_type
            ),
        )
        startup._note_flap_revival(NAME, time.monotonic())
        again = startup.reconcile_driver_restart_failures()
        startup._note_flap_revival(NAME, time.monotonic())
        third = startup.reconcile_driver_restart_failures()

        assert again["alerts_sent"] == 1 and third["alerts_sent"] == 0
        assert flapping.alerts == ["driver_restart_loop"]
        assert startup._FLAP_STOPPED[NAME].undelivered_alert is None

    def test_a_stable_run_after_a_revival_ends_the_episode(self, flapping):
        startup.reconcile_driver_restart_failures()
        startup._note_flap_revival(NAME, time.monotonic())
        flapping.state.update(
            running=True,
            started=_started(startup._FLAP_STABLE_RUN_SECONDS + 1),
            restarts=(0, 0),
        )

        startup.reconcile_driver_restart_failures()

        assert NAME not in startup._FLAP_STOPPED

    def test_a_recent_revival_keeps_the_episode(self, flapping):
        startup.reconcile_driver_restart_failures()
        startup._note_flap_revival(NAME, time.monotonic())
        flapping.state.update(running=True, started=_started(5), restarts=(0, 0))

        startup.reconcile_driver_restart_failures()

        assert NAME in startup._FLAP_STOPPED

    def test_a_driver_brought_back_by_someone_else_alerts_on_its_next_flap(self, flapping):
        startup.reconcile_driver_restart_failures()
        # An operator restarts the driver before its revival is due, and it runs.
        flapping.state.update(
            running=True,
            started=_started(startup._FLAP_STABLE_RUN_SECONDS + 1),
            restarts=(0, 0),
        )
        startup.reconcile_driver_restart_failures()
        assert NAME not in startup._FLAP_STOPPED

        flapping.state.update(
            running=False, restarts=(1, startup.DRIVER_RESTART_LOOP_THRESHOLD + 1)
        )
        again = startup.reconcile_driver_restart_failures()

        assert again["alerts_sent"] == 1
        assert flapping.alerts == ["driver_restart_loop", "driver_restart_loop"]

    def test_a_driver_running_briefly_after_its_stop_keeps_the_episode(self, flapping):
        startup.reconcile_driver_restart_failures()
        flapping.state.update(running=True, started=_started(5), restarts=(0, 0))

        startup.reconcile_driver_restart_failures()

        assert NAME in startup._FLAP_STOPPED

    def test_a_hand_start_long_after_the_stop_keeps_the_episode(self, monkeypatch, flapping):
        startup.reconcile_driver_restart_failures()
        # No revival ran for 700 s (the backend was down); an operator just started it.
        later = time.monotonic() + 700
        monkeypatch.setattr(startup.time, "monotonic", lambda: later)
        flapping.state.update(running=True, started=_started(5), restarts=(0, 0))

        startup.reconcile_driver_restart_failures()

        assert startup._FLAP_STOPPED[NAME].backoff == startup._FLAP_REVIVAL_BACKOFF_INITIAL_SECONDS
        assert NAME in startup._FLAP_STOPPED

    @pytest.mark.parametrize("started", ["", "0001-01-01T00:00:00Z", "garbage"])
    def test_an_unknown_start_time_keeps_the_episode(self, flapping, started):
        startup.reconcile_driver_restart_failures()
        flapping.state.update(running=True, started=started, restarts=(0, 0))

        startup.reconcile_driver_restart_failures()

        assert NAME in startup._FLAP_STOPPED

    def test_the_record_of_an_unlinked_twins_container_is_dropped(self, monkeypatch):
        monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {})
        monkeypatch.setattr(startup, "_list_driver_containers", lambda *, include_stopped: [NAME])
        monkeypatch.setattr(startup, "_inspect_driver_container", lambda name: {})
        startup._record_flap_stop(NAME, {})

        startup.reconcile_driver_restart_failures()

        assert startup._FLAP_STOPPED == {}

    def test_a_record_survives_a_removed_container_its_twin_still_has(self, monkeypatch):
        monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {NAME: TWIN})
        monkeypatch.setattr(startup, "_list_driver_containers", lambda *, include_stopped: [])
        startup._record_flap_stop(NAME, {})
        startup._record_flap_stop("cyberwave-driver-gone", {})

        startup.reconcile_driver_restart_failures()

        assert set(startup._FLAP_STOPPED) == {NAME}

    def test_the_revival_backoff_doubles_up_to_its_cap(self):
        assert startup._record_flap_stop(NAME, {"Image": "sha256:drv"}) is True
        backoffs = []
        for _ in range(6):
            startup._note_flap_revival(NAME, time.monotonic())
            backoffs.append(startup._FLAP_STOPPED[NAME].backoff)
        assert backoffs == [240, 480, 960, 1800, 1800, 1800]
        assert startup._FLAP_STOPPED[NAME].revivals == 6

    def test_a_repeat_flap_waits_out_the_current_backoff(self):
        startup._record_flap_stop(NAME, {})
        startup._note_flap_revival(NAME, time.monotonic())
        before = time.monotonic()

        assert startup._record_flap_stop(NAME, {}) is False

        record = startup._FLAP_STOPPED[NAME]
        assert record.due_at >= before + 240


class TestRevivableFlapStopped:
    @pytest.fixture(autouse=True)
    def _restart_no(self, monkeypatch):
        # The flap detector leaves its stopped container on restart=no.
        monkeypatch.setattr(
            dh,
            "docker_inspect",
            lambda name: {"HostConfig": {"RestartPolicy": {"Name": "no"}}},
        )
        monkeypatch.setattr(startup, "_RESTART_FLOOR_DEFERRED", set())

    def _summary(self) -> dict[str, int]:
        return {
            "skipped_flap_backoff": 0,
            "skipped_restart_no": 0,
            "skipped_restart_floor": 0,
        }

    def test_waits_for_its_backoff(self):
        startup._record_flap_stop(NAME, {})
        summary = self._summary()

        assert startup._revivable_driver_containers([NAME], summary) == []
        assert summary["skipped_flap_backoff"] == 1

    def test_is_revived_despite_restart_no_once_due(self):
        startup._record_flap_stop(NAME, {})
        startup._FLAP_STOPPED[NAME].due_at = time.monotonic() - 1
        summary = self._summary()

        assert startup._revivable_driver_containers([NAME], summary) == [NAME]

    def test_a_declared_restart_no_driver_stays_down(self):
        summary = self._summary()

        assert startup._revivable_driver_containers([NAME], summary) == []
        assert summary["skipped_restart_no"] == 1


# ===========================================================================
# 5. Revival arguments
# ===========================================================================


class TestRevivalLaunch:
    @pytest.fixture(autouse=True)
    def _down_driver(self, monkeypatch):
        monkeypatch.setattr(startup, "_DRIVER_HEALTH_STALE", False)
        monkeypatch.setattr(startup, "_DRIVER_HEALTH_PREVIOUS", {NAME: "down"})
        monkeypatch.setattr(startup, "_CONTAINER_TWIN_MAP", {NAME: TWIN})
        monkeypatch.setattr(startup, "_CONTAINER_RESTART_POLICY", {})
        monkeypatch.setattr(startup, "_RESTART_FLOOR_DEFERRED", set())
        monkeypatch.setattr(startup, "_REMOVED_SINCE", {})
        monkeypatch.setattr(startup, "_LAST_REVIVAL_ATTEMPT_AT", None)
        monkeypatch.setattr(
            dh,
            "docker_inspect",
            lambda name: {"HostConfig": {"RestartPolicy": {"Name": "unless-stopped"}}},
        )
        monkeypatch.setattr(startup, "driver_activation_in_progress", lambda twin=None: False)
        monkeypatch.setattr(startup, "begin_loop_launch", lambda twins: True)
        monkeypatch.setattr(startup, "end_loop_launch", lambda twins: None)
        monkeypatch.setattr(startup, "load_token", lambda: "tok")
        monkeypatch.setattr(startup, "load_saved_fingerprint", lambda: "fp")
        monkeypatch.setattr(startup, "load_environment_uuid", lambda: "env-uuid")

    def test_passes_the_watchdog_and_skips_pulls(self, monkeypatch):
        fetch = MagicMock(return_value=[])
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)
        watchdog = object()

        summary = startup.reconcile_driver_revival(watchdog=watchdog)

        assert summary["revived_attempted"] == 1
        fetch.assert_called_once_with(
            "tok",
            "env-uuid",
            "fp",
            watchdog=watchdog,
            only_container_names={NAME},
            pull_images=False,
        )

    def test_counts_a_flap_revival(self, monkeypatch):
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", MagicMock(return_value=[]))
        startup._record_flap_stop(NAME, {})
        startup._FLAP_STOPPED[NAME].due_at = time.monotonic() - 1

        startup.reconcile_driver_revival()

        record = startup._FLAP_STOPPED[NAME]
        assert record.revivals == 1
        assert record.backoff == 2 * startup._FLAP_REVIVAL_BACKOFF_INITIAL_SECONDS

    def _revive_flap_stopped(self) -> None:
        startup._FLAP_STOPPED[NAME].due_at = time.monotonic() - 1
        startup._LAST_REVIVAL_ATTEMPT_AT = None
        startup.reconcile_driver_revival()

    def test_a_flap_record_no_twin_declares_is_dropped_after_six_passes(self, monkeypatch):
        fetch = MagicMock(return_value=[])
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)
        startup._record_flap_stop(NAME, {"Image": "sha256:drv"})

        # A miss may be a swallowed asset-lookup error, so a few in a row keep the record.
        for _ in range(5):
            self._revive_flap_stopped()
        assert startup._FLAP_STOPPED[NAME].no_spec_passes == 5
        self._revive_flap_stopped()

        assert startup._FLAP_STOPPED == {}
        assert fetch.call_count == 6
        # Left on restart=no, so revival leaves it down from now on.
        monkeypatch.setattr(
            dh, "docker_inspect", lambda name: {"HostConfig": {"RestartPolicy": {"Name": "no"}}}
        )
        startup._LAST_REVIVAL_ATTEMPT_AT = None
        summary = startup.reconcile_driver_revival()
        assert summary["skipped_restart_no"] == 1
        assert fetch.call_count == 6

    def test_a_removed_flap_stopped_driver_also_gets_six_passes(self, monkeypatch):
        fetch = MagicMock(return_value=[])
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)
        monkeypatch.setattr(startup, "_DRIVER_HEALTH_PREVIOUS", {NAME: "removed"})
        monkeypatch.setattr(startup, "_REMOVED_SINCE", {NAME: time.monotonic() - 3600})
        monkeypatch.setattr(startup, "_REMOVED_MISSES", {})
        startup._record_flap_stop(NAME, {})

        # A backend outage answers every pass with no spec; three must not forget it.
        for _ in range(5):
            self._revive_flap_stopped()
        assert NAME in startup._FLAP_STOPPED
        assert startup._CONTAINER_TWIN_MAP == {NAME: TWIN}
        self._revive_flap_stopped()

        assert NAME not in startup._FLAP_STOPPED
        assert NAME not in startup._CONTAINER_TWIN_MAP
        assert fetch.call_count == 6

    def test_a_revival_that_fails_after_create_keeps_the_driver_due(self, monkeypatch):
        def fetch(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            # The launcher removes a container that never reached running and
            # unmaps it through its on_removed callback.
            startup._CONTAINER_TWIN_MAP.pop(NAME, None)
            return [{"twin_uuid": TWIN, "service_name": None, "success": False}]

        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)
        monkeypatch.setattr(startup, "_list_driver_containers", lambda *, include_stopped: [])
        startup._record_flap_stop(NAME, {})

        self._revive_flap_stopped()
        startup.reconcile_driver_restart_failures()

        assert startup._CONTAINER_TWIN_MAP == {NAME: TWIN}
        assert NAME in startup._FLAP_STOPPED

    def test_a_launched_flap_revival_resets_its_misses(self, monkeypatch):
        launched = [{"twin_uuid": TWIN, "service_name": None, "success": False}]
        fetch = MagicMock(side_effect=[[], [], launched, [], []])
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)
        startup._record_flap_stop(NAME, {})

        for _ in range(5):
            self._revive_flap_stopped()

        assert startup._FLAP_STOPPED[NAME].no_spec_passes == 2

    def _stopped_offline(self, monkeypatch) -> list[str]:
        """A flap-stopped driver whose stop alert never reached the backend."""
        events: list[str] = []
        monkeypatch.setattr(
            startup,
            "fetch_and_run_twin_drivers",
            lambda *a, **kw: events.append("launch") or [],
        )
        monkeypatch.setattr(
            startup,
            "_send_alert_for_twin",
            lambda twin, title, description, alert_type, **kw: events.append(
                f"{alert_type}:{twin}:{description}"
            ),
        )
        startup._record_flap_stop(NAME, {})
        startup._FLAP_STOPPED[NAME].undelivered_alert = (TWIN, "restarted 5 times.")
        return events

    def test_an_episode_stopped_only_offline_alerts_at_its_revival(self, monkeypatch):
        events = self._stopped_offline(monkeypatch)

        self._revive_flap_stopped()
        self._revive_flap_stopped()

        alerts = [e for e in events if e != "launch"]
        assert len(alerts) == 1
        assert alerts[0].startswith(f"driver_restart_loop:{TWIN}:restarted 5 times.")
        assert "backend was unreachable" in alerts[0]
        assert events.index("launch") < events.index(alerts[0])
        record = startup._FLAP_STOPPED[NAME]
        assert record.alerted and record.undelivered_alert is None

    def test_a_delayed_alert_that_fails_is_sent_at_the_next_revival(self, monkeypatch):
        events = self._stopped_offline(monkeypatch)
        deliver = startup._send_alert_for_twin

        def offline(*args, **kwargs):
            raise RuntimeError("cloud calls paused")

        monkeypatch.setattr(startup, "_send_alert_for_twin", offline)
        self._revive_flap_stopped()
        assert startup._FLAP_STOPPED[NAME].undelivered_alert is not None

        monkeypatch.setattr(startup, "_send_alert_for_twin", deliver)
        self._revive_flap_stopped()

        assert len([e for e in events if e != "launch"]) == 1
        assert startup._FLAP_STOPPED[NAME].alerted

    def test_an_alerted_episode_sends_nothing_at_its_revival(self, monkeypatch):
        events = self._stopped_offline(monkeypatch)
        startup._FLAP_STOPPED[NAME].alerted = True

        self._revive_flap_stopped()

        assert events == ["launch"]

    def test_skipped_while_the_backend_is_unreachable(self, monkeypatch):
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", _never)
        startup.note_cloud_failure(ConnectionError("down"))

        summary = startup.reconcile_driver_revival()

        assert summary["skipped_backend_unreachable"] == 1
        assert summary["revived_attempted"] == 0

    def test_a_network_error_opens_the_breaker(self, monkeypatch):
        def fetch(*_args: Any, **_kwargs: Any) -> Any:
            raise ConnectionError("down")

        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)

        startup.reconcile_driver_revival()

        assert startup.cloud_reconciliation_paused()

    def test_an_attempt_the_backend_cannot_answer_is_not_a_revival(self, monkeypatch):
        fetch = MagicMock(side_effect=_wrapped_transport_error())
        monkeypatch.setattr(startup, "fetch_and_run_twin_drivers", fetch)
        startup._record_flap_stop(NAME, {})
        record = startup._FLAP_STOPPED[NAME]
        record.due_at = time.monotonic() - 1
        before = (record.revivals, record.backoff, record.due_at)

        startup.reconcile_driver_revival()

        assert startup.cloud_reconciliation_paused()
        assert (record.revivals, record.backoff, record.due_at) == before
        assert startup._LAST_REVIVAL_ATTEMPT_AT is None

        # The first pass after the backend answers revives the driver.
        startup.note_cloud_success()
        fetch.side_effect = None
        fetch.return_value = []

        summary = startup.reconcile_driver_revival()

        assert summary["revived_attempted"] == 1
        assert fetch.call_count == 2
        assert record.revivals == 1
        assert record.backoff == 2 * startup._FLAP_REVIVAL_BACKOFF_INITIAL_SECONDS


# ===========================================================================
# 6. Periodic cleanup keeps flap-stopped drivers
# ===========================================================================


class TestCleanupKeepsFlapStopped:
    def test_cleanup_passes_the_flap_stopped_names_and_images(self, monkeypatch):
        monkeypatch.setattr(startup, "_is_periodic_docker_cleanup_disabled", lambda: False)
        for attribute in (
            "_last_container_prune_time",
            "_last_image_prune_time",
            "_last_volume_prune_time",
        ):
            monkeypatch.setattr(startup, attribute, float("-inf"))
        container_prune = MagicMock(return_value=0)
        image_prune = MagicMock(return_value=True)
        monkeypatch.setattr(dh, "docker_prune_stopped_cyberwave_containers", container_prune)
        monkeypatch.setattr(dh, "docker_prune_unused_images", image_prune)
        monkeypatch.setattr(dh, "docker_prune_dangling_volumes", lambda: True)
        monkeypatch.setattr(dh, "docker_prune_navigation_bundle_volumes", lambda **_kw: True)
        startup._record_flap_stop(NAME, {"Image": "sha256:drv"})

        startup._run_periodic_docker_cleanup()

        assert container_prune.call_args.kwargs["keep_names"] == {NAME}
        assert image_prune.call_args.kwargs["keep"] == {"sha256:drv"}


# ===========================================================================
# 7. Runtime-loop guards
# ===========================================================================


class TestRuntimeLoopGuards:
    @pytest.fixture(autouse=True)
    def _loop(self, monkeypatch):
        startup.shutdown_event.clear()
        self.calls: list[str] = []

        def record(name: str, result: Any = None):
            def step(*_args: Any, **_kwargs: Any) -> Any:
                self.calls.append(name)
                return result

            return step

        monkeypatch.setattr(startup, "reconcile_driver_log_streams", record("logs", 0))
        monkeypatch.setattr(
            startup,
            "reconcile_driver_restart_failures",
            record("flaps", {"inspected": 0, "flapping": 0, "stopped": 0, "alerts_sent": 0}),
        )
        monkeypatch.setattr(startup, "reconcile_driver_health_for_worker", record("health", {}))
        monkeypatch.setattr(startup, "retry_degraded_startup", record("degraded"))
        monkeypatch.setattr(
            startup,
            "reconcile_driver_revival",
            record(
                "revival",
                {
                    key: 0
                    for key in (
                        "down",
                        "skipped_orphan",
                        "revived_attempted",
                    )
                },
            ),
        )
        monkeypatch.setattr(
            startup,
            "reconcile_twin_json_file_sync",
            record("twin_sync", {"tracked": 0, "changed": 0, "synced": 0}),
        )
        monkeypatch.setattr(startup, "reconcile_driver_metadata_restarts", record("metadata"))
        monkeypatch.setattr(startup, "reconcile_camera_config_drift", record("camera"))
        monkeypatch.setattr(startup, "reconcile_sensor_device_presence", record("sensors"))
        monkeypatch.setattr(startup, "_reconcile_worker_watcher", lambda w: w)
        monkeypatch.setattr(startup, "ensure_edge_command_subscription", record("edge_commands"))
        monkeypatch.setattr(startup, "ensure_twin_command_subscriptions", record("twin_commands"))
        from cyberwave_edge_core import driver_credentials

        monkeypatch.setattr(
            driver_credentials, "maybe_renew_driver_tokens", record("token_renewal")
        )
        monkeypatch.setattr(startup, "_EDGE_JSON_SYNC_INTERVAL_LOOPS", 10_000)
        monkeypatch.setattr(startup, "_WORKER_SYNC_INTERVAL_LOOPS", 10_000)
        monkeypatch.setattr(startup, "_edge_json_sync_loop_counter", 0)
        monkeypatch.setattr(startup, "_worker_sync_loop_counter", 0)
        monkeypatch.setattr(startup, "_run_periodic_docker_cleanup", record("cleanup"))
        monkeypatch.setattr(startup, "_graceful_shutdown", lambda w: None)
        from cyberwave_edge_core import local_portal

        monkeypatch.setattr(local_portal, "start_from_env", lambda **_kw: None)
        yield
        startup.shutdown_event.clear()

    def _run_ticks(self, monkeypatch, ticks: int, watchdog: Any = None) -> None:
        waits = [0]
        event = startup.shutdown_event

        def wait(timeout: float | None = None) -> bool:
            waits[0] += 1
            if waits[0] >= ticks:
                event.set()
            return event.is_set()

        monkeypatch.setattr(event, "wait", wait)
        startup.run_runtime_loop(watchdog=watchdog)
        assert waits[0] == ticks

    @pytest.mark.parametrize("deferred_done", [True, False])
    def test_revival_sits_out_the_tick_the_deferred_startup_launched_the_drivers(
        self, monkeypatch, deferred_done
    ):
        """The health snapshot predates that launch: revival would start a down driver twice."""
        skips: list[bool] = []

        def revival(*, skip_revival: bool = False, watchdog: Any = None) -> dict[str, int]:
            skips.append(skip_revival)
            return {"down": 0, "skipped_orphan": 0, "revived_attempted": 0}

        monkeypatch.setattr(startup, "retry_degraded_startup", lambda **_kw: deferred_done)
        monkeypatch.setattr(startup, "reconcile_driver_revival", revival)

        self._run_ticks(monkeypatch, 1)

        assert skips == [deferred_done]

    def test_failing_local_steps_do_not_end_the_loop(self, monkeypatch):
        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("boom")

        for name in (
            "reconcile_driver_log_streams",
            "reconcile_driver_restart_failures",
            "reconcile_twin_json_file_sync",
            "_run_periodic_docker_cleanup",
        ):
            monkeypatch.setattr(startup, name, boom)

        self._run_ticks(monkeypatch, 3)

        assert self.calls.count("revival") == 3

    def test_the_watchdog_is_fed_between_steps(self, monkeypatch):
        watchdog = MagicMock()

        self._run_ticks(monkeypatch, 2, watchdog=watchdog)

        assert watchdog.ping.call_count >= 2 * 6

    def test_a_failing_ping_does_not_end_the_loop(self, monkeypatch):
        watchdog = MagicMock()
        watchdog.ping.side_effect = OSError("watchdog gone")

        self._run_ticks(monkeypatch, 2, watchdog=watchdog)

    def test_cloud_steps_are_skipped_while_the_backend_is_down(self, monkeypatch):
        startup.note_cloud_failure(ConnectionError("down"))

        self._run_ticks(monkeypatch, 2)

        for skipped in ("degraded", "edge_commands", "twin_commands", "token_renewal"):
            assert skipped not in self.calls
        for kept in ("logs", "flaps", "health", "revival", "cleanup"):
            assert self.calls.count(kept) == 2

    def test_cloud_steps_run_while_the_backend_answers(self, monkeypatch):
        self._run_ticks(monkeypatch, 1)

        for step in ("degraded", "edge_commands", "twin_commands", "token_renewal"):
            assert step in self.calls

    def test_a_network_error_in_a_cloud_step_pauses_the_rest(self, monkeypatch):
        def offline() -> None:
            self.calls.append("edge_commands")
            raise ConnectionError("down")

        monkeypatch.setattr(startup, "ensure_edge_command_subscription", offline)

        self._run_ticks(monkeypatch, 2)

        assert self.calls.count("edge_commands") == 1
        assert "twin_commands" not in self.calls

    def test_a_twin_the_backend_keeps_failing_does_not_pause_the_rest(self, monkeypatch):
        def broken_twin(*_args: Any, **_kwargs: Any) -> dict[str, int]:
            self.calls.append("twin_sync")
            startup.note_twin_call_failure(_api_error(503))
            return {"tracked": 1, "changed": 1, "synced": 0}

        monkeypatch.setattr(startup, "reconcile_twin_json_file_sync", broken_twin)

        self._run_ticks(monkeypatch, 3)

        assert self.calls.count("twin_sync") == 3
        for step in ("edge_commands", "twin_commands", "token_renewal"):
            assert self.calls.count(step) == 3


# ===========================================================================
# 8. Shared MQTT client
# ===========================================================================


def _mqtt(*, connected: bool, connecting: bool = False) -> MagicMock:
    mqtt = MagicMock()
    mqtt.connected = connected
    mqtt.connecting_in_background = connecting
    return mqtt


class TestSharedMqttClient:
    @pytest.fixture(autouse=True)
    def _no_client(self, monkeypatch):
        monkeypatch.setattr(startup, "_shared_mqtt_client", None)
        monkeypatch.setattr(startup, "_resolve_mqtt_kwargs", lambda: {})

    def test_a_disconnected_client_is_reused_and_reconnected(self, monkeypatch):
        existing = SimpleNamespace(mqtt=_mqtt(connected=False))
        monkeypatch.setattr(startup, "_shared_mqtt_client", existing)
        monkeypatch.setattr(startup, "Cyberwave", _never)

        assert startup._get_shared_mqtt_client("tok") is existing
        existing.mqtt.connect_in_background.assert_called_once()

    def test_a_client_already_reconnecting_is_left_alone(self, monkeypatch):
        existing = SimpleNamespace(mqtt=_mqtt(connected=False, connecting=True))
        monkeypatch.setattr(startup, "_shared_mqtt_client", existing)
        monkeypatch.setattr(startup, "Cyberwave", _never)

        assert startup._get_shared_mqtt_client("tok") is existing
        existing.mqtt.connect_in_background.assert_not_called()

    def test_a_new_client_reconnects_quickly(self, monkeypatch):
        client = SimpleNamespace(mqtt=_mqtt(connected=True))
        monkeypatch.setattr(startup, "Cyberwave", lambda **_kw: client)

        assert startup._get_shared_mqtt_client("tok") is client
        client.mqtt._client.client.reconnect_delay_set.assert_called_once_with(
            min_delay=1, max_delay=30
        )
        client.mqtt.connect.assert_called_once()
        assert startup._shared_mqtt_client is client

    def test_a_failed_first_connect_keeps_the_client_connecting(self, monkeypatch):
        client = SimpleNamespace(mqtt=_mqtt(connected=False))
        client.mqtt.connect.side_effect = ConnectionError("down")
        monkeypatch.setattr(startup, "Cyberwave", lambda **_kw: client)

        assert startup._get_shared_mqtt_client("tok") is client
        client.mqtt.connect_in_background.assert_called_once()
        assert startup._shared_mqtt_client is client

    def test_the_blocking_connect_is_skipped_while_the_backend_is_down(self, monkeypatch):
        client = SimpleNamespace(mqtt=_mqtt(connected=False))
        monkeypatch.setattr(startup, "Cyberwave", lambda **_kw: client)
        startup.note_cloud_failure(ConnectionError("down"))

        assert startup._get_shared_mqtt_client("tok") is client
        client.mqtt.connect.assert_not_called()
        client.mqtt.connect_in_background.assert_called_once()

    def test_driver_logs_are_not_published_while_disconnected(self):
        client = SimpleNamespace(mqtt=_mqtt(connected=False))

        driver_logs._publish_driver_log_message("line", NAME, mqtt_client=client, mqtt_topic="t")

        client.mqtt.publish.assert_not_called()
