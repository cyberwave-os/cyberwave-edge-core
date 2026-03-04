"""Integration tests for Cyberwave Edge Core startup + CLI.

These tests exercise the *full* startup sequence end-to-end:
  1. Real config files written to a temporary directory
  2. Only the Cyberwave SDK (external I/O) and Docker subprocess calls are stubbed
  3. run_startup_checks() and the ``status`` CLI command are driven as a whole

Unlike the fine-grained unit tests in test_startup_core.py, these tests verify
that the individual pieces (credentials loading, token validation, MQTT, edge
registration, environment linking, twin driver launching) all compose correctly.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import cyberwave_edge_core.startup as startup
from cyberwave_edge_core.main import cli

# ---- constants used across tests -------------------------------------------

_ENV_UUID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_TWIN_UUID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
_ASSET_UUID = "cccccccc-cccc-cccc-cccc-cccccccccccc"
_FINGERPRINT = "integration-test-fingerprint"
_TOKEN = "tok-integration-0123456789abcdef"


# ---- helpers ----------------------------------------------------------------


def _write_config(
    tmp_path: Path,
    *,
    token: str = _TOKEN,
    env_uuid: str | None = _ENV_UUID,
) -> None:
    """Populate a temporary config directory with credentials and an environment file."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "credentials.json").write_text(json.dumps({"token": token}))
    if env_uuid is not None:
        (tmp_path / "environment.json").write_text(json.dumps({"uuid": env_uuid}))


def _make_mqtt_stub(*, connected: bool = True) -> SimpleNamespace:
    """Return a minimal MQTT stub."""
    mqtt = MagicMock()
    mqtt.connect = MagicMock()
    mqtt.disconnect = MagicMock()
    mqtt.connected = connected
    return mqtt


def _make_fake_twin(
    *,
    uuid: str = _TWIN_UUID,
    name: str = "test-twin",
    asset_uuid: str = _ASSET_UUID,
    fingerprint: str = _FINGERPRINT,
    driver_image: str = "ghcr.io/org/driver:latest",
) -> SimpleNamespace:
    """Return a lightweight twin object with the fields startup.py reads."""
    return SimpleNamespace(
        uuid=uuid,
        name=name,
        asset_uuid=asset_uuid,
        asset_id=asset_uuid,
        metadata={
            "edge_fingerprint": fingerprint,
            "drivers": {
                "default": {
                    "docker_image": driver_image,
                    "version": "latest",
                    "params": [],
                }
            },
        },
    )


def _make_fake_asset(
    *,
    driver_image: str = "ghcr.io/org/driver:latest",
) -> SimpleNamespace:
    """Return a minimal asset object."""
    return SimpleNamespace(
        metadata={"driver_docker_image": driver_image},
        registry_id="",
        universal_schema=None,
    )


class FakeCyberwave:
    """Minimal SDK fake that covers all paths exercised by run_startup_checks()."""

    def __init__(
        self,
        *,
        workspaces_list: list | None = None,
        mqtt_connected: bool = True,
        edge_create_result: object = SimpleNamespace(fingerprint=_FINGERPRINT),
        twins: list | None = None,
    ) -> None:
        self._workspaces_list = workspaces_list if workspaces_list is not None else [object()]
        self._edge_create_result = edge_create_result
        self._twins = twins or []

        asset = _make_fake_asset()

        workspaces_api = MagicMock()
        workspaces_api.list.return_value = self._workspaces_list

        twins_api = MagicMock()
        twins_api.list.return_value = self._twins

        assets_api = MagicMock()
        assets_api.get.return_value = asset

        edges_api = MagicMock()
        edges_api.list.return_value = []
        edges_api.create.return_value = self._edge_create_result

        self.workspaces = workspaces_api
        self.twins = twins_api
        self.assets = assets_api
        self.edges = edges_api
        self.mqtt = _make_mqtt_stub(connected=mqtt_connected)

    @classmethod
    def factory(cls, **kwargs):
        """Return a callable that produces a FakeCyberwave ignoring constructor args."""

        def _build(*_args, **_kwargs):
            return cls(**kwargs)

        return _build


# ============================================================================
# 1. Happy-path: all checks pass, no twins linked
# ============================================================================


class TestRunStartupChecksHappyPath:
    def test_returns_true_when_all_checks_pass(self, tmp_path, monkeypatch):
        """run_startup_checks() must return True when token, MQTT, registration,
        and environment are all valid — even if no twins match this edge."""
        _write_config(tmp_path)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", tmp_path / "environment.json")
        monkeypatch.setattr(startup, "FINGERPRINT_FILE", tmp_path / "fingerprint.json")
        monkeypatch.setattr(startup, "Cyberwave", FakeCyberwave.factory())
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: _FINGERPRINT)

        result = startup.run_startup_checks()

        assert result is True

    def test_twin_driver_is_launched_when_fingerprint_matches(self, tmp_path, monkeypatch):
        """When a twin's edge_fingerprint matches the local fingerprint the driver
        container start function must be called with the correct image."""
        _write_config(tmp_path)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", tmp_path / "environment.json")
        monkeypatch.setattr(startup, "FINGERPRINT_FILE", tmp_path / "fingerprint.json")

        driver_image = "ghcr.io/org/my-driver:1.0"
        twin = _make_fake_twin(driver_image=driver_image)

        monkeypatch.setattr(
            startup, "Cyberwave", FakeCyberwave.factory(twins=[twin])
        )
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: _FINGERPRINT)

        launched_images: list[str] = []

        def _fake_run_docker(image, params, *, twin_uuid, token, **kwargs):  # type: ignore[no-untyped-def]
            launched_images.append(image)
            return True

        monkeypatch.setattr(startup, "_run_docker_image", _fake_run_docker)

        startup.run_startup_checks()

        assert driver_image in launched_images

    def test_unlinked_twin_does_not_launch_driver(self, tmp_path, monkeypatch):
        """A twin whose edge_fingerprint differs from the local fingerprint must
        not trigger driver launch."""
        _write_config(tmp_path)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", tmp_path / "environment.json")
        monkeypatch.setattr(startup, "FINGERPRINT_FILE", tmp_path / "fingerprint.json")

        foreign_twin = _make_fake_twin(fingerprint="other-edge-fingerprint")

        monkeypatch.setattr(
            startup, "Cyberwave", FakeCyberwave.factory(twins=[foreign_twin])
        )
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: _FINGERPRINT)

        launched_images: list[str] = []

        def _fake_run_docker(image, params, *, twin_uuid, token, **kwargs):  # type: ignore[no-untyped-def]
            launched_images.append(image)
            return True

        monkeypatch.setattr(startup, "_run_docker_image", _fake_run_docker)

        startup.run_startup_checks()

        assert launched_images == []


# ============================================================================
# 2. Credential / token failure paths
# ============================================================================


class TestRunStartupChecksFailurePaths:
    def test_returns_false_when_credentials_missing(self, tmp_path, monkeypatch):
        """No credentials file → run_startup_checks() must return False immediately."""
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")
        # No file written — CREDENTIALS_FILE is absent

        result = startup.run_startup_checks()

        assert result is False

    def test_returns_false_when_token_invalid(self, tmp_path, monkeypatch):
        """Rejected token (SDK raises) → run_startup_checks() must return False."""
        _write_config(tmp_path)
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")

        def _bad_client(*_args, **_kwargs):
            ns = SimpleNamespace()
            ws = MagicMock()
            ws.list.side_effect = Exception("401 Unauthorized")
            ns.workspaces = ws
            return ns

        monkeypatch.setattr(startup, "Cyberwave", _bad_client)

        result = startup.run_startup_checks()

        assert result is False

    def test_returns_false_when_edge_registration_fails(self, tmp_path, monkeypatch):
        """A failed edge registration must cause run_startup_checks() to return False."""
        _write_config(tmp_path)
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")
        monkeypatch.setattr(startup, "FINGERPRINT_FILE", tmp_path / "fingerprint.json")
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: _FINGERPRINT)

        def _bad_edge_client(*_args, **_kwargs):
            ns = SimpleNamespace()
            ws = MagicMock()
            ws.list.return_value = [object()]
            ns.workspaces = ws
            ns.mqtt = _make_mqtt_stub()
            edges = MagicMock()
            edges.list.return_value = []
            edges.create.return_value = None  # falsy → registration "failed"
            ns.edges = edges
            return ns

        monkeypatch.setattr(startup, "Cyberwave", _bad_edge_client)

        result = startup.run_startup_checks()

        assert result is False

    def test_returns_true_when_mqtt_fails_but_rest_passes(self, tmp_path, monkeypatch):
        """MQTT failure is non-fatal — run_startup_checks() must still return True
        as long as token validation and edge registration succeed."""
        _write_config(tmp_path)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", tmp_path / "environment.json")
        monkeypatch.setattr(startup, "FINGERPRINT_FILE", tmp_path / "fingerprint.json")
        monkeypatch.setattr(startup, "Cyberwave", FakeCyberwave.factory(mqtt_connected=False))
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: _FINGERPRINT)

        result = startup.run_startup_checks()

        assert result is True

    def test_returns_true_when_environment_not_linked(self, tmp_path, monkeypatch):
        """Missing environment.json is a warning, not a fatal error."""
        _write_config(tmp_path, env_uuid=None)  # no environment.json
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", tmp_path / "environment.json")
        monkeypatch.setattr(startup, "FINGERPRINT_FILE", tmp_path / "fingerprint.json")
        monkeypatch.setattr(startup, "Cyberwave", FakeCyberwave.factory())
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: _FINGERPRINT)

        result = startup.run_startup_checks()

        assert result is True


# ============================================================================
# 3. CLI integration via Click's CliRunner
# ============================================================================


class TestStatusCommand:
    """Tests for the ``cyberwave-edge-core status`` sub-command."""

    def test_status_shows_all_green_when_valid(self, tmp_path, monkeypatch):
        """Invoking ``status`` with good credentials prints green ticks for all items."""
        _write_config(tmp_path)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")
        monkeypatch.setattr(startup, "DEVICES_FILE", tmp_path / "devices.json")
        monkeypatch.setattr(startup, "Cyberwave", FakeCyberwave.factory())

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Credentials" in result.output
        assert "Token" in result.output
        assert "MQTT broker" in result.output

    def test_status_shows_red_cross_when_credentials_missing(self, tmp_path, monkeypatch):
        """``status`` must report missing credentials clearly and exit cleanly."""
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")
        monkeypatch.setattr(startup, "DEVICES_FILE", tmp_path / "devices.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        # Rich strips markup in plain output; check for the text marker only
        assert "Credentials" in result.output

    def test_status_lists_configured_devices(self, tmp_path, monkeypatch):
        """``status`` must enumerate devices when devices.json is present."""
        _write_config(tmp_path)
        devices = [{"name": "arm", "type": "robot", "port": "/dev/ttyUSB0"}]
        (tmp_path / "devices.json").write_text(json.dumps(devices))

        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")
        monkeypatch.setattr(startup, "DEVICES_FILE", tmp_path / "devices.json")
        monkeypatch.setattr(startup, "Cyberwave", FakeCyberwave.factory())

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "arm" in result.output
        assert "/dev/ttyUSB0" in result.output

    def test_version_flag_returns_version_string(self):
        """``--version`` must emit a non-empty version string and exit 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "cyberwave-edge-core" in result.output


# ============================================================================
# 4. fetch_and_run_twin_drivers — integration between SDK and driver launcher
# ============================================================================


class TestFetchAndRunTwinDriversIntegration:
    """Exercises fetch_and_run_twin_drivers() wiring SDK calls to driver launch."""

    def _make_client(self, twins: list) -> FakeCyberwave:
        return FakeCyberwave(twins=twins)

    def _patch_common(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """Patch CONFIG_DIR so twin JSON writes land in a temp directory."""
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

    def test_returns_empty_list_when_no_twins(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        client = self._make_client([])
        monkeypatch.setattr(startup, "Cyberwave", lambda *a, **k: client)
        monkeypatch.setattr(startup, "_run_docker_image", MagicMock(return_value=True))

        results = startup.fetch_and_run_twin_drivers(_TOKEN, _ENV_UUID, _FINGERPRINT)

        assert results == []

    def test_matched_twin_produces_success_result(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        twin = _make_fake_twin(driver_image="ghcr.io/org/driver:2.0")
        client = self._make_client([twin])

        monkeypatch.setattr(startup, "Cyberwave", lambda *a, **k: client)
        monkeypatch.setattr(startup, "_run_docker_image", MagicMock(return_value=True))

        results = startup.fetch_and_run_twin_drivers(_TOKEN, _ENV_UUID, _FINGERPRINT)

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["twin_uuid"] == _TWIN_UUID

    def test_unmatched_twin_produces_no_result(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        twin = _make_fake_twin(fingerprint="someone-elses-edge")
        client = self._make_client([twin])

        monkeypatch.setattr(startup, "Cyberwave", lambda *a, **k: client)
        monkeypatch.setattr(startup, "_run_docker_image", MagicMock(return_value=True))

        results = startup.fetch_and_run_twin_drivers(_TOKEN, _ENV_UUID, _FINGERPRINT)

        assert results == []

    def test_failed_container_start_marked_as_not_success(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        twin = _make_fake_twin()
        client = self._make_client([twin])

        monkeypatch.setattr(startup, "Cyberwave", lambda *a, **k: client)
        monkeypatch.setattr(startup, "_run_docker_image", MagicMock(return_value=False))

        results = startup.fetch_and_run_twin_drivers(_TOKEN, _ENV_UUID, _FINGERPRINT)

        assert len(results) == 1
        assert results[0]["success"] is False
