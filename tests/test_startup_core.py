"""Tests for core startup.py functions.

Covers the highest-priority untested areas:
  1. load_token  — missing / malformed credentials file
  2. write_or_update_twin_json_file — deep-merge preserves existing keys
  3. write_or_update_twin_json_file — directory at path replaced by file
  4. load_environment_uuid — retry logic
  5. _remove_cached_twin_json_files — protected files never deleted
"""

from __future__ import annotations

import io
import itertools
import json
import logging
import os
import stat
import subprocess
import threading
import uuid as _uuid_module
from pathlib import Path
from unittest.mock import patch

import pytest

import cyberwave_edge_core.driver_selection as driver_selection
import cyberwave_edge_core.startup as startup

# ===========================================================================
# 0. config dir resolution
# ===========================================================================


class TestResolveConfigDir:
    def test_env_override_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("CYBERWAVE_EDGE_CONFIG_DIR", "/tmp/cw-custom")
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")

        assert startup._resolve_config_dir().as_posix() == "/tmp/cw-custom"

    def test_macos_uses_invoking_user_home_when_running_via_sudo(self, monkeypatch):
        monkeypatch.delenv("CYBERWAVE_EDGE_CONFIG_DIR", raising=False)
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            startup, "_resolve_sudo_user_home", lambda: startup.Path("/Users/alice")
        )
        monkeypatch.setattr(startup.Path, "home", lambda: startup.Path("/var/root"))

        assert startup._resolve_config_dir() == startup.Path("/Users/alice/.cyberwave")

    def test_linux_default_uses_home_dir(self, monkeypatch):
        monkeypatch.delenv("CYBERWAVE_EDGE_CONFIG_DIR", raising=False)
        monkeypatch.setattr(startup, "_resolve_sudo_user_home", lambda: None)
        monkeypatch.setattr(startup.Path, "home", lambda: startup.Path("/home/testuser"))

        assert startup._resolve_config_dir() == startup.Path("/home/testuser/.cyberwave")

    def test_linux_sudo_uses_invoking_user_home(self, monkeypatch):
        monkeypatch.delenv("CYBERWAVE_EDGE_CONFIG_DIR", raising=False)
        monkeypatch.setattr(startup, "_resolve_sudo_user_home", lambda: startup.Path("/home/alice"))
        monkeypatch.setattr(startup.Path, "home", lambda: startup.Path("/root"))

        assert startup._resolve_config_dir() == startup.Path("/home/alice/.cyberwave")

    def test_migrate_legacy_config_copies_json_files(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CYBERWAVE_EDGE_CONFIG_DIR", raising=False)
        legacy_dir = tmp_path / "legacy"
        target_dir = tmp_path / "new"
        legacy_dir.mkdir()
        (legacy_dir / "credentials.json").write_text('{"token":"abc"}')
        (legacy_dir / "environment.json").write_text('{"uuid":"123"}')
        monkeypatch.setattr(startup, "_LEGACY_SYSTEM_CONFIG_DIR", legacy_dir)

        startup._migrate_legacy_config(target_dir)

        assert (target_dir / "credentials.json").exists()
        assert (target_dir / "environment.json").exists()

    def test_migrate_legacy_config_skips_when_env_override_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CYBERWAVE_EDGE_CONFIG_DIR", str(tmp_path / "custom"))
        legacy_dir = tmp_path / "legacy"
        target_dir = tmp_path / "new"
        legacy_dir.mkdir()
        (legacy_dir / "credentials.json").write_text('{"token":"abc"}')
        monkeypatch.setattr(startup, "_LEGACY_SYSTEM_CONFIG_DIR", legacy_dir)

        startup._migrate_legacy_config(target_dir)

        assert not (target_dir / "credentials.json").exists()

    def test_migrate_legacy_config_does_not_overwrite_existing_target_json(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("CYBERWAVE_EDGE_CONFIG_DIR", raising=False)
        legacy_dir = tmp_path / "legacy"
        target_dir = tmp_path / "new"
        legacy_dir.mkdir()
        target_dir.mkdir()
        (legacy_dir / "credentials.json").write_text('{"token":"legacy"}')
        (legacy_dir / "environment.json").write_text('{"uuid":"legacy-env"}')
        (target_dir / "environment.json").write_text('{"uuid":"new-env"}')
        monkeypatch.setattr(startup, "_LEGACY_SYSTEM_CONFIG_DIR", legacy_dir)

        startup._migrate_legacy_config(target_dir)

        assert (target_dir / "credentials.json").exists()
        assert (target_dir / "environment.json").read_text() == '{"uuid":"new-env"}'


# ===========================================================================
# 0b. startup env bootstrap
# ===========================================================================


class TestBootstrapRuntimeEnvVars:
    def test_bootstrap_loads_envs_from_migrated_credentials(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CYBERWAVE_EDGE_CONFIG_DIR", raising=False)
        monkeypatch.delenv("CYBERWAVE_BASE_URL", raising=False)
        monkeypatch.delenv("CYBERWAVE_MQTT_HOST", raising=False)

        legacy_dir = tmp_path / "legacy"
        target_dir = tmp_path / "new"
        legacy_dir.mkdir()
        (legacy_dir / "credentials.json").write_text(
            json.dumps(
                {
                    "envs": {
                        "CYBERWAVE_BASE_URL": " https://api.example.com ",
                        "CYBERWAVE_MQTT_HOST": " mqtt.example.com ",
                    }
                }
            )
        )
        monkeypatch.setattr(startup, "_LEGACY_SYSTEM_CONFIG_DIR", legacy_dir)
        monkeypatch.setattr(startup, "_resolve_config_dir", lambda: target_dir)

        startup._bootstrap_runtime_env_vars()

        assert os.getenv("CYBERWAVE_BASE_URL") == "https://api.example.com"
        assert os.getenv("CYBERWAVE_MQTT_HOST") == "mqtt.example.com"
        assert (target_dir / "credentials.json").exists()

    def test_bootstrap_does_not_overwrite_existing_process_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CYBERWAVE_EDGE_CONFIG_DIR", raising=False)
        monkeypatch.setenv("CYBERWAVE_BASE_URL", "https://already-set.example.com")

        legacy_dir = tmp_path / "legacy"
        target_dir = tmp_path / "new"
        legacy_dir.mkdir()
        (legacy_dir / "credentials.json").write_text(
            json.dumps({"envs": {"CYBERWAVE_BASE_URL": "https://from-file.example.com"}})
        )
        monkeypatch.setattr(startup, "_LEGACY_SYSTEM_CONFIG_DIR", legacy_dir)
        monkeypatch.setattr(startup, "_resolve_config_dir", lambda: target_dir)

        startup._bootstrap_runtime_env_vars()

        assert os.getenv("CYBERWAVE_BASE_URL") == "https://already-set.example.com"

    def test_bootstrap_ignores_blank_and_non_string_env_values(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CYBERWAVE_EDGE_CONFIG_DIR", raising=False)
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.delenv("CW_BOOTSTRAP_VALID", raising=False)
        monkeypatch.delenv("CW_BOOTSTRAP_VALID_2", raising=False)
        monkeypatch.delenv("CW_BOOTSTRAP_BLANK", raising=False)
        monkeypatch.delenv("CW_BOOTSTRAP_NON_STRING", raising=False)
        monkeypatch.delenv("CW_BOOTSTRAP_LIST", raising=False)

        legacy_dir = tmp_path / "legacy"
        target_dir = tmp_path / "new"
        legacy_dir.mkdir()
        (legacy_dir / "credentials.json").write_text(
            json.dumps(
                {
                    "envs": {
                        "CW_BOOTSTRAP_VALID": "  value-one  ",
                        "CW_BOOTSTRAP_VALID_2": "\tvalue-two\n",
                        "CW_BOOTSTRAP_BLANK": "   ",
                        "CW_BOOTSTRAP_NON_STRING": 123,
                        "CW_BOOTSTRAP_LIST": ["x"],
                    }
                }
            )
        )
        monkeypatch.setattr(startup, "_LEGACY_SYSTEM_CONFIG_DIR", legacy_dir)
        monkeypatch.setattr(startup, "_resolve_config_dir", lambda: target_dir)

        startup._bootstrap_runtime_env_vars()

        assert os.getenv("CW_BOOTSTRAP_VALID") == "value-one"
        assert os.getenv("CW_BOOTSTRAP_VALID_2") == "value-two"
        assert os.getenv("CW_BOOTSTRAP_BLANK") is None
        assert os.getenv("CW_BOOTSTRAP_NON_STRING") is None
        assert os.getenv("CW_BOOTSTRAP_LIST") is None


# ===========================================================================
# 1. load_token
# ===========================================================================


class TestLoadToken:
    def test_returns_none_when_credentials_file_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", tmp_path / "credentials.json")
        assert startup.load_token() is None

    def test_returns_none_when_token_key_missing(self, tmp_path, monkeypatch):
        creds = tmp_path / "credentials.json"
        creds.write_text(json.dumps({"envs": {}}))
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", creds)
        assert startup.load_token() is None

    def test_returns_none_when_token_is_empty_string(self, tmp_path, monkeypatch):
        creds = tmp_path / "credentials.json"
        creds.write_text(json.dumps({"token": ""}))
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", creds)
        assert startup.load_token() is None

    def test_returns_none_when_json_is_malformed(self, tmp_path, monkeypatch):
        creds = tmp_path / "credentials.json"
        creds.write_text("{ not valid json }")
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", creds)
        assert startup.load_token() is None

    def test_returns_token_when_file_is_valid(self, tmp_path, monkeypatch):
        creds = tmp_path / "credentials.json"
        creds.write_text(json.dumps({"token": "my-secret-token"}))
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", creds)
        assert startup.load_token() == "my-secret-token"

    def test_logs_loaded_token_only_once_for_unchanged_credentials(
        self, tmp_path, monkeypatch, caplog
    ):
        creds = tmp_path / "credentials.json"
        creds.write_text(json.dumps({"token": "my-secret-token"}))
        monkeypatch.setattr(startup, "CREDENTIALS_FILE", creds)
        startup._reset_logged_token_signature()

        with caplog.at_level(logging.INFO, logger=startup.logger.name):
            assert startup.load_token() == "my-secret-token"
            assert startup.load_token() == "my-secret-token"

        loaded_messages = [
            record.message for record in caplog.records if "Loaded token from" in record.message
        ]
        assert len(loaded_messages) == 1


# ===========================================================================
# 2. write_or_update_twin_json_file — deep-merge preserves existing keys
# ===========================================================================


class TestWriteOrUpdateTwinJsonFileDeepMerge:
    _TWIN_UUID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    def test_creates_new_file_with_asset_embedded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        result = startup.write_or_update_twin_json_file(
            self._TWIN_UUID, {"name": "robot"}, {"model": "x1"}
        )
        assert result is True
        written = json.loads((tmp_path / f"{self._TWIN_UUID}.json").read_text())
        assert written["name"] == "robot"
        assert written["asset"] == {"model": "x1"}

    def test_merge_preserves_locally_set_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        existing = {
            "name": "old-name",
            "metadata": {"sensors_devices": {"camera": "/dev/video0"}, "extra": "keep-me"},
            "local_only_key": "do-not-lose",
        }
        (tmp_path / f"{self._TWIN_UUID}.json").write_text(json.dumps(existing))

        startup.write_or_update_twin_json_file(
            self._TWIN_UUID,
            {"name": "new-name", "metadata": {"sensors_devices": {"camera": "/dev/video1"}}},
            {},
        )
        written = json.loads((tmp_path / f"{self._TWIN_UUID}.json").read_text())

        # New value wins for updated keys
        assert written["name"] == "new-name"
        assert written["metadata"]["sensors_devices"]["camera"] == "/dev/video1"
        # Locally-set sibling key inside the nested dict is preserved
        assert written["metadata"]["extra"] == "keep-me"
        # Top-level local-only key is preserved
        assert written["local_only_key"] == "do-not-lose"

    def test_merge_overwrites_scalars_not_dicts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        existing = {"x": 1, "nested": {"a": 1, "b": 2}}
        (tmp_path / f"{self._TWIN_UUID}.json").write_text(json.dumps(existing))

        startup.write_or_update_twin_json_file(self._TWIN_UUID, {"x": 99, "nested": {"a": 42}}, {})
        written = json.loads((tmp_path / f"{self._TWIN_UUID}.json").read_text())
        assert written["x"] == 99
        assert written["nested"]["a"] == 42
        # "b" existed in existing nested dict and was not in override → preserved
        assert written["nested"]["b"] == 2

    def test_existing_file_is_updated_in_place(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        twin_json_path = tmp_path / f"{self._TWIN_UUID}.json"
        twin_json_path.write_text(json.dumps({"name": "stable", "asset": {"model": "x1"}}))
        original_inode = twin_json_path.stat().st_ino

        startup.write_or_update_twin_json_file(
            self._TWIN_UUID,
            {"name": "updated"},
            {"model": "x2"},
        )

        written = json.loads(twin_json_path.read_text())
        assert twin_json_path.stat().st_ino == original_inode
        assert written == {"name": "updated", "asset": {"model": "x2"}}

    def test_existing_file_remains_valid_when_atomic_write_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        twin_json_path = tmp_path / f"{self._TWIN_UUID}.json"
        twin_json_path.write_text(json.dumps({"name": "stable", "asset": {"model": "x1"}}))

        original_json_dumps = startup.json.dumps

        def _failing_dumps(data, **kwargs):  # type: ignore[no-untyped-def]
            raise TypeError("boom")

        monkeypatch.setattr(startup.json, "dumps", _failing_dumps)

        try:
            startup.write_or_update_twin_json_file(
                self._TWIN_UUID,
                {"name": "new-name"},
                {"model": "x2"},
            )
        except TypeError as exc:
            assert str(exc) == "boom"
        else:
            raise AssertionError("Expected write_or_update_twin_json_file to propagate TypeError")
        finally:
            monkeypatch.setattr(startup.json, "dumps", original_json_dumps)

        written = json.loads(twin_json_path.read_text())
        assert written == {"name": "stable", "asset": {"model": "x1"}}
        assert not list(tmp_path.glob(f"{self._TWIN_UUID}.*.tmp"))


# ===========================================================================
# 3. write_or_update_twin_json_file — directory at path replaced by file
# ===========================================================================


class TestWriteOrUpdateTwinJsonFileDirectoryReplacement:
    _TWIN_UUID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    def test_directory_at_path_is_removed_and_file_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        # Simulate the Docker bind-mount artifact: a directory where the file should be
        dir_path = tmp_path / f"{self._TWIN_UUID}.json"
        dir_path.mkdir()
        (dir_path / "dummy").write_text("content")

        assert dir_path.is_dir()
        result = startup.write_or_update_twin_json_file(
            self._TWIN_UUID, {"name": "robot"}, {"model": "x1"}
        )
        assert result is True
        assert dir_path.is_file(), "Directory should have been replaced by a regular file"
        written = json.loads(dir_path.read_text())
        assert written["name"] == "robot"


# ===========================================================================
# 4. load_environment_uuid — retry logic
# ===========================================================================


class TestLoadEnvironmentUuid:
    _VALID_UUID = "12345678-1234-5678-1234-567812345678"

    def test_returns_none_when_file_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", tmp_path / "environment.json")
        assert startup.load_environment_uuid() is None

    def test_returns_none_for_invalid_uuid_format(self, tmp_path, monkeypatch):
        env_file = tmp_path / "environment.json"
        env_file.write_text(json.dumps({"uuid": "not-a-uuid"}))
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", env_file)
        assert startup.load_environment_uuid() is None

    def test_normalises_uuid_to_lowercase_canonical_form(self, tmp_path, monkeypatch):
        env_file = tmp_path / "environment.json"
        env_file.write_text(json.dumps({"uuid": self._VALID_UUID.upper()}))
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", env_file)
        result = startup.load_environment_uuid()
        assert result == self._VALID_UUID.lower()

    def test_returns_none_when_uuid_field_missing_no_retries(self, tmp_path, monkeypatch):
        env_file = tmp_path / "environment.json"
        env_file.write_text(json.dumps({}))
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", env_file)
        monkeypatch.setattr(startup.time, "sleep", lambda _: None)
        assert startup.load_environment_uuid(retries=0) is None

    def test_retries_until_file_becomes_valid(self, tmp_path, monkeypatch):
        """Simulate a race: file is written mid-boot and becomes valid on attempt 2."""
        env_file = tmp_path / "environment.json"
        # Start with an empty uuid field
        env_file.write_text(json.dumps({"uuid": ""}))
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", env_file)

        call_count = 0

        def _side_effect_sleep(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            # On first sleep (between attempt 1 and 2), write the valid UUID
            if call_count == 1:
                env_file.write_text(json.dumps({"uuid": self._VALID_UUID}))

        monkeypatch.setattr(startup.time, "sleep", _side_effect_sleep)

        result = startup.load_environment_uuid(retries=2, retry_delay_seconds=0.0)
        assert result == self._VALID_UUID
        assert call_count == 1, "sleep should have been called exactly once (one retry)"

    def test_no_sleep_when_valid_on_first_attempt(self, tmp_path, monkeypatch):
        env_file = tmp_path / "environment.json"
        env_file.write_text(json.dumps({"uuid": self._VALID_UUID}))
        monkeypatch.setattr(startup, "ENVIRONMENT_FILE", env_file)

        sleep_calls: list[float] = []
        monkeypatch.setattr(startup.time, "sleep", lambda s: sleep_calls.append(s))

        result = startup.load_environment_uuid(retries=3, retry_delay_seconds=0.1)
        assert result == self._VALID_UUID
        assert sleep_calls == [], "sleep must not be called when the file is valid immediately"


# ===========================================================================
# 5. _remove_cached_twin_json_files — protected files never deleted
# ===========================================================================


class TestRemoveCachedTwinJsonFiles:
    def test_protected_files_are_never_deleted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        for protected_name in startup._PROTECTED_CONFIG_JSON_FILES:
            (tmp_path / protected_name).write_text("{}")

        removed = startup._remove_cached_twin_json_files()

        assert removed == [], "No files should be removed when only protected files exist"
        for protected_name in startup._PROTECTED_CONFIG_JSON_FILES:
            assert (tmp_path / protected_name).exists(), f"{protected_name} must not be deleted"

    def test_uuid_named_files_are_removed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        twin_uuid = str(_uuid_module.uuid4())
        twin_file = tmp_path / f"{twin_uuid}.json"
        twin_file.write_text("{}")

        removed = startup._remove_cached_twin_json_files()

        assert f"{twin_uuid}.json" in removed
        assert not twin_file.exists()

    def test_non_uuid_named_files_are_not_deleted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        custom_file = tmp_path / "my-custom-config.json"
        custom_file.write_text("{}")

        removed = startup._remove_cached_twin_json_files()

        assert removed == []
        assert custom_file.exists(), "Non-UUID named files must not be deleted"

    def test_returns_only_removed_filenames(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        uuid1 = str(_uuid_module.uuid4())
        uuid2 = str(_uuid_module.uuid4())
        (tmp_path / f"{uuid1}.json").write_text("{}")
        (tmp_path / f"{uuid2}.json").write_text("{}")
        (tmp_path / "credentials.json").write_text("{}")  # protected — must survive

        removed = startup._remove_cached_twin_json_files()

        assert set(removed) == {f"{uuid1}.json", f"{uuid2}.json"}
        assert (tmp_path / "credentials.json").exists()


# ===========================================================================
# 6. reconcile_driver_restart_failures — flapping driver detection
# ===========================================================================


class TestReconcileDriverRestartFailures:
    def test_stops_and_alerts_when_restart_threshold_is_exceeded(self, monkeypatch):
        container_name = "cyberwave-driver-1234abcd"
        twin_uuid = "11111111-1111-1111-1111-111111111111"
        restart_counts = iter([0, 1, 2, 3, 4, 5])
        timestamps = itertools.count(start=0, step=10)

        startup._CONTAINER_LAST_RESTART_COUNT.clear()
        startup._CONTAINER_RESTART_HISTORY.clear()
        startup._CONTAINER_TWIN_MAP.clear()
        monkeypatch.setattr(startup, "DRIVER_RESTART_LOOP_THRESHOLD", 4)
        monkeypatch.setattr(startup, "DRIVER_RESTART_LOOP_WINDOW_SECONDS", 60.0)
        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [container_name],
        )
        monkeypatch.setattr(startup.time, "time", lambda: float(next(timestamps)))
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda _name: {
                "RestartCount": next(restart_counts),
                "State": {"Status": "restarting", "Error": "camera unavailable"},
                "Config": {"Env": [f"CYBERWAVE_TWIN_UUID={twin_uuid}"]},
            },
        )

        stopped: list[str] = []
        alerts: list[tuple] = []
        monkeypatch.setattr(
            startup,
            "_stop_driver_container",
            lambda name: stopped.append(name) or True,
        )
        monkeypatch.setattr(
            startup,
            "_send_alert_for_twin",
            lambda *args, **kwargs: alerts.append((args, kwargs)),
        )

        for _ in range(6):
            startup.reconcile_driver_restart_failures()

        assert stopped == [container_name]
        assert len(alerts) == 1
        assert alerts[0][0][0] == twin_uuid
        assert alerts[0][0][3] == "driver_restart_loop"

    def test_does_not_alert_for_sparse_restarts_outside_window(self, monkeypatch):
        container_name = "cyberwave-driver-1234abcd"
        twin_uuid = "22222222-2222-2222-2222-222222222222"
        restart_counts = iter([0, 1, 2, 3, 4, 5])
        timestamps = itertools.count(start=0, step=70)

        startup._CONTAINER_LAST_RESTART_COUNT.clear()
        startup._CONTAINER_RESTART_HISTORY.clear()
        startup._CONTAINER_TWIN_MAP.clear()
        monkeypatch.setattr(startup, "DRIVER_RESTART_LOOP_THRESHOLD", 4)
        monkeypatch.setattr(startup, "DRIVER_RESTART_LOOP_WINDOW_SECONDS", 60.0)
        monkeypatch.setattr(
            startup,
            "_list_driver_containers",
            lambda include_stopped: [container_name],
        )
        monkeypatch.setattr(startup.time, "time", lambda: float(next(timestamps)))
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda _name: {
                "RestartCount": next(restart_counts),
                "State": {"Status": "restarting", "Error": ""},
                "Config": {"Env": [f"CYBERWAVE_TWIN_UUID={twin_uuid}"]},
            },
        )

        stopped: list[str] = []
        alerts: list[tuple] = []
        monkeypatch.setattr(
            startup,
            "_stop_driver_container",
            lambda name: stopped.append(name) or True,
        )
        monkeypatch.setattr(
            startup,
            "_send_alert_for_twin",
            lambda *args, **kwargs: alerts.append((args, kwargs)),
        )

        for _ in range(6):
            startup.reconcile_driver_restart_failures()

        assert stopped == []
        assert alerts == []


# ===========================================================================
# 7. _run_docker_image pull behavior with local fallback
# ===========================================================================


class TestRunDockerImagePullFallback:
    _TWIN_UUID = "99999999-9999-9999-9999-999999999999"

    def _patch_common(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(startup, "load_credentials_envs", lambda: {})
        monkeypatch.setattr(startup, "get_runtime_env_var", lambda *args, **kwargs: None)
        monkeypatch.setattr(startup.time, "sleep", lambda _: None)
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda _name: {"State": {"Status": "running", "Error": ""}},
        )
        monkeypatch.setattr(startup, "_stream_container_logs", lambda *args, **kwargs: None)

    @staticmethod
    def _extract_env_map(docker_run_cmd: list[str]) -> dict[str, str]:
        env_map: dict[str, str] = {}
        for idx, arg in enumerate(docker_run_cmd):
            if arg != "-e" or idx + 1 >= len(docker_run_cmd):
                continue
            key, sep, value = docker_run_cmd[idx + 1].partition("=")
            if sep:
                env_map[key] = value
        return env_map

    def test_uses_local_image_when_pull_fails(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        pull_calls: list[str] = []

        def _fake_pull(*args, **kwargs):  # type: ignore[no-untyped-def]
            pull_calls.append("pull")
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["docker", "pull", "cyberwave-step14-driver:latest"],
                stderr="pull access denied",
            )

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            if cmd[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup, "_pull_docker_image_with_progress", _fake_pull)
        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        assert pull_calls == ["pull"]
        assert any(cmd[:3] == ["docker", "image", "inspect"] for cmd in commands)
        assert any(cmd[:2] == ["docker", "run"] for cmd in commands)

    def test_fails_when_pull_fails_and_image_missing_locally(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        pull_calls: list[str] = []

        def _fake_pull(*args, **kwargs):  # type: ignore[no-untyped-def]
            pull_calls.append("pull")
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["docker", "pull", "cyberwave-step14-driver:latest"],
                stderr="pull access denied",
            )

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            if cmd[:3] == ["docker", "image", "inspect"]:
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=cmd,
                    stderr="No such image",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup, "_pull_docker_image_with_progress", _fake_pull)
        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is False
        assert pull_calls == ["pull"]
        assert not any(cmd[:2] == ["docker", "run"] for cmd in commands)

    def test_forwards_process_cyberwave_env_vars_to_driver_container(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setenv("CYBERWAVE_GO2_IP_ADDR", " 192.168.0.10 ")
        monkeypatch.setenv("CYBERWAVE_EMPTY", "   ")
        monkeypatch.setenv("GO2_IP_ADDR", "192.168.0.10")

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(
            startup, "_pull_docker_image_with_progress", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        env_map = self._extract_env_map(docker_run_cmd)
        assert env_map["CYBERWAVE_GO2_IP_ADDR"] == "192.168.0.10"
        assert "CYBERWAVE_EMPTY" not in env_map
        assert "GO2_IP_ADDR" not in env_map

    def test_process_env_does_not_override_credentials_env_values(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setattr(
            startup,
            "load_credentials_envs",
            lambda: {
                "CYBERWAVE_GO2_IP_ADDR": "10.0.0.2",
                "CYBERWAVE_REGION": "eu-west-1",
            },
        )
        monkeypatch.setenv("CYBERWAVE_GO2_IP_ADDR", "192.168.0.10")
        monkeypatch.setenv("CYBERWAVE_REGION", "us-east-1")
        monkeypatch.setenv("CYBERWAVE_EXTRA", "enabled")

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(
            startup, "_pull_docker_image_with_progress", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        env_map = self._extract_env_map(docker_run_cmd)
        assert env_map["CYBERWAVE_GO2_IP_ADDR"] == "10.0.0.2"
        assert env_map["CYBERWAVE_REGION"] == "eu-west-1"
        assert env_map["CYBERWAVE_EXTRA"] == "enabled"

    def test_runs_macos_bridge_command_before_docker_run(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: False)

        def _runtime_env(name, default=None):  # type: ignore[no-untyped-def]
            if name == "CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND":
                return (
                    "/bin/echo bridge {host_device} {container_device} {twin_uuid} {container_name}"
                )
            return default

        monkeypatch.setattr(startup, "get_runtime_env_var", _runtime_env)

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            ["--device", "/dev/video0:/dev/video0"],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        bridge_cmd = next(cmd for cmd in commands if cmd and cmd[0] == "/bin/echo")
        assert "bridge" in bridge_cmd
        assert "/dev/video0" in bridge_cmd
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        assert "--network" not in docker_run_cmd
        assert "host" not in docker_run_cmd
        assert "--add-host" in docker_run_cmd
        assert "host.docker.internal:host-gateway" in docker_run_cmd
        env_map = self._extract_env_map(docker_run_cmd)
        assert env_map["CYBERWAVE_EDGE_HOST_PLATFORM"] == "darwin"

    def test_macos_driver_container_rewrites_localhost_base_url(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: False)

        def _runtime_env(name, default=None):  # type: ignore[no-untyped-def]
            if name == "CYBERWAVE_BASE_URL":
                return "http://localhost:8000"
            return default

        monkeypatch.setattr(startup, "get_runtime_env_var", _runtime_env)

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        env_map = self._extract_env_map(docker_run_cmd)
        assert env_map["CYBERWAVE_BASE_URL"] == "http://host.docker.internal:8000"

    def test_macos_driver_container_rewrites_localhost_mqtt_host(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: False)

        def _runtime_env(name, default=None):  # type: ignore[no-untyped-def]
            if name == "CYBERWAVE_MQTT_HOST":
                return "localhost"
            return default

        monkeypatch.setattr(startup, "get_runtime_env_var", _runtime_env)

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        env_map = self._extract_env_map(docker_run_cmd)
        assert env_map["CYBERWAVE_MQTT_HOST"] == "host.docker.internal"

    def test_macos_bridge_command_failure_aborts_driver_start(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: False)

        def _runtime_env(name, default=None):  # type: ignore[no-untyped-def]
            if name == "CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND":
                return "/bin/false {host_device}"
            return default

        monkeypatch.setattr(startup, "get_runtime_env_var", _runtime_env)

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            if cmd and cmd[0] == "/bin/false":
                raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="failed")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            ["--device", "/dev/ttyACM0:/dev/ttyACM0"],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is False
        assert any(cmd and cmd[0] == "/bin/false" for cmd in commands)
        assert not any(cmd[:2] == ["docker", "run"] for cmd in commands)

    def test_macos_bridge_resolved_source_sets_video_env_and_strips_video_device(
        self, tmp_path, monkeypatch
    ):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: False)

        def _runtime_env(name, default=None):  # type: ignore[no-untyped-def]
            if name == "CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND":
                return "/bin/echo resolved_device=rtsp://host.docker.internal:8554/cam0"
            if name == "CYBERWAVE_MACOS_STRIP_VIDEO_DEVICE_PARAMS":
                return "true"
            return default

        monkeypatch.setattr(startup, "get_runtime_env_var", _runtime_env)

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            if cmd and cmd[0] == "/bin/echo":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="resolved_device=rtsp://host.docker.internal:8554/cam0\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            ["--device", "/dev/video0:/dev/video0"],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        assert "--device" not in docker_run_cmd
        assert "--device=/dev/video0:/dev/video0" not in docker_run_cmd
        env_map = self._extract_env_map(docker_run_cmd)
        assert env_map["CYBERWAVE_METADATA_VIDEO_DEVICE"] == "rtsp://host.docker.internal:8554/cam0"
        assert json.loads(env_map["CYBERWAVE_EDGE_VIDEO_DEVICE_MAP"]) == {
            "/dev/video0": "rtsp://host.docker.internal:8554/cam0"
        }

    def test_macos_does_not_override_explicit_video_device_env_in_params(
        self, tmp_path, monkeypatch
    ):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: False)

        def _runtime_env(name, default=None):  # type: ignore[no-untyped-def]
            if name == "CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND":
                return "/bin/echo resolved_device=rtsp://host.docker.internal:8554/cam0"
            return default

        monkeypatch.setattr(startup, "get_runtime_env_var", _runtime_env)

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            if cmd and cmd[0] == "/bin/echo":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="resolved_device=rtsp://host.docker.internal:8554/cam0\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [
                "--device",
                "/dev/video0:/dev/video0",
                "-e",
                "CYBERWAVE_METADATA_VIDEO_DEVICE=/dev/video7",
            ],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        env_map = self._extract_env_map(docker_run_cmd)
        assert env_map["CYBERWAVE_METADATA_VIDEO_DEVICE"] == "/dev/video7"

    def test_macos_candidate_mapping_sets_video_env_without_device_params(
        self, tmp_path, monkeypatch
    ):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: False)

        def _runtime_env(name, default=None):  # type: ignore[no-untyped-def]
            if name == "CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND":
                return "/bin/echo resolved_device=rtsp://host.docker.internal:8554/cam-main"
            return default

        monkeypatch.setattr(startup, "get_runtime_env_var", _runtime_env)

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            if cmd and cmd[0] == "/bin/echo":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="resolved_device=rtsp://host.docker.internal:8554/cam-main\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
            macos_bridge_device_candidates=["/dev/video0"],
        )

        assert success is True
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        env_map = self._extract_env_map(docker_run_cmd)
        assert (
            env_map["CYBERWAVE_METADATA_VIDEO_DEVICE"]
            == "rtsp://host.docker.internal:8554/cam-main"
        )
        assert json.loads(env_map["CYBERWAVE_EDGE_VIDEO_DEVICE_MAP"]) == {
            "/dev/video0": "rtsp://host.docker.internal:8554/cam-main"
        }

    def test_normalize_macos_bridge_candidates_supports_custom_host_to_container_mapping(self):
        normalized = startup._normalize_macos_bridge_candidates(
            ["/dev/video2:/dev/video0", " /dev/video5 ", "", "   "]
        )

        assert normalized == [
            ("/dev/video2", "/dev/video0"),
            ("/dev/video5", "/dev/video5"),
        ]

    def test_usbip_active_skips_bridge_for_video_devices(self, tmp_path, monkeypatch):
        """When USB/IP is active, video device mappings bypass the bridge command."""
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: True)

        bridge_calls: list[str] = []

        def _runtime_env(name, default=None):  # type: ignore[no-untyped-def]
            if name == "CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND":
                return "/bin/echo resolved_device=rtsp://host.docker.internal:8554/cam0"
            return default

        monkeypatch.setattr(startup, "get_runtime_env_var", _runtime_env)

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            if cmd and cmd[0] == "/bin/echo":
                bridge_calls.append("bridge")
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="resolved_device=rtsp://host.docker.internal:8554/cam0\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            ["--device", "/dev/video0:/dev/video0"],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        assert bridge_calls == [], (
            "bridge command must NOT run for video devices when USB/IP is active"
        )
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        assert "--pid=host" in docker_run_cmd
        env_map = self._extract_env_map(docker_run_cmd)
        assert env_map.get("CYBERWAVE_USBIP_ENABLED") == "1"
        assert env_map.get("CYBERWAVE_METADATA_VIDEO_DEVICE") == "/dev/video0"

    def test_usbip_active_preserves_device_params_for_video(self, tmp_path, monkeypatch):
        """When USB/IP handles video, --device /dev/video* is NOT stripped."""
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: True)

        monkeypatch.setattr(startup, "get_runtime_env_var", lambda *a, **kw: None)

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            ["--device", "/dev/video0:/dev/video0"],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        assert "--device" in docker_run_cmd
        device_idx = docker_run_cmd.index("--device")
        assert docker_run_cmd[device_idx + 1] == "/dev/video0:/dev/video0"

    def test_usbip_active_still_bridges_non_video_devices(self, tmp_path, monkeypatch):
        """USB/IP only skips video devices; serial devices still use the bridge command."""
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: True)

        def _runtime_env(name, default=None):  # type: ignore[no-untyped-def]
            if name == "CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND":
                return "/bin/echo bridged {host_device}"
            return default

        monkeypatch.setattr(startup, "get_runtime_env_var", _runtime_env)

        bridge_calls: list[list[str]] = []

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            if cmd and cmd[0] == "/bin/echo":
                bridge_calls.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [
                "--device",
                "/dev/ttyACM0:/dev/ttyACM0",
                "--device",
                "/dev/video0:/dev/video0",
            ],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        assert len(bridge_calls) == 1, "only non-video device should trigger bridge"
        assert "/dev/ttyACM0" in bridge_calls[0]


class TestDriverStartingAlertLifecycle:
    """Regression: the ``driver_starting`` alert must be resolved on the
    ``skip_pull=True`` path used by the remote ``restart edge-core`` command.
    Previously the alert was only resolved at "pull complete", so skip-pull
    left it active and the UI showed "Driver starting" forever after a
    restart.
    """

    def test_skip_pull_resolves_alert_when_container_runs(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        spy = MagicMock(name="DriverStartingAlertContext")
        monkeypatch.setattr(startup, "DriverStartingAlertContext", lambda **kw: spy)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.shutil, "which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(startup, "load_credentials_envs", lambda: {})
        monkeypatch.setattr(startup, "get_runtime_env_var", lambda *a, **k: None)
        monkeypatch.setattr(startup.time, "sleep", lambda _: None)
        monkeypatch.setattr(startup, "_stream_container_logs", lambda *a, **k: None)
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda _n: {"State": {"Status": "running", "Error": ""}},
        )
        monkeypatch.setattr(
            startup.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr=""),
        )

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [],
            twin_uuid="11111111-2222-3333-4444-555555555555",
            token="test-token",
            skip_pull=True,
        )

        assert success is True
        spy.resolve.assert_called()
        spy.mark_failed_and_resolve.assert_not_called()

    def test_fetch_and_run_clears_stale_alerts_before_creating_new_ones(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Orphan ``driver_starting`` alerts from a crashed boot must not
        survive the next ``fetch_and_run_twin_drivers`` call."""
        from tests.test_multi_camera_orchestration import (
            FakeAsset,
            FakeTwin,
            TWIN_A,
            _stub_client,
        )

        fingerprint = "edge-fp"
        cam_asset = "asset-a-uuid"
        twin = FakeTwin(
            uuid=TWIN_A,
            name="Camera",
            asset_uuid=cam_asset,
            metadata={
                "edge_fingerprint": fingerprint,
                "drivers": {"default": {"docker_image": "cyberwaveos/camera-driver"}},
            },
        )
        fake_client = _stub_client([twin], {cam_asset: FakeAsset(metadata={})})

        call_order: list[str] = []

        class _FakeAlertCtx:
            @staticmethod
            def resolve_active_for_twin(twin_uuid: str) -> int:
                call_order.append(f"resolve:{twin_uuid}")
                return 1

            def __init__(
                self,
                *,
                twin_uuid: str,
                image: str,
                service_name: str | None = None,
            ) -> None:
                _ = service_name
                call_order.append(f"create:{twin_uuid}")

            def create(self) -> None:
                pass

            def update_metadata(self, metadata_patch: dict, *, force: bool = False) -> None:
                pass

            def mark_failed_and_resolve(
                self, description: str, *, phase: str = "pull_failed"
            ) -> None:
                pass

        monkeypatch.setattr(startup, "Cyberwave", lambda base_url, api_key: fake_client)
        monkeypatch.setattr(startup, "DriverStartingAlertContext", _FakeAlertCtx)
        monkeypatch.setattr(
            startup, "_check_and_alert_sensors_devices", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(startup, "write_or_update_twin_json_file", lambda *a, **kw: True)
        monkeypatch.setattr(
            startup,
            "_pull_driver_images_parallel",
            lambda images, **kw: {img: True for img in images},
        )
        monkeypatch.setattr(
            startup,
            "_run_docker_image",
            lambda *a, **kw: True,
        )

        startup.fetch_and_run_twin_drivers("test-token", "env-uuid", fingerprint)

        assert call_order.index(f"resolve:{TWIN_A}") < call_order.index(f"create:{TWIN_A}")


class TestDriverSelection:
    def test_prefers_platform_specific_driver_by_machine(self, monkeypatch):
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: False)

        image, params, prefer_gpu, gpu_spec = startup._get_best_driver_image_and_params(
            {
                "default": {"docker_image": "cyberwave/default-driver:latest"},
                "macos": {"docker_image": "cyberwave/macos-driver:latest"},
                "darwin-arm64": {
                    "docker_image": "cyberwave/macos-arm-driver:latest",
                    "params": ["--log-level", "debug"],
                },
            }
        )

        assert image == "cyberwave/macos-arm-driver:latest"
        assert params == ["--log-level", "debug"]
        assert prefer_gpu is False
        assert gpu_spec == "all"

    def test_falls_back_to_default_when_no_platform_driver_matches(self, monkeypatch):
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: False)

        image, params, prefer_gpu, gpu_spec = startup._get_best_driver_image_and_params(
            {
                "default": {"docker_image": "cyberwave/default-driver:latest"},
                "macos": {"docker_image": "cyberwave/macos-driver:latest"},
            }
        )

        assert image == "cyberwave/default-driver:latest"
        assert params == []
        assert prefer_gpu is False
        assert gpu_spec == "all"

    def test_prefers_jetson_platform_key_on_jetson(self, monkeypatch):
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "aarch64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: True)

        image, params, prefer_gpu, gpu_spec = startup._get_best_driver_image_and_params(
            {
                "default": {"docker_image": "cyberwave/driver:humble"},
                "linux-aarch64": {"docker_image": "cyberwave/driver:humble"},
                "linux-aarch64-jetson": {
                    "docker_image": "cyberwave/driver:jetson-humble",
                    "params": ["--jetson"],
                    "prefer_gpu": True,
                },
            }
        )

        assert image == "cyberwave/driver:jetson-humble"
        assert params == ["--jetson"]
        assert prefer_gpu is True
        assert gpu_spec == "all"

    def test_prefer_gpu_returned_from_driver_config(self, monkeypatch):
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: False)

        image, params, prefer_gpu, gpu_spec = startup._get_best_driver_image_and_params(
            {
                "default": {
                    "docker_image": "cyberwave/driver:latest",
                    "prefer_gpu": True,
                },
            }
        )

        assert image == "cyberwave/driver:latest"
        assert prefer_gpu is True
        assert gpu_spec == "all"

    def test_gpu_spec_from_driver_config(self, monkeypatch):
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: False)

        _, _, prefer_gpu, gpu_spec = startup._get_best_driver_image_and_params(
            {
                "default": {
                    "docker_image": "cyberwave/driver:latest",
                    "prefer_gpu": True,
                    "gpu": 1,
                },
            }
        )

        assert prefer_gpu is True
        assert gpu_spec == "1"

    def test_gpu_spec_device_selector(self, monkeypatch):
        monkeypatch.setattr(driver_selection.platform, "system", lambda: "Linux")
        monkeypatch.setattr(driver_selection.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(driver_selection, "_jetson_detected", None)
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: False)

        _, _, prefer_gpu, gpu_spec = startup._get_best_driver_image_and_params(
            {
                "default": {
                    "docker_image": "cyberwave/driver:latest",
                    "prefer_gpu": True,
                    "gpu": "device=0,2",
                },
            }
        )

        assert prefer_gpu is True
        assert gpu_spec == "device=0,2"


class TestPullDockerImageWithProgress:
    class _FakeProcess:
        def __init__(self, output: str, *, returncode: int = 0):
            self.stdout = io.StringIO(output)
            self._returncode = returncode
            self.killed = False

        def wait(self, timeout=None):  # type: ignore[no-untyped-def]
            return self._returncode

        def kill(self):
            self.killed = True

    class _FakeMQTT:
        def __init__(self):
            self.topic_prefix = "cw/"
            self.published: list[tuple[str, dict[str, object]]] = []

        def publish(self, topic, payload):  # type: ignore[no-untyped-def]
            self.published.append((topic, payload))

    class _FakeClient:
        def __init__(self):
            self.mqtt = TestPullDockerImageWithProgress._FakeMQTT()

    def test_streams_pull_progress_lines_to_mqtt(self, monkeypatch):
        from cyberwave_edge_core import driver_logs

        fake_client = self._FakeClient()
        monkeypatch.setattr(startup, "_get_shared_mqtt_client", lambda token: fake_client)
        monkeypatch.setattr(
            driver_logs.subprocess,
            "Popen",
            lambda *args, **kwargs: self._FakeProcess(
                "layer-1: Pulling fs layer\rlayer-1: Downloading 10%\r"
                "layer-1: Downloading 10%\nDigest: sha256:abc123\n"
            ),
        )

        startup._pull_docker_image_with_progress(
            "ghcr.io/cyberwave/driver:1.2.3",
            container_name="cyberwave-driver-abcd1234",
            twin_uuid="99999999-9999-9999-9999-999999999999",
            token="test-token",
            timeout=30,
        )

        published_messages = [payload["message"] for _, payload in fake_client.mqtt.published]
        assert published_messages == [
            "docker pull started for image ghcr.io/cyberwave/driver:1.2.3",
            "docker pull: layer-1: Pulling fs layer",
            "docker pull: layer-1: Downloading 10%",
            "docker pull: Digest: sha256:abc123",
            "docker pull completed for image ghcr.io/cyberwave/driver:1.2.3",
        ]
        assert all(
            topic == "cw/cyberwave/twin/99999999-9999-9999-9999-999999999999/driverlog"
            for topic, _ in fake_client.mqtt.published
        )
        assert all(
            payload["driver_image"] == "ghcr.io/cyberwave/driver:1.2.3"
            for _, payload in fake_client.mqtt.published
        )


class TestReconcileDriverLogStreams:
    class _AliveThread:
        @staticmethod
        def is_alive() -> bool:
            return True

    def test_skips_token_reload_when_log_thread_is_already_attached(self, monkeypatch):
        from cyberwave_edge_core import driver_logs

        monkeypatch.setattr(
            startup,
            "_list_running_driver_containers",
            lambda: ["cyberwave-driver-abcd1234"],
        )
        monkeypatch.setattr(
            driver_logs,
            "_CONTAINER_LOG_THREADS",
            {"cyberwave-driver-abcd1234": self._AliveThread()},
        )
        monkeypatch.setattr(
            startup,
            "_CONTAINER_TWIN_MAP",
            {"cyberwave-driver-abcd1234": "99999999-9999-9999-9999-999999999999"},
        )

        load_calls: list[str] = []
        stream_calls: list[str] = []
        monkeypatch.setattr(startup, "load_token", lambda: load_calls.append("load") or "token")
        monkeypatch.setattr(
            driver_logs,
            "_stream_container_logs",
            lambda *args, **kwargs: stream_calls.append("stream"),
        )

        attached = startup.reconcile_driver_log_streams()

        assert attached == 1
        assert load_calls == []
        assert stream_calls == []


class TestBuildDriverLogPayload:
    def test_resolve_container_driver_image_prefers_config_image(self):
        assert (
            startup._resolve_container_driver_image(
                {
                    "Config": {"Image": "ghcr.io/cyberwave/driver:1.2.3"},
                    "Image": "sha256:abc123",
                }
            )
            == "ghcr.io/cyberwave/driver:1.2.3"
        )

    def test_includes_edge_core_and_sdk_versions(self, monkeypatch):
        from cyberwave_edge_core import driver_logs

        monkeypatch.setattr(startup, "EDGE_CORE_VERSION", "0.0.18-test")
        monkeypatch.setattr(startup, "CYBERWAVE_SDK_VERSION", "0.3.20-test")
        monkeypatch.setattr(driver_logs.time, "time", lambda: 1234.5)

        payload = startup._build_driver_log_payload(
            "2026-03-09 12:00:00 ERROR driver failed",
            "cyberwave-driver-test",
            driver_image="ghcr.io/cyberwave/driver:1.2.3",
        )

        assert payload == {
            "type": "driver_log",
            "message": "2026-03-09 12:00:00 ERROR driver failed",
            "level": "ERROR",
            "container_name": "cyberwave-driver-test",
            "source": "edge",
            "timestamp": 1234.5,
            "edge_core_version": "0.0.18-test",
            "sdk_version": "0.3.20-test",
            "driver_image": "ghcr.io/cyberwave/driver:1.2.3",
        }

    def test_omits_optional_fields_when_unavailable(self, monkeypatch):
        from cyberwave_edge_core import driver_logs

        monkeypatch.setattr(startup, "EDGE_CORE_VERSION", "0.0.18-test")
        monkeypatch.setattr(startup, "CYBERWAVE_SDK_VERSION", None)
        monkeypatch.setattr(driver_logs.time, "time", lambda: 99.0)

        payload = startup._build_driver_log_payload(
            "informational startup message",
            "cyberwave-driver-test",
        )

        assert payload["edge_core_version"] == "0.0.18-test"
        assert payload["level"] == "INFO"
        assert "sdk_version" not in payload
        assert "driver_image" not in payload


class TestBuildHostMetricsProvider:
    """The bootstrap publisher hands a closure to the SDK; verify what it returns.

    These tests pin the publish-side contract: which keys land on
    ``edge_health`` and in what shape.  The SDK's ``EdgeHealthCheck`` is
    covered separately under ``cyberwave-python/tests/test_edge_health.py``.
    """

    def test_returns_none_when_both_monitors_absent(self) -> None:
        assert startup._build_host_metrics_provider(None, None) is None

    def test_includes_snapshot_dict_and_consecutive_counter(self) -> None:
        class _FakeSnap:
            def to_publish_dict(self) -> dict[str, float]:
                return {"host_memory_percent": 64.2, "cpu_temp_c": 58.7}

        class _FakeMonitor:
            last_snapshot = _FakeSnap()
            consecutive_critical_count = 0

        provider = startup._build_host_metrics_provider(_FakeMonitor(), None)
        assert provider is not None
        out = provider()
        assert out["host_memory_percent"] == 64.2
        assert out["cpu_temp_c"] == 58.7
        assert out["consecutive_critical"] == 0

    def test_includes_watchdog_layers(self) -> None:
        class _FakeWatchdog:
            def active_layers(self) -> list[str]:
                return ["systemd", "hardware"]

        provider = startup._build_host_metrics_provider(None, _FakeWatchdog())
        assert provider is not None
        assert provider() == {"watchdog_layers": ["systemd", "hardware"]}

    def test_tolerates_monitor_without_snapshot(self) -> None:
        """``last_snapshot is None`` happens before the first ``check()`` call."""

        class _FakeMonitor:
            last_snapshot = None
            consecutive_critical_count = 0

        provider = startup._build_host_metrics_provider(_FakeMonitor(), None)
        assert provider is not None
        out = provider()
        # Counter is still emitted; snapshot keys are simply absent.
        assert out == {"consecutive_critical": 0}

    def test_isolates_failing_subreaders(self) -> None:
        """One broken subreader must not suppress the others."""

        class _BoomMonitor:
            @property
            def last_snapshot(self):  # type: ignore[no-untyped-def]
                raise RuntimeError("monitor exploded")

            @property
            def consecutive_critical_count(self):  # type: ignore[no-untyped-def]
                raise RuntimeError("monitor exploded again")

        class _OkWatchdog:
            def active_layers(self) -> list[str]:
                return ["systemd"]

        provider = startup._build_host_metrics_provider(_BoomMonitor(), _OkWatchdog())
        assert provider is not None
        # Provider must not raise; watchdog field is still present.
        out = provider()
        assert out == {"watchdog_layers": ["systemd"]}


class TestStartupHeartbeatOrdering:
    def test_starts_edge_heartbeat_before_fetching_drivers(self, monkeypatch):
        call_order: list[str] = []

        monkeypatch.setattr(startup, "load_token", lambda: "token-123")
        monkeypatch.setattr(startup, "validate_token", lambda token: True)
        monkeypatch.setattr(startup, "check_mqtt_connection", lambda token: True)
        monkeypatch.setattr(startup, "register_edge", lambda token: True)
        # ``_upload_host_facts_on_startup`` would otherwise try to call the
        # real backend.  Treat it as best-effort (already non-fatal in
        # production) and just record the call ordering.
        monkeypatch.setattr(
            startup,
            "_upload_host_facts_on_startup",
            lambda token: call_order.append("upload_host_facts") or True,
        )
        monkeypatch.setattr(
            startup,
            "load_environment_uuid",
            lambda retries=0, retry_delay_seconds=0.2: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        )
        monkeypatch.setattr(startup, "get_or_create_fingerprint", lambda: "edge-fingerprint")
        monkeypatch.setattr(
            startup,
            "_list_linked_twin_uuids_for_fingerprint",
            lambda token, env_uuid, fingerprint: (
                call_order.append("list_linked_twins") or ["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"]
            ),
        )
        monkeypatch.setattr(
            startup,
            "_start_bootstrap_edge_health_publisher",
            lambda token, twin_uuids, *, edge_id, resource_monitor=None, watchdog=None: (
                call_order.append("start_edge_heartbeat") or True
            ),
        )
        monkeypatch.setattr(
            startup,
            "fetch_and_run_twin_drivers",
            lambda token, env_uuid, fingerprint, **kw: call_order.append("fetch_drivers") or [],
        )
        # Skip step 7 (worker sync) so we're strictly exercising the
        # heartbeat-vs-drivers ordering contract this test is about.
        monkeypatch.setattr(
            startup,
            "_resolve_worker_sync_twin_uuids",
            lambda token, env_uuid, fingerprint: [],
        )

        result = startup.run_startup_checks()

        assert result is True
        assert call_order == [
            "upload_host_facts",
            "list_linked_twins",
            "start_edge_heartbeat",
            "fetch_drivers",
        ]


# ===========================================================================
# _fix_config_dir_ownership
# ===========================================================================


class TestFixConfigDirOwnership:
    def test_noop_on_macos(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup.os, "getuid", lambda: 0)
        monkeypatch.setenv("SUDO_UID", "1000")
        monkeypatch.setenv("SUDO_GID", "1000")
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        lchown_calls: list[tuple] = []
        monkeypatch.setattr(startup.os, "lchown", lambda p, u, g: lchown_calls.append((p, u, g)))
        startup._fix_config_dir_ownership()
        assert lchown_calls == []

    def test_noop_when_not_root(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
        monkeypatch.setattr(startup.os, "getuid", lambda: 1000)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        lchown_calls: list[tuple] = []
        monkeypatch.setattr(startup.os, "lchown", lambda p, u, g: lchown_calls.append((p, u, g)))
        startup._fix_config_dir_ownership()
        assert lchown_calls == []

    def test_noop_when_root_without_sudo_and_root_parent(self, tmp_path: Path, monkeypatch) -> None:
        """systemd-style: root without SUDO_UID and a root-owned parent → no-op."""
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
        monkeypatch.setattr(startup.os, "getuid", lambda: 0)
        monkeypatch.delenv("SUDO_UID", raising=False)
        monkeypatch.delenv("SUDO_GID", raising=False)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        # Simulate a root-owned parent by forcing the helper to return None.
        monkeypatch.setattr(startup, "resolve_config_owner_uid_gid", lambda: None)

        lchown_calls: list[tuple] = []
        monkeypatch.setattr(startup.os, "lchown", lambda p, u, g: lchown_calls.append((p, u, g)))
        startup._fix_config_dir_ownership()
        assert lchown_calls == []

    def test_chowns_under_systemd_via_config_parent_owner(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """systemd-style: root without SUDO_UID, but CONFIG_DIR.parent is user-owned."""
        target_uid, target_gid = 1000, 1000
        (tmp_path / "fingerprint.json").write_text("{}")

        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
        monkeypatch.setattr(startup.os, "getuid", lambda: 0)
        monkeypatch.delenv("SUDO_UID", raising=False)
        monkeypatch.delenv("SUDO_GID", raising=False)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        # Simulate a user-owned parent — helper returns that user's uid/gid.
        monkeypatch.setattr(
            startup,
            "resolve_config_owner_uid_gid",
            lambda: (target_uid, target_gid),
        )

        original_lstat = os.lstat

        def fake_lstat(path):
            result = original_lstat(path)
            return os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    0,
                    0,
                    result.st_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )

        monkeypatch.setattr(startup.os, "lstat", fake_lstat)

        lchown_calls: list[tuple] = []
        monkeypatch.setattr(
            startup.os,
            "lchown",
            lambda p, u, g: lchown_calls.append((p, u, g)),
        )

        startup._fix_config_dir_ownership()

        assert lchown_calls
        chowned_paths = {call[0] for call in lchown_calls}
        assert str(tmp_path) in chowned_paths
        assert str(tmp_path / "fingerprint.json") in chowned_paths
        for _, uid, gid in lchown_calls:
            assert uid == target_uid
            assert gid == target_gid

    def test_chowns_misowned_files_via_sudo(self, tmp_path: Path, monkeypatch) -> None:
        """sudo cyberwave edge start: root process with SUDO_UID/SUDO_GID."""
        target_uid, target_gid = 1000, 1000
        (tmp_path / "credentials.json").write_text("{}")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.json").write_text("{}")

        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
        monkeypatch.setattr(startup.os, "getuid", lambda: 0)
        monkeypatch.setenv("SUDO_UID", str(target_uid))
        monkeypatch.setenv("SUDO_GID", str(target_gid))
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

        original_lstat = os.lstat
        root_owned_paths = {
            str(tmp_path),
            str(tmp_path / "credentials.json"),
            str(tmp_path / "subdir"),
        }

        def fake_lstat(path):
            result = original_lstat(path)
            if str(path) in root_owned_paths:
                return os.stat_result(
                    (
                        result.st_mode,
                        result.st_ino,
                        result.st_dev,
                        result.st_nlink,
                        0,
                        0,
                        result.st_size,
                        result.st_atime,
                        result.st_mtime,
                        result.st_ctime,
                    )
                )
            return os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    target_uid,
                    target_gid,
                    result.st_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )

        monkeypatch.setattr(startup.os, "lstat", fake_lstat)

        lchown_calls: list[tuple] = []
        monkeypatch.setattr(
            startup.os,
            "lchown",
            lambda p, u, g: lchown_calls.append((p, u, g)),
        )

        startup._fix_config_dir_ownership()

        chowned_paths = {call[0] for call in lchown_calls}
        assert str(tmp_path) in chowned_paths
        assert str(tmp_path / "credentials.json") in chowned_paths
        assert str(tmp_path / "subdir") in chowned_paths
        # nested.json is owned by uid 1000 (matches target), should NOT be chowned
        assert str(tmp_path / "subdir" / "nested.json") not in chowned_paths
        for _, uid, gid in lchown_calls:
            assert uid == target_uid
            assert gid == target_gid

    def test_no_duplicate_chown_for_subdirectories(self, tmp_path: Path, monkeypatch) -> None:
        """Each path should be chowned at most once (no double-processing via dirnames)."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file.txt").write_text("data")

        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
        monkeypatch.setattr(startup.os, "getuid", lambda: 0)
        monkeypatch.setenv("SUDO_UID", "1000")
        monkeypatch.setenv("SUDO_GID", "1000")
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

        original_lstat = os.lstat

        def fake_lstat(path):
            result = original_lstat(path)
            return os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    0,
                    0,
                    result.st_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )

        monkeypatch.setattr(startup.os, "lstat", fake_lstat)

        lchown_calls: list[tuple] = []
        monkeypatch.setattr(
            startup.os,
            "lchown",
            lambda p, u, g: lchown_calls.append((p, u, g)),
        )

        startup._fix_config_dir_ownership()

        all_paths = [call[0] for call in lchown_calls]
        assert len(all_paths) == len(set(all_paths)), f"Duplicate chown calls: {all_paths}"

    def test_handles_permission_error_gracefully(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "file.json").write_text("{}")

        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
        monkeypatch.setattr(startup.os, "getuid", lambda: 0)
        monkeypatch.setenv("SUDO_UID", "1000")
        monkeypatch.setenv("SUDO_GID", "1000")
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

        original_lstat = os.lstat

        def fake_lstat(path):
            result = original_lstat(path)
            return os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    0,
                    0,
                    result.st_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )

        monkeypatch.setattr(startup.os, "lstat", fake_lstat)
        monkeypatch.setattr(
            startup.os,
            "lchown",
            lambda p, u, g: (_ for _ in ()).throw(PermissionError("Operation not permitted")),
        )

        # Should not raise
        startup._fix_config_dir_ownership()

    def test_sudo_gid_defaults_to_sudo_uid_when_absent(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "file.json").write_text("{}")

        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
        monkeypatch.setattr(startup.os, "getuid", lambda: 0)
        monkeypatch.setenv("SUDO_UID", "1000")
        monkeypatch.delenv("SUDO_GID", raising=False)
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)

        original_lstat = os.lstat

        def fake_lstat(path):
            result = original_lstat(path)
            return os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    0,
                    0,
                    result.st_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )

        monkeypatch.setattr(startup.os, "lstat", fake_lstat)

        lchown_calls: list[tuple] = []
        monkeypatch.setattr(
            startup.os,
            "lchown",
            lambda p, u, g: lchown_calls.append((p, u, g)),
        )

        startup._fix_config_dir_ownership()

        assert lchown_calls
        for _, uid, gid in lchown_calls:
            assert uid == 1000
            assert gid == 1000


# ===========================================================================
# _ensure_config_subdirs
# ===========================================================================


class TestEnsureConfigSubdirs:
    def test_creates_workers_and_models_non_root(self, tmp_path: Path, monkeypatch) -> None:
        """Non-root caller (macOS dev, Linux user): subdirs get created, no chown."""
        cfg_dir = tmp_path / ".cyberwave"
        monkeypatch.setattr(startup, "CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(startup, "resolve_config_owner_uid_gid", lambda: None)

        chown_calls: list[tuple] = []
        monkeypatch.setattr(
            startup.os,
            "chown",
            lambda p, u, g: chown_calls.append((p, u, g)),
        )

        startup._ensure_config_subdirs()

        assert cfg_dir.is_dir()
        assert (cfg_dir / "workers").is_dir()
        assert (cfg_dir / "models").is_dir()
        assert chown_calls == []

    def test_chowns_created_subdirs_under_systemd(self, tmp_path: Path, monkeypatch) -> None:
        """Root caller with a user-owned parent chowns created subdirs."""
        cfg_dir = tmp_path / ".cyberwave"
        monkeypatch.setattr(startup, "CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(
            startup,
            "resolve_config_owner_uid_gid",
            lambda: (1000, 1000),
        )

        original_stat = Path.stat

        def fake_stat(self):
            result = original_stat(self)
            return os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    0,
                    0,
                    result.st_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )

        monkeypatch.setattr(Path, "stat", fake_stat)

        chown_calls: list[tuple] = []
        monkeypatch.setattr(
            startup.os,
            "chown",
            lambda p, u, g: chown_calls.append((str(p), u, g)),
        )

        startup._ensure_config_subdirs()

        chowned_paths = {call[0] for call in chown_calls}
        assert str(cfg_dir) in chowned_paths
        assert str(cfg_dir / "workers") in chowned_paths
        assert str(cfg_dir / "models") in chowned_paths
        for _, uid, gid in chown_calls:
            assert (uid, gid) == (1000, 1000)

    def test_is_idempotent(self, tmp_path: Path, monkeypatch) -> None:
        """Re-running with existing subdirs is a no-op."""
        cfg_dir = tmp_path / ".cyberwave"
        cfg_dir.mkdir()
        (cfg_dir / "workers").mkdir()
        (cfg_dir / "models").mkdir()
        monkeypatch.setattr(startup, "CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(startup, "resolve_config_owner_uid_gid", lambda: None)

        # Must not raise.
        startup._ensure_config_subdirs()
        startup._ensure_config_subdirs()

    def test_macos_does_not_chown(self, tmp_path: Path, monkeypatch) -> None:
        """On Darwin the helper returns None → subdirs are created but never chowned."""
        cfg_dir = tmp_path / ".cyberwave"
        monkeypatch.setattr(startup, "CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        # getuid is irrelevant on Darwin but set it to 0 to prove the platform
        # gate — not the uid gate — is what prevents chown.
        monkeypatch.setattr(startup.os, "getuid", lambda: 0)

        chown_calls: list[tuple] = []
        monkeypatch.setattr(
            startup.os,
            "chown",
            lambda p, u, g: chown_calls.append((p, u, g)),
        )

        startup._ensure_config_subdirs()

        assert cfg_dir.is_dir()
        assert (cfg_dir / "workers").is_dir()
        assert (cfg_dir / "models").is_dir()
        assert chown_calls == []


# ===========================================================================
# Camera config drift reconciliation
# ===========================================================================


class TestReconcileCameraConfigDrift:
    """Tests for reconcile_camera_config_drift()."""

    def _write_cameras_json(self, config_dir: Path, selected_device: int) -> Path:
        cameras_file = config_dir / "cameras.json"
        cameras_file.write_text(json.dumps({"selected_device": selected_device}))
        return cameras_file

    def test_no_cameras_json_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "_cameras_json_mtime", None)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        assert startup.reconcile_camera_config_drift() is False

    def test_skips_on_macos(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "_cameras_json_mtime", None)
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")

        self._write_cameras_json(tmp_path, 2)
        assert startup.reconcile_camera_config_drift() is False

    def test_first_call_seeds_mtime_without_restart(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "_cameras_json_mtime", None)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        self._write_cameras_json(tmp_path, 2)
        assert startup.reconcile_camera_config_drift() is False

    def test_no_drift_when_mtime_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        cameras_file = self._write_cameras_json(tmp_path, 2)
        mtime = cameras_file.stat().st_mtime
        monkeypatch.setattr(startup, "_cameras_json_mtime", mtime)

        assert startup.reconcile_camera_config_drift() is False

    def test_drift_triggers_restart(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        cameras_file = self._write_cameras_json(tmp_path, 0)
        old_mtime = cameras_file.stat().st_mtime - 10
        monkeypatch.setattr(startup, "_cameras_json_mtime", old_mtime)

        cameras_file.write_text(json.dumps({"selected_device": 2}))

        fake_inspect = {
            "Config": {
                "Env": ["CYBERWAVE_METADATA_VIDEO_DEVICE=/dev/video0"],
            },
        }

        monkeypatch.setattr(
            startup,
            "_list_running_driver_containers",
            lambda: ["cyberwave-driver-abc12345"],
        )
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda name: fake_inspect,
        )

        restart_calls: list[str] = []
        monkeypatch.setattr(
            startup,
            "_start_camera_config_drift_restart",
            lambda token: restart_calls.append(token),
        )
        monkeypatch.setattr(startup, "load_token", lambda: "test-token")

        assert startup.reconcile_camera_config_drift() is True
        assert restart_calls == ["test-token"]

    def test_no_restart_when_container_matches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        cameras_file = self._write_cameras_json(tmp_path, 2)
        old_mtime = cameras_file.stat().st_mtime - 10
        monkeypatch.setattr(startup, "_cameras_json_mtime", old_mtime)

        cameras_file.write_text(json.dumps({"selected_device": 2}))

        fake_inspect = {
            "Config": {
                "Env": ["CYBERWAVE_METADATA_VIDEO_DEVICE=/dev/video2"],
            },
        }

        monkeypatch.setattr(
            startup,
            "_list_running_driver_containers",
            lambda: ["cyberwave-driver-abc12345"],
        )
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda name: fake_inspect,
        )

        assert startup.reconcile_camera_config_drift() is False

    def test_no_restart_without_token(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        cameras_file = self._write_cameras_json(tmp_path, 0)
        old_mtime = cameras_file.stat().st_mtime - 10
        monkeypatch.setattr(startup, "_cameras_json_mtime", old_mtime)

        cameras_file.write_text(json.dumps({"selected_device": 2}))

        fake_inspect = {
            "Config": {
                "Env": ["CYBERWAVE_METADATA_VIDEO_DEVICE=/dev/video0"],
            },
        }

        monkeypatch.setattr(
            startup,
            "_list_running_driver_containers",
            lambda: ["cyberwave-driver-abc12345"],
        )
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda name: fake_inspect,
        )
        monkeypatch.setattr(startup, "load_token", lambda: None)

        assert startup.reconcile_camera_config_drift() is False

    def test_per_twin_mapping_triggers_restart_for_mismatched_twin(self, tmp_path, monkeypatch):
        """A twin whose mapping no longer matches its container should trigger a restart."""
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        cameras_file = tmp_path / "cameras.json"
        cameras_file.write_text(json.dumps({"selected_device": 0}))
        old_mtime = cameras_file.stat().st_mtime - 10
        monkeypatch.setattr(startup, "_cameras_json_mtime", old_mtime)

        cameras_file.write_text(
            json.dumps(
                {
                    "selected_device": 0,
                    "twin_to_device": {
                        "twin-a": 0,
                        "twin-b": 2,
                    },
                }
            )
        )

        inspect_by_container = {
            "cyberwave-driver-a": {
                "Config": {
                    "Env": [
                        "CYBERWAVE_TWIN_UUID=twin-a",
                        "CYBERWAVE_METADATA_VIDEO_DEVICE=/dev/video0",
                    ],
                },
            },
            "cyberwave-driver-b": {
                "Config": {
                    "Env": [
                        "CYBERWAVE_TWIN_UUID=twin-b",
                        "CYBERWAVE_METADATA_VIDEO_DEVICE=/dev/video0",
                    ],
                },
            },
        }

        monkeypatch.setattr(
            startup,
            "_list_running_driver_containers",
            lambda: list(inspect_by_container.keys()),
        )
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda name: inspect_by_container[name],
        )

        restart_calls: list[str] = []
        monkeypatch.setattr(
            startup,
            "_start_camera_config_drift_restart",
            lambda token: restart_calls.append(token),
        )
        monkeypatch.setattr(startup, "load_token", lambda: "test-token")

        assert startup.reconcile_camera_config_drift() is True
        assert restart_calls == ["test-token"]

    def test_per_twin_mapping_no_restart_when_all_match(self, tmp_path, monkeypatch):
        """Every running container already matches its twin's mapping → no restart."""
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        cameras_file = tmp_path / "cameras.json"
        cameras_file.write_text(json.dumps({"selected_device": 0}))
        old_mtime = cameras_file.stat().st_mtime - 10
        monkeypatch.setattr(startup, "_cameras_json_mtime", old_mtime)

        cameras_file.write_text(
            json.dumps(
                {
                    "selected_device": 0,
                    "twin_to_device": {
                        "twin-a": 0,
                        "twin-b": 2,
                    },
                }
            )
        )

        inspect_by_container = {
            "cyberwave-driver-a": {
                "Config": {
                    "Env": [
                        "CYBERWAVE_TWIN_UUID=twin-a",
                        "CYBERWAVE_METADATA_VIDEO_DEVICE=/dev/video0",
                    ],
                },
            },
            "cyberwave-driver-b": {
                "Config": {
                    "Env": [
                        "CYBERWAVE_TWIN_UUID=twin-b",
                        "CYBERWAVE_METADATA_VIDEO_DEVICE=/dev/video2",
                    ],
                },
            },
        }

        monkeypatch.setattr(
            startup,
            "_list_running_driver_containers",
            lambda: list(inspect_by_container.keys()),
        )
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda name: inspect_by_container[name],
        )

        assert startup.reconcile_camera_config_drift() is False

    def test_drift_restart_runs_in_background_worker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        cameras_file = self._write_cameras_json(tmp_path, 0)
        old_mtime = cameras_file.stat().st_mtime - 10
        monkeypatch.setattr(startup, "_cameras_json_mtime", old_mtime)
        cameras_file.write_text(json.dumps({"selected_device": 2}))

        fake_inspect = {
            "Config": {
                "Env": ["CYBERWAVE_METADATA_VIDEO_DEVICE=/dev/video0"],
            },
        }
        monkeypatch.setattr(
            startup,
            "_list_running_driver_containers",
            lambda: ["cyberwave-driver-abc12345"],
        )
        monkeypatch.setattr(
            startup,
            "_inspect_driver_container",
            lambda name: fake_inspect,
        )
        monkeypatch.setattr(startup, "load_token", lambda: "test-token")

        restart_calls: list[str] = []
        started_threads: list[threading.Thread] = []
        real_thread = threading.Thread

        class _ImmediateThread(real_thread):
            def start(self):
                started_threads.append(self)
                self.run()

        monkeypatch.setattr(
            startup,
            "_perform_edge_core_restart",
            lambda token: restart_calls.append(token) or {},
        )
        monkeypatch.setattr(startup.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(startup, "_CAMERA_DRIFT_RESTART_IN_PROGRESS", False)

        assert startup.reconcile_camera_config_drift() is True
        assert len(started_threads) == 1
        assert restart_calls == ["test-token"]

    def test_does_not_inspect_edge_health_payloads(self):
        """CYB-2004 regression: drift detection must stay decoupled from edge_health.

        ``reconcile_camera_config_drift`` reads only ``cameras.json``
        mtime and ``docker inspect`` env vars; the ``edge_health`` MQTT
        payload (which carries per-stream ``stream_config`` blocks
        post-CYB-2004) plays no role.  A future refactor that would
        wire the two together — for example, gating restarts on the
        running driver's actual ``stream_config.source`` rather than
        the container env — would silently couple this function's
        behaviour to the schema's evolution.  This test pins the
        non-coupling at bytecode level so such a refactor cannot land
        without an explicit, reviewed edit here.

        We inspect the function's referenced names (``co_names``) and
        any nested functions rather than raw source so the test does
        not trip on the audit notes in the function's own docstring.
        """

        def _all_referenced_names(func) -> set[str]:
            """Names this function references, including inside nested defs."""
            code = func.__code__
            collected: set[str] = set(code.co_names) | set(code.co_varnames)
            collected |= set(code.co_freevars)
            for const in code.co_consts:
                # Nested function code objects show up as constants;
                # recurse so the test catches "the coupling is hidden in
                # a local helper" refactors too.
                if hasattr(const, "co_names"):
                    collected |= set(const.co_names)
                    collected |= set(const.co_varnames)
            return collected

        referenced = _all_referenced_names(startup.reconcile_camera_config_drift)

        forbidden = (
            "edge_health",
            "stream_config",
            "_EDGE_HEALTH_CHECK",
            "register_stream_config",
            "_collect_host_metrics",
        )
        offenders = sorted(referenced & set(forbidden))
        assert not offenders, (
            "reconcile_camera_config_drift now references "
            f"{offenders!r}; the CYB-2004 audit notes this function "
            "must remain decoupled from edge_health.  If the coupling "
            "is intentional, update the docstring and this test "
            "together."
        )


class TestLoadSelectedCameraDevice:
    """Tests for _load_selected_camera_device()."""

    def _patch_config_paths(self, monkeypatch, config_dir: Path) -> None:
        monkeypatch.setattr(startup, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(startup, "EDGE_JSON_FILE", config_dir / "edge.json")

    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        self._patch_config_paths(monkeypatch, tmp_path)
        assert startup._load_selected_camera_device() is None
        assert startup._load_selected_camera_device("any-twin") is None

    def test_prefers_edge_json_metadata_over_cameras_json(self, tmp_path, monkeypatch):
        """CYB-1763: edge.json metadata.cameras is the source of truth."""
        self._patch_config_paths(monkeypatch, tmp_path)
        (tmp_path / "cameras.json").write_text(json.dumps({"selected_device": 0}))
        (tmp_path / "edge.json").write_text(
            json.dumps({"metadata": {"cameras": {"selected_device": 5}}})
        )

        assert startup._read_cameras_config() == {"selected_device": 5}
        assert startup._load_selected_camera_device() == "/dev/video5"

    def test_falls_back_to_cameras_json_when_edge_json_missing(self, tmp_path, monkeypatch):
        self._patch_config_paths(monkeypatch, tmp_path)
        (tmp_path / "cameras.json").write_text(json.dumps({"selected_device": 2}))

        assert startup._read_cameras_config() == {"selected_device": 2}
        assert startup._load_selected_camera_device() == "/dev/video2"

    def test_falls_back_when_edge_metadata_cameras_empty(self, tmp_path, monkeypatch):
        self._patch_config_paths(monkeypatch, tmp_path)
        (tmp_path / "cameras.json").write_text(json.dumps({"selected_device": 3}))
        (tmp_path / "edge.json").write_text(json.dumps({"metadata": {"cameras": {}}}))

        assert startup._read_cameras_config() == {"selected_device": 3}
        assert startup._load_selected_camera_device() == "/dev/video3"

    def test_falls_back_to_selected_device(self, tmp_path, monkeypatch):
        self._patch_config_paths(monkeypatch, tmp_path)
        (tmp_path / "cameras.json").write_text(json.dumps({"selected_device": 3}))
        assert startup._load_selected_camera_device() == "/dev/video3"
        assert startup._load_selected_camera_device("twin-without-mapping") == "/dev/video3"

    def test_per_twin_mapping_takes_precedence(self, tmp_path, monkeypatch):
        self._patch_config_paths(monkeypatch, tmp_path)
        (tmp_path / "cameras.json").write_text(
            json.dumps(
                {
                    "selected_device": 0,
                    "twin_to_device": {"twin-a": 2, "twin-b": 4},
                }
            )
        )
        assert startup._load_selected_camera_device("twin-a") == "/dev/video2"
        assert startup._load_selected_camera_device("twin-b") == "/dev/video4"
        # Unknown twin falls back to selected_device.
        assert startup._load_selected_camera_device("twin-c") == "/dev/video0"
        # No twin_uuid still uses the fallback.
        assert startup._load_selected_camera_device() == "/dev/video0"

    def test_invalid_mapping_values_are_ignored(self, tmp_path, monkeypatch):
        self._patch_config_paths(monkeypatch, tmp_path)
        (tmp_path / "cameras.json").write_text(
            json.dumps(
                {
                    "selected_device": 1,
                    "twin_to_device": {"twin-a": "not-a-number"},
                }
            )
        )
        assert startup._load_selected_camera_device("twin-a") == "/dev/video1"


class TestEdgeJsonFileHelpers:
    """Tests for edge.json read/write helpers (CYB-1763)."""

    def test_write_or_update_edge_json_file_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "EDGE_JSON_FILE", tmp_path / "edge.json")
        edge_data = {
            "uuid": "edge-1",
            "metadata": {"cameras": {"selected_device": 1}},
        }

        assert startup.write_or_update_edge_json_file(edge_data) is True
        assert startup._read_edge_json() == edge_data

    def test_read_edge_json_returns_none_for_invalid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        edge_json = tmp_path / "edge.json"
        monkeypatch.setattr(startup, "EDGE_JSON_FILE", edge_json)
        edge_json.write_text("{not-json")

        assert startup._read_edge_json() is None


class TestLoadAudioStreamUrlForTwin:
    """Tests for _load_audio_stream_url_for_twin() (macOS microphone bridge)."""

    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        assert startup._load_audio_stream_url_for_twin("twin-a") is None

    def test_resolves_mapped_twin(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        (tmp_path / "audio_streams.json").write_text(
            json.dumps(
                {
                    "twin_to_stream_url": {
                        "twin-a": "http://host.docker.internal:8101",
                    }
                }
            )
        )
        assert (
            startup._load_audio_stream_url_for_twin("twin-a")
            == "http://host.docker.internal:8101"
        )


class TestEnsureLinuxMicrophoneDockerParams:
    def test_appends_snd_and_audio_group_for_microphone_image(self, monkeypatch):
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
        params = startup._ensure_linux_microphone_docker_params(
            "cyberwave/generic-microphone-driver:latest",
            [],
        )
        assert "-v" in params
        assert "/dev/snd:/dev/snd" in params
        assert "--device-cgroup-rule" in params
        assert "116" in params
        assert "--group-add" in params
        assert "audio" in params

    def test_noop_on_darwin(self, monkeypatch):
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        params = ["-v", "/dev/snd:/dev/snd"]
        assert (
            startup._ensure_linux_microphone_docker_params(
                "cyberwave/generic-microphone-driver:latest",
                params,
            )
            == params
        )

    def test_replaces_static_snd_device_with_bind_mount(self, monkeypatch):
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
        params = startup._ensure_linux_microphone_docker_params(
            "cyberwave/generic-microphone-driver:latest",
            ["--device", "/dev/snd:/dev/snd", "--group-add", "audio"],
        )
        assert "--device" not in params
        assert "/dev/snd:/dev/snd" in params
        assert "--device-cgroup-rule" in params
        assert params.count("audio") == 1

    def test_idempotent_when_snd_bind_mount_already_present(self, monkeypatch):
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
        params = [
            "-v",
            "/dev/snd:/dev/snd",
            "--device-cgroup-rule",
            "c 116:* rmw",
            "--group-add",
            "audio",
        ]
        assert (
            startup._ensure_linux_microphone_docker_params(
                "cyberwave/generic-microphone-driver:latest",
                params,
            )
            == params
        )


class TestMacosAudioStreamInjection:
    _TWIN_UUID = "046aa803-b3e7-46a4-8c3d-9c877fb772ab"

    def test_injects_audio_url_from_json(self, tmp_path, monkeypatch):
        base = TestRunDockerImagePullFallback()
        base._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        (tmp_path / "audio_streams.json").write_text(
            json.dumps(
                {
                    "twin_to_stream_url": {
                        self._TWIN_UUID: "http://host.docker.internal:8101",
                    }
                }
            )
        )
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: False)

        def _fake_run(cmd, **kwargs):
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(
            startup, "_pull_docker_image_with_progress", lambda *a, **kw: None
        )
        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave/generic-microphone-driver:latest",
            [],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )
        assert success is True
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        env_map = base._extract_env_map(docker_run_cmd)
        assert env_map["CYBERWAVE_METADATA_AUDIO_DEVICE"] == (
            "http://host.docker.internal:8101"
        )

    def test_injects_capture_settings_from_audio_streams_json(
        self, tmp_path, monkeypatch
    ):
        base = TestRunDockerImagePullFallback()
        base._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []
        (tmp_path / "audio_streams.json").write_text(
            json.dumps(
                {
                    "twin_to_stream_url": {
                        self._TWIN_UUID: "http://host.docker.internal:8101",
                    },
                    "capture_sample_rate": 32000,
                    "channels": 2,
                }
            )
        )
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(startup, "_is_usbip_server_running", lambda: False)

        def _fake_run(cmd, **kwargs):
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(
            startup, "_pull_docker_image_with_progress", lambda *a, **kw: None
        )
        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave/generic-microphone-driver:latest",
            [],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )
        assert success is True
        docker_run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
        env_map = base._extract_env_map(docker_run_cmd)
        assert env_map["CYBERWAVE_METADATA_AUDIO_SAMPLE_RATE"] == "32000"
        assert env_map["CYBERWAVE_METADATA_AUDIO_CHANNELS"] == "2"


class TestLoadCameraStreamUrlForTwin:
    """Tests for _load_camera_stream_url_for_twin() (macOS multi-camera mapping)."""

    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        assert startup._load_camera_stream_url_for_twin("twin-a") is None

    def test_returns_none_without_twin_uuid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        (tmp_path / "camera_streams.json").write_text(
            json.dumps({"twin_to_stream_url": {"twin-a": "http://host.docker.internal:8091"}})
        )
        assert startup._load_camera_stream_url_for_twin(None) is None
        assert startup._load_camera_stream_url_for_twin("") is None

    def test_resolves_mapped_twin(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        (tmp_path / "camera_streams.json").write_text(
            json.dumps(
                {
                    "twin_to_stream_url": {
                        "twin-a": "http://host.docker.internal:8091",
                        "twin-b": "http://host.docker.internal:8092",
                    }
                }
            )
        )
        assert (
            startup._load_camera_stream_url_for_twin("twin-a") == "http://host.docker.internal:8091"
        )
        assert (
            startup._load_camera_stream_url_for_twin("twin-b") == "http://host.docker.internal:8092"
        )
        assert startup._load_camera_stream_url_for_twin("twin-c") is None

    def test_ignores_invalid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        (tmp_path / "camera_streams.json").write_text("not-json")
        assert startup._load_camera_stream_url_for_twin("twin-a") is None

    def test_ignores_non_string_values(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        (tmp_path / "camera_streams.json").write_text(
            json.dumps({"twin_to_stream_url": {"twin-a": 8091}})
        )
        assert startup._load_camera_stream_url_for_twin("twin-a") is None


# ===========================================================================
# Symmetric worker restart on edge-core restart command
# ===========================================================================
#
# Regression coverage for the asymmetric-restart bug where every edge-core
# restart command stopped/removed the worker container but never restarted
# it, leaving the worker (and every workflow-driven feature: frame filter,
# ML inference, detection overlays, …) down for up to one
# ``reconcile_worker_lifecycle`` cycle (~5 min).


class TestStartWorkerContainerAfterRestart:
    """Tests for ``_start_worker_container_after_restart``."""

    @staticmethod
    def _seed_active_workflow(tmp_path: Path) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir()
        (workers_dir / "wf_demo.py").write_text("# stub\n")

    def test_starts_worker_when_active_workflows_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        self._seed_active_workflow(tmp_path)
        monkeypatch.setattr(
            startup, "_resolve_worker_sync_twin_uuids", lambda *_: ["twin-a", "twin-b"]
        )
        captured: dict = {}
        monkeypatch.setattr(
            startup, "_start_worker_after_drivers", lambda **kw: captured.update(kw)
        )

        result = startup._start_worker_container_after_restart("tok", "env", "fp")

        assert result is True
        assert captured == {
            "token": "tok",
            "environment_uuid": "env",
            "twin_uuids": ["twin-a", "twin-b"],
        }

    @pytest.mark.parametrize(
        "seed",
        [
            pytest.param(lambda _p: None, id="no-workers-dir"),
            pytest.param(lambda p: (p / "workers").mkdir(), id="empty-workers-dir"),
        ],
    )
    def test_skips_when_no_active_workflows(self, tmp_path, monkeypatch, seed):
        """Mirrors ``reconcile_worker_lifecycle``'s gate so we don't pull
        ``cyberwaveos/edge-ml-worker`` on an idle node (CYB-1766)."""
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        seed(tmp_path)

        def fail(**_kw):
            raise AssertionError("worker start must not run when no wf_*.py exist")

        monkeypatch.setattr(startup, "_start_worker_after_drivers", fail)

        assert startup._start_worker_container_after_restart("tok", "env", "fp") is False

    @pytest.mark.parametrize(
        "patch_target",
        ["_resolve_worker_sync_twin_uuids", "_start_worker_after_drivers"],
    )
    def test_failures_are_swallowed_best_effort(self, tmp_path, monkeypatch, patch_target):
        """Helper must never propagate — the runtime loop's reconcile is
        the documented recovery path."""
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        self._seed_active_workflow(tmp_path)
        monkeypatch.setattr(startup, "_resolve_worker_sync_twin_uuids", lambda *_: ["twin-a"])
        monkeypatch.setattr(startup, "_start_worker_after_drivers", lambda **_kw: None)

        def boom(*_a, **_kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(startup, patch_target, boom)

        assert startup._start_worker_container_after_restart("tok", "env", "fp") is False


class TestPerformEdgeCoreRestart:
    """Pin the symmetric stop+start ordering inside ``_perform_edge_core_restart``."""

    def test_stop_and_start_worker_are_paired(self, tmp_path, monkeypatch):
        """Pre-fix, only the stop half ran. This test fails the moment
        someone removes the symmetric start call."""
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        events: list[str] = []

        # Stub everything outside the restart's own logic.
        for name, fn in {
            "_stop_worker_container_for_restart": lambda: events.append("stop_worker"),
            "_remove_cached_twin_json_files": lambda: ["a.json"],
            "_stop_and_prune_driver_containers": lambda: [],
            "load_environment_uuid": lambda **_kw: "env-uuid",
            "get_or_create_fingerprint": lambda: "fp",
            "stop_zenoh_router": lambda _e: None,
            "start_zenoh_router": lambda _cfg, _env: events.append("start_zenoh") or True,
            "fetch_and_run_twin_drivers": (
                lambda _t, _e, _f, **kw: events.append("start_drivers") or [{"success": True}]
            ),
            "_stop_bootstrap_edge_health_publisher": lambda: None,
            "_start_worker_container_after_restart": (
                lambda _t, _e, _f: events.append("start_worker") or True
            ),
        }.items():
            monkeypatch.setattr(startup, name, fn)

        # Zenoh router enabled so we exercise the symmetric router path too.
        monkeypatch.setattr(
            startup, "_get_zenoh_config", lambda: type("Cfg", (), {"router_enabled": True})()
        )

        summary = startup._perform_edge_core_restart("test-token")

        assert events == ["stop_worker", "start_zenoh", "start_drivers", "start_worker"], (
            f"Symmetric ordering broken: {events}"
        )
        assert summary["worker_started"] is True
        assert summary["drivers_started"] == 1


# ===========================================================================
# Camera config source selection (edge.json vs cameras.json)
# ===========================================================================


class TestReadCamerasConfig:
    """Tests for _read_cameras_config() preference order (CYB-1763)."""

    def test_prefers_edge_json_metadata_cameras(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "EDGE_JSON_FILE", tmp_path / "edge.json")

        (tmp_path / "edge.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "cameras": {
                            "selected_device": 1,
                            "devices": ["/dev/video1"],
                        }
                    }
                }
            )
        )
        (tmp_path / "cameras.json").write_text(
            json.dumps({"selected_device": 99, "devices": ["/dev/video99"]})
        )

        config = startup._read_cameras_config()

        assert config == {
            "selected_device": 1,
            "devices": ["/dev/video1"],
        }

    def test_falls_back_to_cameras_json_when_edge_json_missing_cameras(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "EDGE_JSON_FILE", tmp_path / "edge.json")

        (tmp_path / "edge.json").write_text(json.dumps({"metadata": {}}))
        (tmp_path / "cameras.json").write_text(
            json.dumps({"selected_device": 2, "devices": ["/dev/video2"]})
        )

        config = startup._read_cameras_config()

        assert config == {"selected_device": 2, "devices": ["/dev/video2"]}

    def test_returns_none_when_no_camera_config_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup, "EDGE_JSON_FILE", tmp_path / "edge.json")

        assert startup._read_cameras_config() is None


class TestWriteOrUpdateEdgeJsonFile:
    def test_writes_edge_json_atomically(self, tmp_path, monkeypatch):
        edge_json = tmp_path / "edge.json"
        monkeypatch.setattr(startup, "EDGE_JSON_FILE", edge_json)

        payload = {"uuid": "edge-123", "metadata": {"cameras": {"selected_device": 0}}}

        assert startup.write_or_update_edge_json_file(payload) is True
        assert json.loads(edge_json.read_text()) == payload
