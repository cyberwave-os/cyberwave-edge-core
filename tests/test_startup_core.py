"""Tests for core startup.py functions.

Covers the highest-priority untested areas:
  1. load_token  — missing / malformed credentials file
  2. write_or_update_twin_json_file — deep-merge preserves existing keys
  3. write_or_update_twin_json_file — directory at path replaced by file
  4. load_environment_uuid — retry logic
  5. _remove_cached_twin_json_files — protected files never deleted
"""
from __future__ import annotations

import itertools
import json
import subprocess
import uuid as _uuid_module

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
        monkeypatch.setattr(startup, "_resolve_sudo_user_home", lambda: startup.Path("/Users/alice"))
        monkeypatch.setattr(startup.Path, "home", lambda: startup.Path("/var/root"))

        assert startup._resolve_config_dir() == startup.Path("/Users/alice/.cyberwave")

    def test_linux_default_remains_etc_cyberwave(self, monkeypatch):
        monkeypatch.delenv("CYBERWAVE_EDGE_CONFIG_DIR", raising=False)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        assert startup._resolve_config_dir() == startup.Path("/etc/cyberwave")

    def test_migrate_legacy_macos_config_copies_json_files(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CYBERWAVE_EDGE_CONFIG_DIR", raising=False)
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        legacy_dir = tmp_path / "legacy"
        target_dir = tmp_path / "new"
        legacy_dir.mkdir()
        (legacy_dir / "credentials.json").write_text('{"token":"abc"}')
        (legacy_dir / "environment.json").write_text('{"uuid":"123"}')
        monkeypatch.setattr(startup, "_LEGACY_MACOS_CONFIG_DIR", legacy_dir)

        startup._migrate_legacy_macos_config(target_dir)

        assert (target_dir / "credentials.json").exists()
        assert (target_dir / "environment.json").exists()

    def test_migrate_legacy_macos_config_skips_when_env_override_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CYBERWAVE_EDGE_CONFIG_DIR", str(tmp_path / "custom"))
        monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
        legacy_dir = tmp_path / "legacy"
        target_dir = tmp_path / "new"
        legacy_dir.mkdir()
        (legacy_dir / "credentials.json").write_text('{"token":"abc"}')
        monkeypatch.setattr(startup, "_LEGACY_MACOS_CONFIG_DIR", legacy_dir)

        startup._migrate_legacy_macos_config(target_dir)

        assert not (target_dir / "credentials.json").exists()


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

        startup.write_or_update_twin_json_file(
            self._TWIN_UUID, {"x": 99, "nested": {"a": 42}}, {}
        )
        written = json.loads((tmp_path / f"{self._TWIN_UUID}.json").read_text())
        assert written["x"] == 99
        assert written["nested"]["a"] == 42
        # "b" existed in existing nested dict and was not in override → preserved
        assert written["nested"]["b"] == 2


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

    def test_uses_local_image_when_pull_fails(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            if cmd[:2] == ["docker", "pull"]:
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=cmd,
                    stderr="pull access denied",
                )
            if cmd[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is True
        assert any(cmd[:3] == ["docker", "image", "inspect"] for cmd in commands)
        assert any(cmd[:2] == ["docker", "run"] for cmd in commands)

    def test_fails_when_pull_fails_and_image_missing_locally(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch)
        commands: list[list[str]] = []

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            commands.append(list(cmd))
            if cmd[:2] == ["docker", "pull"]:
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=cmd,
                    stderr="pull access denied",
                )
            if cmd[:3] == ["docker", "image", "inspect"]:
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=cmd,
                    stderr="No such image",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(startup.subprocess, "run", _fake_run)

        success = startup._run_docker_image(
            "cyberwave-step14-driver:latest",
            [],
            twin_uuid=self._TWIN_UUID,
            token="test-token",
        )

        assert success is False
        assert any(cmd[:2] == ["docker", "pull"] for cmd in commands)
        assert not any(cmd[:2] == ["docker", "run"] for cmd in commands)
