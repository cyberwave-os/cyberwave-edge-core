"""Tests for core startup.py functions.

Covers the highest-priority untested areas:
  1. load_token  — missing / malformed credentials file
  2. write_or_update_twin_json_file — deep-merge preserves existing keys
  3. write_or_update_twin_json_file — directory at path replaced by file
  4. load_environment_uuid — retry logic
  5. _remove_cached_twin_json_files — protected files never deleted
"""
from __future__ import annotations

import json
import uuid as _uuid_module
from pathlib import Path
from unittest.mock import patch

import cyberwave_edge_core.startup as startup


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
        real_sleep = startup.time.sleep

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
