"""Tests for cyberwave_edge_core.zenoh_config.

Covers:
  1. ZenohConfig env-var resolution (CYBERWAVE_DATA_BACKEND, ZENOH_CONNECT,
     ZENOH_SHARED_MEMORY, ZENOH_ROUTER_ENABLED, ZENOH_ROUTER_IMAGE,
     ZENOH_ROUTER_PORT).
  2. build_zenoh_env_vars — correct env dict for zenoh / filesystem modes.
  3. validate_zenoh_config — mode strings and warning conditions.
  4. ZenohConfig.router_container_name — deterministic naming.
  5. start_zenoh_router / stop_zenoh_router — subprocess mocking.
  6. startup.py integration — Zenoh env vars injected into driver containers
     via _run_docker_image.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

import cyberwave_edge_core.zenoh_config as zc_module
from cyberwave_edge_core.zenoh_config import (
    ZenohConfig,
    ZenohDiagnostics,
    build_zenoh_env_vars,
    start_zenoh_router,
    stop_zenoh_router,
    validate_zenoh_config,
)

# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def _config(**kwargs) -> ZenohConfig:
    """Build a ZenohConfig with explicit fields (bypasses env-var resolution)."""
    cfg = object.__new__(ZenohConfig)
    cfg.data_backend = kwargs.get("data_backend", "zenoh")
    cfg.connect_endpoints = kwargs.get("connect_endpoints", [])
    cfg.shared_memory = kwargs.get("shared_memory", False)
    cfg.router_enabled = kwargs.get("router_enabled", False)
    cfg.router_image = kwargs.get("router_image", "eclipse/zenoh:latest")
    cfg.router_port = kwargs.get("router_port", 7447)
    return cfg


# ===========================================================================
# 1. ZenohConfig env-var resolution
# ===========================================================================


class TestZenohConfigEnvResolution:
    def test_default_backend_is_zenoh(self, monkeypatch):
        monkeypatch.delenv("CYBERWAVE_DATA_BACKEND", raising=False)
        cfg = ZenohConfig()
        assert cfg.data_backend == "zenoh"

    def test_backend_overridden_by_env(self, monkeypatch):
        monkeypatch.setenv("CYBERWAVE_DATA_BACKEND", "filesystem")
        cfg = ZenohConfig()
        assert cfg.data_backend == "filesystem"

    def test_connect_endpoints_parsed_from_env(self, monkeypatch):
        monkeypatch.setenv("ZENOH_CONNECT", "tcp/10.0.0.1:7447, tcp/10.0.0.2:7447")
        cfg = ZenohConfig()
        assert cfg.connect_endpoints == ["tcp/10.0.0.1:7447", "tcp/10.0.0.2:7447"]

    def test_connect_endpoints_empty_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("ZENOH_CONNECT", raising=False)
        cfg = ZenohConfig()
        assert cfg.connect_endpoints == []

    def test_shared_memory_true_from_env(self, monkeypatch):
        for value in ("1", "true", "yes", "on", "True", "YES"):
            monkeypatch.setenv("ZENOH_SHARED_MEMORY", value)
            cfg = ZenohConfig()
            assert cfg.shared_memory is True, f"expected True for ZENOH_SHARED_MEMORY={value!r}"

    def test_shared_memory_false_from_env(self, monkeypatch):
        monkeypatch.setenv("ZENOH_SHARED_MEMORY", "false")
        cfg = ZenohConfig()
        assert cfg.shared_memory is False

    def test_shared_memory_false_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("ZENOH_SHARED_MEMORY", raising=False)
        cfg = ZenohConfig()
        assert cfg.shared_memory is False

    def test_router_enabled_from_env(self, monkeypatch):
        monkeypatch.setenv("ZENOH_ROUTER_ENABLED", "true")
        cfg = ZenohConfig()
        assert cfg.router_enabled is True

    def test_router_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("ZENOH_ROUTER_ENABLED", raising=False)
        cfg = ZenohConfig()
        assert cfg.router_enabled is False

    def test_router_image_overridden_by_env(self, monkeypatch):
        monkeypatch.setenv("ZENOH_ROUTER_IMAGE", "myrepo/zenoh-router:1.2.3")
        cfg = ZenohConfig()
        assert cfg.router_image == "myrepo/zenoh-router:1.2.3"

    def test_router_image_default(self, monkeypatch):
        monkeypatch.delenv("ZENOH_ROUTER_IMAGE", raising=False)
        cfg = ZenohConfig()
        assert cfg.router_image == "eclipse/zenoh:latest"

    def test_router_port_overridden_by_env(self, monkeypatch):
        monkeypatch.setenv("ZENOH_ROUTER_PORT", "17447")
        cfg = ZenohConfig()
        assert cfg.router_port == 17447

    def test_router_port_default(self, monkeypatch):
        monkeypatch.delenv("ZENOH_ROUTER_PORT", raising=False)
        cfg = ZenohConfig()
        assert cfg.router_port == 7447

    def test_router_port_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("ZENOH_ROUTER_PORT", "not-a-number")
        cfg = ZenohConfig()
        assert cfg.router_port == 7447

    def test_explicit_fields_bypass_env(self, monkeypatch):
        monkeypatch.setenv("CYBERWAVE_DATA_BACKEND", "filesystem")
        monkeypatch.setenv("ZENOH_SHARED_MEMORY", "true")
        cfg = ZenohConfig(data_backend="zenoh", shared_memory=False)
        assert cfg.data_backend == "zenoh"
        assert cfg.shared_memory is False


# ===========================================================================
# 2. ZenohConfig properties
# ===========================================================================


class TestZenohConfigProperties:
    def test_is_zenoh_true_for_zenoh_backend(self):
        cfg = _config(data_backend="zenoh")
        assert cfg.is_zenoh is True

    def test_is_zenoh_false_for_filesystem(self):
        cfg = _config(data_backend="filesystem")
        assert cfg.is_zenoh is False

    def test_peer_to_peer_when_no_endpoints(self):
        cfg = _config(connect_endpoints=[])
        assert cfg.peer_to_peer is True

    def test_not_peer_to_peer_when_endpoints_set(self):
        cfg = _config(connect_endpoints=["tcp/localhost:7447"])
        assert cfg.peer_to_peer is False

    def test_router_container_name_uses_env_uuid_prefix(self):
        cfg = _config()
        assert cfg.router_container_name("abc12345-xxxx") == "cyberwave-zenoh-router-abc12345"

    def test_router_container_name_short_uuid(self):
        cfg = _config()
        assert cfg.router_container_name("ab") == "cyberwave-zenoh-router-ab"

    def test_router_container_name_empty_uuid(self):
        cfg = _config()
        assert cfg.router_container_name("") == "cyberwave-zenoh-router-default"


# ===========================================================================
# 3. build_zenoh_env_vars
# ===========================================================================


class TestBuildZenohEnvVars:
    def test_zenoh_backend_includes_data_backend_key(self):
        cfg = _config(data_backend="zenoh")
        env = build_zenoh_env_vars(cfg)
        assert env["CYBERWAVE_DATA_BACKEND"] == "zenoh"

    def test_filesystem_backend_returns_only_data_backend_key(self):
        cfg = _config(data_backend="filesystem")
        env = build_zenoh_env_vars(cfg)
        assert list(env.keys()) == ["CYBERWAVE_DATA_BACKEND"]
        assert env["CYBERWAVE_DATA_BACKEND"] == "filesystem"

    def test_shared_memory_false_sets_false_string(self):
        cfg = _config(shared_memory=False)
        env = build_zenoh_env_vars(cfg)
        assert env["ZENOH_SHARED_MEMORY"] == "false"

    def test_shared_memory_true_sets_true_string(self):
        cfg = _config(shared_memory=True)
        env = build_zenoh_env_vars(cfg)
        assert env["ZENOH_SHARED_MEMORY"] == "true"

    def test_connect_endpoints_set_when_non_empty(self):
        cfg = _config(connect_endpoints=["tcp/10.0.0.1:7447", "tcp/10.0.0.2:7447"])
        env = build_zenoh_env_vars(cfg)
        assert env["ZENOH_CONNECT"] == "tcp/10.0.0.1:7447,tcp/10.0.0.2:7447"

    def test_connect_endpoints_absent_when_empty(self):
        cfg = _config(connect_endpoints=[])
        env = build_zenoh_env_vars(cfg)
        assert "ZENOH_CONNECT" not in env

    def test_all_values_are_strings(self):
        cfg = _config(shared_memory=True, connect_endpoints=["tcp/localhost:7447"])
        env = build_zenoh_env_vars(cfg)
        for key, value in env.items():
            assert isinstance(key, str), f"key {key!r} is not str"
            assert isinstance(value, str), f"value for {key!r} is not str"


# ===========================================================================
# 4. validate_zenoh_config (ZenohDiagnostics)
# ===========================================================================


class TestValidateZenohConfig:
    def test_filesystem_backend_returns_warning(self):
        cfg = _config(data_backend="filesystem")
        diag = validate_zenoh_config(cfg)
        assert diag.shared_memory_active is False
        assert any("not 'zenoh'" in w or "disabled" in w for w in diag.warnings)

    def test_zenoh_p2p_no_warnings_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Run as Linux to avoid the macOS-specific SHM warning that would fire
        # when shared_memory=True on Darwin.
        monkeypatch.setattr(zc_module.platform, "system", lambda: "Linux")
        cfg = _config(data_backend="zenoh", shared_memory=True, connect_endpoints=[])
        diag = validate_zenoh_config(cfg)
        assert "peer-to-peer" in diag.mode
        # No SHM warning when shared_memory is enabled on Linux
        shm_warnings = [w for w in diag.warnings if "ZENOH_SHARED_MEMORY" in w]
        assert not shm_warnings

    def test_shm_disabled_does_not_generate_warning(self):
        """``shared_memory=False`` is the intentional default for
        containerised deployments (see the ``ZENOH_SHARED_MEMORY`` note in
        ``zenoh_config`` module docstring).  It must not emit an operator
        warning, otherwise every default startup logs a scary line."""
        cfg = _config(data_backend="zenoh", shared_memory=False)
        diag = validate_zenoh_config(cfg)
        assert not any(
            "TCP loopback" in w or "ZENOH_SHARED_MEMORY is not enabled" in w for w in diag.warnings
        )

    def test_router_enabled_without_connect_generates_warning(self):
        cfg = _config(data_backend="zenoh", router_enabled=True, connect_endpoints=[])
        diag = validate_zenoh_config(cfg)
        assert any("ZENOH_CONNECT" in w for w in diag.warnings)

    def test_connect_set_without_router_generates_warning(self):
        cfg = _config(
            data_backend="zenoh",
            router_enabled=False,
            connect_endpoints=["tcp/localhost:7447"],
        )
        diag = validate_zenoh_config(cfg)
        assert any("ZENOH_ROUTER_ENABLED" in w or "external router" in w for w in diag.warnings)

    def test_mode_includes_router_connected_when_endpoints_set(self):
        cfg = _config(
            data_backend="zenoh",
            router_enabled=True,
            connect_endpoints=["tcp/localhost:7447"],
            shared_memory=True,
        )
        diag = validate_zenoh_config(cfg)
        assert "router" in diag.mode.lower()

    def test_shared_memory_active_reflects_config(self):
        cfg_on = _config(shared_memory=True)
        cfg_off = _config(shared_memory=False)
        assert validate_zenoh_config(cfg_on).shared_memory_active is True
        assert validate_zenoh_config(cfg_off).shared_memory_active is False

    def test_returns_zenoh_diagnostics_instance(self):
        cfg = _config()
        diag = validate_zenoh_config(cfg)
        assert isinstance(diag, ZenohDiagnostics)

    def test_connect_endpoints_reflected_in_diagnostics(self):
        endpoints = ["tcp/a:7447", "tcp/b:7447"]
        cfg = _config(connect_endpoints=endpoints, router_enabled=True)
        diag = validate_zenoh_config(cfg)
        assert diag.connect_endpoints == endpoints

    def test_macos_shm_warning(self, monkeypatch):
        import cyberwave_edge_core.zenoh_config as zmod

        monkeypatch.setattr(zmod.platform, "system", lambda: "Darwin")
        cfg = _config(shared_memory=True)
        diag = validate_zenoh_config(cfg)
        assert any("macOS" in w for w in diag.warnings)


# ===========================================================================
# 5. Router container lifecycle (subprocess mocking)
# ===========================================================================


class TestStartZenohRouter:
    def test_returns_true_when_router_disabled(self):
        cfg = _config(router_enabled=False)
        result = start_zenoh_router(cfg, "env-uuid-1234")
        assert result is True

    def test_returns_false_when_docker_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        cfg = _config(router_enabled=True)
        result = start_zenoh_router(cfg, "env-uuid-1234")
        assert result is False

    def test_returns_true_when_already_running(self, monkeypatch):
        import cyberwave_edge_core.zenoh_config as zmod

        monkeypatch.setattr(zmod, "_is_router_running", lambda name: True)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")
        cfg = _config(router_enabled=True)
        result = start_zenoh_router(cfg, "env-uuid-1234")
        assert result is True

    def test_calls_docker_run_with_correct_args(self, monkeypatch):
        import cyberwave_edge_core.zenoh_config as zmod

        monkeypatch.setattr(zmod, "_is_router_running", lambda name: False)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

        captured_cmds: list[list[str]] = []

        def _fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr(zmod.subprocess, "run", _fake_run)
        monkeypatch.setattr(zmod.platform, "system", lambda: "Linux")

        cfg = _config(
            router_enabled=True,
            router_image="eclipse/zenoh:0.11.0",
            router_port=17447,
        )
        result = start_zenoh_router(cfg, "abcd1234-xxxx")

        assert result is True
        # Find the docker run command (not the docker rm -f)
        run_cmd = next(c for c in captured_cmds if "run" in c and "--detach" in c)
        assert "eclipse/zenoh:0.11.0" in run_cmd
        assert "--restart" in run_cmd
        assert "unless-stopped" in run_cmd
        assert "--network" in run_cmd
        assert "host" in run_cmd
        assert "cyberwave-zenoh-router-abcd1234" in run_cmd

    def test_returns_false_on_docker_error(self, monkeypatch):
        import cyberwave_edge_core.zenoh_config as zmod

        monkeypatch.setattr(zmod, "_is_router_running", lambda name: False)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

        def _fake_run(cmd, **kwargs):
            if "--detach" in cmd:
                raise subprocess.CalledProcessError(1, cmd, stderr="image not found")
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr(zmod.subprocess, "run", _fake_run)
        monkeypatch.setattr(zmod.platform, "system", lambda: "Linux")

        cfg = _config(router_enabled=True)
        result = start_zenoh_router(cfg, "env-uuid-1234")
        assert result is False

    def test_returns_false_on_timeout(self, monkeypatch):
        import cyberwave_edge_core.zenoh_config as zmod

        monkeypatch.setattr(zmod, "_is_router_running", lambda name: False)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

        def _fake_run(cmd, **kwargs):
            if "--detach" in cmd:
                raise subprocess.TimeoutExpired(cmd, 60)
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr(zmod.subprocess, "run", _fake_run)
        monkeypatch.setattr(zmod.platform, "system", lambda: "Linux")

        cfg = _config(router_enabled=True)
        result = start_zenoh_router(cfg, "env-uuid-1234")
        assert result is False

    def test_macos_uses_add_host_not_network_host(self, monkeypatch):
        import cyberwave_edge_core.zenoh_config as zmod

        monkeypatch.setattr(zmod, "_is_router_running", lambda name: False)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr(zmod.platform, "system", lambda: "Darwin")

        captured_cmds: list[list[str]] = []

        def _fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr(zmod.subprocess, "run", _fake_run)

        cfg = _config(router_enabled=True)
        start_zenoh_router(cfg, "abcd1234")

        run_cmd = next(c for c in captured_cmds if "run" in c and "--detach" in c)
        assert "--add-host" in run_cmd
        assert "--network" not in run_cmd


class TestStopZenohRouter:
    def test_returns_true_when_docker_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert stop_zenoh_router("env-uuid-1234") is True

    def test_calls_docker_rm_with_correct_name(self, monkeypatch):
        import cyberwave_edge_core.zenoh_config as zmod

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

        captured: list[list[str]] = []

        def _fake_run(cmd, **kwargs):
            captured.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        monkeypatch.setattr(zmod.subprocess, "run", _fake_run)

        result = stop_zenoh_router("abcd1234-xxxx")
        assert result is True
        assert captured[0] == ["docker", "rm", "-f", "cyberwave-zenoh-router-abcd1234"]

    def test_returns_true_even_when_container_not_found(self, monkeypatch):
        import cyberwave_edge_core.zenoh_config as zmod

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

        def _fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stderr = "No such container"
            return result

        monkeypatch.setattr(zmod.subprocess, "run", _fake_run)

        result = stop_zenoh_router("env-uuid-1234")
        assert result is True

    def test_returns_false_on_exception(self, monkeypatch):
        import cyberwave_edge_core.zenoh_config as zmod

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

        def _fake_run(cmd, **kwargs):
            raise OSError("docker not found")

        monkeypatch.setattr(zmod.subprocess, "run", _fake_run)

        result = stop_zenoh_router("env-uuid-1234")
        assert result is False


# ===========================================================================
# 6. startup.py integration — Zenoh env vars injected into driver containers
# ===========================================================================


class TestZenohEnvInjectionInStartup:
    """Verify that _run_docker_image passes Zenoh env vars to docker run."""

    def _make_mock_subprocess(self, captured_cmds: list[list[str]]) -> MagicMock:
        """Return a mock subprocess.run that records commands and simulates success."""

        def _fake_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = ""
            mock.stderr = ""
            return mock

        m = MagicMock(side_effect=_fake_run)
        m.CalledProcessError = subprocess.CalledProcessError
        m.TimeoutExpired = subprocess.TimeoutExpired
        return m

    def test_zenoh_data_backend_injected_into_driver_container(self, monkeypatch, tmp_path):
        import cyberwave_edge_core.startup as startup

        captured: list[list[str]] = []

        # Patch subprocess so docker commands don't actually run.
        monkeypatch.setattr(startup.subprocess, "run", self._make_mock_subprocess(captured))
        monkeypatch.setattr(startup.shutil, "which", lambda name: "/usr/bin/docker")

        # Patch the zenoh config so we control what's returned.
        test_cfg = _config(data_backend="zenoh", shared_memory=False)
        monkeypatch.setattr(startup, "_get_zenoh_config", lambda: test_cfg)

        # Patch away side-effectful helpers.
        monkeypatch.setattr(startup, "_pull_docker_image_with_progress", lambda *a, **kw: None)
        monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda image: True)
        _running = {"State": {"Status": "running"}}
        monkeypatch.setattr(startup, "_inspect_driver_container", lambda name: _running)
        monkeypatch.setattr(startup, "_stream_container_logs", lambda *a, **kw: None)
        monkeypatch.setattr(startup, "DriverStartingAlertContext", MagicMock())
        monkeypatch.setattr(startup, "get_runtime_env_var", lambda name, default=None: None)
        monkeypatch.setattr(startup, "load_credentials_envs", lambda: {})
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        startup._run_docker_image(
            "cyberwaveos/test-driver",
            [],
            twin_uuid="aaaabbbb-cccc-dddd-eeee-111122223333",
            token="test-token",
        )

        # Find the docker run command
        run_cmds = [c for c in captured if len(c) > 2 and c[0] == "docker" and c[1] == "run"]
        assert run_cmds, "No docker run command captured"
        run_args = run_cmds[0]

        # Collect all -e KEY=VALUE pairs
        env_pairs: dict[str, str] = {}
        for i, arg in enumerate(run_args):
            if arg == "-e" and i + 1 < len(run_args):
                key, _, value = run_args[i + 1].partition("=")
                env_pairs[key] = value

        assert "CYBERWAVE_DATA_BACKEND" in env_pairs, (
            f"CYBERWAVE_DATA_BACKEND not found in env pairs: {env_pairs}"
        )
        assert env_pairs["CYBERWAVE_DATA_BACKEND"] == "zenoh"
        assert "ZENOH_SHARED_MEMORY" in env_pairs
        assert env_pairs["ZENOH_SHARED_MEMORY"] == "false"

    def test_zenoh_connect_injected_when_endpoints_configured(self, monkeypatch, tmp_path):
        import cyberwave_edge_core.startup as startup

        captured: list[list[str]] = []
        monkeypatch.setattr(startup.subprocess, "run", self._make_mock_subprocess(captured))
        monkeypatch.setattr(startup.shutil, "which", lambda name: "/usr/bin/docker")

        test_cfg = _config(
            data_backend="zenoh",
            shared_memory=True,
            connect_endpoints=["tcp/10.0.0.1:7447"],
        )
        monkeypatch.setattr(startup, "_get_zenoh_config", lambda: test_cfg)
        monkeypatch.setattr(startup, "_pull_docker_image_with_progress", lambda *a, **kw: None)
        monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda image: True)
        _running = {"State": {"Status": "running"}}
        monkeypatch.setattr(startup, "_inspect_driver_container", lambda name: _running)
        monkeypatch.setattr(startup, "_stream_container_logs", lambda *a, **kw: None)
        monkeypatch.setattr(startup, "DriverStartingAlertContext", MagicMock())
        monkeypatch.setattr(startup, "get_runtime_env_var", lambda name, default=None: None)
        monkeypatch.setattr(startup, "load_credentials_envs", lambda: {})
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        startup._run_docker_image(
            "cyberwaveos/test-driver",
            [],
            twin_uuid="aaaabbbb-cccc-dddd-eeee-111122223333",
            token="test-token",
        )

        run_cmds = [c for c in captured if len(c) > 2 and c[0] == "docker" and c[1] == "run"]
        run_args = run_cmds[0]

        env_pairs: dict[str, str] = {}
        for i, arg in enumerate(run_args):
            if arg == "-e" and i + 1 < len(run_args):
                key, _, value = run_args[i + 1].partition("=")
                env_pairs[key] = value

        assert env_pairs.get("ZENOH_CONNECT") == "tcp/10.0.0.1:7447"
        assert env_pairs.get("ZENOH_SHARED_MEMORY") == "true"

    def test_driver_param_override_takes_precedence_over_zenoh_default(self, monkeypatch, tmp_path):
        """Explicit -e CYBERWAVE_DATA_BACKEND=filesystem in driver params must win."""
        import cyberwave_edge_core.startup as startup

        captured: list[list[str]] = []
        monkeypatch.setattr(startup.subprocess, "run", self._make_mock_subprocess(captured))
        monkeypatch.setattr(startup.shutil, "which", lambda name: "/usr/bin/docker")

        test_cfg = _config(data_backend="zenoh", shared_memory=False)
        monkeypatch.setattr(startup, "_get_zenoh_config", lambda: test_cfg)
        monkeypatch.setattr(startup, "_pull_docker_image_with_progress", lambda *a, **kw: None)
        monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda image: True)
        _running = {"State": {"Status": "running"}}
        monkeypatch.setattr(startup, "_inspect_driver_container", lambda name: _running)
        monkeypatch.setattr(startup, "_stream_container_logs", lambda *a, **kw: None)
        monkeypatch.setattr(startup, "DriverStartingAlertContext", MagicMock())
        monkeypatch.setattr(startup, "get_runtime_env_var", lambda name, default=None: None)
        monkeypatch.setattr(startup, "load_credentials_envs", lambda: {})
        monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

        # Driver metadata params contain an explicit override
        params = ["-e", "CYBERWAVE_DATA_BACKEND=filesystem"]

        startup._run_docker_image(
            "cyberwaveos/test-driver",
            params,
            twin_uuid="aaaabbbb-cccc-dddd-eeee-111122223333",
            token="test-token",
        )

        run_cmds = [c for c in captured if len(c) > 2 and c[0] == "docker" and c[1] == "run"]
        run_args = run_cmds[0]

        env_pairs: dict[str, str] = {}
        for i, arg in enumerate(run_args):
            if arg == "-e" and i + 1 < len(run_args):
                key, _, value = run_args[i + 1].partition("=")
                env_pairs[key] = value

        # The driver-specified value must override the Zenoh default.
        assert env_pairs.get("CYBERWAVE_DATA_BACKEND") == "filesystem"
