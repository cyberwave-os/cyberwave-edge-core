"""Tests for edge remote access (frpc tunnel) orchestration (CYB-4074)."""

from __future__ import annotations

import base64
import logging
import pwd
import stat
import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import cyberwave_edge_core.remote_access as ra
import cyberwave_edge_core.startup as startup

SESSION_A = "0b6f3c5e-1111-4222-8333-444455556666"
SESSION_B = "9d1e2f3a-aaaa-4bbb-8ccc-ddddeeeeffff"
EDGE_UUID = "5f0c7d0e-0000-4000-8000-000000000001"


def _credential(session_id: str = SESSION_A, token: str = "relay-token-secret") -> dict:
    return {
        "session_id": session_id,
        "token": token,
        "proxy_name": f"ssh-{session_id}",
        "remote_port": 20001,
        "local_port": 22,
        "expires_at": "2026-09-24T12:00:00Z",
    }


class FakeClient:
    def __init__(self) -> None:
        self.edge_key = "edge-key-secret"
        self.calls: list[tuple[str, Any]] = []
        self.states: list[Any] = []
        self.claim_error: ra.RemoteAccessHTTPError | None = None
        self.credential_errors: list[ra.RemoteAccessHTTPError] = []
        self.host_keys: list | None = None
        self.ssh_user: str | None = None

    def claim_key(self) -> dict:
        self.calls.append(("claim", None))
        if self.claim_error:
            raise self.claim_error
        return {"ok": True}

    def fetch_credential(self, session_id: str, host_keys: list, ssh_user: str = "") -> dict:
        self.calls.append(("credential", session_id))
        self.host_keys = host_keys
        self.ssh_user = ssh_user
        if self.credential_errors:
            raise self.credential_errors.pop(0)
        return _credential(session_id)

    def session_state(self, session_id: str) -> dict:
        self.calls.append(("state", session_id))
        state = self.states.pop(0)
        if isinstance(state, Exception):
            raise state
        return state

    def report_failure(self, session_id: str, error: str) -> None:
        self.calls.append(("failure", (session_id, error)))


class FakeDocker:
    """Stands in for the docker CLI: tracks the one remote-access container."""

    def __init__(self) -> None:
        self.status = "none"
        self.run_args: list[list[str]] = []
        self.removed = 0
        self.config_at_run: str | None = None
        self.logs = ""

    def run(self, args, **kwargs):
        if args[:2] == ["docker", "run"]:
            self.run_args.append(args)
            self.config_at_run = ra.frpc_config_path().read_text()
            self.status = "running"
            return subprocess.CompletedProcess(args, 0, "cid\n", "")
        if args[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(args, 0, self.logs, "")
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(args, 0, "1\n", "")
        raise AssertionError(f"unexpected docker call {args}")

    def rm(self, name, **kwargs):
        assert name == ra.CONTAINER_NAME
        self.removed += 1
        self.status = "none"
        return True


class SyncThread:
    """Runs the target on start(), except long-lived watchdogs which are recorded."""

    started: list[Any] = []

    def __init__(self, *, target, args=(), name="", daemon=False):
        self.target, self.args, self.name = target, args, name

    def start(self):
        SyncThread.started.append(self)
        if not self.name.startswith("remote-access-watchdog"):
            self.target(*self.args)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    for name in (ra.RELAY_HOST_ENV, ra.RELAY_PORT_ENV, ra.RELAY_CA_FILE_ENV, ra.SSH_USER_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("SUDO_USER", raising=False)
    # The developer's own account would otherwise leak into these assertions.
    monkeypatch.setattr(ra, "_primary_login_user", lambda: "")
    monkeypatch.setattr(startup, "load_credentials_envs", lambda: {})
    docker = FakeDocker()
    monkeypatch.setattr(ra.subprocess, "run", docker.run)
    monkeypatch.setattr(ra, "docker_rm", docker.rm)
    monkeypatch.setattr(ra, "docker_container_status", lambda name: docker.status)
    monkeypatch.setattr(ra, "docker_available", lambda: True)
    monkeypatch.setattr(ra, "build_user_args", lambda: [])
    monkeypatch.setattr(ra, "_docker_host_network", lambda: True)
    ssh_dir = tmp_path / "ssh"
    ssh_dir.mkdir()
    monkeypatch.setattr(ra, "SSH_HOST_KEY_DIR", ssh_dir)
    SyncThread.started = []
    monkeypatch.setattr(ra.threading, "Thread", SyncThread)

    manager = ra.RemoteAccessManager()
    client = FakeClient()
    monkeypatch.setattr(manager, "_client", lambda: client)
    monkeypatch.setattr(ra, "_MANAGER", manager)
    return SimpleNamespace(
        manager=manager, client=client, docker=docker, tmp=tmp_path, ssh_dir=ssh_dir
    )


# -- edge key ----------------------------------------------------------------


def test_edge_key_created_once_with_private_perms(env):
    key = ra.load_or_create_edge_key()
    path = env.tmp / "remote_access_key"
    assert len(key) >= ra.EDGE_KEY_MIN_LENGTH
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ra.load_or_create_edge_key() == key
    assert path.read_text().strip() == key


def test_claim_conflict_keeps_local_key_and_logs_admin_reset(env, caplog):
    key = ra.load_or_create_edge_key()
    env.client.claim_error = ra.RemoteAccessHTTPError(409, "different key")
    with caplog.at_level(logging.ERROR):
        assert env.manager.ensure_key_claimed() is False
    assert "ADMIN must reset" in caplog.text
    assert key not in caplog.text
    assert ra.load_or_create_edge_key() == key


def test_http_client_sends_bearer_and_edge_key(monkeypatch):
    import httpx

    seen: dict[str, Any] = {}

    def fake_request(method, url, *, headers, json, timeout):
        seen.update(method=method, url=url, headers=headers, json=json)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(httpx, "request", fake_request)
    client = ra._BackendClient(
        base_url="https://api.example.com/", token="tok", edge_uuid=EDGE_UUID, edge_key="k" * 43
    )
    client.report_failure(SESSION_A, "boom")
    assert seen["method"] == "POST"
    assert seen["url"] == (
        f"https://api.example.com/api/v1/edges/{EDGE_UUID}"
        f"/remote-access/sessions/{SESSION_A}/failure"
    )
    assert seen["headers"] == {"Authorization": "Bearer tok", "X-Cyberwave-Edge-Key": "k" * 43}
    assert seen["json"] == {"error": "boom"}


# -- frpc config ---------------------------------------------------------------


def test_frpc_config_uses_relay_from_env_not_payload(env, monkeypatch):
    monkeypatch.setenv(ra.RELAY_HOST_ENV, "relay.test.example")
    monkeypatch.setenv(ra.RELAY_PORT_ENV, "7443")

    startup._HANDLED_EDGE_COMMAND_REQUEST_IDS.clear()
    startup._handle_edge_command_message(
        "edges/x/command",
        {
            "command": "open_tunnel",
            "request_id": "open-1",
            "session_id": SESSION_A,
            "server_addr": "evil.example",
            "relay_host": "evil.example",
        },
    )

    config = env.docker.config_at_run
    assert config is not None
    assert 'serverAddr = "relay.test.example"' in config
    assert "serverPort = 7443" in config
    assert "evil.example" not in config
    assert f'metadatas.session_id = "{SESSION_A}"' in config
    assert 'metadatas.token = "relay-token-secret"' in config
    assert "loginFailExit = true" in config
    assert "transport.tls.enable = true" in config
    assert f'transport.tls.trustedCaFile = "{ra.CONTAINER_SYSTEM_CA_BUNDLE}"' in config
    assert 'transport.tls.serverName = "relay.test.example"' in config
    assert f'name = "ssh-{SESSION_A}"' in config
    assert 'type = "tcp"' in config
    assert 'localIP = "127.0.0.1"' in config
    assert "localPort = 22" in config
    assert "remotePort = 20001" in config

    path = ra.frpc_config_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    args = env.docker.run_args[0]
    assert args[args.index("--name") + 1] == ra.CONTAINER_NAME
    assert args[args.index("--network") + 1] == "host"
    assert args[args.index("--restart") + 1] == "no"
    assert f"{path}:/etc/frp/frpc.toml:ro" in args
    assert args[-3:] == [ra.FRPC_IMAGE, "-c", "/etc/frp/frpc.toml"]


def test_frpc_config_defaults_and_escaping():
    config = ra.render_frpc_config(
        session_id=SESSION_A,
        credential=_credential(token='a"b\\c'),
        relay_host=ra.DEFAULT_RELAY_HOST,
        relay_port=ra.DEFAULT_RELAY_PORT,
        trusted_ca_file=ra.CONTAINER_CA_PATH,
    )
    assert 'serverAddr = "relay.edge.cyberwave.com"' in config
    assert "serverPort = 7000" in config
    assert 'metadatas.token = "a\\"b\\\\c"' in config
    assert 'transport.tls.trustedCaFile = "/etc/frp/relay-ca.pem"' in config


def test_relay_ca_file_is_mounted(env, monkeypatch):
    ca = env.tmp / "relay-ca.pem"
    ca.write_text("pem")
    monkeypatch.setenv(ra.RELAY_CA_FILE_ENV, str(ca))
    monkeypatch.setenv(ra.RELAY_HOST_ENV, "localhost")
    env.manager.open_tunnel(SESSION_A)
    args = env.docker.run_args[0]
    config = env.docker.config_at_run
    assert f"{ca}:{ra.CONTAINER_CA_PATH}:ro" in args
    assert f'transport.tls.trustedCaFile = "{ra.CONTAINER_CA_PATH}"' in config
    assert 'transport.tls.serverName = "localhost"' in config
    assert ra.CONTAINER_SYSTEM_CA_BUNDLE not in config


def test_default_relay_verified_against_system_bundle(env):
    env.manager.open_tunnel(SESSION_A)
    config = env.docker.config_at_run
    assert f'transport.tls.trustedCaFile = "{ra.CONTAINER_SYSTEM_CA_BUNDLE}"' in config
    assert f'transport.tls.serverName = "{ra.DEFAULT_RELAY_HOST}"' in config
    assert not any(ra.CONTAINER_CA_PATH in a for a in env.docker.run_args[0])


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "host.docker.internal"])
def test_local_dev_relay_skips_verification(env, monkeypatch, host):
    monkeypatch.setenv(ra.RELAY_HOST_ENV, host)
    env.manager.open_tunnel(SESSION_A)
    config = env.docker.config_at_run
    assert f"serverAddr = {ra._toml_string(host)}" in config
    assert "trustedCaFile" not in config
    assert "serverName" not in config


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_docker_desktop_reaches_host_via_gateway(env, monkeypatch, host):
    monkeypatch.setattr(ra, "_docker_host_network", lambda: False)
    monkeypatch.setenv(ra.RELAY_HOST_ENV, host)
    env.manager.open_tunnel(SESSION_A)
    config = env.docker.config_at_run
    assert 'serverAddr = "host.docker.internal"' in config
    assert 'localIP = "host.docker.internal"' in config
    assert "trustedCaFile" not in config
    assert "--network" not in env.docker.run_args[0]


def test_docker_desktop_keeps_remote_relay_and_verifies(env, monkeypatch):
    monkeypatch.setattr(ra, "_docker_host_network", lambda: False)
    monkeypatch.setenv(ra.RELAY_HOST_ENV, "relay.test.example")
    env.manager.open_tunnel(SESSION_A)
    config = env.docker.config_at_run
    assert 'serverAddr = "relay.test.example"' in config
    assert 'transport.tls.serverName = "relay.test.example"' in config
    assert 'localIP = "host.docker.internal"' in config


# -- ssh host keys ---------------------------------------------------------------


def _pubkey(key_type: str) -> str:
    blob = len(key_type).to_bytes(4, "big") + key_type.encode() + b"\x00\x00\x00\x20" + b"k" * 32
    return base64.b64encode(blob).decode()


def test_host_keys_read_without_comments_and_skip_malformed(env):
    ed = _pubkey("ssh-ed25519")
    ecdsa = _pubkey("ecdsa-sha2-nistp256")
    (env.ssh_dir / "ssh_host_ed25519_key.pub").write_text(f"ssh-ed25519 {ed} root@edge\n")
    (env.ssh_dir / "ssh_host_ecdsa_key.pub").write_text(f"\necdsa-sha2-nistp256 {ecdsa}\n")
    (env.ssh_dir / "ssh_host_rsa_key.pub").write_text(f"ssh-rsa {ed} mismatched-type\n")
    assert ra.read_ssh_host_keys() == [
        {"type": "ssh-ed25519", "key": ed},
        {"type": "ecdsa-sha2-nistp256", "key": ecdsa},
    ]

    (env.ssh_dir / "ssh_host_ed25519_key.pub").write_text("ssh-ed25519 not*base64\n")
    (env.ssh_dir / "ssh_host_ecdsa_key.pub").write_text("garbage\n")
    assert ra.read_ssh_host_keys() == []


def test_open_sends_host_keys_with_credential_request(env):
    ed = _pubkey("ssh-ed25519")
    (env.ssh_dir / "ssh_host_ed25519_key.pub").write_text(f"ssh-ed25519 {ed} root@edge\n")
    env.manager.open_tunnel(SESSION_A)
    assert env.client.host_keys == [{"type": "ssh-ed25519", "key": ed}]


@pytest.mark.parametrize("host_keys", [[], [{"type": "ssh-ed25519", "key": "AAAA"}]])
def test_credential_request_always_sends_host_keys_body(monkeypatch, host_keys):
    import httpx

    seen: dict[str, Any] = {}

    def fake_request(method, url, *, headers, json, timeout):
        seen.update(method=method, url=url, json=json)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(httpx, "request", fake_request)
    client = ra._BackendClient(
        base_url="https://api.example.com", token="tok", edge_uuid=EDGE_UUID, edge_key="k" * 43
    )
    client.fetch_credential(SESSION_A, host_keys)
    assert seen["method"] == "POST"
    assert seen["url"].endswith(f"/remote-access/sessions/{SESSION_A}/credential")
    assert seen["json"] == {"host_keys": host_keys}


# -- ssh login user ----------------------------------------------------------


def _passwd(name: str, uid: int, shell: str = "/bin/bash"):
    return pwd.struct_passwd((name, "x", uid, uid, "", f"/home/{name}", shell))


def test_detect_ssh_user_prefers_the_explicit_override(env, monkeypatch):
    monkeypatch.setenv(ra.SSH_USER_ENV, "operator")
    monkeypatch.setenv("SUDO_USER", "ubuntu")
    assert ra.detect_ssh_login_user() == "operator"


def test_detect_ssh_user_falls_back_to_the_invoking_account(env, monkeypatch):
    monkeypatch.setenv("SUDO_USER", "ubuntu")
    assert ra.detect_ssh_login_user() == "ubuntu"


@pytest.mark.parametrize("bad", ["root", "", "   ", "Ubuntu", "ubuntu; rm -rf /", "u" * 33])
def test_detect_ssh_user_rejects_unusable_names(env, monkeypatch, bad):
    """A name the backend would reject must not displace a usable fallback."""
    monkeypatch.setenv(ra.SSH_USER_ENV, bad)
    monkeypatch.setattr(ra, "_primary_login_user", lambda: "jetson")
    assert ra.detect_ssh_login_user() == "jetson"


def test_detect_ssh_user_is_empty_when_nothing_is_usable(env):
    assert ra.detect_ssh_login_user() == ""


def test_primary_login_user_picks_the_lowest_uid_login_account(monkeypatch):
    monkeypatch.setattr(
        pwd,
        "getpwall",
        lambda: [
            _passwd("root", 0),
            _passwd("daemon", 1, "/usr/sbin/nologin"),
            _passwd("jetson", 1001),
            _passwd("ubuntu", 1000),
            _passwd("nobody", 65534),
        ],
    )
    assert ra._primary_login_user() == "ubuntu"


@pytest.mark.parametrize("shell", ["/usr/sbin/nologin", "/bin/false"])
def test_primary_login_user_skips_accounts_that_cannot_log_in(monkeypatch, shell):
    monkeypatch.setattr(
        pwd,
        "getpwall",
        lambda: [_passwd("root", 0), _passwd("svc", 1000, shell), _passwd("pi", 1002)],
    )
    assert ra._primary_login_user() == "pi"


def test_primary_login_user_is_empty_on_a_device_with_no_login_account(monkeypatch):
    monkeypatch.setattr(pwd, "getpwall", lambda: [_passwd("root", 0), _passwd("nobody", 65534)])
    assert ra._primary_login_user() == ""


def test_open_sends_the_detected_ssh_user(env, monkeypatch):
    monkeypatch.setenv(ra.SSH_USER_ENV, "ubuntu")
    env.manager.open_tunnel(SESSION_A)
    assert env.client.ssh_user == "ubuntu"


def test_open_sends_no_ssh_user_when_detection_finds_none(env):
    env.manager.open_tunnel(SESSION_A)
    assert env.client.ssh_user == ""


def test_credential_request_includes_ssh_user_only_when_known(monkeypatch):
    import httpx

    seen: dict[str, Any] = {}

    def fake_request(method, url, *, headers, json, timeout):
        seen.update(json=json)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(httpx, "request", fake_request)
    client = ra._BackendClient(
        base_url="https://api.example.com", token="tok", edge_uuid=EDGE_UUID, edge_key="k" * 43
    )
    client.fetch_credential(SESSION_A, [], "ubuntu")
    assert seen["json"] == {"host_keys": [], "ssh_user": "ubuntu"}
    client.fetch_credential(SESSION_A, [], "")
    assert seen["json"] == {"host_keys": []}


# -- open / close ----------------------------------------------------------------


def test_open_claims_key_then_starts_frpc_and_watchdog(env):
    env.manager.open_tunnel(SESSION_A)
    assert [c[0] for c in env.client.calls] == ["claim", "credential"]
    assert env.docker.status == "running"
    assert [t.name for t in SyncThread.started] == [f"remote-access-watchdog-{SESSION_A[:8]}"]


def test_open_reclaims_key_after_admin_reset(env):
    env.manager.ensure_key_claimed()
    env.client.credential_errors = [ra.RemoteAccessHTTPError(409, "No edge key is claimed")]
    env.manager.open_tunnel(SESSION_A)
    assert [c[0] for c in env.client.calls] == ["claim", "credential", "claim", "credential"]
    assert env.docker.status == "running"


def test_open_same_session_is_noop(env):
    env.manager.open_tunnel(SESSION_A)
    env.manager.open_tunnel(SESSION_A)
    assert len(env.docker.run_args) == 1
    assert [c[0] for c in env.client.calls] == ["claim", "credential"]


def test_open_other_session_stops_current_first(env):
    env.manager.open_tunnel(SESSION_A)
    first = env.manager._tunnel
    env.manager.open_tunnel(SESSION_B)
    assert first.stop_event.is_set()
    assert env.manager._tunnel.session_id == SESSION_B
    assert len(env.docker.run_args) == 2
    assert f'metadatas.session_id = "{SESSION_B}"' in env.docker.config_at_run


def test_close_tunnel_via_mqtt_removes_container_and_config(env):
    startup._HANDLED_EDGE_COMMAND_REQUEST_IDS.clear()
    startup._handle_edge_command_message(
        {"command": "open_tunnel", "request_id": "o-1", "session_id": SESSION_A}
    )
    tunnel = env.manager._tunnel
    assert ra.frpc_config_path().exists()
    assert ra.session_file_path().read_text().strip() == SESSION_A
    assert stat.S_IMODE(ra.session_file_path().stat().st_mode) == 0o600

    startup._handle_edge_command_message(
        {"command": "close_tunnel", "request_id": "c-1", "session_id": SESSION_B}
    )
    assert env.manager._tunnel is tunnel

    startup._handle_edge_command_message(
        {"command": "close_tunnel", "request_id": "c-2", "session_id": SESSION_A}
    )
    assert env.manager._tunnel is None
    assert tunnel.stop_event.is_set()
    assert env.docker.status == "none"
    assert not ra.frpc_config_path().exists()
    assert not ra.session_file_path().exists()


def test_failed_start_forgets_session(env, monkeypatch):
    env.manager.open_tunnel(SESSION_A)
    monkeypatch.setattr(ra, "docker_available", lambda: False)
    env.manager.open_tunnel(SESSION_B)
    assert env.manager._tunnel is None
    assert not ra.session_file_path().exists()
    assert ("failure", (SESSION_B, "docker is not available on this edge")) in env.client.calls


def test_dispatch_dedupes_and_rejects_bad_session_id(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(ra, "handle_command", lambda command, payload: calls.append(command))
    startup._HANDLED_EDGE_COMMAND_REQUEST_IDS.clear()
    payload = {"command": "open_tunnel", "request_id": "dup", "session_id": SESSION_A}
    startup._handle_edge_command_message("edges/x/command", payload)
    startup._handle_edge_command_message("edges/x/command", payload)
    assert calls == ["open_tunnel"]


def test_handle_command_rejects_non_uuid_session_id(monkeypatch):
    manager = MagicMock()
    monkeypatch.setattr(ra, "_MANAGER", manager)
    ra.handle_command("open_tunnel", {"session_id": "../../key"})
    manager.open_tunnel.assert_not_called()


def test_restart_command_still_spawns_restart_worker(monkeypatch):
    monkeypatch.setattr(startup, "load_token", lambda: "tok")
    monkeypatch.setattr(startup, "_perform_edge_core_restart", MagicMock())
    handle = MagicMock()
    monkeypatch.setattr(ra, "handle_command", handle)
    startup._HANDLED_EDGE_COMMAND_REQUEST_IDS.clear()
    started: list[tuple] = []

    class FakeThread:
        def __init__(self, *, target, args=(), name="", daemon=False):
            started.append((target, args))

        def start(self):
            pass

    monkeypatch.setattr(startup.threading, "Thread", FakeThread)
    startup._handle_edge_command_message(
        "edges/x/command",
        {"command": "restart_edge_core", "request_id": "r-1", "alert_uuid": "al-1"},
    )
    assert started == [(startup._run_edge_core_restart_worker, ("r-1", "al-1"))]
    handle.assert_not_called()


# -- watchdog --------------------------------------------------------------------


@pytest.fixture
def fast_watchdog(monkeypatch):
    monkeypatch.setattr(ra, "STARTUP_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(ra, "WATCHDOG_INTERVAL_SECONDS", 0.0)


def _run_watchdog(env) -> None:
    watchdog = SyncThread.started[-1]
    watchdog.target(*watchdog.args)


def test_watchdog_stops_tunnel_when_session_not_live(env, fast_watchdog):
    env.manager.open_tunnel(SESSION_A)
    env.client.states = [
        {"session_id": SESSION_A, "status": "active", "live": True},
        {"session_id": SESSION_A, "status": "closed", "live": False},
    ]
    _run_watchdog(env)
    assert env.manager._tunnel is None
    assert env.docker.status == "none"
    assert not ra.frpc_config_path().exists()
    assert not ra.session_file_path().exists()
    assert [c[0] for c in env.client.calls].count("state") == 2
    assert "failure" not in [c[0] for c in env.client.calls]


def test_watchdog_stops_on_404_but_survives_transient_errors(env, fast_watchdog):
    env.manager.open_tunnel(SESSION_A)
    env.client.states = [
        ra.RemoteAccessHTTPError(None, "ConnectError"),
        ra.RemoteAccessHTTPError(502, "bad gateway"),
        ra.RemoteAccessHTTPError(404, "Not Found"),
    ]
    _run_watchdog(env)
    assert env.manager._tunnel is None
    assert [c[0] for c in env.client.calls].count("state") == 3


def test_watchdog_reports_early_exit_with_redacted_logs(env, fast_watchdog):
    env.manager.open_tunnel(SESSION_A)
    env.docker.status = "exited"
    env.docker.logs = (
        "\x1b[1;33m[W] login to the server failed: session is not live "
        "relay-token-secret edge-key-secret\x1b[0m\n"
    )
    _run_watchdog(env)

    failures = [c[1] for c in env.client.calls if c[0] == "failure"]
    assert len(failures) == 1
    session_id, error = failures[0]
    assert session_id == SESSION_A
    assert "login to the server failed" in error
    assert "relay-token-secret" not in error
    assert "edge-key-secret" not in error
    assert "\x1b" not in error
    assert env.manager._tunnel is None
    assert not ra.frpc_config_path().exists()
    assert not ra.session_file_path().exists()


def test_watchdog_exits_quietly_after_close(env, fast_watchdog):
    env.manager.open_tunnel(SESSION_A)
    env.manager.close_tunnel(SESSION_A)
    _run_watchdog(env)
    assert [c[0] for c in env.client.calls] == ["claim", "credential"]


# -- startup / shutdown ----------------------------------------------------------


def test_on_edge_resolved_removes_stale_tunnel_once_and_claims(env):
    ra.write_frpc_config("stale")
    env.docker.status = "running"
    env.manager.on_edge_resolved(EDGE_UUID)
    assert env.docker.removed == 1
    assert not ra.frpc_config_path().exists()
    assert env.client.calls == [("claim", None)]

    env.manager.on_edge_resolved(EDGE_UUID)
    assert env.docker.removed == 1


def test_shutdown_stops_tunnel_but_keeps_session_for_resume(env):
    env.manager.open_tunnel(SESSION_A)
    env.manager.shutdown()
    assert env.docker.status == "none"
    assert env.manager._tunnel is None
    assert not ra.frpc_config_path().exists()
    assert ra.session_file_path().read_text().strip() == SESSION_A
    assert "failure" not in [c[0] for c in env.client.calls]


def _saved_session_restart(env, content: str) -> None:
    ra._write_private(ra.session_file_path(), content)
    ra.write_frpc_config("stale")
    env.docker.status = "running"
    env.manager.on_edge_resolved(EDGE_UUID)


def test_restart_resumes_live_saved_session(env):
    env.client.states = [{"session_id": SESSION_A, "status": "active", "live": True}]
    _saved_session_restart(env, SESSION_A + "\n")
    assert env.client.calls == [("claim", None), ("state", SESSION_A), ("credential", SESSION_A)]
    assert env.manager._tunnel.session_id == SESSION_A
    assert env.docker.status == "running"
    assert f'metadatas.session_id = "{SESSION_A}"' in env.docker.config_at_run
    assert ra.session_file_path().read_text().strip() == SESSION_A


@pytest.mark.parametrize(
    "state",
    [
        {"session_id": SESSION_A, "status": "closed", "live": False},
        ra.RemoteAccessHTTPError(404, "Not Found"),
    ],
)
def test_restart_drops_saved_session_that_is_not_live(env, state):
    env.client.states = [state]
    _saved_session_restart(env, SESSION_A)
    assert env.client.calls == [("claim", None), ("state", SESSION_A)]
    assert env.manager._tunnel is None
    assert env.docker.status == "none"
    assert not ra.session_file_path().exists()


def test_restart_keeps_saved_session_on_transient_state_error(env):
    env.client.states = [ra.RemoteAccessHTTPError(None, "ConnectError")]
    _saved_session_restart(env, SESSION_A)
    assert env.manager._tunnel is None
    assert ra.session_file_path().exists()


def test_restart_discards_invalid_saved_session(env):
    _saved_session_restart(env, "../../not-a-uuid")
    assert env.client.calls == [("claim", None)]
    assert not ra.session_file_path().exists()
