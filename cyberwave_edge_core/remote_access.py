"""Edge remote access: SSH through the frp relay (CYB-4074).

The backend sends ``open_tunnel`` / ``close_tunnel`` with a session id only.
Edge-core fetches the relay token over REST with its per-edge key, runs frpc
in the ``cyberwave-remote-access`` container and polls the session state as a
watchdog. The relay address comes from local config, never from the payload,
and the relay itself re-checks every login, proxy and connection. The running
session id is kept on disk so a restarted edge-core resumes a still-live session.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .docker_args import build_log_args
from .docker_helpers import (
    build_user_args,
    docker_available,
    docker_container_status,
    docker_rm,
)
from .utils import atomic_write_text

logger = logging.getLogger(__name__)

COMMAND_OPEN_TUNNEL = "open_tunnel"
COMMAND_CLOSE_TUNNEL = "close_tunnel"
COMMANDS = frozenset({COMMAND_OPEN_TUNNEL, COMMAND_CLOSE_TUNNEL})

CONTAINER_NAME = "cyberwave-remote-access"
# fatedier/frpc:v0.71.0, multi-arch index digest.
FRPC_IMAGE = (
    "fatedier/frpc:v0.71.0@sha256:99ece6a2b62cfc68731e0df289af804ff1c699911cfc47871856434f1d6d53ee"
)
CONTAINER_CONFIG_PATH = "/etc/frp/frpc.toml"
CONTAINER_CA_PATH = "/etc/frp/relay-ca.pem"
# Present in the pinned frpc image.
CONTAINER_SYSTEM_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
DOCKER_HOST_GATEWAY = "host.docker.internal"

SESSION_FILENAME = "session"
SSH_HOST_KEY_DIR = Path("/etc/ssh")
SSH_HOST_KEY_FILES = (
    "ssh_host_ed25519_key.pub",
    "ssh_host_ecdsa_key.pub",
    "ssh_host_rsa_key.pub",
)

RELAY_HOST_ENV = "CYBERWAVE_REMOTE_ACCESS_RELAY_HOST"
RELAY_PORT_ENV = "CYBERWAVE_REMOTE_ACCESS_RELAY_PORT"
RELAY_CA_FILE_ENV = "CYBERWAVE_REMOTE_ACCESS_RELAY_CA_FILE"
SSH_USER_ENV = "CYBERWAVE_REMOTE_ACCESS_SSH_USER"
# Mirrors the backend's validation: a name it would reject is not worth sending.
_SSH_USER_RE = re.compile(r"^[a-z_][a-z0-9_.-]{0,31}$")
_MIN_LOGIN_UID = 1000
_MAX_LOGIN_UID = 65534
DEFAULT_RELAY_HOST = "relay.edge.cyberwave.com"
# 443, not a port of our own: corporate allowlists permit 80/443 only.
DEFAULT_RELAY_PORT = 443

EDGE_KEY_HEADER = "X-Cyberwave-Edge-Key"
EDGE_KEY_FILENAME = "remote_access_key"
EDGE_KEY_MIN_LENGTH = 32

HTTP_TIMEOUT_SECONDS = 15.0
DOCKER_RUN_TIMEOUT_SECONDS = 300
# frpc exits within this window when the relay rejects its first login
# (``loginFailExit``); dialing an unreachable relay alone takes up to 10 s.
STARTUP_GRACE_SECONDS = 20.0
STARTUP_POLL_SECONDS = 1.0
WATCHDOG_INTERVAL_SECONDS = 30.0
FAILURE_LOG_LINES = 20
FAILURE_MESSAGE_MAX_CHARS = 1500

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_EXITED_STATUSES = frozenset({"exited", "dead", "none"})
_EDGE_KEY_LOCK = threading.Lock()


class RemoteAccessHTTPError(Exception):
    def __init__(self, status_code: Optional[int], message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _startup() -> Any:
    from . import startup  # noqa: PLC0415

    return startup


def edge_key_path() -> Path:
    return _startup().CONFIG_DIR / EDGE_KEY_FILENAME


def frpc_config_dir() -> Path:
    return _startup().CONFIG_DIR / "remote-access"


def frpc_config_path() -> Path:
    return frpc_config_dir() / "frpc.toml"


def session_file_path() -> Path:
    return frpc_config_dir() / SESSION_FILENAME


def _docker_host_network() -> bool:
    """Only native Linux Docker shares the host network namespace.

    On Docker Desktop, ``--network host`` is the VM's network, not the host sshd.
    """
    return sys.platform == "linux"


def load_or_create_edge_key() -> str:
    """Return this edge's remote-access key, creating it on first use."""
    path = edge_key_path()
    with _EDGE_KEY_LOCK:
        try:
            existing = path.read_text().strip()
        except FileNotFoundError:
            existing = ""
        if existing:
            if os.name != "nt" and (path.stat().st_mode & 0o077):
                os.chmod(path, 0o600)
            return existing

        path.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_urlsafe(32)
        atomic_write_text(path, key + "\n")
        logger.info("Created remote access edge key at %s", path)
        return key


def relay_address() -> tuple[str, int]:
    get = _startup().get_runtime_env_var
    host = get(RELAY_HOST_ENV, DEFAULT_RELAY_HOST) or DEFAULT_RELAY_HOST
    raw_port = get(RELAY_PORT_ENV, str(DEFAULT_RELAY_PORT))
    try:
        port = int(str(raw_port))
        if not 0 < port < 65536:
            raise ValueError
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", RELAY_PORT_ENV, raw_port, DEFAULT_RELAY_PORT)
        port = DEFAULT_RELAY_PORT
    return host, port


def relay_ca_file() -> Optional[str]:
    return _startup().get_runtime_env_var(RELAY_CA_FILE_ENV) or None


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _container_relay_host(relay_host: str) -> str:
    if not _docker_host_network() and _is_loopback(relay_host):
        return DOCKER_HOST_GATEWAY
    return relay_host


def _relay_tls_ca(relay_host: str, ca_file: Optional[str]) -> Optional[str]:
    """In-container CA bundle frpc verifies the relay against; None skips verification.

    The local dev frps serves a self-signed certificate.
    """
    if ca_file:
        return CONTAINER_CA_PATH
    if _is_loopback(relay_host) or relay_host.lower() == DOCKER_HOST_GATEWAY:
        return None
    return CONTAINER_SYSTEM_CA_BUNDLE


def _parse_host_key(line: str) -> Optional[dict[str, str]]:
    parts = line.split()
    if len(parts) < 2:
        return None
    key_type, key = parts[0], parts[1]
    try:
        blob = base64.b64decode(key, validate=True)
    except (binascii.Error, ValueError):
        return None
    # The wire blob starts with the length-prefixed key type.
    name_len = int.from_bytes(blob[:4], "big")
    if blob[4 : 4 + name_len] != key_type.encode():
        return None
    return {"type": key_type, "key": key}


def read_ssh_host_keys() -> list[dict[str, str]]:
    """Public host keys (type and key only, never the comment) for the backend to pin."""
    keys = []
    for name in SSH_HOST_KEY_FILES:
        try:
            text = (SSH_HOST_KEY_DIR / name).read_text()
        except (OSError, UnicodeDecodeError):
            continue
        line = next((ln for ln in text.splitlines() if ln.strip()), "")
        parsed = _parse_host_key(line)
        if parsed is None:
            logger.debug("Remote access: skipping malformed host key %s", name)
            continue
        keys.append(parsed)
    return keys


def _valid_login_user(value: str | None) -> str:
    """The name if an operator could log in as it, else ""."""
    user = (value or "").strip()
    if user == "root" or not _SSH_USER_RE.match(user):
        return ""
    return user


def _primary_login_user() -> str:
    """The lowest-UID real login account, the way ``ngrok/boot.sh`` picks one."""
    # Deferred: ``startup.py`` imports this module at load, so a top-level
    # ``pwd`` would fail the whole package's import on a platform without it.
    import pwd  # noqa: PLC0415

    try:
        entries = pwd.getpwall()
    except OSError:
        return ""
    candidates = [
        (entry.pw_uid, _valid_login_user(entry.pw_name))
        for entry in entries
        if _MIN_LOGIN_UID <= entry.pw_uid < _MAX_LOGIN_UID
        and not entry.pw_shell.endswith(("nologin", "false"))
    ]
    usable = [(uid, name) for uid, name in candidates if name]
    return min(usable)[1] if usable else ""


def detect_ssh_login_user() -> str:
    """The account an operator should SSH in as, or "" to leave it to the backend.

    Edge-core runs as root under systemd, so ``id -un`` would name an account
    that normally cannot log in at all -- hence the explicit override first,
    then the invoking account, then the device's primary login user.
    """
    for candidate in (os.environ.get(SSH_USER_ENV), os.environ.get("SUDO_USER")):
        user = _valid_login_user(candidate)
        if user:
            return user
    return _primary_login_user()


def _toml_string(value: str) -> str:
    out = []
    for ch in value:
        if ch in ('"', "\\"):
            out.append("\\" + ch)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _port(value: Any, name: str) -> int:
    port = int(value)
    if not 0 < port < 65536:
        raise ValueError(f"{name} out of range: {value!r}")
    return port


def render_frpc_config(
    *,
    session_id: str,
    credential: dict[str, Any],
    relay_host: str,
    relay_port: int,
    trusted_ca_file: Optional[str] = None,
    tls_server_name: Optional[str] = None,
    local_ip: str = "127.0.0.1",
) -> str:
    """Build the frpc TOML (frp v0.71) for one session.

    Without ``trusted_ca_file`` frpc skips relay certificate verification.
    """
    lines = [
        f"serverAddr = {_toml_string(relay_host)}",
        f"serverPort = {_port(relay_port, 'relay port')}",
        "loginFailExit = true",
        f"metadatas.session_id = {_toml_string(session_id)}",
        f"metadatas.token = {_toml_string(str(credential['token']))}",
        "transport.heartbeatInterval = 30",
        "transport.heartbeatTimeout = 90",
        "transport.tls.enable = true",
    ]
    if trusted_ca_file:
        lines.append(f"transport.tls.trustedCaFile = {_toml_string(trusted_ca_file)}")
    if tls_server_name:
        lines.append(f"transport.tls.serverName = {_toml_string(tls_server_name)}")
    lines += [
        'log.to = "console"',
        "",
        "[[proxies]]",
        f"name = {_toml_string(str(credential['proxy_name']))}",
        'type = "tcp"',
        f"localIP = {_toml_string(local_ip)}",
        f"localPort = {_port(credential['local_port'], 'local_port')}",
        f"remotePort = {_port(credential['remote_port'], 'remote_port')}",
        "",
    ]
    return "\n".join(lines)


def _write_private(path: Path, content: str) -> None:
    directory = frpc_config_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        os.chmod(directory, 0o700)
    atomic_write_text(path, content)


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Failed to remove %s: %s", path, exc)


def write_frpc_config(content: str) -> Path:
    path = frpc_config_path()
    _write_private(path, content)
    return path


def remove_frpc_config() -> None:
    _unlink_quietly(frpc_config_path())


def save_session(session_id: str) -> None:
    """Remember the running session so a restarted edge-core can resume it."""
    try:
        _write_private(session_file_path(), session_id + "\n")
    except OSError as exc:
        logger.warning("Remote access: failed to save session %s: %s", session_id, exc)


def load_saved_session() -> Optional[str]:
    path = session_file_path()
    try:
        raw = path.read_text().strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Remote access: unreadable %s: %s", path, exc)
        raw = ""
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        logger.warning("Remote access: discarding invalid saved session %r", raw[:64])
        forget_session()
        return None


def forget_session() -> None:
    _unlink_quietly(session_file_path())


def build_docker_run_args(config_path: Path, ca_file: Optional[str] = None) -> list[str]:
    network = ["--network", "host"] if _docker_host_network() else []
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        *network,
        "--restart",
        "no",
        "--read-only",
        "--security-opt",
        "no-new-privileges",
        *build_user_args(),
        *build_log_args(),
        "-v",
        f"{config_path}:{CONTAINER_CONFIG_PATH}:ro",
    ]
    if ca_file:
        args += ["-v", f"{ca_file}:{CONTAINER_CA_PATH}:ro"]
    args += [FRPC_IMAGE, "-c", CONTAINER_CONFIG_PATH]
    return args


def _container_exit_summary(secrets_to_redact: list[str]) -> str:
    """Exit code plus the last frpc log lines, trimmed and redacted."""
    exit_code: Any = "unknown"
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.ExitCode}}", CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            exit_code = result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    logs = ""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(FAILURE_LOG_LINES), CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=10,
        )
        logs = (result.stdout or "") + (result.stderr or "")
    except (subprocess.TimeoutExpired, OSError):
        pass
    return _redact(f"frpc exited (code {exit_code}): {logs.strip()}", secrets_to_redact)


def _redact(text: str, secrets_to_redact: list[str]) -> str:
    text = _ANSI_ESCAPE.sub("", text)
    for value in secrets_to_redact:
        if value:
            text = text.replace(value, "***")
    if len(text) > FAILURE_MESSAGE_MAX_CHARS:
        text = "…" + text[-(FAILURE_MESSAGE_MAX_CHARS - 1) :]
    return text


class _BackendClient:
    """Raw HTTP to the edge remote-access endpoints (not in the public SDK)."""

    def __init__(self, *, base_url: str, token: str, edge_uuid: str, edge_key: str) -> None:
        self._prefix = f"{base_url.rstrip('/')}/api/v1/edges/{edge_uuid}/remote-access"
        self.edge_key = edge_key
        self._headers = {
            "Authorization": f"Bearer {token}",
            EDGE_KEY_HEADER: edge_key,
        }

    def _request(
        self, method: str, path: str, *, json_body: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        import httpx  # noqa: PLC0415

        try:
            response = httpx.request(
                method,
                f"{self._prefix}{path}",
                headers=self._headers,
                json=json_body,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            raise RemoteAccessHTTPError(None, f"{type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            detail = ""
            try:
                detail = str(response.json().get("detail", ""))
            except Exception:
                detail = response.text[:200]
            raise RemoteAccessHTTPError(response.status_code, detail)
        try:
            data = response.json()
        except ValueError as exc:
            raise RemoteAccessHTTPError(response.status_code, "invalid JSON response") from exc
        if not isinstance(data, dict):
            raise RemoteAccessHTTPError(response.status_code, "unexpected response shape")
        return data

    def claim_key(self) -> dict[str, Any]:
        return self._request("POST", "/key")

    def fetch_credential(
        self, session_id: str, host_keys: list[dict[str, str]], ssh_user: str = ""
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"host_keys": host_keys}
        # Omitted rather than sent empty, so an older backend is unaffected.
        if ssh_user:
            body["ssh_user"] = ssh_user
        return self._request("POST", f"/sessions/{session_id}/credential", json_body=body)

    def session_state(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/sessions/{session_id}/state")

    def report_failure(self, session_id: str, error: str) -> None:
        self._request("POST", f"/sessions/{session_id}/failure", json_body={"error": error})


@dataclass
class _Tunnel:
    session_id: str
    token: str
    stop_event: threading.Event = field(default_factory=threading.Event)


class RemoteAccessManager:
    """Owns the single frpc container; every transition runs under one lock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tunnel: Optional[_Tunnel] = None
        self._edge_uuid: Optional[str] = None
        self._key_claimed = False
        self._initialized = False

    # -- setup -------------------------------------------------------------

    def on_edge_resolved(self, edge_uuid: str) -> None:
        """Clear any tunnel left by a previous process, claim the key, resume its session."""
        with self._lock:
            self._edge_uuid = edge_uuid
            if self._initialized:
                return
            self._initialized = True
            self._remove_container_and_config()
        threading.Thread(
            target=self._claim_key_and_resume,
            name="remote-access-key-claim",
            daemon=True,
        ).start()

    def _claim_key_and_resume(self) -> None:
        try:
            self.ensure_key_claimed()
        except Exception:
            logger.exception("Remote access key claim failed")
        try:
            self._resume_saved_session()
        except Exception:
            logger.exception("Remote access session resume failed")

    def _resume_saved_session(self) -> None:
        """Reopen the tunnel a crashed or restarted edge-core left while still live."""
        session_id = load_saved_session()
        if session_id is None:
            return
        client = self._client()
        if client is None:
            return
        try:
            state = client.session_state(session_id)
        except RemoteAccessHTTPError as exc:
            logger.warning(
                "Remote access: state of saved session %s unavailable (%s): %s",
                session_id,
                exc.status_code,
                exc,
            )
            if exc.status_code in (403, 404, 409):
                forget_session()
            return
        if not state.get("live"):
            logger.info(
                "Remote access: saved session %s is %s; not resuming",
                session_id,
                state.get("status") or "not live",
            )
            forget_session()
            return
        with self._lock:
            if self._tunnel is not None:
                return
            logger.info("Remote access: resuming tunnel for session %s", session_id)
            self.open_tunnel(session_id)

    def _client(self) -> Optional[_BackendClient]:
        startup = _startup()
        token = startup.load_token()
        if not token:
            logger.warning("Remote access: no API token available")
            return None
        edge_uuid = self._edge_uuid or startup._resolve_edge_uuid(token)
        if not edge_uuid:
            logger.warning("Remote access: edge UUID unavailable")
            return None
        self._edge_uuid = edge_uuid
        base_url = (
            startup.get_runtime_env_var("CYBERWAVE_BASE_URL", startup.DEFAULT_API_URL)
            or startup.DEFAULT_API_URL
        )
        return _BackendClient(
            base_url=base_url,
            token=token,
            edge_uuid=edge_uuid,
            edge_key=load_or_create_edge_key(),
        )

    def ensure_key_claimed(self, client: Optional[_BackendClient] = None) -> bool:
        if self._key_claimed:
            return True
        client = client or self._client()
        if client is None:
            return False
        try:
            client.claim_key()
        except RemoteAccessHTTPError as exc:
            if exc.status_code == 409:
                logger.error(
                    "Remote access: this edge's backend record already holds a different "
                    "key (reinstalled device?). A workspace ADMIN must reset the remote "
                    "access key for this edge in the Cyberwave UI; the local key at %s "
                    "is kept.",
                    edge_key_path(),
                )
            elif exc.status_code == 400:
                logger.error("Remote access: backend rejected the edge key: %s", exc)
            else:
                logger.warning("Remote access: key claim failed (%s): %s", exc.status_code, exc)
            return False
        self._key_claimed = True
        logger.info("Remote access edge key claimed")
        return True

    # -- commands ------------------------------------------------------------

    def open_tunnel(self, session_id: str) -> None:
        with self._lock:
            current = self._tunnel
            if current is not None and current.session_id == session_id:
                if docker_container_status(CONTAINER_NAME) == "running":
                    logger.info("Remote access tunnel for session %s already running", session_id)
                    return
            if current is not None:
                if current.session_id != session_id:
                    logger.info(
                        "Remote access: stopping tunnel for session %s to open %s",
                        current.session_id,
                        session_id,
                    )
                self._stop_locked()

            client = self._client()
            if client is None or not self.ensure_key_claimed(client):
                logger.warning("Remote access: cannot open tunnel for session %s", session_id)
                forget_session()
                return

            try:
                credential = self._fetch_credential(client, session_id)
            except RemoteAccessHTTPError as exc:
                logger.warning(
                    "Remote access: credential for session %s refused (%s): %s",
                    session_id,
                    exc.status_code,
                    exc,
                )
                forget_session()
                return

            token = str(credential.get("token") or "")
            error = self._start_container(session_id, credential)
            if error:
                logger.warning("Remote access tunnel for session %s failed: %s", session_id, error)
                self._stop_locked()
                self._report_failure(client, session_id, _redact(error, [token]))
                return

            save_session(session_id)
            tunnel = _Tunnel(session_id=session_id, token=token)
            self._tunnel = tunnel
            threading.Thread(
                target=self._watch,
                args=(tunnel, client),
                name=f"remote-access-watchdog-{session_id[:8]}",
                daemon=True,
            ).start()
            logger.info("Remote access tunnel started for session %s", session_id)

    def close_tunnel(self, session_id: str) -> None:
        with self._lock:
            current = self._tunnel
            if current is None or current.session_id != session_id:
                logger.info("Remote access: no running tunnel for session %s to close", session_id)
                return
            self._stop_locked()
            logger.info("Remote access tunnel closed for session %s", session_id)

    def shutdown(self) -> None:
        """Stop frpc on edge-core exit, keeping the session file so the next start resumes."""
        with self._lock:
            current = self._tunnel
            if current is None:
                return
            self._stop_locked(keep_session=True)
        logger.info("Remote access tunnel for session %s stopped on shutdown", current.session_id)

    # -- internals -------------------------------------------------------------

    def _fetch_credential(self, client: _BackendClient, session_id: str) -> dict[str, Any]:
        """Fetch the relay credential, re-claiming the key once if the backend lost it.

        An ADMIN key reset answers 409, a key claimed by another process 403.
        """
        host_keys = read_ssh_host_keys()
        ssh_user = detect_ssh_login_user()
        try:
            return client.fetch_credential(session_id, host_keys, ssh_user)
        except RemoteAccessHTTPError as exc:
            if exc.status_code not in (403, 409):
                raise
            self._key_claimed = False
            if not self.ensure_key_claimed(client):
                raise
        return client.fetch_credential(session_id, host_keys, ssh_user)

    def _start_container(self, session_id: str, credential: dict[str, Any]) -> Optional[str]:
        if not docker_available():
            return "docker is not available on this edge"
        ca_file = relay_ca_file()
        if ca_file and not os.path.isfile(ca_file):
            return f"relay CA file {ca_file} not found"
        relay_host, relay_port = relay_address()
        trusted_ca = _relay_tls_ca(relay_host, ca_file)
        try:
            content = render_frpc_config(
                session_id=session_id,
                credential=credential,
                relay_host=_container_relay_host(relay_host),
                relay_port=relay_port,
                trusted_ca_file=trusted_ca,
                tls_server_name=relay_host if trusted_ca else None,
                local_ip="127.0.0.1" if _docker_host_network() else DOCKER_HOST_GATEWAY,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return f"invalid relay credential: {exc}"
        config_path = write_frpc_config(content)
        docker_rm(CONTAINER_NAME)
        try:
            result = subprocess.run(
                build_docker_run_args(config_path, ca_file),
                capture_output=True,
                text=True,
                timeout=DOCKER_RUN_TIMEOUT_SECONDS,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return f"docker run failed: {exc}"
        if result.returncode != 0:
            return f"docker run failed: {(result.stderr or result.stdout).strip()}"
        logger.info(
            "Remote access: frpc started for session %s (relay %s:%d, remote port %s)",
            session_id,
            relay_host,
            relay_port,
            credential.get("remote_port"),
        )
        return None

    def _stop_locked(self, *, keep_session: bool = False) -> None:
        if self._tunnel is not None:
            self._tunnel.stop_event.set()
            self._tunnel = None
        self._remove_container_and_config()
        if not keep_session:
            forget_session()

    def _remove_container_and_config(self) -> None:
        docker_rm(CONTAINER_NAME)
        remove_frpc_config()

    def _report_failure(self, client: _BackendClient, session_id: str, error: str) -> None:
        try:
            client.report_failure(session_id, error)
        except RemoteAccessHTTPError as exc:
            logger.warning(
                "Remote access: failure report for session %s failed (%s): %s",
                session_id,
                exc.status_code,
                exc,
            )

    def _watch(self, tunnel: _Tunnel, client: _BackendClient) -> None:
        """Stop frpc when it exits or the backend no longer reports the session live.

        Expiry is deliberately not tracked locally: a renewal only moves it
        server-side.
        """
        started = time.monotonic()
        next_state_poll = started + WATCHDOG_INTERVAL_SECONDS
        while True:
            in_grace = time.monotonic() - started < STARTUP_GRACE_SECONDS
            interval = STARTUP_POLL_SECONDS if in_grace else WATCHDOG_INTERVAL_SECONDS
            if tunnel.stop_event.wait(interval):
                return

            if docker_container_status(CONTAINER_NAME) in _EXITED_STATUSES:
                summary = _container_exit_summary([tunnel.token, client.edge_key])
                with self._lock:
                    if self._tunnel is not tunnel:
                        return
                    self._stop_locked()
                logger.warning(
                    "Remote access tunnel for session %s exited: %s", tunnel.session_id, summary
                )
                self._report_failure(client, tunnel.session_id, summary)
                return

            if time.monotonic() < next_state_poll:
                continue
            next_state_poll = time.monotonic() + WATCHDOG_INTERVAL_SECONDS
            reason = self._state_stop_reason(client, tunnel.session_id)
            if reason is None:
                continue
            with self._lock:
                if self._tunnel is not tunnel:
                    return
                self._stop_locked()
            logger.info(
                "Remote access tunnel for session %s stopped: %s", tunnel.session_id, reason
            )
            return

    @staticmethod
    def _state_stop_reason(client: _BackendClient, session_id: str) -> Optional[str]:
        try:
            state = client.session_state(session_id)
        except RemoteAccessHTTPError as exc:
            if exc.status_code in (403, 404, 409):
                return f"session state returned {exc.status_code}"
            logger.debug("Remote access state poll for session %s failed: %s", session_id, exc)
            return None
        if not state.get("live"):
            return f"session is {state.get('status') or 'not live'}"
        return None


_MANAGER = RemoteAccessManager()


def on_edge_resolved(edge_uuid: str) -> None:
    _MANAGER.on_edge_resolved(edge_uuid)


def shutdown() -> None:
    _MANAGER.shutdown()


def handle_command(command: str, payload: dict[str, Any]) -> None:
    """Dispatch an ``open_tunnel`` / ``close_tunnel`` MQTT command off-thread."""
    raw_session_id = str(payload.get("session_id") or "").strip()
    try:
        session_id = str(uuid.UUID(raw_session_id))
    except ValueError:
        logger.warning("Ignoring %s with invalid session_id %r", command, raw_session_id)
        return
    target = _MANAGER.open_tunnel if command == COMMAND_OPEN_TUNNEL else _MANAGER.close_tunnel
    logger.info("Received remote access command %s for session %s", command, session_id)
    threading.Thread(
        target=_run_command,
        args=(target, command, session_id),
        name=f"remote-access-{command}-{session_id[:8]}",
        daemon=True,
    ).start()


def _run_command(target: Any, command: str, session_id: str) -> None:
    try:
        target(session_id)
    except Exception:
        logger.exception("Remote access %s for session %s failed", command, session_id)
