"""Least-privilege credentials for driver containers.

Historically every driver container received the edge's own API token (the
pairing user's full credential) as ``CYBERWAVE_API_KEY``. For a driver that runs
scoped (below), Edge Core instead asks the backend for a *driver token* bound to
the one twin the container drives (plus its child twins)::

    POST /api/v1/twins/{twin_uuid}/driver-token
         {"child_twin_uuids": [...], "edge_fingerprint": "..."}
    POST /api/v1/twins/{twin_uuid}/driver-token/{token_uuid}/renew

and hands that to the driver instead. What the token may reach is decided and
enforced by the backend; this module only obtains, caches and renews it.

* The token is cached per twin in ``<config dir>/driver_tokens/<twin>.json``
  (mode 0600) and reused across launches, so every service of a multi-service
  driver shares one token and a restart does not mint a new one.
* A token close to expiry is renewed (same secret, longer lease); the runtime
  loop also renews periodically, so a driver that runs for months is not cut
  off. A token the backend no longer recognises is replaced by a fresh one.
* A backend that predates the endpoint answers 404: the driver then receives
  the edge token as before, with a WARNING. Other failures fall back the same
  way so a backend hiccup never stops a robot from starting.

``CYBERWAVE_EDGE_DRIVER_TOKEN_MODE`` selects the behaviour:

* ``auto`` (default): scoped only for driver images that opt in with the
  ``com.cyberwave.driver-token.scoped`` label (the Vector Guardian does); every
  other driver keeps the edge token as before. A fleet-wide credential change
  should not ride along with one driver: drivers opt in once their topic needs
  are known to fit their twin's scope, and the default can flip after a soak.
* ``scoped``: as above for every driver.
* ``legacy``: always pass the edge token (previous behaviour).
* ``strict``: like ``scoped`` for every driver, but never fall back; the
  launch is refused when no driver token can be obtained.

A scoped container must not be able to read the edge token any other way, so
:func:`apply_to_container` also scrubs it from the environment and
:func:`mask_mount_args` hides ``credentials.json`` and the token cache behind the
read-only config-directory mount; a driver that runs scoped is not started when
that masking fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from cyberwave.mqtt_identity import mqtt_username_for_token

from .utils import atomic_write_text

logger = logging.getLogger(__name__)

MODE_ENV = "CYBERWAVE_EDGE_DRIVER_TOKEN_MODE"
MODE_AUTO = "auto"
MODE_SCOPED = "scoped"
MODE_LEGACY = "legacy"
MODE_STRICT = "strict"
DEFAULT_MODE = MODE_AUTO
_MODES = frozenset({MODE_AUTO, MODE_SCOPED, MODE_LEGACY, MODE_STRICT})
# Image label a driver sets ("true") to run with a twin-scoped token in auto mode.
SCOPED_TOKEN_LABEL = "com.cyberwave.driver-token.scoped"

TOKEN_DIR_NAME = "driver_tokens"
PUBLIC_CREDENTIALS_NAME = "driver_credentials.public.json"
CONTAINER_CONFIG_DIR = "/app/.cyberwave"

def _float_env(name: str, default: float) -> float:
    """A numeric knob that never costs the edge its drivers: bad values fall back."""
    raw = os.getenv(name, "")
    try:
        value = float(raw) if raw.strip() else default
    except ValueError:
        logger.warning("Ignoring %s=%r; using %s", name, raw, default)
        return default
    return value if value > 0 else default


# Renew once less than this is left on the lease.
REFRESH_BEFORE = timedelta(days=_float_env("CYBERWAVE_EDGE_DRIVER_TOKEN_REFRESH_DAYS", 7.0))
RENEW_CHECK_INTERVAL_SECONDS = _float_env(
    "CYBERWAVE_EDGE_DRIVER_TOKEN_RENEW_CHECK_SECONDS", 3600.0
)
# A driver launch must not wait long on an unreachable backend: the fallback
# credential is already in hand, launches are sequential, and the renewal loop
# mints a scoped token as soon as the backend answers again.
HTTP_TIMEOUT_SECONDS = _float_env("CYBERWAVE_EDGE_DRIVER_TOKEN_TIMEOUT_S", 2.5)

# Broker-credential overrides drivers honour ahead of CYBERWAVE_API_KEY (the
# Python SDK, the aerial kit and the C++ driver base each read some of them).
# In scoped mode they are all pinned to the driver token, so no client can fall
# back to an operator- or edge-supplied broker identity.
_MQTT_USERNAME_ENV_KEYS = ("CYBERWAVE_MQTT_USERNAME", "CYBERWAVE_MQTT_USER")
_MQTT_PASSWORD_ENV_KEYS = ("CYBERWAVE_MQTT_PASSWORD", "CYBERWAVE_MQTT_API_TOKEN")
# Substrings marking a credentials.json env value as secret: never copied into
# the public view a driver can read.
_SECRET_ENV_MARKERS = ("TOKEN", "KEY", "PASSWORD", "SECRET", "CREDENTIAL")

HttpPost = Callable[..., Any]

_warned_unsupported = False
_cache_lock = threading.Lock()
_last_renew_check = 0.0


@dataclass(frozen=True)
class DriverCredential:
    """The API key a driver container receives and whether it is scoped."""

    api_key: str
    scoped: bool
    token_uuid: Optional[str] = None
    expires_at: Optional[datetime] = None


class _UnsupportedError(Exception):
    """The backend has no driver-token endpoint (HTTP 404)."""


class _NotRenewableError(Exception):
    """The backend no longer recognises the cached token (HTTP 410/404)."""


def resolve_mode() -> str:
    raw = (os.getenv(MODE_ENV) or "").strip().lower()
    if not raw:
        return DEFAULT_MODE
    if raw not in _MODES:
        logger.warning("Unknown %s=%r; using %s", MODE_ENV, raw, DEFAULT_MODE)
        return DEFAULT_MODE
    return raw


def effective_mode(*, image_opts_in: bool, mode: Optional[str] = None) -> str:
    """The mode one driver launch runs in: ``auto`` resolves per image."""
    mode = mode or resolve_mode()
    if mode == MODE_AUTO:
        return MODE_SCOPED if image_opts_in else MODE_LEGACY
    return mode


def edge_rotation_key(fingerprint: Optional[str]) -> str:
    """Stable, opaque per-edge rotation key derived from the fingerprint."""
    if not fingerprint:
        return ""
    return "fp-" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def token_dir(config_dir: Path) -> Path:
    return Path(config_dir) / TOKEN_DIR_NAME


def _cache_path(config_dir: Path, twin_uuid: str) -> Path:
    safe = "".join(ch for ch in twin_uuid if ch.isalnum() or ch == "-")
    return token_dir(config_dir) / f"{safe}.json"


def _fingerprint_of(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _load_cache(config_dir: Path, twin_uuid: str) -> Optional[dict[str, Any]]:
    path = _cache_path(config_dir, twin_uuid)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("token") or not data.get("uuid"):
        return None
    return data


def _write_cache(config_dir: Path, twin_uuid: str, data: dict[str, Any]) -> None:
    directory = token_dir(config_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(directory, 0o700)
    atomic_write_text(_cache_path(config_dir, twin_uuid), json.dumps(data, indent=2))


def _drop_cache(config_dir: Path, twin_uuid: str) -> None:
    try:
        _cache_path(config_dir, twin_uuid).unlink()
    except OSError:
        pass


def _cache_matches(
    data: dict[str, Any],
    *,
    base_url: str,
    edge_token: str,
    child_twin_uuids: list[str],
) -> bool:
    """Whether a cached token was minted for this exact launch context.

    A different backend, a re-paired edge (new edge token) or a different set
    of child twins each need a token minted for them.
    """
    return (
        data.get("base_url") == base_url.rstrip("/")
        and data.get("edge_token_sha256") == _fingerprint_of(edge_token)
        and sorted(data.get("child_twin_uuids") or []) == sorted(child_twin_uuids)
    )


# --------------------------------------------------------------------------- #
# Backend calls
# --------------------------------------------------------------------------- #


def _default_post(url: str, *, json_body: dict[str, Any], token: str) -> Any:
    import httpx  # noqa: PLC0415 -- keep import cost off module load

    return httpx.post(
        url,
        json=json_body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=HTTP_TIMEOUT_SECONDS,
    )


def _endpoint(base_url: str, twin_uuid: str) -> str:
    return f"{base_url.rstrip('/')}/api/v1/twins/{twin_uuid}/driver-token"


def _mint(
    *,
    post: HttpPost,
    base_url: str,
    edge_token: str,
    twin_uuid: str,
    child_twin_uuids: list[str],
    edge_key: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {"child_twin_uuids": child_twin_uuids}
    if edge_key:
        body["edge_fingerprint"] = edge_key
    response = post(_endpoint(base_url, twin_uuid), json_body=body, token=edge_token)
    status = int(getattr(response, "status_code", 0))
    if status == 404:
        raise _UnsupportedError()
    if status not in (200, 201):
        raise RuntimeError(f"driver-token mint returned HTTP {status}")
    payload = response.json()
    if not isinstance(payload, dict) or not payload.get("token") or not payload.get("uuid"):
        raise RuntimeError("driver-token mint returned no token")
    return payload


def _renew(
    *, post: HttpPost, base_url: str, edge_token: str, twin_uuid: str, token_uuid: str
) -> dict[str, Any]:
    url = f"{_endpoint(base_url, twin_uuid)}/{token_uuid}/renew"
    response = post(url, json_body={}, token=edge_token)
    status = int(getattr(response, "status_code", 0))
    if status in (404, 410):
        raise _NotRenewableError()
    if status != 200:
        raise RuntimeError(f"driver-token renew returned HTTP {status}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("driver-token renew returned no body")
    return payload


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def _credential_from_cache(data: dict[str, Any]) -> DriverCredential:
    return DriverCredential(
        api_key=str(data["token"]),
        scoped=True,
        token_uuid=str(data["uuid"]),
        expires_at=_parse_ts(data.get("expires_at")),
    )


def _fallback(mode: str, edge_token: str, reason: str) -> Optional[DriverCredential]:
    if mode == MODE_STRICT:
        logger.error(
            "No twin-scoped driver token (%s) and %s=strict: refusing the launch",
            reason,
            MODE_ENV,
        )
        return None
    logger.warning(
        "Passing the edge's full API token to the driver container (%s). Set "
        "%s=legacy to silence this, or upgrade the backend for twin-scoped "
        "driver tokens.",
        reason,
        MODE_ENV,
    )
    return DriverCredential(api_key=edge_token, scoped=False)


def resolve_driver_credential(
    *,
    twin_uuid: str,
    edge_token: str,
    base_url: str,
    config_dir: Path,
    child_twin_uuids: Iterable[str] = (),
    edge_fingerprint: Optional[str] = None,
    post: Optional[HttpPost] = None,
    now: Optional[datetime] = None,
    mode: Optional[str] = None,
) -> Optional[DriverCredential]:
    """The credential a driver container for ``twin_uuid`` should receive.

    ``mode`` is this launch's :func:`effective_mode` (default: the configured
    mode, with ``auto`` treated as scoped). ``None`` only in strict mode, when
    the launch must be refused.
    """
    global _warned_unsupported
    mode = mode or resolve_mode()
    if mode == MODE_AUTO:
        mode = MODE_SCOPED
    if mode == MODE_LEGACY:
        return DriverCredential(api_key=edge_token, scoped=False)

    post = post or _default_post
    now = now or datetime.now(timezone.utc)
    children = sorted({str(c).strip() for c in child_twin_uuids if str(c).strip()} - {twin_uuid})
    base = base_url.rstrip("/")

    with _cache_lock:
        cached = _load_cache(config_dir, twin_uuid)
        if cached is not None and not _cache_matches(
            cached, base_url=base, edge_token=edge_token, child_twin_uuids=children
        ):
            cached = None
        if cached is not None:
            credential = _credential_from_cache(cached)
            expires_at = credential.expires_at
            if expires_at is not None and expires_at - now > REFRESH_BEFORE:
                return credential
            try:
                renewed = _renew(
                    post=post,
                    base_url=base,
                    edge_token=edge_token,
                    twin_uuid=twin_uuid,
                    token_uuid=str(cached["uuid"]),
                )
                cached["expires_at"] = renewed.get("expires_at")
                _write_cache(config_dir, twin_uuid, cached)
                return _credential_from_cache(cached)
            except _NotRenewableError:
                logger.info("Cached driver token for %s is no longer valid", twin_uuid[:8])
                _drop_cache(config_dir, twin_uuid)
            except Exception as exc:
                if expires_at is not None and expires_at > now:
                    logger.warning(
                        "Could not renew driver token for %s (%s); using it until %s",
                        twin_uuid[:8],
                        exc,
                        expires_at.isoformat(),
                    )
                    return credential

        try:
            payload = _mint(
                post=post,
                base_url=base,
                edge_token=edge_token,
                twin_uuid=twin_uuid,
                child_twin_uuids=children,
                edge_key=edge_rotation_key(edge_fingerprint),
            )
        except _UnsupportedError:
            if not _warned_unsupported:
                _warned_unsupported = True
                logger.warning("Backend has no driver-token endpoint (HTTP 404)")
            return _fallback(mode, edge_token, "backend does not support driver tokens")
        except Exception as exc:
            return _fallback(mode, edge_token, f"driver-token request failed: {exc}")

        data = {
            "token": payload["token"],
            "uuid": payload["uuid"],
            "expires_at": payload.get("expires_at"),
            "twin_uuid": twin_uuid,
            "child_twin_uuids": children,
            "base_url": base,
            "edge_token_sha256": _fingerprint_of(edge_token),
        }
        try:
            _write_cache(config_dir, twin_uuid, data)
        except OSError as exc:
            logger.warning("Could not cache driver token for %s: %s", twin_uuid[:8], exc)
        logger.info(
            "Using twin-scoped driver token %s for %s (expires %s)",
            str(payload["uuid"])[:8],
            twin_uuid[:8],
            payload.get("expires_at"),
        )
        return _credential_from_cache(data)


def renew_cached_driver_tokens(
    *,
    twin_uuids: Iterable[str],
    edge_token: str,
    base_url: str,
    config_dir: Path,
    post: Optional[HttpPost] = None,
    now: Optional[datetime] = None,
) -> int:
    """Renew cached tokens of running drivers whose lease is running out.

    Same secret, longer lease: the running containers keep working. Returns
    the number renewed. Tokens the backend rejects are dropped from the cache
    so the next launch mints a fresh one.
    """
    if resolve_mode() == MODE_LEGACY:
        return 0
    post = post or _default_post
    now = now or datetime.now(timezone.utc)
    base = base_url.rstrip("/")
    renewed = 0
    for twin_uuid in sorted(set(twin_uuids)):
        with _cache_lock:
            cached = _load_cache(config_dir, twin_uuid)
            if cached is None or cached.get("edge_token_sha256") != _fingerprint_of(edge_token):
                continue
            expires_at = _parse_ts(cached.get("expires_at"))
            if expires_at is not None and expires_at - now > REFRESH_BEFORE:
                continue
            try:
                payload = _renew(
                    post=post,
                    base_url=base,
                    edge_token=edge_token,
                    twin_uuid=twin_uuid,
                    token_uuid=str(cached["uuid"]),
                )
            except _NotRenewableError:
                logger.warning(
                    "Driver token for %s was revoked or expired; the next launch mints a new one",
                    twin_uuid[:8],
                )
                _drop_cache(config_dir, twin_uuid)
                continue
            except Exception as exc:
                logger.warning("Could not renew driver token for %s: %s", twin_uuid[:8], exc)
                continue
            cached["expires_at"] = payload.get("expires_at")
            _write_cache(config_dir, twin_uuid, cached)
            renewed += 1
    return renewed


def maybe_renew_driver_tokens(
    *,
    twin_uuids: Callable[[], Iterable[str]],
    edge_token: Callable[[], Optional[str]],
    base_url: Callable[[], str],
    config_dir: Path,
) -> int:
    """Throttled :func:`renew_cached_driver_tokens` for the runtime loop.

    Inputs are callables so a tick that is not due costs nothing.
    """
    global _last_renew_check
    if resolve_mode() == MODE_LEGACY:
        return 0
    now = time.monotonic()
    if _last_renew_check and now - _last_renew_check < RENEW_CHECK_INTERVAL_SECONDS:
        return 0
    _last_renew_check = now
    token = edge_token()
    if not token:
        return 0
    return renew_cached_driver_tokens(
        twin_uuids=twin_uuids(),
        edge_token=token,
        base_url=base_url(),
        config_dir=config_dir,
    )


# --------------------------------------------------------------------------- #
# Container hardening
# --------------------------------------------------------------------------- #


def apply_to_container(
    container_env: dict[str, str], credential: DriverCredential, edge_token: str
) -> None:
    """Install ``credential`` in ``container_env`` (in place).

    For a scoped credential, also remove every way the edge token could still
    reach the driver (any value equal to it) and pin the MQTT credential
    overrides to the driver token. The backend requires a scoped token to
    connect as ``cwh_<sha256(token)>``; setting it here covers clients that do
    not derive it themselves.
    """
    container_env["CYBERWAVE_API_KEY"] = credential.api_key
    if not credential.scoped:
        return
    for key in list(container_env):
        if key != "CYBERWAVE_API_KEY" and edge_token and container_env[key] == edge_token:
            del container_env[key]
    username = mqtt_username_for_token(credential.api_key)
    for key in _MQTT_USERNAME_ENV_KEYS:
        container_env[key] = username
    for key in _MQTT_PASSWORD_ENV_KEYS:
        container_env[key] = credential.api_key


def _public_envs(credentials_file: Path) -> dict[str, str]:
    try:
        data = json.loads(credentials_file.read_text())
    except (OSError, ValueError):
        return {}
    envs = data.get("envs") if isinstance(data, dict) else None
    if not isinstance(envs, dict):
        return {}
    return {
        key: value
        for key, value in envs.items()
        if isinstance(key, str)
        and isinstance(value, str)
        and not any(marker in key.upper() for marker in _SECRET_ENV_MARKERS)
    }


def mask_mount_args(config_dir: Path) -> list[str]:
    """Docker args hiding the edge's secrets behind the config-dir mount.

    ``credentials.json`` is replaced by a copy holding only its non-secret
    ``envs`` (some drivers read their audio/video settings from there) and the
    driver-token cache is shadowed by an empty tmpfs, so one driver cannot read
    another twin's token.
    """
    config_dir = Path(config_dir)
    public_file = config_dir / PUBLIC_CREDENTIALS_NAME
    envs = _public_envs(config_dir / "credentials.json")
    atomic_write_text(public_file, json.dumps({"envs": envs}, indent=2), mode=0o644)
    return [
        "-v",
        f"{public_file}:{CONTAINER_CONFIG_DIR}/credentials.json:ro",
        "--tmpfs",
        f"{CONTAINER_CONFIG_DIR}/{TOKEN_DIR_NAME}:ro,size=64k",
    ]
