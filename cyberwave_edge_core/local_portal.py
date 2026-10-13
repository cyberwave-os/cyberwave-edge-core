"""Edge local portal: an opt-in, read-only LAN index of the pages drivers serve.

Some drivers serve a page of their own on the edge's network (a plan download
page for a ground-station browser, a camera preview, a status page) and
declare it in their ``cw-driver.yml``::

    local_http:
      - name: plans
        port: 8765
        path: /
        service: gateway          # multi-service drivers: which container serves it
        audience: lan             # "host" entries are never listed here
        description: Flight plans for the ground station

The backend compiles that onto the driver profile
(``drivers.<profile>.local_http``), which reaches the edge inside the twin JSON
Edge Core already writes to its config directory. With
``CYBERWAVE_EDGE_LOCAL_PORTAL=on`` this module serves, on
``CYBERWAVE_EDGE_LOCAL_PORTAL_BIND`` (default ``0.0.0.0:8700``):

* ``GET /``             an HTML page linking to every declared page of every
                        running driver;
* ``GET /portal.json``  the same listing, machine-readable;
* ``GET /healthz``      liveness of the portal itself.

What it deliberately does NOT do (v1):

* proxy anything: links point straight at ``http://<edge-ip>:<port><path>``,
  so the portal adds an index and nothing else to the edge's attack surface;
* expose secrets: the listing is built from a fixed set of fields (twin name,
  registry id, page name/description/port/path, container health). Tokens,
  environment variables and the rest of the twin metadata never reach it;
* advertise over mDNS: no mDNS library is a dependency of Edge Core, and adding
  one for this is not worth it yet. Operators reach the portal by IP or by the
  host name their network already resolves.

Off by default. When on, it never blocks the runtime loop: docker and the twin
files are read by a background refresher and requests are answered from that
snapshot.
"""

from __future__ import annotations

import html
import ipaddress
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from .container_spec import SERVICE_LABEL, TWIN_LABEL
from .docker_helpers import DRIVER_CONTAINER_PREFIX, docker_ps_by_prefix
from .driver_selection import _matched_driver_profile
from .utils import is_driver_twin_json_file

logger = logging.getLogger(__name__)

ENV_ENABLED = "CYBERWAVE_EDGE_LOCAL_PORTAL"
ENV_BIND = "CYBERWAVE_EDGE_LOCAL_PORTAL_BIND"
ENV_HOST = "CYBERWAVE_EDGE_LOCAL_PORTAL_HOST"
ENV_REFRESH = "CYBERWAVE_EDGE_LOCAL_PORTAL_REFRESH_SECONDS"
DEFAULT_BIND = "0.0.0.0:8700"
DEFAULT_REFRESH_SECONDS = 10.0
SCHEMA = "cyberwave.edge.local_portal.v1"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_AUDIENCE_LAN = "lan"
_SCHEMES = frozenset({"http", "https"})
_SLUG_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")
# Same rule as the backend catalog: absolute path, RFC 3986 path characters,
# no query or fragment (so no link here can carry a token).
_PATH_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~!$&'()*+,;=:@%/"
)
_HOST_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")
_MAX_ENTRIES_PER_TWIN = 16
_MAX_TEXT = 200

#: ``{twin uuid: {service: docker status}}``; a single-service driver's service is ``""``.
ContainerLister = Callable[[], dict[str, dict[str, str]]]


# -- configuration -----------------------------------------------------------


def portal_enabled(env: Mapping[str, str] = os.environ) -> bool:
    """True only when ``CYBERWAVE_EDGE_LOCAL_PORTAL`` is explicitly switched on."""
    return env.get(ENV_ENABLED, "").strip().lower() in _TRUE_VALUES


def parse_bind(value: Optional[str]) -> tuple[str, int]:
    """Parse ``host:port``, ``[v6]:port``, ``:port`` or ``port`` into ``(host, port)``.

    An empty value means :data:`DEFAULT_BIND`. A bare port binds every IPv4
    interface, like the default.
    """
    text = (value or "").strip() or DEFAULT_BIND
    if text.startswith("["):
        host, sep, port_text = text[1:].partition("]:")
        if not sep:
            raise ValueError(f"invalid bind address {value!r}")
    elif ":" in text:
        host, _, port_text = text.rpartition(":")
    else:
        host, port_text = "", text
    host = host or "0.0.0.0"
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError(f"invalid port in bind address {value!r}") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"port out of range in bind address {value!r}")
    if host != "localhost":
        try:
            ipaddress.ip_address(host)
        except ValueError:
            raise ValueError(
                f"bind host must be an IP address or localhost (got {host!r})"
            ) from None
    return host, port


def _refresh_seconds(env: Mapping[str, str]) -> float:
    try:
        return max(1.0, float(env.get(ENV_REFRESH, "") or DEFAULT_REFRESH_SECONDS))
    except ValueError:
        return DEFAULT_REFRESH_SECONDS


def _clean_advertised_host(value: Optional[str]) -> Optional[str]:
    """An operator-set host name or IP for links, or ``None`` when unusable."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        address = ipaddress.ip_address(text.strip("[]"))
    except ValueError:
        if len(text) <= 253 and set(text) <= _HOST_CHARS and not text.startswith("-"):
            return text
        logger.warning("Ignoring invalid %s=%r", ENV_HOST, value)
        return None
    return f"[{address}]" if address.version == 6 else str(address)


# -- reading declarations ----------------------------------------------------


def _slug(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > 63 or text[0] in "-_" or not set(text) <= _SLUG_CHARS:
        return None
    return text


def _short_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text[:_MAX_TEXT] or None


def normalize_local_http_entry(raw: Any) -> Optional[dict[str, Any]]:
    """One ``local_http`` entry reduced to the fields the portal may show.

    The backend validates declarations at seed time, but a twin's metadata is
    editable after that, so the edge re-checks rather than trusting it. Entries
    that are invalid, or not meant for the LAN, return ``None``. Only the keys
    listed here can ever reach the portal output.
    """
    if not isinstance(raw, dict):
        return None
    name = _slug(raw.get("name"))
    port = raw.get("port")
    if name is None or isinstance(port, bool) or not isinstance(port, int):
        return None
    if not 1 <= port <= 65535:
        return None
    path = raw.get("path", "/")
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or path.startswith("//")
        or len(path) > 512
        or not set(path) <= _PATH_CHARS
        or "/../" in f"{path}/"
    ):
        return None
    if raw.get("audience", _AUDIENCE_LAN) != _AUDIENCE_LAN:
        return None
    scheme = raw.get("scheme", "http")
    if scheme not in _SCHEMES:
        return None
    service = raw.get("service")
    if service is not None:
        service = _slug(service)
        if service is None:
            return None
    entry: dict[str, Any] = {"name": name, "port": port, "path": path, "scheme": scheme}
    if service:
        entry["service"] = service
    description = _short_text(raw.get("description"))
    if description:
        entry["description"] = description
    return entry


def _attach_to_twin_uuid(twin_json: dict[str, Any]) -> Optional[str]:
    """The parent twin a twin JSON is attached to, read like Edge Core's launcher."""
    attach_to = twin_json.get("attach_to_twin_uuid")
    metadata = twin_json.get("metadata")
    if not attach_to and isinstance(metadata, dict):
        attach_to = metadata.get("attach_to_twin_uuid")
    return str(attach_to) if attach_to else None


def _asset_registry_id(twin_json: dict[str, Any]) -> Optional[str]:
    asset = twin_json.get("asset")
    if not isinstance(asset, dict):
        return None
    registry_id = asset.get("registry_id")
    asset_metadata = asset.get("metadata")
    if not registry_id and isinstance(asset_metadata, dict):
        registry_id = asset_metadata.get("registry_id")
    return str(registry_id).strip() if registry_id else None


def _child_registry_ids_by_parent(twins: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    """``{parent twin uuid: registry ids of the twins attached to it}``.

    The input to the launcher's profile resolution: a parent whose drivers
    declare a profile per child registry id launches the one its children match.
    """
    by_parent: dict[str, set[str]] = {}
    for twin_json in twins.values():
        parent = _attach_to_twin_uuid(twin_json)
        registry_id = _asset_registry_id(twin_json)
        if parent and registry_id:
            by_parent.setdefault(parent, set()).add(registry_id)
    return by_parent


def local_http_for_twin(
    twin_json: dict[str, Any], *, child_registry_ids: Optional[set[str]] = None
) -> list[dict[str, Any]]:
    """LAN pages a twin's driver declares, from its twin JSON file.

    Same precedence Edge Core uses to launch the driver: twin
    ``metadata.drivers`` wins over the asset's, and within it the profile
    :func:`~.driver_selection._matched_driver_profile` resolves for this host
    and the twin's attached children; a profile without ``local_http`` falls
    back to ``default``'s.
    """
    metadata = twin_json.get("metadata")
    asset = twin_json.get("asset")
    asset_metadata = asset.get("metadata") if isinstance(asset, dict) else None
    drivers = metadata.get("drivers") if isinstance(metadata, dict) else None
    if not drivers and isinstance(asset_metadata, dict):
        drivers = asset_metadata.get("drivers")
    if not isinstance(drivers, dict) or not drivers:
        return []
    _key, profile = _matched_driver_profile(drivers, child_registry_ids)
    raw_entries = profile.get("local_http") if isinstance(profile, dict) else None
    if raw_entries is None:
        default = drivers.get("default")
        raw_entries = default.get("local_http") if isinstance(default, dict) else None
    if not isinstance(raw_entries, list):
        return []
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_entries[:_MAX_ENTRIES_PER_TWIN]:
        entry = normalize_local_http_entry(raw)
        if entry is None or entry["name"] in seen:
            continue
        seen.add(entry["name"])
        entries.append(entry)
    return entries


def _health_from_status(status: str) -> str:
    """``docker ps`` Status column -> healthy / unhealthy / starting / running."""
    lowered = status.lower()
    if "(unhealthy)" in lowered:
        return "unhealthy"
    if "(healthy)" in lowered:
        return "healthy"
    if "health: starting" in lowered:
        return "starting"
    return "running"


def list_running_driver_containers() -> dict[str, dict[str, str]]:
    """``{twin uuid: {service: docker status}}`` for Edge Core's running driver containers.

    Ownership comes from the ``io.cyberwave.edge.twin`` / ``io.cyberwave.edge.service``
    labels Edge Core puts on every driver container (service ``""`` for a
    single-service driver), not from the container name, whose 8-character twin
    prefix is not unique. A container without the twin label is not listed.
    """
    fmt = "\t".join(
        ("{{.Status}}", '{{.Label "' + TWIN_LABEL + '"}}', '{{.Label "' + SERVICE_LABEL + '"}}')
    )
    containers: dict[str, dict[str, str]] = {}
    for line in docker_ps_by_prefix(DRIVER_CONTAINER_PREFIX, fmt=fmt):
        status, twin_uuid, service = (line.split("\t") + ["", ""])[:3]
        if twin_uuid.strip():
            containers.setdefault(twin_uuid.strip(), {})[service.strip()] = status.strip()
    return containers


class _TwinFileCache:
    """Parsed twin JSON files, re-read only when their mtime changes."""

    def __init__(self) -> None:
        self._entries: dict[Path, tuple[float, Optional[dict[str, Any]]]] = {}

    def load(self, path: Path) -> Optional[dict[str, Any]]:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            self._entries.pop(path, None)
            return None
        cached = self._entries.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict):
            data = None
        self._entries[path] = (mtime, data)
        return data

    def prune(self, keep: set[Path]) -> None:
        for path in list(self._entries):
            if path not in keep:
                del self._entries[path]


def _twin_json_files(config_dir: Path) -> list[Path]:
    try:
        return sorted(path for path in config_dir.glob("*.json") if is_driver_twin_json_file(path))
    except OSError:
        return []


def build_listing(
    config_dir: Path,
    containers: dict[str, str],
    *,
    cache: Optional[_TwinFileCache] = None,
) -> list[dict[str, Any]]:
    """Running drivers with declared LAN pages, host-independent.

    A driver is listed while at least one of its containers is running. A page
    whose ``service`` container is not running stays listed with
    ``health: "stopped"`` and no link.
    """
    cache = cache or _TwinFileCache()
    files = _twin_json_files(config_dir)
    cache.prune(set(files))
    twins = {path.stem: twin_json for path in files if (twin_json := cache.load(path)) is not None}
    child_registry_ids = _child_registry_ids_by_parent(twins)
    drivers: list[dict[str, Any]] = []
    for twin_uuid, twin_json in twins.items():
        own = containers.get(twin_uuid)
        if not own:
            continue
        entries = local_http_for_twin(
            twin_json, child_registry_ids=child_registry_ids.get(twin_uuid)
        )
        if not entries:
            continue
        pages = []
        for entry in entries:
            service = entry.get("service")
            if service:
                status = own.get(service)
            else:
                status = own.get("") or next(iter(own.values()))
            pages.append(
                {
                    **entry,
                    "health": _health_from_status(status) if status is not None else "stopped",
                }
            )
        healths = {page["health"] for page in pages}
        drivers.append(
            {
                "id": twin_uuid[:8],
                "twin_name": _short_text(twin_json.get("name")) or twin_uuid[:8],
                "registry_id": _short_text(_asset_registry_id(twin_json)),
                "health": _overall_health(healths),
                "pages": pages,
            }
        )
    drivers.sort(key=lambda driver: (driver["twin_name"].lower(), driver["id"]))
    return drivers


def _overall_health(healths: set[str]) -> str:
    for state in ("unhealthy", "stopped", "starting", "running"):
        if state in healths:
            return state
    return "healthy"


def render_portal_json(
    drivers: list[dict[str, Any]], host: str, *, generated_at: str
) -> dict[str, Any]:
    """The ``/portal.json`` document: listing plus a link per running page."""
    out_drivers = []
    for driver in drivers:
        pages = []
        for page in driver["pages"]:
            item = {
                key: page[key]
                for key in ("name", "description", "port", "path", "service", "health")
                if key in page
            }
            item["url"] = (
                f"{page['scheme']}://{host}:{page['port']}{page['path']}"
                if page["health"] != "stopped"
                else None
            )
            pages.append(item)
        out_drivers.append({**driver, "pages": pages})
    return {"schema": SCHEMA, "generated_at": generated_at, "host": host, "drivers": out_drivers}


_STYLE = (
    "body{font-family:system-ui,sans-serif;margin:0 auto;max-width:48rem;padding:1rem;"
    "color:#1b1b1f;background:#fff}h1{font-size:1.4rem}section{border:1px solid #ccc;"
    "border-radius:6px;padding:.75rem 1rem;margin:.75rem 0}h2{font-size:1.1rem;margin:0}"
    ".meta{color:#555;font-size:.85rem}li{margin:.4rem 0}.h{font-size:.8rem;padding:0 .4rem;"
    "border-radius:4px;background:#eee}.healthy{background:#d8f5dd}.unhealthy,.stopped"
    "{background:#fbdada}a{font-weight:600}"
    "@media(prefers-color-scheme:dark){body{background:#151518;color:#eee}section{border-color:#444}"
    ".meta{color:#aaa}.h{background:#333}.healthy{background:#1f4d29}.unhealthy,.stopped"
    "{background:#5c2020}a{color:#8ab4ff}}"
)


def render_portal_html(document: dict[str, Any]) -> str:
    """Static HTML for ``/``. Every value is escaped; no scripts, no forms."""
    esc = html.escape
    parts = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        "<meta name=viewport content='width=device-width,initial-scale=1'>",
        f"<title>Edge local pages</title><style>{_STYLE}</style></head><body>",
        "<h1>Edge local pages</h1>",
        f"<p class=meta>Pages served by drivers on this edge ({esc(document['host'])}). "
        f"Updated {esc(document['generated_at'])}. "
        "<a href=/portal.json>portal.json</a></p>",
    ]
    if not document["drivers"]:
        parts.append("<p>No running driver declares a local page.</p>")
    for driver in document["drivers"]:
        registry = f" &middot; {esc(driver['registry_id'])}" if driver.get("registry_id") else ""
        parts.append(
            f"<section><h2>{esc(driver['twin_name'])}</h2>"
            f"<div class=meta>{esc(driver['id'])}{registry} &middot; "
            f"<span class='h {esc(driver['health'])}'>{esc(driver['health'])}</span></div><ul>"
        )
        for page in driver["pages"]:
            label = esc(page["name"])
            link = (
                f"<a href='{esc(page['url'])}' rel=noreferrer>{label}</a>" if page["url"] else label
            )
            description = f" &mdash; {esc(page['description'])}" if page.get("description") else ""
            parts.append(
                f"<li>{link}{description} <span class='h {esc(page['health'])}'>"
                f"{esc(page['health'])}</span></li>"
            )
        parts.append("</ul></section>")
    parts.append("</body></html>")
    return "".join(parts)


# -- server ------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class LocalPortal:
    """The portal server and its background refresher.

    ``container_lister`` and ``clock`` exist for tests; production uses
    ``docker ps`` and wall time.
    """

    def __init__(
        self,
        *,
        config_dir: Path,
        bind: tuple[str, int],
        advertised_host: Optional[str] = None,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
        container_lister: Optional[ContainerLister] = None,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.bind = bind
        self.advertised_host = _clean_advertised_host(advertised_host)
        self.refresh_seconds = refresh_seconds
        self._container_lister = container_lister or list_running_driver_containers
        self._cache = _TwinFileCache()
        self._lock = threading.Lock()
        self._drivers: list[dict[str, Any]] = []
        self._generated_at: Optional[str] = None
        self._refreshed_monotonic: Optional[float] = None
        self._stop = threading.Event()
        self._server: Optional[ThreadingHTTPServer] = None
        self._threads: list[threading.Thread] = []

    # Snapshot -------------------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the listing. Called by the refresher; safe to call directly."""
        try:
            drivers = build_listing(self.config_dir, self._container_lister(), cache=self._cache)
        except Exception:
            logger.debug("Local portal refresh failed", exc_info=True)
            return
        with self._lock:
            self._drivers = drivers
            self._generated_at = _utc_now_iso()
            self._refreshed_monotonic = time.monotonic()

    def snapshot(self) -> tuple[list[dict[str, Any]], str, Optional[float]]:
        with self._lock:
            age = (
                None
                if self._refreshed_monotonic is None
                else time.monotonic() - self._refreshed_monotonic
            )
            return self._drivers, self._generated_at or _utc_now_iso(), age

    def document(self, local_address: str) -> dict[str, Any]:
        """``/portal.json`` as seen by a client that reached us on *local_address*."""
        drivers, generated_at, _age = self.snapshot()
        return render_portal_json(
            drivers, self._link_host(local_address), generated_at=generated_at
        )

    def _link_host(self, local_address: str) -> str:
        # The address the client actually reached us on, never the request's
        # Host header: a client-controlled value must not end up in links.
        if self.advertised_host:
            return self.advertised_host
        try:
            address = ipaddress.ip_address(local_address.split("%", 1)[0])
        except ValueError:
            return "localhost"
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return f"[{address}]" if address.version == 6 else str(address)

    # Lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Bind and serve in daemon threads. Raises ``OSError`` if the bind fails."""
        server = _server_class(self.bind[0])(self.bind, _make_handler(self))
        self._server = server
        self._stop.clear()
        serve = threading.Thread(target=server.serve_forever, name="edge-local-portal", daemon=True)
        refresher = threading.Thread(
            target=self._refresh_loop, name="edge-local-portal-refresh", daemon=True
        )
        self._threads = [serve, refresher]
        serve.start()
        refresher.start()

    @property
    def server_address(self) -> tuple[str, int]:
        if self._server is None:
            return self.bind
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(self.refresh_seconds)

    def stop(self) -> None:
        self._stop.set()
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join(timeout=2)
        self._threads = []


def _server_class(host: str) -> type[ThreadingHTTPServer]:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET

    class _PortalServer(ThreadingHTTPServer):
        address_family = family
        daemon_threads = True

    return _PortalServer


def _make_handler(portal: LocalPortal) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        server_version = "cyberwave-edge-portal"
        sys_version = ""
        timeout = 5  # a stalled client cannot hold a thread forever

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            self._respond(include_body=True)

        def do_HEAD(self) -> None:  # noqa: N802 - http.server API
            self._respond(include_body=False)

        def _respond(self, *, include_body: bool) -> None:
            route = self.path.split("?", 1)[0]
            if route in ("/", "/index.html"):
                document = portal.document(self.connection.getsockname()[0])
                self._send(
                    200, "text/html; charset=utf-8", render_portal_html(document), include_body
                )
            elif route == "/portal.json":
                document = portal.document(self.connection.getsockname()[0])
                self._send(200, "application/json", json.dumps(document, indent=2), include_body)
            elif route == "/healthz":
                drivers, _generated_at, age = portal.snapshot()
                body = {
                    "status": "ok",
                    "drivers": len(drivers),
                    "refreshed_seconds_ago": None if age is None else round(age, 1),
                }
                self._send(200, "application/json", json.dumps(body), include_body)
            else:
                self._send(
                    404, "application/json", json.dumps({"error": "not found"}), include_body
                )

        def _send(self, status: int, content_type: str, body: str, include_body: bool) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
            )
            self.end_headers()
            if include_body:
                self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.debug("local portal: " + format, *args)

    return _Handler


# -- module-level lifecycle (what startup.py calls) ---------------------------

_ACTIVE: Optional[LocalPortal] = None
_ACTIVE_LOCK = threading.Lock()


def start_from_env(
    *,
    config_dir: Path,
    env: Mapping[str, str] = os.environ,
) -> Optional[LocalPortal]:
    """Start the portal if ``CYBERWAVE_EDGE_LOCAL_PORTAL`` is on; otherwise do nothing.

    Never raises: a bad bind value or a busy port is logged and the edge keeps
    running without the portal.
    """
    global _ACTIVE
    if not portal_enabled(env):
        return None
    with _ACTIVE_LOCK:
        if _ACTIVE is not None:
            return _ACTIVE
        try:
            bind = parse_bind(env.get(ENV_BIND))
            portal = LocalPortal(
                config_dir=config_dir,
                bind=bind,
                advertised_host=env.get(ENV_HOST),
                refresh_seconds=_refresh_seconds(env),
            )
            portal.start()
        except (ValueError, OSError) as exc:
            logger.warning("Edge local portal not started: %s", exc)
            return None
        except Exception:
            logger.exception("Edge local portal not started")
            return None
        _ACTIVE = portal
    host, port = portal.server_address
    logger.info("Edge local portal serving read-only driver index on %s:%d", host, port)
    return portal


def shutdown() -> None:
    """Stop the portal if it is running. Safe to call when it never started."""
    global _ACTIVE
    with _ACTIVE_LOCK:
        portal, _ACTIVE = _ACTIVE, None
    if portal is not None:
        try:
            portal.stop()
        except Exception:
            logger.debug("Error stopping edge local portal", exc_info=True)
