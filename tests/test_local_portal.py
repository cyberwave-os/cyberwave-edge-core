"""Edge local portal: opt-in, read-only LAN index of driver-declared pages."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from cyberwave_edge_core import local_portal
from cyberwave_edge_core.container_spec import SERVICE_LABEL, TWIN_LABEL
from cyberwave_edge_core.docker_helpers import driver_container_name

TWIN_A = "11111111-2222-3333-4444-555555555555"
TWIN_B = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
TWIN_PLAIN = "99999999-8888-7777-6666-555555555555"
SECRET = "cw_live_SUPERSECRET_token"

PLANS = {
    "name": "plans",
    "port": 8765,
    "path": "/",
    "service": "gateway",
    "audience": "lan",
    "scheme": "http",
    "description": "Flight plans for the ground station",
}


def _twin(
    name: str,
    *,
    asset_drivers: dict[str, Any] | None = None,
    twin_drivers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"api_key": SECRET, "notes": SECRET}
    if twin_drivers is not None:
        metadata["drivers"] = twin_drivers
    return {
        "uuid": "ignored",
        "name": name,
        "metadata": metadata,
        "asset": {
            "registry_id": "acme/aircraft",
            "metadata": {"drivers": asset_drivers or {}, "secret": SECRET},
        },
    }


def _guardian_drivers(**extra: Any) -> dict[str, Any]:
    return {
        "default": {
            "docker_image": "acme/driver:1.0.0",
            "shared_env": {"CYBERWAVE_API_KEY": SECRET},
            "local_http": [PLANS],
            **extra,
        }
    }


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / f"{TWIN_A}.json").write_text(
        json.dumps(_twin("Guardian 1", asset_drivers=_guardian_drivers()))
    )
    (tmp_path / f"{TWIN_PLAIN}.json").write_text(
        json.dumps(_twin("Plain arm", asset_drivers={"default": {"docker_image": "x"}}))
    )
    # Never twin files: protected names and non-UUID stems.
    (tmp_path / "credentials.json").write_text(json.dumps({"token": SECRET}))
    (tmp_path / "notes.json").write_text(json.dumps(_twin("x", asset_drivers=_guardian_drivers())))
    return tmp_path


def _running(
    *containers: tuple[str, str], status: str = "Up 3 minutes (healthy)"
) -> dict[str, dict[str, str]]:
    """What the container lister reports for running ``(twin uuid, service)`` containers."""
    running: dict[str, dict[str, str]] = {}
    for twin_uuid, service in containers:
        running.setdefault(twin_uuid, {})[service] = status
    return running


# -- opt-in ------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "off", "0", "false", "no", "maybe"])
def test_portal_is_off_unless_explicitly_enabled(value: str | None) -> None:
    env = {} if value is None else {local_portal.ENV_ENABLED: value}
    assert local_portal.portal_enabled(env) is False


@pytest.mark.parametrize("value", ["on", "ON", "1", "true", "yes"])
def test_portal_enable_values(value: str) -> None:
    assert local_portal.portal_enabled({local_portal.ENV_ENABLED: value}) is True


def test_start_from_env_does_nothing_when_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("portal must not be constructed when disabled")

    monkeypatch.setattr(local_portal, "LocalPortal", _boom)
    assert local_portal.start_from_env(config_dir=tmp_path, env={}) is None
    assert local_portal._ACTIVE is None
    local_portal.shutdown()  # safe when never started


def test_runtime_loop_wiring_is_opt_in() -> None:
    source = Path(local_portal.__file__).with_name("startup.py").read_text()
    assert "local_portal.start_from_env(config_dir=CONFIG_DIR)" in source
    assert "local_portal.shutdown()" in source


# -- bind configuration ------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ("0.0.0.0", 8700)),
        ("", ("0.0.0.0", 8700)),
        ("127.0.0.1:8701", ("127.0.0.1", 8701)),
        ("192.168.1.20:80", ("192.168.1.20", 80)),
        (":9000", ("0.0.0.0", 9000)),
        ("9001", ("0.0.0.0", 9001)),
        ("[::1]:8700", ("::1", 8700)),
        ("[::]:8700", ("::", 8700)),
        ("localhost:8700", ("localhost", 8700)),
    ],
)
def test_parse_bind(value: str | None, expected: tuple[str, int]) -> None:
    assert local_portal.parse_bind(value) == expected


@pytest.mark.parametrize(
    "value", ["0.0.0.0:0", "0.0.0.0:70000", "0.0.0.0:http", "edge.example:80", "[::1]8700"]
)
def test_parse_bind_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        local_portal.parse_bind(value)


def test_bad_bind_is_logged_not_raised(tmp_path: Path) -> None:
    env = {local_portal.ENV_ENABLED: "on", local_portal.ENV_BIND: "nope:x"}
    assert local_portal.start_from_env(config_dir=tmp_path, env=env) is None


def test_busy_port_is_logged_not_raised(tmp_path: Path) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        env = {local_portal.ENV_ENABLED: "on", local_portal.ENV_BIND: f"127.0.0.1:{port}"}
        assert local_portal.start_from_env(config_dir=tmp_path, env=env) is None


# -- reading declarations ----------------------------------------------------


def test_twin_metadata_drivers_win_over_asset() -> None:
    twin = _twin(
        "t",
        asset_drivers=_guardian_drivers(),
        twin_drivers={"default": {"local_http": [{"name": "status", "port": 9000}]}},
    )
    assert [e["name"] for e in local_http_for(twin)] == ["status"]


def test_profile_without_local_http_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cyberwave_edge_core.driver_selection._platform_driver_keys", lambda: ["linux-x86_64"]
    )
    drivers = _guardian_drivers()
    drivers["linux-x86_64"] = {"docker_image": "acme/driver:1.0.0-amd64"}
    assert [e["name"] for e in local_http_for(_twin("t", asset_drivers=drivers))] == ["plans"]

    drivers["linux-x86_64"]["local_http"] = [{"name": "amd64-only", "port": 9100}]
    assert [e["name"] for e in local_http_for(_twin("t", asset_drivers=drivers))] == ["amd64-only"]


def local_http_for(twin: dict[str, Any]) -> list[dict[str, Any]]:
    return local_portal.local_http_for_twin(twin)


def test_entries_are_rechecked_and_reduced_to_known_fields() -> None:
    drivers = {
        "default": {
            "local_http": [
                {**PLANS, "token": SECRET, "headers": {"Authorization": SECRET}},
                {"name": "hidden", "port": 9000, "audience": "host"},
                {"name": "query", "port": 9001, "path": f"/?token={SECRET}"},
                {"name": "Bad Name", "port": 9002},
                {"name": "badport", "port": "9003"},
                {"name": "ftp", "port": 21, "scheme": "ftp"},
                {"name": "plans", "port": 9999},  # duplicate name
                "not-a-mapping",
            ]
        }
    }
    entries = local_http_for(_twin("t", asset_drivers=drivers))
    assert entries == [
        {
            "name": "plans",
            "port": 8765,
            "path": "/",
            "scheme": "http",
            "service": "gateway",
            "description": "Flight plans for the ground station",
        }
    ]


def test_twin_without_drivers_has_no_entries() -> None:
    assert local_http_for({"name": "t"}) == []
    assert local_http_for(_twin("t", asset_drivers={"default": {"local_http": "nope"}})) == []


def test_parent_profile_follows_attached_child_registry_ids(tmp_path: Path) -> None:
    """Same profile the launcher picks: a child's registry id selects its profile."""
    drivers = _guardian_drivers()
    drivers["acme/gimbal"] = {"local_http": [{"name": "gimbal", "port": 9200}]}
    (tmp_path / f"{TWIN_A}.json").write_text(json.dumps(_twin("Guardian", asset_drivers=drivers)))
    child = _twin("Gimbal")
    child["attach_to_twin_uuid"] = TWIN_A
    child["asset"]["registry_id"] = "acme/gimbal"
    (tmp_path / f"{TWIN_B}.json").write_text(json.dumps(child))

    [driver] = local_portal.build_listing(tmp_path, _running((TWIN_A, "")))
    assert [page["name"] for page in driver["pages"]] == ["gimbal"]

    assert [e["name"] for e in local_http_for(_twin("t", asset_drivers=drivers))] == ["plans"]


# -- listing -----------------------------------------------------------------


def test_container_lister_reads_ownership_from_edge_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling = TWIN_A[:8] + "-0000-0000-0000-000000000000"  # same 8-char name prefix
    lines = [
        f"Up 3 minutes (healthy)\t{TWIN_A}\tgateway",
        f"Up 1 minute\t{TWIN_A}\tvideo",
        f"Up 2 hours\t{sibling}",  # single-service: empty service label stripped
        "Up 5 minutes",  # no edge labels: not attributed to any twin
    ]
    calls: list[tuple[str, str]] = []

    def fake_ps(prefix: str, *, include_stopped: bool = False, fmt: str = "") -> list[str]:
        calls.append((prefix, fmt))
        return lines

    monkeypatch.setattr(local_portal, "docker_ps_by_prefix", fake_ps)
    assert local_portal.list_running_driver_containers() == {
        TWIN_A: {"gateway": "Up 3 minutes (healthy)", "video": "Up 1 minute"},
        sibling: {"": "Up 2 hours"},
    }
    [(prefix, fmt)] = calls
    assert driver_container_name(TWIN_A).startswith(prefix)
    assert fmt.split("\t") == [
        "{{.Status}}",
        f'{{{{.Label "{TWIN_LABEL}"}}}}',
        f'{{{{.Label "{SERVICE_LABEL}"}}}}',
    ]




def test_listing_includes_only_running_drivers_with_pages(config_dir: Path) -> None:
    assert local_portal.build_listing(config_dir, {}) == []

    containers = _running(
        (TWIN_A, "gateway"),
        (TWIN_A, "video"),
        (TWIN_PLAIN, ""),
    )
    [driver] = local_portal.build_listing(config_dir, containers)
    assert driver["id"] == TWIN_A[:8]
    assert driver["twin_name"] == "Guardian 1"
    assert driver["registry_id"] == "acme/aircraft"
    assert driver["health"] == "healthy"
    assert [page["name"] for page in driver["pages"]] == ["plans"]


def test_page_of_a_stopped_service_has_no_link(config_dir: Path) -> None:
    containers = _running((TWIN_A, "video"), status="Up 1 minute")
    [driver] = local_portal.build_listing(config_dir, containers)
    assert driver["pages"][0]["health"] == "stopped"
    assert driver["health"] == "stopped"
    document = local_portal.render_portal_json([driver], "10.0.0.5", generated_at="t")
    assert document["drivers"][0]["pages"][0]["url"] is None


@pytest.mark.parametrize(
    ("status", "health"),
    [
        ("Up 2 hours (healthy)", "healthy"),
        ("Up 5 seconds (health: starting)", "starting"),
        ("Up 1 minute (unhealthy)", "unhealthy"),
        ("Up 1 minute", "running"),
    ],
)
def test_health_comes_from_docker_status(config_dir: Path, status: str, health: str) -> None:
    containers = _running((TWIN_A, "gateway"), status=status)
    [driver] = local_portal.build_listing(config_dir, containers)
    assert driver["pages"][0]["health"] == health


def test_single_container_driver_without_service(tmp_path: Path) -> None:
    drivers = {"default": {"local_http": [{"name": "preview", "port": 8080, "path": "/live"}]}}
    (tmp_path / f"{TWIN_B}.json").write_text(json.dumps(_twin("Cam", asset_drivers=drivers)))
    [driver] = local_portal.build_listing(
        tmp_path, _running((TWIN_B, ""), status="Up 1 minute")
    )
    assert driver["pages"][0]["health"] == "running"
    document = local_portal.render_portal_json([driver], "192.168.1.20", generated_at="t")
    assert document["drivers"][0]["pages"][0]["url"] == "http://192.168.1.20:8080/live"


def test_unreadable_twin_file_is_skipped(config_dir: Path) -> None:
    (config_dir / f"{TWIN_B}.json").write_text("{not json")
    containers = _running((TWIN_A, "gateway"), (TWIN_B, ""))
    assert [d["id"] for d in local_portal.build_listing(config_dir, containers)] == [TWIN_A[:8]]


# -- output ------------------------------------------------------------------


def test_portal_json_shape(config_dir: Path) -> None:
    containers = _running((TWIN_A, "gateway"))
    drivers = local_portal.build_listing(config_dir, containers)
    document = local_portal.render_portal_json(
        drivers, "192.168.1.20", generated_at="2026-10-10T00:00:00Z"
    )
    assert document == {
        "schema": "cyberwave.edge.local_portal.v1",
        "generated_at": "2026-10-10T00:00:00Z",
        "host": "192.168.1.20",
        "drivers": [
            {
                "id": TWIN_A[:8],
                "twin_name": "Guardian 1",
                "registry_id": "acme/aircraft",
                "health": "healthy",
                "pages": [
                    {
                        "name": "plans",
                        "description": "Flight plans for the ground station",
                        "port": 8765,
                        "path": "/",
                        "service": "gateway",
                        "health": "healthy",
                        "url": "http://192.168.1.20:8765/",
                    }
                ],
            }
        ],
    }


def test_no_secrets_or_metadata_leak_into_output(config_dir: Path) -> None:
    containers = _running((TWIN_A, "gateway"))
    drivers = local_portal.build_listing(config_dir, containers)
    document = local_portal.render_portal_json(drivers, "10.0.0.5", generated_at="t")
    rendered = json.dumps(document) + local_portal.render_portal_html(document)
    for leaked in (SECRET, "shared_env", "docker_image", "CYBERWAVE_API_KEY", TWIN_A):
        assert leaked not in rendered


def test_html_escapes_driver_supplied_text(tmp_path: Path) -> None:
    drivers = {
        "default": {
            "local_http": [{"name": "p", "port": 80, "description": "<img src=x onerror=alert(1)>"}]
        }
    }
    (tmp_path / f"{TWIN_B}.json").write_text(
        json.dumps(_twin("<script>alert(1)</script>", asset_drivers=drivers))
    )
    listing = local_portal.build_listing(tmp_path, _running((TWIN_B, "")))
    page = local_portal.render_portal_html(
        local_portal.render_portal_json(listing, "10.0.0.5", generated_at="t")
    )
    assert "<script>" not in page and "<img" not in page
    assert "&lt;script&gt;" in page


def test_empty_listing_renders(tmp_path: Path) -> None:
    document = local_portal.render_portal_json([], "10.0.0.5", generated_at="t")
    assert document["drivers"] == []
    assert "No running driver declares a local page" in local_portal.render_portal_html(document)


@pytest.mark.parametrize(
    ("override", "local", "expected"),
    [
        (None, "192.168.1.20", "192.168.1.20"),
        (None, "::ffff:192.168.1.20", "192.168.1.20"),
        (None, "fe80::1%eth0", "[fe80::1]"),
        ("edge-hangar.local", "192.168.1.20", "edge-hangar.local"),
        ("10.1.2.3", "192.168.1.20", "10.1.2.3"),
        ("bad host/<x>", "192.168.1.20", "192.168.1.20"),
    ],
)
def test_link_host(tmp_path: Path, override: str | None, local: str, expected: str) -> None:
    portal = local_portal.LocalPortal(
        config_dir=tmp_path, bind=("127.0.0.1", 0), advertised_host=override
    )
    assert portal._link_host(local) == expected


# -- live server -------------------------------------------------------------


def _get(url: str, *, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], str]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode()


def test_server_serves_index_json_and_healthz(config_dir: Path) -> None:
    calls: list[int] = []

    def lister() -> dict[str, str]:
        calls.append(1)
        return _running((TWIN_A, "gateway"))

    portal = local_portal.LocalPortal(
        config_dir=config_dir, bind=("127.0.0.1", 0), container_lister=lister
    )
    portal.start()
    try:
        portal.refresh()
        host, port = portal.server_address
        assert host == "127.0.0.1"
        base = f"http://127.0.0.1:{port}"

        status, headers, body = _get(f"{base}/healthz")
        assert status == 200
        health = json.loads(body)
        assert health["status"] == "ok" and health["drivers"] == 1

        # A forged Host header must not steer the links.
        status, headers, body = _get(f"{base}/portal.json", headers={"Host": "evil.example"})
        assert status == 200 and headers["Content-Type"] == "application/json"
        document = json.loads(body)
        assert document["drivers"][0]["pages"][0]["url"] == "http://127.0.0.1:8765/"
        assert "evil.example" not in body

        status, headers, body = _get(f"{base}/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert "default-src 'none'" in headers["Content-Security-Policy"]
        assert headers["Cache-Control"] == "no-store"
        assert "Python" not in headers.get("Server", "")
        assert "href='http://127.0.0.1:8765/'" in body

        assert _get(f"{base}/../credentials.json")[0] == 404
        assert _get(f"{base}/admin")[0] == 404

        post = urllib.request.Request(f"{base}/", data=b"x", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(post, timeout=5)
        assert exc_info.value.code == 501
    finally:
        portal.stop()
    assert calls, "refresher/lister was used"


def test_start_from_env_binds_as_configured_and_shuts_down(config_dir: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {
        local_portal.ENV_ENABLED: "on",
        local_portal.ENV_BIND: f"127.0.0.1:{port}",
        local_portal.ENV_REFRESH: "60",
    }
    portal = local_portal.start_from_env(config_dir=config_dir, env=env)
    try:
        assert portal is not None
        assert portal.server_address == ("127.0.0.1", port)
        assert local_portal.start_from_env(config_dir=config_dir, env=env) is portal
        assert _get(f"http://127.0.0.1:{port}/healthz")[0] == 200
    finally:
        local_portal.shutdown()
    assert local_portal._ACTIVE is None
    with socket.socket() as again:  # the port is released
        again.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        again.bind(("127.0.0.1", port))
        again.listen(1)
