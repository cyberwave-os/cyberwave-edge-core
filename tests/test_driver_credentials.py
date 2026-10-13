"""Twin-scoped driver tokens: request, cache, renewal, fallback and hardening."""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from cyberwave.mqtt_identity import mqtt_username_for_token

from cyberwave_edge_core import driver_credentials as dc

EDGE_TOKEN = "cw_" + "e" * 64
DRIVER_TOKEN = "cw_" + "d" * 64
BASE_URL = "https://api.example.test"
TWIN = "11111111-1111-1111-1111-111111111111"
CHILD = "22222222-2222-2222-2222-222222222222"
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


class _Response:
    def __init__(self, status_code: int, body: Any = None) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body


class _Backend:
    """Records calls and answers like the driver-token endpoint."""

    def __init__(self, *, mint_status: int = 201, renew_status: int = 200) -> None:
        self.mint_status = mint_status
        self.renew_status = renew_status
        self.calls: list[tuple[str, dict[str, Any], str]] = []
        self.minted = 0

    def __call__(self, url: str, *, json_body: dict[str, Any], token: str) -> _Response:
        self.calls.append((url, json_body, token))
        if url.endswith("/renew"):
            return _Response(
                self.renew_status,
                {
                    "uuid": "tok-1",
                    "token": "",
                    "expires_at": (NOW + timedelta(days=30)).isoformat(),
                },
            )
        if self.mint_status != 201:
            return _Response(self.mint_status, {"detail": "nope"})
        self.minted += 1
        return _Response(
            201,
            {
                "uuid": f"tok-{self.minted}",
                "token": DRIVER_TOKEN if self.minted == 1 else DRIVER_TOKEN[:-1] + "f",
                "expires_at": (NOW + timedelta(days=30)).isoformat(),
            },
        )


@pytest.fixture
def scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dc.MODE_ENV, dc.MODE_SCOPED)


def _resolve(tmp_path: Path, backend: _Backend, *, now: datetime = NOW, **kwargs: Any):
    params: dict[str, Any] = {
        "twin_uuid": TWIN,
        "edge_token": EDGE_TOKEN,
        "base_url": BASE_URL,
        "config_dir": tmp_path,
        "child_twin_uuids": [CHILD],
        "edge_fingerprint": "jetson-abc123",
        "post": backend,
        "now": now,
    }
    params.update(kwargs)
    return dc.resolve_driver_credential(**params)


def test_scoped_token_requested_and_cached(tmp_path: Path, scoped: None) -> None:
    backend = _Backend()
    credential = _resolve(tmp_path, backend)

    assert credential == dc.DriverCredential(
        api_key=DRIVER_TOKEN,
        scoped=True,
        token_uuid="tok-1",
        expires_at=NOW + timedelta(days=30),
    )
    url, body, token = backend.calls[0]
    assert url == f"{BASE_URL}/api/v1/twins/{TWIN}/driver-token"
    assert token == EDGE_TOKEN
    assert body["child_twin_uuids"] == [CHILD]
    assert body["edge_fingerprint"] == dc.edge_rotation_key("jetson-abc123")
    assert body["edge_fingerprint"].startswith("fp-")

    cache = tmp_path / dc.TOKEN_DIR_NAME / f"{TWIN}.json"
    assert json.loads(cache.read_text())["token"] == DRIVER_TOKEN
    if os.name != "nt":
        assert stat.S_IMODE(cache.stat().st_mode) == 0o600
        assert stat.S_IMODE(cache.parent.stat().st_mode) == 0o700
        assert EDGE_TOKEN not in cache.read_text()

    # A second launch (another service, a restart) reuses it offline.
    assert _resolve(tmp_path, backend).api_key == DRIVER_TOKEN
    assert len(backend.calls) == 1


def test_cache_not_reused_for_other_edge_token_or_children(tmp_path: Path, scoped: None) -> None:
    backend = _Backend()
    _resolve(tmp_path, backend)
    _resolve(tmp_path, backend, child_twin_uuids=[])
    _resolve(tmp_path, backend, edge_token="cw_" + "x" * 64)
    assert backend.minted == 3


def test_token_near_expiry_is_renewed_not_reminted(tmp_path: Path, scoped: None) -> None:
    backend = _Backend()
    _resolve(tmp_path, backend)
    later = NOW + timedelta(days=25)  # 5 days left < REFRESH_BEFORE (7)
    credential = _resolve(tmp_path, backend, now=later)
    assert credential.api_key == DRIVER_TOKEN
    assert backend.calls[-1][0].endswith("/driver-token/tok-1/renew")
    assert backend.minted == 1


def test_revoked_cached_token_is_replaced(tmp_path: Path, scoped: None) -> None:
    backend = _Backend()
    _resolve(tmp_path, backend)
    backend.renew_status = 410
    credential = _resolve(tmp_path, backend, now=NOW + timedelta(days=25))
    assert backend.minted == 2
    assert credential.token_uuid == "tok-2"


def test_renew_failure_keeps_unexpired_token(tmp_path: Path, scoped: None) -> None:
    backend = _Backend()
    _resolve(tmp_path, backend)
    backend.renew_status = 503
    credential = _resolve(tmp_path, backend, now=NOW + timedelta(days=25))
    assert credential.scoped and credential.api_key == DRIVER_TOKEN
    assert backend.minted == 1


def test_old_backend_404_falls_back_to_edge_token(
    tmp_path: Path, scoped: None, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING")
    credential = _resolve(tmp_path, _Backend(mint_status=404))
    assert credential == dc.DriverCredential(api_key=EDGE_TOKEN, scoped=False)
    assert "full API token" in caplog.text
    assert not (tmp_path / dc.TOKEN_DIR_NAME / f"{TWIN}.json").exists()


def test_backend_error_falls_back_unless_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(dc.MODE_ENV, dc.MODE_SCOPED)
    assert _resolve(tmp_path, _Backend(mint_status=500)).scoped is False
    monkeypatch.setenv(dc.MODE_ENV, dc.MODE_STRICT)
    assert _resolve(tmp_path, _Backend(mint_status=404)) is None
    assert _resolve(tmp_path, _Backend(mint_status=500)) is None


def test_legacy_mode_never_calls_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dc.MODE_ENV, dc.MODE_LEGACY)
    backend = _Backend()
    assert _resolve(tmp_path, backend) == dc.DriverCredential(EDGE_TOKEN, scoped=False)
    assert backend.calls == []


def test_default_mode_is_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dc, "DEFAULT_MODE", dc.MODE_AUTO)
    monkeypatch.delenv(dc.MODE_ENV, raising=False)
    assert dc.resolve_mode() == dc.MODE_AUTO
    monkeypatch.setenv(dc.MODE_ENV, "bogus")
    assert dc.resolve_mode() == dc.MODE_AUTO


def test_auto_mode_is_scoped_only_for_images_that_opt_in() -> None:
    """PR #4622 review: scoped tokens must not flip every driver on every edge."""
    assert dc.effective_mode(mode=dc.MODE_AUTO, image_opts_in=True) == dc.MODE_SCOPED
    assert dc.effective_mode(mode=dc.MODE_AUTO, image_opts_in=False) == dc.MODE_LEGACY
    # An explicit operator choice applies to every driver, label or not.
    for mode in (dc.MODE_SCOPED, dc.MODE_STRICT, dc.MODE_LEGACY):
        assert dc.effective_mode(mode=mode, image_opts_in=False) == mode
        assert dc.effective_mode(mode=mode, image_opts_in=True) == mode


def test_auto_resolves_to_a_scoped_token_when_asked(tmp_path: Path) -> None:
    backend = _Backend()
    credential = dc.resolve_driver_credential(
        twin_uuid=TWIN,
        edge_token=EDGE_TOKEN,
        base_url=BASE_URL,
        config_dir=tmp_path,
        post=backend,
        now=NOW,
        mode=dc.MODE_AUTO,
    )
    assert credential is not None and credential.scoped
    assert backend.calls


def test_periodic_renewal_extends_running_drivers(tmp_path: Path, scoped: None) -> None:
    backend = _Backend()
    _resolve(tmp_path, backend)
    assert (
        dc.renew_cached_driver_tokens(
            twin_uuids=[TWIN],
            edge_token=EDGE_TOKEN,
            base_url=BASE_URL,
            config_dir=tmp_path,
            post=backend,
            now=NOW + timedelta(days=1),
        )
        == 0
    )
    renewed = dc.renew_cached_driver_tokens(
        twin_uuids=[TWIN, "unknown"],
        edge_token=EDGE_TOKEN,
        base_url=BASE_URL,
        config_dir=tmp_path,
        post=backend,
        now=NOW + timedelta(days=24),
    )
    assert renewed == 1
    backend.renew_status = 410
    dc.renew_cached_driver_tokens(
        twin_uuids=[TWIN],
        edge_token=EDGE_TOKEN,
        base_url=BASE_URL,
        config_dir=tmp_path,
        post=backend,
        now=NOW + timedelta(days=60),
    )
    assert not (tmp_path / dc.TOKEN_DIR_NAME / f"{TWIN}.json").exists()


def test_apply_to_container_scrubs_edge_token() -> None:
    env = {
        "CYBERWAVE_API_KEY": EDGE_TOKEN,
        "CYBERWAVE_MQTT_PASSWORD": EDGE_TOKEN,
        "CYBERWAVE_MQTT_USER": "alice",
        "CYBERWAVE_SOMETHING": EDGE_TOKEN,
        "CYBERWAVE_TWIN_UUID": TWIN,
    }
    dc.apply_to_container(env, dc.DriverCredential(DRIVER_TOKEN, scoped=True), EDGE_TOKEN)
    assert EDGE_TOKEN not in env.values()
    assert env["CYBERWAVE_API_KEY"] == DRIVER_TOKEN
    assert env["CYBERWAVE_MQTT_PASSWORD"] == DRIVER_TOKEN
    assert env["CYBERWAVE_MQTT_API_TOKEN"] == DRIVER_TOKEN
    expected_username = mqtt_username_for_token(DRIVER_TOKEN)
    assert env["CYBERWAVE_MQTT_USERNAME"] == expected_username
    assert env["CYBERWAVE_MQTT_USER"] == expected_username
    assert env["CYBERWAVE_TWIN_UUID"] == TWIN


def test_apply_to_container_legacy_is_unchanged() -> None:
    env = {"CYBERWAVE_MQTT_USER": "alice"}
    dc.apply_to_container(env, dc.DriverCredential(EDGE_TOKEN, scoped=False), EDGE_TOKEN)
    assert env == {"CYBERWAVE_MQTT_USER": "alice", "CYBERWAVE_API_KEY": EDGE_TOKEN}


def test_mask_mount_args_hide_secrets(tmp_path: Path) -> None:
    (tmp_path / "credentials.json").write_text(
        json.dumps(
            {
                "token": EDGE_TOKEN,
                "envs": {
                    "CYBERWAVE_BASE_URL": BASE_URL,
                    "CYBERWAVE_MACOS_AUDIO_PLAYBACK_URL": "http://host:8101",
                    "CYBERWAVE_API_KEY": EDGE_TOKEN,
                    "CYBERWAVE_MQTT_PASSWORD": "secret",
                },
            }
        )
    )
    args = dc.mask_mount_args(tmp_path)
    public = tmp_path / dc.PUBLIC_CREDENTIALS_NAME
    assert args == [
        "-v",
        f"{public}:/app/.cyberwave/credentials.json:ro",
        "--tmpfs",
        "/app/.cyberwave/driver_tokens:ro,size=64k",
    ]
    content = json.loads(public.read_text())
    assert content == {
        "envs": {
            "CYBERWAVE_BASE_URL": BASE_URL,
            "CYBERWAVE_MACOS_AUDIO_PLAYBACK_URL": "http://host:8101",
        }
    }


# --------------------------------------------------------------------------- #
# Launcher wiring
# --------------------------------------------------------------------------- #


def _launch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, credential: Any) -> list[str]:
    """Run ``_run_docker_image`` with docker stubbed; return the docker argv."""
    from cyberwave_edge_core import driver_launcher, startup

    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(startup, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(driver_launcher.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(driver_launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(startup, "_resolve_driver_image_tag", lambda image: image)
    monkeypatch.setattr(startup, "_docker_image_exists_locally", lambda _image: True)
    monkeypatch.setattr(startup, "_resolve_mqtt_kwargs", lambda: {})
    monkeypatch.setattr(startup, "load_credentials_envs", lambda: {})
    monkeypatch.setattr(startup, "_get_zenoh_config", lambda: None)
    monkeypatch.setattr(startup, "build_zenoh_env_vars", lambda _cfg: {})
    monkeypatch.setattr(
        "cyberwave_edge_core.docker_helpers.docker_inspect_running_containers", lambda: []
    )
    monkeypatch.setattr(
        "cyberwave_edge_core.docker_launch.remove_existing_container", lambda *_a, **_k: True
    )
    monkeypatch.setattr(driver_launcher, "docker_image_id", lambda _image: "sha256:abc")

    def _launch(**kwargs: Any) -> bool:
        captured["argv"] = list(kwargs["run_argv"])
        return True

    monkeypatch.setattr("cyberwave_edge_core.docker_launch.launch_detached_container", _launch)
    monkeypatch.setattr(driver_launcher, "_resolve_driver_credential", lambda **_kwargs: credential)

    class _Alert:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def create(self) -> None:
            pass

        def update_metadata(self, *_a: Any, **_k: Any) -> None:
            pass

        def resolve(self) -> None:
            pass

        def mark_failed_and_resolve(self, *_a: Any, **_k: Any) -> None:
            captured["failed"] = list(_a)

    monkeypatch.setattr(startup, "DriverStartingAlertContext", _Alert)
    ok = driver_launcher._run_docker_image(
        "cyberwaveos/test-driver:latest",
        [],
        twin_uuid=TWIN,
        token=EDGE_TOKEN,
        skip_pull=True,
    )
    captured["ok"] = [str(ok)]
    return captured.get("argv", []) if ok else []


def test_launcher_passes_scoped_token_and_masks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CYBERWAVE_MQTT_PASSWORD", EDGE_TOKEN)
    argv = _launch_env(monkeypatch, tmp_path, dc.DriverCredential(DRIVER_TOKEN, scoped=True))
    assert argv, "launch failed"
    assert f"CYBERWAVE_API_KEY={DRIVER_TOKEN}" in argv
    assert not any(EDGE_TOKEN in arg for arg in argv)
    assert f"{tmp_path / dc.PUBLIC_CREDENTIALS_NAME}:/app/.cyberwave/credentials.json:ro" in argv
    assert "/app/.cyberwave/driver_tokens:ro,size=64k" in argv


def test_launcher_legacy_keeps_edge_token(monkeypatch, tmp_path: Path) -> None:
    argv = _launch_env(monkeypatch, tmp_path, dc.DriverCredential(EDGE_TOKEN, scoped=False))
    assert f"CYBERWAVE_API_KEY={EDGE_TOKEN}" in argv
    assert not any("driver_tokens" in arg for arg in argv)


def test_launcher_refuses_in_strict_without_token(monkeypatch, tmp_path: Path) -> None:
    assert _launch_env(monkeypatch, tmp_path, None) == []


def test_launcher_refuses_a_scoped_driver_when_masking_fails(monkeypatch, tmp_path: Path) -> None:
    """Fail closed (PR #4622 review): a scoped container must not read the edge token."""

    def _mask_fails(_config_dir: Path) -> list[str]:
        raise OSError("read-only file system")

    monkeypatch.setattr(dc, "mask_mount_args", _mask_fails)
    assert _launch_env(monkeypatch, tmp_path, dc.DriverCredential(DRIVER_TOKEN, scoped=True)) == []
    # A legacy driver never masks, so it is unaffected.
    assert _launch_env(monkeypatch, tmp_path, dc.DriverCredential(EDGE_TOKEN, scoped=False))


@pytest.mark.parametrize(
    ("configured", "label", "expected"),
    [
        (None, True, "scoped"),  # default auto + opt-in label
        (None, False, None),  # default auto, no label: the edge token as before
        ("scoped", False, "scoped"),
        ("legacy", True, None),
    ],
)
def test_launcher_resolves_the_mode_per_image(monkeypatch, tmp_path: Path, configured, label, expected) -> None:
    from types import SimpleNamespace

    from cyberwave_edge_core import driver_launcher

    if configured:
        monkeypatch.setenv(dc.MODE_ENV, configured)
    else:
        monkeypatch.delenv(dc.MODE_ENV, raising=False)
        monkeypatch.setattr(dc, "DEFAULT_MODE", dc.MODE_AUTO)
    stub = SimpleNamespace(
        get_runtime_env_var=lambda _name, default: default,
        DEFAULT_API_URL=BASE_URL,
        get_or_create_fingerprint=lambda: "fp",
        CONFIG_DIR=tmp_path,
    )
    monkeypatch.setattr(driver_launcher, "_startup", lambda: stub)
    seen: list[str] = []
    monkeypatch.setattr(
        driver_launcher,
        "_image_has_capability_label",
        lambda _image, name: seen.append(name) or label,
    )
    modes: list[str] = []

    def _resolve(**kwargs: Any) -> dc.DriverCredential:
        modes.append(kwargs["mode"])
        return dc.DriverCredential(DRIVER_TOKEN, scoped=True)

    monkeypatch.setattr(dc, "resolve_driver_credential", _resolve)

    credential = driver_launcher._resolve_driver_credential(
        twin_uuid=TWIN, token=EDGE_TOKEN, child_camera_twin_uuids=None, image="img"
    )

    if expected is None:
        assert credential == dc.DriverCredential(EDGE_TOKEN, scoped=False) and modes == []
    else:
        assert modes == [expected] and credential.scoped
    if configured is None:
        assert seen == [dc.SCOPED_TOKEN_LABEL]


def test_bad_numeric_env_knobs_fall_back_instead_of_breaking_import(monkeypatch):
    """PR #4622 review: a typo here used to stop Edge Core from launching any driver."""
    import importlib

    import cyberwave_edge_core.driver_credentials as module

    monkeypatch.setenv("CYBERWAVE_EDGE_DRIVER_TOKEN_REFRESH_DAYS", "oops")
    monkeypatch.setenv("CYBERWAVE_EDGE_DRIVER_TOKEN_RENEW_CHECK_SECONDS", "-5")
    monkeypatch.setenv("CYBERWAVE_EDGE_DRIVER_TOKEN_TIMEOUT_S", "")
    try:
        reloaded = importlib.reload(module)
        assert reloaded.REFRESH_BEFORE.days == 7
        assert reloaded.RENEW_CHECK_INTERVAL_SECONDS == 3600.0
        assert reloaded.HTTP_TIMEOUT_SECONDS == 2.5
    finally:
        monkeypatch.undo()
        importlib.reload(module)
