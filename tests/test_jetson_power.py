from __future__ import annotations

from types import SimpleNamespace

from cyberwave_edge_core import jetson_power


def test_auto_preserves_the_existing_host_power_policy(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        jetson_power.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )

    assert jetson_power.configure_jetson_nvpmodel("auto") is True
    assert calls == []


def test_rejects_non_numeric_remote_metadata() -> None:
    assert jetson_power.configure_jetson_nvpmodel("2; rm -rf /") is False


def test_applies_a_numeric_model_id_on_a_jetson_host(monkeypatch) -> None:
    monkeypatch.setattr(jetson_power.platform, "system", lambda: "Linux")
    monkeypatch.setattr(jetson_power.shutil, "which", lambda _name: "/usr/sbin/nvpmodel")
    monkeypatch.setattr(
        jetson_power.Path,
        "read_text",
        lambda *_args, **_kwargs: "< POWER_MODEL ID=2 NAME=MAXN_SUPER >",
    )
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(jetson_power.subprocess, "run", fake_run)

    assert jetson_power.configure_jetson_nvpmodel("2") is True
    assert calls == [["/usr/sbin/nvpmodel", "-m", "2"]]


def test_normalizes_leading_zeroes_and_accepts_supported_three_digit_ids(monkeypatch) -> None:
    monkeypatch.setattr(jetson_power.platform, "system", lambda: "Linux")
    monkeypatch.setattr(jetson_power.shutil, "which", lambda _name: "/usr/sbin/nvpmodel")
    monkeypatch.setattr(
        jetson_power.Path,
        "read_text",
        lambda *_args, **_kwargs: "< POWER_MODEL ID=100 NAME=CUSTOM >",
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        jetson_power.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )

    assert jetson_power.configure_jetson_nvpmodel("0100") is True
    assert calls == [["/usr/sbin/nvpmodel", "-m", "100"]]


def test_discovers_supported_model_ids_from_the_host_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        jetson_power.Path,
        "read_text",
        lambda *_args, **_kwargs: """
            < POWER_MODEL ID=0 NAME=10W >
            < POWER_MODEL ID=1 NAME=25W >
            < POWER_MODEL ID=2 NAME=MAXN_SUPER >
        """,
    )

    assert jetson_power.supported_jetson_nvpmodel_ids() == {"0", "1", "2"}


def test_does_not_apply_an_id_unsupported_by_this_jetson(monkeypatch) -> None:
    monkeypatch.setattr(jetson_power.platform, "system", lambda: "Linux")
    monkeypatch.setattr(jetson_power.shutil, "which", lambda _name: "/usr/sbin/nvpmodel")
    monkeypatch.setattr(
        jetson_power.Path,
        "read_text",
        lambda *_args, **_kwargs: "< POWER_MODEL ID=0 NAME=10W >",
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        jetson_power.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )

    assert jetson_power.configure_jetson_nvpmodel("2") is False
    assert calls == []


def test_reboot_required_is_not_reported_as_an_applied_mode(monkeypatch) -> None:
    monkeypatch.setattr(jetson_power.platform, "system", lambda: "Linux")
    monkeypatch.setattr(jetson_power.shutil, "which", lambda _name: "/usr/sbin/nvpmodel")
    monkeypatch.setattr(
        jetson_power.Path,
        "read_text",
        lambda *_args, **_kwargs: "< POWER_MODEL ID=2 NAME=MAXN_SUPER >",
    )
    received_kwargs = {}

    def fake_run(_command, **kwargs):
        received_kwargs.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="NVPM WARN: Reboot required", stderr="")

    monkeypatch.setattr(jetson_power.subprocess, "run", fake_run)

    assert jetson_power.configure_jetson_nvpmodel("2") is False
    assert received_kwargs["stdin"] is jetson_power.subprocess.DEVNULL


def test_resolves_one_host_wide_power_request() -> None:
    assert jetson_power.resolve_jetson_nvpmodel_request(["auto", None, "2", " 2 "]) == "2"


def test_resolves_padded_and_unpadded_host_wide_power_requests_as_one_mode() -> None:
    assert jetson_power.resolve_jetson_nvpmodel_request(["2", "02"]) == "2"


def test_rejects_conflicting_host_wide_power_requests() -> None:
    assert jetson_power.resolve_jetson_nvpmodel_request(["0", "2"]) is None
