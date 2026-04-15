"""Tests for the hidden edge-core SDK self-check command."""

from click.testing import CliRunner

import cyberwave_edge_core.main as main_module


def test_selfcheck_sdk_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "_load_sdk_default_api", lambda: object())

    result = CliRunner().invoke(main_module.cli, ["__selfcheck_sdk"])

    assert result.exit_code == 0
    assert "sdk-rest-ok" in result.output


def test_selfcheck_sdk_fails_when_rest_client_missing(monkeypatch) -> None:
    def _raise_import_error():
        raise ImportError("missing rest client")

    monkeypatch.setattr(main_module, "_load_sdk_default_api", _raise_import_error)

    result = CliRunner().invoke(main_module.cli, ["__selfcheck_sdk"])

    assert result.exit_code == 1
    assert "sdk-rest-missing" in result.output
