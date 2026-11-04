"""Tests for the hidden edge-core self-check commands."""

import ssl

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


class _Connection:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def version(self) -> str:
        return "TLSv1.3"


class _Context:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def wrap_socket(self, sock, server_hostname: str) -> _Connection:
        if self.error:
            raise self.error
        return _Connection()


def test_selfcheck_tls_reports_a_verified_server(monkeypatch) -> None:
    monkeypatch.setattr(main_module.socket, "create_connection", lambda *a, **k: _Connection())
    monkeypatch.setattr(main_module.ssl, "create_default_context", lambda: _Context())

    result = CliRunner().invoke(main_module.cli, ["__selfcheck_tls", "example.com"])

    assert result.exit_code == 0
    assert "tls-ok: example.com (TLSv1.3)" in result.output


def test_selfcheck_tls_fails_when_no_ca_verifies_the_server(monkeypatch) -> None:
    error = ssl.SSLCertVerificationError("unable to get local issuer certificate")
    monkeypatch.setattr(main_module.socket, "create_connection", lambda *a, **k: _Connection())
    monkeypatch.setattr(main_module.ssl, "create_default_context", lambda: _Context(error))

    result = CliRunner().invoke(main_module.cli, ["__selfcheck_tls", "example.com"])

    assert result.exit_code == 1
    assert "tls-failed: example.com" in result.output
