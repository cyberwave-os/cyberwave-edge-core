"""Tests for the CA bundle the frozen binary points OpenSSL at."""

import ssl

import certifi

from cyberwave_edge_core import ca_bundle

# OpenSSL built on AlmaLinux, run on Debian: neither default path exists.
MISSING = ssl.DefaultVerifyPaths(
    cafile=None,
    capath=None,
    openssl_cafile_env="SSL_CERT_FILE",
    openssl_cafile="/etc/pki/tls/cert.pem",
    openssl_capath_env="SSL_CERT_DIR",
    openssl_capath="/etc/pki/tls/certs",
)
HOST_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"


def test_a_ca_file_or_dir_named_in_the_environment_is_kept() -> None:
    for environ in ({"SSL_CERT_FILE": "/opt/ca.pem"}, {"SSL_CERT_DIR": "/opt/certs"}):
        assert ca_bundle.select_ca_bundle(environ, MISSING, lambda _path: True) is None


def test_a_default_ca_file_that_exists_is_kept() -> None:
    found = MISSING._replace(cafile="/etc/pki/tls/cert.pem")

    assert ca_bundle.select_ca_bundle({}, found, lambda _path: True) is None


def test_a_default_ca_dir_alone_still_takes_the_host_bundle() -> None:
    empty_dir = MISSING._replace(capath="/etc/pki/tls/certs")
    selected = ca_bundle.select_ca_bundle({}, empty_dir, lambda path: path == HOST_BUNDLE)

    assert selected == HOST_BUNDLE


def test_missing_default_paths_take_the_host_bundle() -> None:
    selected = ca_bundle.select_ca_bundle({}, MISSING, lambda path: path == HOST_BUNDLE)

    assert selected == HOST_BUNDLE


def test_without_a_host_bundle_certifi_is_used() -> None:
    assert ca_bundle.select_ca_bundle({}, MISSING, lambda _path: False) == certifi.where()


def test_ensure_sets_ssl_cert_file(monkeypatch, tmp_path) -> None:
    bundle = tmp_path / "ca-certificates.crt"
    bundle.write_text("")
    monkeypatch.setattr(ca_bundle, "HOST_CA_BUNDLES", (str(bundle),))
    monkeypatch.setattr(ca_bundle.ssl, "get_default_verify_paths", lambda: MISSING)
    environ: dict[str, str] = {}

    assert ca_bundle.ensure_ca_bundle(environ) == str(bundle)
    assert environ == {"SSL_CERT_FILE": str(bundle)}
