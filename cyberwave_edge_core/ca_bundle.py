"""CA certificates for the frozen binary.

PyInstaller bundles the build image's OpenSSL, which looks for CAs where that image
keeps them: /etc/pki/tls on the AlmaLinux build. Debian and Ubuntu keep them under
/etc/ssl, so there every TLS verification, MQTT's included, would find no CA.
"""

from __future__ import annotations

import os
import ssl
from collections.abc import Callable, Mapping, MutableMapping

# Where Linux distributions and macOS keep their CA bundle.
HOST_CA_BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",  # Debian, Ubuntu, Alpine
    "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL, Fedora
    "/etc/ssl/ca-bundle.pem",  # openSUSE
    "/etc/ssl/cert.pem",  # macOS
)


def select_ca_bundle(
    environ: Mapping[str, str],
    default_paths: ssl.DefaultVerifyPaths,
    isfile: Callable[[str], bool],
) -> str | None:
    """The file ``SSL_CERT_FILE`` should name, or None to leave OpenSSL's lookup alone.

    None when the environment names CAs or OpenSSL's own default CA file exists. Its
    default directory does not count: it can exist with no CAs in it, and
    ``SSL_CERT_FILE`` adds to it rather than replacing it. The host's bundle comes
    before certifi's, so CAs installed on the host keep working.
    """
    if environ.get("SSL_CERT_FILE") or environ.get("SSL_CERT_DIR"):
        return None
    if default_paths.cafile:
        return None
    for bundle in HOST_CA_BUNDLES:
        if isfile(bundle):
            return bundle
    try:
        import certifi
    except ImportError:
        return None
    return certifi.where()


def ensure_ca_bundle(environ: MutableMapping[str, str] = os.environ) -> str | None:
    """Set ``SSL_CERT_FILE`` when OpenSSL would find no CA; return what was set.

    Runs as a PyInstaller runtime hook, before anything creates a TLS context.
    """
    bundle = select_ca_bundle(environ, ssl.get_default_verify_paths(), os.path.isfile)
    if bundle:
        environ["SSL_CERT_FILE"] = bundle
    return bundle
