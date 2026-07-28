"""Version helpers for Cyberwave Edge Core."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version

STATIC_VERSION = "0.1.8"

try:
    from ._build_version import BUILD_VERSION
except ImportError:
    BUILD_VERSION = None


def get_version() -> str:
    """Resolve build, installed, or source version in that order."""
    if BUILD_VERSION:
        return BUILD_VERSION

    try:
        return metadata_version("cyberwave-edge-core")
    except PackageNotFoundError:
        return STATIC_VERSION
    except Exception:
        return STATIC_VERSION
