"""Safe, opt-in Jetson power-mode selection for edge-managed drivers."""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

JETSON_NVP_MODEL_ENV = "CYBERWAVE_JETSON_NVP_MODEL"
NVP_MODEL_CONFIG_PATH = Path("/etc/nvpmodel.conf")
_NVP_MODEL_ID_PATTERN = re.compile(r"<\s*POWER_MODEL\b[^>]*\bID\s*=\s*([0-9]+)\b")


def _normalize_nvpmodel_id(value: str | None) -> str | None:
    """Normalize one explicit numeric nvpmodel request.

    The vendor tool treats padded and unpadded numeric IDs as the same mode.
    Keep host-wide conflict detection aligned with the value eventually passed to
    ``nvpmodel``.
    """
    normalized = (value or "").strip().lower()
    if not normalized or normalized == "auto":
        return None
    return normalized.lstrip("0") or "0"


def supported_jetson_nvpmodel_ids() -> set[str] | None:
    """Return the power-mode IDs declared by this Jetson's nvpmodel config.

    NVIDIA ships a device- and JetPack-specific configuration, so a mode ID is
    only meaningful on the device that declares it.  ``None`` means the
    configuration could not be read or parsed and should not be treated as an
    empty set of supported modes.
    """
    try:
        config = NVP_MODEL_CONFIG_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        logger.warning("Could not read Jetson nvpmodel configuration: %s", exc)
        return None

    supported_ids = {
        model_id.lstrip("0") or "0" for model_id in _NVP_MODEL_ID_PATTERN.findall(config)
    }
    if not supported_ids:
        logger.warning("Jetson nvpmodel configuration contains no power-mode IDs")
        return None
    return supported_ids


def resolve_jetson_nvpmodel_request(values: Iterable[str | None]) -> str | None:
    """Return one host-wide explicit request, or reject conflicting requests.

    nvpmodel is a host-wide setting. Empty values and ``auto`` deliberately make
    no request, so they may coexist with one explicit selection.
    """
    requested_ids = {
        normalized
        for value in values
        if (normalized := _normalize_nvpmodel_id(value)) is not None
    }
    if len(requested_ids) > 1:
        logger.error(
            "Conflicting %s values on one Jetson host: %s; preserving the current power mode",
            JETSON_NVP_MODEL_ENV,
            ", ".join(sorted(requested_ids)),
        )
        return None
    return next(iter(requested_ids), None)


def configure_jetson_nvpmodel(value: str | None) -> bool:
    """Apply an explicitly selected nvpmodel mode, when this is a Jetson host.

    The value comes from the matched driver's ``shared_env``. It is deliberately
    an integer rather than an executable command: remote twin metadata may select
    a documented power profile, but cannot execute arbitrary host shell input.
    IDs are checked against this Jetson's ``/etc/nvpmodel.conf`` because they are
    not portable between Jetson variants. Empty and ``auto`` preserve the host's
    existing policy.
    """
    requested = _normalize_nvpmodel_id(value)
    if requested is None:
        return True
    if not re.fullmatch(r"[0-9]+", requested):
        logger.error(
            "%s must be a numeric nvpmodel id or 'auto'; ignoring %r",
            JETSON_NVP_MODEL_ENV,
            value,
        )
        return False
    if platform.system() != "Linux":
        logger.warning(
            "%s=%s ignored: nvpmodel is only available on Linux Jetson hosts",
            JETSON_NVP_MODEL_ENV,
            requested,
        )
        return False
    executable = shutil.which("nvpmodel")
    if executable is None:
        logger.warning(
            "%s=%s ignored: nvpmodel is unavailable on this host",
            JETSON_NVP_MODEL_ENV,
            requested,
        )
        return False
    supported_ids = supported_jetson_nvpmodel_ids()
    if supported_ids is None:
        logger.warning(
            "%s=%s ignored: unable to determine supported nvpmodel IDs on this host",
            JETSON_NVP_MODEL_ENV,
            requested,
        )
        return False
    if requested not in supported_ids:
        logger.warning(
            "%s=%s ignored: supported nvpmodel IDs are %s",
            JETSON_NVP_MODEL_ENV,
            requested,
            ", ".join(sorted(supported_ids, key=int)),
        )
        return False
    try:
        result = subprocess.run(
            [executable, "-m", requested],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not apply Jetson nvpmodel %s: %s", requested, exc)
        return False
    output = (result.stderr or "") + (result.stdout or "")
    if re.search(r"\breboot required\b", output, re.IGNORECASE):
        logger.warning(
            "Jetson nvpmodel %s requires a full reboot before it takes effect",
            requested,
        )
        return False
    if result.returncode != 0:
        logger.warning(
            "Jetson nvpmodel %s failed: %s",
            requested,
            (output or "unknown error").strip()[:500],
        )
        return False
    logger.info("Applied Jetson nvpmodel id %s before starting driver services", requested)
    return True
