"""Hailo Gate 3 — host/HEF architecture compatibility preflight.

The Hailo SDK runtime tries to guard against architecture mismatch at HEF
load time (a Hailo-8 binary will not run on a Hailo-8L device, and vice
versa), but on HailoRT 4.23.0 the Python binding does not expose the HEF's
compiled arch via any of the historical attr names. The fallback is then
``vdevice.configure(hef, …)`` raising HailoRT's native incompatibility
error deep inside the worker container — which surfaces to the operator
as a cryptic "firmware handshake" log line.

Gate 3 catches that **before** Edge Core downloads the ``.hef`` from the
Hailo Model Zoo:

1. Edge Core fetches the catalog entry for ``<model_id>``.
2. The seeded ``metadata.hw_arch`` (``"hailo8"`` or ``"hailo8l"``) tells
   us which device the HEF was compiled for.
3. We probe the connected Hailo accelerator's architecture via the host
   ``hailortcli fw-control identify`` command.
4. If both are known and disagree → raise :class:`HailoArchMismatch` with
   a pointed message naming the correct sibling slug.

The check is intentionally **silent** in every situation where we cannot
make a confident negative determination:

* Non-Hailo models (anything other than ``.hef`` / ``edge_runtime ==
  "hailo"``) — Gate 3 is a no-op.
* No ``/dev/hailo0`` on the host (operator pre-staging a HEF for a
  device that will be attached later, or running Edge Core on a CI box)
  — skip with a debug log.
* ``hailortcli`` not installed on the host (HailoRT driver-only install,
  or the userspace package is missing) — skip with a one-line warning.
  The SDK's in-container fallback and the Gate 4 entrypoint still
  protect against actual incompatibilities at start time.
* Catalog metadata missing ``hw_arch`` — skip silently (older seeds /
  custom user uploads).

The cost of a single ``hailortcli fw-control identify`` invocation is
~30 ms on a Pi 5 + AI HAT+, paid only on the cache-miss download path.
"""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


#: Host device node exposed by the Hailo PCIe kernel driver. Kept as a
#: module-level constant so :func:`host_hailo_device_present` and the
#: worker_manager copy stay trivially aligned on the same path. (The
#: worker_manager exposes its own ``HAILO_DEVICE_PATH`` for backward
#: compatibility with mocking patterns in its tests; both point at the
#: same string and any change here must be mirrored there.)
HAILO_DEVICE_PATH = "/dev/hailo0"

#: ``hailortcli fw-control identify`` emits this line for every connected
#: device. Captured as a regex once at import time because the probe is on
#: the model-download fast path on cache miss.
_ARCH_LINE_RE = re.compile(r"Device Architecture:\s*(\S+)", re.IGNORECASE)

#: Hard upper bound on the time we wait for ``hailortcli`` to finish.
#: A healthy invocation returns in ~30 ms on a Pi 5 + AI HAT+; anything
#: longer is almost certainly a wedged driver and Edge Core should not
#: stall the download on it.
_HAILORTCLI_TIMEOUT_S = 3.0


class HailoArchMismatchError(RuntimeError):
    """Catalog HEF arch and connected Hailo accelerator arch disagree.

    Raised by :func:`preflight_hailo_arch` when both sides are known and
    they're different. The message names the offending model_id, the
    requested arch, and the device's actual arch so the operator can pick
    the correct sibling slug (``<base>_h8`` vs ``<base>_h8l``).
    """


def normalize_arch(raw: str) -> str:
    """Canonicalize arch strings emitted by hailortcli or HEFs.

    Inputs we've seen across HailoRT versions::

        "HAILO8"               → "hailo8"
        "Hailo-8"              → "hailo8"
        "HAILO_ARCH_HAILO8"    → "hailo8"
        "HAILO_ARCH_HAILO_8L"  → "hailo8l"
        ""                     → ""

    Strategy: lowercase, drop ``hailo_arch_`` prefix, strip non-alnum.
    Mirrors :func:`cyberwave.models.runtimes.hailo_rt._normalize_arch`
    in the SDK — kept duplicated here on purpose to avoid an edge-core →
    SDK import dependency just for one tiny string transform.
    """
    if not raw:
        return ""
    s = raw.strip().lower()
    s = re.sub(r"^hailo[_\-]?arch[_\-]?", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def host_hailo_device_present() -> bool:
    """Return True when ``/dev/hailo0`` exists on the host.

    Linux-only — HailoRT's PCIe driver only ships for Linux, and the
    device node never materializes on macOS / Windows. Returns False on
    non-Linux hosts without inspecting the filesystem.
    """
    if platform.system() != "Linux":
        return False
    return Path(HAILO_DEVICE_PATH).exists()


def host_hailo_arch() -> Optional[str]:
    """Return the normalized arch of the connected Hailo accelerator.

    Implementation calls ``hailortcli fw-control identify``, parses the
    ``Device Architecture: HAILO8[L]`` line, and normalizes the value via
    :func:`normalize_arch`. Returns ``None`` (no exception) when the
    arch cannot be determined for any reason — ``hailortcli`` missing
    from ``$PATH``, non-zero exit, parse miss, or timeout. The intent
    is "best-effort positive identification": :func:`preflight_hailo_arch`
    only raises when both sides report a value and they disagree.
    """
    cli = shutil.which("hailortcli")
    if not cli:
        return None

    try:
        completed = subprocess.run(
            [cli, "fw-control", "identify"],
            capture_output=True,
            text=True,
            timeout=_HAILORTCLI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("hailortcli fw-control identify failed: %s", exc)
        return None

    if completed.returncode != 0:
        logger.debug(
            "hailortcli fw-control identify exited %d: %s",
            completed.returncode,
            (completed.stderr or completed.stdout or "").strip()[:200],
        )
        return None

    match = _ARCH_LINE_RE.search(completed.stdout or "")
    if not match:
        logger.debug("hailortcli output did not include a 'Device Architecture' line")
        return None

    return normalize_arch(match.group(1))


def _extract_hef_arch(catalog_entry: dict[str, Any]) -> str:
    """Pull ``hw_arch`` out of a catalog response (top-level or metadata).

    Mirrors :func:`cyberwave_edge_core.model_manager._extract_runtime`'s
    "check both top-level and ``metadata.*``" pattern. Returns an empty
    string when the catalog has no opinion (older seeds, custom uploads)
    — the preflight then silently no-ops.
    """
    top = catalog_entry.get("hw_arch")
    if isinstance(top, str) and top.strip():
        return normalize_arch(top)
    metadata = catalog_entry.get("metadata") or {}
    if isinstance(metadata, dict):
        nested = metadata.get("hw_arch")
        if isinstance(nested, str) and nested.strip():
            return normalize_arch(nested)
    return ""


def _is_hailo_model(catalog_entry: dict[str, Any]) -> bool:
    """True iff *catalog_entry* describes a Hailo HEF artifact.

    Two independent signals — either is sufficient:

    * ``edge_runtime == "hailo"`` (catalog-driven), or
    * the filename / model_external_id ends in ``.hef`` (artifact-driven).

    Either is sufficient because some user-uploaded catalog rows may
    legitimately set one but not the other (e.g. a hand-written
    ``edge_runtime`` with an ambiguous extension, or a ``.hef`` upload
    that forgot to set ``edge_runtime``).
    """
    metadata = catalog_entry.get("metadata") or {}

    def _coalesce_str(*sources: Any) -> str:
        for src in sources:
            if isinstance(src, str) and src.strip():
                return src.strip().lower()
        return ""

    runtime = _coalesce_str(
        catalog_entry.get("edge_runtime"),
        catalog_entry.get("runtime"),
        metadata.get("edge_runtime") if isinstance(metadata, dict) else None,
        metadata.get("runtime") if isinstance(metadata, dict) else None,
    )
    if runtime == "hailo":
        return True

    name_sources = [
        catalog_entry.get("filename"),
        catalog_entry.get("model_external_id"),
    ]
    if isinstance(metadata, dict):
        name_sources.extend(
            [
                metadata.get("filename"),
                metadata.get("edge_model_path"),
            ]
        )
    for src in name_sources:
        if isinstance(src, str) and src.strip().lower().endswith(".hef"):
            return True

    return False


def preflight_hailo_arch(
    catalog_entry: dict[str, Any],
    model_id: str,
    *,
    sibling_slug_hint: Optional[str] = None,
) -> None:
    """Gate 3 — fail fast on a known HEF/device arch mismatch.

    Called from the model-manager download path *before* any bytes hit
    the wire. Silent in every situation where Edge Core cannot make a
    confident negative determination (see module docstring); raises
    :class:`HailoArchMismatchError` only when both the catalog and the
    host report a value and they disagree.

    Args:
        catalog_entry: the dict returned by
            :meth:`ModelManager._fetch_catalog_entry`. Inspected for
            ``edge_runtime``, filename, and ``hw_arch``.
        model_id: the human-readable model slug, used only in error
            messages so the operator knows which entry triggered the
            mismatch.
        sibling_slug_hint: optional. When provided, included in the
            error message as the recommended replacement (e.g.
            ``yolov8s_h8l`` when the user picked ``yolov8s_h8`` on a
            Hailo-8L host). When omitted we derive a best-effort hint
            from *model_id*'s ``_h8`` / ``_h8l`` suffix.
    """
    if not _is_hailo_model(catalog_entry):
        return

    if not host_hailo_device_present():
        logger.debug(
            "Hailo preflight: %s is a HEF but %s is not present on this host; "
            "skipping arch check (operator may be pre-staging for a later attach).",
            model_id,
            HAILO_DEVICE_PATH,
        )
        return

    hef_arch = _extract_hef_arch(catalog_entry)
    if not hef_arch:
        logger.debug(
            "Hailo preflight: catalog entry for %s has no metadata.hw_arch; "
            "skipping arch check (older seed / custom upload).",
            model_id,
        )
        return

    device_arch = host_hailo_arch()
    if not device_arch:
        logger.warning(
            "Hailo preflight: could not determine the connected accelerator's "
            "arch (hailortcli not installed or probe failed). Continuing with "
            "download for %s; the SDK and Gate-4 entrypoint will still catch "
            "mismatches at worker start time.",
            model_id,
        )
        return

    if device_arch != hef_arch:
        sibling = sibling_slug_hint or _suggest_sibling_slug(model_id, device_arch)
        raise HailoArchMismatchError(
            f"Hailo HEF '{model_id}' was compiled for {hef_arch!r}, but the "
            f"connected accelerator at {HAILO_DEVICE_PATH} reports {device_arch!r}. "
            f"Pick the matching catalog sibling"
            f"{f' ({sibling!r})' if sibling else ''} and redeploy."
        )

    logger.info(
        "Hailo preflight: %s arch %s matches connected accelerator. OK.",
        model_id,
        hef_arch,
    )


def _suggest_sibling_slug(model_id: str, device_arch: str) -> str:
    """Best-effort recommendation for the correct sibling slug.

    The seed catalog suffixes Hailo entries with ``_h8`` / ``_h8l`` (see
    ``seed_models._yolov8_hailo_catalog_entries``). When the operator
    picked the wrong sibling we can usually flip the suffix in-place;
    when the slug doesn't match the convention we return an empty
    string and the error message degrades gracefully.
    """
    base, _, suffix = model_id.rpartition("_")
    if suffix == "h8" and device_arch == "hailo8l":
        return f"{base}_h8l"
    if suffix == "h8l" and device_arch == "hailo8":
        return f"{base}_h8"
    return ""
