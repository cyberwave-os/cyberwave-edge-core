"""What the edge remembers about staged driver updates, on disk.

Everything lives under ``CONFIG_DIR/driver-updates/``. It is a subdirectory with a
non-UUID name, so the twin-JSON reconcilers that own ``CONFIG_DIR/<uuid>.json`` never
touch it, and an edge-core restart keeps it.

* ``<request_id>.json`` -- a *staged record*: the images of a release the edge has
  pulled and verified, and what is needed to start them later. Written when staging
  finishes, so its presence means "staged".
* ``pulls.json`` -- when each image was last pulled, so the image garbage collector
  does not remove something that was pulled moments ago.
* ``previous.json`` -- the image set a twin ran before an update, kept for rollback until
  the new version has been healthy for a while (written by activation).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from .utils import atomic_write_text

logger = logging.getLogger(__name__)

PULLS_FILE = "pulls.json"
PREVIOUS_FILE = "previous.json"
#: How long a pull stays in ``pulls.json``; also the garbage collector's grace period.
RECENT_PULL_SECONDS = 2 * 60 * 60

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


_PULLS_LOCK = threading.Lock()


def updates_dir() -> Path:
    from . import startup

    return startup.CONFIG_DIR / "driver-updates"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable driver-update state %s: %s", path, exc)
        return None


def _write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True))


def is_valid_request_id(request_id: str) -> bool:
    """A plain name whose record file cannot be one of the bookkeeping files."""
    reserved = {PULLS_FILE, PREVIOUS_FILE}
    return (
        bool(_SAFE_ID.match(request_id))
        and request_id not in reserved
        and f"{request_id}.json" not in reserved
    )


def staged_record_path(request_id: str) -> Path:
    """Where the staged record of *request_id* lives; refuses an id that is not a plain name."""
    if not is_valid_request_id(request_id):
        raise ValueError(f"not a usable request_id: {request_id!r}")
    return updates_dir() / f"{request_id}.json"


def read_staged_record(request_id: str) -> Optional[dict[str, Any]]:
    data = _read_json(staged_record_path(request_id))
    return data if isinstance(data, dict) else None


def write_staged_record(record: Mapping[str, Any]) -> None:
    _write_json(staged_record_path(str(record["request_id"])), dict(record))


def read_staged_records() -> list[dict[str, Any]]:
    directory = updates_dir()
    if not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name in (PULLS_FILE, PREVIOUS_FILE):
            continue
        data = _read_json(path)
        if isinstance(data, dict):
            records.append(data)
    return records


def staged_records_unreadable() -> bool:
    """Whether some staged record cannot be read, so the set of staged images is unknown."""
    directory = updates_dir()
    if not directory.is_dir():
        return False
    return any(
        not isinstance(_read_json(path), dict)
        for path in directory.glob("*.json")
        if path.name not in (PULLS_FILE, PREVIOUS_FILE)
    )


def delete_staged_record(request_id: str) -> None:
    """Forget a staged update once it has been switched to, or has failed.

    Its images stop being protected as a staged update; the ones in use stay protected by
    their containers, and the old ones by the rollback set.
    """
    try:
        staged_record_path(request_id).unlink()
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        logger.warning("Could not delete staged record %s", request_id, exc_info=True)


def delete_superseded_records(record: Mapping[str, Any]) -> None:
    """Delete the other staged records of the same twin and package as *record*.

    A record that names no twin or no package is never matched: two unrelated ones would
    otherwise look like the same update.
    """
    if not record.get("twin_uuid") or not record.get("package"):
        return
    for other in read_staged_records():
        if (
            other.get("request_id") != record.get("request_id")
            and other.get("twin_uuid") == record.get("twin_uuid")
            and other.get("package") == record.get("package")
            and other.get("request_id")
        ):
            delete_staged_record(str(other["request_id"]))


def staged_images() -> set[str]:
    """Every image reference some staged record names."""
    return {
        str(image["image"])
        for record in read_staged_records()
        for image in record.get("images") or []
        if isinstance(image, dict) and image.get("image")
    }


def staged_digests() -> set[str]:
    """Every image digest some staged record names (``sha256:...``)."""
    return {
        str(image["digest"])
        for record in read_staged_records()
        for image in record.get("images") or []
        if isinstance(image, dict) and image.get("digest")
    }


def _fresh_pulls(moment: float) -> dict[str, float]:
    pulls = _read_json(updates_dir() / PULLS_FILE)
    if not isinstance(pulls, dict):
        return {}
    return {
        ref: when
        for ref, when in pulls.items()
        if isinstance(when, (int, float)) and moment - when < RECENT_PULL_SECONDS
    }


def record_pull(image: str, *, now: Optional[float] = None) -> bool:
    """Note that *image* was just pulled; False if it could not be written.

    Never raises: bookkeeping must not break a pull. Staging does check the result, since
    its up-front hold is what keeps the image cleanup off the images it is about to need.
    """
    try:
        moment = time.time() if now is None else now
        # Staging pulls several images at once; the read-modify-write must not lose one.
        with _PULLS_LOCK:
            pulls = _fresh_pulls(moment)
            pulls[image] = moment
            _write_json(updates_dir() / PULLS_FILE, pulls)
        return True
    except Exception:
        logger.warning("Could not record the pull of %s", image, exc_info=True)
        return False


def recent_pulls(*, now: Optional[float] = None) -> set[str]:
    return set(_fresh_pulls(time.time() if now is None else now))


def previous_image_sets(*, now: Optional[float] = None) -> set[str]:
    """Images kept for rollback whose keep-until time has not passed."""
    moment = time.time() if now is None else now
    previous = _read_json(updates_dir() / PREVIOUS_FILE)
    if not isinstance(previous, dict):
        return set()
    images: set[str] = set()
    for entry in previous.values():
        if isinstance(entry, dict) and float(entry.get("keep_until") or 0) > moment:
            images.update(str(image) for image in entry.get("images") or [])
    return images
