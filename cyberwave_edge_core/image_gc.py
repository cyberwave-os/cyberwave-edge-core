"""Remove the Docker images nothing needs, and keep the ones a staged update will.

This replaces ``docker image prune --all --filter until=2h``. That prune compares an
image's *creation* time, so an image pulled a moment ago but built days ago was fair
game, and the images of a staged update (pulled now, started whenever the operator
switches, possibly days later) would have been removed by the next periodic cleanup.

An image is removed only when **all** of these hold, so this never removes more than the
old prune did:

* no container, running or stopped, uses it;
* it was created more than two hours ago (the old guard for fresh local builds);
* no staged record names it, and it was not pulled in the last two hours;
* it is not part of a rollback set that is still being kept (unless *aggressive*, used
  when the disk is short).

Oldest first, so a failure part-way leaves the freshest images alone.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from . import driver_update_state as state
from .docker_helpers import docker_available, parse_docker_timestamp_ns

logger = logging.getLogger(__name__)

#: The old prune's ``until=2h`` guard.
MIN_AGE_SECONDS = 2 * 60 * 60
_DOCKER_TIMEOUT_SECONDS = 120


@dataclass
class _Image:
    id: str
    created: float
    tags: list[str] = field(default_factory=list)
    digests: list[str] = field(default_factory=list)

    def names(self) -> set[str]:
        return {_familiar(name) for name in (*self.tags, *self.digests)}


_DEFAULT_REGISTRIES = ("docker.io/", "index.docker.io/", "registry-1.docker.io/")


def _familiar(reference: str) -> str:
    """*reference* the way Docker lists it: no default registry, ``library/``, or ``:latest``.

    A manifest may spell an image ``docker.io/org/x:1`` or ``x``; ``docker image ls`` shows
    ``org/x:1`` and ``x:latest``. Comparing the raw strings would leave such an image
    unprotected.
    """
    ref = reference.strip()
    for prefix in _DEFAULT_REGISTRIES:
        if ref.startswith(prefix):
            ref = ref[len(prefix) :]
            break
    if ref.startswith("library/"):
        ref = ref[len("library/") :]
    if "@" not in ref and ":" not in ref.rsplit("/", 1)[-1]:
        ref += ":latest"
    return ref


def _docker(*args: str) -> str:
    result = subprocess.run(
        ["docker", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=_DOCKER_TIMEOUT_SECONDS,
    )
    return result.stdout


def _parse_created(value: str) -> float:
    """Epoch seconds of Docker's ``.Created``; 0 if unreadable.

    Zero reads as "very old", so an image Docker gave an odd timestamp is still collectable.
    """
    nanoseconds = parse_docker_timestamp_ns(value)
    return 0.0 if nanoseconds is None else float(nanoseconds // 1_000_000_000)


def _list_images() -> list[_Image]:
    ids = _docker("image", "ls", "--all", "--quiet", "--no-trunc").split()
    ids = list(dict.fromkeys(ids))
    if not ids:
        return []
    out = _docker(
        "image",
        "inspect",
        "--format",
        "{{.Id}}\t{{.Created}}\t{{json .RepoTags}}\t{{json .RepoDigests}}",
        *ids,
    )
    images: list[_Image] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        try:
            tags = json.loads(parts[2]) or []
            digests = json.loads(parts[3]) or []
        except ValueError:
            continue
        images.append(
            _Image(
                id=parts[0].strip(),
                created=_parse_created(parts[1]),
                tags=[str(tag) for tag in tags],
                digests=[str(digest) for digest in digests],
            )
        )
    return images


def _images_in_use() -> set[str]:
    containers = _docker("ps", "--all", "--quiet", "--no-trunc").split()
    if not containers:
        return set()
    return set(_docker("inspect", "--format", "{{.Image}}", *containers).split())


def _protected_ids(
    images: Iterable[_Image], references: set[str], digests: frozenset[str] = frozenset()
) -> set[str]:
    """IDs of the images any of *references* (a tag or a digest reference) names.

    An image whose repo digest is one of *digests* is protected too, whatever its tags say:
    a tag can move, or be spelled differently, and the digest is what staging verified.
    """
    wanted = {_familiar(reference) for reference in references}
    return {
        image.id
        for image in images
        if image.names() & wanted
        or any(digest.endswith(f"@{d}") for digest in image.digests for d in digests)
    }


def gc_driver_images(
    *,
    aggressive: bool = False,
    now: Optional[float] = None,
    keep: Iterable[str] = (),
) -> bool:
    """Remove unused images as described in the module docstring; True when it could run.

    *aggressive* also drops rollback sets that are still inside their keep period, for a
    disk that is short. Staged images, images in use and images pulled in the last two
    hours are never removed, nor the images *keep* names (an image ID or reference).
    """
    if not docker_available():
        return False
    moment = time.time() if now is None else now
    try:
        images = _list_images()
        in_use = _images_in_use()
    except subprocess.CalledProcessError as exc:
        logger.warning("Docker image cleanup failed: %s", exc)
        return False
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Docker image cleanup error: %s", exc)
        return False

    if state.staged_records_unreadable():
        # Without every record the keep-set is incomplete, and a missing one would let the
        # images it names be removed.
        logger.warning("Docker image cleanup skipped: a staged update record is unreadable")
        return False

    kept = state.staged_images() | state.recent_pulls(now=moment)
    if not aggressive:
        kept |= state.previous_image_sets(now=moment)
    keep = {str(item) for item in keep if item}
    kept |= {item for item in keep if not item.startswith("sha256:")}
    protected = (
        _protected_ids(images, kept, frozenset(state.staged_digests()))
        | in_use
        | {item for item in keep if item.startswith("sha256:")}
    )

    removable = sorted(
        (
            image
            for image in images
            if image.id not in protected and moment - image.created >= MIN_AGE_SECONDS
        ),
        key=lambda image: image.created,
    )
    removed = 0
    for image in removable:
        # An image with several tags is gone only once every tag is.
        targets = image.tags or [image.id]
        try:
            _docker("rmi", *targets)
            removed += 1
        except subprocess.CalledProcessError as exc:
            logger.info("Not removing image %s: %s", image.tags or image.id, exc.stderr or exc)
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Removing image %s failed: %s", image.tags or image.id, exc)
            break
    logger.info(
        "Docker image cleanup: removed %d of %d unused image(s)%s",
        removed,
        len(removable),
        " (aggressive)" if aggressive else "",
    )
    return True
