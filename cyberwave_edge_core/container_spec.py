"""Resolved container specs and the ``io.cyberwave.edge.*`` labels derived from them.

Every container edge-core creates for a driver or the ML worker carries:

- ``io.cyberwave.edge.twin``: the twin the container serves (drivers only);
- ``io.cyberwave.edge.service``: the service of a multi-service driver, or
  ``worker`` for the ML worker (absent for a single-service driver);
- ``io.cyberwave.edge.spec-hash``: sha256 of a canonical JSON of the resolved
  spec, see :meth:`DockerRunSpec.canonical`;
- ``io.cyberwave.edge.created-by``: ``cyberwave-edge-core/<version>``.

The spec hash exists so a later edge-core process can tell whether a running
container still matches what it would create now, and keep it instead of
recreating it (adoption, edge-core auto-update Phase D / EC6). It is therefore
computed from the *same* data that becomes the ``docker run`` argv:
:meth:`DockerRunSpec.run_argv` renders the argv and :meth:`DockerRunSpec.spec_hash`
hashes the options, image, command and post-create networks it holds. A flag
added to the argv is covered by construction, with no list of hashed fields to
keep in step.

The canonical form normalises only what Docker treats as unordered, so an
edge-core version that merely reorders flags does not restart every driver:

- the relative order of different flags is ignored (options are grouped by flag);
- short aliases collapse onto their long form (``-e`` / ``--env``), and
  ``--flag=value`` is the same as ``--flag value``;
- ``--env`` and ``--label`` become key -> value maps with Docker's last-wins rule;
- repeatable, set-like flags (mounts, devices, capabilities, ...) are sorted;
- every other repeated flag keeps its order, since Docker's last value wins.

The edge labels themselves are excluded: the hash cannot cover itself, and the
``created-by`` version must not change the hash, or every edge-core update
would look like a spec change.

Only the hash is stored. Environment values (API tokens included) never reach a
label; the hash of a document containing a high-entropy token reveals nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from . import __version__

EDGE_LABEL_PREFIX = "io.cyberwave.edge."
TWIN_LABEL = f"{EDGE_LABEL_PREFIX}twin"
SERVICE_LABEL = f"{EDGE_LABEL_PREFIX}service"
SPEC_HASH_LABEL = f"{EDGE_LABEL_PREFIX}spec-hash"
CREATED_BY_LABEL = f"{EDGE_LABEL_PREFIX}created-by"

WORKER_SERVICE = "worker"

# Bump when the canonical form changes meaning; the new hash then differs from
# every hash an older edge-core wrote, and adoption recreates those containers.
SPEC_SCHEMA_VERSION = 1

_FLAG_ALIASES = {
    "-a": "--attach",
    "-c": "--cpu-shares",
    "-d": "--detach",
    "-e": "--env",
    "-h": "--hostname",
    "-i": "--interactive",
    "-l": "--label",
    "-m": "--memory",
    "-p": "--publish",
    "-P": "--publish-all",
    "-t": "--tty",
    "-u": "--user",
    "-v": "--volume",
    "-w": "--workdir",
}

# Repeatable flags whose occurrences form a set: Docker applies all of them and
# their order carries no meaning, so they are sorted.
_SET_FLAGS = frozenset(
    {
        "--add-host",
        "--cap-add",
        "--cap-drop",
        "--device",
        "--device-cgroup-rule",
        "--dns",
        "--dns-option",
        "--dns-search",
        "--env-file",
        "--expose",
        "--group-add",
        "--label-file",
        "--link",
        "--log-opt",
        "--mount",
        "--publish",
        "--security-opt",
        "--storage-opt",
        "--sysctl",
        "--tmpfs",
        "--ulimit",
        "--volume",
        "--volumes-from",
    }
)

# Flags whose values are ``KEY[=VALUE]`` and follow Docker's last-wins rule.
_MAP_FLAGS = frozenset({"--env", "--label"})


def created_by() -> str:
    """Value of the ``created-by`` label for this edge-core build."""
    return f"cyberwave-edge-core/{__version__}"


def _group_options(options: Sequence[str]) -> list[tuple[str, Optional[str]]]:
    """Split flat option tokens into ``(flag, value)`` pairs.

    A token starting with ``-`` is a flag. ``--flag=value`` carries its value
    inline; otherwise the next token is the value unless it is itself a flag,
    in which case this one is a boolean switch (value ``None``). The image is
    not part of *options*, so no list of boolean flags is needed to find it.
    A value that is not preceded by a flag (never produced by edge-core) is
    kept as a ``("", value)`` pair so it still counts towards the hash.
    """
    pairs: list[tuple[str, Optional[str]]] = []
    index = 0
    while index < len(options):
        token = options[index]
        if not token.startswith("-"):
            pairs.append(("", token))
            index += 1
            continue
        if token.startswith("--") and "=" in token:
            flag, _, value = token.partition("=")
            pairs.append((_FLAG_ALIASES.get(flag, flag), value))
            index += 1
            continue
        flag = _FLAG_ALIASES.get(token, token)
        nxt = options[index + 1] if index + 1 < len(options) else None
        if nxt is not None and not nxt.startswith("-"):
            pairs.append((flag, nxt))
            index += 2
        else:
            pairs.append((flag, None))
            index += 1
    return pairs


def _is_edge_label(value: Optional[str]) -> bool:
    return value is not None and value.startswith(EDGE_LABEL_PREFIX)


def canonical_options(options: Sequence[str]) -> dict[str, Any]:
    """Canonical, order-normalised form of ``docker run`` option tokens."""
    grouped: dict[str, list[Optional[str]]] = {}
    for flag, value in _group_options(options):
        if flag == "--label" and _is_edge_label(value):
            continue
        grouped.setdefault(flag, []).append(value)

    canonical: dict[str, Any] = {}
    for flag, values in grouped.items():
        if flag in _MAP_FLAGS:
            mapping: dict[str, Optional[str]] = {}
            for raw in values:
                key, sep, val = (raw or "").partition("=")
                mapping[key] = val if sep else None
            canonical[flag] = mapping
        elif flag in _SET_FLAGS:
            canonical[flag] = sorted(values, key=lambda v: (v is None, v or ""))
        else:
            canonical[flag] = values
    return canonical


@dataclass(frozen=True)
class DockerRunSpec:
    """Everything edge-core passes to Docker when it creates one container.

    *options* are the ``docker run`` flags and their values, exactly as they go
    on the command line (without the edge labels, which this class adds).
    *extra_networks* are attached with ``docker network connect`` between
    create and start. *image_id* is the local image ID (``sha256:...``) the
    reference resolved to, when known: a moving tag such as ``:dev`` re-pulled
    onto a new build keeps its reference but not its ID.
    """

    options: tuple[str, ...]
    image: str
    command: tuple[str, ...] = ()
    extra_networks: tuple[str, ...] = ()
    image_id: Optional[str] = None

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": SPEC_SCHEMA_VERSION,
            "image": self.image,
            "image_id": self.image_id,
            "options": canonical_options(self.options),
            "command": list(self.command),
            "extra_networks": sorted(self.extra_networks),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def spec_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def labels(
        self, *, twin_uuid: Optional[str] = None, service: Optional[str] = None
    ) -> dict[str, str]:
        labels: dict[str, str] = {}
        if twin_uuid:
            labels[TWIN_LABEL] = twin_uuid
        if service:
            labels[SERVICE_LABEL] = service
        labels[SPEC_HASH_LABEL] = self.spec_hash()
        labels[CREATED_BY_LABEL] = created_by()
        return labels

    def run_argv(
        self, *, twin_uuid: Optional[str] = None, service: Optional[str] = None
    ) -> list[str]:
        """The ``docker run`` argv for this spec, edge labels included."""
        label_args: list[str] = []
        for key, value in self.labels(twin_uuid=twin_uuid, service=service).items():
            label_args += ["--label", f"{key}={value}"]
        return ["docker", "run", *self.options, *label_args, self.image, *self.command]
