"""Driver image selection based on platform and child asset registry IDs.

Extracted from startup.py — picks the best driver image and docker params
from the asset metadata ``drivers`` dict, considering the current OS/arch
and any child-twin registry overrides.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import platform
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_jetson_detected: Optional[bool] = None


def is_jetson() -> bool:
    """Detect NVIDIA Jetson hardware via ``/etc/nv_tegra_release``.

    Also honours the ``CYBERWAVE_PLATFORM_VARIANT=jetson`` env override.
    """
    global _jetson_detected
    if _jetson_detected is not None:
        return _jetson_detected

    override = os.environ.get("CYBERWAVE_PLATFORM_VARIANT", "").strip().lower()
    if override == "jetson":
        _jetson_detected = True
        return True

    _jetson_detected = Path("/etc/nv_tegra_release").exists()
    return _jetson_detected


def _platform_driver_keys() -> list[str]:
    """Profile keys this host matches, most specific first.

    Module level so every resolver shares ONE answer to "which profile is this
    host". It used to be nested inside `_get_best_driver_image_and_params`, which
    was fine while that was the only caller; a second copy is the drift
    `select_driver_image` warns about -- a cloud node reading the Jetson variant
    of a profile is silent, because both variants look plausible.
    """
    system_name = platform.system().lower()
    machine_name = platform.machine().lower()

    platform_aliases: list[str]
    if system_name == "darwin":
        platform_aliases = ["darwin", "macos", "mac", "osx"]
    elif system_name == "linux":
        platform_aliases = ["linux"]
    elif system_name == "windows":
        platform_aliases = ["windows", "win32"]
    else:
        platform_aliases = [system_name]

    keys: list[str] = []

    if machine_name and system_name == "linux" and is_jetson():
        keys.append(f"linux-{machine_name}-jetson")

    if machine_name:
        keys.extend(f"{alias}-{machine_name}" for alias in platform_aliases)

    keys.extend(platform_aliases)
    return keys


def _matched_driver_profile(
    drivers: Dict[str, Dict[str, Any]],
    child_registry_ids: Optional[set[str]] = None,
) -> tuple[str, Dict[str, Any]]:
    """The profile in *drivers* this host resolves to, and the key it was found under.

    Resolution order, unchanged: a child-registry match, then the platform keys
    above, then ``default``. Returned rather than consumed so callers that want
    something other than the image -- ``select_driver_shared_env`` -- read the
    SAME profile the image came from instead of re-deriving it.
    """
    normalized_child_registry_ids = {
        registry_id.strip()
        for registry_id in (child_registry_ids or set())
        if isinstance(registry_id, str) and registry_id.strip()
    }
    if normalized_child_registry_ids and len(drivers) > 1:
        for driver_name, driver_config in drivers.items():
            if driver_name == "default":
                continue
            if driver_name not in normalized_child_registry_ids:
                continue
            return driver_name, driver_config

    for platform_key in _platform_driver_keys():
        if platform_key in drivers:
            return platform_key, drivers[platform_key]

    return "default", drivers.get("default")  # type: ignore[return-value]


def select_driver_shared_env(
    drivers: Dict[str, Dict[str, Any]],
    *,
    child_registry_ids: Optional[set[str]] = None,
) -> Dict[str, str]:
    """``shared_env`` the matched profile declares, or ``{}``.

    WHY THIS IS PUBLIC. A twin's ``drivers`` metadata could always carry
    ``shared_env``, and `_get_driver_services` applies it -- but only on the
    inline-``services`` form. The image-declared path returns
    ``(specs, {}, [])`` by construction, so a twin migrated from ``services`` to
    a bare ``docker_image`` had its ``shared_env`` SILENTLY dropped. That cost a
    day: the Go2's ``WAIT_FOR_START_MAPPING: "false"`` disappeared, the sim
    variant's ``${WAIT_FOR_START_MAPPING:-true}`` default took over, and SLAM sat
    behind a start gate that nothing opened while the map simply never appeared.

    Deliberately env ONLY, and deliberately not merged here. The image owns its
    graph -- that is the point of declaring it -- so this hands back values for a
    caller to layer UNDER the declaration, letting the compose file opt a
    parameter in with `${NAME:-default}` and keeping every other key the
    declaration's own. Returning the whole profile would invite injecting
    volumes or commands past the translator's allow-list.

    Values are stringified: compose placeholders are expanded by a shell, where
    everything is a string, and a JSON number would otherwise reach `--env` as
    `1.0` or `True` depending on the writer's language.
    """
    if not isinstance(drivers, dict) or not drivers:
        return {}
    try:
        _name, config = _matched_driver_profile(drivers, child_registry_ids)
    except Exception:
        return {}
    if not isinstance(config, dict):
        return {}
    raw = config.get("shared_env")
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): ("true" if value is True else "false" if value is False else str(value))
        for key, value in raw.items()
        if value is not None
    }


def _get_best_driver_image_and_params(
    drivers: Dict[str, Dict[str, Any]],
    child_registry_ids: Optional[set[str]] = None,
) -> tuple[str, list[str], bool, str]:
    """Select the best driver image for this platform.

    Returns ``(docker_image, params, prefer_gpu, gpu_spec)`` where
    *prefer_gpu* is a hint that the driver benefits from ``--gpus``
    when an NVIDIA runtime is available, and *gpu_spec* controls which
    GPUs are exposed (``"all"`` by default, or a count/device selector
    like ``"1"`` or ``"device=0,1"``).

    "drivers": {
        "default": {
            "docker_image": "helloworld",
            "version": "0.1.0",
            "params": ["--param1", "--param2"],
            "prefer_gpu": true,
            "gpu": "all"
        },
        "linux-aarch64-jetson": {
            "docker_image": "helloworld:jetson-humble",
            "params": ["--param1", "--param2"],
            "prefer_gpu": true,
            "gpu": 1
        },
    },
    """

    def _extract(
        driver_name: str,
        driver_config: Any,
    ) -> tuple[str, list[str], bool, str]:
        if not isinstance(driver_config, dict):
            raise ValueError(f"Invalid config for driver '{driver_name}'")
        if not driver_config.get("docker_image") or not isinstance(
            driver_config["docker_image"], str
        ):
            raise ValueError(f"No docker_image specified for driver '{driver_name}'")
        raw_params = driver_config.get("params")
        if raw_params is None:
            params: list[str] = []
        elif isinstance(raw_params, list) and all(isinstance(param, str) for param in raw_params):
            params = raw_params
        else:
            raise ValueError(f"Invalid params for driver '{driver_name}'")
        prefer_gpu = bool(driver_config.get("prefer_gpu", False))
        gpu_spec = str(driver_config.get("gpu", "all"))
        return driver_config["docker_image"], params, prefer_gpu, gpu_spec

    # Selection lives in `_matched_driver_profile` so `select_driver_shared_env`
    # reads the same profile this image came from.
    driver_name, driver_config = _matched_driver_profile(drivers, child_registry_ids)
    return _extract(driver_name, driver_config)


# ---------------------------------------------------------------------------
# Multi-container service support
# ---------------------------------------------------------------------------


@dataclass
class _ServiceSpec:
    """Describes one service within a multi-container driver stack."""

    image: str
    name: str
    command: list[str] | None = None
    env: dict[str, str] = field(default_factory=dict)
    params: list[str] = field(default_factory=list)
    prefer_gpu: bool = False
    gpu_spec: str = "all"
    # Networks beyond the first, attached after create. `docker create` honours
    # only one --network before Docker 25, and the simulation plant proxy has to
    # span two: the cyberwave-sim Zenoh network and the stack's own segment.
    extra_networks: list[str] = field(default_factory=list)
    # ``{dependency service name: compose condition}``. The launcher waits on
    # Docker's health state for ``service_healthy``; anything else is already
    # satisfied by starting in dependency order.
    depends_on: dict[str, str] = field(default_factory=dict)


def _get_driver_services(
    drivers: Dict[str, Dict[str, Any]],
    child_registry_ids: Optional[set[str]] = None,
) -> tuple[list[_ServiceSpec], dict[str, str], list[str]] | None:
    """Extract multi-container service definitions from the drivers dict.

    Returns ``(services, shared_env, shared_params)`` when the matched
    platform config contains a ``services`` array, or ``None`` when the
    config uses the legacy single-image ``docker_image`` key.

    The platform resolution order is identical to
    :func:`_get_best_driver_image_and_params`.
    """

    def _resolve_platform_driver_keys() -> list[str]:
        system_name = platform.system().lower()
        machine_name = platform.machine().lower()

        platform_aliases: list[str]
        if system_name == "darwin":
            platform_aliases = ["darwin", "macos", "mac", "osx"]
        elif system_name == "linux":
            platform_aliases = ["linux"]
        elif system_name == "windows":
            platform_aliases = ["windows", "win32"]
        else:
            platform_aliases = [system_name]

        keys: list[str] = []

        if machine_name and system_name == "linux" and is_jetson():
            keys.append(f"linux-{machine_name}-jetson")

        if machine_name:
            keys.extend(f"{alias}-{machine_name}" for alias in platform_aliases)

        keys.extend(platform_aliases)
        return keys

    def _match_config() -> Dict[str, Any] | None:
        normalized_child_registry_ids = {
            rid.strip()
            for rid in (child_registry_ids or set())
            if isinstance(rid, str) and rid.strip()
        }
        if normalized_child_registry_ids and len(drivers) > 1:
            for driver_name, driver_config in drivers.items():
                if driver_name == "default":
                    continue
                if driver_name in normalized_child_registry_ids and isinstance(driver_config, dict):
                    return driver_config

        for platform_key in _resolve_platform_driver_keys():
            cfg = drivers.get(platform_key)
            if isinstance(cfg, dict):
                return cfg

        cfg = drivers.get("default")
        return cfg if isinstance(cfg, dict) else None

    config = _match_config()
    if config is None or "services" not in config:
        return None

    raw_services = config["services"]
    if not isinstance(raw_services, list) or not raw_services:
        return None

    specs: list[_ServiceSpec] = []
    for idx, svc in enumerate(raw_services):
        if not isinstance(svc, dict):
            raise ValueError(f"services[{idx}] is not a dict")
        image = svc.get("image")
        name = svc.get("name")
        if not image or not isinstance(image, str):
            raise ValueError(f"services[{idx}] missing required 'image' string")
        if not name or not isinstance(name, str):
            raise ValueError(f"services[{idx}] missing required 'name' string")

        raw_cmd = svc.get("command")
        command: list[str] | None = None
        if raw_cmd is not None:
            if isinstance(raw_cmd, list) and all(isinstance(c, str) for c in raw_cmd):
                command = raw_cmd
            else:
                raise ValueError(f"services[{idx}].command must be a list of strings")

        raw_env = svc.get("env")
        env: dict[str, str] = {}
        if raw_env is not None:
            if isinstance(raw_env, dict):
                env = {str(k): str(v) for k, v in raw_env.items()}
            else:
                raise ValueError(f"services[{idx}].env must be a dict")

        raw_params = svc.get("params")
        params: list[str] = []
        if raw_params is not None:
            if isinstance(raw_params, list) and all(isinstance(p, str) for p in raw_params):
                params = raw_params
            else:
                raise ValueError(f"services[{idx}].params must be a list of strings")

        specs.append(
            _ServiceSpec(
                image=image,
                name=name,
                command=command,
                env=env,
                params=params,
                prefer_gpu=bool(svc.get("prefer_gpu", False)),
                gpu_spec=str(svc.get("gpu", "all")),
            )
        )

    shared_env: dict[str, str] = {}
    raw_shared_env = config.get("shared_env")
    if isinstance(raw_shared_env, dict):
        shared_env = {str(k): str(v) for k, v in raw_shared_env.items()}

    shared_params: list[str] = []
    raw_shared_params = config.get("shared_params")
    if isinstance(raw_shared_params, list):
        shared_params = [str(p) for p in raw_shared_params]

    return specs, shared_env, shared_params


# ---------------------------------------------------------------------------
# Image-declared topology (Docker Compose labels)
# ---------------------------------------------------------------------------
#
# A driver image can declare its own service graph as rendered Compose JSON in a
# label, so twin metadata only has to name the image. CI renders each variant
# with `docker compose config --no-interpolate --format json`, which validates
# the file, expands YAML anchors and normalises `depends_on` — the edge needs no
# Compose CLI, only a base64 decode and a JSON parse.
#
# Metadata still wins: `_get_driver_services` is consulted first, so a twin that
# carries a `services` array behaves exactly as before and this path stays
# dormant until that array is removed. See CYB-3469.

_COMPOSE_LABEL_PREFIX = "io.cyberwave.driver.compose"
_COMPOSE_SCHEMA_LABEL = f"{_COMPOSE_LABEL_PREFIX}.schema"
_CHANNEL_TAG_LABEL = "io.cyberwave.image.channel-tag"
# CANONICAL EXPLANATION -- the Dockerfile, the build workflow and the two test
# suites point here rather than restating it.
#
# Schema 2 carries the rendered graph base64-encoded, because
# `docker/build-push-action` parses `build-args` as CSV and strips the quotes off
# a field that is entirely a quoted token (`"-lc"` in a compose `command`), which
# truncated schema 1's raw JSON at the first such token. base64's alphabet has
# neither character, so the payload is inert to that parser and any other. The
# unencoded compose files ship at /opt/cyberwave/ for a human to read.
_SUPPORTED_COMPOSE_SCHEMA = 2

# Plain ``${NAME}`` only. Compose's ``${NAME:-default}`` form is left alone so
# it reaches the container shell that is meant to expand it.
_PLAIN_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Which Compose service keys this module actually acts on. A key outside both
# sets is NOT ignored -- it is refused, because ignoring it is the failure this
# exists to prevent: `docker compose config` in CI validates the file, so an
# author who writes `privileged: true`, `devices:` or `cap_add:` gets a green
# build, a green label, and a container on the robot that quietly does not have
# what it asked for. A container missing its USB device still reports healthy,
# so nothing downstream notices either.
#
# The primary gate is a build-time assertion over the checked-in files
# (`drivers/tests/unitree/test_go2_compose_topology.py`); this runtime check is
# the backstop for an image built before that assertion existed. Verified
# against the rendered output of `docker compose config --no-interpolate
# --format json` on both variants (Compose v5.1.1): it injects nothing at
# service level, so every key reaching this function was written by hand.
_TRANSLATED_SERVICE_KEYS = frozenset(
    {
        "image",
        "command",
        "environment",
        "networks",
        "network_mode",
        "volumes",
        "healthcheck",
        "depends_on",
        "stop_signal",
        "x-cyberwave",
    }
)
# Declared in the file, deliberately overridden by Edge Core, and so safe to
# ignore rather than refuse -- the launcher sets its own restart policy, names
# its own containers and owns log rotation. Two owners for one setting is worse
# than one, and the launcher is the one that has to reconcile the result.
_EDGE_CORE_OWNED_SERVICE_KEYS = frozenset({"restart", "container_name", "logging"})

# The variant Edge Core resolves for a physical twin. Simulation callers pass
# their own; there is deliberately no auto-detection, because guessing wrong
# starts the wrong plant.
DEFAULT_COMPOSE_VARIANT = "robot"

# The variant is a property of the ORCHESTRATOR, not the twin: edge-core runs on
# a robot, simulation on a cloud node, and neither hosts the other. Deliberately
# never derived from twin metadata -- guessing wrong starts a sim proxy against a
# real robot, or a hardware driver reaching for a radio that is not there.
# Unknown values are not rejected here (a non-ROS stack may declare its own); a
# typo surfaces at the label lookup, which logs it and falls back to metadata.
COMPOSE_VARIANT_ENV = "CYBERWAVE_DRIVER_COMPOSE_VARIANT"

# Which release train the SIBLING images resolve to. `${CW_CHANNEL_TAG}` normally
# comes from the driver image's own `io.cyberwave.image.channel-tag` label, which
# is right in production but wrong for a test that pins ONE image: pinning
# `go2-ros2-driver:humble-pr-123-sha-abc` leaves the label saying `humble-dev`,
# so three of four images are not the ones under test and the run reports green
# for a graph it never assembled. Set this to pin the whole graph to a train.
CHANNEL_TAG_ENV = "CYBERWAVE_DRIVER_CHANNEL_TAG"

# The SECOND train, and the reason it needs one of its own.
#
# `${CW_CHANNEL_TAG}` names the driver stack: go2-ros2-driver, ros2-nav2 and
# ros2-slam are built together by edge-ros2-go2-driver-build-and-push.yml, whose
# tags carry a ROS distro (`humble-pr-<n>-sha-<sha12>`, `jetson-humble`, ...).
# ros2-sim-proxy is not in that stack. It is built by
# ros2-sim-runtime-build-and-push.yml alongside the sim runtime, and its tags
# have no distro dimension at all -- `pr-<n>-<sha12>` on a PR, the moving
# `<branch>` tag on a push to dev/staging/production.
#
# So no rendering of CW_CHANNEL_TAG can ever name a tag ros2-sim-proxy
# publishes, and pointing the sim graph's `plant` at it asked for an image that
# by construction does not exist. That is not a config mistake to be worked
# around per environment -- it is one variable standing for two release trains.
SIM_PROXY_TAG_ENV = "CYBERWAVE_SIM_PROXY_TAG"

# Where the proxy tag comes from when nothing pins it: the deployment
# environment, because that is exactly what the proxy's branch pushes publish.
# The same set, for the same reason, as `ros2_autonomy.resolve_image()` uses to
# pick the sim runtime image -- the other image on this train.
_PUBLISHED_PROXY_ENVIRONMENTS = frozenset({"dev", "staging", "production"})

# `dev` rather than `latest`, which is never published for this image.
_FALLBACK_PROXY_TAG = "dev"


def _image_labels(image: str) -> Optional[Dict[str, str]]:
    """Every label on a *locally present* image, or None when it is not there.

    One ``docker image inspect`` for all three labels rather than one each: three
    CLI+daemon round-trips per twin per reconcile pass, each with its own 15s
    timeout, to pull three strings off one image. A dict lookup also returns
    "absent" on its own, where a Go template ``index`` renders the literal
    ``<no value>`` that every caller had to fold in by hand.

    None, not ``{}``, on a non-zero exit: "not on this host" and "here and
    carries no labels" are different answers, and only the second means the
    image genuinely declares nothing.
    """
    try:
        proc = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                image,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        parsed = json.loads(proc.stdout or "null")
    except json.JSONDecodeError:
        logger.warning("Image %s returned unparseable labels", image)
        return None
    # `.Config.Labels` is JSON null for an image with no labels at all.
    if not isinstance(parsed, dict):
        return {}
    return {str(k): "" if v is None else str(v) for k, v in parsed.items()}


def _substitute(value: str, subs: Dict[str, str]) -> str:
    """Expand ``${NAME}`` placeholders and collapse Compose's ``$$`` escape.

    Only the names in *subs* are expanded. Anything else — ``${ENABLE_RVIZ2:-false}``
    inside a service command, for instance — must survive untouched so the
    container's own shell expands it at runtime.

    Names not in *subs* fall back to Edge Core's own environment, because that
    is what Compose would have done at ``up`` time and we deferred interpolation
    to here — ``${CYBERWAVE_SIM_DOCKER_NETWORK}`` in a network name has to become
    a real network. The pattern deliberately does not match Compose's
    default syntax, so ``${ENABLE_RVIZ2:-false}`` inside a service command
    survives for the container's own shell to expand.

    An unset name is left as the literal ``${NAME}`` rather than blanked the way
    Compose would: an obviously wrong ``--network ${...}`` is far easier to
    diagnose in a docker error than an empty argument.

    ``$$`` is Compose's escape for a literal ``$``; Compose collapses it when it
    starts a container, so a launcher that skips Compose has to do it or a
    healthcheck reaches the container with a literal double dollar.
    """
    for name, replacement in subs.items():
        value = value.replace(f"${{{name}}}", replacement)

    def _from_env(match: "re.Match[str]") -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if resolved is None:
            logger.warning(
                "Compose placeholder ${%s} is unset in edge-core's environment; leaving it literal",
                name,
            )
            return match.group(0)
        return resolved

    value = _PLAIN_PLACEHOLDER.sub(_from_env, value)
    return value.replace("$$", "$")


def _compose_env(raw: Any) -> Dict[str, str]:
    """Normalise a Compose ``environment`` block.

    ``docker compose config`` emits a mapping when no override touched the key
    and a ``["K=v"]`` list when one did, so both shapes arrive from the same
    file depending on which variant was layered on.
    """
    if isinstance(raw, dict):
        return {str(k): "" if v is None else str(v) for k, v in raw.items()}
    out: Dict[str, str] = {}
    for item in raw or []:
        key, _, value = str(item).partition("=")
        if key:
            out[key] = value
    return out


def _compose_healthcheck_params(
    healthcheck: Any,
    subs: Dict[str, str],
) -> list[str]:
    """Translate a Compose ``healthcheck`` into ``docker create --health-*`` flags.

    Without this the healthchecks a Compose file declares are inert on the edge:
    Docker reports no health state, so ``depends_on: {condition:
    service_healthy}`` has nothing to wait on and every dependent starts as soon
    as its dependency is *created*.

    ``config`` normalises durations to strings like ``"5s"``, which the CLI
    accepts verbatim. ``test`` arrives as a list whose first element is the kind:

      ``NONE``       -> disable any healthcheck baked into the image
      ``CMD-SHELL``  -> one shell string, exactly what ``--health-cmd`` runs
      ``CMD``        -> exec form, which the CLI cannot express; joined into a
                        shell string, which differs only for arguments that need
                        quoting. Compose files here use CMD-SHELL.
    """
    if not isinstance(healthcheck, dict) or not healthcheck:
        return []
    if healthcheck.get("disable"):
        return ["--no-healthcheck"]

    raw_test = healthcheck.get("test")
    if isinstance(raw_test, str):
        test = ["CMD-SHELL", raw_test]
    elif isinstance(raw_test, list):
        test = [str(part) for part in raw_test]
    else:
        return []
    if not test:
        return []

    kind, rest = test[0], test[1:]
    if kind == "NONE":
        return ["--no-healthcheck"]
    if kind in {"CMD-SHELL", "CMD"}:
        if not rest:
            return []
        command = rest[0] if kind == "CMD-SHELL" and len(rest) == 1 else shlex.join(rest)
    else:
        # No kind prefix at all: Compose treats the whole list as a shell string.
        command = shlex.join(test)

    params = ["--health-cmd", _substitute(command, subs)]
    for key, flag in (
        ("interval", "--health-interval"),
        ("timeout", "--health-timeout"),
        ("start_period", "--health-start-period"),
        ("start_interval", "--health-start-interval"),
    ):
        value = healthcheck.get(key)
        if value:
            params += [flag, _substitute(str(value), subs)]
    retries = healthcheck.get("retries")
    if retries is not None:
        params += ["--health-retries", str(int(retries))]
    return params


def _compose_dependency_conditions(service: Dict[str, Any]) -> dict[str, str]:
    """Map ``depends_on`` to ``{dependency: condition}``.

    ``config`` always renders the long form, but the short list form is still
    valid input, and a bare list means ``service_started`` — which is what the
    launcher does anyway, so it needs no wait.
    """
    raw = service.get("depends_on")
    if isinstance(raw, list):
        return {str(name): "service_started" for name in raw}
    if not isinstance(raw, dict):
        return {}
    conditions: dict[str, str] = {}
    for name, spec in raw.items():
        if isinstance(spec, dict):
            conditions[str(name)] = str(spec.get("condition") or "service_started")
        else:
            conditions[str(name)] = "service_started"
    return conditions


def _compose_service_params(
    service: Dict[str, Any],
    networks: Dict[str, Any],
    subs: Dict[str, str],
) -> tuple[list[str], list[str]]:
    """Translate the Compose fields Edge Core's launcher understands.

    Returns ``(docker flags, networks to attach after create)``. Networking, bind
    mounts and the healthcheck are read: everything else about the container
    (restart policy, log rotation, naming) is Edge Core's to decide, and letting
    the Compose file override those would give two owners for one setting.
    """
    params: list[str] = []
    extra_networks: list[str] = []

    network_mode = service.get("network_mode")
    if isinstance(network_mode, str) and network_mode:
        params += ["--network", _substitute(network_mode, subs)]
    else:
        # `networks: [simnet]` renders as {"simnet": None}; the real Docker
        # network name lives on the top-level networks block, which is where an
        # `external: true` name has to be looked up.
        # The first network is the one `docker create` takes; the rest are
        # connected between create and start, because the simulation plant proxy
        # deliberately spans two (Zenoh out to the cyberwave-sim plant, DDS in to
        # the ROS graph) so physics and the ROS graph never share a segment.
        #
        # "First" here is NOT the author's choice, and must not be read as one.
        # `docker compose config` renders `networks:` as a mapping sorted
        # ALPHABETICALLY, not in file order -- v5.1.1 renders
        # `networks: [zzz_first, aaa_second]` as `{"aaa_second": ..., "zzz_first":
        # ...}`. This loop's `index == 0` therefore only means "the one network
        # `docker create` can take before Docker 25", which is an artefact of the
        # CLI, not a declaration.
        #
        # Nothing in the Go2 graph depends on which of a service's networks
        # becomes eth0: both are NAT'd bridges, so egress works either way, and
        # Zenoh and DDS each discover on their own segment. `plant` resolves
        # simnet first only because `simnet` sorts before `stacknet` -- a
        # coincidence, which `test_the_plant_declares_simnet_first` in
        # drivers/tests/unitree/test_go2_compose_topology.py pins so a rename
        # that flips it fails the build instead of silently swapping eth0.
        #
        # If a stack ever genuinely needs to choose its default route, the
        # control is `--network name=x,gw-priority=N` (Docker 28+), not the order
        # of this loop.
        for index, alias in enumerate(service.get("networks") or {}):
            declared = networks.get(alias) or {}
            name = declared.get("name") if isinstance(declared, dict) else None
            resolved = _substitute(str(name or alias), subs)
            if index == 0:
                params += ["--network", resolved]
            else:
                extra_networks.append(resolved)

    for volume in service.get("volumes") or []:
        if not isinstance(volume, dict):
            continue
        source = volume.get("source")
        target = volume.get("target")
        if not source or not target:
            continue
        mount = f"{_substitute(str(source), subs)}:{_substitute(str(target), subs)}"
        if volume.get("read_only"):
            mount += ":ro"
        params += ["-v", mount]

    params += _compose_healthcheck_params(service.get("healthcheck"), subs)

    # WHY A DECLARATION NEEDS THIS. Docker's default stop signal is SIGTERM, and
    # the ROS 2 launcher does not act on it: measured on the Go2 sim graph, PID 1
    # `ros2 launch` ignores SIGTERM and is SIGKILLed when `docker stop` gives up
    # (21s), while the same process on SIGINT shuts its node tree down in ~1s and
    # exits 0. Every service in that graph paid the full timeout, so a teardown
    # took 95 seconds and each node died abruptly rather than shutting down --
    # which for `slam` means a map database killed mid-flush.
    #
    # Declared per service rather than baked into the images because it is a
    # property of what PID 1 is, and that is what the `command` here decides.
    stop_signal = service.get("stop_signal")
    if stop_signal:
        params += ["--stop-signal", _substitute(str(stop_signal), subs)]

    return params, extra_networks


def _order_by_dependencies(
    services: Dict[str, Dict[str, Any]],
) -> list[str]:
    """Service names in an order that starts every dependency first.

    Required, not a nicety: ``docker compose config`` emits services
    alphabetically, so following document order would start ``bridges`` before
    ``plant``. Health *gating* is a separate follow-up — this only fixes the
    sequence.

    A dependency that is not part of this variant is ignored rather than fatal:
    the assertions in the driver tests already reject that, and refusing to boot
    a robot over it would be a worse failure than starting in a slightly wrong
    order. A cycle raises, because there is no defensible order for one.
    """
    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in ordered or name not in services:
            return
        if name in visiting:
            raise ValueError(f"depends_on cycle through service {name!r}")
        visiting.add(name)
        for dependency in services[name].get("depends_on") or {}:
            visit(dependency)
        visiting.discard(name)
        ordered.append(name)

    for name in services:
        visit(name)
    return ordered


def _specs_from_compose(
    doc: Dict[str, Any],
    subs: Dict[str, str],
) -> tuple[list[_ServiceSpec], dict[str, str], list[str]] | None:
    """Map a rendered Compose document onto ``_ServiceSpec`` objects.

    Returns the same ``(services, shared_env, shared_params)`` triple as
    :func:`_get_driver_services` so the launch path downstream is identical. The
    Compose document carries no shared env or params of its own — anchors are
    already expanded into each service by ``config`` — so both are empty and
    every value is per-service.
    """
    raw_services = doc.get("services")
    if not isinstance(raw_services, dict) or not raw_services:
        return None
    networks = doc.get("networks") if isinstance(doc.get("networks"), dict) else {}

    specs: list[_ServiceSpec] = []
    for name in _order_by_dependencies(raw_services):
        service = raw_services[name]
        if not isinstance(service, dict):
            raise ValueError(f"compose service {name!r} is not a mapping")
        unsupported = sorted(
            set(service) - _TRANSLATED_SERVICE_KEYS - _EDGE_CORE_OWNED_SERVICE_KEYS
        )
        if unsupported:
            raise ValueError(
                f"compose service {name!r} declares {unsupported}, which Edge Core "
                "does not translate. Refusing the whole graph rather than starting "
                "it without them: a container that silently lacks a device, a "
                "capability or a privilege still reports healthy, so the loss "
                "would surface as unexplained robot behaviour instead of an error."
            )

        image = service.get("image")
        if not image or not isinstance(image, str):
            raise ValueError(f"compose service {name!r} has no image")

        raw_command = service.get("command")
        command: list[str] | None = None
        if isinstance(raw_command, list):
            command = [_substitute(str(part), subs) for part in raw_command]
        elif isinstance(raw_command, str) and raw_command:
            # Compose allows a bare string; Edge Core always execs a list.
            command = ["/bin/sh", "-c", _substitute(raw_command, subs)]

        extension = service.get("x-cyberwave") or {}
        if not isinstance(extension, dict):
            extension = {}

        params, extra_networks = _compose_service_params(service, networks, subs)

        specs.append(
            _ServiceSpec(
                image=_substitute(image, subs),
                name=name,
                command=command,
                env={
                    key: _substitute(value, subs)
                    for key, value in _compose_env(service.get("environment")).items()
                },
                params=params,
                prefer_gpu=bool(extension.get("prefer_gpu", False)),
                gpu_spec=str(extension.get("gpu", "all")),
                extra_networks=extra_networks,
                depends_on=_compose_dependency_conditions(service),
            )
        )

    return specs, {}, []


def select_driver_image(
    drivers: Dict[str, Dict[str, Any]],
    *,
    child_registry_ids: Optional[set[str]] = None,
) -> str:
    """The driver image a twin's ``drivers`` metadata names for THIS host.

    The public entry point for orchestrators outside Edge Core that resolve the
    same twin metadata. cyberwave-sim needs it: the graph it starts is declared
    by the driver image's Compose labels, so it has to name the image before it
    can read them, and it must pick the same one the robot would rather than
    reimplement the platform matching (the drift that would cause — a cloud node
    reading labels off the Jetson variant of an image — is silent, because both
    variants declare a plausible graph).

    Same resolution order as the robot path: a child-registry match, then
    platform-specific keys (``linux-aarch64-jetson``, ``linux-x86_64``,
    ``linux``), then ``default``.

    Raises ``ValueError`` when the matched profile names no single
    ``docker_image`` — which includes a profile written in the multi-service
    ``services`` form, whose flat list of containers has no variants and so
    cannot answer "which image declares this graph".
    """
    image, _params, _prefer_gpu, _gpu = _get_best_driver_image_and_params(
        drivers, child_registry_ids=child_registry_ids
    )
    return image


def _resolve_compose_variant(variant: Optional[str]) -> tuple[str, str]:
    """Return ``(variant name, where it came from)``.

    Precedence: an explicit argument, then ``CYBERWAVE_DRIVER_COMPOSE_VARIANT``,
    then :data:`DEFAULT_COMPOSE_VARIANT`. The source is returned so the caller
    can log it — when a twin comes up with the wrong set of containers, "which
    variant did this process pick, and why" is the first question.
    """
    if variant:
        return variant, "argument"
    from_env = os.environ.get(COMPOSE_VARIANT_ENV, "").strip()
    if from_env:
        return from_env, COMPOSE_VARIANT_ENV
    return DEFAULT_COMPOSE_VARIANT, "default"


def _label_forensics(payload: str, *, edge: int = 60) -> str:
    """Length, digest and both ends of a label that failed to parse.

    Only ever called on a failure path. A healthy run must not print this: the
    decoded graph is kilobytes, and the CI step that surfaces these logs is
    size-capped -- burying a real verdict under a base64 dump is the exact
    failure mode that cost this suite two diagnostic rounds already.

    Length is the primary signal. Truncation and quote-stripping both show up as
    a payload that is *nearly* right, and comparing the number against what CI
    rendered settles in one read what a JSONDecodeError offset only hints at. The
    digest lets two sides be compared without shipping the payload itself.
    """
    digest = hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:12]
    if len(payload) <= edge * 2:
        return f"len={len(payload)} sha256={digest} body={payload!r}"
    return f"len={len(payload)} sha256={digest} head={payload[:edge]!r} tail={payload[-edge:]!r}"


def _resolve_channel_tag(
    image: str,
    labels: Dict[str, str],
    channel_tag: Optional[str] = None,
) -> tuple[str, str]:
    """Return ``(channel tag, where it came from)``.

    Precedence: an explicit argument, then :data:`CHANNEL_TAG_ENV`, then the
    image's own ``io.cyberwave.image.channel-tag`` label, then the tag in the
    image reference. The source is returned so the caller can log it -- when a
    graph comes up on the wrong images, "which train did the siblings resolve to,
    and why" is the question, and the answer is otherwise invisible: every
    sibling is a plausible image that pulls and runs.

    The image's own tag is a last resort rather than a peer of the label. They
    disagree exactly when an image is re-tagged -- a PR build tagged
    `humble-pr-123-sha-abc` still carries `channel-tag=humble-dev` -- and in
    production the label is the truthful one, because it records the train the
    image was BUILT in rather than the name someone gave it afterwards.
    """
    if channel_tag:
        return channel_tag, "argument"
    from_env = os.environ.get(CHANNEL_TAG_ENV, "").strip()
    if from_env:
        return from_env, CHANNEL_TAG_ENV
    from_label = labels.get(_CHANNEL_TAG_LABEL, "")
    if from_label:
        return from_label, _CHANNEL_TAG_LABEL
    if ":" in image:
        return image.rsplit(":", 1)[1], "image reference"
    return "", "unset"


def _resolve_sim_proxy_tag(sim_proxy_tag: Optional[str] = None) -> tuple[str, str]:
    """Return ``(sim-proxy tag, where it came from)``.

    Precedence mirrors :func:`_resolve_channel_tag` -- explicit argument, then
    the env var, then a default -- so the two trains are resolved and logged the
    same way. The source is returned for the same reason: when a graph comes up
    on the wrong images, "which train, and why" is the question, and every
    sibling is a plausible image that pulls and runs.

    NO IMAGE LABEL in the chain, unlike the channel tag. The label
    ``io.cyberwave.image.channel-tag`` belongs to the driver image whose graph is
    being read, and it records the DRIVER's train; ros2-sim-proxy is a different
    image from a different workflow, so that label can say nothing about it.
    Reading it here is precisely the bug this function exists to fix.

    The default is the deployment environment because that is what the proxy's
    branch pushes publish. An unknown or unset environment falls back to ``dev``
    rather than ``latest``, which is never published -- the same trade
    ``ros2_autonomy.resolve_image()`` makes for the sim runtime image.
    """
    if sim_proxy_tag:
        return sim_proxy_tag, "argument"
    from_env = os.environ.get(SIM_PROXY_TAG_ENV, "").strip()
    if from_env:
        return from_env, SIM_PROXY_TAG_ENV
    environment = (
        (os.environ.get("CYBERWAVE_ENVIRONMENT") or os.environ.get("ENVIRONMENT") or "")
        .strip()
        .lower()
    )
    if environment in _PUBLISHED_PROXY_ENVIRONMENTS:
        return environment, "CYBERWAVE_ENVIRONMENT"
    return _FALLBACK_PROXY_TAG, "default"


def compose_substitutions(
    *, channel_tag: str, twin_uuid: str = "", sim_proxy_tag: Optional[str] = None
) -> Dict[str, str]:
    """Every ``${NAME}`` a rendered declaration may carry, and what it becomes.

    One function because there are two callers, and when they were two literals
    they drifted: the sim-scenario harness resolves the same declaration to
    start the plant itself, and a key added here but not there reached a
    container as the LITERAL ``${CW_TWIN_NS}``. Nothing rejected it -- it became
    a ROS namespace of that name, and the first thing to notice was a topic
    probe failing on ``//odom``.

    ``CW_SIM_PROXY_TAG`` is the sim plant's image tag. It is NOT
    ``CW_CHANNEL_TAG``: ros2-sim-proxy is built by a different workflow, with a
    tag grammar that carries no ROS distro, so the driver stack's channel never
    names a tag it publishes. See :func:`_resolve_sim_proxy_tag`.

    ``CW_TWIN_NS`` is the ROS namespace every service in a twin's graph shares,
    built by exactly the transformation the driver images' ``docker-entrypoint``
    applies to CYBERWAVE_TWIN_UUID -- lowercase, dashes to underscores, ``twin_``
    prefix. A service whose image has that entrypoint needs nothing; one whose
    image does not (the sim plant runs ros2-sim-proxy) sets ROS_NAMESPACE from
    this.
    """
    substitutions = {"CW_CHANNEL_TAG": channel_tag}
    # A SECOND train, resolved separately -- see `_resolve_sim_proxy_tag`. Always
    # present rather than conditional: an unrendered `${CW_SIM_PROXY_TAG}` would
    # reach docker as a literal image tag, which is the same class of failure as
    # the `${CW_TWIN_NS}` namespace above, and just as quiet.
    substitutions["CW_SIM_PROXY_TAG"] = _resolve_sim_proxy_tag(sim_proxy_tag)[0]
    if twin_uuid:
        substitutions["CW_TWIN8"] = twin_uuid[:8]
        substitutions["CW_TWIN_NS"] = "twin_" + twin_uuid.lower().replace("-", "_")
    return substitutions


def get_image_declared_services(
    image: str,
    *,
    variant: Optional[str] = None,
    twin_uuid: str = "",
    channel_tag: Optional[str] = None,
) -> tuple[list[_ServiceSpec], dict[str, str], list[str]] | None:
    """Service graph *image* declares for *variant*, or None when it declares none.

    *image* must already be pulled — the labels are read with
    ``docker image inspect``, so this runs after the driver image is on disk and
    before the sibling images it names are pulled.

    *twin_uuid* resolves ``${CW_TWIN8}``, which the simulation variant uses to
    name its per-twin ROS network. Same 8-character prefix Edge Core already uses
    for container names, so the two stay legible side by side in ``docker ps``.
    It also resolves ``${CW_TWIN_NS}`` -- ``twin_<uuid with underscores>``, the
    ROS namespace the graph shares.

    *channel_tag* resolves ``${CW_CHANNEL_TAG}``, the train the sibling images
    come from. Normally left to the image's own label; pass it (or set
    :data:`CHANNEL_TAG_ENV`) to pin a whole graph to one train, which a test that
    pins a single image must do or it silently tests one image out of four.

    None rather than an exception for every "this image says nothing" case: an
    image built before these labels existed is not an error, it simply falls
    through to the twin-metadata path.
    """
    labels = _image_labels(image)
    if labels is None:
        return None
    schema_raw = labels.get(_COMPOSE_SCHEMA_LABEL, "")
    if not schema_raw:
        return None
    try:
        schema = int(schema_raw)
    except ValueError:
        schema = 0
    if schema != _SUPPORTED_COMPOSE_SCHEMA:
        # Refuse rather than guess. A newer image may use x-cyberwave keys this
        # edge-core would silently drop, and starting a partial graph is worse
        # than falling back to metadata.
        logger.error(
            "Image %s declares compose schema %r; this edge-core supports %d. "
            "Falling back to twin metadata.",
            image,
            schema_raw,
            _SUPPORTED_COMPOSE_SCHEMA,
        )
        return None

    name, source = _resolve_compose_variant(variant)
    encoded = labels.get(f"{_COMPOSE_LABEL_PREFIX}.{name}", "")
    raw = ""
    if encoded:
        try:
            raw = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            logger.exception(
                "Image %s has a compose label for variant %r that is not base64 (%s)",
                image,
                name,
                _label_forensics(encoded),
            )
            return None
    if not raw or raw == "{}":
        # Naming the source tells the two very different causes apart: an image
        # that genuinely ships only one variant, versus a typo in
        # CYBERWAVE_DRIVER_COMPOSE_VARIANT on this host.
        logger.error(
            "Image %s declares compose schema %d but no variant %r (from %s)",
            image,
            schema,
            name,
            source,
        )
        return None

    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        # The traceback carries the character offset; on its own that says where
        # the reader gave up and nothing about why. This label reached a consumer
        # 88 characters short -- `docker/build-push-action` CSV-parses build-args,
        # and a comma-delimited field that is ENTIRELY one quoted token loses its
        # quotes, so 44 quote pairs vanished and the JSON failed at exactly the
        # first stripped pair. Length plus the two ends make that shape visible
        # here instead of requiring the image config blob to be pulled by hand,
        # and the digest lets a reader compare against what CI rendered.
        logger.exception(
            "Image %s has a malformed compose label for variant %r (%s)",
            image,
            name,
            _label_forensics(raw),
        )
        return None
    if not isinstance(doc, dict):
        logger.error("Image %s compose label for %r is not an object", image, name)
        return None

    channel_tag, channel_source = _resolve_channel_tag(image, labels, channel_tag)
    proxy_tag, proxy_source = _resolve_sim_proxy_tag()
    substitutions = compose_substitutions(
        channel_tag=channel_tag, twin_uuid=twin_uuid, sim_proxy_tag=proxy_tag
    )

    try:
        resolved = _specs_from_compose(doc, substitutions)
    except ValueError:
        logger.exception("Image %s declares an unusable compose graph for %r", image, name)
        return None

    if resolved is not None:
        logger.info(
            "Using image-declared topology from %s (variant=%s from %s, channel=%s "
            "from %s, sim-proxy=%s from %s): %s",
            image,
            name,
            source,
            channel_tag or "<unset>",
            channel_source,
            proxy_tag,
            proxy_source,
            ", ".join(spec.name for spec in resolved[0]),
        )
    return resolved
