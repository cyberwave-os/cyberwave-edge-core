"""Driver image selection based on platform and child asset registry IDs.

Extracted from startup.py — picks the best driver image and docker params
from the asset metadata ``drivers`` dict, considering the current OS/arch
and any child-twin registry overrides.
"""

from __future__ import annotations

import logging
import os
import platform
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
            return _extract(driver_name, driver_config)

    for platform_key in _resolve_platform_driver_keys():
        if platform_key not in drivers:
            continue
        return _extract(platform_key, drivers[platform_key])

    default_driver = drivers.get("default")
    return _extract("default", default_driver)


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

        specs.append(_ServiceSpec(
            image=image,
            name=name,
            command=command,
            env=env,
            params=params,
            prefer_gpu=bool(svc.get("prefer_gpu", False)),
            gpu_spec=str(svc.get("gpu", "all")),
        ))

    shared_env: dict[str, str] = {}
    raw_shared_env = config.get("shared_env")
    if isinstance(raw_shared_env, dict):
        shared_env = {str(k): str(v) for k, v in raw_shared_env.items()}

    shared_params: list[str] = []
    raw_shared_params = config.get("shared_params")
    if isinstance(raw_shared_params, list):
        shared_params = [str(p) for p in raw_shared_params]

    return specs, shared_env, shared_params
