"""Driver image selection based on platform and child asset registry IDs.

Extracted from startup.py — picks the best driver image and docker params
from the asset metadata ``drivers`` dict, considering the current OS/arch
and any child-twin registry overrides.
"""

from __future__ import annotations

import logging
import os
import platform
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
