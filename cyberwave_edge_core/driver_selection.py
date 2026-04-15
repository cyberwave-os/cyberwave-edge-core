"""Driver image selection based on platform and child asset registry IDs.

Extracted from startup.py — picks the best driver image and docker params
from the asset metadata ``drivers`` dict, considering the current OS/arch
and any child-twin registry overrides.
"""

from __future__ import annotations

import platform
from typing import Any, Dict, Optional


def _get_best_driver_image_and_params(
    drivers: Dict[str, Dict[str, Any]],
    child_registry_ids: Optional[set[str]] = None,
) -> tuple[str, list[str]]:
    """
    Given a list of drivers specified in the metadata of the asset,
    and given the hardware where the edge is running,
    Returns:
    - The best driver to run.
    - A list of parameters to pass to the driver when doing docker run
    If any non-default driver key matches one of the child asset registry IDs,
    that driver is preferred over ``default``.

    "drivers": {
        "default": {
            "docker_image": "helloworld",
            "version": "0.1.0",
            "params": ["--param1", "--param2"],
        },
        "mac": {
            "docker_image": "helloworld",
            "version": "0.1.0",
            "params": ["--param1", "--param2"],
        },
    },
    """
    def _extract_driver_image_and_params(
        driver_name: str,
        driver_config: Any,
    ) -> tuple[str, list[str]]:
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
        return driver_config["docker_image"], params

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

        if machine_name:
            machine_specific = [f"{alias}-{machine_name}" for alias in platform_aliases]
            return machine_specific + platform_aliases
        return platform_aliases

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
            return _extract_driver_image_and_params(driver_name, driver_config)

    for platform_key in _resolve_platform_driver_keys():
        if platform_key not in drivers:
            continue
        return _extract_driver_image_and_params(platform_key, drivers[platform_key])

    default_driver = drivers.get("default")
    return _extract_driver_image_and_params("default", default_driver)
