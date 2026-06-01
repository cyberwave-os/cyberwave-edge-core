"""Driver container launch helpers."""

from __future__ import annotations

import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import time
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _startup():
    from . import startup as _startup_mod

    return _startup_mod


def _run_macos_device_bridge_commands(
    *,
    params: list[str],
    twin_uuid: str,
    container_name: str,
    additional_device_mappings: Optional[list[tuple[str, str]]] = None,
    usbip_active: bool = False,
) -> tuple[bool, dict[str, str]]:
    """Best-effort macOS host-bridge hook for linux-only ``--device`` mappings.

    If ``CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND`` is set, each extracted device
    mapping triggers one command execution. Supported template variables:
      - ``{host_device}``
      - ``{container_device}``
      - ``{twin_uuid}``
      - ``{container_name}``
      - ``{config_dir}``

    When *usbip_active* is True, ``/dev/video*`` mappings are skipped because
    USB/IP handles them transparently via the container entrypoint.
    """
    s = _startup()
    if platform.system() != "Darwin":
        return True, {}

    explicit_device_mappings = s._extract_docker_device_mappings(params)
    device_mappings: list[tuple[str, str]] = []
    usbip_handled_video_devices: dict[str, str] = {}
    seen_mappings: set[tuple[str, str]] = set()
    for mapping in explicit_device_mappings + (additional_device_mappings or []):
        if mapping in seen_mappings:
            continue
        seen_mappings.add(mapping)
        host_device, container_device = mapping
        if usbip_active and (
            s._is_video_device_path(host_device) or s._is_video_device_path(container_device)
        ):
            logger.info(
                "USB/IP active — skipping bridge command for video device %s:%s "
                "(will be attached via USB/IP in container entrypoint)",
                host_device,
                container_device,
            )
            usbip_handled_video_devices[container_device] = container_device
            continue
        device_mappings.append(mapping)
    if not device_mappings:
        return True, usbip_handled_video_devices

    bridge_command_template = (
        s.get_runtime_env_var("CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND", "") or ""
    ).strip()
    if not bridge_command_template:
        logger.warning(
            "Driver uses --device mappings on macOS but CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND "
            "is not configured; hardware access will likely fail"
        )
        resolved = {container: container for _, container in device_mappings}
        resolved.update(usbip_handled_video_devices)
        return True, resolved

    resolved_device_map: dict[str, str] = dict(usbip_handled_video_devices)

    for host_device, container_device in device_mappings:
        try:
            rendered_command = bridge_command_template.format(
                host_device=host_device,
                container_device=container_device,
                twin_uuid=twin_uuid,
                container_name=container_name,
                config_dir=str(s.CONFIG_DIR),
            )
        except Exception as exc:
            logger.error(
                "Invalid CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND template %r: %s",
                bridge_command_template,
                exc,
            )
            return False, {}

        try:
            command_parts = shlex.split(rendered_command)
            if not command_parts:
                logger.error(
                    "CYBERWAVE_MACOS_DEVICE_BRIDGE_COMMAND rendered to an empty command"
                )
                return False, {}
            result = subprocess.run(
                command_parts,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            resolved_device_map[container_device] = s._parse_bridge_resolved_device(
                result.stdout or "",
                fallback=container_device,
            )
        except (
            ValueError,
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            logger.error(
                "Failed to run macOS hardware bridge command for device %s (%s): %s",
                host_device,
                container_device,
                exc,
            )
            return False, {}
    return True, resolved_device_map


def _run_docker_image(
    image: str,
    params: list[str],
    *,
    twin_uuid: str,
    token: str,
    child_camera_twin_uuids: Optional[list[str]] = None,
    macos_bridge_device_candidates: Optional[list[str]] = None,
    skip_pull: bool = False,
    prefer_gpu: bool = False,
    gpu_spec: str = "all",
    service_name: str | None = None,
    command: list[str] | None = None,
    service_env: dict[str, str] | None = None,
    driver_alert_ctx: Optional[Any] = None,
) -> bool:
    """Run a driver Docker container for a twin.

    When *skip_pull* is False (the default) the image is pulled first.
    Set *skip_pull* to True when images have already been fetched by an
    earlier parallel-pull phase.

    *driver_alert_ctx*, when provided, reuses an alert that was already
    created during the parallel-pull phase so the user sees a continuous
    ``driver_starting`` lifecycle.  When ``None`` a fresh alert is
    created (backwards-compatible with callers that skip the bulk pull).

    The container is started in detached mode with ``--restart unless-stopped``
    so it persists across reboots.  Environment variables are passed so the
    driver can authenticate with the Cyberwave backend and know which twin it
    controls.

    Returns ``True`` if the container was started successfully.
    """
    s = _startup()
    if not shutil.which("docker"):
        logger.error("Docker is not installed or not in PATH")
        return False

    if service_name:
        container_name = f"cyberwave-driver-{twin_uuid[:8]}-{service_name}"
    else:
        container_name = f"cyberwave-driver-{twin_uuid[:8]}"
    image = s._resolve_driver_image_tag(image)
    params = s._ensure_linux_microphone_docker_params(image, params)
    runtime_environment = (
        s.get_runtime_env_var("CYBERWAVE_ENVIRONMENT", s.DEFAULT_ENVIRONMENT) or s.DEFAULT_ENVIRONMENT
    ).lower()

    # Remove any existing container with the same name (idempotent re-runs)
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=30,
    )

    if driver_alert_ctx is None:
        driver_alert_ctx = s.DriverStartingAlertContext(
            twin_uuid=twin_uuid,
            image=image,
            service_name=service_name,
        )
        driver_alert_ctx.create()

    if skip_pull:
        if not s._docker_image_exists_locally(image):
            logger.error("Image %s not available locally (skip_pull=True)", image)
            driver_alert_ctx.mark_failed_and_resolve(
                f"Driver image {image} not available locally after pull phase.",
                phase="image_missing",
            )
            return False
        driver_alert_ctx.update_metadata({"phase": "pull_skipped"}, force=True)
    else:
        try:
            s._pull_docker_image_with_progress(
                image,
                container_name=container_name,
                twin_uuid=twin_uuid,
                token=token,
                driver_alert_ctx=driver_alert_ctx,
            )
        except subprocess.CalledProcessError as exc:
            err_tail = (exc.stderr or "").strip() or "unknown error"
            if s._docker_image_exists_locally(image):
                logger.warning(
                    "Failed to pull docker image %s (%s); using local image copy",
                    image,
                    err_tail,
                )
                driver_alert_ctx.update_metadata(
                    {
                        "phase": "pull_failed_using_local",
                        "last_error": err_tail[:500],
                    },
                    force=True,
                )
            else:
                logger.error("Failed to pull docker image %s: %s", image, exc.stderr)
                driver_alert_ctx.mark_failed_and_resolve(
                    f"Failed to pull driver image {image}. {err_tail[:300]}",
                    phase="pull_failed",
                )
                return False
        except subprocess.TimeoutExpired:
            if s._docker_image_exists_locally(image):
                logger.warning(
                    "Docker pull timed out for image %s; using local image copy",
                    image,
                )
                driver_alert_ctx.update_metadata(
                    {"phase": "pull_timeout_using_local"}, force=True
                )
            else:
                logger.error("Docker pull timed out for image: %s", image)
                driver_alert_ctx.mark_failed_and_resolve(
                    f"Timed out pulling driver image {image}.",
                    phase="pull_timeout",
                )
                return False
        except OSError as exc:
            if s._docker_image_exists_locally(image):
                logger.warning(
                    "Docker pull OS error for image %s; using local image copy: %s",
                    image,
                    exc,
                )
                driver_alert_ctx.update_metadata(
                    {"phase": "pull_oserror_using_local", "last_error": str(exc)[:500]},
                    force=True,
                )
            else:
                logger.error("Docker pull failed for image %s: %s", image, exc)
                driver_alert_ctx.mark_failed_and_resolve(
                    f"Could not pull driver image {image}: {exc}",
                    phase="pull_oserror",
                )
                return False
        else:
            # NOTE: deliberately not "pull_complete" — the frontend treats that
            # as a terminal phase and would briefly drop the spinner between
            # pull-finished and container-running.  ``starting_container`` keeps
            # the spinner on through the docker-run + health-probe window.
            driver_alert_ctx.update_metadata(
                {"phase": "starting_container"}, force=True
            )

    # Build env vars for the container
    container_env: dict[str, str] = {
        "CYBERWAVE_TWIN_UUID": twin_uuid,
        "CYBERWAVE_API_KEY": token,
        "CYBERWAVE_EDGE_HOST_PLATFORM": platform.system().lower(),
    }
    if child_camera_twin_uuids:
        normalized_child_uuids = [str(child_uuid).strip() for child_uuid in child_camera_twin_uuids]
        normalized_child_uuids = [child_uuid for child_uuid in normalized_child_uuids if child_uuid]
        if normalized_child_uuids:
            child_uuids_csv = ",".join(dict.fromkeys(normalized_child_uuids))
            container_env["CYBERWAVE_CHILD_TWIN_UUIDS"] = child_uuids_csv

    explicit_params_env = s._extract_docker_env_map(params)

    # On Linux, read the selected camera device from cameras.json so camera
    # drivers open the correct /dev/video* instead of defaulting to index 0.
    if platform.system() == "Linux" and "CYBERWAVE_METADATA_VIDEO_DEVICE" not in explicit_params_env:
        selected_video_device = s._load_selected_camera_device(twin_uuid)
        if selected_video_device is not None:
            container_env.setdefault("CYBERWAVE_METADATA_VIDEO_DEVICE", selected_video_device)

    macos_bridge_mappings = s._normalize_macos_bridge_candidates(macos_bridge_device_candidates)

    # Determine USB/IP state early so the bridge function can skip video
    # devices that USB/IP will handle transparently inside the container.
    usbip_active = platform.system() == "Darwin" and s._is_usbip_server_running()

    # Check for an explicit MJPEG camera stream URL early.  When the user has
    # configured one, video device bridge resolution is pointless because the
    # driver will consume the HTTP stream instead of /dev/video*.
    #
    # Resolution order (most-specific wins):
    #   1) ``camera_streams.json['twin_to_stream_url'][twin_uuid]`` — set by
    #      the CLI installer when the user mapped multiple camera twins to
    #      distinct AVFoundation cameras.
    #   2) ``CYBERWAVE_MACOS_CAMERA_STREAM_URL`` runtime env var — legacy
    #      single-camera fallback.
    _macos_camera_stream_url: Optional[str] = None
    _macos_audio_stream_url: Optional[str] = None
    if platform.system() == "Darwin":
        _per_twin = s._load_camera_stream_url_for_twin(twin_uuid)
        if _per_twin:
            _macos_camera_stream_url = _per_twin
        else:
            _raw = s.get_runtime_env_var("CYBERWAVE_MACOS_CAMERA_STREAM_URL")
            if _raw and _raw.strip():
                _macos_camera_stream_url = _raw.strip()

        _per_twin_audio = s._load_audio_stream_url_for_twin(twin_uuid)
        if _per_twin_audio:
            _macos_audio_stream_url = _per_twin_audio
        else:
            _raw_audio = s.get_runtime_env_var("CYBERWAVE_MACOS_AUDIO_STREAM_URL")
            if _raw_audio and _raw_audio.strip():
                _macos_audio_stream_url = _raw_audio.strip()

    macos_bridge_ok, macos_resolved_devices = _run_macos_device_bridge_commands(
        params=params,
        twin_uuid=twin_uuid,
        container_name=container_name,
        additional_device_mappings=macos_bridge_mappings,
        usbip_active=usbip_active,
    )
    if not macos_bridge_ok:
        driver_alert_ctx.mark_failed_and_resolve(
            f"macOS device bridge setup failed for image {image}.",
            phase="macos_bridge_failed",
        )
        return False

    if platform.system() == "Darwin" and not _macos_camera_stream_url:
        video_device_map = {
            container_device: resolved_device
            for container_device, resolved_device in macos_resolved_devices.items()
            if s._is_video_device_path(container_device)
        }
        if video_device_map:
            container_env.setdefault(
                "CYBERWAVE_EDGE_VIDEO_DEVICE_MAP",
                json.dumps(video_device_map, separators=(",", ":")),
            )
            first_resolved_video_device = next(iter(video_device_map.values()))
            if "CYBERWAVE_METADATA_VIDEO_DEVICE" not in explicit_params_env:
                container_env.setdefault(
                    "CYBERWAVE_METADATA_VIDEO_DEVICE",
                    first_resolved_video_device,
                )

            # When USB/IP handles video devices, the container will see
            # /dev/video* natively after entrypoint attachment — don't strip
            # --device params and don't rewrite paths to RTSP URLs.
            if not usbip_active:
                should_strip_video_devices = s._resolve_bool_env_var(
                    "CYBERWAVE_MACOS_STRIP_VIDEO_DEVICE_PARAMS",
                    default=True,
                )
                if should_strip_video_devices and any(
                    resolved != container_device
                    for container_device, resolved in video_device_map.items()
                ):
                    params = s._strip_video_device_mappings(params)

    base_url = s.get_runtime_env_var("CYBERWAVE_BASE_URL")
    if base_url:
        container_env["CYBERWAVE_BASE_URL"] = s._rewrite_macos_container_base_url(base_url)
    mqtt_host = s.get_runtime_env_var("CYBERWAVE_MQTT_HOST")
    if mqtt_host:
        container_env["CYBERWAVE_MQTT_HOST"] = s._rewrite_macos_container_hostname(
            mqtt_host
        )
    if runtime_environment != "production":
        container_env["CYBERWAVE_ENVIRONMENT"] = runtime_environment

    # Also forward additional CYBERWAVE_* env vars persisted by the CLI.
    for key, value in s.load_credentials_envs().items():
        if key.startswith("CYBERWAVE_"):
            if key in explicit_params_env:
                continue
            container_env.setdefault(key, value)

    # Forward CYBERWAVE_* from the edge core process environment so that
    # host-set vars (e.g. systemd Environment=, /etc/environment) reach
    # the driver container. E.g. CYBERWAVE_GO2_IP_ADDR for the Go2 driver.
    # setdefault avoids overwriting vars we or credentials already set.
    for key, value in os.environ.items():
        if key.startswith("CYBERWAVE_") and isinstance(value, str) and value.strip():
            if key in explicit_params_env:
                continue
            container_env.setdefault(key, value.strip())

    # Auto-infer MQTT TLS when port 8883 is configured but USE_TLS is absent.
    # Port 8883 is the IANA-assigned MQTT-over-TLS port; C++ drivers (unlike the
    # Python SDK) do not auto-detect this and need the explicit flag.
    if (
        "CYBERWAVE_MQTT_USE_TLS" not in container_env
        and "CYBERWAVE_MQTT_USE_TLS" not in explicit_params_env
    ):
        mqtt_port = container_env.get("CYBERWAVE_MQTT_PORT", "")
        if mqtt_port == "8883":
            container_env["CYBERWAVE_MQTT_USE_TLS"] = "true"

    container_env["CYBERWAVE_EDGE_CONFIG_DIR"] = "/app/.cyberwave"

    # Inject Zenoh transport configuration so drivers that use cw.data.publish()
    # automatically pick up the correct backend and router settings.  Drivers
    # that do not use the SDK data layer simply ignore these variables.
    # Use setdefault so that any per-driver override in explicit_params_env takes
    # precedence (driver metadata can always override with -e KEY=val in params).
    zenoh_env = s.build_zenoh_env_vars(s._get_zenoh_config())
    for key, value in zenoh_env.items():
        if key not in explicit_params_env:
            container_env.setdefault(key, value)

    # On macOS, enable USB/IP passthrough when the host server is running.
    # --pid=host lets the container use nsenter to access Docker Desktop's
    # pre-installed usbip tools; CYBERWAVE_USBIP_ENABLED tells the entrypoint
    # to auto-attach devices (serial + video).
    pid_args: list[str] = []
    if usbip_active:
        pid_args = ["--pid=host"]
        container_env.setdefault("CYBERWAVE_USBIP_ENABLED", "1")
        has_video_devices = any(
            s._is_video_device_path(d) for d in macos_resolved_devices
        )
        if has_video_devices and not _macos_camera_stream_url:
            container_env.setdefault("CYBERWAVE_USBIP_VIDEO_TIMEOUT_SECS", "8")

    # When the user has configured a macOS MJPEG camera stream URL, force
    # it as the video device.  This takes priority over bridge-resolved
    # /dev/video* paths and USB/IP video passthrough (which is often
    # unreliable for high-bandwidth video).
    if _macos_camera_stream_url:
        container_env["CYBERWAVE_METADATA_VIDEO_DEVICE"] = _macos_camera_stream_url
        logger.info(
            "macOS camera stream URL override: %s (usbip_active=%s)",
            _macos_camera_stream_url,
            usbip_active,
        )
    elif platform.system() == "Darwin" and macos_bridge_mappings and not usbip_active:
        logger.warning(
            "macOS camera twin %s has no MJPEG stream URL configured. "
            "The driver container will likely fail to open /dev/video* "
            "because Docker Desktop does not expose host cameras. "
            "Run: cyberwave edge install --reconfigure-camera",
            twin_uuid[:8],
        )
        try:
            s._send_alert_for_twin(
                twin_uuid,
                "Camera not configured for macOS",
                "This camera twin has no MJPEG stream URL configured. Docker "
                "Desktop on macOS cannot pass /dev/video* devices to containers. "
                "Run 'cyberwave edge install --reconfigure-camera' to set up "
                "camera streaming.",
                "macos_camera_not_configured",
                severity="warning",
            )
        except Exception as exc:
            logger.debug(
                "Could not send macos_camera_not_configured alert: %s", exc
            )

    if _macos_audio_stream_url:
        container_env["CYBERWAVE_METADATA_AUDIO_DEVICE"] = _macos_audio_stream_url
        logger.info(
            "macOS audio stream URL override: %s",
            _macos_audio_stream_url,
        )
        bridge_rate, bridge_channels = s._load_audio_stream_capture_settings()
        if (
            bridge_rate is not None
            and "CYBERWAVE_METADATA_AUDIO_SAMPLE_RATE" not in explicit_params_env
        ):
            container_env.setdefault(
                "CYBERWAVE_METADATA_AUDIO_SAMPLE_RATE", str(bridge_rate)
            )
        if (
            bridge_channels is not None
            and "CYBERWAVE_METADATA_AUDIO_CHANNELS" not in explicit_params_env
        ):
            container_env.setdefault(
                "CYBERWAVE_METADATA_AUDIO_CHANNELS", str(bridge_channels)
            )
    elif (
        platform.system() == "Darwin"
        and s._is_generic_microphone_driver_image(image)
        and "CYBERWAVE_METADATA_AUDIO_DEVICE" not in explicit_params_env
    ):
        logger.warning(
            "macOS microphone twin %s has no host audio bridge URL configured. "
            "Docker Desktop cannot pass CoreAudio into Linux containers. "
            "Run: cyberwave edge install --reconfigure-microphone",
            twin_uuid[:8],
        )
        try:
            s._send_alert_for_twin(
                twin_uuid,
                "Microphone not configured for macOS",
                "This microphone twin has no host PCM stream URL. On macOS, "
                "run 'cyberwave edge install --reconfigure-microphone' to start "
                "the ffmpeg audio bridge, then restart edge-core.",
                "macos_microphone_not_configured",
                severity="warning",
            )
        except Exception as exc:
            logger.debug(
                "Could not send macos_microphone_not_configured alert: %s", exc
            )

    if service_env:
        container_env.update(service_env)

    env_vars: List[str] = []
    for key, value in container_env.items():
        env_vars += ["-e", f"{key}={value}"]

    twin_json_file = s.CONFIG_DIR / f"{twin_uuid}.json"
    if twin_json_file.is_file():
        env_vars += ["-v", f"{twin_json_file}:/app/{twin_uuid}.json"]
        env_vars += ["-e", f"CYBERWAVE_TWIN_JSON_FILE=/app/{twin_uuid}.json"]
    # Mount the edge config directory read-only so driver containers cannot
    # tamper with credentials.json or other sensitive config files.
    env_vars += ["-v", f"{s.CONFIG_DIR}:/app/.cyberwave:ro"]
    # SO101 drivers need write access to so101_lib/ for calibrations and URDF
    # downloads.  Mount that subdirectory read-write as an overlay.
    so101_lib_dir = s.CONFIG_DIR / "so101_lib"
    if so101_lib_dir.is_dir() or "so101" in image.lower():
        so101_lib_dir.mkdir(parents=True, exist_ok=True)
        env_vars += ["-v", f"{so101_lib_dir}:/app/.cyberwave/so101_lib"]

    network_args = s._build_driver_network_args(params)

    gpu_args: list[str] = []
    if prefer_gpu and platform.system() == "Linux":
        from .docker_helpers import docker_has_nvidia_default_runtime, docker_has_nvidia_runtime

        if docker_has_nvidia_runtime() and docker_has_nvidia_default_runtime():
            gpu_value = gpu_spec or "all"
            gpu_args = ["--gpus", gpu_value]
            logger.info(
                "NVIDIA runtime detected with default daemon config — "
                "enabling GPU passthrough (--gpus %s) for %s",
                gpu_value,
                container_name,
            )
        elif docker_has_nvidia_runtime():
            logger.info(
                "NVIDIA runtime is available but not the default in "
                "/etc/docker/daemon.json — skipping --gpus for %s. "
                "Set \"default-runtime\": \"nvidia\" in "
                "/etc/docker/daemon.json to enable GPU passthrough.",
                container_name,
            )
        else:
            logger.debug(
                "prefer_gpu is set for %s but no NVIDIA runtime found",
                container_name,
            )

    cmd = [
        "docker",
        "run",
        "--detach",
        "--init",
        "--stop-timeout",
        "5",
        "--restart",
        "unless-stopped",
        "--privileged",
        *gpu_args,
        *pid_args,
        *network_args,
        "--name",
        container_name,
        *params,
        *env_vars,
        image,
        *(command or []),
    ]
    if logger.isEnabledFor(logging.DEBUG):
        debug_env_vars: list[str] = []
        for index, item in enumerate(env_vars):
            if item != "-e" or index + 1 >= len(env_vars):
                continue
            key, sep, value = env_vars[index + 1].partition("=")
            if sep and key == "CYBERWAVE_API_KEY":
                value = f"{value[:6]}…{value[-4:]}" if len(value) > 12 else "***"
            debug_env_vars.append(f"{key}{sep}{value}" if sep else env_vars[index + 1])

        debug_cmd = [
            (
                f"CYBERWAVE_API_KEY={arg.split('=', 1)[1][:6]}…{arg.split('=', 1)[1][-4:]}"
                if arg.startswith("CYBERWAVE_API_KEY=") and len(arg.split("=", 1)[1]) > 12
                else "CYBERWAVE_API_KEY=***"
                if arg.startswith("CYBERWAVE_API_KEY=")
                else arg
            )
            for arg in cmd
        ]
        logger.debug(
            "Docker run debug inputs for %s: image=%s params=%s env_vars=%s",
            container_name,
            image,
            params,
            debug_env_vars,
        )
        logger.debug("Docker run command args for %s: %s", container_name, debug_cmd)
    logger.info("Starting docker container %s from image %s", container_name, image)
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        s._CONTAINER_TWIN_MAP[container_name] = twin_uuid
        s._stream_container_logs(container_name, twin_uuid=twin_uuid, token=token)

        # A detached `docker run` can still fail immediately (e.g. missing USB
        # hardware causes rapid crashes). Verify that the container reaches and
        # stays in a running state for a brief window.
        for _ in range(5):
            inspect_data = s._inspect_driver_container(container_name)
            if not inspect_data:
                time.sleep(1.0)
                continue
            state = inspect_data.get("State") if isinstance(inspect_data.get("State"), dict) else {}
            status = str(state.get("Status", "")).lower()
            if status == "running":
                driver_alert_ctx.update_metadata(
                    {"phase": "container_running"}, force=True
                )
                driver_alert_ctx.resolve()
                return True
            if status in {"restarting", "exited", "dead"}:
                logger.error(
                    "Driver container %s failed to start cleanly (status=%s error=%s)",
                    container_name,
                    status,
                    str(state.get("Error", "")).strip() or "none",
                )
                driver_alert_ctx.mark_failed_and_resolve(
                    (
                        f"Driver container {container_name} failed to start cleanly "
                        f"(status={status})."
                    ),
                    phase="container_unhealthy",
                )
                return False
            time.sleep(1.0)

        logger.warning(
            "Driver container %s did not reach a stable running state within startup probe window",
            container_name,
        )
        # Probe window elapsed without confirmation; the container may still
        # come up successfully, so close the alert as resolved (the caller
        # surfaces a separate ``driver_start_failure`` alert if needed).
        driver_alert_ctx.update_metadata(
            {"phase": "container_probe_unconfirmed"}, force=True
        )
        driver_alert_ctx.resolve()
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to start container %s: %s", container_name, exc.stderr)
        driver_alert_ctx.mark_failed_and_resolve(
            f"Failed to start container {container_name}.",
            phase="docker_run_failed",
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("Docker run timed out for image: %s", image)
        driver_alert_ctx.mark_failed_and_resolve(
            f"Docker run timed out for image {image}.",
            phase="docker_run_timeout",
        )
        return False
