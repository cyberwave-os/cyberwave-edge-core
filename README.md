# Cyberwave Edge-Core

The edge core component acts as an orchestrator of the Cyberwave edge drivers.

## Quickstart (Linux machines)

SSH into the device where you want to install the edge-core.

```bash
# Install the CLI (one time setup)
curl -fsSL https://cyberwave.com/install.sh | bash

# Use the CLI to complete the edge setup
sudo cyberwave edge install
```

The cyberwave-cli will ask you to log in with your Cyberwave credentials and will then proceed with installing `cyberwave-edge-core` on your edge device.

> Don't have a Cyberwave account? Get one at [cyberwave.com](https://cyberwave.com)

### Configuration

`cyberwave-edge-core` reads the edge configuration from `/etc/cyberwave/` on Linux and `~/.cyberwave/` on macOS (overridable via the `CYBERWAVE_EDGE_CONFIG_DIR` environment variable, which is set in the systemd unit):

| File               | Description               |
| ------------------ | ------------------------- |
| `credentials.json` | API token, workspace info |
| `fingerprint.json` | Device fingerprint        |
| `environment.json` | Linked environment UUID   |

Of the files above, `cyberwave-edge-core` needs only `credentials.json`. You can easily populate it with the `cyberwave` command line tool (CLI) as described in the quickstart.

The file `fingerprint.json` is populated by `cyberwave-edge-core` itself.

## How the edge-core works

Once its started (either via `cyberwave-edge-core` or via service) the edge-core does the following:

1. It checks if the credentials stored in `credentials.json` are valid
2. It connects to the backend MQTT and checks if the connection is up and running
3. It registers the `edge` device it's running on, and updates its registration record (each `edge` device being identified by a unique edge fingerprint)
4. It downloads the latest environment from the backend and resolves the twins linked to the edge fingerprint
5. It starts drivers for linked twins, with one special case for attached camera twins:
   - if a linked twin is a camera child twin (`attach_to_twin_uuid`) of another linked twin, edge-core does **not** start a dedicated driver for that camera child
   - edge-core passes camera child UUIDs to the parent driver via `CYBERWAVE_CHILD_TWIN_UUIDS`

### Remote restart (via Edge REST API)

`POST /api/v1/edges/{uuid}/restart-core` publishes an MQTT command to:

`edges/{edge_uuid}/command`

with `{"command":"restart_edge_core", ...}` payload.

The running edge-core subscribes to that topic using the Cyberwave SDK and,
when it receives the command, it:

1. Deletes cached twin JSON objects under the edge config directory
2. Stops/removes edge-managed driver containers and prunes stopped containers
3. Re-downloads the linked environment twins and starts drivers again

## Writing compatible drivers

A Cyberwave `driver` is a Docker image that is capable of interacting with the device's hardware, sending and receiving data from the Cyberwave backend. Every time the edge-core starts a driver Docker image, it does so by defining the following environment variables:

- `CYBERWAVE_TWIN_UUID`
- `CYBERWAVE_API_KEY`
- `CYBERWAVE_TWIN_JSON_FILE`
- `CYBERWAVE_CHILD_TWIN_UUIDS` (optional, comma-separated)

`CYBERWAVE_CHILD_TWIN_UUIDS` is present when the driver twin has attached camera child twins in the same linked set. Drivers can use this to discover and coordinate child twins without relying on extra camera selection prompts.

### Driver failure handling contract

Drivers must exit with a **non-zero** code when they cannot access the required hardware (e.g. due to a missing `/dev/video*` device or because of a disconnected USB peripheral). This lets the edge-core reliably detect startup failures and restart loops.

Runtime behavior of the edge-core:

- If a driver container fails to enter a stable running state, the edge-core raises a `driver_start_failure` alert.
- If a driver restarts more than `4` times in `60` seconds, the edge-core marks it as flapping, stops the container, and raises a `driver_restart_loop` alert with a troubleshooting link.

Optional edge-core env vars:

- `CYBERWAVE_DRIVER_RESTART_LOOP_THRESHOLD` (default: `4`)
- `CYBERWAVE_DRIVER_RESTART_LOOP_WINDOW_SECONDS` (default: `60`)
- `CYBERWAVE_DRIVER_TROUBLESHOOTING_URL` (default: `https://docs.cyberwave.com`)

### Twin JSON file

`CYBERWAVE_TWIN_JSON_FILE` is an absolute path to a JSON file, which is writable by the driver. The JSON represents a complete digital twin object (instance) as well as its complete catalog twin object (asset). It is represented in the same way as in the API, including the whole metadata field, schema and abilities. [Twin reference here](https://docs.cyberwave.com/api-reference/rest/TwinSchema), [Asset reference here](https://docs.cyberwave.com/api-reference/rest/AssetSchema).

A driver can change the twin JSON file. The edge-core will, when connectivity is present, sync it with the one in the backend.

### Twin metadata

When writing drivers, use the official Cyberwave SDK to communicate with the backend, as it will abstract a bunch of complexity in the MQTT handshake, REST API authentication, and more.

Once you've written a driver, you can register it by adding its details to the twin's metadata, or to the asset's metadata if you own it. Metadata editing is currently manual — switch to **Advanced editing** in the environment view or the asset editor.

> Note: If you change the metadata on the asset, every twin created out of that asset from that moment on will have the same metadata as the asset, as the starting point

The driver object in the metadata looks like this:

```json
"drivers": {
    "default": {
        "docker_image":"cyberwaveos/so101-driver", // this is either a public image on the Docker hub or on your own registry
        "version": "0.0.1", // this field is optional
        "params" : ["--network local", "--add-host host.docker.internal:host-gateway"] // this is also optional
    }
}
```

For an example, check how the camera driver handles the TWIN JSON file.

### Passing environment variables to drivers (`metadata["drivers"]`)

To inject environment variables into a driver container, use the `params` field with Docker's `-e` flag:

```json
"drivers": {
  "default": {
    "docker_image": "cyberwaveos/go2-native-driver",
    "params": ["-e", "MY_VAR=value", "-e", "ANOTHER_VAR=value2"]
  }
}
```

Each `-e` must be its own element in the array, followed by the `KEY=value` string as the next element. This is equivalent to passing `-e MY_VAR=value` on the `docker run` command line.

This is useful for driver-specific configuration that varies per device, such as IP addresses, credentials, or feature flags that cannot be stored in the twin's `edge_configs` metadata.

### Runtime configuration for drivers (`metadata["edge_configs"]`)

Drivers and edge services should treat `metadata["edge_configs"]` as the source of truth for per-device runtime configuration.
Edge identity should be stored at `metadata["edge_fingerprint"]` (not duplicated inside `edge_configs`).

> **Runtime access**: The core passes the full twin JSON (including `metadata`) to every driver via the `CYBERWAVE_TWIN_JSON_FILE` environment variable. Drivers can read `edge_configs` from that file at startup to obtain per-device settings — for example, selecting the right camera source or IP address for the current machine. This is the recommended way to pass device-specific configuration to a driver without hardcoding values in the image.

- Type: object/dictionary
- Value: binding object (`object`)

Canonical shape:

```json
"edge_fingerprint": "macbook-pro-a1b2c3d4e5f6",
"edge_configs": {
  "camera_config": {
    "camera_id": "front",
    "source": "rtsp://user:pass@192.168.1.20/stream",
    "fps": 10,
    "resolution": "VGA",
    "camera_type": "cv2"
  }
}
```

Field notes:

- `edge_fingerprint` (recommended): fingerprint of the edge currently serving this twin (top-level metadata field).
- `camera_config` (recommended): per-device camera/runtime config consumed by edge drivers.
- Do not store `edge_uuid`, `registered_at`, `last_sync`, `last_ip_address`, or `status_data` inside `edge_configs`.

Backward compatibility:

- Older records may still use the legacy map shape (`edge_configs[fingerprint] = {...}`).
- Older records may store camera settings in `cameras[0]` or as top-level fields.
- New writers should always write to `camera_config`.
- Do not rely on `PUT /api/v1/edges/{uuid}/twins/{twin_uuid}/camera-config`; it is deprecated. Update twin metadata instead.

## Advanced usage

### Manual install and usage

```bash
# Install the registry signing key:

curl -fsSL "https://packages.buildkite.com/cyberwave/cyberwave-edge-core/gpgkey" | gpg --dearmor -o /etc/apt/keyrings/cyberwave_cyberwave-edge-core-archive-keyring.gpg

# Configure the source:

echo -e "deb [signed-by=/etc/apt/keyrings/cyberwave_cyberwave-edge-core-archive-keyring.gpg] https://packages.buildkite.com/cyberwave/cyberwave-edge-core/any/ any main\ndeb-src [signed-by=/etc/apt/keyrings/cyberwave_cyberwave-edge-core-archive-keyring.gpg] https://packages.buildkite.com/cyberwave/cyberwave-edge-core/any/ any main" > /etc/apt/sources.list.d/buildkite-cyberwave-cyberwave-edge-core.list

# Run the edge-core (startup checks + full runtime loop: starts drivers, etc.)
cyberwave-edge-core

# Show current credential, token, and MQTT status (read-only)
cyberwave-edge-core status

# Show version
cyberwave-edge-core --version
```

### Other env vars

To run against another env:

```bash
export CYBERWAVE_ENVIRONMENT="yourenv"
export CYBERWAVE_BASE_URL="https://yourbaseurl"
cyberwave-edge-core
```

To control log verbosity (default: `INFO`):

```bash
export CYBERWAVE_EDGE_LOG_LEVEL="DEBUG"
cyberwave-edge-core
```

Or from the CLI

```bash
sudo CYBERWAVE_ENVIRONMENT="yourenv" CYBERWAVE_BASE_URL="https://yourbaseurl" cyberwave edge install
```
