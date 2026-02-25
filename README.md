# Cyberwave Edge Core

This Edge component acts as an orchestrator of your Cyberwave edge components.

## Quickstart (Linux machines)

```bash
# Install the CLI (one time setup)
curl -fsSL "https://packages.buildkite.com/cyberwave/cyberwave-cli/gpgkey" | sudo gpg --dearmor -o /etc/apt/keyrings/cyberwave_cyberwave-cli-archive-keyring.gpg

# Configure the source
echo -e "deb [signed-by=/etc/apt/keyrings/cyberwave_cyberwave-cli-archive-keyring.gpg] https://packages.buildkite.com/cyberwave/cyberwave-cli/any/ any main\ndeb-src [signed-by=/etc/apt/keyrings/cyberwave_cyberwave-cli-archive-keyring.gpg] https://packages.buildkite.com/cyberwave/cyberwave-cli/any/ any main" | sudo tee /etc/apt/sources.list.d/buildkite-cyberwave-cyberwave-cli.list > /dev/null

# Install the CLI
sudo apt update && sudo apt install cyberwave-cli

# Use the CLI to complete the edge setup
sudo cyberwave edge install
```

The cyberwave-cli will ask you to log in with your Cyberwave credentials and then will proceed installing the `cyberwave-edge-core` on your edge device.

> Don't have a Cyberwave account? Get one at [cyberwave.com](https://cyberwave.com)

### Configuration

The edge core reads configuration from `/etc/cyberwave/` (overridable via the `CYBERWAVE_EDGE_CONFIG_DIR` environment variable, which is set in the systemd unit):

| File               | Description               |
| ------------------ | ------------------------- |
| `credentials.json` | API token, workspace info |
| `fingerprint.json` | Device fingerprint        |
| `environment.json` | Linked environment UUID   |

Of the files above, the `core` needs only the `credentials.json` file. You can easily populate it with the `cyberwave-cli` as described in the quickstart.

The `fingerprint.json` is populated by the core itself.

## How the edge works

Once it's started (either via CLI or via service) the core does the following:

1. Checks if the credentials stored in `credentials.json` are valid
2. Connects to the backend MQTT and checks if the connection is up and running
3. Registers the `edge` device is running on, or updates its registration record. Each `edge` device is defined by a unique hardware fingerprint
4. Downloads the latest environment from the backend and downloads the list of devices connected to the edge
5. For each `twin`, present in the environment, and connected to the edge: It starts the twin's docker driver image

## Writing compatible drivers

A Cyberwave driver is a Docker image that is capable of interacting with the device's hardware, sending and getting data from the Cyberwave backend. Every time the core starts a driver Docker image, the `core` does so by defining the following environment variables:

- `CYBERWAVE_TWIN_UUID`
- `CYBERWAVE_API_KEY`
- `CYBERWAVE_TWIN_JSON_FILE`

The Cyberwave twin JSON file is an absolute path to a JSON file. The JSON file is writable by the driver. It represents a complete twin object as well as its complete asset object. It represented in the same way that is it in the API, including the whole metadata field, schema and abilities. [Twin reference here](https://docs.cyberwave.com/api-reference/rest/TwinSchema), [Asset reference here](https://docs.cyberwave.com/api-reference/rest/AssetSchema).

As a driver, you can change the JSON file. The core will, when connectivity is present, sync it with the one in the backend.

When writing drivers, use the official Cyberwave SDK to communicate with the backend, as it will abstract a bunch of complexity in the MQTT handshake, REST API authentication, and more.

Once you wrote a driver, you can add its details in the twin's metadata (or the asset's metadata if you own it). Right now the edit is manual and directly in the metadata. To edit the metadata, you can switch to Advanced editing in the environment or in the asset editing.

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

### `metadata["edge_configs"]` format (for edge <-> twin binding)

Drivers and edge services should treat `metadata["edge_configs"]` as the source of truth for per-device runtime configuration.

- Type: object/dictionary
- Value: binding object (`object`)

Canonical shape:

```json
"edge_configs": {
  "edge_fingerprint": "macbook-pro-a1b2c3d4e5f6",
  "camera_config": {
    "camera_id": "front",
    "source": "rtsp://user:pass@192.168.1.20/stream",
    "fps": 10,
    "resolution": "VGA",
    "camera_type": "cv2"
  },
  "device_info": {
    "hostname": "edge-macbook",
    "platform": "Darwin-arm64"
  },
  "registered_at": "2026-02-24T10:11:12.000000+00:00",
  "last_sync": "2026-02-24T10:15:00.000000+00:00",
  "edge_uuid": "8c1f72a0-5cb5-4f85-9d57-c170b50d4dbe",
  "last_ip_address": "192.168.1.42",
  "status_data": {
    "uptime_seconds": 1234,
    "streams": {
      "front": {
        "fps": 9.8,
        "frames_sent": 10000
      }
    }
  }
}
```

Field notes:

- `edge_fingerprint` (recommended): fingerprint of the edge currently serving this twin.
- `camera_config` (recommended): per-device camera/runtime config consumed by edge drivers.
- `device_info` (optional): descriptive hardware info (`hostname`, `platform`, etc.).
- `registered_at`, `last_sync` (optional): ISO-8601 timestamps.
- `edge_uuid` (optional): UUID of the Edge record associated with this fingerprint.
- `last_ip_address`, `status_data` (optional): runtime heartbeat/status details.

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

# Run all startup checks (validate token, MQTT, devices, environment)
cyberwave-edge-core

# Show current credential, token, MQTT, and device status
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

Or from the CLI

```bash
sudo CYBERWAVE_ENVIRONMENT="yourenv" CYBERWAVE_BASE_URL="https://yourbaseurl" cyberwave edge install
```
