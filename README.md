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
- `CYBERWAVE_TOKEN`
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

To run against dev:

```bash
export CYBERWAVE_ENVIRONMENT="dev"
export CYBERWAVE_API_URL="https://api-dev.cyberwave.com"
# Optional SDK alias (kept for compatibility):
export CYBERWAVE_BASE_URL="https://api-dev.cyberwave.com"
cyberwave-edge-core
```
