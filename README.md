# Cyberwave Edge Core

This Edge component acts as an orchestrator of your Cyberwave edge components.

## Quickstart (Linux machines)

```bash
# ssh into your edge device then
sudo apt-get install cyberwave-cli

sudo cyberwave edge
```

The cyberwave-cli will ask you to log in with your Cyberwave credentials and then will proceed installing the `cyberwave-edge-core` on your edge device.

> Don't have a Cyberwave account? Get one at [https://cyberwave.com](cyberwave.com)

## Manual usage

```bash
sudo apt-get install cyberwave-edge-core

# Run all startup checks (validate token, MQTT, devices, environment)
cyberwave-edge-core

# Show current credential, token, MQTT, and device status
cyberwave-edge-core status

# Show version
cyberwave-edge-core --version
```

To run against dev:

```bash
export CYBERWAVE_ENVIRONMENT="dev"
export CYBERWAVE_BASE_URL="https://api-dev.cyberwave.com"
cyberwave-edge-core
```

### Configuration

The edge core reads configuration from `~/.cyberwave/`:

| File               | Description               |
| ------------------ | ------------------------- |
| `credentials.json` | API token, environment ID |
| `fingerprint.json` | Device fingerprint        |
| `environment.json` | Linked environment UUID   |

Of the files above, the `core` needs only the `credentials.json` file. You can easily populate it with the `cyberwave-cli` as described in the quickstart. The

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

The Cyberwave twin JSON file is an absolute path to a JSON file. The JSON file is writable by the driver. It represents a complete twin object as well as its complete asset object. It represented in the same way that is it in the API, including the whole metadata field, schema and abilities. [https://docs.cyberwave.com/api-reference/rest/TwinSchema](Twin reference here), [https://docs.cyberwave.com/api-reference/rest/AssetSchema](Asset reference here).

As a driver, you can change the JSON file. The core will, when connectivity is present, sync it with the one in the backend.
