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
```

## TODO

The edge core should:

1. Assume it is starting on boot
2. ~~Check if it can connect to MQTT and if the token saved in ~/.cyberwave/credentials.json is valid~~
3. ~~Check an internal file (~/.cyberwave/devices.json) to see if any device is set up~~
4. ~~Get or create the edge with the Cyberwave Backend (device fingerprint, persisted in ~/.cyberwave/fingerprint.json)~~
5. Check if the MQTT connection works DONE
6. ~~Check if there is an environment connected to the edge (in ~/.cyberwave/environment.json, format: {"uuid":"unique-uuid-of-the-environment"})~~
7. ~~If so, call the API and download the twins for that particular environment. Check if any digital twin of the environments have the fingerprint of the edge. If so, for every twin: call the assets API with the UUID of the asset of that twin; in the asset's metadata, look for the driver_docker_image field; then run the docker image~~
