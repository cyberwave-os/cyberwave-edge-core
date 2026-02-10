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
4. Check if it has the correct driver for that device (we'll do this later)
5. Get or create the edge with the Cyberwave Backend (device fingerprint)
6. Start listening to MQTT commands for the device
