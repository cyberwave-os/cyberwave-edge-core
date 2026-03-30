# CYB-1546: Edge Core — Worker Container, Model Manager, Zenoh Infrastructure

## Implementation Plan

**Issue:** [CYB-1546](https://linear.app/cyberwave-spa/issue/CYB-1546/edge-core-worker-container-model-manager-zenoh-infrastructure)
**Parent Epic:** CYB-1498 (Edge ML Models)
**Coordinates with:** CYB-1544 (SDK Data Layer), CYB-1545 (SDK Worker API), CYB-1547 (Driver Integration)

---

## Overview

Edge Core currently manages **driver containers** (one per twin) via the Docker CLI. This issue extends Edge Core with three new capabilities:

1. **Worker Manager** — manage a dedicated worker container per edge device
2. **Model Manager** — download, cache, and mount ML model weights
3. **Zenoh Infrastructure Wiring** — inject transport env vars and configure mounts so workers and drivers share the Zenoh data bus

The design follows the target architecture in `README.md` and `cyberwave-edge-runtime/README.md`, keeping Edge Core as an orchestrator that manages lifecycle, not a hot-path relay.

---

## Current State

### What exists (`cyberwave_edge_core/startup.py`)

| Capability | Status |
|---|---|
| Docker container lifecycle (create/start/stop/restart) via `docker` CLI subprocess | Done |
| Driver containers: `cyberwave-driver-{uuid_short}`, one per twin | Done |
| Environment variable injection (`CYBERWAVE_*` from credentials, host env, metadata params) | Done |
| Config dir bind-mount (`CONFIG_DIR → /app/.cyberwave`) | Done |
| Twin JSON file mount + deep-merge + API sync | Done |
| Container log streaming to logger + MQTT | Done |
| Restart/flap detection (`reconcile_driver_restart_failures`) | Done |
| Remote restart via MQTT (`restart_edge_core`) | Done |
| Docker image pull with progress + local fallback | Done |

### What does NOT exist

- No worker container concept (only driver containers)
- No model download/cache/mount logic
- No Zenoh references anywhere in edge-core
- No shared-memory or Zenoh environment wiring
- No file-watching or reload signal for worker changes

---

## Architecture Decisions

### AD-1: One worker container per edge device (not per twin)

Driver containers are 1:1 with twins because each twin has dedicated hardware. Workers consume data from **all** twins in the environment, so one shared container is simpler and avoids model duplication.

- Container name: `cyberwave-worker-{env_uuid_short}`
- Mounts: config dir + `~/.cyberwave/workers/` + `~/.cyberwave/models/`
- Env: `CYBERWAVE_DATA_BACKEND=zenoh`, `ZENOH_CONNECT`, twin UUIDs, API key

### AD-2: Model cache at `~/.cyberwave/models/`

Models are shared across workers. A single host-level cache directory avoids re-downloading the same weights when multiple workers reference the same model.

- Linux: `/etc/cyberwave/models/` (follows existing config dir convention)
- macOS: `~/.cyberwave/models/`
- Container mount: `{models_dir} → /app/models/` (read-only)

### AD-3: Zenoh peer-to-peer by default, optional router

The Zenoh data bus runs in peer-to-peer mode (all containers on the same Docker network discover each other). An optional Zenoh router container is only needed for:
- MQTT bridge (forwarding events to cloud)
- Multi-hop topologies (cross-device inference)

Edge Core sets `ZENOH_CONNECT` only when a router is configured.

### AD-4: Worker restart on file change, not hot-reload

When workers are added/removed/changed, Edge Core **restarts the worker container** rather than sending an in-process reload signal. This keeps the worker runtime simple and avoids state corruption from partially-loaded models.

### AD-5: Model Manager is Edge Core-internal, exposed via local API

The SDK calls `cw.models.load("yolov8n")` which resolves to a local file path. Edge Core pre-downloads models before starting the worker container. No runtime HTTP API is needed initially — the model cache is a bind-mounted directory.

---

## User-Facing CLI Surface

The primary user interface for worker management is a `worker` subcommand group on the `cyberwave-edge-core` CLI:

```
cyberwave-edge-core worker start      # Start the worker container
cyberwave-edge-core worker stop       # Stop the worker container
cyberwave-edge-core worker restart    # Restart the worker container
cyberwave-edge-core worker status     # Show worker container state, loaded workers, models
cyberwave-edge-core worker logs       # Stream worker container logs
```

These are the only edge-core-level worker commands. Worker **file** management (`add`, `remove`, `list`) belongs in the main `cyberwave` CLI (CYB-1548 scope):

```
cyberwave worker add detect.py        # Copy worker file to ~/.cyberwave/workers/
cyberwave worker remove detect        # Remove worker file
cyberwave worker list                 # List installed workers
```

---

## Deliverables and Implementation Steps

### Step 1: Internal Refactor — Extract Docker Helpers

**File:** `cyberwave_edge_core/docker_helpers.py` (new, internal only)

Extract reusable Docker CLI wrappers from `startup.py` into a shared internal module. These are **not user-facing** — they're implementation plumbing so driver and worker container management don't duplicate Docker subprocess logic.

`startup.py` calls these helpers instead of inlining `subprocess.run(["docker", ...])` calls. No new CLI commands, no behavior change.

**Acceptance:** Existing driver container lifecycle works identically after refactor.

---

### Step 2: Worker Manager — Container Lifecycle + CLI Commands

**Files:**
- `cyberwave_edge_core/worker_manager.py` (new) — internal lifecycle logic
- `cyberwave_edge_core/main.py` (modified) — `worker` CLI subcommand group

**CLI commands added to `main.py`:**

```python
@cli.group()
def worker():
    """Manage the edge worker container."""

@worker.command()
def start():
    """Start the worker container (downloads models, wires Zenoh)."""

@worker.command()
def stop():
    """Stop the worker container."""

@worker.command()
def restart():
    """Restart the worker container (re-scans workers, re-ensures models)."""

@worker.command()
def status():
    """Show worker container state, loaded workers, cached models."""

@worker.command()
def logs():
    """Stream worker container logs."""
```

**Internal `WorkerManager` class** (in `worker_manager.py`) backs the CLI commands:

```
class WorkerManager:
    def start(self) → bool
    def stop(self) → bool
    def restart(self) → bool
    def status(self) → WorkerStatus
    def logs(self, *, follow: bool = True) → None
```

**Container specification:**

| Aspect | Value |
|---|---|
| Name | `cyberwave-worker-{env_uuid[:8]}` |
| Image | Configurable; default `cyberwaveos/ml-worker:latest` |
| Restart policy | `unless-stopped` |
| Network | Same Docker network as driver containers |
| GPU | `--gpus all` if NVIDIA runtime detected (`docker info --format '{{.Runtimes}}'` contains `nvidia`) |
| Privileged | No (workers don't need hardware access) |

**Environment variables injected:**

| Variable | Source |
|---|---|
| `CYBERWAVE_API_KEY` | credentials.json |
| `CYBERWAVE_TWIN_UUIDS` | Comma-separated list of environment twin UUIDs |
| `CYBERWAVE_ENVIRONMENT_UUID` | environment.json |
| `CYBERWAVE_DATA_BACKEND` | `zenoh` (hardcoded) |
| `ZENOH_CONNECT` | Only if router configured; otherwise peer-to-peer |
| `CYBERWAVE_EDGE_CONFIG_DIR` | `/app/.cyberwave` |
| `CYBERWAVE_BASE_URL` | Runtime env |
| `CYBERWAVE_MQTT_HOST` | Runtime env |
| `CYBERWAVE_ENVIRONMENT` | Runtime env (non-production) |
| `CYBERWAVE_*` | Forward from credentials.json envs + host env (same pattern as drivers) |

**Volumes:**

| Host path | Container path | Mode |
|---|---|---|
| `{CONFIG_DIR}` | `/app/.cyberwave` | rw |
| `{CONFIG_DIR}/workers/` | `/app/workers/` | ro |
| `{CONFIG_DIR}/models/` | `/app/models/` | ro |

**Integration with automatic boot path (`startup.py`):**

- After `fetch_and_run_twin_drivers()`, call `worker_manager.start()` which:
  1. Checks if workers directory has any `.py` files
  2. If yes, ensures models are cached, then starts the worker container
  3. If no, skips (no worker container needed)
- `run_runtime_loop()` adds a reconciliation step for the worker container (log following, restart detection)
- Remote restart (`restart_edge_core` MQTT handler) also calls `worker_manager.stop()` and then `worker_manager.start()`

**Acceptance:** `cyberwave-edge-core worker start` starts the container. Worker container also starts automatically alongside drivers on boot when workers exist.

---

### Step 3: Model Manager — Download and Cache

**File:** `cyberwave_edge_core/model_manager.py` (new)

Manages model weight downloads and local cache.

```
class ModelManager:
    def __init__(self, cache_dir: Path, api_token: str, base_url: str):
        ...

    def ensure_model(self, model_id: str) -> Path:
        """Download model if not cached, return local path."""

    def ensure_models(self, model_ids: list[str]) -> dict[str, Path]:
        """Batch ensure, returns {model_id: path}."""

    def list_cached_models(self) -> list[CachedModel]:
        """List all models in local cache."""

    def evict_model(self, model_id: str) -> bool:
        """Remove a model from cache."""

    def cache_size_bytes(self) -> int:
        """Total cache size."""
```

**Cache layout:**

```
~/.cyberwave/models/
├── manifest.json              # {model_id: {path, size, downloaded_at, source_url, checksum}}
├── yolov8n/
│   ├── yolov8n.pt             # Weight file
│   └── metadata.json          # {model_id, runtime, download_url, checksum, size}
├── background-subtraction/
│   └── ...
└── custom/
    └── {uuid}/                # User-uploaded models
        └── ...
```

**Download flow:**

1. Check `manifest.json` for existing entry + file existence + checksum match
2. If cache hit → return path immediately
3. If cache miss → call backend API: `GET /api/v1/ml-models/{model_id}` to get download URL from `MLModel.metadata`
4. Download weights to a temp file, verify checksum, move to cache
5. Update `manifest.json`
6. Return path

**Worker model requirements discovery:**

- Scan `*.py` files in `~/.cyberwave/workers/` for `cw.models.load("...")` calls using a simple regex
- Also check `cyberwave.yml` if present (the `models:` field lists pre-download requirements)
- `ensure_models()` is called before starting/restarting the worker container

**Error handling:**

- Network failure during download → log warning, skip model, let worker handle gracefully at runtime
- Corrupt cache file → re-download on next ensure
- Disk full → log error, report via MQTT alert

**Acceptance:** Models are pre-downloaded before worker container starts. Cache prevents duplicate downloads across restarts and across workers.

---

### Step 4: Worker File Watching and Reload

**File:** `cyberwave_edge_core/worker_watcher.py` (new)

Monitors `~/.cyberwave/workers/` for changes and triggers worker container restart.

**Implementation approach:**

Use a polling loop (not inotify) for cross-platform compatibility and Docker bind-mount reliability:

1. Every N seconds (configurable, default 5s), compute a hash of all `*.py` files in the workers directory (sorted names + mtimes + sizes)
2. Compare with previous hash
3. If changed:
   - Log which files were added/removed/modified
   - Call `model_manager.ensure_models()` for any new model requirements
   - Call `worker_manager.restart_worker_container()`
   - Update stored hash

This runs as part of the existing `run_runtime_loop()` reconciliation cycle (already polls every ~15s).

**Worker set changes detected:**
- New `.py` file added → restart
- Existing `.py` file modified → restart
- `.py` file removed → restart
- Non-`.py` files → ignored

**Acceptance:** Adding/removing/modifying worker files triggers predictable container restart.

---

### Step 5: Zenoh Infrastructure Wiring

**Changes to:** `cyberwave_edge_core/startup.py` and `cyberwave_edge_core/worker_manager.py`

Ensure all managed containers (drivers + worker) receive consistent Zenoh transport configuration.

**For driver containers** (modification to `_run_docker_image`):

Add these environment variables to all driver containers:
- `CYBERWAVE_DATA_BACKEND=zenoh`
- `ZENOH_CONNECT` (if router configured)

This is additive — existing drivers that don't use the SDK data layer ignore these vars. Drivers that adopt `cw.data.publish()` (CYB-1547) will pick them up automatically.

**For worker container** (in worker_manager):

Same Zenoh env vars as above, plus shared memory configuration:
- `ZENOH_SHM_ENABLED=true` (when all containers are on the same host)

**Docker network:**

All driver containers and the worker container must be on the same Docker network for Zenoh peer-to-peer discovery. Today, driver container networking is configured via `_build_driver_network_args()`.

Options:
- If `--network host` is used (common on Linux edge devices): Zenoh peers discover each other via multicast — no extra config needed
- If a custom Docker network is used: create a shared `cyberwave-edge` Docker network and attach all containers

**Optional Zenoh router:**

The Zenoh router is only needed for MQTT bridge or multi-hop topologies. It is not started by default.

When configured (via `credentials.json` envs or metadata):
- Edge Core starts a `eclipse/zenoh:latest` container: `cyberwave-zenoh-router-{env_uuid_short}`
- Sets `ZENOH_CONNECT=tcp/{router_ip}:7447` in all service containers
- Router is started before drivers and workers

**Acceptance:** Zenoh transport settings are consistently applied to all managed containers.

---

### Step 6: Integration with Remote Restart and Cache Cleanup

**Changes to:** `cyberwave_edge_core/startup.py`

Update the existing `restart_edge_core` MQTT handler:

1. Stop worker container
2. Stop all driver containers (existing behavior)
3. Clear cached twin JSON files (existing behavior)
4. Optionally clear model cache (configurable, default: preserve)
5. Re-run startup: fetch twins → run drivers → ensure models → start worker container

Update `_remove_cached_twin_json_files()` to NOT delete the models directory (preserve model cache across restarts by default).

Add new MQTT command: `restart_workers` — restarts only the worker container without touching drivers:
- Topic: `cyberwave/edge/{fingerprint}/restart_workers`
- Behavior: stop worker → re-scan workers → re-ensure models → start worker

**Acceptance:** Remote restart handles worker container alongside drivers. Model cache persists across restarts.

---

### Step 7: Operational Logging and Status

**Changes to:** `cyberwave_edge_core/main.py`, `cyberwave_edge_core/startup.py`

The top-level `cyberwave-edge-core status` command is extended to include worker/model/Zenoh lines:

```
Cyberwave Edge Core — Status

  ✓ Credentials (0.001s)
  ✓ Token (0.234s)
  ✓ MQTT broker (0.102s)
  ✓ Driver containers: 2 running
  ✓ Worker container: running (3 workers loaded)
  ✓ Model cache: 2 models (156 MB)
  ✓ Zenoh: peer-to-peer (no router)
```

The `cyberwave-edge-core worker status` subcommand shows detailed worker info:

```
Worker container: cyberwave-worker-a1b2c3d4 (running)

  Workers:
    detect_people.py    (custom)     loaded
    wf_abc123_alert.py  (workflow)   loaded

  Models:
    yolov8n             156 MB   ultralytics   cached
    background-sub      12 MB    opencv        cached

  Zenoh: peer-to-peer (no router)
```

Add structured logging throughout:
- Worker container start/stop/restart events
- Model download start/complete/failure events
- Worker file change detection events
- Zenoh configuration applied

Publish worker status to MQTT for cloud visibility:
- Topic: `cyberwave/edge/{fingerprint}/worker_status`
- Payload: `{workers: [{name, origin, status}], models: [{id, size, runtime}], container_status}`

**Acceptance:** Both `status` and `worker status` show correct state. Events logged and published to MQTT.

---

## File Layout After Implementation

```
cyberwave_edge_core/
├── __init__.py                  # (existing) version
├── main.py                      # (modified) `worker` CLI subcommand group added
├── startup.py                   # (modified) worker lifecycle integration, Zenoh env wiring
├── utils.py                     # (existing) DriverStartingAlertContext
├── docker_helpers.py            # (new) internal Docker CLI wrappers — not user-facing
├── worker_manager.py            # (new) WorkerManager class backing CLI + boot path
├── model_manager.py             # (new) model download/cache/eviction
└── worker_watcher.py            # (new) file-change detection and reload trigger
```

---

## Implementation Order

```
Step 1: Internal Docker helpers refactor
  │     (refactor, no behavior change — safe to land independently)
  │
  ├── Step 2: Worker Manager + CLI commands
  │     (uses docker_helpers internally; exposes `worker start/stop/restart/status/logs`)
  │
  ├── Step 3: Model Manager
  │     (independent of worker manager, can be developed in parallel)
  │
  │     Step 4: Worker Watcher
  │       (depends on worker_manager + model_manager)
  │
  ├── Step 5: Zenoh Wiring
  │     (touches startup.py + worker_manager — can start after Step 2)
  │
  ├── Step 6: Remote Restart Integration
  │     (depends on Steps 2-5)
  │
  └── Step 7: Status and Logging
        (depends on Steps 2-3, can be developed incrementally)
```

Steps 2 and 3 can be developed in parallel after Step 1. Step 5 can start as soon as Step 2 is ready. Steps 6 and 7 integrate everything.

---

## Dependency Map

### Depends on (input from other issues)

| Issue | What this issue needs | Status |
|---|---|---|
| CYB-1544 | `CYBERWAVE_DATA_BACKEND` env var convention; Zenoh key-expression patterns | In Progress |
| CYB-1545 | Worker runtime entrypoint image (`cyberwaveos/ml-worker`); `cw.models.load()` contract for model path resolution | In Progress |

### Provides to (output for other issues)

| Issue | What this issue provides |
|---|---|
| CYB-1547 | Zenoh env vars injected into driver containers; Docker network shared with workers |
| CYB-1548 | Worker directory mount point; restart-on-change behavior for `wf_*.py` files |

### Internal dependencies on existing edge-core code

| Area | Risk |
|---|---|
| `_run_docker_image()` in `startup.py` | Moderate — env var injection and volume mount logic must be extended, not duplicated |
| `run_runtime_loop()` in `startup.py` | Low — additive reconciliation step |
| `_stop_and_prune_driver_containers()` | Low — add worker container to prune list |
| MQTT command handlers | Low — additive new command topic |

---

## Testing Strategy

### Unit tests

| Test | What it validates |
|---|---|
| `test_model_manager_cache_hit` | Cached model returns path without download |
| `test_model_manager_cache_miss` | Missing model triggers download + cache write |
| `test_model_manager_checksum_mismatch` | Corrupt cache triggers re-download |
| `test_model_manager_manifest_persistence` | Manifest survives process restart |
| `test_worker_watcher_detects_add` | New `.py` file triggers restart callback |
| `test_worker_watcher_detects_modify` | Modified `.py` file triggers restart callback |
| `test_worker_watcher_detects_remove` | Removed `.py` file triggers restart callback |
| `test_worker_watcher_ignores_non_py` | Non-`.py` files don't trigger restart |
| `test_worker_manager_env_vars` | Worker container gets correct env vars |
| `test_worker_manager_volumes` | Worker container mounts workers + models dirs |
| `test_worker_manager_gpu_detection` | GPU flag set when NVIDIA runtime available |
| `test_zenoh_env_applied_to_drivers` | Driver containers receive Zenoh env vars |

### Integration tests

| Test | What it validates |
|---|---|
| End-to-end boot with one worker + one model | Worker container starts with correct mounts/env after model download |
| Cache reuse across two workers using same model | Model downloaded once, both workers see it |
| Reload behavior after worker file add | Adding `.py` to workers dir → container restarts |
| Reload behavior after worker file remove | Removing `.py` from workers dir → container restarts |
| Remote restart preserves model cache | MQTT restart → models dir untouched → worker restarts faster |

### Manual verification

| Scenario | Steps |
|---|---|
| Local boot test | Install edge-core on test device → configure environment → add worker file → verify worker container starts with GPU + Zenoh env |
| Status command | Run `cyberwave-edge-core status` → verify worker/model/Zenoh status lines |
| MQTT restart | Publish restart command → verify worker + drivers restart cleanly |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Worker base image (`cyberwaveos/ml-worker`) not yet built | Blocked — worker container can't start | Build a minimal base image as part of this issue; use `cyberwave-edge-runtime/runtime-services/workers/ml-worker/` layout from README |
| SDK `cw.models.load()` not yet implemented (CYB-1545) | Workers can't load models at runtime | Model Manager pre-downloads to cache path; SDK loads from that path. Decouple download (Edge Core) from load (SDK) |
| Zenoh peer-to-peer may not work on all Docker network configurations | Inter-container data flow fails | Default to `--network host` on Linux; document Docker network requirements |
| Large model downloads on constrained networks | Slow/failed worker startup | Download with retry + resume; report progress via MQTT; let workers start even if some models fail |
| `startup.py` is ~3000 lines | Merge conflicts; hard to review | Step 1 extracts docker helpers first, reducing the blast radius of later steps |

---

## Out of Scope

Per the issue definition:
- Driver payload implementation details (CYB-1547)
- Workflow codegen itself (CYB-1548)
- SDK data layer internals (CYB-1544)
- SDK worker API internals (CYB-1545)
- Cloud worker runtime (CYB-1549)
