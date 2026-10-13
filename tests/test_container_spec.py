"""Tests for the container spec hash and the ``io.cyberwave.edge.*`` labels.

The hash is what a later edge-core uses to decide whether a running driver or
worker container can be kept (adoption, EC6). So it must be:

- stable: the same spec hashes the same across runs, key orders and flag orders;
- complete: any change to anything passed to ``docker run`` changes it.

Completeness is checked against the argv the real launch paths build: the
labels on it are recomputed from the argv itself, and dropping any single
option from it must change the hash. A new flag added to either launch path is
therefore covered without editing this file, and one that slipped past the
hash (built outside the spec, or dropped by the canonical form) fails here.
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import cyberwave_edge_core.docker_launch as docker_launch
import cyberwave_edge_core.driver_launcher as driver_launcher
import cyberwave_edge_core.startup as startup
import cyberwave_edge_core.worker_manager as wm_module
from cyberwave_edge_core import __version__
from cyberwave_edge_core.container_spec import (
    CLOUD_NODE_ID_ENV,
    CLOUD_NODE_LABEL,
    CREATED_BY_LABEL,
    EDGE_LABEL_PREFIX,
    LIVE_WORKLOADS_ENV,
    SERVICE_LABEL,
    SPEC_HASH_LABEL,
    TWIN_LABEL,
    WORKER_SERVICE,
    WORKLOAD_LABEL,
    WORKLOAD_UUID_ENV,
    DockerRunSpec,
    _group_options,
    live_workloads,
    workload_labels,
)
from cyberwave_edge_core.docker_helpers import docker_image_id
from cyberwave_edge_core.worker_manager import WorkerManager
from tests.test_multi_service_driver import _patch_driver_container_launch

_IMAGE = "cyberwaveos/so101-driver:dev"
_IMAGE_ID = "sha256:" + "a" * 64

_BASE_OPTIONS = (
    "--detach",
    "--init",
    "--restart",
    "unless-stopped",
    "--privileged",
    "--network",
    "host",
    "--name",
    "cyberwave-driver-aabbccdd",
    "-e",
    "CYBERWAVE_TWIN_UUID=aabbccdd",
    "-e",
    "CYBERWAVE_API_KEY=super-secret-token",
    "-v",
    "/etc/cyberwave:/app/.cyberwave:ro",
    "-v",
    "/data:/data",
    "--device",
    "/dev/ttyACM0:/dev/ttyACM0",
    "--gpus",
    "all",
)


def _spec(**overrides: Any) -> DockerRunSpec:
    fields: dict[str, Any] = {
        "options": _BASE_OPTIONS,
        "image": _IMAGE,
        "command": ("python", "-m", "driver"),
        "extra_networks": ("cyberwave-sim",),
        "image_id": _IMAGE_ID,
    }
    fields.update(overrides)
    return DockerRunSpec(**fields)


def _pairs_to_options(pairs: list[tuple[str, str | None]]) -> tuple[str, ...]:
    out: list[str] = []
    for flag, value in pairs:
        if flag:
            out.append(flag)
        if value is not None:
            out.append(value)
    return tuple(out)


def _labels_from_argv(argv: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for flag, value in _group_options(argv[2:]):
        if flag == "--label" and value and value.startswith(EDGE_LABEL_PREFIX):
            key, _, val = value.partition("=")
            labels[key] = val
    return labels


def _spec_from_run_argv(
    argv: list[str],
    *,
    image: str,
    image_id: str | None,
    extra_networks: tuple[str, ...] = (),
) -> DockerRunSpec:
    """Rebuild the spec from the argv handed to Docker, edge labels excluded."""
    assert argv[:2] == ["docker", "run"]
    image_idx = argv.index(image)
    options = tuple(
        token
        for flag, value in _group_options(argv[2:image_idx])
        if not (flag == "--label" and value and value.startswith(EDGE_LABEL_PREFIX))
        for token in ([flag] if flag else []) + ([value] if value is not None else [])
    )
    return DockerRunSpec(
        options=options,
        image=image,
        command=tuple(argv[image_idx + 1 :]),
        extra_networks=extra_networks,
        image_id=image_id,
    )


def _assert_every_option_is_hashed(spec: DockerRunSpec) -> None:
    """Dropping any single flag (with its value) from the argv changes the hash."""
    baseline = spec.spec_hash()
    pairs = _group_options(spec.options)
    assert pairs, "expected a non-empty docker run argv"
    for index, pair in enumerate(pairs):
        reduced = _pairs_to_options(pairs[:index] + pairs[index + 1 :])
        mutated = DockerRunSpec(
            options=reduced,
            image=spec.image,
            command=spec.command,
            extra_networks=spec.extra_networks,
            image_id=spec.image_id,
        )
        assert mutated.spec_hash() != baseline, f"{pair!r} is not covered by the spec hash"


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


class TestStability:
    def test_same_spec_same_hash(self) -> None:
        assert _spec().spec_hash() == _spec().spec_hash()

    def test_hash_is_pinned_across_runs(self) -> None:
        """A golden value: fails if the canonical form drifts between versions.

        If this changes on purpose, bump ``SPEC_SCHEMA_VERSION`` and update the
        value: every hash an older edge-core wrote stops matching, so adoption
        will recreate those containers once.
        """
        spec = DockerRunSpec(
            options=("--restart", "unless-stopped", "-e", "A=1", "-v", "/a:/a"),
            image="img:tag",
            image_id="sha256:" + "0" * 64,
        )
        assert spec.canonical_json() == (
            '{"command":[],"extra_networks":[],"image":"img:tag",'
            '"image_id":"sha256:' + "0" * 64 + '",'
            '"options":{"--env":{"A":"1"},"--restart":["unless-stopped"],'
            '"--volume":["/a:/a"]},"schema":1}'
        )
        assert spec.spec_hash() == (
            "56d79a62a46365b60f22b66db62062d3851aaf33728d6ed830a3ecf988333fec"
        )

    def test_env_order_does_not_matter(self) -> None:
        a = _spec(options=("-e", "A=1", "-e", "B=2"))
        b = _spec(options=("-e", "B=2", "-e", "A=1"))
        assert a.spec_hash() == b.spec_hash()

    def test_flag_order_does_not_matter(self) -> None:
        pairs = _group_options(_BASE_OPTIONS)
        reordered = _pairs_to_options(list(reversed(pairs)))
        assert _spec(options=reordered).spec_hash() == _spec().spec_hash()

    def test_set_like_flags_are_sorted(self) -> None:
        a = _spec(options=("-v", "/a:/a", "-v", "/b:/b", "--device", "/dev/x", "--device", "/y"))
        b = _spec(options=("--device", "/y", "-v", "/b:/b", "--device", "/dev/x", "-v", "/a:/a"))
        assert a.spec_hash() == b.spec_hash()

    def test_aliases_and_equals_form_are_equivalent(self) -> None:
        a = _spec(options=("-e", "A=1", "-v", "/a:/a", "--network", "host"))
        b = _spec(options=("--env", "A=1", "--volume=/a:/a", "--network=host"))
        assert a.spec_hash() == b.spec_hash()

    def test_env_last_value_wins_like_docker(self) -> None:
        a = _spec(options=("-e", "A=old", "-e", "A=new"))
        b = _spec(options=("-e", "A=new"))
        assert a.spec_hash() == b.spec_hash()

    def test_extra_network_order_does_not_matter(self) -> None:
        a = _spec(extra_networks=("n1", "n2"))
        b = _spec(extra_networks=("n2", "n1"))
        assert a.spec_hash() == b.spec_hash()

    def test_edge_labels_do_not_feed_the_hash(self) -> None:
        """The hash cannot cover itself, and created-by must not restart drivers."""
        labeled = _spec(
            options=(
                *_BASE_OPTIONS,
                "--label",
                f"{SPEC_HASH_LABEL}=whatever",
                "--label",
                f"{CREATED_BY_LABEL}=cyberwave-edge-core/9.9.9",
            )
        )
        assert labeled.spec_hash() == _spec().spec_hash()

    def test_version_does_not_feed_the_hash(self) -> None:
        with patch("cyberwave_edge_core.container_spec.__version__", "99.0.0"):
            other = _spec()
            assert other.labels()[CREATED_BY_LABEL] == "cyberwave-edge-core/99.0.0"
            assert other.spec_hash() == _spec().spec_hash()


# ---------------------------------------------------------------------------
# Sensitivity: any change to a field changes the hash
# ---------------------------------------------------------------------------


_MUTATIONS: dict[str, dict[str, Any]] = {
    "image reference": {"image": "cyberwaveos/so101-driver:staging"},
    "image id": {"image_id": "sha256:" + "b" * 64},
    "image id unknown": {"image_id": None},
    "command": {"command": ("python", "-m", "other")},
    "no command": {"command": ()},
    "extra network": {"extra_networks": ("cyberwave-sim", "ros")},
    "env value": {"options": tuple(o.replace("aabbccdd", "11223344") for o in _BASE_OPTIONS)},
    "env added": {"options": (*_BASE_OPTIONS, "-e", "NEW=1")},
    "env token": {"options": tuple(o.replace("super-secret", "rotated") for o in _BASE_OPTIONS)},
    "mount": {"options": (*_BASE_OPTIONS, "-v", "/extra:/extra")},
    "mount mode": {"options": tuple(o.replace(":ro", ":rw") for o in _BASE_OPTIONS)},
    "device": {"options": (*_BASE_OPTIONS, "--device", "/dev/video0")},
    "network": {"options": tuple("bridge" if o == "host" else o for o in _BASE_OPTIONS)},
    "restart policy": {
        "options": tuple("always" if o == "unless-stopped" else o for o in _BASE_OPTIONS)
    },
    "gpus": {"options": tuple("device=0" if o == "all" else o for o in _BASE_OPTIONS)},
    "entrypoint": {"options": (*_BASE_OPTIONS, "--entrypoint", "/bin/sh")},
    "container name": {
        "options": tuple(o + "-x" if o.startswith("cyberwave-driver") else o for o in _BASE_OPTIONS)
    },
    "boolean dropped": {"options": tuple(o for o in _BASE_OPTIONS if o != "--privileged")},
    "user label": {"options": (*_BASE_OPTIONS, "--label", "com.example=1")},
    "runtime": {"options": (*_BASE_OPTIONS, "--runtime", "nvidia")},
}


@pytest.mark.parametrize("change", sorted(_MUTATIONS))
def test_any_field_change_changes_hash(change: str) -> None:
    assert _spec(**_MUTATIONS[change]).spec_hash() != _spec().spec_hash()


def test_every_option_of_the_reference_spec_is_hashed() -> None:
    _assert_every_option_is_hashed(_spec())


def test_order_significant_repeats_keep_their_order() -> None:
    """Docker takes the last value of a single-valued flag; so does the hash."""
    a = _spec(options=("--network", "a", "--network", "b"))
    b = _spec(options=("--network", "b", "--network", "a"))
    assert a.spec_hash() != b.spec_hash()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestLabels:
    def test_driver_labels(self) -> None:
        labels = _spec().labels(twin_uuid="aabbccdd-full-uuid", service="driver")
        assert labels == {
            TWIN_LABEL: "aabbccdd-full-uuid",
            SERVICE_LABEL: "driver",
            SPEC_HASH_LABEL: _spec().spec_hash(),
            CREATED_BY_LABEL: f"cyberwave-edge-core/{__version__}",
        }

    def test_single_service_driver_has_no_service_label(self) -> None:
        assert SERVICE_LABEL not in _spec().labels(twin_uuid="t")

    def test_no_raw_env_value_in_labels(self) -> None:
        argv = _spec().run_argv(twin_uuid="aabbccdd", service="driver")
        label_values = [v for f, v in _group_options(argv[2:]) if f == "--label"]
        assert label_values
        assert not any("super-secret-token" in (v or "") for v in label_values)
        assert not any("CYBERWAVE_API_KEY" in (v or "") for v in label_values)

    def test_extra_labels_do_not_change_the_hash(self) -> None:
        spec = _spec()
        labels = spec.labels(extra={WORKLOAD_LABEL: "wl-1"})
        assert labels[WORKLOAD_LABEL] == "wl-1"
        assert labels[SPEC_HASH_LABEL] == spec.spec_hash()

    def test_run_argv_shape(self) -> None:
        spec = _spec()
        argv = spec.run_argv(twin_uuid="t", service="s")
        assert argv[:2] == ["docker", "run"]
        assert argv[2 : 2 + len(_BASE_OPTIONS)] == list(_BASE_OPTIONS)
        image_idx = argv.index(_IMAGE)
        assert argv[image_idx + 1 :] == list(spec.command)
        # Labels sit between the options and the image.
        assert argv[image_idx - 2] == "--label"
        assert _labels_from_argv(argv)[SPEC_HASH_LABEL] == spec.spec_hash()


class TestWorkloadLabels:
    def test_from_the_cloud_node_environment(self) -> None:
        env = {CLOUD_NODE_ID_ENV: "node-a", WORKLOAD_UUID_ENV: "wl-env"}
        assert workload_labels(env=env) == {CLOUD_NODE_LABEL: "node-a", WORKLOAD_LABEL: "wl-env"}
        assert workload_labels("wl-1", env=env)[WORKLOAD_LABEL] == "wl-1"

    def test_empty_outside_a_cloud_node_workload(self) -> None:
        assert workload_labels("wl-1", env={}) == {}
        assert workload_labels(env={CLOUD_NODE_ID_ENV: "node-a"}) == {}

    def test_live_workloads(self) -> None:
        assert live_workloads({LIVE_WORKLOADS_ENV: "a, b,,"}) == {"a", "b"}
        assert live_workloads({}) == set()

    def test_env_names_match_the_cloud_node(self) -> None:
        """The cloud node sets these variables; it does not import edge-core."""
        source = (
            Path(__file__).resolve().parents[2]
            / "cyberwave-cloud-nodes/cyberwave-cloud-node/cyberwave_cloud_node/workload_cleanup.py"
        )
        if not source.exists():
            pytest.skip("cloud node not checked out beside edge-core")
        constants = {
            node.targets[0].id: node.value.value
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        assert constants["CLOUD_NODE_ID_ENV"] == CLOUD_NODE_ID_ENV
        assert constants["WORKLOAD_UUID_ENV"] == WORKLOAD_UUID_ENV
        assert constants["LIVE_WORKLOADS_ENV"] == LIVE_WORKLOADS_ENV


# ---------------------------------------------------------------------------
# docker_image_id
# ---------------------------------------------------------------------------


class TestDockerImageId:
    def test_returns_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cyberwave_edge_core.docker_helpers.docker_available", lambda: True)
        monkeypatch.setattr(
            "cyberwave_edge_core.docker_helpers.subprocess.run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=f"{_IMAGE_ID}\n"),
        )
        assert docker_image_id(_IMAGE) == _IMAGE_ID

    @pytest.mark.parametrize(
        "result",
        [
            subprocess.CompletedProcess([], 1, stdout="", stderr="No such image"),
            subprocess.CompletedProcess([], 0, stdout="garbage\n"),
            subprocess.TimeoutExpired("docker", 10),
            OSError("boom"),
        ],
    )
    def test_unknown_is_none_and_never_raises(
        self, monkeypatch: pytest.MonkeyPatch, result: Any
    ) -> None:
        def _run(cmd: list[str], **kw: Any) -> Any:
            if isinstance(result, BaseException):
                raise result
            return result

        monkeypatch.setattr("cyberwave_edge_core.docker_helpers.docker_available", lambda: True)
        monkeypatch.setattr("cyberwave_edge_core.docker_helpers.subprocess.run", _run)
        assert docker_image_id(_IMAGE) is None

    def test_none_without_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cyberwave_edge_core.docker_helpers.docker_available", lambda: False)
        assert docker_image_id(_IMAGE) is None


# ---------------------------------------------------------------------------
# Real launch paths: the label hashes exactly the argv handed to Docker
# ---------------------------------------------------------------------------


class TestDriverLaunchLabels:
    def _launch(self, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, Any]:
        captured_cmds: list[list[str]] = []
        _patch_driver_container_launch(monkeypatch, captured_cmds)
        monkeypatch.setattr(driver_launcher, "docker_image_id", lambda image: _IMAGE_ID)
        launched: dict[str, Any] = {}

        def _fake_launch(**kw: Any) -> bool:
            launched.update(kw)
            return True

        monkeypatch.setattr(docker_launch, "launch_detached_container", _fake_launch)
        assert startup._run_docker_image(
            _IMAGE,
            kwargs.pop("params", ["-v", "/opt/data:/data", "--device", "/dev/ttyACM0"]),
            twin_uuid="aabbccdd-1234-5678-9012-abcdef012345",
            token="test-token-value-1234567890",
            skip_pull=True,
            **kwargs,
        )
        return launched

    def test_labels_hash_the_argv_passed_to_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        launched = self._launch(
            monkeypatch,
            service_name="driver",
            command=["ros2", "launch", "x"],
            extra_networks=["ros-net"],
            prefer_gpu=False,
        )
        argv: list[str] = launched["run_argv"]
        labels = _labels_from_argv(argv)
        assert labels[TWIN_LABEL] == "aabbccdd-1234-5678-9012-abcdef012345"
        assert labels[SERVICE_LABEL] == "driver"
        assert labels[CREATED_BY_LABEL] == f"cyberwave-edge-core/{__version__}"

        rebuilt = _spec_from_run_argv(
            argv,
            image=_IMAGE,
            image_id=_IMAGE_ID,
            extra_networks=tuple(launched["extra_networks"]),
        )
        assert rebuilt.command == ("ros2", "launch", "x")
        assert rebuilt.extra_networks == ("ros-net",)
        assert labels[SPEC_HASH_LABEL] == rebuilt.spec_hash()
        _assert_every_option_is_hashed(rebuilt)

    def test_no_raw_env_in_driver_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        argv = self._launch(monkeypatch)["run_argv"]
        labels = _labels_from_argv(argv)
        assert SERVICE_LABEL not in labels
        assert "CYBERWAVE_API_KEY=test-token-value-1234567890" in argv  # in the env...
        assert not any("test-token-value" in v for v in labels.values())  # ...not a label

    def test_same_inputs_same_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first = _labels_from_argv(self._launch(monkeypatch)["run_argv"])
        second = _labels_from_argv(self._launch(monkeypatch)["run_argv"])
        assert first[SPEC_HASH_LABEL] == second[SPEC_HASH_LABEL]

    def test_param_change_changes_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first = _labels_from_argv(self._launch(monkeypatch, params=[])["run_argv"])
        second = _labels_from_argv(self._launch(monkeypatch, params=["-e", "EXTRA=1"])["run_argv"])
        assert first[SPEC_HASH_LABEL] != second[SPEC_HASH_LABEL]


class TestWorkerLaunchLabels:
    def test_labels_hash_the_argv_passed_to_docker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workers_dir = tmp_path / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        (workers_dir / "model.py").write_text("pass\n")
        manager = WorkerManager(
            config_dir=tmp_path,
            environment_uuid="aabbccdd-eeff-0011-2233-445566778899",
            token="worker-secret-token-123",
            twin_uuids=["twin-uuid-1"],
            labels={CLOUD_NODE_LABEL: "node-a", WORKLOAD_LABEL: "wl-1"},
        )

        run_calls: list[list[str]] = []

        def fake_run(cmd: list, **kwargs: object) -> MagicMock:
            run_calls.append(cmd)
            m = MagicMock()
            m.returncode = 0
            return m

        monkeypatch.setattr(wm_module, "docker_available", lambda: True)
        monkeypatch.setattr(wm_module, "docker_container_status", lambda name: "none")
        monkeypatch.setattr(wm_module, "docker_has_nvidia_runtime", lambda: True)
        monkeypatch.setattr(wm_module, "docker_rm", lambda name, **kw: True)
        monkeypatch.setattr(wm_module, "docker_image_id", lambda image: _IMAGE_ID)
        # Never reach a real daemon: this test is about the argv, not the pull.
        monkeypatch.setattr(
            WorkerManager, "_pull_worker_image_with_progress", lambda self, image: True
        )
        monkeypatch.setattr(wm_module.subprocess, "run", fake_run)
        monkeypatch.setattr(wm_module.time, "sleep", lambda s: None)
        monkeypatch.setattr(
            "cyberwave_edge_core.startup.get_runtime_env_var",
            lambda name, default=None: default,
        )
        monkeypatch.setattr("cyberwave_edge_core.startup.load_credentials_envs", lambda: {})
        monkeypatch.setattr("os.environ", {})

        with patch.object(
            wm_module, "docker_inspect", return_value={"State": {"Status": "running"}}
        ):
            assert manager._run_container() is True

        argv = next(c for c in run_calls if c[:2] == ["docker", "run"])
        image = argv[-1]
        labels = _labels_from_argv(argv)
        assert TWIN_LABEL not in labels
        assert labels[SERVICE_LABEL] == WORKER_SERVICE
        assert labels[CREATED_BY_LABEL] == f"cyberwave-edge-core/{__version__}"
        assert labels[CLOUD_NODE_LABEL] == "node-a"
        assert labels[WORKLOAD_LABEL] == "wl-1"
        assert not any("worker-secret-token" in v for v in labels.values())

        rebuilt = _spec_from_run_argv(argv, image=image, image_id=_IMAGE_ID)
        assert labels[SPEC_HASH_LABEL] == rebuilt.spec_hash()
        assert "--gpus" in json.dumps(rebuilt.canonical())
        _assert_every_option_is_hashed(rebuilt)
