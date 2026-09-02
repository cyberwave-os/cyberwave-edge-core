"""Tests for the image-declared driver topology (CYB-3469).

The graph arrives as the JSON `docker compose config --no-interpolate` emits,
carried in an image label. These tests pin the parts that bit during design:

  * `docker compose config` renders services **alphabetically**, so launch order
    has to come from `depends_on` or `bridges` starts before `plant`.
  * `environment` comes back as a mapping when nothing overrode it and as a
    ``["K=v"]`` list when a variant did — both shapes from the same file.
  * interpolation is deferred to the edge, so `${CW_CHANNEL_TAG}` must expand
    while `${VAR:-default}` must survive for the container's own shell, and
    Compose's `$$` escape must collapse.
  * every "this image says nothing useful" case falls back to metadata rather
    than raising, because an older image is not an error.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

import cyberwave_edge_core.driver_selection as ds


@pytest.fixture(autouse=True)
def _no_inherited_variant(monkeypatch: Any) -> None:
    """Keep the host's own orchestrator setting out of these tests.

    Every test below that omits ``variant=`` is asserting the *default* path, so
    a developer or CI runner that exports CYBERWAVE_DRIVER_COMPOSE_VARIANT must
    not change what they mean.
    """
    monkeypatch.delenv(ds.COMPOSE_VARIANT_ENV, raising=False)
    monkeypatch.delenv(ds.CHANNEL_TAG_ENV, raising=False)
    # The sim-proxy train resolves from the deployment environment when nothing
    # pins it, so a developer's own CYBERWAVE_ENVIRONMENT would otherwise decide
    # what the default tests mean.
    monkeypatch.delenv(ds.SIM_PROXY_TAG_ENV, raising=False)
    monkeypatch.delenv("CYBERWAVE_ENVIRONMENT", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)


def _labels(monkeypatch: Any, mapping: dict[str, str]) -> None:
    """Serve *mapping* as the image's labels; anything else reads as absent."""
    monkeypatch.setattr(ds, "_image_labels", lambda image: dict(mapping))


def _encode(doc: dict[str, Any]) -> str:
    """Encode *doc* the way the driver build bakes it: base64 over compact JSON."""
    return base64.b64encode(json.dumps(doc).encode("utf-8")).decode("ascii")


def _label_set(doc: dict[str, Any], *, schema: str = "2", variant: str = "robot") -> dict[str, str]:
    return {
        ds._COMPOSE_SCHEMA_LABEL: schema,
        f"{ds._COMPOSE_LABEL_PREFIX}.{variant}": _encode(doc),
        ds._CHANNEL_TAG_LABEL: "jetson-humble-staging",
    }


ROBOT_DOC: dict[str, Any] = {
    "x-cyberwave": {"schema": 1, "variant": "robot", "state_owner": "navigation_bridge"},
    "services": {
        # Alphabetical, exactly as `config` emits it — bridges before plant.
        "bridges": {
            "image": "cyberwaveos/go2-ros2-driver:${CW_CHANNEL_TAG}",
            "network_mode": "host",
            "depends_on": {"plant": {"condition": "service_healthy", "required": True}},
            "x-cyberwave": {"prefer_gpu": True},
            "command": ["bash", "-lc", "ros2 launch x.py rviz:=${ENABLE_RVIZ2:-false}"],
        },
        "plant": {
            "image": "cyberwaveos/go2-ros2-driver:${CW_CHANNEL_TAG}",
            "network_mode": "host",
            "environment": {"ROS_DOMAIN_ID": "42"},
            "volumes": [
                {"type": "bind", "source": "/etc/cyberwave/x", "target": "/data"},
                {
                    "type": "bind",
                    "source": "/etc/localtime",
                    "target": "/etc/localtime",
                    "read_only": True,
                },
            ],
            "healthcheck": {"test": ["CMD-SHELL", "source /opt/ros/$$ROS_DISTRO/setup.bash"]},
        },
    },
}


# ---------------------------------------------------------------------------
# fallback behaviour — none of these may raise
# ---------------------------------------------------------------------------


def test_an_absent_image_is_not_an_image_with_no_labels(monkeypatch: Any) -> None:
    """None and {} are different answers, and both fall back.

    This replaces a test for the Go-template ``<no value>`` literal, which a
    single ``{{json .Config.Labels}}`` read no longer has to special-case: a
    missing key is simply absent from the dict. The distinction that still
    matters is the one kept here -- "the image is not on this host" versus "it
    is here and declares nothing".
    """
    monkeypatch.setattr(ds, "_image_labels", lambda image: None)
    assert ds.get_image_declared_services("img:tag") is None
    monkeypatch.setattr(ds, "_image_labels", lambda image: {})
    assert ds.get_image_declared_services("img:tag") is None


def test_unknown_schema_refuses(monkeypatch: Any) -> None:
    _labels(monkeypatch, _label_set(ROBOT_DOC, schema="99"))
    assert ds.get_image_declared_services("img:tag") is None


def test_non_numeric_schema_refuses(monkeypatch: Any) -> None:
    _labels(monkeypatch, _label_set(ROBOT_DOC, schema="latest"))
    assert ds.get_image_declared_services("img:tag") is None


def test_missing_variant_falls_back(monkeypatch: Any) -> None:
    _labels(monkeypatch, _label_set(ROBOT_DOC, variant="robot"))
    assert ds.get_image_declared_services("img:tag", variant="sim") is None


def test_empty_variant_falls_back(monkeypatch: Any) -> None:
    """The Dockerfile's ARG default: a variant this image does not ship."""
    labels = _label_set(ROBOT_DOC)
    labels[f"{ds._COMPOSE_LABEL_PREFIX}.robot"] = ""
    _labels(monkeypatch, labels)
    assert ds.get_image_declared_services("img:tag") is None


def test_empty_object_variant_falls_back(monkeypatch: Any) -> None:
    """An encoded but empty graph reads as "no such variant", not as a graph."""
    labels = _label_set(ROBOT_DOC)
    labels[f"{ds._COMPOSE_LABEL_PREFIX}.robot"] = _encode({})
    _labels(monkeypatch, labels)
    assert ds.get_image_declared_services("img:tag") is None


def test_malformed_json_falls_back(monkeypatch: Any) -> None:
    """Decodes as base64, but the bytes underneath are not JSON."""
    labels = _label_set(ROBOT_DOC)
    labels[f"{ds._COMPOSE_LABEL_PREFIX}.robot"] = base64.b64encode(b"{not json").decode("ascii")
    _labels(monkeypatch, labels)
    assert ds.get_image_declared_services("img:tag") is None


def test_unencoded_payload_falls_back(monkeypatch: Any) -> None:
    """Raw JSON in a schema-2 label is refused rather than parsed.

    Schema 1 put raw JSON here, and `docker/build-push-action`'s CSV parsing of
    `build-args` stripped the quotes off every comma-delimited quoted token on
    the way in -- so the raw form is exactly the shape that cannot be trusted.
    Refusing it keeps a mixed-encoding image from half-starting.
    """
    labels = _label_set(ROBOT_DOC)
    labels[f"{ds._COMPOSE_LABEL_PREFIX}.robot"] = json.dumps(ROBOT_DOC)
    _labels(monkeypatch, labels)
    assert ds.get_image_declared_services("img:tag") is None


def test_schema_one_refuses(monkeypatch: Any) -> None:
    """An image built before the base64 label must fall back, not be reread.

    Its payload lost quotes in the build, so "parse what is there" would either
    fail confusingly or -- worse, if a future graph happened to survive the
    mangling -- start a graph that is not the declared one.
    """
    labels = _label_set(ROBOT_DOC, schema="1")
    labels[f"{ds._COMPOSE_LABEL_PREFIX}.robot"] = json.dumps(ROBOT_DOC)
    _labels(monkeypatch, labels)
    assert ds.get_image_declared_services("img:tag") is None


def test_an_untranslated_key_is_refused_rather_than_dropped(monkeypatch: Any) -> None:
    """A key Edge Core cannot honour must not be silently ignored.

    `docker compose config` accepts every legal Compose key, so `privileged`,
    `devices` and `cap_add` all reach the label. Translating four keys and
    quietly dropping the rest is how a robot ends up with a container that
    reports healthy while lacking the hardware access it declared. Refuse the
    graph instead, and say which keys did it.
    """
    _labels(
        monkeypatch,
        _label_set({"services": {"plant": {"image": "p:dev", "privileged": True}}}),
    )
    assert ds.get_image_declared_services("img:dev") is None


def test_the_refusal_names_the_offending_keys(monkeypatch: Any, caplog: Any) -> None:
    """ "Unusable compose graph" alone sends a reader to the wrong place."""
    _labels(
        monkeypatch,
        _label_set(
            {
                "services": {
                    "plant": {
                        "image": "p:dev",
                        "devices": ["/dev/ttyUSB0"],
                        "cap_add": ["SYS_ADMIN"],
                    }
                }
            }
        ),
    )
    with caplog.at_level("ERROR"):
        assert ds.get_image_declared_services("img:dev") is None
    assert "cap_add" in caplog.text and "devices" in caplog.text


def test_edge_core_owned_keys_are_ignored_not_refused(monkeypatch: Any) -> None:
    """`restart` is declared in every service and overridden by the launcher.

    Refusing it would reject the real files; two owners for one setting is the
    thing being avoided, and the launcher is the owner that has to reconcile a
    restart loop.
    """
    _labels(
        monkeypatch,
        _label_set({"services": {"plant": {"image": "p:dev", "restart": "unless-stopped"}}}),
    )
    resolved = ds.get_image_declared_services("img:dev")
    assert resolved is not None
    assert [spec.name for spec in resolved[0]] == ["plant"]


def test_service_without_image_falls_back(monkeypatch: Any) -> None:
    doc = {"services": {"plant": {"command": ["true"]}}}
    _labels(monkeypatch, _label_set(doc))
    assert ds.get_image_declared_services("img:tag") is None


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------


def test_dependencies_start_before_dependents(monkeypatch: Any) -> None:
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert [s.name for s in specs][0] == "plant", "plant must not start after bridges"


def test_dependency_cycle_is_rejected(monkeypatch: Any) -> None:
    doc = {
        "services": {
            "a": {"image": "i", "depends_on": {"b": {}}},
            "b": {"image": "i", "depends_on": {"a": {}}},
        }
    }
    _labels(monkeypatch, _label_set(doc))
    # Surfaces as a fallback rather than a crashed boot, but must never be an
    # arbitrary order.
    assert ds.get_image_declared_services("img:tag") is None


def test_dependency_outside_the_variant_is_ignored(monkeypatch: Any) -> None:
    """The sim variant omits services the robot variant depends on."""
    doc = {"services": {"nav2": {"image": "i", "depends_on": {"nav-bridge": {}}}}}
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert [s.name for s in specs] == ["nav2"]


# ---------------------------------------------------------------------------
# substitution
# ---------------------------------------------------------------------------


def test_channel_tag_is_substituted(monkeypatch: Any) -> None:
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert all("${CW_CHANNEL_TAG}" not in s.image for s in specs)
    assert all(s.image.endswith(":jetson-humble-staging") for s in specs)


def test_channel_tag_falls_back_to_the_image_tag(monkeypatch: Any) -> None:
    labels = _label_set(ROBOT_DOC)
    del labels[ds._CHANNEL_TAG_LABEL]
    _labels(monkeypatch, labels)
    specs, _, _ = ds.get_image_declared_services("cyberwaveos/go2-ros2-driver:humble-dev")
    assert all(s.image.endswith(":humble-dev") for s in specs)


def test_the_channel_tag_env_pins_every_sibling(monkeypatch: Any) -> None:
    """The regression test for a run that tests one image out of four.

    A CI leg pins the driver image to a PR tag, but the label it carries still
    says the train it was BUILT in (`jetson-humble-staging` here). Without an
    override every sibling — nav2, slam, the proxy — resolves to that train, so
    the run assembles a graph that is one PR image plus three from elsewhere and
    then reports green for it. Nothing about that is visible: each sibling is a
    real image that pulls and starts.
    """
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    monkeypatch.setenv(ds.CHANNEL_TAG_ENV, "humble-pr-3749-sha-abc123")

    specs, _, _ = ds.get_image_declared_services("go2-ros2-driver:humble-pr-3749-sha-abc123")

    assert specs, "the graph did not resolve at all"
    assert all(s.image.endswith(":humble-pr-3749-sha-abc123") for s in specs), (
        f"siblings resolved to {sorted({s.image for s in specs})}"
    )


def test_an_explicit_channel_tag_beats_the_environment(monkeypatch: Any) -> None:
    """Same precedence as the variant: a caller that knows wins over the host."""
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    monkeypatch.setenv(ds.CHANNEL_TAG_ENV, "from-env")
    specs, _, _ = ds.get_image_declared_services("img:tag", channel_tag="from-argument")
    assert all(s.image.endswith(":from-argument") for s in specs)


def test_the_label_still_wins_when_nothing_overrides_it(monkeypatch: Any) -> None:
    """Production behaviour is unchanged: the image says which train it is on.

    The image reference is deliberately a different tag, so a fallback creeping
    ahead of the label would fail here rather than pass by coincidence.
    """
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    specs, _, _ = ds.get_image_declared_services("go2-ros2-driver:some-other-tag")
    assert all(s.image.endswith(":jetson-humble-staging") for s in specs)


def test_the_resolved_channel_source_is_logged(monkeypatch: Any, caplog: Any) -> None:
    """ "Which train did the siblings come from, and why" has to be greppable."""
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    monkeypatch.setenv(ds.CHANNEL_TAG_ENV, "humble-pr-1")
    with caplog.at_level("INFO"):
        ds.get_image_declared_services("img:tag")
    rendered = [r.getMessage() for r in caplog.records]
    assert any("channel=humble-pr-1" in line and ds.CHANNEL_TAG_ENV in line for line in rendered), (
        rendered
    )


def test_shell_default_syntax_survives(monkeypatch: Any) -> None:
    """``${VAR:-default}`` belongs to the container's shell, not to us."""
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    bridges = next(s for s in specs if s.name == "bridges")
    assert "${ENABLE_RVIZ2:-false}" in bridges.command[-1]


def test_plain_placeholder_resolves_from_the_environment(monkeypatch: Any) -> None:
    """Compose would expand this at ``up`` time; deferring means we must."""
    monkeypatch.setenv("CYBERWAVE_SIM_DOCKER_NETWORK", "sim-net")
    doc = {
        "networks": {"simnet": {"name": "${CYBERWAVE_SIM_DOCKER_NETWORK}", "external": True}},
        "services": {"plant": {"image": "i:t", "networks": {"simnet": None}}},
    }
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert specs[0].params == ["--network", "sim-net"]


def test_unset_plain_placeholder_stays_literal(monkeypatch: Any) -> None:
    """Blanking it the way Compose does would hide the misconfiguration."""
    monkeypatch.delenv("CYBERWAVE_SIM_DOCKER_NETWORK", raising=False)
    doc = {
        "networks": {"simnet": {"name": "${CYBERWAVE_SIM_DOCKER_NETWORK}"}},
        "services": {"plant": {"image": "i:t", "networks": {"simnet": None}}},
    }
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert specs[0].params == ["--network", "${CYBERWAVE_SIM_DOCKER_NETWORK}"]


def test_compose_dollar_escape_collapses() -> None:
    assert ds._substitute("source /opt/ros/$$ROS_DISTRO/x", {}) == "source /opt/ros/$ROS_DISTRO/x"


# ---------------------------------------------------------------------------
# field mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"A": "1", "B": "2"}, {"A": "1", "B": "2"}),
        (["A=1", "B=2"], {"A": "1", "B": "2"}),
        (["A="], {"A": ""}),
        ({"A": None}, {"A": ""}),
        (None, {}),
    ],
)
def test_environment_accepts_both_shapes(raw: Any, expected: dict[str, str]) -> None:
    assert ds._compose_env(raw) == expected


def test_volumes_become_bind_flags(monkeypatch: Any) -> None:
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    plant = next(s for s in specs if s.name == "plant")
    assert "-v" in plant.params
    assert "/etc/cyberwave/x:/data" in plant.params
    assert "/etc/localtime:/etc/localtime:ro" in plant.params, "read_only must map to :ro"


def test_prefer_gpu_comes_from_the_extension_field(monkeypatch: Any) -> None:
    """Not deploy.resources: Edge Core degrades where Compose would hard-fail."""
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    by_name = {s.name: s for s in specs}
    assert by_name["bridges"].prefer_gpu is True
    assert by_name["plant"].prefer_gpu is False


def test_string_command_is_wrapped_for_exec(monkeypatch: Any) -> None:
    doc = {"services": {"plant": {"image": "i:t", "command": "ros2 run a b"}}}
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert specs[0].command == ["/bin/sh", "-c", "ros2 run a b"]


def test_compose_carries_no_shared_env_or_params(monkeypatch: Any) -> None:
    """`config` expands anchors into each service, so there is nothing shared."""
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    _, shared_env, shared_params = ds.get_image_declared_services("img:tag")
    assert shared_env == {}
    assert shared_params == []


# ---------------------------------------------------------------------------
# precedence
# ---------------------------------------------------------------------------


def test_metadata_services_win_over_a_label() -> None:
    """The escape hatch: a twin pinned to its own topology keeps it."""
    metadata = {
        "default": {
            "services": [{"name": "only", "image": "from-metadata:tag"}],
        }
    }
    resolved = ds._get_driver_services(metadata)
    assert resolved is not None
    specs, _, _ = resolved
    assert [s.image for s in specs] == ["from-metadata:tag"]


# ---------------------------------------------------------------------------
# the sim membrane
# ---------------------------------------------------------------------------


def test_multi_network_service_splits_primary_from_extras(monkeypatch: Any) -> None:
    """The sim plant proxy sits on two networks by design.

    Zenoh out to the cyberwave-sim plant, DDS in to the ROS graph — that split is
    what keeps physics off the ROS segment. `docker create` takes only the first;
    the rest are connected between create and start (see
    test_multi_network_attach.py). Order follows the file, so the author decides
    which one create gets.
    """
    doc = {
        "networks": {
            "stacknet": {"name": "cyberwave-stack-abcd1234", "driver": "bridge"},
            "simnet": {"name": "sim-net", "external": True},
        },
        "services": {
            "plant": {"image": "i:t", "networks": {"stacknet": None, "simnet": None}},
        },
    }
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert specs[0].params == ["--network", "cyberwave-stack-abcd1234"]
    assert specs[0].extra_networks == ["sim-net"]


def test_twin_uuid_resolves_the_per_twin_network_name(monkeypatch: Any) -> None:
    """Two sims on one host must not share a ROS network."""
    doc = {
        "networks": {"stacknet": {"name": "cyberwave-stack-${CW_TWIN8}"}},
        "services": {"plant": {"image": "i:t", "networks": {"stacknet": None}}},
    }
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services(
        "img:tag", twin_uuid="abcd1234-5678-90ab-cdef-000000000000"
    )
    assert specs[0].params == ["--network", "cyberwave-stack-abcd1234"]


def test_single_network_service_still_resolves(monkeypatch: Any) -> None:
    doc = {
        "networks": {"stacknet": {"driver": "bridge"}},
        "services": {"nav2": {"image": "i:t", "networks": {"stacknet": None}}},
    }
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert specs[0].params == ["--network", "stacknet"]


# --- healthcheck translation ------------------------------------------------
#
# Declared in the Compose file but silently dropped before this: Docker reported
# no health state, so `depends_on: {condition: service_healthy}` had nothing to
# wait on and every dependent started as soon as its dependency was *created*.


def _health_flags(params: list[str]) -> dict[str, str]:
    """`--health-*` flags as a mapping, so assertions ignore ordering."""
    flags: dict[str, str] = {}
    for index, item in enumerate(params):
        if item.startswith("--health-") and index + 1 < len(params):
            flags[item] = params[index + 1]
        elif item == "--no-healthcheck":
            flags[item] = ""
    return flags


def test_healthcheck_becomes_docker_health_flags(monkeypatch: Any) -> None:
    doc = {
        "services": {
            "plant": {
                "image": "i:t",
                "healthcheck": {
                    "test": ["CMD-SHELL", "ros2 node list | grep -q go2_driver_node"],
                    "interval": "5s",
                    "timeout": "10s",
                    "start_period": "10s",
                    "retries": 24,
                },
            }
        }
    }
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert _health_flags(specs[0].params) == {
        "--health-cmd": "ros2 node list | grep -q go2_driver_node",
        "--health-interval": "5s",
        "--health-timeout": "10s",
        "--health-start-period": "10s",
        "--health-retries": "24",
    }


def test_healthcheck_collapses_the_compose_dollar_escape(monkeypatch: Any) -> None:
    """`$$ROS_DISTRO` in the file must reach the container as `$ROS_DISTRO`.

    The health command runs inside the container, so the expansion belongs to
    the container's shell — not to Edge Core, and not to Compose's renderer.
    """
    doc = {
        "services": {
            "plant": {
                "image": "i:t",
                "healthcheck": {"test": ["CMD-SHELL", "source /opt/ros/$$ROS_DISTRO/setup.bash"]},
            }
        }
    }
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert _health_flags(specs[0].params) == {
        "--health-cmd": "source /opt/ros/$ROS_DISTRO/setup.bash"
    }


@pytest.mark.parametrize(
    "test_value",
    [["NONE"], "NONE"],
    ids=["list", "string"],
)
def test_healthcheck_none_disables_the_image_default(monkeypatch: Any, test_value: Any) -> None:
    """An image can bake in its own healthcheck; NONE must switch it off."""
    doc = {"services": {"plant": {"image": "i:t", "healthcheck": {"test": test_value}}}}
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    if test_value == "NONE":
        # A bare string is a shell command to Compose, not the NONE sentinel.
        assert _health_flags(specs[0].params) == {"--health-cmd": "NONE"}
    else:
        assert "--no-healthcheck" in specs[0].params


def test_healthcheck_disable_flag_is_honoured(monkeypatch: Any) -> None:
    doc = {"services": {"plant": {"image": "i:t", "healthcheck": {"disable": True}}}}
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert specs[0].params == ["--no-healthcheck"]


def test_service_without_healthcheck_gets_no_health_flags(monkeypatch: Any) -> None:
    doc = {"services": {"nav2": {"image": "i:t"}}}
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert _health_flags(specs[0].params) == {}


def test_exec_form_healthcheck_is_joined_into_a_shell_string(monkeypatch: Any) -> None:
    """`--health-cmd` has no exec form, so CMD is joined with quoting preserved."""
    doc = {
        "services": {
            "plant": {
                "image": "i:t",
                "healthcheck": {"test": ["CMD", "/bin/probe", "--flag", "two words"]},
            }
        }
    }
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    assert _health_flags(specs[0].params) == {"--health-cmd": "/bin/probe --flag 'two words'"}


# --- depends_on conditions --------------------------------------------------


def test_depends_on_conditions_are_carried_on_the_spec(monkeypatch: Any) -> None:
    """Ordering was already derived from depends_on; the condition is the half
    that decides whether the launcher waits for health or just for start."""
    doc = {
        "services": {
            "plant": {"image": "i:t"},
            "nav2": {"image": "i:t", "depends_on": {"plant": {"condition": "service_healthy"}}},
            "map-stream-bridge": {
                "image": "i:t",
                "depends_on": {"nav2": {"condition": "service_started"}},
            },
        }
    }
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    by_name = {s.name: s for s in specs}
    assert by_name["plant"].depends_on == {}
    assert by_name["nav2"].depends_on == {"plant": "service_healthy"}
    assert by_name["map-stream-bridge"].depends_on == {"nav2": "service_started"}


def test_short_form_depends_on_means_service_started(monkeypatch: Any) -> None:
    """A bare list carries no condition, and started is what ordering already
    guarantees — so it must not make the launcher block on health."""
    doc = {
        "services": {
            "plant": {"image": "i:t"},
            "nav2": {"image": "i:t", "depends_on": ["plant"]},
        }
    }
    _labels(monkeypatch, _label_set(doc))
    specs, _, _ = ds.get_image_declared_services("img:tag")
    by_name = {s.name: s for s in specs}
    assert by_name["nav2"].depends_on == {"plant": "service_started"}


# --- variant selection ------------------------------------------------------
#
# Which variant a process resolves decides which plant starts, so the
# precedence is a contract rather than a convenience: an explicit argument from
# a caller that knows (the simulation dispatcher), then the deployment's
# environment, then the physical-robot default that Edge Core has always
# implied. Nothing here inspects the twin — see COMPOSE_VARIANT_ENV.


def test_argument_wins_over_the_environment(monkeypatch: Any) -> None:
    monkeypatch.setenv(ds.COMPOSE_VARIANT_ENV, "robot")
    assert ds._resolve_compose_variant("sim") == ("sim", "argument")


def test_environment_wins_over_the_default(monkeypatch: Any) -> None:
    """How a cyberwave-sim cloud node selects the sim topology without every
    call site having to plumb the variant through."""
    monkeypatch.setenv(ds.COMPOSE_VARIANT_ENV, "sim")
    assert ds._resolve_compose_variant(None) == ("sim", ds.COMPOSE_VARIANT_ENV)


def test_default_is_the_physical_robot(monkeypatch: Any) -> None:
    assert ds._resolve_compose_variant(None) == (ds.DEFAULT_COMPOSE_VARIANT, "default")
    assert ds.DEFAULT_COMPOSE_VARIANT == "robot"


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_environment_is_not_a_variant(monkeypatch: Any, blank: str) -> None:
    """An unset variable and one exported empty by a shell wrapper or a compose
    ``VAR:`` line have to mean the same thing, or a deployment silently asks for
    a label named ``io.cyberwave.driver.compose.`` and falls back to metadata."""
    monkeypatch.setenv(ds.COMPOSE_VARIANT_ENV, blank)
    assert ds._resolve_compose_variant(None) == (ds.DEFAULT_COMPOSE_VARIANT, "default")


def test_an_unknown_variant_is_not_rejected_here(monkeypatch: Any) -> None:
    """No allow-list: a non-ROS stack may declare variants this module has
    never heard of. A typo surfaces as a missing label, which logs the image and
    the variant asked for and falls back to metadata."""
    monkeypatch.setenv(ds.COMPOSE_VARIANT_ENV, "hil-bench")
    assert ds._resolve_compose_variant(None) == ("hil-bench", ds.COMPOSE_VARIANT_ENV)


def test_the_environment_selects_the_sim_topology_end_to_end(monkeypatch: Any) -> None:
    """The whole point of the variable: the same image and the same call site
    resolve a different graph on a cloud node than on a robot."""
    labels = _label_set(ROBOT_DOC, variant="robot")
    sim_doc = {"services": {"plant": {"image": "i:t", "networks": ["simnet"]}}}
    labels[f"{ds._COMPOSE_LABEL_PREFIX}.sim"] = _encode(sim_doc)
    _labels(monkeypatch, labels)

    robot_specs, _, _ = ds.get_image_declared_services("img:tag")
    assert sorted(s.name for s in robot_specs) == ["bridges", "plant"]

    monkeypatch.setenv(ds.COMPOSE_VARIANT_ENV, "sim")
    sim_specs, _, _ = ds.get_image_declared_services("img:tag")
    assert [s.name for s in sim_specs] == ["plant"]
    assert "--network" in sim_specs[0].params


def test_the_resolved_variant_and_its_source_are_logged(monkeypatch: Any, caplog: Any) -> None:
    """When a twin comes up with the wrong containers, "which variant did this
    process pick, and why" is the first question — it must be answerable from
    the log alone."""
    monkeypatch.setenv(ds.COMPOSE_VARIANT_ENV, "robot")
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    with caplog.at_level("INFO", logger=ds.logger.name):
        ds.get_image_declared_services("img:tag")
    assert f"variant=robot from {ds.COMPOSE_VARIANT_ENV}" in caplog.text


# ── failure-path forensics ───────────────────────────────────────────────────
# The JSONDecodeError offset says where the reader gave up, not why. This label
# once reached a consumer 88 characters short -- build-push-action CSV-parses
# build-args, and a field that is ENTIRELY one quoted token loses its quotes, so
# 44 quote pairs vanished. Finding that needed the image config blob pulled by
# hand. Length plus both ends puts the shape in the log instead.


def test_a_malformed_label_logs_its_length_and_digest(monkeypatch: Any, caplog: Any) -> None:
    labels = _label_set(ROBOT_DOC)
    truncated = json.dumps(ROBOT_DOC)[:-40]  # a plausible truncation
    labels[f"{ds._COMPOSE_LABEL_PREFIX}.robot"] = base64.b64encode(truncated.encode()).decode(
        "ascii"
    )
    _labels(monkeypatch, labels)
    with caplog.at_level("ERROR"):
        assert ds.get_image_declared_services("img:tag") is None
    blob = caplog.text
    assert f"len={len(truncated)}" in blob, blob
    assert "sha256=" in blob and "head=" in blob and "tail=" in blob, blob


def test_a_non_base64_label_also_reports_its_length(monkeypatch: Any, caplog: Any) -> None:
    labels = _label_set(ROBOT_DOC)
    labels[f"{ds._COMPOSE_LABEL_PREFIX}.robot"] = "!!! not base64 !!!"
    _labels(monkeypatch, labels)
    with caplog.at_level("ERROR"):
        assert ds.get_image_declared_services("img:tag") is None
    assert "len=18" in caplog.text, caplog.text


def test_a_healthy_run_never_dumps_the_label(monkeypatch: Any, caplog: Any) -> None:
    """The decoded graph is kilobytes and the CI step that shows these is capped.

    Burying a verdict under a base64 dump is the exact failure mode that already
    cost this suite two diagnostic rounds, so the forensics stay on the failure
    path only.
    """
    _labels(monkeypatch, _label_set(ROBOT_DOC))
    with caplog.at_level("DEBUG"):
        assert ds.get_image_declared_services("img:tag") is not None
    assert "sha256=" not in caplog.text
    assert "head=" not in caplog.text


def test_a_short_payload_is_shown_whole_rather_than_split():
    """Below the edge threshold, head/tail would just repeat the same bytes."""
    out = ds._label_forensics("{}")
    assert "body=" in out and "head=" not in out


# --- the sim-proxy train ------------------------------------------------------
#
# ros2-sim-proxy is the one service in the sim graph that is NOT from the driver
# stack. It is built by ros2-sim-runtime-build-and-push.yml, whose tags carry no
# ROS distro (`pr-<n>-<sha12>`, or the moving `<branch>`), while the driver, nav2
# and slam are built by edge-ros2-go2-driver-build-and-push.yml as
# `<distro>-pr-<n>-sha-<sha12>`. Rendering CW_CHANNEL_TAG for the proxy asked for
# a tag that repository has never published — the plant could not be pulled on
# any run, which is why the declared sim graph had never come up.


def test_the_sim_proxy_tag_is_not_the_driver_channel(monkeypatch: Any) -> None:
    """The regression itself. One variable cannot name two release trains."""
    monkeypatch.setenv("CYBERWAVE_ENVIRONMENT", "production")
    subs = ds.compose_substitutions(channel_tag="humble-pr-3749-sha-7f29d2c1c975")
    assert subs["CW_CHANNEL_TAG"] == "humble-pr-3749-sha-7f29d2c1c975"
    assert subs["CW_SIM_PROXY_TAG"] == "production"


def test_the_sim_proxy_tag_defaults_to_the_deployment_environment(
    monkeypatch: Any,
) -> None:
    """What the proxy's branch pushes actually publish: dev / staging / production."""
    for env_var in ("CYBERWAVE_ENVIRONMENT", "ENVIRONMENT"):
        for value in ("dev", "staging", "production"):
            monkeypatch.setenv(env_var, value)
            assert ds._resolve_sim_proxy_tag() == (value, "CYBERWAVE_ENVIRONMENT")
        monkeypatch.delenv(env_var, raising=False)


def test_an_unpublished_environment_falls_back_to_dev_not_latest() -> None:
    """`local` is a real value on a developer box and names no published tag;
    `latest` is never published for this image, so it is not the fallback."""
    assert ds._resolve_sim_proxy_tag() == ("dev", "default")


def test_the_env_var_pins_the_proxy_for_a_test_build(monkeypatch: Any) -> None:
    """How CI pins the whole graph to one head: the driver channel and the proxy
    tag come from two different workflows, so they are pinned separately."""
    monkeypatch.setenv("CYBERWAVE_ENVIRONMENT", "production")
    monkeypatch.setenv(ds.SIM_PROXY_TAG_ENV, "pr-3749-7f29d2c1c975")
    assert ds._resolve_sim_proxy_tag() == (
        "pr-3749-7f29d2c1c975",
        ds.SIM_PROXY_TAG_ENV,
    )


def test_an_explicit_argument_outranks_the_env_var(monkeypatch: Any) -> None:
    monkeypatch.setenv(ds.SIM_PROXY_TAG_ENV, "from-env")
    assert ds._resolve_sim_proxy_tag("explicit") == ("explicit", "argument")


def test_the_proxy_tag_is_always_substituted(monkeypatch: Any) -> None:
    """Always present rather than conditional on a twin uuid: an unrendered
    `${CW_SIM_PROXY_TAG}` would reach docker as a literal tag, the same quiet
    failure as the `${CW_TWIN_NS}` namespace this map already guards."""
    assert "CW_SIM_PROXY_TAG" in ds.compose_substitutions(channel_tag="humble")
    assert "CW_SIM_PROXY_TAG" in ds.compose_substitutions(
        channel_tag="humble", twin_uuid="abcd1234-0000-0000-0000-000000000000"
    )


# ── twin shared_env on the image-declared path ────────────────────────────────
#
# `drivers.<profile>.shared_env` was honoured by `_get_driver_services` (the
# inline-`services` form) and silently dropped by the image-declared form, whose
# translator returns `(specs, {}, [])`. A twin migrated from one to the other
# lost its values with no error: the Go2's `WAIT_FOR_START_MAPPING: "false"` went
# missing, the sim variant's `${WAIT_FOR_START_MAPPING:-true}` default took over,
# and SLAM sat behind a start gate nothing opened.


def test_shared_env_comes_from_the_matched_profile(monkeypatch: Any) -> None:
    """The SAME profile the image came from, so a cloud node cannot read the
    Jetson variant's values while running the default variant's image."""
    monkeypatch.setattr(ds, "_platform_driver_keys", lambda: ["linux-aarch64-jetson"])
    drivers = {
        "default": {"docker_image": "drv:x86", "shared_env": {"WHO": "default"}},
        "linux-aarch64-jetson": {
            "docker_image": "drv:jetson",
            "shared_env": {"WHO": "jetson"},
        },
    }
    assert ds.select_driver_shared_env(drivers) == {"WHO": "jetson"}
    assert ds.select_driver_image(drivers) == "drv:jetson"


def test_shared_env_falls_back_to_default_like_the_image_does(monkeypatch: Any) -> None:
    monkeypatch.setattr(ds, "_platform_driver_keys", lambda: ["linux-x86_64", "linux"])
    drivers = {"default": {"docker_image": "drv:x86", "shared_env": {"A": "1"}}}
    assert ds.select_driver_shared_env(drivers) == {"A": "1"}


def test_values_are_stringified_for_the_shell_that_expands_them(monkeypatch: Any) -> None:
    """Compose placeholders are expanded by a shell, where everything is a
    string. A JSON bool must not reach `--env` as Python's `True`."""
    monkeypatch.setattr(ds, "_platform_driver_keys", lambda: ["linux"])
    drivers = {
        "default": {
            "docker_image": "drv:x86",
            "shared_env": {"B": False, "T": True, "N": 15, "F": 1.5, "S": "x"},
        }
    }
    assert ds.select_driver_shared_env(drivers) == {
        "B": "false",
        "T": "true",
        "N": "15",
        "F": "1.5",
        "S": "x",
    }


def test_a_none_value_is_dropped_rather_than_sent_as_the_string_none(
    monkeypatch: Any,
) -> None:
    """`--env NAME=None` is worse than an unset variable: it defeats the
    `${NAME:-default}` fallback the declaration relies on."""
    monkeypatch.setattr(ds, "_platform_driver_keys", lambda: ["linux"])
    drivers = {"default": {"docker_image": "d:t", "shared_env": {"KEEP": "1", "DROP": None}}}
    assert ds.select_driver_shared_env(drivers) == {"KEEP": "1"}


@pytest.mark.parametrize(
    "drivers",
    [
        {},
        None,
        "not-a-dict",
        {"default": {"docker_image": "d:t"}},
        {"default": {"docker_image": "d:t", "shared_env": "not-a-dict"}},
        {"default": {"docker_image": "d:t", "shared_env": []}},
        {"default": "not-a-dict"},
    ],
)
def test_nothing_to_say_is_an_empty_dict_never_an_error(drivers: Any) -> None:
    """Every one of these means "the twin declares no shared env", which is the
    state every twin was in before this existed -- not a reason to fail a
    workload mid-start."""
    assert ds.select_driver_shared_env(drivers) == {}
