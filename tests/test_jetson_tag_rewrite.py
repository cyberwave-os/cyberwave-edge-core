"""The Jetson tag rewrite only touches references that actually carry a tag."""

from __future__ import annotations

import pytest

import cyberwave_edge_core.driver_selection as driver_selection
import cyberwave_edge_core.startup as startup

DIGEST_IMAGE = "cyberwaveos/kinova-gen3-driver@sha256:" + "a" * 64


class TestJetsonTagRewrite:
    @pytest.fixture(autouse=True)
    def _on_jetson(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(driver_selection, "is_jetson", lambda: True)

    def test_digest_reference_is_kept(self) -> None:
        assert startup._maybe_rewrite_jetson_tag(DIGEST_IMAGE, "kinova") == DIGEST_IMAGE

    def test_tag_and_digest_reference_is_kept(self) -> None:
        image = "cyberwaveos/kinova-gen3-driver:humble@sha256:" + "a" * 64
        assert startup._maybe_rewrite_jetson_tag(image, "kinova") == image

    def test_registry_port_is_not_a_tag(self) -> None:
        image = "registry.local:5000/cyberwaveos/kinova-gen3-driver"
        assert startup._maybe_rewrite_jetson_tag(image, "kinova") == image

    def test_plain_tag_still_rewritten(self) -> None:
        assert (
            startup._maybe_rewrite_jetson_tag("cyberwaveos/go2-ros2-driver:humble", "go2")
            == "cyberwaveos/go2-ros2-driver:jetson-humble"
        )

    def test_tagged_image_on_registry_with_port_still_rewritten(self) -> None:
        assert (
            startup._maybe_rewrite_jetson_tag("registry.local:5000/cyberwaveos/go2:humble", "go2")
            == "registry.local:5000/cyberwaveos/go2:jetson-humble"
        )


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("cyberwaveos/go2", ("cyberwaveos/go2", None)),
        ("cyberwaveos/go2:humble", ("cyberwaveos/go2", "humble")),
        ("registry.local:5000/cyberwaveos/go2", ("registry.local:5000/cyberwaveos/go2", None)),
        (
            "registry.local:5000/cyberwaveos/go2:humble",
            ("registry.local:5000/cyberwaveos/go2", "humble"),
        ),
    ],
)
def test_split_image_tag(image: str, expected: tuple[str, str | None]) -> None:
    assert startup._split_image_tag(image) == expected


class TestResolveDriverImageTag:
    @pytest.fixture(autouse=True)
    def _dev_edge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CYBERWAVE_ENVIRONMENT", "dev")

    def test_untagged_image_gets_environment_tag(self) -> None:
        assert startup._resolve_driver_image_tag("cyberwaveos/go2") == "cyberwaveos/go2:dev"

    def test_registry_port_is_not_a_tag(self) -> None:
        image = "registry.local:5000/cyberwaveos/go2"
        assert startup._resolve_driver_image_tag(image) == f"{image}:dev"

    def test_tagged_image_is_kept(self) -> None:
        image = "registry.local:5000/cyberwaveos/go2:humble"
        assert startup._resolve_driver_image_tag(image) == image

    def test_digest_reference_is_kept(self) -> None:
        assert startup._resolve_driver_image_tag(DIGEST_IMAGE) == DIGEST_IMAGE
