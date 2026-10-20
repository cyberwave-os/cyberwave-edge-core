from __future__ import annotations

import sys
import types

import pytest

from cyberwave_edge_core import startup

URL = "http://backend/api/v1/twins/x"


@pytest.fixture
def rest(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """The generated REST module, or a stand-in where the SDK was installed without it (CI)."""
    try:
        from cyberwave.rest import rest as generated
    except ImportError:
        pass
    else:
        return generated

    def request(  # type: ignore[no-untyped-def]
        self, method, url, headers=None, body=None, post_params=None, _request_timeout=None
    ):
        raise AssertionError("replaced by each test")

    module = types.ModuleType("cyberwave.rest.rest")
    module.RESTClientObject = type("RESTClientObject", (), {"request": request})  # type: ignore[attr-defined]
    package = types.ModuleType("cyberwave.rest")
    package.rest = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cyberwave.rest", package)
    monkeypatch.setitem(sys.modules, "cyberwave.rest.rest", module)
    return module


def test_requests_without_a_timeout_get_the_bound(
    monkeypatch: pytest.MonkeyPatch, rest: types.ModuleType
) -> None:
    seen: list[object] = []

    def _record(  # type: ignore[no-untyped-def]
        self,
        method,
        url,
        headers=None,
        body=None,
        post_params=None,
        _request_timeout=None,
        *,
        stream=False,
    ):
        seen.append((_request_timeout, stream))

    monkeypatch.setattr(rest.RESTClientObject, "request", _record)
    startup.bound_sdk_request_timeouts(10.0)
    startup.bound_sdk_request_timeouts(10.0)

    client = object.__new__(rest.RESTClientObject)
    client.request("GET", URL)
    client.request("GET", URL, _request_timeout=None)
    client.request("GET", URL, _request_timeout=3)
    client.request("GET", URL, None, None, None, (1, 2))
    client.request("GET", URL, None, None, None, None)
    client.request("GET", URL, stream=True)

    assert seen == [
        (10.0, False),
        (10.0, False),
        (3, False),
        ((1, 2), False),
        (10.0, False),
        (10.0, True),
    ]


def test_the_timeout_is_found_wherever_the_signature_puts_it(
    monkeypatch: pytest.MonkeyPatch, rest: types.ModuleType
) -> None:
    seen: list[object] = []

    def _record(  # type: ignore[no-untyped-def]
        self,
        method,
        url,
        headers=None,
        body=None,
        post_params=None,
        query=None,
        _request_timeout=None,
    ):
        seen.append((query, _request_timeout))

    monkeypatch.setattr(rest.RESTClientObject, "request", _record)
    startup.bound_sdk_request_timeouts(10.0)

    client = object.__new__(rest.RESTClientObject)
    client.request("GET", URL, None, None, None, {"q": 1})
    client.request("GET", URL, None, None, None, None, 4)

    assert seen == [({"q": 1}, 10.0), (None, 4)]
