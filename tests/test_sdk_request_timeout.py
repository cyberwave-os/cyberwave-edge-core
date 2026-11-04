from __future__ import annotations

import socket
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


@pytest.fixture(autouse=True)
def _restore_sdk_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """bound_sdk_request_timeouts replaces the SDK's default; undo it after each test."""
    from cyberwave import client as sdk_client

    if hasattr(sdk_client, "_default_request_timeout"):
        monkeypatch.setattr(
            sdk_client, "_default_request_timeout", sdk_client._default_request_timeout
        )


def test_the_sdk_default_timeout_becomes_the_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """An SDK that fills its own 300 s default would otherwise preempt the bound."""
    from cyberwave import client as sdk_client

    monkeypatch.setattr(
        sdk_client,
        "_default_request_timeout",
        lambda timeout: (10.0, float(timeout)),
        raising=False,
    )

    startup.bound_sdk_request_timeouts(10.0)

    assert sdk_client._default_request_timeout(300) == (4.0, 10.0)


def test_the_bound_default_accepts_the_sdk_body_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK passes ``has_body``; a one-argument override failed every SDK call.

    A call with a body keeps the connect bound: without one, a PUT on a silent
    uplink blocks for the kernel's SYN retries, past the watchdog.
    """
    from cyberwave import client as sdk_client

    monkeypatch.setattr(
        sdk_client,
        "_default_request_timeout",
        lambda timeout, *, has_body=False: (10.0, float(timeout)),
        raising=False,
    )

    startup.bound_sdk_request_timeouts(10.0)

    assert sdk_client._default_request_timeout(300, has_body=False) == (4.0, 10.0)
    assert sdk_client._default_request_timeout(300, has_body=True) == (4.0, 10.0)


def test_a_call_through_the_sdk_client_gets_the_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through Cyberwave.call_api, on an SDK that sets its own default."""
    import inspect
    from unittest.mock import MagicMock

    from cyberwave import Cyberwave
    from cyberwave import client as sdk_client

    if not hasattr(sdk_client, "_default_request_timeout"):
        pytest.skip("installed SDK sets no default timeout")
    if not hasattr(pytest.importorskip("cyberwave.rest"), "DefaultApi"):
        pytest.skip("installed SDK has no generated REST client")
    monkeypatch.delenv("CYBERWAVE_API_TIMEOUT", raising=False)
    startup.bound_sdk_request_timeouts()
    client = Cyberwave(base_url="http://localhost:8000", api_key="k")
    request = MagicMock(return_value=MagicMock())
    client._api_client.rest_client.request = request
    serialize = client.api._src_app_api_assets_list_assets_serialize
    kwargs: dict = {
        "_request_auth": None,
        "_content_type": None,
        "_headers": None,
        "_host_index": 0,
    }
    for name, param in inspect.signature(serialize).parameters.items():
        if name not in kwargs and param.default is inspect.Parameter.empty:
            kwargs[name] = None

    client._api_client.call_api(*serialize(**kwargs))

    assert request.call_args.kwargs["_request_timeout"] == (4.0, 10.0)


def test_a_call_with_a_body_through_the_sdk_client_gets_a_connect_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A twin update or alert POST must not connect without a bound on a dead uplink."""
    from unittest.mock import MagicMock

    from cyberwave import Cyberwave
    from cyberwave import client as sdk_client

    if not hasattr(sdk_client, "_default_request_timeout"):
        pytest.skip("installed SDK sets no default timeout")
    if not hasattr(pytest.importorskip("cyberwave.rest"), "DefaultApi"):
        pytest.skip("installed SDK has no generated REST client")
    monkeypatch.delenv("CYBERWAVE_API_TIMEOUT", raising=False)
    startup.bound_sdk_request_timeouts()
    client = Cyberwave(base_url="http://localhost:8000", api_key="k")
    request = MagicMock(return_value=MagicMock())
    client._api_client.rest_client.request = request

    client._api_client.call_api("PUT", "http://localhost:8000/api/v1/twins/t", {}, {"metadata": {}})

    assert request.call_args.kwargs["_request_timeout"] == (4.0, 10.0)


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

    bound = (startup.SDK_CONNECT_TIMEOUT_SECONDS, 10.0)
    assert seen == [
        (bound, False),
        (bound, False),
        (3, False),
        ((1, 2), False),
        (bound, False),
        (bound, True),
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

    assert seen == [({"q": 1}, (startup.SDK_CONNECT_TIMEOUT_SECONDS, 10.0)), (None, 4)]


def test_a_short_read_bound_also_caps_the_connect(
    monkeypatch: pytest.MonkeyPatch, rest: types.ModuleType
) -> None:
    seen: list[object] = []

    def _record(  # type: ignore[no-untyped-def]
        self, method, url, headers=None, body=None, post_params=None, _request_timeout=None
    ):
        seen.append(_request_timeout)

    monkeypatch.setattr(rest.RESTClientObject, "request", _record)
    startup.bound_sdk_request_timeouts(2.0)

    object.__new__(rest.RESTClientObject).request("GET", URL)

    assert seen == [(2.0, 2.0)]


def _ignore(  # type: ignore[no-untyped-def]
    self, method, url, headers=None, body=None, post_params=None, _request_timeout=None
):
    return None


class _PoolManager:
    def __init__(self, **pool_kw: object) -> None:
        self.connection_pool_kw = dict(pool_kw)
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1


def test_the_pool_retries_only_a_failed_connect_once(
    monkeypatch: pytest.MonkeyPatch, rest: types.ModuleType
) -> None:
    monkeypatch.setattr(rest.RESTClientObject, "request", _ignore)
    startup.bound_sdk_request_timeouts()

    client = object.__new__(rest.RESTClientObject)
    client.pool_manager = _PoolManager()
    client.request("GET", URL)
    client.request("GET", URL)

    retries = client.pool_manager.connection_pool_kw["retries"]
    assert (retries.connect, retries.read, retries.status, retries.other) == (1, 1, 0, 0)
    assert retries.respect_retry_after_header is False
    # Built once per client, and the pools already open are dropped so they pick it up.
    assert client.pool_manager.cleared == 1


def test_a_retries_the_sdk_configured_is_kept(
    monkeypatch: pytest.MonkeyPatch, rest: types.ModuleType
) -> None:
    monkeypatch.setattr(rest.RESTClientObject, "request", _ignore)
    startup.bound_sdk_request_timeouts()

    client = object.__new__(rest.RESTClientObject)
    client.pool_manager = _PoolManager(retries=7)
    client.request("GET", URL)

    assert client.pool_manager.connection_pool_kw["retries"] == 7
    assert client.pool_manager.cleared == 0


def test_a_failed_dns_lookup_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolver already took its own timeout (10 s on the Jetson); a second lookup fails the same way."""
    import socket

    import urllib3
    from urllib3.util import connection

    pytest.importorskip("urllib3.exceptions").NameResolutionError  # noqa: B018 - urllib3 >= 2
    lookups: list[str] = []

    def _no_dns(host, *args, **kwargs):  # type: ignore[no-untyped-def]
        lookups.append(host)
        raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")

    monkeypatch.setattr(connection.socket, "getaddrinfo", _no_dns)
    pool = urllib3.PoolManager(retries=startup._bounded_sdk_retries())

    with pytest.raises(urllib3.exceptions.MaxRetryError):
        pool.request("GET", "http://backend.invalid/api/v1/twins/x")

    assert lookups == ["backend.invalid"]


def test_a_failed_connect_is_still_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib3
    from urllib3.util import connection

    attempts: list[object] = []

    def _refused(address, *args, **kwargs):  # type: ignore[no-untyped-def]
        attempts.append(address)
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(connection, "create_connection", _refused)
    pool = urllib3.PoolManager(retries=startup._bounded_sdk_retries())

    with pytest.raises(urllib3.exceptions.MaxRetryError):
        pool.request("GET", "http://backend.invalid/api/v1/twins/x")

    assert len(attempts) == 2


def _serve_dropping_first(count: int) -> tuple[socket.socket, list[bytes]]:
    """A local HTTP server that closes the first *count* connections after reading the request."""
    import threading

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    requests: list[bytes] = []

    def _serve() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                requests.append(conn.recv(65536))
                if len(requests) > count:
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
                    )

    threading.Thread(target=_serve, daemon=True).start()
    return server, requests


def test_a_connection_dropped_mid_get_is_retried_once() -> None:
    """A keep-alive connection the peer or a NAT closed is not an outage: no breaker for it."""
    import urllib3

    server, requests = _serve_dropping_first(1)
    try:
        pool = urllib3.PoolManager(retries=startup._bounded_sdk_retries())
        response = pool.request("GET", f"http://127.0.0.1:{server.getsockname()[1]}/api/v1/twins/x")
    finally:
        server.close()

    assert response.status == 200
    assert len(requests) == 2


def test_a_post_whose_connection_dropped_is_not_sent_twice() -> None:
    import urllib3

    server, requests = _serve_dropping_first(1)
    try:
        pool = urllib3.PoolManager(retries=startup._bounded_sdk_retries())
        with pytest.raises(urllib3.exceptions.ProtocolError):
            port = server.getsockname()[1]
            pool.request("POST", f"http://127.0.0.1:{port}/api/v1/twins", body=b"{}")
    finally:
        server.close()

    assert len(requests) == 1


def test_a_silent_server_is_not_waited_for_twice() -> None:
    import threading

    import urllib3

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    accepted: list[socket.socket] = []

    def _accept_and_stay_silent() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            accepted.append(conn)

    threading.Thread(target=_accept_and_stay_silent, daemon=True).start()
    try:
        pool = urllib3.PoolManager(retries=startup._bounded_sdk_retries(), timeout=0.2)
        with pytest.raises(urllib3.exceptions.MaxRetryError) as failed:
            pool.request("GET", f"http://127.0.0.1:{server.getsockname()[1]}/api/v1/twins/x")
    finally:
        server.close()
        for conn in accepted:
            conn.close()

    assert isinstance(failed.value.reason, urllib3.exceptions.ReadTimeoutError)
    assert len(accepted) == 1


def test_any_answer_closes_the_breaker(
    monkeypatch: pytest.MonkeyPatch, rest: types.ModuleType
) -> None:
    """Not only at the two call sites that close it: an edge with no twin files has neither."""
    status = {"value": 404}

    def _answer(  # type: ignore[no-untyped-def]
        self, method, url, headers=None, body=None, post_params=None, _request_timeout=None
    ):
        return types.SimpleNamespace(status=status["value"])

    monkeypatch.setattr(rest.RESTClientObject, "request", _answer)
    startup.bound_sdk_request_timeouts()
    client = object.__new__(rest.RESTClientObject)
    client.pool_manager = _PoolManager()
    startup.note_cloud_failure(ConnectionError("down"))
    # The pause ran out; the next call reaches the backend.
    monkeypatch.setattr(startup, "_cloud_paused_until", 0.0)

    status["value"] = 503
    client.request("GET", URL)
    assert startup._cloud_backoff_seconds

    status["value"] = 404
    client.request("GET", URL)
    assert not startup._cloud_backoff_seconds


def test_requests_fail_at_once_while_the_backend_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, rest: types.ModuleType
) -> None:
    """Every caller, not only the runtime loop's cloud steps, which check the breaker themselves."""
    sent: list[str] = []

    def _send(self, method, url, headers=None, body=None, post_params=None, _request_timeout=None):  # type: ignore[no-untyped-def]
        sent.append(url)

    monkeypatch.setattr(rest.RESTClientObject, "request", _send)
    startup.bound_sdk_request_timeouts()
    client = object.__new__(rest.RESTClientObject)
    startup.note_cloud_failure(ConnectionError("Temporary failure in name resolution"))
    paused_until = startup._cloud_paused_until

    with pytest.raises(ConnectionError) as failed:
        client.request("GET", URL)

    assert sent == []
    # Counted as in flight: the backoff is not extended by the calls it stops.
    assert startup.note_cloud_failure(failed.value) is True
    assert startup._cloud_paused_until == paused_until

    startup.note_cloud_success()
    client.request("GET", URL)
    assert sent == [URL]


def test_a_backend_that_answered_with_an_error_does_not_stop_requests(
    monkeypatch: pytest.MonkeyPatch, rest: types.ModuleType
) -> None:
    """A 503 costs no timeout, so only the loop's cloud steps sit out the pause."""
    sent: list[str] = []

    def _send(self, method, url, headers=None, body=None, post_params=None, _request_timeout=None):  # type: ignore[no-untyped-def]
        sent.append(url)

    monkeypatch.setattr(rest.RESTClientObject, "request", _send)
    startup.bound_sdk_request_timeouts()
    client = object.__new__(rest.RESTClientObject)

    class Unavailable(Exception):
        status_code = 503

    assert startup.note_cloud_failure(Unavailable()) is True
    assert startup.cloud_reconciliation_paused()

    client.request("GET", URL)
    assert sent == [URL]

    # Until the backend also stops being reachable.
    startup.note_cloud_failure(ConnectionError("Temporary failure in name resolution"))
    with pytest.raises(ConnectionError):
        client.request("GET", URL)
