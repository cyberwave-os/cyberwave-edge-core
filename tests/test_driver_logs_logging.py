import io
import logging

from cyberwave_edge_core import driver_logs


class _FakeProcess:
    def __init__(self, output: str):
        self.stdout = io.StringIO(output)

    def wait(self, timeout: int | None = None) -> int:
        return 0

    def kill(self) -> None:
        return None


def test_follow_container_logs_does_not_emit_per_line_debug_noise(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(driver_logs.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        driver_logs.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess("hello from container\n"),
    )

    caplog.set_level(logging.DEBUG, logger="cyberwave_edge_core.driver_logs")

    driver_logs._follow_container_logs("cyberwave-driver-test")

    assert not any(
        "Container log line received" in record.getMessage() for record in caplog.records
    )
