"""Cyberwave Edge Core CLI entry point."""

import logging
import os
import sys
import time

import click
from rich.console import Console

from .startup import (
    check_mqtt_connection,
    load_token,
    run_runtime_loop,
    run_startup_checks,
    validate_token,
)

console = Console()
LOG_LEVEL_ENV_VAR = "CYBERWAVE_EDGE_LOG_LEVEL"


def _resolve_log_level() -> int:
    """Resolve logger level from env var with INFO fallback."""
    raw_level = os.getenv(LOG_LEVEL_ENV_VAR, "INFO").upper()
    return getattr(logging, raw_level, logging.INFO)


# Configure logging so info/warning/error messages appear in journald.
# The systemd journal captures stderr; use a clear format so log lines
# are easy to filter with journalctl.
logging.basicConfig(
    level=_resolve_log_level(),
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)

# In a PyInstaller frozen binary, importlib.metadata cannot find packages by
# name because they are not "installed" in the traditional sense.
# Resolve the version at import time with a safe fallback so --version never
# raises RuntimeError in a frozen environment.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("cyberwave-edge-core")
except Exception:
    _VERSION = "unknown"


@click.group(invoke_without_command=True)
@click.version_option(version=_VERSION, prog_name="cyberwave-edge-core")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Cyberwave Edge Core — orchestrator for edge components."""
    if ctx.invoked_subcommand is None:
        # Boot path: run all startup checks
        if not run_startup_checks():
            sys.exit(1)
        try:
            run_runtime_loop()
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Received stop signal, shutting down edge-core")


@cli.command()
def status() -> None:
    """Show current credential, token, and MQTT status."""
    console.print("\n[bold]Cyberwave Edge Core — Status[/bold]\n")

    _t0 = time.perf_counter()
    token = load_token()
    if not token:
        console.print(f"  [red]✗[/red] Credentials [dim]({(time.perf_counter() - _t0):.3f}s)[/dim]")
        console.print("  [dim]—[/dim] Token")
        console.print("  [dim]—[/dim] MQTT broker")
        console.print()
        return

    console.print(f"  [green]✓[/green] Credentials [dim]({(time.perf_counter() - _t0):.3f}s)[/dim]")

    _t0 = time.perf_counter()
    token_ok = validate_token(token)
    if token_ok:
        console.print(f"  [green]✓[/green] Token [dim]({(time.perf_counter() - _t0):.3f}s)[/dim]")
    else:
        console.print(f"  [red]✗[/red] Token [dim]({(time.perf_counter() - _t0):.3f}s)[/dim]")
        console.print("  [dim]—[/dim] MQTT broker")
        console.print()
        return

    _t0 = time.perf_counter()
    mqtt_ok = check_mqtt_connection(token)
    if mqtt_ok:
        console.print(f"  [green]✓[/green] MQTT broker [dim]({(time.perf_counter() - _t0):.3f}s)[/dim]")
    else:
        console.print(f"  [red]✗[/red] MQTT broker [dim]({(time.perf_counter() - _t0):.3f}s)[/dim]")

    console.print()


def main() -> None:
    """Entry point for PyInstaller binary."""
    cli()


if __name__ == "__main__":
    main()
