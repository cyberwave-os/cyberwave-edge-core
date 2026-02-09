"""Cyberwave Edge Core CLI entry point."""

import sys

import click
from rich.console import Console

from .startup import (
    check_mqtt_connection,
    load_devices,
    load_token,
    run_startup_checks,
    validate_token,
)

console = Console()


@click.group(invoke_without_command=True)
@click.version_option(package_name="cyberwave-edge-core")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Cyberwave Edge Core — orchestrator for edge components."""
    if ctx.invoked_subcommand is None:
        # Boot path: run all startup checks
        if not run_startup_checks():
            sys.exit(1)


@cli.command()
def status() -> None:
    """Show current credential, token, and MQTT status."""
    console.print("\n[bold]Cyberwave Edge Core — status[/bold]\n")

    token = load_token()
    if not token:
        console.print("  Credentials: [red]not found[/red]")
        console.print("  Token:       [dim]—[/dim]")
        console.print("  MQTT:        [dim]—[/dim]")
        console.print()
        return

    console.print("  Credentials: [green]found[/green]")

    if validate_token(token):
        console.print("  Token:       [green]valid[/green]")
    else:
        console.print("  Token:       [red]invalid / unreachable[/red]")
        console.print("  MQTT:        [dim]—[/dim]")
        console.print()
        return

    if check_mqtt_connection(token):
        console.print("  MQTT:        [green]connected[/green]")
    else:
        console.print("  MQTT:        [red]unreachable[/red]")

    devices = load_devices()
    if devices:
        console.print(f"  Devices:     [green]{len(devices)} configured[/green]")
        for dev in devices:
            console.print(f"               {dev.name} [dim]({dev.type})[/dim] @ {dev.port}")
    else:
        console.print("  Devices:     [yellow]none[/yellow]")

    console.print()


def main() -> None:
    """Entry point for PyInstaller binary."""
    cli()


if __name__ == "__main__":
    main()
