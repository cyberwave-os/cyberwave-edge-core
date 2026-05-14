"""Cyberwave Edge Core CLI entry point."""

import logging
import os
import signal
import sys
import time

import click
from rich.console import Console

from . import __version__
from .startup import (
    CONFIG_DIR,
    DEFAULT_API_URL,
    check_mqtt_connection,
    get_runtime_env_var,
    load_environment_uuid,
    load_token,
    run_runtime_loop,
    run_startup_checks,
    shutdown_event,
    validate_token,
)

console = Console()
LOG_LEVEL_ENV_VAR = "CYBERWAVE_EDGE_LOG_LEVEL"


def _resolve_log_level() -> int:
    """Resolve logger level from env var with INFO fallback."""
    raw_level = os.getenv(LOG_LEVEL_ENV_VAR, "INFO").upper()
    return getattr(logging, raw_level, logging.INFO)


# Configure logging so info/warning/error messages appear in journald / log files.
# Include timestamps so macOS LaunchAgent log files are readable; on Linux
# journalctl adds its own but the ISO prefix is still harmless.
logging.basicConfig(
    level=_resolve_log_level(),
    format="%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)


def _sigterm_handler(signum: int, frame: object) -> None:
    logger.info("Received SIGTERM — initiating graceful shutdown")
    shutdown_event.set()


def _load_sdk_default_api():
    """Import and return the generated SDK DefaultApi type."""
    from cyberwave.rest import DefaultApi

    return DefaultApi


def run_sdk_selfcheck() -> int:
    """Verify the packaged runtime contains the generated REST SDK."""
    try:
        default_api = _load_sdk_default_api()
    except Exception as exc:
        click.echo(f"sdk-rest-missing: {exc}", err=True)
        return 1

    if default_api is None:
        click.echo("sdk-rest-missing: DefaultApi unavailable", err=True)
        return 1

    click.echo("sdk-rest-ok")
    return 0


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="cyberwave-edge-core")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Cyberwave Edge Core — orchestrator for edge components."""
    if ctx.invoked_subcommand is None:
        from .resource_monitor import SystemResourceMonitor
        from .startup import LOG_FOLLOWER_RECONCILE_INTERVAL_SECONDS
        from .watchdog import ProcessWatchdog, protect_edge_core_oom

        signal.signal(signal.SIGTERM, _sigterm_handler)

        protect_edge_core_oom()
        watchdog = ProcessWatchdog()
        resource_monitor = SystemResourceMonitor()

        # Prime the monitor so the bootstrap edge_health publisher started
        # inside run_startup_checks() can ship host pressure on its very
        # first heartbeat instead of "metric absent" for ~30 s.
        resource_monitor.check()

        if not run_startup_checks(resource_monitor=resource_monitor, watchdog=watchdog):
            sys.exit(1)

        # Two-phase watchdog handoff: send READY=1 *immediately* after the
        # blocking boot work (driver images pulled, MQTT connected, twin
        # sync reconciled) so systemd transitions the unit to ``active``
        # well within ``TimeoutStartSec``.  ``mark_ready`` is not
        # PID-restricted, so it works even when this process is a
        # PyInstaller --onefile child whose PID differs from systemd's
        # ``MainPID`` / ``$WATCHDOG_PID`` (the silent-drop bug that
        # caused the 0.1.4.1459 → 0.1.4.1463 start-timeout regression).
        # Periodic ``WATCHDOG=1`` pings, which *are* PID-restricted, are
        # then driven from the runtime loop in this same process.
        watchdog.mark_ready()
        watchdog.start_pinging(ping_interval_seconds=LOG_FOLLOWER_RECONCILE_INTERVAL_SECONDS)
        try:
            run_runtime_loop(
                watchdog=watchdog,
                resource_monitor=resource_monitor,
            )
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, shutting down edge-core")
            shutdown_event.set()
        finally:
            watchdog.stop()


@cli.command(name="__selfcheck_sdk", hidden=True)
def selfcheck_sdk() -> None:
    """Hidden packaged-runtime check for generated SDK imports."""
    raise click.exceptions.Exit(run_sdk_selfcheck())


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
    elapsed = time.perf_counter() - _t0
    if mqtt_ok:
        console.print(f"  [green]✓[/green] MQTT broker [dim]({elapsed:.3f}s)[/dim]")
    else:
        console.print(f"  [red]✗[/red] MQTT broker [dim]({elapsed:.3f}s)[/dim]")

    console.print()


# ---------------------------------------------------------------------------
# Worker subcommand group
# ---------------------------------------------------------------------------


@cli.group()
def worker() -> None:
    """Manage the edge worker container."""


def _get_worker_manager() -> "WorkerManager":  # type: ignore[name-defined]  # noqa: F821
    """Build a WorkerManager from the current edge configuration."""
    from .startup import _list_linked_twin_uuids_for_fingerprint, get_or_create_fingerprint
    from .worker_manager import WorkerManager, resolve_worker_image

    token = load_token()
    if not token:
        console.print("[red]No credentials found. Run 'cyberwave login' first.[/red]")
        sys.exit(1)

    environment_uuid = load_environment_uuid()
    if not environment_uuid:
        console.print(
            "[red]No linked environment found. "
            "Run 'cyberwave link' to associate this edge with an environment.[/red]"
        )
        sys.exit(1)

    twin_uuids: list[str] = []
    if environment_uuid:
        try:
            fingerprint = get_or_create_fingerprint()
            if fingerprint:
                twin_uuids = _list_linked_twin_uuids_for_fingerprint(
                    token, environment_uuid, fingerprint
                )
        except Exception:
            logger.debug("Failed to resolve twin UUIDs for environment", exc_info=True)

    return WorkerManager(
        config_dir=CONFIG_DIR,
        environment_uuid=environment_uuid,
        token=token,
        twin_uuids=twin_uuids,
        image=resolve_worker_image(),
    )


@worker.command(name="start")
def worker_start() -> None:
    """Start the worker container (ensures models are cached first)."""
    wm = _get_worker_manager()
    ok = wm.start()
    if ok:
        console.print(
            f"[green]✓[/green] Worker container [bold]{wm._container_name}[/bold] started"
        )
    else:
        console.print(
            f"[red]✗[/red] Failed to start worker container [bold]{wm._container_name}[/bold]"
        )
        sys.exit(1)


@worker.command(name="stop")
def worker_stop() -> None:
    """Stop the worker container."""
    wm = _get_worker_manager()
    ok = wm.stop()
    if ok:
        console.print(
            f"[green]✓[/green] Worker container [bold]{wm._container_name}[/bold] stopped"
        )
    else:
        console.print("[red]✗[/red] Failed to stop worker container")
        sys.exit(1)


@worker.command(name="restart")
def worker_restart() -> None:
    """Restart the worker container (re-scans workers, re-ensures models)."""
    wm = _get_worker_manager()
    ok = wm.restart()
    if ok:
        console.print(
            f"[green]✓[/green] Worker container [bold]{wm._container_name}[/bold] restarted"
        )
    else:
        console.print("[red]✗[/red] Failed to restart worker container")
        sys.exit(1)


@worker.command(name="status")
def worker_status() -> None:
    """Show worker container state, loaded workers, cached models, and health."""
    from .model_manager import ModelManager
    from .worker_health import WorkerHealthMonitor

    wm = _get_worker_manager()
    # Attach a health monitor so status() includes restart / health data.
    health_monitor = WorkerHealthMonitor(container_name=wm._container_name)
    wm.set_health_monitor(health_monitor)
    ws = wm.status()

    status_color = (
        "green"
        if ws.status == "running"
        else ("yellow" if ws.status in {"restarting", "created"} else "red")
    )
    cb_suffix = (
        " [bold red][circuit-breaker tripped][/bold red]" if ws.circuit_breaker_tripped else ""
    )
    console.print(
        f"\n[bold]Worker container:[/bold] {ws.container_name} "
        f"([{status_color}]{ws.status}[/{status_color}]){cb_suffix}\n"
    )

    if ws.worker_files:
        console.print("[bold]  Workers:[/bold]")
        for wf in ws.worker_files:
            console.print(f"    {wf}")
    else:
        console.print("  [dim]No worker files found in workers directory[/dim]")

    models_dir = wm._models_dir()
    mm = ModelManager(
        cache_dir=models_dir,
        api_token=load_token() or "",
        base_url=get_runtime_env_var("CYBERWAVE_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL,
    )
    cached = mm.list_cached_models()
    if cached:
        console.print("\n[bold]  Models:[/bold]")
        for model in cached:
            console.print(f"    {model.model_id:<30} {model.size_mb():.1f} MB")
    else:
        console.print("\n  [dim]No cached models[/dim]")

    gpu_label = "[green]enabled[/green]" if ws.gpu_enabled else "[dim]not detected[/dim]"
    console.print(f"\n  GPU: {gpu_label}")
    console.print(f"  Workers dir: {ws.workers_dir}")
    console.print(f"  Models dir:  {ws.models_dir}")

    # Health / restart summary.
    console.print("\n[bold]  Health:[/bold]")
    console.print(f"    Total restarts:  {ws.restart_count}")
    console.print(f"    Recent restarts: {ws.recent_restarts} (5-min window)")
    if ws.circuit_breaker_tripped:
        console.print("    [red]Circuit-breaker: TRIPPED — automatic restarts suppressed[/red]")
    else:
        console.print("    Circuit-breaker: [green]closed[/green]")
    if ws.health_state and ws.health_state.uptime_seconds is not None:
        uptime = ws.health_state.uptime_seconds
        console.print(f"    Uptime:          {uptime:.0f}s")
    console.print()


@worker.command(name="health")
def worker_health() -> None:
    """Show detailed worker health: restart history and circuit-breaker state."""
    from .worker_health import WorkerHealthMonitor

    wm = _get_worker_manager()
    health_monitor = WorkerHealthMonitor(container_name=wm._container_name)
    wm.set_health_monitor(health_monitor)
    ws = wm.status()
    hs = ws.health_state

    console.print(f"\n[bold]Worker Health — {wm._container_name}[/bold]\n")

    status_color = (
        "green"
        if ws.status == "running"
        else ("yellow" if ws.status in {"restarting", "created"} else "red")
    )
    console.print(f"  Container status: [{status_color}]{ws.status}[/{status_color}]")

    if hs is not None:
        healthy_label = "[green]healthy[/green]" if hs.is_healthy else "[red]unhealthy[/red]"
        ready_label = "[green]ready[/green]" if hs.is_ready else "[yellow]not ready[/yellow]"
        console.print(f"  Health:           {healthy_label}")
        console.print(f"  Readiness:        {ready_label}")

        if hs.uptime_seconds is not None:
            console.print(f"  Uptime:           {hs.uptime_seconds:.0f}s")

        console.print("\n  [bold]Restart accounting:[/bold]")
        console.print(f"    Total:    {hs.restart_count}")
        console.print(f"    Recent:   {hs.recent_restarts} (5-min window)")

        if hs.circuit_breaker_tripped:
            import datetime

            tripped_ts = (
                datetime.datetime.fromtimestamp(hs.circuit_breaker_tripped_at).isoformat()
                if hs.circuit_breaker_tripped_at
                else "unknown"
            )
            console.print(f"\n  [bold red]Circuit-breaker: TRIPPED[/bold red] at {tripped_ts}")
            console.print("  Automatic restarts are suppressed until the 5-minute window clears.")
        else:
            console.print("\n  Circuit-breaker: [green]closed[/green]")

        if hs.restart_records:
            console.print(f"\n  [bold]Restart history ({len(hs.restart_records)} events):[/bold]")
            import datetime

            for rec in hs.restart_records[-10:]:
                ts = datetime.datetime.fromtimestamp(rec.timestamp).strftime("%H:%M:%S")
                ok_label = "[green]ok[/green]" if rec.success else "[red]failed[/red]"
                console.print(f"    {ts}  {rec.reason:<30} {ok_label}")
        else:
            console.print("\n  [dim]No restarts recorded in this session[/dim]")
    else:
        console.print("\n  [dim]Health monitor not available[/dim]")

    console.print()


@worker.command(name="logs")
@click.option("--no-follow", is_flag=True, default=False, help="Print logs without following.")
def worker_logs(no_follow: bool) -> None:
    """Stream worker container logs."""
    wm = _get_worker_manager()
    wm.logs(follow=not no_follow)


def main() -> None:
    """Entry point for PyInstaller binary."""
    cli()


if __name__ == "__main__":
    main()
