"""desert CLI: orchestrator (Textual + embedded API) or worker (swarm peer with logs)."""

from __future__ import annotations

import logging
import os
import pathlib
import sys

import typer
from rich.logging import RichHandler

from cli.orchestrator_app import run_tui
from cli.serve import start_orchestrator_api

app = typer.Typer(no_args_is_help=True, help="Desert distributed agents — orchestrator TUI or worker.")


def _silence_tui_loggers(log_file: pathlib.Path) -> None:
    """Keep the Textual screen clean: route all stdlib logging to a file and
    mute libraries that emit at INFO (httpx per-request, uvicorn lifecycle,
    libp2p handshake). Without this, their lines scribble over the TUI."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(fh)
    root.setLevel(logging.WARNING)
    for name in (
        "httpx",
        "httpcore",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "libp2p",
        "multiaddr",
        "asyncio",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.propagate = True


@app.command("orchestrator")
def orchestrator_cmd(
    http_host: str = typer.Option(
        "0.0.0.0",
        "--http-host",
        help="Bind address for FastAPI (use 0.0.0.0 in Docker).",
    ),
    http_port: int = typer.Option(8000, "--http-port", help="HTTP API port."),
    connect_url: str | None = typer.Option(
        None,
        "--connect-url",
        help="URL the TUI polls (default: http://127.0.0.1:<http-port>).",
    ),
    uvicorn_log: str = typer.Option("warning", "--uvicorn-log", help="Uvicorn log level."),
    log_file: pathlib.Path = typer.Option(
        pathlib.Path(os.environ.get("DESERT_LOG_FILE") or "/tmp/desert-orchestrator.log"),
        "--log-file",
        help="Where backend logs are written (never printed over the TUI).",
    ),
) -> None:
    """Run FastAPI + libp2p orchestrator and open the full-screen TUI."""
    os.environ.setdefault("ORCH_HOST", http_host)
    os.environ.setdefault("ORCH_PORT", str(http_port))
    _silence_tui_loggers(log_file)
    start_orchestrator_api(http_host, http_port, log_level=uvicorn_log)
    # Defensive: Textual owns stdout for the alt screen; ensure no stray print
    # from libs hits the TUI. Stderr still goes to the terminal on crash.
    sys.stdout.flush()
    ui_url = connect_url or f"http://127.0.0.1:{http_port}"
    run_tui(ui_url)


@app.command(
    "worker",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def worker_cmd(ctx: typer.Context) -> None:
    """Run a swarm worker peer. Streams Rich-formatted logs to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    for noisy in ("httpx", "httpcore", "libp2p", "multiaddr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from worker.worker import run as worker_run

    extra = list(ctx.args)
    worker_run(argv=extra if extra else None)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
