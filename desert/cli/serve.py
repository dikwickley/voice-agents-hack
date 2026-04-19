"""Start FastAPI + libp2p orchestrator in a background thread (for CLI TUI)."""

from __future__ import annotations

import threading
import time

import httpx
import uvicorn


def wait_for_health(port: int, timeout_sec: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_sec
    url = f"http://127.0.0.1:{port}/health"
    last_err: str | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = str(e)
        time.sleep(0.15)
    raise RuntimeError(f"orchestrator did not become healthy at {url}: {last_err}")


def start_orchestrator_api(host: str, port: int, *, log_level: str = "warning") -> threading.Thread:
    # `log_config=None` keeps uvicorn from replacing the root logging config
    # we set up in cli.main._silence_tui_loggers, so every emitter (including
    # httpx used by the TUI) keeps going to the file handler instead of stdout.
    config = uvicorn.Config(
        "backend.orchestrator.main:app",
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
        log_config=None,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()
    wait_for_health(port)
    return thread
