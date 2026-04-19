"""Desert bootstrap: HTTP API + libp2p peer on the same GossipSub topic as desert."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI, HTTPException

from bootstrap_server.p2p_thread import get_bootstrap_swarm_node, start_bootstrap_swarm_thread
from p2p.bootstrap_http_addrs import multiaddrs_for_bootstrap_get_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_bootstrap_swarm_thread()
    yield


app = FastAPI(title="Desert Bootstrap", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    node = get_bootstrap_swarm_node()
    return {
        "ok": True,
        "p2p_ready": bool(node and node.host),
        "peer_id": (node.peer_id if node and node.host else None),
    }


@app.get("/v1/bootstrap")
async def bootstrap() -> dict[str, Any]:
    node = get_bootstrap_swarm_node()
    if node is None or node.host is None:
        raise HTTPException(status_code=503, detail="libp2p not ready")
    addrs = multiaddrs_for_bootstrap_get_response(node.self_multiaddrs())
    if not addrs:
        raise HTTPException(
            status_code=503,
            detail=(
                "no dialable (non-loopback) multiaddrs. "
                "If bootstrap runs in Docker and desert runs in other containers, set "
                "DESERT_P2P_ANNOUNCE_ADDR to e.g. /dns4/host.docker.internal/tcp/4001 on the "
                "host-published bootstrap, or omit announce and rely on the container LAN IP. "
                "Native-only: set BOOTSTRAP_INCLUDE_LOOPBACK=1."
            ),
        )
    return {"peer_id": node.peer_id, "multiaddrs": addrs}


def run() -> None:
    import uvicorn

    host = os.environ.get("BOOTSTRAP_HTTP_HOST", "0.0.0.0")
    port = int(os.environ.get("BOOTSTRAP_HTTP_PORT", "8090"))
    uvicorn.run("bootstrap_server.main:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    run()
