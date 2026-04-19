"""FastAPI orchestrator: HTTP surface for the TUI; all worker RPC is libp2p."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.orchestrator.store import JobStore
from p2p.orchestrator_service import get_swarm_node, start_orchestrator_swarm_thread

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_TOOLS = ["youtube_scraper", "doc_analyzer", "web_search"]
store = JobStore()


class CreateJobRequest(BaseModel):
    challenge: str = Field(..., min_length=1)
    parallel: int | None = Field(default=None)
    tools: list[str] | None = None
    force_cloud_fallback: bool = False

    @field_validator("parallel")
    @classmethod
    def parallel_range(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if not 1 <= v <= 32:
            raise ValueError("parallel must be 1..32 or null for auto")
        return v


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    start_orchestrator_swarm_thread(store=store, aio_loop=loop)
    log.info("orchestrator swarm started")
    yield


app = FastAPI(title="Desert Orchestrator", version="0.4.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    stats = await store.worker_stats()
    node = get_swarm_node()
    return {
        "ok": True,
        **stats,
        "swarm_peers": (len(node.peers()) if node else 0),
    }


@app.get("/v1/workers")
async def list_workers() -> dict[str, Any]:
    node = get_swarm_node()
    workers: list[dict[str, Any]] = []
    if node is not None:
        import time

        now = time.monotonic()
        for rec in node.peers_by_role("worker"):
            workers.append(
                {
                    "worker_id": rec.worker_id or rec.peer_id[:12],
                    "peer_id": rec.peer_id,
                    "multiaddrs": rec.multiaddrs,
                    "caps": rec.caps,
                    "last_seen_sec_ago": round(max(0.0, now - rec.last_seen), 2),
                }
            )
    return {"count": len(workers), "workers": workers}


@app.post("/v1/jobs")
async def create_job(body: CreateJobRequest) -> dict[str, str]:
    tools = list(body.tools) if body.tools else list(DEFAULT_TOOLS)
    node = get_swarm_node()
    nodes_available = len(node.peers_by_role("worker")) if node else 0
    job_id = await store.create_job(
        body.challenge,
        body.parallel,
        tools,
        force_cloud_fallback=body.force_cloud_fallback,
        nodes_available=nodes_available,
    )
    log.info(
        "job %s parallel=%s force_cloud=%s",
        job_id,
        body.parallel,
        body.force_cloud_fallback,
    )
    return {"job_id": job_id}


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    data = await store.get_job_public(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="unknown job")
    return data


def run() -> None:
    import uvicorn

    host = os.environ.get("ORCH_HOST", "0.0.0.0")
    port = int(os.environ.get("ORCH_PORT", "8000"))
    uvicorn.run("backend.orchestrator.main:app", host=host, port=port, reload=False)
