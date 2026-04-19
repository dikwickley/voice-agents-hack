"""Worker: join the swarm, serve ``/desert/task/2.0.0``, run the pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import sys
import uuid
from typing import Any

import trio

from p2p.swarm import SwarmNode
from p2p.task_service import register_worker_handler
from worker.cactus_pipeline import CactusPipeline, summarize_outbound_messages
from worker.tools import tools_json_subset

log = logging.getLogger(__name__)


async def _run_worker(worker_id: str) -> None:
    pipeline = CactusPipeline(send=_noop_send)

    async def runner(task: dict[str, Any]) -> tuple[str, str, str | None]:
        return await trio.to_thread.run_sync(_run_one_task_sync, pipeline, task)

    node = SwarmNode(role="worker", caps=["llm", "tools"], worker_id=worker_id)
    async with trio.open_nursery() as nursery:
        async def setup(host, _sn: SwarmNode) -> None:
            register_worker_handler(host, runner)
            log.info("worker %s swarm peer=%s ready", worker_id, _sn.peer_id[:12] + "…")

        await node.run(nursery, setup=setup)


def _run_one_task_sync(pipeline: CactusPipeline, task: dict[str, Any]) -> tuple[str, str, str | None]:
    tj = tools_json_subset(list(task.get("tools") or []))

    async def _inner() -> tuple[str, str, str | None]:
        await pipeline.ensure_loaded(tj)
        force_cloud = bool(task.get("force_cloud_fallback"))
        async with pipeline.capture_outbound() as buf:
            await pipeline.process_text_prompt(
                str(task["task_id"]),
                str(task.get("prompt") or ""),
                force_cloud=force_cloud,
            )
        return summarize_outbound_messages(buf)

    return asyncio.run(_inner())


async def _noop_send(_msg: dict[str, Any]) -> None:
    pass


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def run(argv: list[str] | None = None) -> None:
    """CLI entry point. ``argv`` is optional; otherwise sys.argv is used."""
    logging.getLogger("libp2p").setLevel(logging.WARNING)
    logging.getLogger("multiaddr").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(prog="desert-worker")
    parser.add_argument("--worker-id", default=None)

    if argv is None:
        raw = list(sys.argv[1:])
        if raw and raw[0] == "worker":
            raw = raw[1:]
        argv = raw
    args = parser.parse_args(argv)
    wid = args.worker_id or _default_worker_id()
    trio.run(_run_worker, wid)
