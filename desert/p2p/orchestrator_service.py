"""Orchestrator-side libp2p glue: joins the swarm + drives the dispatcher.

Runs in its own trio thread (spawned from the FastAPI lifespan) so the trio
event loop and the uvicorn/asyncio loop don't fight each other.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from typing import Any

import trio

from p2p.orchestrator_bridge import run_dispatcher
from p2p.swarm import SwarmNode

log = logging.getLogger(__name__)

_lock = threading.Lock()
_node: SwarmNode | None = None


def get_swarm_node() -> SwarmNode | None:
    with _lock:
        return _node


async def _run_orchestrator_swarm(store: Any, aio_loop: asyncio.AbstractEventLoop) -> None:
    global _node
    node = SwarmNode(role="orchestrator", caps=["dispatch"])
    with _lock:
        _node = node

    async with trio.open_nursery() as nursery:
        async def setup(host, sn: SwarmNode) -> None:
            # The orchestrator doesn't serve /desert/task/2.0.0 itself; only
            # workers do. It subscribes to swarm announces + runs the
            # dispatcher loop below.

            async def _mirror_workers(rec) -> None:
                if rec.role == "worker":
                    label = rec.worker_id or rec.peer_id[:12]
                    try:
                        await trio.to_thread.run_sync(
                            lambda: asyncio.run_coroutine_threadsafe(
                                store.touch_worker(label), aio_loop
                            ).result(timeout=5)
                        )
                    except Exception:
                        log.debug("store.touch_worker failed", exc_info=True)

            sn.on_peer(lambda rec: trio.lowlevel.spawn_system_task(_mirror_workers, rec))
            nursery.start_soon(run_dispatcher, host, sn, store, aio_loop)

        await node.run(nursery, setup=setup)


def start_orchestrator_swarm_thread(
    *,
    store: Any,
    aio_loop: asyncio.AbstractEventLoop,
) -> None:
    def runner() -> None:
        try:
            trio.run(
                functools.partial(
                    _run_orchestrator_swarm,
                    store,
                    aio_loop,
                )
            )
        except Exception:
            log.exception("orchestrator swarm thread exited")

    threading.Thread(target=runner, name="desert-swarm", daemon=True).start()
