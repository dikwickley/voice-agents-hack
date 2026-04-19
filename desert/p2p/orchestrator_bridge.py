"""Trio loop that drains the JobStore's pending queue onto swarm workers.

The FastAPI store is asyncio; libp2p is trio. We keep a tiny bridge: a trio
nursery task polls the store via ``run_coroutine_threadsafe`` (the store's
asyncio loop lives on the uvicorn thread), pulls one pending task, picks a
live worker peer from the swarm table, and dispatches it via
``/desert/task/2.0.0``. Results flow back the same way.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Future
from typing import Any

import trio

from libp2p.abc import IHost

from p2p.swarm import PeerRecord, SwarmNode
from p2p.task_service import dispatch_task

log = logging.getLogger(__name__)

# How often we poll the store when idle (there's no async signal today).
IDLE_POLL_SEC = 0.35
# How long to wait for at least one worker to show up on the swarm before
# dispatching the first task of a job.
WORKER_WAIT_MAX_SEC = 60.0


def _await_async(coro, aio_loop: asyncio.AbstractEventLoop) -> Any:
    fut: Future = asyncio.run_coroutine_threadsafe(coro, aio_loop)
    return fut.result(timeout=600)


class _RoundRobin:
    """Fair worker picker: least-recently-used first."""

    def __init__(self) -> None:
        self._last_used: dict[str, float] = {}

    def pick(self, candidates: list[PeerRecord]) -> PeerRecord | None:
        if not candidates:
            return None
        # Least recently used; unknown peers get priority.
        now = time.monotonic()
        ranked = sorted(
            candidates,
            key=lambda r: self._last_used.get(r.peer_id, 0.0),
        )
        chosen = ranked[0]
        self._last_used[chosen.peer_id] = now
        return chosen


async def run_dispatcher(
    host: IHost,
    swarm: SwarmNode,
    store: Any,
    aio_loop: asyncio.AbstractEventLoop,
) -> None:
    """Main loop; spawned as a trio task in the swarm nursery."""
    rr = _RoundRobin()

    while True:
        try:
            task = await trio.to_thread.run_sync(
                lambda: _await_async(store.pop_pending(), aio_loop)
            )
        except Exception as e:
            log.warning("store.pop_pending failed: %s", e)
            await trio.sleep(IDLE_POLL_SEC)
            continue

        if task is None:
            await trio.sleep(IDLE_POLL_SEC)
            continue

        # Wait for a worker to appear in the swarm table.
        picked: PeerRecord | None = None
        waited = 0.0
        while picked is None:
            picked = rr.pick(swarm.peers_by_role("worker"))
            if picked is not None:
                break
            if waited >= WORKER_WAIT_MAX_SEC:
                break
            await trio.sleep(0.5)
            waited += 0.5
        if picked is None:
            log.error("no worker in swarm; marking task %s failed", task.id)
            await trio.to_thread.run_sync(
                lambda: _await_async(
                    store.assign(task.id, "orchestrator"), aio_loop
                )
            )
            await trio.to_thread.run_sync(
                lambda: _await_async(
                    store.complete_task(
                        task.id,
                        "orchestrator",
                        text="",
                        inference_source="local",
                        error="no worker available",
                    ),
                    aio_loop,
                )
            )
            continue

        worker_label = picked.worker_id or picked.peer_id[:12]
        await trio.to_thread.run_sync(
            lambda: _await_async(store.assign(task.id, worker_label), aio_loop)
        )
        trio.lowlevel.spawn_system_task(
            _dispatch_and_complete, host, swarm, picked, task, store, aio_loop
        )


async def _dispatch_and_complete(
    host: IHost,
    swarm: SwarmNode,
    peer: PeerRecord,
    task: Any,
    store: Any,
    aio_loop: asyncio.AbstractEventLoop,
) -> None:
    from libp2p.peer.id import ID as PeerID  # local import avoids cycles

    payload = {
        "task_id": task.id,
        "job_id": task.job_id,
        "kind": task.kind,
        "prompt": task.prompt,
        "tools": list(task.tools),
        "force_cloud_fallback": task.force_cloud_fallback,
    }

    # Ensure we have an open connection + stored addrs; swarm announces
    # populate the peerstore but a fresh peer may need a quick dial.
    info = swarm.peer_info(peer.peer_id)
    if info is not None:
        try:
            await host.connect(info)
        except Exception as e:
            log.warning("dial worker %s failed: %s", peer.peer_id[:12], e)

    worker_label = peer.worker_id or peer.peer_id[:12]
    text, source, error = "", "local", None
    try:
        reply = await dispatch_task(host, PeerID.from_base58(peer.peer_id), payload)
        text = str(reply.get("text") or "")
        source = str(reply.get("inference_source") or "local")
        error = reply.get("error")
    except TimeoutError as e:
        error = str(e)
    except Exception as e:
        log.exception("task %s on %s failed", task.id, worker_label)
        error = str(e)

    try:
        await trio.to_thread.run_sync(
            lambda: _await_async(
                store.complete_task(
                    task.id,
                    worker_label,
                    text=text,
                    inference_source=source,
                    error=error,
                ),
                aio_loop,
            )
        )
    except Exception:
        log.exception("store.complete_task")
