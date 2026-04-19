"""Task RPC over libp2p.

One stream per task. Orchestrator writes a ``TaskRequest`` line, worker replies
with a single ``TaskResult`` line and closes. This replaces the old
register/claim/complete polling: workers are push-receivers, the orchestrator
picks who to dispatch to from the swarm table.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

import trio

from libp2p.abc import IHost
from libp2p.custom_types import TProtocol
from libp2p.network.stream.net_stream import INetStream
from libp2p.peer.id import ID

from p2p import TASK_PROTOCOL
from p2p.codec import read_msg, write_msg

log = logging.getLogger(__name__)
PROTOCOL_ID = TProtocol(TASK_PROTOCOL)

# Upper bound on the whole task — local inference can be slow, cloud fallback
# can stall on a flaky network. 10 minutes is generous for reduce steps too.
TASK_STREAM_TIMEOUT_SEC = 600.0


# ── worker side ──────────────────────────────────────────────────────────

TaskRunner = Callable[[dict[str, Any]], Awaitable[tuple[str, str, str | None]]]
"""Given a task dict, return (text, inference_source, error)."""


def register_worker_handler(host: IHost, runner: TaskRunner) -> None:
    """Attach the ``/desert/task/2.0.0`` handler on the worker's host."""

    async def handler(stream: INetStream) -> None:
        try:
            with trio.move_on_after(TASK_STREAM_TIMEOUT_SEC) as scope:
                req = await read_msg(stream)
                task = req.get("task") or {}
                task_id = task.get("task_id") or "?"
                log.info(
                    "task in id=%s kind=%s job=%s",
                    task_id,
                    task.get("kind"),
                    task.get("job_id"),
                )
                text, source, err = await runner(task)
                await write_msg(
                    stream,
                    {
                        "task_id": task_id,
                        "text": text,
                        "inference_source": source,
                        "error": err,
                    },
                )
            if scope.cancelled_caught:
                log.warning("task stream timed out")
        except Exception:
            log.exception("task handler")
            try:
                await write_msg(stream, {"error": "handler exception"})
            except Exception:
                pass
        finally:
            try:
                await stream.close()
            except Exception:
                pass

    host.set_stream_handler(PROTOCOL_ID, handler)


# ── orchestrator side ────────────────────────────────────────────────────


async def dispatch_task(
    host: IHost,
    peer_id: ID,
    task: dict[str, Any],
    *,
    timeout_sec: float = TASK_STREAM_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Open a task stream, send ``task``, return the worker's reply dict.

    Raises ``TimeoutError`` if the worker doesn't reply inside ``timeout_sec``.
    """
    stream: INetStream | None = None
    with trio.move_on_after(timeout_sec) as scope:
        stream = await host.new_stream(peer_id, [PROTOCOL_ID])
        try:
            await write_msg(stream, {"task": task})
            reply = await read_msg(stream)
            return reply or {}
        finally:
            try:
                if stream is not None:
                    await stream.close()
            except Exception:
                pass
    if scope.cancelled_caught:
        raise TimeoutError(f"task {task.get('task_id')} on {peer_id} timed out")
    return {}
