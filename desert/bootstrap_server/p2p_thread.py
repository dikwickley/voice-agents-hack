"""Run ``SwarmNode(role=bootstrap)`` in a trio thread; expose handle for HTTP."""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any

import trio

from p2p.swarm import SwarmNode

log = logging.getLogger(__name__)

_lock = threading.Lock()
_node: SwarmNode | None = None


def get_bootstrap_swarm_node() -> SwarmNode | None:
    with _lock:
        return _node


async def _run_bootstrap_swarm() -> None:
    global _node
    node = SwarmNode(role="bootstrap", caps=["bootstrap"])
    with _lock:
        _node = node
    async with trio.open_nursery() as nursery:
        await node.run(nursery, setup=None)


def start_bootstrap_swarm_thread() -> None:
    def runner() -> None:
        try:
            trio.run(functools.partial(_run_bootstrap_swarm))
        except Exception:
            log.exception("bootstrap swarm thread exited")

    threading.Thread(target=runner, name="desert-bootstrap-p2p", daemon=True).start()
