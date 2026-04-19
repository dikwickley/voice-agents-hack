"""Desert worker entry point.

The worker is a pure swarm peer: it joins the shared GossipSub topic and
serves the ``/desert/task/2.0.0`` protocol. Anything orchestrator-specific
lives in ``backend.orchestrator``.
"""

from __future__ import annotations

from worker.worker_p2p import run

__all__ = ["run"]
