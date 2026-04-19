"""libp2p layer for desert (trio).

- Bootstrap HTTP ``GET /v1/bootstrap`` → seed multiaddrs (``DESERT_BOOTSTRAP_URL``).
- ``desert/swarm/v1`` on GossipSub carries per-peer presence records.
- ``/desert/task/2.0.0`` is a one-shot NDJSON orchestrator → worker RPC.
"""

TASK_PROTOCOL = "/desert/task/2.0.0"
