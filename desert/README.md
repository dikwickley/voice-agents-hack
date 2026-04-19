# Desert

Distributed map → reduce over libp2p using **two images**:

| Image | Role |
|--------|------|
| `desert-bootstrap` | libp2p peer + `GET /v1/bootstrap` (dialable multiaddrs; loopback filtered for Docker-safe defaults). |
| `desert` | **Worker** (default) or **orchestrator** (`DESERT_MODE=orchestrator`). Uses `DESERT_BOOTSTRAP_URL` to join the mesh. |

Orchestrator HTTP is for the Textual TUI only; tasks use `/desert/task/2.0.0` over libp2p.

## Quick start

See repo **`CLI.md`** for `docker run` commands, `BOOTSTRAP_INCLUDE_LOOPBACK`, and troubleshooting.

## Dev (no Docker)

```bash
cd desert && uv sync
BOOTSTRAP_INCLUDE_LOOPBACK=1 DESERT_P2P_ANNOUNCE_ADDR=/ip4/127.0.0.1/tcp/4001 uv run desert-bootstrap
DESERT_BOOTSTRAP_URL=http://127.0.0.1:8090 uv run desert worker
DESERT_BOOTSTRAP_URL=http://127.0.0.1:8090 uv run desert orchestrator
```

## Layout

- `p2p/swarm.py` — GossipSub + bootstrap HTTP client
- `p2p/bootstrap_http_addrs.py` — safe `/v1/bootstrap` address list
- `bootstrap_server/` — FastAPI + libp2p bootstrap peer
- `docker/Dockerfile.bootstrap` / `Dockerfile.desert`
