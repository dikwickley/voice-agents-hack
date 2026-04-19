# Desert — two images (localhost first)

1. **`desert-bootstrap`** — libp2p peer + HTTP `GET /v1/bootstrap` (dialable multiaddrs).
2. **`desert`** — worker (default) or orchestrator (TUI). Set **`DESERT_BOOTSTRAP_URL`** to the bootstrap HTTP base URL.

There is no mDNS: the bootstrap server is the rendezvous.

## Why bootstrap must not advertise `127.0.0.1` to other containers

`GET /v1/bootstrap` **drops loopback** multiaddrs by default. Inside a container, `127.0.0.1` is that container, not the host or another container—workers would dial themselves and Noise would fail. For processes on the **same machine without Docker**, set **`BOOTSTRAP_INCLUDE_LOOPBACK=1`** on the bootstrap container.

## Build

From repo root:

```bash
docker build -f desert/docker/Dockerfile.bootstrap -t desert-bootstrap:latest .
docker build -f desert/docker/Dockerfile.desert -t desert:latest .
```

### Bake `GEMINI_API_KEY` into `desert` (recommended)

Defaults in the image: **local `google/gemma-3-270m-it`**, **`DESERT_CLOUD_FALLBACK=1`**, **`DESERT_GEMINI_MODEL=gemini-3-flash-preview`**. You usually only need to pass the API key and keep the rest implicit.

```bash
# From repo root (inline key — be careful on shared machines).
docker build -f desert/docker/Dockerfile.desert -t desert:latest \
  --build-arg GEMINI_API_KEY="YOUR_GEMINI_API_KEY" \
  .

# Or read from desert/.env (line: GEMINI_API_KEY=...)
export GEMINI_API_KEY="$(grep -E '^GEMINI_API_KEY=' desert/.env | cut -d= -f2-)"
docker build -f desert/docker/Dockerfile.desert -t desert:latest \
  --build-arg GEMINI_API_KEY="${GEMINI_API_KEY}" \
  .
```

Optional: `--build-arg BAKE_LLM_MODEL_ID=google/gemma-3-270m-it`, `--build-arg DESERT_CLOUD_FALLBACK=1`, `--build-arg DESERT_GEMINI_MODEL=...` to override. At run time: `-e GEMINI_API_KEY=...` still works.

## Docker: bootstrap + workers (typical)

**Terminal A — bootstrap** (published P2P + HTTP; loopback stripped from `/v1/bootstrap` so clients get the container LAN address, e.g. `172.17.x.x`)

```bash
docker run --rm -p 8090:8090 -p 4001:4001 desert-bootstrap:latest
```

**Terminal B — worker(s)**

```bash
docker run --rm \
  -e DESERT_BOOTSTRAP_URL=http://host.docker.internal:8090 \
  desert:latest
```

**Terminal C — orchestrator**

```bash
docker run --rm -it -p 8000:8000 \
  -e DESERT_MODE=orchestrator \
  -e DESERT_BOOTSTRAP_URL=http://host.docker.internal:8090 \
  desert:latest
```

On Linux, add `--add-host=host.docker.internal:host-gateway` to **worker and orchestrator** if `host.docker.internal` is missing, or use `http://172.17.0.1:8090` for the HTTP URL only (bootstrap API must still return P2P addrs your network can reach—usually the bootstrap container’s `172.17.x.x` from `docker inspect`).

If you **must** advertise a hostname for P2P (e.g. host gateway), run bootstrap with:

```bash
docker run --rm -p 8090:8090 -p 4001:4001 \
  -e DESERT_P2P_ANNOUNCE_ADDR=/dns4/host.docker.internal/tcp/4001 \
  desert-bootstrap:latest
```

## Environment

| Variable | Where | Meaning |
|----------|--------|---------|
| `DESERT_BOOTSTRAP_URL` | desert | Base URL of bootstrap HTTP (e.g. `http://host.docker.internal:8090`) |
| `DESERT_P2P_ANNOUNCE_ADDR` | libp2p peers | Optional; multiaddr **others** use to dial this node |
| `DESERT_P2P_LISTEN_PORT` | all | TCP listen port (default `4001`) |
| `BOOTSTRAP_HTTP_PORT` | bootstrap | HTTP port (default `8090`) |
| `BOOTSTRAP_INCLUDE_LOOPBACK` | bootstrap | Set `1` only for native dev (all processes on host); never for cross-container |

## Without Docker

Terminal 1:

```bash
cd desert && uv sync
BOOTSTRAP_INCLUDE_LOOPBACK=1 DESERT_P2P_ANNOUNCE_ADDR=/ip4/127.0.0.1/tcp/4001 uv run desert-bootstrap
```

Terminal 2 / 3:

```bash
DESERT_BOOTSTRAP_URL=http://127.0.0.1:8090 uv run desert worker
DESERT_BOOTSTRAP_URL=http://127.0.0.1:8090 uv run desert orchestrator
```

## Troubleshooting

- **Noise / handshake / `127.0.0.1` in errors** — do not publish loopback to other containers; use default bootstrap image (filters loopback) or fix `DESERT_P2P_ANNOUNCE_ADDR`.
- **`503` on `/v1/bootstrap`** — no non-loopback addrs; run bootstrap in Docker without `127.0.0.1` announce, or set `BOOTSTRAP_INCLUDE_LOOPBACK=1` for native-only.
- **`DESERT_BOOTSTRAP_URL` unset** — set it on every desert container.
