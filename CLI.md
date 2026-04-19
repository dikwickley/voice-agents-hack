# Desert — native run (uv)

## Quickstart

After a one-time setup (see [Prereqs](#prereqs-once)), three terminals from `desert/`:

```bash
# Terminal A — bootstrap (HTTP :8090, libp2p :4001)
uv run desert-bootstrap

# Terminal B — orchestrator TUI:  
uv run desert orchestrator
# uv run python -c "from backend.orchestrator.main import run; run()"

# Terminal C — worker (joins the swarm, serves /desert/task/2.0.0)
uv run desert worker
```

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/workers | python3 -m json.tool
```

`desert/.env` is auto-loaded by every entrypoint, so `GEMINI_API_KEY`, `DESERT_CLOUD_FALLBACK`, etc. flow through with no `--env-file` / no `export`. The worker adds `cactus/python` to `sys.path` on import — no `PYTHONPATH` needed. The bootstrap pins libp2p `:4001` and clients default `DESERT_BOOTSTRAP_URL` to `http://127.0.0.1:8090`, so a same-host loop needs **zero** environment overrides on the command line.

---

Three processes, one host: **bootstrap**, **worker(s)**, **orchestrator**. They meet via the bootstrap's HTTP `GET /v1/bootstrap` endpoint (returns dialable libp2p multiaddrs) and then talk over a shared GossipSub topic (`desert/swarm/v1`). There is no mDNS.

## Prereqs (once)

1. Python deps for desert:

   ```bash
   cd desert
   uv sync
   ```

2. Build Cactus shared library + download the default model. The worker uses `libcactus.dylib`/`.so` via `ctypes`, loaded from `desert/cactus/cactus/build/`.

   ```bash
   cd desert/cactus
   source ./setup                       # one-time: sets up cactus CLI + venv
   source venv/bin/activate && cactus build --python
   cactus download google/gemma-3-270m-it
   ```

   The worker also needs an ASR model; it will be fetched automatically on first request (`openai/whisper-tiny`) into `desert/cactus/weights/`.

3. (Optional) Drop a `desert/.env` with keys you don't want on the command line:

   ```ini
   GEMINI_API_KEY=...
   DESERT_CLOUD_FALLBACK=1
   DESERT_GEMINI_MODEL=gemini-3-flash-preview
   ```

   All three entrypoints (`cli.main`, `bootstrap_server.main`, `backend.orchestrator.main`) call `dotenv.load_dotenv(desert/.env)` before reading `os.environ`.

## Run (three terminals)

All commands run from `desert/`. Each line below shows the **minimum** you need; every flag is optional.

**Terminal A — bootstrap**

```bash
uv run desert-bootstrap
```

Binds HTTP on `:8090` and libp2p on `:4001` by default. `GET /v1/bootstrap` returns the host's LAN multiaddr (loopback is stripped by default; see Troubleshooting).

**Terminal B — orchestrator API**

```bash
uv run python -c "from backend.orchestrator.main import run; run()"
```

Or with the full TUI:

```bash
uv run desert orchestrator
```

Defaults to `http://0.0.0.0:8000`. Override with `ORCH_HOST` / `ORCH_PORT` env vars if needed.

**Terminal C — worker**

```bash
uv run desert worker
```

Every invocation gets its own random libp2p port (`find_free_port()`), so you can run several workers back-to-back without port collisions. Add `--worker-id <name>` to make them easier to tell apart in `/v1/workers`.

## Smoke test via the orchestrator API

```bash
# Is the orchestrator up and has it seen the worker?
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/workers

# Submit a one-shot job
JID=$(curl -s -X POST http://127.0.0.1:8000/v1/jobs \
  -H 'content-type: application/json' \
  -d '{"challenge":"Say hello in one sentence.","parallel":1}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')

# Poll until done
curl -s http://127.0.0.1:8000/v1/jobs/$JID | python3 -m json.tool
```

On the first job the worker blocks briefly while loading ASR + LLM weights; subsequent jobs are hot. `inference_source` is `local` for on-device Cactus output and `gemini` when the cloud fallback kicks in (set `force_cloud_fallback: true` in the request body to force it).

## Environment

| Variable | Where | Default | Meaning |
|----------|-------|---------|---------|
| `DESERT_BOOTSTRAP_URL` | worker, orchestrator | `http://127.0.0.1:8090` | Base URL of the bootstrap HTTP |
| `DESERT_P2P_LISTEN_PORT` | all | `4001` (bootstrap) / `0` else | TCP port for libp2p (`0` → kernel picks free) |
| `DESERT_P2P_ANNOUNCE_ADDR` | all | unset | Optional explicit multiaddr to advertise |
| `BOOTSTRAP_HTTP_HOST` / `BOOTSTRAP_HTTP_PORT` | bootstrap | `0.0.0.0:8090` | HTTP bind |
| `BOOTSTRAP_INCLUDE_LOOPBACK` | bootstrap | unset | Include `127.0.0.1` / `::1` in `/v1/bootstrap`. **Leave unset**; see Troubleshooting |
| `ORCH_HOST` / `ORCH_PORT` | orchestrator | `0.0.0.0:8000` | FastAPI bind |
| `DESERT_LLM_MODEL_ID` | worker | `google/gemma-3-270m-it` | Cactus LLM id |
| `DESERT_ASR_MODEL` | worker | `openai/whisper-tiny` | Cactus ASR id |
| `DESERT_LLM_WEIGHTS` | worker | unset | Override LLM weights dir (skip `ensure_model`) |
| `DESERT_CLOUD_FALLBACK` | worker | `1` | Allow Gemini fallback after local |
| `GEMINI_API_KEY` | worker | unset | Required for cloud fallback / `force_cloud_fallback` |
| `DESERT_GEMINI_MODEL` | worker | `gemini-3-flash-preview` | Gemini model id |
| `DESERT_MOCK` | worker | `0` | `1` → skip Cactus load; return canned replies (p2p-only tests) |

## Troubleshooting

- **Orchestrator shows `workers: []` forever** — almost always means the GossipSub mesh didn't form. The usual cause: `/v1/bootstrap` returned more than one seed address and libp2p (0.6.0) raced two concurrent `/meshsub/1.0.0` stream handshakes over the same peer, one of which gets dropped with `MultiselectClient handshake: read failed`. Fix: leave `BOOTSTRAP_INCLUDE_LOOPBACK` unset and don't set `DESERT_P2P_ANNOUNCE_ADDR` on the bootstrap — the default filter returns exactly one non-loopback multiaddr (your LAN IP), which every process on the same Mac can dial.

- **`/v1/bootstrap` returns 503 "no dialable (non-loopback) multiaddrs"** — the host has no non-loopback IPv4 interface. Either bring up Wi-Fi / Ethernet, or set `BOOTSTRAP_INCLUDE_LOOPBACK=1` **and** be aware of the race above (may require restarting clients until the mesh settles).

- **`❌ No IPv4+TCP addresses for <peer>`** — cosmetic log from libp2p when it skips an IPv6-only seed. Safe to ignore.

- **`job failed: force cloud failed: set GEMINI_API_KEY`** — the worker doesn't have the key in `os.environ`. Make sure `desert/.env` sets `GEMINI_API_KEY=...`; every entrypoint auto-loads that file. If you run the worker from somewhere else, point at it explicitly: `uv run --env-file /path/to/.env desert worker`.

- **`Cactus library not found at .../libcactus.dylib`** on the worker — run `cactus build --python` in `desert/cactus/`.

- **`LLM weights not found`** on the worker — run `cactus download google/gemma-3-270m-it` (or whatever `DESERT_LLM_MODEL_ID` points to).
