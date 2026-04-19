"""Fetch libp2p seed multiaddrs from the desert bootstrap HTTP server."""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


def fetch_bootstrap_multiaddrs(base_url: str, *, timeout: float = 10.0) -> list[str]:
    """GET ``{base_url}/v1/bootstrap`` → ``multiaddrs`` list."""
    url = base_url.rstrip("/") + "/v1/bootstrap"
    r = httpx.get(url, timeout=timeout)
    r.raise_for_status()
    data: dict[str, Any] = r.json()
    raw = data.get("multiaddrs") or []
    return [str(x).strip() for x in raw if str(x).strip()]
