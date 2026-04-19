"""Filter multiaddrs returned by ``GET /v1/bootstrap``.

Loopback addresses (`/ip4/127.0.0.1/…`, `/ip6/::1/…`) are valid when every peer
runs on the **same** host. They are **wrong** when desert runs in other Docker
containers: from inside a container, ``127.0.0.1`` is that container, not the
bootstrap host, so libp2p dials the wrong process and Noise handshakes fail.

By default we strip loopback. Set ``BOOTSTRAP_INCLUDE_LOOPBACK=1`` for native
single-host dev only.
"""

from __future__ import annotations

import os


def _is_loopback_multiaddr(s: str) -> bool:
    return "/ip4/127.0.0.1/" in s or "/ip6/::1/" in s


def multiaddrs_for_bootstrap_get_response(addrs: list[str]) -> list[str]:
    """Dedupe, optionally drop loopback (default: drop)."""
    include_loopback = os.environ.get("BOOTSTRAP_INCLUDE_LOOPBACK", "").lower() in (
        "1",
        "true",
        "yes",
    )
    seen: set[str] = set()
    out: list[str] = []
    for a in addrs:
        if not a or not a.startswith("/"):
            continue
        if not include_loopback and _is_loopback_multiaddr(a):
            continue
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out
