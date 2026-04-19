"""Newline-delimited JSON on libp2p streams."""

from __future__ import annotations

import json
from typing import Any

from libp2p.network.stream.net_stream import INetStream


MAX_LINE = 8 * 1024 * 1024


async def write_msg(stream: INetStream, obj: Any) -> None:
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
    if len(line) > MAX_LINE:
        raise ValueError("message too large")
    await stream.write(line)


async def read_msg(stream: INetStream) -> dict[str, Any]:
    buf = bytearray()
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > MAX_LINE:
            raise ValueError("message too large")
        if b"\n" in buf:
            break
    line, _, _ = bytes(buf).partition(b"\n")
    if not line:
        return {}
    return json.loads(line.decode("utf-8"))
