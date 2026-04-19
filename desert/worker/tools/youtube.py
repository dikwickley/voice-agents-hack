"""YouTube metadata + comments via yt-dlp (best-effort, no API key)."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _pick_video_url(url: str | None, query: str | None) -> str | None:
    if url and ("youtube.com" in url or "youtu.be" in url):
        return url
    if query:
        # Let yt-dlp resolve search: prefix
        return f"ytsearch1:{query}"
    return url


def run(args: dict[str, Any]) -> dict[str, Any]:
    url = _pick_video_url(args.get("url"), args.get("query"))
    if not url:
        return {"error": "need url or query"}

    out: dict[str, Any] = {"url": url, "title": None, "description": None, "comments": [], "trend_summary": ""}

    try:
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--dump-single-json",
            "--no-warnings",
            "--extractor-args",
            "youtube:player_client=android",
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            log.warning("yt-dlp failed: %s", proc.stderr[:500])
            return {**out, "error": proc.stderr.strip() or "yt-dlp failed"}

        meta = json.loads(proc.stdout)
        out["title"] = meta.get("title")
        out["description"] = (meta.get("description") or "")[:2000]
        out["view_count"] = meta.get("view_count")
        out["like_count"] = meta.get("like_count")

        # Comments: use yt-dlp comment extraction if available
        vid = meta.get("id") or url
        real_url = meta.get("webpage_url") or url
        comments = _try_comments(real_url, vid)
        out["comments"] = comments[:15]

        likes = meta.get("like_count") or 0
        views = meta.get("view_count") or 0
        out["trend_summary"] = _trend_line(views, likes, len(comments))
        return out
    except subprocess.TimeoutExpired:
        return {**out, "error": "timeout"}
    except Exception as e:
        log.exception("youtube tool")
        return {**out, "error": str(e)}


def _try_comments(page_url: str, video_id: str) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory() as td:
            jfile = Path(td) / "comments.json"
            cmd = [
                "yt-dlp",
                "--skip-download",
                "--write-comments",
                "--no-warnings",
                "-o",
                str(Path(td) / "v"),
                page_url,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if proc.returncode != 0:
                return comments
            # yt-dlp writes <filename>.info.json with comments inside
            for p in Path(td).glob("*.info.json"):
                data = json.loads(p.read_text())
                for c in (data.get("comments") or [])[:20]:
                    if isinstance(c, dict):
                        comments.append(
                            {
                                "author": c.get("author"),
                                "text": (c.get("text") or "")[:500],
                                "like_count": c.get("like_count"),
                            }
                        )
                    elif isinstance(c, str):
                        comments.append({"author": None, "text": c[:500]})
                break
    except Exception as e:
        log.debug("comments extract: %s", e)
    if not comments:
        # placeholder so tool still returns something useful
        comments = [{"author": None, "text": "(no comments extracted; metadata only)"}]
    return comments


def _trend_line(views: int, likes: int, n_comments: int) -> str:
    if not views:
        return "Engagement snapshot unavailable."
    er = (likes / max(views, 1)) * 100
    return f"Approx. engagement: {likes} likes / {views} views ({er:.2f}% like-rate), {n_comments} comment samples."
