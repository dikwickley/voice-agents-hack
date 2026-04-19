"""Web search tool: DuckDuckGo (no key) or Tavily (optional API key)."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)


def _search_ddg(query: str, max_results: int) -> dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"error": "empty query"}
    max_results = max(1, min(10, int(max_results or 5)))
    out: dict[str, Any] = {"provider": "duckduckgo", "query": q, "results": []}
    try:
        r = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=15.0,
            follow_redirects=True,
        )
        r.raise_for_status()
        data = r.json()
        abstract = (data.get("AbstractText") or "").strip()
        src = (data.get("AbstractURL") or "").strip()
        if abstract:
            out["results"].append({"title": data.get("Heading") or "Summary", "snippet": abstract, "url": src})
        for t in (data.get("RelatedTopics") or [])[: max_results - len(out["results"])]:
            if isinstance(t, dict) and "Text" in t:
                out["results"].append(
                    {
                        "title": (t.get("Text") or "")[:120],
                        "snippet": t.get("Text") or "",
                        "url": t.get("FirstURL") or "",
                    }
                )
        if not out["results"] and data.get("Answer"):
            out["results"].append({"title": "Instant answer", "snippet": str(data["Answer"]), "url": ""})
        return out
    except Exception as e:
        log.exception("ddg search")
        return {"error": str(e), "provider": "duckduckgo", "query": q}


def _search_tavily(query: str, max_results: int) -> dict[str, Any]:
    key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if not key:
        return {"error": "TAVILY_API_KEY not set", "provider": "tavily"}
    q = (query or "").strip()
    if not q:
        return {"error": "empty query"}
    max_results = max(1, min(10, int(max_results or 5)))
    try:
        r = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": q, "max_results": max_results},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
        results = []
        for item in (data.get("results") or [])[:max_results]:
            results.append(
                {
                    "title": item.get("title") or "",
                    "snippet": item.get("content") or "",
                    "url": item.get("url") or "",
                }
            )
        return {"provider": "tavily", "query": q, "results": results}
    except Exception as e:
        log.exception("tavily search")
        return {"error": str(e), "provider": "tavily", "query": q}


def run(args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query") or args.get("q") or "")
    max_results = int(args.get("max_results") or 5)
    provider = (args.get("provider") or os.environ.get("WEB_SEARCH_PROVIDER") or "ddg").lower()
    if provider in ("tavily", "tvly"):
        return _search_tavily(query, max_results)
    return _search_ddg(query, max_results)
