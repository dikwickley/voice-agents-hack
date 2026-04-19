"""Tool dispatcher for LLM function calling."""

from __future__ import annotations

from worker.tools import doc_analyzer, web_search, youtube


def run_tool(name: str, args: dict) -> dict:
    if name == "youtube_scraper":
        return youtube.run(args)
    if name == "doc_analyzer":
        return doc_analyzer.run(args)
    if name == "web_search":
        return web_search.run(args)
    return {"error": f"unknown tool: {name}"}


def tools_json() -> str:
    import json

    specs = [
        {
            "type": "function",
            "function": {
                "name": "youtube_scraper",
                "description": "Fetch YouTube video metadata and top comments from a video URL or search query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Full YouTube video URL",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query if no URL",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current facts, news, or references. Uses DuckDuckGo by default; set WEB_SEARCH_PROVIDER=tavily and TAVILY_API_KEY for Tavily.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max snippets to return (1–10)",
                        },
                        "provider": {
                            "type": "string",
                            "description": "ddg or tavily (optional; can use env WEB_SEARCH_PROVIDER)",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "doc_analyzer",
                "description": "Analyze document text for summary and risk tags.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Document text to analyze",
                        },
                    },
                    "required": ["text"],
                },
            },
        },
    ]
    return json.dumps(specs)


def tools_json_subset(names: list[str]) -> str:
    """OpenAI-style tool list filtered by function name (empty list → no tools)."""
    import json

    if not names:
        return "[]"
    want = set(names)
    specs = json.loads(tools_json())
    out = [s for s in specs if s.get("function", {}).get("name") in want]
    return json.dumps(out)
