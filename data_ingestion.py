"""
data_ingestion.py — Data Ingestion Module.

Fetches recent news using the Tavily API (Search for AI agents).
Optimised for speed and reliability using the Tavily Python SDK.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import structlog

logger = structlog.get_logger(__name__)

# ── Data structures ─────────────────────────────────────────


@dataclass(frozen=True)
class NewsItem:
    """A single piece of news relevant to a Polymarket market."""

    title: str
    summary: str
    source: str
    url: str = ""
    published_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Tavily collector ────────────────────────────────────────


async def fetch_tavily(
    query: str,
    api_key: str,
    *,
    max_results: int = 5,
) -> list[NewsItem]:
    """Fetch recent news from Tavily Search API.

    Uses the topic="news" parameter for more relevant and timely results.
    """
    if not api_key:
        logger.error("tavily_api_key_missing", hint="Check your TAVILY_API_KEY environment variable")
        return []

    try:
        from tavily import AsyncTavilyClient
    except ImportError:
        logger.error("tavily_import_error", hint="pip install tavily-python")
        return []

    client = AsyncTavilyClient(api_key=api_key)

    try:
        # According to documentation:
        # topic="news" is optimized for latest news.
        # search_depth="basic" is faster and costs 1 credit (vs 2 for advanced).
        # We use basic to conserve the 1,000 credit limit for 39+ markets.
        response = await client.search(
            query=query,
            topic="news",
            search_depth="basic",
            max_results=max_results,
        )
        results = response.get("results", [])
    except Exception as exc:
        logger.error("tavily_error", error=repr(exc), query=query[:80])
        return []

    items: list[NewsItem] = []
    for r in results:
        items.append(
            NewsItem(
                title=r.get("title", "No Title"),
                summary=r.get("content", ""),
                source=r.get("url", "Tavily News"),
                url=r.get("url", ""),
            )
        )

    logger.info("tavily_fetched", count=len(items), query=query[:80])
    return items


# ── Unified collector ───────────────────────────────────────


async def collect_news(
    query: str,
    tavily_api_key: str = "",
) -> list[NewsItem]:
    """Fetch news from Tavily Search.

    Returns
    -------
    list[NewsItem]
        Deduplicated news items.
    """
    if not tavily_api_key:
        return []

    items = await fetch_tavily(query, tavily_api_key)
    
    # Simple deduplication by title (case-insensitive)
    seen = set()
    unique = []
    for item in items:
        key = item.title.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
            
    logger.info("news_collected", total=len(unique), query=query[:80])
    return unique
