"""
data_ingestion.py — Data Ingestion Module.

Fetches recent news using the Tavily API (Search for AI agents).
Replaces legacy DuckDuckGo and NewsAPI collectors.
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
    """Fetch recent news/context from Tavily Search.

    Parameters
    ----------
    query : str
        Search query (market question or keywords).
    api_key : str
        Tavily.com API key.
    max_results : int
        Maximum number of results to return.

    Returns
    -------
    list[NewsItem]
        Found articles; empty list on any error.
    """
    if not api_key:
        logger.error("tavily_api_key_missing", hint="Check your TAVILY_API_KEY environment variable")
        return []

    try:
        from tavily import TavilyClient
    except ImportError:
        logger.error("tavily_import_error", hint="pip install tavily-python")
        return []

    def _sync_search():
        client = TavilyClient(api_key=api_key)
        # We use 'search' for general results. 
        # For trading, 'news' or advanced depth might be better, but 'search' is the most versatile.
        return client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )

    try:
        response = await asyncio.to_thread(_sync_search)
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
                source=r.get("url", "Tavily"),
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
    """Fetch news from Tavily.

    Returns
    -------
    list[NewsItem]
        Deduplicated news items.
    """
    # Currently only Tavily is used as requested by user.
    items = await fetch_tavily(query, tavily_api_key)
    
    # Simple deduplication by title
    seen = set()
    unique = []
    for item in items:
        if item.title not in seen:
            seen.add(item.title)
            unique.append(item)
            
    logger.info("news_collected", total=len(unique), query=query[:80])
    return unique
