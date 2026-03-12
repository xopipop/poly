"""
data_ingestion.py — Data Ingestion Module.

Asynchronously fetches recent news from multiple sources:
  • DuckDuckGo Search  (primary, no API key required)
  • News API           (optional, requires key from newsapi.org)

Returns a unified list of ``NewsItem`` objects for downstream analysis.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import aiohttp
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


# ── DuckDuckGo collector (primary, no key) ──────────────────


async def fetch_duckduckgo(
    query: str,
    *,
    max_results: int = 15,
    time_filter: str = "d",  # "d" = past day, "w" = past week
) -> list[NewsItem]:
    """Fetch recent news from DuckDuckGo Search.

    Uses the ``duckduckgo-search`` library (synchronous) wrapped in
    ``asyncio.to_thread`` to keep the event loop responsive.

    Parameters
    ----------
    query : str
        Search query (market question or keywords).
    max_results : int
        Maximum number of results to return.
    time_filter : str
        Recency filter: ``"d"`` = past 24 h, ``"w"`` = past week.

    Returns
    -------
    list[NewsItem]
        Found articles; empty list on any error.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.error(
            "duckduckgo_import_error",
            hint="pip install duckduckgo-search",
        )
        return []

    def _search() -> list[dict]:
        """Run the synchronous DuckDuckGo search."""
        with DDGS() as ddgs:
            return list(
                ddgs.news(
                    keywords=query,
                    max_results=max_results,
                    timelimit=time_filter,
                )
            )

    try:
        raw_results = await asyncio.wait_for(
            asyncio.to_thread(_search),
            timeout=20.0,
        )
    except asyncio.TimeoutError:
        logger.error("duckduckgo_timeout", query=query[:80])
        return []
    except Exception as exc:
        logger.error("duckduckgo_error", error=str(exc), query=query[:80])
        return []

    items: list[NewsItem] = []
    for r in raw_results:
        title = r.get("title", "")
        body = r.get("body", "") or r.get("description", "")
        source = r.get("source", "DuckDuckGo")
        url = r.get("url", "") or r.get("link", "")

        if not title:
            continue

        items.append(
            NewsItem(
                title=title,
                summary=body,
                source=source,
                url=url,
            )
        )

    logger.info("duckduckgo_fetched", count=len(items), query=query[:80])
    return items


# ── News API collector (optional, requires key) ─────────────

NEWS_API_BASE = "https://newsapi.org/v2/everything"


async def fetch_news_api(
    query: str,
    api_key: str,
    *,
    language: str = "en",
    page_size: int = 20,
    session: aiohttp.ClientSession | None = None,
) -> list[NewsItem]:
    """Fetch articles from News API matching *query*.

    Returns an empty list (not an exception) on non-critical errors so
    the pipeline can continue with other sources.
    """
    if not api_key:
        logger.debug("news_api_key_missing", hint="Skipping News API")
        return []

    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }

    own_session = session is None
    session = session or aiohttp.ClientSession()

    try:
        async with session.get(
            NEWS_API_BASE,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error("news_api_error", status=resp.status, body=body[:300])
                return []

            data = await resp.json()

        items: list[NewsItem] = []
        for article in data.get("articles", []):
            items.append(
                NewsItem(
                    title=article.get("title", ""),
                    summary=(
                        article.get("description", "")
                        or article.get("content", "")
                    ),
                    source=article.get("source", {}).get("name", "NewsAPI"),
                    url=article.get("url", ""),
                )
            )
        logger.info("news_api_fetched", count=len(items), query=query[:80])
        return items

    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.error("news_api_network_error", error=str(exc))
        return []
    finally:
        if own_session:
            await session.close()


# ── Unified collector ───────────────────────────────────────


def _deduplicate(items: list[NewsItem]) -> list[NewsItem]:
    """Remove duplicates by normalised title."""
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for item in items:
        key = item.title.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


async def collect_news(
    query: str,
    news_api_key: str = "",
    *,
    max_ddg_results: int = 15,
) -> list[NewsItem]:
    """Run all collectors in parallel, merge, and deduplicate.

    DuckDuckGo is always used (no key required).
    News API is used only when *news_api_key* is provided.

    Returns
    -------
    list[NewsItem]
        Combined, deduplicated news items (newest first).
    """
    tasks = [fetch_duckduckgo(query, max_results=max_ddg_results)]

    # Optionally add News API
    if news_api_key:
        tasks.append(fetch_news_api(query, news_api_key))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items: list[NewsItem] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error("collector_exception", error=str(result))
            continue
        if isinstance(result, list):
            all_items.extend(result)

    all_items = _deduplicate(all_items)
    logger.info("news_collected", total=len(all_items), query=query[:80])
    return all_items
