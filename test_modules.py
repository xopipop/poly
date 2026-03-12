"""
test_modules.py — Smoke tests for data_ingestion and analysis modules.

Run:  python test_modules.py

• DuckDuckGo search test runs always (no API key required).
• OpenAI end-to-end test runs only when OPENAI_API_KEY is set.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

_PASS = "\033[92m✓ PASS\033[0m"
_FAIL = "\033[91m✗ FAIL\033[0m"
_SKIP = "\033[93m⊘ SKIP\033[0m"

passed = failed = skipped = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    tag = _PASS if ok else _FAIL
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag}  {name}{suffix}")


def skip(name: str, reason: str = "") -> None:
    global skipped
    skipped += 1
    suffix = f"  ({reason})" if reason else ""
    print(f"  {_SKIP}  {name}{suffix}")


# ────────────────────────────────────────────────────────────
# 1. Data Ingestion — DuckDuckGo
# ────────────────────────────────────────────────────────────


async def test_duckduckgo() -> None:
    print("\n── Data Ingestion: DuckDuckGo ─────────────────────")

    from data_ingestion import NewsItem, fetch_duckduckgo

    items = await fetch_duckduckgo("Federal Reserve interest rates", max_results=5)

    report(
        "fetch_duckduckgo returns list",
        isinstance(items, list),
    )
    report(
        "at least 1 result",
        len(items) >= 1,
        detail=f"got {len(items)}",
    )

    if items:
        first = items[0]
        report(
            "items are NewsItem",
            isinstance(first, NewsItem),
        )
        report(
            "title is non-empty",
            bool(first.title.strip()),
            detail=repr(first.title[:60]),
        )


# ────────────────────────────────────────────────────────────
# 2. Data Ingestion — collect_news unified
# ────────────────────────────────────────────────────────────


async def test_collect_news() -> None:
    print("\n── Data Ingestion: collect_news ────────────────────")

    from data_ingestion import collect_news

    items = await collect_news(
        "Will the Fed cut interest rates in May?",
        news_api_key="",  # no key — only DuckDuckGo used
        max_ddg_results=5,
    )

    report(
        "collect_news returns list",
        isinstance(items, list),
    )
    report(
        "at least 1 result (DuckDuckGo only)",
        len(items) >= 1,
        detail=f"got {len(items)}",
    )


# ────────────────────────────────────────────────────────────
# 3. Analysis — JSON parsing (unit test, no API call)
# ────────────────────────────────────────────────────────────


def test_parse_response() -> None:
    print("\n── Analysis: _parse_response ───────────────────────")

    from analysis import AnalysisResult, LLMAnalyzer

    # Valid response
    valid_json = json.dumps({
        "reasoning": "Strong macro signals point to a rate cut.",
        "probability": 72,
        "confidence": 0.85,
    })
    result = LLMAnalyzer._parse_response(valid_json)
    report("valid JSON → AnalysisResult", isinstance(result, AnalysisResult))
    report("probability == 72", result.probability == 72)
    report("confidence == 0.85", abs(result.confidence - 0.85) < 1e-6)
    report("reasoning preserved", "rate cut" in result.reasoning.lower())

    # Out-of-range values clamped
    clamped = LLMAnalyzer._parse_response(json.dumps({
        "reasoning": "test",
        "probability": 150,
        "confidence": 2.5,
    }))
    report("probability clamped to 100", clamped.probability == 100)
    report("confidence clamped to 1.0", clamped.confidence == 1.0)

    # Malformed JSON fallback
    bad = LLMAnalyzer._parse_response("this is not json at all")
    report("malformed JSON → prob 50", bad.probability == 50)
    report("malformed JSON → conf 0.0", bad.confidence == 0.0)


# ────────────────────────────────────────────────────────────
# 4. Analysis — live OpenAI call (optional)
# ────────────────────────────────────────────────────────────


async def test_openai_live() -> None:
    print("\n── Analysis: OpenAI live call ──────────────────────")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        skip("live OpenAI call", reason="OPENAI_API_KEY not set")
        return

    from analysis import AnalysisResult, LLMAnalyzer
    from data_ingestion import NewsItem

    analyzer = LLMAnalyzer(api_key=api_key, model=os.getenv("OPENAI_MODEL", "gpt-4o"))

    mock_news = [
        NewsItem(
            title="Fed officials signal openness to rate cut amid slowing economy",
            summary="Multiple Federal Reserve governors indicated in recent speeches "
                    "that a rate cut could be on the table if economic data continues "
                    "to weaken.",
            source="Reuters",
        ),
        NewsItem(
            title="US inflation falls to 2.1 % in latest CPI report",
            summary="Consumer prices rose just 2.1 % year-over-year, the lowest "
                    "reading in three years, bolstering expectations of monetary "
                    "easing.",
            source="Bloomberg",
        ),
    ]

    result = await analyzer.estimate_probability(
        market_question="Will the Fed cut interest rates in May?",
        news_items=mock_news,
    )

    report("returns AnalysisResult", isinstance(result, AnalysisResult))
    report(
        "probability in [0, 100]",
        0 <= result.probability <= 100,
        detail=f"got {result.probability}",
    )
    report(
        "confidence in [0.0, 1.0]",
        0.0 <= result.confidence <= 1.0,
        detail=f"got {result.confidence}",
    )
    report(
        "reasoning non-empty",
        bool(result.reasoning.strip()),
        detail=repr(result.reasoning[:80]),
    )

    print(f"\n  📊  Full LLM response:\n{json.dumps(result.as_dict(), indent=2, ensure_ascii=False)}")


# ────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 56)
    print("  Polymarket Bot — Module Tests")
    print("=" * 56)

    await test_duckduckgo()
    await test_collect_news()
    test_parse_response()
    await test_openai_live()

    print(f"\n{'=' * 56}")
    total = passed + failed + skipped
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped  (total {total})")
    print("=" * 56)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
