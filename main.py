"""
main.py — Orchestrator for the Polymarket Trading Bot.

Full async pipeline per market:

    Gamma API price  ──┐
                       ├──▶  Trading Logic  ──▶  Execution
    Data Ingestion ──▶ Analysis ──┘
    (DuckDuckGo/News)  (LLM)

Supports:
  • Single market   (--condition-id + --question)
  • Multi-market    (--markets-file markets.json)
  • Continuous loop (--loop --interval 300)

Usage::

    # Single market, one-shot
    python main.py \\
        --condition-id 0xabc...def \\
        --token-id-yes 12345 \\
        --question "Will the Fed cut interest rates in May?" \\

    # Multi-market from JSON file, continuous
    python main.py --markets-file markets.json --loop --interval 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging
import logging.config

import aiohttp
import structlog

from analysis import AnalysisResult, LLMAnalyzer
from config import Settings, load_settings
from data_ingestion import collect_news
from execution import (
    AuthenticationError,
    InsufficientBalanceError,
    OrderSubmissionError,
    PolymarketExecutor,
)
from trading_logic import Signal, TradeSignal, generate_signal

# ═══════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processors": [
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=False),
            ],
        },
        "colored": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processors": [
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "polybot.log",
            "formatter": "plain",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    }
})

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
)

logger = structlog.get_logger("polybot")


# ═══════════════════════════════════════════════════════════
#  Market definition
# ═══════════════════════════════════════════════════════════


@dataclass
class MarketConfig:
    """One tracked Polymarket market."""

    condition_id: str
    token_id_yes: str
    token_id_no: str
    question: str


def load_markets_file(path: str) -> list[MarketConfig]:
    """Load a list of markets from a JSON file.

    Expected format::

        [
          {
            "condition_id": "0x...",
            "token_id_yes": "123...",
            "token_id_no":  "456...",
            "question": "Will X happen?"
          },
          ...
        ]
    """
    file = Path(path)
    if not file.exists():
        logger.error("markets_file_not_found", path=path)
        sys.exit(1)

    with file.open() as f:
        raw: list[dict[str, str]] = json.load(f)

    markets = [
        MarketConfig(
            condition_id=m["condition_id"],
            token_id_yes=m["token_id_yes"],
            token_id_no=m.get("token_id_no", ""),
            question=m["question"],
        )
        for m in raw
    ]
    logger.info("markets_loaded", count=len(markets), source=path)
    return markets


# ═══════════════════════════════════════════════════════════
#  Gamma API — Market Price Fetcher
# ═══════════════════════════════════════════════════════════


async def fetch_market_price(
    token_id_yes: str,
    clob_api_url: str = "https://clob.polymarket.com",
    *,
    session: aiohttp.ClientSession | None = None,
) -> dict[str, float]:
    """Query the Polymarket CLOB API midpoint for current market prices.

    Parameters
    ----------
    token_id_yes : str
        The market's YES outcome token ID.
    clob_api_url : str
        CLOB API base URL.

    Returns
    -------
    dict
        ``{"yes": float, "no": float}``  — current prices.
        Falls back to ``{"yes": 0.5, "no": 0.5}`` on error.
    """
    url = f"{clob_api_url}/midpoint"
    params = {"token_id": token_id_yes}
    fallback = {"yes": 0.5, "no": 0.5}

    own_session = session is None
    session = session or aiohttp.ClientSession()

    try:
        async with session.get(
            url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                if resp.status == 404:
                    logger.warning("clob_price_not_found", token_id=token_id_yes)
                else:
                    logger.error(
                        "clob_api_error",
                        status=resp.status,
                        body=body[:300],
                    )
                return fallback

            data = await resp.json()

        mid_price = float(data.get("mid", 0.5))
        
        # Round to 4 decimal places
        yes_price = round(mid_price, 4)
        no_price = round(1.0 - mid_price, 4)

        prices = {"yes": yes_price, "no": no_price}
        logger.info(
            "clob_price",
            token_id=token_id_yes[:16] + "...",
            yes=f"${yes_price:.4f}",
            no=f"${no_price:.4f}",
        )
        return prices

    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.error("clob_api_network_error", error=str(exc))
        return fallback
    except Exception as exc:
        logger.error("clob_api_unexpected_error", error=str(exc))
        return fallback
    finally:
        if own_session:
            await session.close()


# ═══════════════════════════════════════════════════════════
#  Graceful shutdown
# ═══════════════════════════════════════════════════════════

_shutdown_event = asyncio.Event()


def _handle_shutdown(sig: int, _frame: object) -> None:
    logger.info("shutdown_signal", signal=sig)
    _shutdown_event.set()


# ═══════════════════════════════════════════════════════════
#  Pipeline — single market
# ═══════════════════════════════════════════════════════════


async def run_pipeline(
    market: MarketConfig,
    settings: Settings,
    executor: PolymarketExecutor,
    analyzer: LLMAnalyzer,
) -> None:
    """One full cycle for a single market.

    Steps:
        1. Fetch current price (Gamma API)
        2. Collect news       (Data Ingestion)
        3. LLM analysis       (Analysis)
        4. EV + Kelly sizing  (Trading Logic)
        5. Execute order      (Execution)
    """

    logger.info(
        "pipeline_start",
        market=market.question[:60],
        condition_id=market.condition_id[:16] + "...",
    )

    # ── 1. Current market price via CLOB Midpoint ───────────
    logger.info("pipeline_step", step="1_clob_price")
    prices = await fetch_market_price(
        token_id_yes=market.token_id_yes,
    )
    market_price_yes = prices["yes"]

    logger.info(
        "market_price",
        yes=f"${market_price_yes:.4f}",
        no=f"${prices['no']:.4f}",
    )

    # ── 2. Data Ingestion ───────────────────────────────────
    logger.info("pipeline_step", step="2_data_ingestion")
    news = await collect_news(
        query=market.question,
        news_api_key=settings.news_api_key,
    )

    if not news:
        logger.warning(
            "no_news_found",
            question=market.question[:60],
        )
        return

    logger.info(
        "news_summary",
        total_articles=len(news),
        top_headlines=[n.title[:50] for n in news[:3]],
    )

    # ── 3. LLM Analysis ────────────────────────────────────
    logger.info("pipeline_step", step="3_llm_analysis")
    analysis: AnalysisResult = await analyzer.estimate_probability(
        market.question,
        news,
    )

    logger.info(
        "analysis_result",
        probability=f"{analysis.probability}%",
        confidence=f"{analysis.confidence:.2f}",
        reasoning=analysis.reasoning[:100],
    )

    # ── 4. Trading Logic: EV + Kelly ────────────────────────
    logger.info("pipeline_step", step="4_trading_logic")

    # Balance check
    balance = await executor.get_usdc_balance()
    logger.info("wallet_balance", usdc=f"${balance:.2f}")

    trade_signal: TradeSignal = generate_signal(
        llm_prob=analysis.probability_float,
        market_price_yes=market_price_yes,
        balance_usd=balance,
        confidence=analysis.confidence,
        edge_threshold=settings.edge_threshold,
        kelly_multiplier=settings.kelly_multiplier,
        max_risk_pct=settings.max_risk_pct,
        min_bet_usd=settings.min_bet_usd,
        polygon_fee=settings.polygon_fee,
        reasoning=analysis.reasoning,
    )

    logger.info(
        "signal_result",
        signal=trade_signal.signal.value,
        llm_prob=f"{trade_signal.llm_probability:.2%}",
        market_price=f"{trade_signal.market_price:.2%}",
        edge=f"{trade_signal.edge:+.2%}",
        ev=f"{trade_signal.ev:+.4f}",
        confidence=f"{trade_signal.confidence:.2f}",
        kelly_full=f"{trade_signal.kelly_full:.4f}",
        kelly_frac=f"{trade_signal.kelly_fractional:.4f}",
        order_usd=f"${trade_signal.order_size_usd:.2f}",
    )

    # ── 5. Execution ────────────────────────────────────────
    if trade_signal.signal == Signal.HOLD:
        logger.info("no_trade", reason=trade_signal.reasoning[:120])
        return

    logger.info("pipeline_step", step="5_execution")

    try:
        if trade_signal.signal == Signal.BUY_YES:
            result = await executor.buy_yes(
                token_id=market.token_id_yes,
                size_usd=trade_signal.order_size_usd,
            )
        elif trade_signal.signal == Signal.BUY_NO:
            if not market.token_id_no:
                logger.warning(
                    "buy_no_skip",
                    reason="No token_id_no configured for this market",
                )
                return
            result = await executor.buy_yes(
                token_id=market.token_id_no,
                size_usd=trade_signal.order_size_usd,
            )
        else:
            return

        logger.info(
            "trade_executed",
            order_id=result.order_id,
            success=result.success,
            signal=trade_signal.signal.value,
            size_usd=f"${trade_signal.order_size_usd:.2f}",
        )

    except InsufficientBalanceError as exc:
        logger.error("trade_aborted_balance", reason=str(exc))
    except OrderSubmissionError as exc:
        logger.error("trade_aborted_order", reason=str(exc))
    except Exception as exc:
        logger.error("trade_unexpected_error", error=str(exc))

    logger.info(
        "pipeline_complete",
        market=market.question[:60],
    )


# ── Render Free Tier Health Check Server ────────────────────

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is alive")

    def log_message(self, format, *args):
        # Silence standard HTTP logs to keep polybot.log clean
        return

def run_health_server(port: int):
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    logger.info("health_server_started", port=port)
    server.serve_forever()

# ═══════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════


async def main(args: argparse.Namespace) -> None:
    """Boot all modules and run the pipeline loop."""

    # Start health check server for Render Free Tier if PORT is provided
    port = os.environ.get("PORT")
    if port:
        threading.Thread(target=run_health_server, args=(int(port),), daemon=True).start()

    # ── Load config ─────────────────────────────────────────
    try:
        settings = load_settings()
    except Exception as exc:
        print(f"\n  Configuration error: {exc}\n", file=sys.stderr)
        print(
            "Copy .env.example -> .env and fill in your real values.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Build market list ───────────────────────────────────
    if args.markets_file:
        markets = load_markets_file(args.markets_file)
    else:
        # Single market from CLI args
        if not args.condition_id or not args.question:
            print(
                "Error: provide either --markets-file or "
                "(--condition-id + --token-id-yes + --question).",
                file=sys.stderr,
            )
            sys.exit(1)
        markets = [
            MarketConfig(
                condition_id=args.condition_id,
                token_id_yes=args.token_id_yes or "",
                token_id_no=args.token_id_no or "",
                question=args.question,
            )
        ]

    # ── Initialize modules ──────────────────────────────────
    analyzer = LLMAnalyzer(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )

    logger.info("executor_init", rpc_url=settings.polygon_rpc_url)
    executor = PolymarketExecutor(
        private_key=settings.private_key,
        host=settings.polymarket_host,
        chain_id=settings.polymarket_chain_id,
        polygon_rpc=settings.polygon_rpc_url,
        usdc_address=settings.usdc_contract,
        proxy_wallet=settings.proxy_wallet,
        poly_api_key=settings.poly_api_key,
        poly_api_secret=settings.poly_api_secret,
        poly_api_passphrase=settings.poly_api_passphrase,
    )

    try:
        await executor.initialize()
    except AuthenticationError as exc:
        logger.error("startup_failed", reason=str(exc))
        sys.exit(1)

    logger.info(
        "bot_ready",
        markets=len(markets),
        loop=args.loop,
        interval=args.interval,
        kelly_mult=settings.kelly_multiplier,
        max_risk=f"{settings.max_risk_pct:.1%}",
    )

    # ── Main loop ───────────────────────────────────────────
    cycle = 0
    while True:
        cycle += 1
        logger.info("cycle_start", cycle=cycle, markets=len(markets))

        for market in markets:
            if _shutdown_event.is_set():
                break
            try:
                await run_pipeline(
                    market=market,
                    settings=settings,
                    executor=executor,
                    analyzer=analyzer,
                )
            except KeyboardInterrupt:
                _shutdown_event.set()
                break
            except Exception as exc:
                logger.error(
                    "pipeline_error",
                    market=market.question[:60],
                    error=str(exc),
                )

        logger.info("cycle_complete", cycle=cycle)

        if not args.loop or _shutdown_event.is_set():
            break

        # Wait for next cycle or shutdown
        logger.info("sleeping", seconds=args.interval)
        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=args.interval,
            )
            break  # shutdown requested
        except asyncio.TimeoutError:
            continue  # next cycle

    logger.info("bot_stopped", total_cycles=cycle)


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polymarket Trading Bot — OSINT + LLM + Kelly + CLOB",
    )

    # Multi-market mode
    parser.add_argument(
        "--markets-file",
        default=None,
        help="Path to JSON file with market definitions",
    )

    # Single-market mode
    parser.add_argument(
        "--condition-id",
        default=None,
        help="Polymarket condition ID (hex string)",
    )
    parser.add_argument(
        "--token-id-yes",
        default=None,
        help="YES outcome token ID",
    )
    parser.add_argument(
        "--token-id-no",
        default=None,
        help="NO outcome token ID",
    )
    parser.add_argument(
        "--question",
        default=None,
        help='Market question, e.g. "Will BTC hit $100k by June 2025?"',
    )

    # Loop mode
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously instead of one-shot",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between cycles in loop mode (default: 300)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    asyncio.run(main(cli()))
