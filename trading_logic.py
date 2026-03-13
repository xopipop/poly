"""
trading_logic.py — Trading Logic & Risk Management Module.

Mathematical models:
  • Expected Value (EV) — determines whether a bet is profitable.
  • Kelly Criterion     — optimal bet sizing.
  • Fractional Kelly    — conservative sizing to smooth drawdowns.
  • Confidence scaling  — LLM confidence adjusts position size.

All formulas are documented inline with LaTeX-style comments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════


class Signal(str, Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    HOLD = "HOLD"


@dataclass(frozen=True)
class TradeSignal:
    """Immutable result of signal generation."""

    signal: Signal
    llm_probability: float      # LLM estimate, 0–1
    market_price: float         # current YES share price, 0–1
    edge: float                 # llm_prob - market_price
    ev: float                   # expected value per $1 bet
    confidence: float           # LLM meta-confidence, 0–1
    kelly_full: float           # full Kelly fraction
    kelly_fractional: float     # fractional Kelly (after multiplier)
    order_size_usd: float       # final bet size in USD
    reasoning: str              # LLM reasoning text


# ═══════════════════════════════════════════════════════════
#  1.  Expected Value (EV)
# ═══════════════════════════════════════════════════════════


def calculate_ev(
    p: float,
    market_price: float,
    *,
    fee: float = 0.0,
) -> float:
    """Compute the Expected Value of a $1 bet on a binary outcome.

    ── Formula ──────────────────────────────────────────────
    On Polymarket each YES share costs ``c`` (market_price) and
    pays $1 if the event resolves YES, $0 otherwise.

        EV = p × (1 - c) - (1 - p) × c - fee

    Which simplifies to:

        EV = p - c - fee

    ─────────────────────────────────────────────────────────

    Parameters
    ----------
    p : float
        Estimated probability of YES (0–1), from LLM.
    market_price : float
        Current YES share price on Polymarket (0–1).
    fee : float
        Estimated transaction cost per $1 bet (gas + protocol
        fees on Polygon, e.g. ~0.0002).

    Returns
    -------
    float
        Expected profit per $1 wagered.  Positive = +EV.
    """
    # EV = p × payout_if_win - (1-p) × cost - fee
    #    = p × (1 - c) - (1 - p) × c - fee
    #    = p - c - fee
    ev = p - market_price - fee

    logger.debug(
        "ev_calculated",
        p=f"{p:.4f}",
        market_price=f"{market_price:.4f}",
        fee=f"{fee:.6f}",
        ev=f"{ev:+.4f}",
        is_positive_ev=ev > 0,
    )
    return ev


# ═══════════════════════════════════════════════════════════
#  2.  Kelly Criterion
# ═══════════════════════════════════════════════════════════


def kelly_fraction(p: float, b: float) -> float:
    """Compute the full Kelly fraction for optimal bet sizing.

    ── Formula ──────────────────────────────────────────────
    The Kelly criterion for a binary bet with probability ``p``
    and decimal odds ``b`` (net payout per $1 risked):

        f* = (b × p - q) / b

    where  q = 1 - p.

    For a Polymarket YES share at price ``c``:
        • You pay ``c``, and receive $1 on a win.
        • Net odds:  b = (1 - c) / c
        • f* = (b × p - q) / b = p - q / b
    ─────────────────────────────────────────────────────────

    Parameters
    ----------
    p : float
        Estimated win probability (0–1).
    b : float
        Decimal odds = net payout per $1 risked.
        For Polymarket:  ``b = (1 - price) / price``.

    Returns
    -------
    float
        Optimal fraction of bankroll to bet (≥ 0).
        Returns 0 if the bet has no edge.
    """
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0

    q = 1.0 - p
    #  f* = (b × p - q) / b
    f = (b * p - q) / b

    return max(f, 0.0)


# ═══════════════════════════════════════════════════════════
#  3.  Position Sizing  (Fractional Kelly × Confidence)
# ═══════════════════════════════════════════════════════════


def compute_position_size(
    p: float,
    market_price: float,
    balance_usd: float,
    confidence: float,
    *,
    kelly_multiplier: float = 0.25,
    max_risk_pct: float = 0.02,
    min_bet_usd: float = 0.5,
) -> tuple[float, float, float]:
    """Calculate the dollar size of a bet with risk controls.

    ── Pipeline ─────────────────────────────────────────────
    1. Compute full Kelly fraction  ``f*``
    2. Apply Fractional Kelly:      ``f_frac = f* × kelly_multiplier``
       (reduces variance; typical values 0.10–0.25)
    3. Scale by LLM confidence:     ``f_adj  = f_frac × confidence``
       (if the model is uncertain, bet proportionally less)
    4. Apply hard cap:              ``f_final = min(f_adj, max_risk_pct)``
       (NEVER risk more than 2-3 % of bankroll on a single bet)
    5. Convert to dollars:          ``size = f_final × balance``
    6. Floor to $1.00 minimum.
    ─────────────────────────────────────────────────────────

    Parameters
    ----------
    p : float
        LLM win probability (0–1).
    market_price : float
        Share price for the direction we're betting (0–1).
    balance_usd : float
        Current wallet balance in USDC.
    confidence : float
        LLM confidence (0–1).
    kelly_multiplier : float
        Fractional Kelly scaling factor (default 0.25 = quarter-Kelly).
    max_risk_pct : float
        Hard cap as fraction of bankroll (default 0.02 = 2 %).
    min_bet_usd : float
        Minimum meaningful bet (below this → skip).

    Returns
    -------
    tuple[float, float, float]
        ``(full_kelly, fractional_kelly_adjusted, order_size_usd)``
    """
    # Step 1 — decimal odds for Kelly
    if market_price <= 0 or market_price >= 1:
        return 0.0, 0.0, 0.0

    b = (1.0 - market_price) / market_price  # net odds
    kf_full = kelly_fraction(p, b)

    # Step 2 — Fractional Kelly
    kf_frac = kf_full * kelly_multiplier

    # Step 3 — Scale by LLM confidence
    kf_adj = kf_frac * max(0.0, min(1.0, confidence))

    # Step 4 — Hard cap
    kf_capped = min(kf_adj, max_risk_pct)

    # Step 5 — Convert to dollars (round down to cents)
    order_size = math.floor(kf_capped * balance_usd * 100) / 100

    # Step 6 — Minimum bet floor
    if order_size < min_bet_usd:
        order_size = 0.0

    logger.info(
        "position_size",
        kelly_full=f"{kf_full:.4f}",
        kelly_frac=f"{kf_frac:.4f}",
        kelly_conf_adj=f"{kf_adj:.4f}",
        kelly_capped=f"{kf_capped:.4f}",
        confidence=f"{confidence:.2f}",
        order_usd=f"${order_size:.2f}",
        balance_usd=f"${balance_usd:.2f}",
    )

    return kf_full, kf_adj, order_size


# ═══════════════════════════════════════════════════════════
#  4.  Signal Generator  (combines EV + sizing)
# ═══════════════════════════════════════════════════════════


def generate_signal(
    llm_prob: float,
    market_price_yes: float,
    balance_usd: float,
    confidence: float,
    *,
    edge_threshold: float = 0.10,
    kelly_multiplier: float = 0.25,
    max_risk_pct: float = 0.02,
    min_bet_usd: float = 0.50,
    polygon_fee: float = 0.0002,
    reasoning: str = "",
) -> TradeSignal:
    """End-to-end decision: should we trade, in which direction, how much?

    Parameters
    ----------
    llm_prob : float
        LLM-estimated probability of YES (0–1).
    market_price_yes : float
        Current YES share price on Polymarket (0–1).
    balance_usd : float
        Available USDC balance.
    confidence : float
        LLM meta-confidence (0–1).
    edge_threshold : float
        Minimum |edge| to consider trading.
    kelly_multiplier : float
        Fractional Kelly multiplier (e.g. 0.25).
    max_risk_pct : float
        Hard cap per trade as fraction of bankroll.
    min_bet_usd : float
        Minimum meaningful bet (below this → skip).
    polygon_fee : float
        Estimated Polygon transaction cost per $1 bet.
    reasoning : str
        LLM reasoning (forwarded for logging).
    """

    edge = llm_prob - market_price_yes

    # ── 1.  Edge check ──────────────────────────────────────
    if abs(edge) < edge_threshold:
        logger.info(
            "signal_hold_no_edge",
            edge=f"{edge:+.2%}",
            threshold=f"{edge_threshold:.0%}",
        )
        return TradeSignal(
            signal=Signal.HOLD,
            llm_probability=llm_prob,
            market_price=market_price_yes,
            edge=edge,
            ev=0.0,
            confidence=confidence,
            kelly_full=0.0,
            kelly_fractional=0.0,
            order_size_usd=0.0,
            reasoning=f"Edge {edge:+.2%} below threshold {edge_threshold:.0%}. {reasoning}",
        )

    # ── 2.  Direction ───────────────────────────────────────
    if edge > 0:
        # LLM thinks YES is under-priced → buy YES
        direction = Signal.BUY_YES
        p = llm_prob
        price = market_price_yes
    else:
        # LLM thinks YES is over-priced → buy NO
        direction = Signal.BUY_NO
        p = 1.0 - llm_prob
        price = 1.0 - market_price_yes

    # ── 3.  Expected Value ──────────────────────────────────
    ev = calculate_ev(p, price, fee=polygon_fee)

    if ev <= 0:
        logger.info(
            "signal_hold_negative_ev",
            direction=direction.value,
            ev=f"{ev:+.4f}",
        )
        return TradeSignal(
            signal=Signal.HOLD,
            llm_probability=llm_prob,
            market_price=market_price_yes,
            edge=edge,
            ev=ev,
            confidence=confidence,
            kelly_full=0.0,
            kelly_fractional=0.0,
            order_size_usd=0.0,
            reasoning=f"Negative EV ({ev:+.4f}) after fees. {reasoning}",
        )

    # ── 4.  Position sizing ─────────────────────────────────
    kf_full, kf_adj, order_size = compute_position_size(
        p=p,
        market_price=price,
        balance_usd=balance_usd,
        confidence=confidence,
        kelly_multiplier=kelly_multiplier,
        max_risk_pct=max_risk_pct,
        min_bet_usd=min_bet_usd,
    )

    if order_size <= 0:
        return TradeSignal(
            signal=Signal.HOLD,
            llm_probability=llm_prob,
            market_price=market_price_yes,
            edge=edge,
            ev=ev,
            confidence=confidence,
            kelly_full=kf_full,
            kelly_fractional=kf_adj,
            order_size_usd=0.0,
            reasoning=f"Position too small after risk controls. {reasoning}",
        )

    # ── 5.  Emit actionable signal ──────────────────────────
    signal = TradeSignal(
        signal=direction,
        llm_probability=llm_prob,
        market_price=market_price_yes,
        edge=edge,
        ev=ev,
        confidence=confidence,
        kelly_full=kf_full,
        kelly_fractional=kf_adj,
        order_size_usd=order_size,
        reasoning=reasoning,
    )

    logger.info(
        "trade_signal",
        direction=direction.value,
        edge=f"{edge:+.2%}",
        ev=f"{ev:+.4f}",
        confidence=f"{confidence:.2f}",
        kelly_full=f"{kf_full:.4f}",
        kelly_adj=f"{kf_adj:.4f}",
        size_usd=f"${order_size:.2f}",
    )

    return signal
