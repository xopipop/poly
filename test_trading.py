"""
test_trading.py — Unit tests for trading_logic module.

Run:  python test_trading.py

Tests the mathematical core:
  • calculate_ev — known inputs and edge cases
  • kelly_fraction — standard and edge cases
  • compute_position_size — full pipeline with fractional Kelly × confidence
  • generate_signal — integration: direction, EV gate, sizing
"""

from __future__ import annotations

import sys

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

_PASS = "PASS"
_FAIL = "FAIL"

passed = failed = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    tag = _PASS if ok else _FAIL
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}]  {name}{suffix}")


def approx(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) < tol


# ────────────────────────────────────────────────────────────
# 1.  calculate_ev
# ────────────────────────────────────────────────────────────


def test_calculate_ev() -> None:
    print("\n-- calculate_ev ---------------------------------")
    from trading_logic import calculate_ev

    # p=0.75, price=0.55 => EV = 0.75 - 0.55 = 0.20
    ev = calculate_ev(0.75, 0.55)
    report("p=0.75, price=0.55 => EV~0.20", approx(ev, 0.20), f"ev={ev:.4f}")

    # With fee
    ev2 = calculate_ev(0.75, 0.55, fee=0.001)
    report("same with fee=0.001 => EV~0.199", approx(ev2, 0.199), f"ev={ev2:.4f}")

    # Negative EV: p=0.40, price=0.55 => EV = -0.15
    ev3 = calculate_ev(0.40, 0.55)
    report("p=0.40, price=0.55 => EV~-0.15", approx(ev3, -0.15), f"ev={ev3:.4f}")

    # Break-even: p=0.55, price=0.55 => EV~0
    ev4 = calculate_ev(0.55, 0.55)
    report("p=price => EV~0", approx(ev4, 0.0), f"ev={ev4:.4f}")


# ────────────────────────────────────────────────────────────
# 2.  kelly_fraction
# ────────────────────────────────────────────────────────────


def test_kelly_fraction() -> None:
    print("\n-- kelly_fraction --------------------------------")
    from trading_logic import kelly_fraction

    # p=0.60, odds=1.0 (even money) => f* = (1*0.6 - 0.4)/1 = 0.20
    kf = kelly_fraction(0.60, 1.0)
    report("p=0.60, b=1.0 => f*=0.20", approx(kf, 0.20), f"kf={kf:.4f}")

    # p=0.50, b=1.0 => f*=0 (no edge)
    kf2 = kelly_fraction(0.50, 1.0)
    report("p=0.50, b=1.0 => f*=0", approx(kf2, 0.0), f"kf={kf2:.4f}")

    # Edge case: p=0, should return 0
    kf3 = kelly_fraction(0.0, 1.0)
    report("p=0 => f*=0", kf3 == 0.0)

    # p=0.75, price=0.55 => b = 0.45/0.55 ~ 0.8182
    # f* = (0.8182 * 0.75 - 0.25) / 0.8182 ~ 0.4444
    b = 0.45 / 0.55
    kf4 = kelly_fraction(0.75, b)
    report("p=0.75, price=0.55 => f*~0.4444", approx(kf4, 0.4444, tol=0.001), f"kf={kf4:.4f}")


# ────────────────────────────────────────────────────────────
# 3.  compute_position_size
# ────────────────────────────────────────────────────────────


def test_compute_position_size() -> None:
    print("\n-- compute_position_size -------------------------")
    from trading_logic import compute_position_size

    # p=0.75, price=0.55, balance=1000, confidence=0.80, kelly_mult=0.25
    kf_full, kf_adj, size = compute_position_size(
        p=0.75,
        market_price=0.55,
        balance_usd=1000.0,
        confidence=0.80,
        kelly_multiplier=0.25,
        max_risk_pct=0.02,
    )

    report("full Kelly > 0", kf_full > 0, f"kf_full={kf_full:.4f}")
    report("adjusted Kelly < full Kelly", kf_adj < kf_full, f"kf_adj={kf_adj:.4f}")
    report("order_size > 0", size > 0, f"size=${size:.2f}")
    report("order_size <= 2% of balance", size <= 20.0, f"size=${size:.2f}")

    # Low confidence should reduce size (use higher cap to avoid masking)
    _, _, size_high = compute_position_size(
        p=0.75,
        market_price=0.55,
        balance_usd=1000.0,
        confidence=0.80,
        kelly_multiplier=0.25,
        max_risk_pct=0.20,
    )
    _, _, size_low_conf = compute_position_size(
        p=0.75,
        market_price=0.55,
        balance_usd=1000.0,
        confidence=0.20,
        kelly_multiplier=0.25,
        max_risk_pct=0.20,
    )
    report(
        "low confidence => smaller size",
        size_low_conf < size_high,
        f"high_conf=${size_high:.2f}, low_conf=${size_low_conf:.2f}",
    )

    # No edge: p=0.50, price=0.50 => Kelly=0 => size=0
    _, _, size_no_edge = compute_position_size(
        p=0.50,
        market_price=0.50,
        balance_usd=1000.0,
        confidence=0.90,
    )
    report("no edge => size=0", size_no_edge == 0.0, f"size=${size_no_edge:.2f}")


# ────────────────────────────────────────────────────────────
# 4.  generate_signal
# ────────────────────────────────────────────────────────────


def test_generate_signal() -> None:
    print("\n-- generate_signal -------------------------------")
    from trading_logic import Signal, generate_signal

    # Strong edge: LLM=0.75, market=0.55, balance=5000
    sig = generate_signal(
        llm_prob=0.75,
        market_price_yes=0.55,
        balance_usd=5000.0,
        confidence=0.85,
        edge_threshold=0.10,
        kelly_multiplier=0.25,
        max_risk_pct=0.02,
    )
    report("BUY_YES on positive edge", sig.signal == Signal.BUY_YES, sig.signal.value)
    report("EV > 0", sig.ev > 0, f"ev={sig.ev:+.4f}")
    report("order_size > 0", sig.order_size_usd > 0, f"${sig.order_size_usd:.2f}")
    report("order_size <= 2% cap", sig.order_size_usd <= 100.0)

    # Negative edge: LLM=0.35, market=0.55
    sig2 = generate_signal(
        llm_prob=0.35,
        market_price_yes=0.55,
        balance_usd=5000.0,
        confidence=0.80,
        edge_threshold=0.10,
    )
    report("BUY_NO on negative edge", sig2.signal == Signal.BUY_NO, sig2.signal.value)

    # No edge: LLM~market
    sig3 = generate_signal(
        llm_prob=0.56,
        market_price_yes=0.55,
        balance_usd=5000.0,
        confidence=0.80,
        edge_threshold=0.10,
    )
    report("HOLD when edge < threshold", sig3.signal == Signal.HOLD, sig3.signal.value)

    # Low balance: even with edge, too small
    sig4 = generate_signal(
        llm_prob=0.75,
        market_price_yes=0.55,
        balance_usd=10.0,
        confidence=0.50,
        kelly_multiplier=0.10,
        max_risk_pct=0.02,
    )
    report("HOLD on tiny balance", sig4.signal == Signal.HOLD, f"size=${sig4.order_size_usd:.2f}")


# ────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 52)
    print("  Trading Logic — Unit Tests")
    print("=" * 52)

    test_calculate_ev()
    test_kelly_fraction()
    test_compute_position_size()
    test_generate_signal()

    print(f"\n{'=' * 52}")
    total = passed + failed
    print(f"  Results: {passed} passed, {failed} failed  (total {total})")
    print("=" * 52)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
