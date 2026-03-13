"""
config.py — Centralized configuration loader.

Loads all settings from .env via pydantic-settings.
Validates that critical secrets are present at startup.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Wallet ──────────────────────────────────────────────
    private_key: str = Field(
        ...,
        description="EVM private key (hex, no 0x prefix)",
    )
    proxy_wallet: str | None = Field(
        default=None,
        description="Optional: The Polymarket smart wallet (Safe) address for this EOA",
    )

    # ── Polymarket API Credentials ──────────────────────────
    poly_api_key: str | None = Field(
        default=None,
        description="Polymarket Level 2 API Auth Key",
    )
    poly_api_secret: str | None = Field(
        default=None,
        description="Polymarket Level 2 API Auth Secret",
    )
    poly_api_passphrase: str | None = Field(
        default=None,
        description="Polymarket Level 2 API Auth Passphrase",
    )

    # ── Polymarket ──────────────────────────────────────────
    polymarket_host: str = Field(
        default="https://clob.polymarket.com",
        description="Polymarket CLOB API base URL",
    )
    polymarket_chain_id: int = Field(default=137)

    # ── OpenAI ──────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o")

    # ── News API ────────────────────────────────────────────
    news_api_key: str = Field(default="", description="newsapi.org API key")
    tavily_api_key: str = Field(default="", description="tavily.com API key")

    # ── Risk management ─────────────────────────────────────
    max_risk_pct: float = Field(
        default=0.10,
        ge=0.001,
        le=0.20,
        description="Max fraction of bankroll per trade (Kelly cap)",
    )
    edge_threshold: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Min edge (|LLM_prob - market_price|) to trigger a trade",
    )
    kelly_multiplier: float = Field(
        default=0.25,
        ge=0.05,
        le=1.0,
        description="Fractional Kelly multiplier (0.25 = quarter-Kelly)",
    )
    min_bet_usd: float = Field(
        default=0.50,
        ge=0.10,
        description="Minimum dollar volume allowed for a trade",
    )
    polygon_fee: float = Field(
        default=0.0002,
        ge=0.0,
        le=0.01,
        description="Estimated Polygon tx fee per $1 bet",
    )

    # ── Gamma API ───────────────────────────────────────────
    gamma_api_url: str = Field(
        default="https://gamma-api.polymarket.com",
        description="Polymarket Gamma API base URL",
    )

    # ── Polygon RPC ─────────────────────────────────────────
    polygon_rpc_url: str = Field(default="https://polygon.drpc.org")
    usdc_contract: str = Field(
        default="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    )

    # ── Validators ──────────────────────────────────────────
    @field_validator("private_key")
    @classmethod
    def _strip_0x(cls, v: str) -> str:
        return v.removeprefix("0x").strip()

    @field_validator("polygon_rpc_url", mode="before")
    @classmethod
    def _fallback_rpc(cls, v: str) -> str:
        if not v or not v.strip():
            return "https://polygon.drpc.org"
        return v

def load_settings() -> Settings:
    """Load and validate settings.  Raises ``ValidationError`` early
    if any required variable is missing."""
    return Settings()  # type: ignore[call-arg]
