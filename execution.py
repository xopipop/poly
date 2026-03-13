"""
execution.py — Execution Module (Polymarket CLOB API).

Responsibilities:
  1. L1 → L2 authentication (EIP-712 signature → API credentials).
  2. Orderbook retrieval for a given token_id.
  3. USDC balance check on Polygon via web3.
  4. Signed limit-order creation and submission.

All operations include structured error handling and logging.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import structlog
from eth_account import Account
from eth_account.signers.local import LocalAccount
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    OrderArgs,
    OrderType,
)
from py_clob_client.constants import POLYGON
from web3 import Web3

logger = structlog.get_logger(__name__)

# ── Custom exceptions ───────────────────────────────────────


class InsufficientBalanceError(Exception):
    """Raised when the wallet does not have enough USDC for the order."""


class OrderSubmissionError(Exception):
    """Raised when the CLOB API rejects or fails to process an order."""


class AuthenticationError(Exception):
    """Raised when L1 or L2 authentication fails."""


# ── Data structures ─────────────────────────────────────────


@dataclass(frozen=True)
class OrderbookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class Orderbook:
    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]
    best_bid: float
    best_ask: float
    spread: float
    token_id: str


@dataclass(frozen=True)
class OrderResult:
    success: bool
    order_id: str
    message: str
    details: dict[str, Any]


# ── ERC-20 minimal ABI for USDC balance query ──────────────

USDC_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]


# ═══════════════════════════════════════════════════════════
#  PolymarketExecutor  — the main class
# ═══════════════════════════════════════════════════════════


class PolymarketExecutor:
    """Handles all interactions with the Polymarket CLOB API.

    Usage::

        executor = PolymarketExecutor(
            private_key="abcdef...",
            host="https://clob.polymarket.com",
            polygon_rpc="https://polygon.llamarpc.com",
            usdc_address="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        )
        await executor.initialize()          # L1 → L2 auth
        book = await executor.get_orderbook(token_id)
        result = await executor.place_limit_order(...)
    """

    def __init__(
        self,
        private_key: str,
        host: str = "https://clob.polymarket.com",
        chain_id: int = POLYGON,
        polygon_rpc: str = "https://polygon.llamarpc.com",
        usdc_address: str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        proxy_wallet: str | None = None,
        poly_api_key: str | None = None,
        poly_api_secret: str | None = None,
        poly_api_passphrase: str | None = None,
    ) -> None:
        self._private_key = private_key
        self._host = host
        self._chain_id = chain_id
        self._polygon_rpc = polygon_rpc
        self._usdc_address = usdc_address
        self._proxy_wallet = proxy_wallet

        # Derived at init
        self._account: LocalAccount = Account.from_key(private_key)
        raw_address = self._proxy_wallet if self._proxy_wallet else self._account.address
        self._address: str = Web3.to_checksum_address(raw_address)
        
        if self._proxy_wallet:
            logger.info("executor_wallet_proxy", funder=self._proxy_wallet, signer=self._account.address)
        else:
            logger.info("executor_wallet", address=self._address)

        # Will be populated by initialize() or pre-filled if provided
        self._client: ClobClient | None = None
        self._api_creds: ApiCreds | None = None

        if poly_api_key and poly_api_secret and poly_api_passphrase:
            self._api_creds = ApiCreds(
                api_key=poly_api_key,
                api_secret=poly_api_secret,
                api_passphrase=poly_api_passphrase,
            )
            logger.info("api_creds_preloaded")

        # web3 for balance checks
        self._w3 = Web3(Web3.HTTPProvider(polygon_rpc))
        self._usdc_contract = self._w3.eth.contract(
            address=Web3.to_checksum_address(usdc_address),
            abi=USDC_ABI,
        )

    # ── 1. Authentication ───────────────────────────────────

    async def initialize(self) -> None:
        """Perform L1 → L2 authentication and cache API credentials,
        unless they were explicitly provided in the constructor.
        """
        logger.info("auth_start", address=self._address)
        try:
            if self._api_creds is None:
                # Need to derive credentials from L1 signature
                client = ClobClient(
                    self._host,
                    key=self._private_key,
                    chain_id=self._chain_id,
                    funder=self._proxy_wallet,
                )

                # Derive L2 API credentials
                api_creds = client.derive_api_key()
                if not api_creds or not api_creds.api_key:
                    raise AuthenticationError("derive_api_key returned empty credentials")
                
                self._api_creds = api_creds

            # Create or Re-create client with full L2 credentials
            self._client = ClobClient(
                self._host,
                key=self._private_key,
                chain_id=self._chain_id,
                creds=self._api_creds,
                funder=self._proxy_wallet,
            )

            logger.info(
                "auth_success",
                api_key=self._api_creds.api_key[:8] + "...",
                address=self._address,
            )

        except AuthenticationError:
            raise
        except Exception as exc:
            logger.error("auth_failed", error=str(exc))
            raise AuthenticationError(f"Authentication failed: {exc}") from exc

    def _ensure_initialized(self) -> ClobClient:
        if self._client is None:
            raise AuthenticationError(
                "Executor not initialized. Call await executor.initialize() first."
            )
        return self._client

    # ── 2. Orderbook retrieval ──────────────────────────────

    async def get_orderbook(self, token_id: str) -> Orderbook:
        """Fetch the L2 orderbook for a specific outcome token.

        Parameters
        ----------
        token_id : str
            The Polymarket token ID (long numeric string) for the YES or NO share.

        Returns
        -------
        Orderbook
            Parsed orderbook with best bid/ask and spread.
        """
        client = self._ensure_initialized()

        logger.info("orderbook_fetch", token_id=token_id[:16] + "...")

        try:
            # py_clob_client wraps GET /book?token_id=...
            raw_book = await asyncio.to_thread(client.get_order_book, token_id)

            bids = [
                OrderbookLevel(price=float(level.price), size=float(level.size))
                for level in (raw_book.bids or [])
            ]
            asks = [
                OrderbookLevel(price=float(level.price), size=float(level.size))
                for level in (raw_book.asks or [])
            ]

            # Sort: bids descending, asks ascending
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            best_bid = bids[0].price if bids else 0.0
            best_ask = asks[0].price if asks else 1.0
            spread = best_ask - best_bid

            book = Orderbook(
                bids=bids,
                asks=asks,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                token_id=token_id,
            )

            logger.info(
                "orderbook_received",
                best_bid=best_bid,
                best_ask=best_ask,
                spread=f"{spread:.4f}",
                bid_levels=len(bids),
                ask_levels=len(asks),
            )
            return book

        except Exception as exc:
            logger.error("orderbook_error", token_id=token_id[:16], error=str(exc))
            raise

    # ── 3. USDC balance on Polygon ──────────────────────────

    async def get_usdc_balance(self) -> float:
        """Query on-chain USDC balance for the wallet.

        Returns the balance in human-readable units (e.g. 150.50 USDC).
        """
        try:
            raw_balance: int = await asyncio.to_thread(
                self._usdc_contract.functions.balanceOf(self._address).call
            )
            decimals: int = await asyncio.to_thread(
                self._usdc_contract.functions.decimals().call
            )
            balance = float(Decimal(raw_balance) / Decimal(10**decimals))

            logger.info("usdc_balance", balance=balance, address=self._address)
            return balance

        except Exception as exc:
            logger.error("balance_check_failed", error=str(exc))
            raise

    # ── 4. Place a signed limit order ───────────────────────

    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        *,
        check_balance: bool = True,
    ) -> OrderResult:
        """Create, sign (EIP-712), and submit a limit order.

        Parameters
        ----------
        token_id : str
            Token ID for the YES or NO outcome share.
        side : str
            ``"BUY"`` or ``"SELL"``.
        price : float
            Limit price (0.01 – 0.99).
        size : float
            Number of shares (≈ dollars at price=1.0).
        check_balance : bool
            If True, verify USDC balance before submitting.

        Returns
        -------
        OrderResult
            Success/failure with order ID and API response.

        Raises
        ------
        InsufficientBalanceError
            If wallet USDC balance < required cost.
        OrderSubmissionError
            If API rejects the order.
        AuthenticationError
            If executor is not initialized.
        """
        client = self._ensure_initialized()

        # ── Validate inputs ─────────────────────────────────
        if not (0.01 <= price <= 0.99):
            raise ValueError(f"Price must be 0.01–0.99, got {price}")
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        if side.upper() not in ("BUY", "SELL"):
            raise ValueError(f"Side must be BUY or SELL, got {side}")

        side_upper = side.upper()
        cost_usd = price * size if side_upper == "BUY" else 0.0

        logger.info(
            "order_prepare",
            token_id=token_id[:16] + "...",
            side=side_upper,
            price=price,
            size=size,
            cost_usd=f"{cost_usd:.2f}",
        )

        # ── Balance check ───────────────────────────────────
        if check_balance and side_upper == "BUY":
            balance = await self.get_usdc_balance()
            if balance < cost_usd:
                raise InsufficientBalanceError(
                    f"Insufficient USDC: need ${cost_usd:.2f}, have ${balance:.2f}"
                )

        # ── Build & sign order ──────────────────────────────
        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side_upper,
            )

            # create_and_post_order:
            #   1. Builds the EIP-712 typed-data struct.
            #   2. Signs it with the private key (L1).
            #   3. Sends the signed payload to CLOB API (L2 auth via HMAC headers).
            response = await asyncio.to_thread(
                client.create_and_post_order, order_args
            )

            # ── Parse response ──────────────────────────────
            if isinstance(response, dict):
                success = response.get("success", False)
                order_id = response.get("orderID", response.get("order_id", ""))
                error_msg = response.get("errorMsg", response.get("error", ""))
            else:
                # Some SDK versions return an object
                success = getattr(response, "success", False)
                order_id = getattr(response, "orderID", "") or getattr(
                    response, "order_id", ""
                )
                error_msg = getattr(response, "errorMsg", "") or getattr(
                    response, "error", ""
                )

            if not success:
                raise OrderSubmissionError(
                    f"Order rejected by CLOB API: {error_msg or 'unknown error'}"
                )

            result = OrderResult(
                success=True,
                order_id=str(order_id),
                message="Order submitted successfully",
                details=response if isinstance(response, dict) else {"raw": str(response)},
            )

            logger.info(
                "order_submitted",
                order_id=result.order_id,
                side=side_upper,
                price=price,
                size=size,
            )
            return result

        except (InsufficientBalanceError, OrderSubmissionError):
            raise
        except Exception as exc:
            logger.error(
                "order_failed",
                error=str(exc),
                side=side_upper,
                price=price,
                size=size,
            )
            raise OrderSubmissionError(f"Order submission failed: {exc}") from exc

    # ── 5. Helper: get mid-price for a token ────────────────

    async def get_mid_price(self, token_id: str) -> float:
        """Return the mid-price between best bid and best ask."""
        book = await self.get_orderbook(token_id)
        mid = (book.best_bid + book.best_ask) / 2.0
        return round(mid, 4)

    # ── 6. Convenience: buy YES shares ──────────────────────

    async def buy_yes(
        self,
        token_id: str,
        size_usd: float,
        *,
        price: float | None = None,
        slippage: float = 0.02,
    ) -> OrderResult:
        """Buy YES shares at best ask + slippage or at a specific price.

        Parameters
        ----------
        token_id : str
            YES token ID.
        size_usd : float
            Dollar amount to spend.
        price : float | None
            If None, uses best_ask + slippage as the limit price.
        slippage : float
            Slippage tolerance added to best ask (default 2%).
        """
        if price is None:
            book = await self.get_orderbook(token_id)
            price = min(book.best_ask + slippage, 0.99)

        # shares = dollars / price
        size = round(size_usd / price, 2)

        return await self.place_limit_order(
            token_id=token_id,
            side="BUY",
            price=round(price, 2),
            size=size,
        )
