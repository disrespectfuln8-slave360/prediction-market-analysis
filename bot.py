"""Kalshi trading bot entry point.

Usage:
    uv run bot.py [--interval SECONDS] [--max-positions N] [--edge PP] [--spend CENTS]

Requires KALSHI_API_KEY in .env or environment.
Requires historical trade data in data/kalshi/ (run `uv run main.py index` first).
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from src.trading.kalshi.bot import TradingBot
from src.trading.kalshi.strategy import MispricingStrategy
from src.trading.kalshi.trading_client import KalshiTradingClient


def main():
    parser = argparse.ArgumentParser(description="Kalshi mispricing bot")
    parser.add_argument("--interval", type=int, default=None, help="Poll interval in seconds (default: 30 for --btc, 60 otherwise)")
    parser.add_argument("--max-positions", type=int, default=20, help="Max concurrent open positions (default: 20)")
    parser.add_argument("--edge", type=float, default=5.0, help="Minimum edge in percentage points (default: 5.0)")
    parser.add_argument("--spend", type=int, default=500, help="Max spend per bet in cents (default: 500 = $5)")
    parser.add_argument("--min-trades", type=int, default=50, help="Min historical trades at a price to trust the data (default: 50)")
    parser.add_argument("--btc", action="store_true", help="Only trade BTC markets (fast-paced short-duration bets)")
    parser.add_argument("--filter", type=str, default=None, help="Only trade markets whose ticker contains this string (e.g. ETH, SOL)")
    args = parser.parse_args()

    api_key = os.environ.get("KALSHI_API_KEY")
    if not api_key:
        raise SystemExit("KALSHI_API_KEY is not set. Add it to your .env file.")

    ticker_filter = "BTC" if args.btc else args.filter
    poll_interval = args.interval or (30 if args.btc else 60)

    strategy = MispricingStrategy(
        min_edge_pp=args.edge,
        max_spend_cents=args.spend,
        min_historical_trades=args.min_trades,
        ticker_filter=ticker_filter,
    )

    with KalshiTradingClient(api_key=api_key) as client:
        bot = TradingBot(
            client=client,
            strategy=strategy,
            poll_interval=poll_interval,
            max_open_positions=args.max_positions,
        )
        bot.run()


if __name__ == "__main__":
    main()
