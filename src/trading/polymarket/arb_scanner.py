"""Cross-platform arbitrage scanner: Kalshi <-> Polymarket.

How it works
------------
For each Kalshi sports market (ATP, WTA, MLB) that has active prices:
  1. Find the matching Polymarket market by team/player name.
  2. Compare: kalshi_yes_ask + polymarket_no_ask (= 100 - poly_yes_bid)
     If combined < MIN_COMBINED_CENTS, the gap is locked-in profit.
  3. Execute the Kalshi leg immediately (we have auth).
  4. Execute the Polymarket leg immediately too, if a Polymarket trading
     client was supplied (see run_arb_loop's poly_trading_client arg).

Why 95c threshold
-----------------
Polymarket charges ~2% maker fee (2c per 100c). We budget 5c margin to
cover both sides' fees plus slippage risk on the slower Polymarket leg.

Execution model
---------------
This scanner runs in its own thread alongside the two existing Kalshi bots.
When it finds a gap it places both legs directly: the Kalshi leg via
kalshi_client, the Polymarket leg via poly_trading_client (when provided).
If the Kalshi leg fills but the Polymarket leg then fails, you're left
holding a naked, undhedged Kalshi position instead of a guaranteed-profit
pair — this is logged as CRITICAL since nothing here can undo an already
-filled taker order. A signal file is still written as a manual-fallback
audit record for every attempted Polymarket leg.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from src.trading.polymarket.client import PolymarketClient, PolyMarket
from src.trading.polymarket.matcher import find_polymarket_match

logger = logging.getLogger(__name__)

# Must have this much free profit after both legs' costs to execute.
MIN_COMBINED_CENTS = 95  # buy both sides for < 95¢ → at least 5¢ profit
SCAN_INTERVAL_SECONDS = 300  # check every 5 minutes

# Manual-fallback audit record of every attempted Polymarket leg, kept
# inside the repo's own data dir (not some other project's directory).
ARB_SIGNAL_FILE = Path(
    os.environ.get("POLYMARKET_ARB_SIGNAL_FILE", "data/polymarket/arb_signals.json")
)

# Persisted set of Kalshi tickers already executed this session.
# Survives process restarts so we never double-execute the same arb leg.
_EXECUTED_FILE = Path("data/kalshi/arb_executed.json")


def _load_executed() -> set[str]:
    if _EXECUTED_FILE.exists():
        try:
            return set(json.loads(_EXECUTED_FILE.read_text()))
        except Exception:
            pass
    return set()


def _save_executed(executed: set[str]) -> None:
    try:
        _EXECUTED_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(_EXECUTED_FILE) + ".tmp"
        Path(tmp).write_text(json.dumps(sorted(executed)))
        os.replace(tmp, _EXECUTED_FILE)
    except Exception as exc:
        logger.warning("ARB: failed to persist executed set: %s", exc)


@dataclass
class ArbOpportunity:
    label: str
    kalshi_ticker: str
    kalshi_side: str
    kalshi_ask: int           # what we'll pay on Kalshi
    poly_condition_id: str    # needed to place the Polymarket leg
    poly_slug: str            # Polymarket market slug
    poly_token_id: str        # Polymarket YES token (for CLOB reference)
    poly_side: str            # which side of Polymarket to buy
    poly_ask: int             # what we'd pay on Polymarket
    combined_cost: int        # kalshi_ask + poly_ask (should be < 100)
    profit_cents: int         # 100 - combined_cost per contract
    poly_question: str


def _events_are_equivalent(kalshi_close: Optional[datetime], poly_end_date: str) -> bool:
    """Return True only if the Kalshi market and Polymarket market cover the same event.

    Strategy: if both end on the same calendar day, they're almost certainly
    the same event (e.g., Wimbledon final on July 12, MLB game on July 4).
    If they differ by more than 1 day, they're different events (e.g., Kalshi
    individual match vs Polymarket tournament winner).
    """
    if not kalshi_close or not poly_end_date:
        return False
    try:
        poly_dt = datetime.fromisoformat(poly_end_date).replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    diff_days = abs((kalshi_close.date() - poly_dt.date()).days)
    return diff_days <= 1


def scan_once(
    kalshi_client,
    poly_client: PolymarketClient,
    series_list: list[str],
    executed_tickers: set[str],
) -> list[ArbOpportunity]:
    """One pass: compare Kalshi sports prices vs Polymarket, return arb opps.

    Only flags GUARANTEED arb — where Kalshi and Polymarket cover the exact
    same binary event (same player/team, same day outcome).  Price divergences
    on different-event pairs (e.g., Kalshi individual match vs Polymarket
    tournament winner) are logged as SIGNAL, not as executable arb.

    Returns only guaranteed arb opportunities (combined cost < MIN_COMBINED_CENTS).
    """
    opps: list[ArbOpportunity] = []

    # Fetch current Kalshi markets with active prices
    markets = kalshi_client.list_liquid_open_markets(series_list=series_list)
    priced = [m for m in markets if m.yes_ask and m.yes_ask > 0 and m.yes_bid and m.yes_bid > 0]

    if not priced:
        logger.debug("Arb scan: no Kalshi sports markets with active prices")
        return []

    # Fetch Polymarket markets for next 30 days
    now = datetime.now(timezone.utc)
    end_min = now.strftime("%Y-%m-%d")
    end_max = (now + timedelta(days=30)).strftime("%Y-%m-%d")
    poly_markets: list[PolyMarket] = poly_client.get_active_markets(
        end_date_min=end_min, end_date_max=end_max, limit=200
    )

    logger.info("Arb scan: %d Kalshi priced sports markets, %d Polymarket markets",
                len(priced), len(poly_markets))

    for km in priced:
        if km.ticker in executed_tickers:
            continue

        pair = find_polymarket_match(km.ticker, poly_markets)
        if not pair:
            continue

        pm = pair.poly_market

        # Only treat as guaranteed arb when both markets close on the same day.
        # If they resolve on different days they're different events and "arb"
        # can result in both legs losing (e.g., player wins today but exits later).
        same_event = _events_are_equivalent(km.close_time, pm.end_date)

        # Forward: YES on Kalshi + NO on Polymarket
        kalshi_yes_ask = km.yes_ask
        poly_no_ask = 100 - pm.yes_bid if pm.yes_bid else None

        if kalshi_yes_ask and poly_no_ask and same_event:
            combined_fwd = kalshi_yes_ask + poly_no_ask
            if combined_fwd < MIN_COMBINED_CENTS:
                profit = 100 - combined_fwd
                opps.append(ArbOpportunity(
                    label=pair.label + " (YES/NO)",
                    kalshi_ticker=km.ticker,
                    kalshi_side="yes",
                    kalshi_ask=kalshi_yes_ask,
                    poly_condition_id=pm.condition_id,
                    poly_slug=pm.slug,
                    poly_token_id=pm.no_token_id,
                    poly_side="no",
                    poly_ask=poly_no_ask,
                    combined_cost=combined_fwd,
                    profit_cents=profit,
                    poly_question=pm.question,
                ))
                logger.warning(
                    "GUARANTEED ARB: %s  Kalshi YES@%d¢ + Poly NO@%d¢ = %d¢  profit=+%d¢",
                    pair.label, kalshi_yes_ask, poly_no_ask, combined_fwd, profit,
                )

        # Reverse: NO on Kalshi + YES on Polymarket
        kalshi_no_ask = km.no_ask
        poly_yes_ask = pm.yes_ask

        if kalshi_no_ask and poly_yes_ask and same_event:
            combined_rev = kalshi_no_ask + poly_yes_ask
            if combined_rev < MIN_COMBINED_CENTS:
                profit = 100 - combined_rev
                opps.append(ArbOpportunity(
                    label=pair.label + " (NO/YES)",
                    kalshi_ticker=km.ticker,
                    kalshi_side="no",
                    kalshi_ask=kalshi_no_ask,
                    poly_condition_id=pm.condition_id,
                    poly_slug=pm.slug,
                    poly_token_id=pm.yes_token_id,
                    poly_side="yes",
                    poly_ask=poly_yes_ask,
                    combined_cost=combined_rev,
                    profit_cents=profit,
                    poly_question=pm.question,
                ))
                logger.warning(
                    "GUARANTEED ARB: %s  Kalshi NO@%d¢ + Poly YES@%d¢ = %d¢  profit=+%d¢",
                    pair.label, kalshi_no_ask, poly_yes_ask, combined_rev, profit,
                )

        # Signal: log price divergences only when events are equivalent or gap is huge
        # Filters out noise from comparing different-day events (e.g., match vs tournament).
        if pm.yes_ask and km.yes_ask and pm.yes_ask > 0:
            diff = abs(pm.yes_ask - km.yes_ask)
            if same_event or diff >= 40:  # same event OR huge price gap worth noting
                direction = "Kalshi CHEAPER" if km.yes_ask < pm.yes_ask else "Poly CHEAPER"
                logger.info(
                    "SIGNAL [%s]: %s  Kalshi YES=%d¢  Poly YES=%d¢  gap=%d¢%s",
                    direction, pair.label, km.yes_ask, pm.yes_ask, diff,
                    "" if same_event else "  (different events — anomalous pricing)",
                )

    return sorted(opps, key=lambda o: o.profit_cents, reverse=True)


def _write_poly_signal(opp: ArbOpportunity, executed: bool, error: Optional[str] = None):
    """Append a record of an attempted Polymarket leg to the local audit file.

    This is a fallback record for manual review, not the execution mechanism
    — run_arb_loop executes the Polymarket leg directly when a trading client
    is supplied. Kept so a human can always double-check what was attempted.
    """
    try:
        ARB_SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if ARB_SIGNAL_FILE.exists():
            try:
                existing = json.loads(ARB_SIGNAL_FILE.read_text())
            except Exception:
                existing = []

        signal = {
            "time": datetime.now(timezone.utc).isoformat(),
            "source": "kalshi_arb",
            "label": opp.label,
            "kalshi_ticker": opp.kalshi_ticker,
            "kalshi_side": opp.kalshi_side,
            "kalshi_ask_cents": opp.kalshi_ask,
            "condition_id": opp.poly_condition_id,
            "slug": opp.poly_slug,
            "question": opp.poly_question,
            "poly_side": opp.poly_side,
            "poly_price": opp.poly_ask / 100,
            "profit_cents": opp.profit_cents,
            "executed": executed,
            "error": error,
        }
        existing.append(signal)
        tmp = str(ARB_SIGNAL_FILE) + ".tmp"
        Path(tmp).write_text(json.dumps(existing, indent=2))
        os.replace(tmp, ARB_SIGNAL_FILE)
    except Exception as exc:
        logger.warning("ARB: failed to write signal file: %s", exc)


def run_arb_loop(
    kalshi_client,
    stop_event: threading.Event,
    sports_series: list[str],
    execute_kalshi: bool = True,
    poly_trading_client=None,
):
    """Background thread: scan for Kalshi<->Polymarket arb opportunities.

    When an opportunity is found, both legs are executed directly:
    - Kalshi leg via kalshi_client (always, when execute_kalshi=True).
    - Polymarket leg via poly_trading_client, if one was supplied — buys the
      outcome token directly (a normal BUY order, not a short), the same way
      src/trading/polymarket/bot.py already bets either side of a market.

    If the Kalshi leg fills but the Polymarket leg then fails, the position
    is left naked (no way to un-fill an already-executed taker order) — this
    is logged as CRITICAL so it gets noticed immediately.

    Args:
        kalshi_client: Authenticated KalshiTradingClient (or paper wrapper).
        stop_event: Set to stop the loop.
        sports_series: Kalshi series to monitor (e.g. ["KXATPMATCH", "KXMLBGAME"]).
        execute_kalshi: If True, auto-place the Kalshi leg.
        poly_trading_client: Authenticated PolymarketTradingClient (or paper
            wrapper). If None, the Polymarket leg is only recorded to the
            signal file for manual execution.
    """
    executed: set[str] = _load_executed()

    with PolymarketClient() as poly:
        while not stop_event.is_set():
            try:
                opps = scan_once(kalshi_client, poly, sports_series, executed)

                for opp in opps:
                    logger.info(
                        "=== ARB OPPORTUNITY ===\n"
                        "  Event:       %s\n"
                        "  Kalshi:      %s %s @ %d¢\n"
                        "  Polymarket:  %s %s @ ~%d¢  (%s)\n"
                        "               Question: %s\n"
                        "  Locked profit: +%d¢ per contract pair",
                        opp.label,
                        opp.kalshi_side.upper(), opp.kalshi_ticker, opp.kalshi_ask,
                        opp.poly_side.upper(), opp.poly_token_id[:30], opp.poly_ask,
                        "auto-executing" if poly_trading_client else "manual — no poly client configured",
                        opp.poly_question,
                        opp.profit_cents,
                    )

                    kalshi_filled = False
                    if execute_kalshi and opp.kalshi_ticker not in executed:
                        try:
                            balance = kalshi_client.get_balance()
                            if balance < opp.kalshi_ask:
                                logger.warning(
                                    "ARB: insufficient balance (%d¢) for Kalshi leg @ %d¢ — skipping",
                                    balance, opp.kalshi_ask,
                                )
                            else:
                                order = kalshi_client.place_order(
                                    ticker=opp.kalshi_ticker,
                                    side=opp.kalshi_side,
                                    count=1,  # conservative: 1 contract to test
                                    yes_price=opp.kalshi_ask if opp.kalshi_side == "yes" else 100 - opp.kalshi_ask,
                                )
                                logger.info("ARB: Kalshi leg placed — order_id=%s", order.order_id)
                                kalshi_filled = True
                                executed.add(opp.kalshi_ticker)
                                _save_executed(executed)
                        except Exception as exc:
                            logger.warning("ARB: Kalshi leg failed: %s", exc)

                    poly_error = None
                    poly_filled = False
                    if kalshi_filled and poly_trading_client and opp.poly_condition_id:
                        outcome = "Yes" if opp.poly_side == "yes" else "No"
                        try:
                            poly_balance = poly_trading_client.get_balance()
                            if poly_balance < opp.poly_ask:
                                poly_error = f"insufficient Polymarket balance ({poly_balance}c < {opp.poly_ask}c)"
                                logger.error(
                                    "ARB CRITICAL: Kalshi leg filled on %s but Polymarket leg skipped (%s) "
                                    "— naked position, no guaranteed-profit pair. Close manually.",
                                    opp.kalshi_ticker, poly_error,
                                )
                            else:
                                poly_order = poly_trading_client.place_order(
                                    condition_id=opp.poly_condition_id,
                                    outcome=outcome,
                                    size=1,
                                    price=opp.poly_ask,
                                    side="BUY",
                                )
                                logger.info("ARB: Polymarket leg placed — order_id=%s", poly_order.order_id)
                                poly_filled = True
                        except Exception as exc:
                            poly_error = str(exc)
                            logger.error(
                                "ARB CRITICAL: Kalshi leg filled on %s but Polymarket leg FAILED (%s) "
                                "— naked position, no guaranteed-profit pair. Close manually.",
                                opp.kalshi_ticker, exc,
                            )

                    if kalshi_filled:
                        _write_poly_signal(opp, executed=poly_filled, error=poly_error)

                if not opps:
                    logger.debug("Arb scan: no opportunities found across %d series", len(sports_series))

            except Exception as exc:
                logger.warning("Arb loop error: %s", exc)

            stop_event.wait(timeout=SCAN_INTERVAL_SECONDS)

    logger.info("Arb loop stopped.")
