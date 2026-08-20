"""An autonomous, calibration-gated trading bot for polymarket.us tennis
totals markets.

Same architecture as sports_props_bot.py, reusing what's genuinely generic
rather than re-deriving money-critical logic a second time:
  - apex.risk: pure deterministic Kelly/edge/reserve sizing math, reused
    unmodified.
  - apex.memory.ApexMemory: bankroll/trade/lesson ledger, reused with its
    own separate ledger file so bankroll state is never shared with any
    other bot on this account.
  - Execution paranoia (IOC orders, position-readback instead of trusting
    the order response, float fill counts never int()'d): same lineage as
    every other bot on this account.

Scope: TOTALS ONLY (categories ending in "_game_totals"/"_set_totals").
Spreads (categories ending in "_spreads") are explicitly excluded -- real
finding 2026-08-18: totals verify cleanly against real ESPN match data
(see tennis_stats_verification.py), but spreads don't resolve under any
sign convention tried. Trading an unverifiable settlement convention with
real money is exactly the mistake the MLB settlement-verification work this
same day exists to prevent; this bot simply doesn't go there yet.

No cross-bot position-reconciliation guard is needed the way sports_props_
bot.py needs one against the ladder bot: tennis's asc-*/tsc-* slugs can
never match any of the ladder bot's three arbitrage families (confirmed
live -- the "gte" family requires the slug to end in "-gte{digits}", tennis
slugs end in "pt{digits}"; "before_round" and "by_date" don't match either).
Safe by construction, same as Apex's tc-temp-* prefix.

Portfolio construction
-----------------------
Holds at most one open position per (tour, matchup, date) at a time -- a
game-totals bet and a set-totals bet on the SAME match both depend on that
one match's real outcome, so they're correlated the same way multiple stat
props on one MLB game are.

Separately, at most one open position per (category, side, decile)
calibration bucket at a time, built in from day one rather than retrofitted
-- see sports_props_bot.py's own module docstring for the real 2026-08-17
incident (4 of 6 real trades drawing on one shared calibration signal and
losing together) that this guard exists to prevent there. Same fix, applied
here before it was ever needed instead of after.
"""

from __future__ import annotations

import logging
import time

from src.trading.polymarket_us.apex import risk
from src.trading.polymarket_us.apex.memory import ApexMemory
from src.trading.polymarket_us.paper_client import _settlement_to_cents
from src.trading.polymarket_us.strategy import taker_fee_cents
from src.trading.polymarket_us.tennis_prop_calibration import TennisPropCalibrationEngine, _decile_low
from src.trading.polymarket_us.tennis_prop_quote_logger import parse_tennis_slug

logger = logging.getLogger(__name__)

FILL_POLL_ATTEMPTS = 5
FILL_POLL_DELAY = 1.0
FILL_EPSILON = 1e-6
UNKNOWN_DEPTH_CONTRACTS = 1

TOTALS_SUFFIXES = ("_game_totals", "_set_totals")


def _fmt_qty(x: float) -> str:
    return f"{x:g}"


def _model_prob(grounding) -> float:
    """Same convention as sports_props_bot.py's -- the decile's own
    historically-measured actual rate, not the market's current price."""
    return (grounding.calibration_actual_rate or 0.0) / 100.0


def _is_totals_meta(meta: dict) -> bool:
    return meta["category"].endswith(TOTALS_SUFFIXES)


def _tour_of(category: str) -> str:
    return category.split("_", 1)[0]


def _game_key(slug: str) -> tuple | None:
    meta = parse_tennis_slug(slug)
    if meta is None or not _is_totals_meta(meta):
        return None
    return (_tour_of(meta["category"]), meta["matchup"], meta["date"])


def _bucket_key(category: str, side: str, price_cents: int) -> tuple:
    return (category, side, _decile_low(price_cents))


def _trade_bucket_key(trade: dict) -> tuple | None:
    meta = parse_tennis_slug(trade["market_slug"])
    if meta is None:
        return None
    return _bucket_key(meta["category"], trade["side"], trade["price_cents"])


class TennisPropsBot:
    def __init__(
        self,
        client,
        memory: ApexMemory | None = None,
        calibration: TennisPropCalibrationEngine | None = None,
        poll_interval: int = 300,
    ):
        self.client = client
        self.memory = memory or ApexMemory(label="tennis_props")
        self.calibration = calibration or TennisPropCalibrationEngine()
        self.poll_interval = poll_interval
        self._session_started = False

    # ── Position readback (same paranoia as every bot on this account) ──────

    def _net_position(self, slug: str) -> float:
        for p in self.client.get_positions():
            if p.market_slug == slug:
                return float(p.position)
        return 0.0

    def _position_after_order(self, slug: str, before: float) -> float:
        latest = before
        for _ in range(FILL_POLL_ATTEMPTS):
            time.sleep(FILL_POLL_DELAY)
            latest = self._net_position(slug)
            if latest != before:
                return latest
        return latest

    # ── Settlement reconciliation ────────────────────────────────────────────

    def _reconcile_settlements(self) -> None:
        for trade in self.memory.open_trades():
            slug = trade["market_slug"]
            try:
                market = self.client.get_market(slug)
            except Exception as exc:
                logger.debug("[tennis_props] settlement check failed for %s: %s", slug, exc)
                continue
            if not market.closed:
                continue
            try:
                settlement = self.client.get_settlement(slug)
            except Exception as exc:
                logger.debug("[tennis_props] no settlement yet for %s: %s", slug, exc)
                continue
            settlement_cents = _settlement_to_cents(settlement)
            if settlement_cents is None:
                continue
            outcome_yes = settlement_cents >= 50

            won = outcome_yes if trade["side"] == "yes" else (not outcome_yes)
            payout_cents = round(trade["count"] * 100) if won else 0
            pnl_cents = self.memory.record_settlement(trade["id"], outcome_yes, payout_cents)
            logger.info(
                "[tennis_props] SETTLED %s -> %s  %sx %s  pnl=%+d¢  bankroll=$%.2f",
                slug,
                "YES" if outcome_yes else "NO",
                _fmt_qty(trade["count"]),
                trade["side"].upper(),
                pnl_cents or 0,
                self.memory.bankroll_cents / 100,
            )
            miss = abs(trade["model_prob"] - (1.0 if won else 0.0))
            if miss >= 0.5 and (pnl_cents or 0) < 0:
                self.memory.add_lesson(
                    f"{slug}: model gave this side {trade['model_prob']:.0%} and it lost -- "
                    f"a {miss:.0%} miss. Recheck this decile's calibration on the next reload."
                )

    # ── Live-position reconciliation ────────────────────────────────────────

    def _reconcile_live_positions(self) -> None:
        """No cross-bot exclusion needed -- see module docstring. Any live
        tennis totals slug this bot doesn't recognize as its own is a real
        anomaly, not a sibling bot's legitimate position."""
        try:
            live_positions = self.client.get_positions()
        except Exception as exc:
            logger.warning("[tennis_props] position reconciliation check failed: %s", exc)
            return
        live_slugs = {
            p.market_slug
            for p in live_positions
            if abs(p.position) > FILL_EPSILON and _game_key(p.market_slug) is not None
        }
        tracked_slugs = {t["market_slug"] for t in self.memory.open_trades()}

        stray = live_slugs - tracked_slugs
        if stray:
            self.memory.halt(
                f"live account holds untracked tennis-totals position(s) not in this bot's ledger: {sorted(stray)}"
            )

        missing = tracked_slugs - live_slugs
        if missing:
            logger.warning(
                "[tennis_props] memory shows open position(s) the live account no longer holds: %s "
                "(may be a settlement not yet reconciled)",
                sorted(missing),
            )

    # ── Bucket selection ─────────────────────────────────────────────────────

    def _best_bucket_for_match(self, buckets: list) -> tuple | None:
        """Highest-edge tradeable bucket across every totals line for one
        match, or None. Priced off the bulk scan's quotes as a coarse
        filter; the caller re-quotes the chosen candidate before sizing."""
        best = None  # (edge, side, slug, meta)
        for slug, meta, market in buckets:
            if market.yes_ask is not None:
                grounding = self.calibration.evaluate(meta["category"], "yes", market.yes_ask)
                if grounding.calibration_trustworthy:
                    edge = risk.edge_fraction(_model_prob(grounding), market.yes_ask)
                    if best is None or edge > best[0]:
                        best = (edge, "yes", slug, meta, grounding)
            if market.yes_bid is not None:
                no_price = 100 - market.yes_bid
                grounding = self.calibration.evaluate(meta["category"], "no", no_price)
                if grounding.calibration_trustworthy:
                    edge = risk.edge_fraction(_model_prob(grounding), no_price)
                    if best is None or edge > best[0]:
                        best = (edge, "no", slug, meta, grounding)
        return best

    def _requote_bucket(self, slug: str, side: str) -> tuple | None:
        try:
            fresh = self.client.get_market(slug)
        except Exception as exc:
            logger.warning("[tennis_props] re-quote failed for %s: %s", slug, exc)
            return None
        if side == "yes":
            if fresh.yes_ask is None:
                return None
            depth = fresh.ask_depth if fresh.ask_depth is not None else UNKNOWN_DEPTH_CONTRACTS
            return fresh.yes_ask, fresh.yes_ask, depth
        if fresh.yes_bid is None:
            return None
        depth = fresh.bid_depth if fresh.bid_depth is not None else UNKNOWN_DEPTH_CONTRACTS
        return 100 - fresh.yes_bid, fresh.yes_bid, depth

    # ── Order placement ──────────────────────────────────────────────────────

    def _place_decision(
        self,
        side: str,
        slug: str,
        meta: dict,
        order_yes_price: int,
        cost_price_cents: int,
        model_prob_for_side: float,
        decision: risk.ApexDecision,
    ) -> int:
        market_implied = cost_price_cents / 100.0

        before = self._net_position(slug)
        try:
            self.client.place_order(
                market_slug=slug,
                side=side,
                count=decision.count,
                yes_price=order_yes_price,
                immediate_or_cancel=True,
            )
        except Exception as exc:
            logger.warning("[tennis_props] %s order rejected on %s: %s", side.upper(), slug, exc)
            raw = self._position_after_order(slug, before) - before
            filled = raw if side == "yes" else -raw
            if abs(filled) <= FILL_EPSILON:
                return 0
            logger.error(
                "[tennis_props] CRITICAL: %s order on %s reported an error but left %s contracts -- recording it",
                side.upper(),
                slug,
                _fmt_qty(filled),
            )
        else:
            raw = self._position_after_order(slug, before) - before
            filled = raw if side == "yes" else -raw
            if abs(filled) <= FILL_EPSILON:
                logger.info("[tennis_props] %s leg on %s did not fill (IOC) -- no position taken", side.upper(), slug)
                return 0

        if filled < 0:
            self.memory.halt(
                f"{slug}: position moved opposite to the traded side ({side}) -- "
                f"observed delta {_fmt_qty(filled)}, expected non-negative"
            )
            return 0

        fee_cents = round(filled * taker_fee_cents(cost_price_cents, 1))
        cost_cents = round(filled * cost_price_cents) + fee_cents
        if cost_cents <= 0:
            cost_cents = 1

        grounding = self.calibration.evaluate(meta["category"], side, cost_price_cents)
        rationale = (
            f"{meta['category']} {side.upper()} at {cost_price_cents}c: {grounding.calibration_note}, "
            f"edge {decision.edge_fraction:+.1%}, reason: {decision.reason}"
        )
        self.memory.record_open(
            market_slug=slug,
            side=side,
            count=filled,
            price_cents=cost_price_cents,
            cost_cents=cost_cents,
            model_prob=model_prob_for_side,
            market_implied_prob=market_implied,
            edge_pct=decision.edge_fraction,
            kelly_fraction_used=decision.kelly_fraction_used,
            rationale=rationale,
        )
        logger.info(
            "[tennis_props] %s %sx %s @ %d¢ (+%d¢ fee)  edge=%+.1f%%  cost=$%.2f  %s",
            side.upper(),
            _fmt_qty(filled),
            slug,
            cost_price_cents,
            fee_cents,
            decision.edge_fraction * 100,
            cost_cents / 100,
            rationale,
        )
        return cost_cents

    # ── Scan loop ────────────────────────────────────────────────────────────

    def _scan_once(self) -> None:
        self.memory.roll_daily_window()
        self._reconcile_settlements()

        if self.memory.is_terminated():
            logger.error(
                "[tennis_props] TERMINATED -- %s. Re-fund and reset the ledger by hand to resume.",
                self.memory.terminated_reason,
            )
            return

        self._reconcile_live_positions()
        if self.memory.is_halted():
            logger.error(
                "[tennis_props] HALTED -- %s. Resolve by hand and clear anomaly_halted_reason "
                "in the ledger file to resume.",
                self.memory.anomaly_halted_reason,
            )
            return
        if self.memory.is_halted_today():
            logger.info("[tennis_props] Halted for today -- %s. Observing only.", self.memory.daily_halted_reason)
            return

        # Reload before checking staleness -- same ordering fix as every
        # other bot on this account.
        self.calibration.reload()
        if self.calibration.data_is_stale():
            logger.warning(
                "[tennis_props] Calibration data is stale or missing -- is "
                "tennis_prop_quote_logger.py running? Observing only this scan."
            )
            return

        try:
            markets = self.client.list_all_markets(active=True, closed=False)
        except Exception as exc:
            logger.warning("[tennis_props] market scan failed: %s", exc)
            return

        buckets_by_match: dict[tuple, list] = {}
        for m in markets:
            meta = parse_tennis_slug(m.slug)
            if meta is None or not _is_totals_meta(meta):
                continue
            key = (_tour_of(meta["category"]), meta["matchup"], meta["date"])
            buckets_by_match.setdefault(key, []).append((m.slug, meta, m))
        if not buckets_by_match:
            logger.info("[tennis_props] no open totals markets found this scan")
            return

        held_matches = {_game_key(t["market_slug"]) for t in self.memory.open_trades()}
        held_buckets = {k for k in (_trade_bucket_key(t) for t in self.memory.open_trades()) if k is not None}

        try:
            balance_cents = self.client.get_balance()
        except Exception as exc:
            logger.warning("[tennis_props] balance check failed: %s", exc)
            return
        self.memory.check_external_balance(balance_cents)
        if self.memory.is_terminated():
            logger.error("[tennis_props] TERMINATED -- %s", self.memory.terminated_reason)
            return
        cash_cents = min(self.memory.cash_cents, balance_cents)

        for match, buckets in buckets_by_match.items():
            if match in held_matches:
                continue
            candidate = self._best_bucket_for_match(buckets)
            if candidate is None:
                continue
            _coarse_edge, side, slug, meta, _stale_grounding = candidate

            requote = self._requote_bucket(slug, side)
            if requote is None:
                continue
            cost_price_cents, order_yes_price, depth = requote

            # Re-evaluate at the fresh requoted price, not the bulk-scan
            # price -- same staleness reasoning as sports_props_bot.py's
            # identical re-check.
            grounding = self.calibration.evaluate(meta["category"], side, cost_price_cents)
            if not grounding.calibration_trustworthy:
                logger.debug(
                    "[tennis_props] PASS %s (%s side): no longer trustworthy at the requoted price (%s)",
                    slug,
                    side,
                    grounding.calibration_note,
                )
                continue

            bucket_key = _bucket_key(meta["category"], side, cost_price_cents)
            if bucket_key in held_buckets:
                logger.info(
                    "[tennis_props] PASS %s (%s side): already holding a position drawing on the same "
                    "(category, side, decile) signal %s -- not an independent bet",
                    slug,
                    side,
                    bucket_key,
                )
                continue

            model_prob_for_side = _model_prob(grounding)
            decision = risk.evaluate(
                model_prob=model_prob_for_side,
                price_cents=cost_price_cents,
                bankroll_cents=self.memory.bankroll_cents,
                cash_cents=cash_cents,
                max_contracts=depth,
            )
            if decision.action == "pass":
                logger.debug("[tennis_props] PASS %s (%s side): %s", slug, side, decision.reason)
                continue

            spent = self._place_decision(
                side, slug, meta, order_yes_price, cost_price_cents, model_prob_for_side, decision
            )
            cash_cents -= spent
            if spent > 0:
                held_buckets.add(bucket_key)
            if self.memory.is_halted():
                return

    def run_forever(self, stop_event=None) -> None:
        if not self._session_started:
            logger.info("\n%s", self.memory.session_banner())
            self._session_started = True
        while True:
            try:
                self._scan_once()
            except Exception as exc:
                logger.warning("[tennis_props] scan error: %s", exc)
            if stop_event is not None:
                if stop_event.wait(timeout=self.poll_interval):
                    break
            else:
                time.sleep(self.poll_interval)
