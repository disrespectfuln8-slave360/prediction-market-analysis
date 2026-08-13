"""Apex -- an autonomous, edge-gated trading agent for polymarket.us weather markets.

Architecture and why it's scoped this way
------------------------------------------
The persona this module implements ("Apex") was specified as an LLM that
estimates true probabilities from general reasoning and bets whenever it
perceives edge. This repo already tried that shape once -- the crypto/macro
mispricing bots mentioned in strategy.py's module docstring -- and removed
them for relying on a miscalibrated model with no supporting data. This
implementation keeps the persona's bankroll/risk rules and output format,
but changes where the probability comes from and who is allowed to decide a
trade:

  - Probability: weather_model.py's NOAA-forecast-implied model, the one
    probability source in this repo with measured calibration against real
    settlements (Brier/ECE/MCE -- see weather_forecast_calibration.py). No
    other polymarket.us category has an equivalent grounded model here, so
    Apex trades only polymarket.us's daily-high-temperature bucket markets.
    Everything else is out of scope, not silently priced off a guess.
  - Sizing: risk.py, pure deterministic Kelly/edge/reserve math. llm.py is
    only ever called after a decision is already final, to write the
    display rationale -- it never becomes the thing deciding how much real
    money moves.
  - Execution: the same paranoia as the sibling ladder-arb bot (see
    src/trading/polymarket_us/bot.py's module docstring for the incidents
    that taught it): IOC orders, position readback instead of trusting the
    order response (this exchange has been observed reporting empty
    executions for orders that filled in full, in both directions), and
    fractional contract counts kept as floats end to end.

Portfolio construction
-----------------------
Same-city, same-day bucket markets are draws from the same forecast, so
they are highly correlated. Apex holds at most one open position per
(city, local-date) at a time -- the single best-edge bucket, never a stack.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from src.trading.polymarket_us.apex import llm, risk
from src.trading.polymarket_us.apex.memory import ApexMemory
from src.trading.polymarket_us.apex.weather_model import WeatherGroundingEngine
from src.trading.polymarket_us.paper_client import _settlement_to_cents
from src.trading.polymarket_us.strategy import taker_fee_cents

logger = logging.getLogger(__name__)

# Same rationale as LadderArbBot: this exchange's order-creation response has
# been observed reporting zero fills for orders that filled completely, so
# fill state is only ever read back from get_positions(), never trusted from
# the order response itself.
FILL_POLL_ATTEMPTS = 5
FILL_POLL_DELAY = 1.0
FILL_EPSILON = 1e-6

# Depth is only known when a fresh per-market BBO fetch succeeds; if it comes
# back missing, cap size to the smallest unit rather than trade blind against
# an unknown book -- same convention as LadderArbBot.UNKNOWN_DEPTH_CONTRACTS.
UNKNOWN_DEPTH_CONTRACTS = 1


def _fmt_qty(x: float) -> str:
    """Compact display for a fractional contract count."""
    return f"{x:g}"


def _city_from_slug(slug: str) -> Optional[str]:
    parts = slug.split("-")
    if len(parts) >= 3 and parts[0] == "tc" and parts[1] == "temp":
        return parts[2]
    return None


class ApexBot:
    def __init__(
        self,
        client,
        memory: Optional[ApexMemory] = None,
        grounding: Optional[WeatherGroundingEngine] = None,
        poll_interval: int = 300,
    ):
        self.client = client
        self.memory = memory or ApexMemory()
        self.grounding = grounding or WeatherGroundingEngine()
        self.poll_interval = poll_interval
        self._session_started = False

    # ── Position readback (same paranoia as the ladder bot) ─────────────────

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
                logger.debug("[apex] settlement check failed for %s: %s", slug, exc)
                continue
            if not market.closed:
                continue
            try:
                settlement = self.client.get_settlement(slug)
            except Exception as exc:
                logger.debug("[apex] no settlement yet for %s: %s", slug, exc)
                continue
            settlement_cents = _settlement_to_cents(settlement)
            if settlement_cents is None:
                continue
            outcome_yes = settlement_cents >= 50

            won = outcome_yes if trade["side"] == "yes" else (not outcome_yes)
            payout_cents = round(trade["count"] * 100) if won else 0
            pnl_cents = self.memory.record_settlement(trade["id"], outcome_yes, payout_cents)
            logger.info(
                "[apex] SETTLED %s -> %s  %sx %s  pnl=%+d¢  bankroll=$%.2f",
                slug, "YES" if outcome_yes else "NO", _fmt_qty(trade["count"]), trade["side"].upper(),
                pnl_cents or 0, self.memory.bankroll_cents / 100,
            )
            miss = abs(trade["model_prob"] - (1.0 if won else 0.0))
            if miss >= 0.5 and (pnl_cents or 0) < 0:
                self.memory.add_lesson(
                    f"{slug}: model gave this side {trade['model_prob']:.0%} and it lost -- "
                    f"a {miss:.0%} miss. Recheck this decile's calibration on the next reload."
                )

    # ── Live-position reconciliation ────────────────────────────────────────

    def _reconcile_live_positions(self) -> None:
        """Cross-check memory's open trades against the real account's live
        positions. An untracked live position -- one the exchange shows
        that memory doesn't know about -- would silently defeat the
        one-position-per-city guard (bot.py's held_cities is built purely
        from memory) and leave the bankroll ledger blind to its risk.
        Carrying that unknown risk is worse than halting, so this stops
        the bot rather than trading around a ledger it can no longer
        trust, matching the sibling ladder bot's own philosophy."""
        try:
            live_positions = self.client.get_positions()
        except Exception as exc:
            logger.warning("[apex] position reconciliation check failed: %s", exc)
            return
        # Scope to Apex's own weather markets -- this account may hold other
        # live positions (e.g. the sibling ladder bot's) that are none of
        # Apex's concern and must not trigger a halt here.
        live_slugs = {
            p.market_slug for p in live_positions
            if abs(p.position) > FILL_EPSILON and _city_from_slug(p.market_slug) is not None
        }
        tracked_slugs = {t["market_slug"] for t in self.memory.open_trades()}

        stray = live_slugs - tracked_slugs
        if stray:
            self.memory.halt(
                f"live account holds untracked weather position(s) not in Apex's ledger: {sorted(stray)}"
            )

        missing = tracked_slugs - live_slugs
        if missing:
            logger.warning(
                "[apex] memory shows open position(s) the live account no longer holds: %s "
                "(may be a settlement not yet reconciled)", sorted(missing),
            )

    # ── Bucket selection ─────────────────────────────────────────────────────

    def _best_bucket_for_city(self, city: str, buckets: list) -> Optional[tuple]:
        """Highest-edge tradeable bucket in one city-day, or None. Priced off
        the bulk scan's quotes as a coarse filter -- the caller re-quotes the
        chosen candidate against a fresh per-market BBO before sizing, since
        a scan across thousands of markets takes long enough for prices to
        move (same lesson LadderArbBot's module docstring documents)."""
        best = None  # (edge, side, slug, grounding)
        for slug, low, high, market in buckets:
            grounding = self.grounding.evaluate_bucket(city, low, high)
            if grounding is None or not grounding.calibration_trustworthy:
                continue
            if market.yes_ask is not None:
                edge = risk.edge_fraction(grounding.model_prob, market.yes_ask)
                if best is None or edge > best[0]:
                    best = (edge, "yes", slug, grounding)
            if market.yes_bid is not None:
                no_price = 100 - market.yes_bid
                edge = risk.edge_fraction(1.0 - grounding.model_prob, no_price)
                if best is None or edge > best[0]:
                    best = (edge, "no", slug, grounding)
        return best

    def _requote_bucket(self, slug: str, side: str) -> Optional[tuple]:
        """(cost_price_cents, order_yes_price, depth) against a fresh BBO, or
        None if the side is unquoted. cost_price_cents is what a contract on
        this side actually costs (used for risk/edge math and display);
        order_yes_price is what place_order's yes_price parameter expects,
        which for a "no" order is the YES-equivalent price, not the NO cost
        -- see PolymarketUSTradingClient.place_order's docstring."""
        try:
            fresh = self.client.get_market(slug)
        except Exception as exc:
            logger.warning("[apex] re-quote failed for %s: %s", slug, exc)
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
        self, side: str, slug: str, order_yes_price: int, cost_price_cents: int,
        grounding, decision: risk.ApexDecision,
    ) -> int:
        """Returns the cents actually spent (0 if nothing filled), so the
        caller can track its own per-scan budget against real spend rather
        than the pre-fee amount risk.evaluate() planned for."""
        model_prob_for_side = grounding.model_prob if side == "yes" else (1.0 - grounding.model_prob)
        market_implied = cost_price_cents / 100.0

        before = self._net_position(slug)
        try:
            self.client.place_order(
                market_slug=slug, side=side,
                count=decision.count, yes_price=order_yes_price, immediate_or_cancel=True,
            )
        except Exception as exc:
            logger.warning("[apex] %s order rejected on %s: %s", side.upper(), slug, exc)
            # This exchange's order-creation response has been observed
            # lying about fills in both directions (see the ladder bot's
            # module docstring) -- an exception here doesn't mean nothing
            # happened, so check the real position before walking away.
            # Poll exactly like the success path below (same propagation
            # lag applies) rather than trusting a single immediate read.
            raw = self._position_after_order(slug, before) - before
            filled = raw if side == "yes" else -raw
            if abs(filled) <= FILL_EPSILON:
                return 0
            logger.error(
                "[apex] CRITICAL: %s order on %s reported an error but left %s contracts -- recording it",
                side.upper(), slug, _fmt_qty(filled),
            )
        else:
            # A float, never int()'d -- this exchange fills some markets in
            # fractional contracts (see the ladder bot's module docstring
            # for the incident that taught this repo that lesson).
            raw = self._position_after_order(slug, before) - before
            filled = raw if side == "yes" else -raw
            if abs(filled) <= FILL_EPSILON:
                logger.info("[apex] %s leg on %s did not fill (IOC) -- no position taken", side.upper(), slug)
                return 0

        if filled < 0:
            # The position moved opposite to the side actually traded --
            # something else touched this slug, or the readback is lying
            # in a way that isn't just "zero". Recording a fabricated
            # "successful" fill in the intended direction here would
            # corrupt the ledger; halting and leaving it for a human is
            # the same call this exchange's other paranoia makes.
            self.memory.halt(
                f"{slug}: position moved opposite to the traded side ({side}) -- "
                f"observed delta {_fmt_qty(filled)}, expected non-negative"
            )
            return 0

        # Fee is charged on every real fill (verified against this
        # account's own activity — see strategy.py's THETA_TAKER) but
        # risk.py's stake/count math is priced off the raw contract price
        # only, so it has to be added back in here for the ledger to match
        # what the exchange actually takes. Per-contract rate scaled by the
        # float fill count, matching how risk.py already treats the fee
        # for edge/Kelly purposes.
        fee_cents = round(filled * taker_fee_cents(cost_price_cents, 1))
        cost_cents = round(filled * cost_price_cents) + fee_cents
        rationale = llm.generate_rationale({
            "action": "buy", "market_slug": slug, "city": grounding.city,
            "low": grounding.low, "high": grounding.high,
            "model_prob": model_prob_for_side, "market_implied_prob": market_implied,
            "edge_pct": decision.edge_fraction, "calibration_note": grounding.calibration_note,
            "stake_cents": cost_cents, "count": filled, "reason": decision.reason,
            "bankroll_cents": self.memory.bankroll_cents,
            "recent_lessons": self.memory.recent_lessons(3),
        })
        self.memory.record_open(
            market_slug=slug, side=side, count=filled, price_cents=cost_price_cents,
            cost_cents=cost_cents, model_prob=model_prob_for_side,
            market_implied_prob=market_implied, edge_pct=decision.edge_fraction,
            kelly_fraction_used=decision.kelly_fraction_used, rationale=rationale["rationale"],
        )
        self._print_decision_block(slug, side, market_implied, model_prob_for_side, decision, cost_cents, rationale)
        logger.info(
            "[apex] %s %sx %s @ %d¢ (+%d¢ fee)  edge=%+.1f%%  cost=$%.2f",
            side.upper(), _fmt_qty(filled), slug, cost_price_cents, fee_cents,
            decision.edge_fraction * 100, cost_cents / 100,
        )
        return cost_cents

    def _print_decision_block(self, slug, side, market_implied, model_prob, decision, cost_cents, rationale) -> None:
        """Renders the persona's required output format from real,
        already-computed numbers -- never from an LLM's own arithmetic."""
        print(
            f"Market: {slug}\n"
            f"Current Price: {side.upper()} {market_implied * 100:.0f}¢\n"
            f"My Probability: {model_prob * 100:.1f}%\n"
            f"Edge: {decision.edge_fraction * 100:+.1f}%\n"
            f"Position Size: ${cost_cents / 100:.2f} "
            f"({decision.kelly_fraction_used * 100:.2f}% of bankroll)\n"
            f"Action: BUY {side.upper()}\n"
            f"Rationale: {rationale['rationale']}\n"
            f"Residual risks: {rationale['residual_risks']}\n"
            f"Memory Update: trade logged, bankroll now ${self.memory.bankroll_cents / 100:.2f}\n"
        )

    # ── Scan loop ────────────────────────────────────────────────────────────

    def _scan_once(self) -> None:
        self.memory.roll_daily_window()
        self._reconcile_settlements()

        if self.memory.is_terminated():
            logger.error(
                "[apex] TERMINATED -- %s. Re-fund and reset the ledger by hand to resume.",
                self.memory.terminated_reason,
            )
            return

        self._reconcile_live_positions()
        if self.memory.is_halted():
            logger.error(
                "[apex] HALTED -- %s. Resolve by hand and clear anomaly_halted_reason "
                "in the ledger file to resume.", self.memory.anomaly_halted_reason,
            )
            return
        if self.memory.is_halted_today():
            logger.info("[apex] Halted for today -- %s. Observing only.", self.memory.daily_halted_reason)
            return

        # Reload BEFORE checking staleness -- the check reads
        # self.grounding's cached _history_max_date, which is only ever
        # updated by reload_calibration(). Checking staleness first means
        # a history file that didn't exist when the engine was constructed
        # (the normal case: Apex starts before the one-off backfill script
        # has ever been run) can never be discovered, because the early
        # return on "stale" would permanently prevent reload_calibration()
        # from running again.
        self.grounding.reload_calibration()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.grounding.history_is_stale(today):
            logger.warning(
                "[apex] Weather calibration history is stale or missing -- run "
                "scripts/backfill_weather_history.py before trading. Observing only this scan."
            )
            return

        try:
            markets = self.client.list_all_markets(active=True, closed=False)
        except Exception as exc:
            logger.warning("[apex] market scan failed: %s", exc)
            return
        buckets_by_city = self.grounding.discover_open_buckets(markets)
        if not buckets_by_city:
            logger.info("[apex] no open weather bucket markets found this scan")
            return

        # Same-city, same-day buckets are draws from one forecast and are
        # highly correlated -- never stack a second position in a city
        # already held (see module docstring).
        held_cities = {
            _city_from_slug(t["market_slug"]) for t in self.memory.open_trades()
        }

        try:
            balance_cents = self.client.get_balance()
        except Exception as exc:
            logger.warning("[apex] balance check failed: %s", exc)
            return
        # Independent check against the real account, not just the tracked
        # ledger -- bankroll_cents can drift from reality (see memory.py's
        # check_external_balance docstring), and only ever in the direction
        # that understates risk, so this is the backstop.
        self.memory.check_external_balance(balance_cents)
        if self.memory.is_terminated():
            logger.error("[apex] TERMINATED -- %s", self.memory.terminated_reason)
            return
        cash_cents = min(self.memory.cash_cents, balance_cents)

        for city, buckets in buckets_by_city.items():
            if city in held_cities:
                continue
            candidate = self._best_bucket_for_city(city, buckets)
            if candidate is None:
                continue
            _coarse_edge, side, slug, grounding = candidate

            requote = self._requote_bucket(slug, side)
            if requote is None:
                continue
            cost_price_cents, order_yes_price, depth = requote

            model_prob_for_side = grounding.model_prob if side == "yes" else (1.0 - grounding.model_prob)
            decision = risk.evaluate(
                model_prob=model_prob_for_side, price_cents=cost_price_cents,
                bankroll_cents=self.memory.bankroll_cents, cash_cents=cash_cents,
                max_contracts=depth,
            )
            if decision.action == "pass":
                logger.debug("[apex] PASS %s (%s side): %s", slug, side, decision.reason)
                continue

            spent = self._place_decision(side, slug, order_yes_price, cost_price_cents, grounding, decision)
            cash_cents -= spent  # actual spend including fees, not the pre-fee plan
            if self.memory.is_halted():
                # _place_decision just found something it can't safely
                # reason about (e.g. a wrong-direction fill) -- stop
                # opening new positions in other cities this scan too.
                return

    def run_forever(self, stop_event=None) -> None:
        if not self._session_started:
            logger.info("\n%s", self.memory.session_banner())
            self._session_started = True
        while True:
            try:
                self._scan_once()
            except Exception as exc:
                logger.warning("[apex] scan error: %s", exc)
            if stop_event is not None:
                if stop_event.wait(timeout=self.poll_interval):
                    break
            else:
                time.sleep(self.poll_interval)
