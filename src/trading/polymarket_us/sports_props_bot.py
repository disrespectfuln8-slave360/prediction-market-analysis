"""An autonomous, calibration-gated trading bot for polymarket.us MLB/NFL
player-prop markets.

Same architecture as apex/bot.py, reusing what's genuinely generic from it
rather than re-deriving money-critical logic a second time:
  - apex.risk: pure deterministic Kelly/edge/reserve sizing math. Nothing
    in it references weather -- it takes a probability, a price, and a
    bankroll, so it's reused here unmodified.
  - apex.memory.ApexMemory: bankroll/trade/lesson ledger with the same
    three stop states (terminated / anomaly-halted / daily-halted).
    Nothing in it references weather either -- reused with its own,
    separate ledger file so the two bots never share bankroll state.
  - Execution paranoia (IOC orders, position-readback instead of trusting
    the order response, float fill counts never int()'d): copied from
    apex/bot.py, which copied it from the sibling ladder bot's own
    incident history. Same exchange, same lies, same fix.

What's NOT reused: apex/llm.py (its template and system prompt describe
the NOAA forecast model by name -- would be a false claim here, see
sports_props_llm.py) and weather_model.py (see sports_prop_calibration.py
for why this platform's own price history is the calibration source
instead of an independent forecast).

Portfolio construction
-----------------------
Multiple props on the SAME game are correlated (a high-scoring game lifts
many players' stat lines at once) the same way same-city-same-day weather
buckets are draws from one forecast. Holds at most one open position per
(sport, matchup, date) at a time -- the single best-edge bucket across
every stat/player in that game, never a stack.

A second, separate correlation this guard does NOT cover: two DIFFERENT
games can independently produce a candidate priced into the exact same
(sport, stat, side, decile) calibration bucket -- model_prob for both comes
from that one bucket's single calibration_actual_rate, so they are not two
independent bets, they are the same underlying signal bet twice. Real
incident 2026-08-17: 4 of 6 real trades that day all drew on the identical
(mlb, tb, yes, 10-20%) bucket and lost together, not four independent
0.20-ish-probability draws. Guarded separately below (held_buckets) since
it's orthogonal to the per-game guard -- two different games, same signal.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from src.trading.polymarket_us import sports_props_llm as llm
from src.trading.polymarket_us.apex import risk
from src.trading.polymarket_us.apex.memory import ApexMemory
from src.trading.polymarket_us.bot import HELD_SLUGS_PATH as LADDER_HELD_SLUGS_PATH
from src.trading.polymarket_us.nfl_prop_quote_logger import parse_nfl_prop_slug
from src.trading.polymarket_us.paper_client import _settlement_to_cents
from src.trading.polymarket_us.sports_prop_calibration import SportsPropCalibrationEngine, _decile_low
from src.trading.polymarket_us.sports_prop_quote_logger import parse_prop_slug
from src.trading.polymarket_us.strategy import taker_fee_cents

logger = logging.getLogger(__name__)

# A bit more than the ladder bot's own 60s poll interval, to tolerate normal
# scheduling jitter without falsely distrusting fresh data -- but tight
# enough that a genuinely dead/stopped ladder bot is noticed rather than
# trusted forever. Missing or stale falls back to the empty set (the older,
# more cautious behavior), never to trusting data that might no longer be true.
LADDER_HELD_SLUGS_STALE_SECS = 600


def _ladder_bot_held_slugs() -> set[str]:
    try:
        raw = json.loads(LADDER_HELD_SLUGS_PATH.read_text(encoding="utf-8"))
        updated = datetime.fromisoformat(raw["updated_ts"])
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        if age > LADDER_HELD_SLUGS_STALE_SECS:
            return set()
        return set(raw.get("slugs", []))
    except Exception:
        return set()


FILL_POLL_ATTEMPTS = 5
FILL_POLL_DELAY = 1.0
FILL_EPSILON = 1e-6
UNKNOWN_DEPTH_CONTRACTS = 1


def _fmt_qty(x: float) -> str:
    return f"{x:g}"


def _model_prob(grounding) -> float:
    """The "model probability" this bot bets against is the decile's own
    historically-measured actual rate -- not the market's current price
    (grounding.implied_prob), which would make edge zero by construction.
    Only meaningful once calibration_trustworthy is True, which guarantees
    calibration_actual_rate is a real measured float, not None."""
    return (grounding.calibration_actual_rate or 0.0) / 100.0


def _parse_any_prop_slug(slug: str) -> dict | None:
    """Tries every per-game-matchup sport family this bot trades: MLB (the
    shared sports_prop_quote_logger.py) and NFL (its own dedicated module,
    split out 2026-08-17 so NFL's per-game matchup props get their own data
    file -- see nfl_prop_quote_logger.py's docstring for why only that one
    NFL shape, out of three real ones, is tracked at all). Each parser
    already rejects slugs outside its own sport, so trying both in sequence
    is safe -- a slug can only ever match one."""
    meta = parse_prop_slug(slug)
    if meta is not None:
        return meta
    return parse_nfl_prop_slug(slug)


def _game_key(slug: str) -> tuple | None:
    meta = _parse_any_prop_slug(slug)
    if meta is None:
        return None
    return (meta["sport"], meta["matchup"], meta["date"])


def _bucket_key(sport: str, stat: str, side: str, price_cents: int) -> tuple:
    """Same decile grouping SportsPropCalibrationEngine itself uses to key
    calibration_actual_rate -- two simultaneously open positions landing in
    the same bucket are drawing on one shared measured win rate, not two
    independent edges. See module docstring's Portfolio construction note."""
    return (sport, stat, side, _decile_low(price_cents))


def _trade_bucket_key(trade: dict) -> tuple | None:
    meta = _parse_any_prop_slug(trade["market_slug"])
    if meta is None:
        return None
    return _bucket_key(meta["sport"], meta["stat"], trade["side"], trade["price_cents"])


class SportsPropsBot:
    def __init__(
        self,
        client,
        memory: ApexMemory | None = None,
        calibration: SportsPropCalibrationEngine | None = None,
        poll_interval: int = 300,
        allowed_sports: tuple[str, ...] = ("mlb", "nfl"),
    ):
        self.client = client
        self.memory = memory or ApexMemory()
        self.calibration = calibration or SportsPropCalibrationEngine()
        self.poll_interval = poll_interval
        self.allowed_sports = allowed_sports
        self._session_started = False

    # ── Position readback (same paranoia as apex/bot.py) ─────────────────────

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
                logger.debug("[sports_props] settlement check failed for %s: %s", slug, exc)
                continue
            if not market.closed:
                continue
            try:
                settlement = self.client.get_settlement(slug)
            except Exception as exc:
                logger.debug("[sports_props] no settlement yet for %s: %s", slug, exc)
                continue
            settlement_cents = _settlement_to_cents(settlement)
            if settlement_cents is None:
                continue
            outcome_yes = settlement_cents >= 50

            won = outcome_yes if trade["side"] == "yes" else (not outcome_yes)
            payout_cents = round(trade["count"] * 100) if won else 0
            pnl_cents = self.memory.record_settlement(trade["id"], outcome_yes, payout_cents)
            logger.info(
                "[sports_props] SETTLED %s -> %s  %sx %s  pnl=%+d¢  bankroll=$%.2f",
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
        """Same guard as apex/bot.py's, scoped to this bot's own market
        family. This account is shared with the ladder bot (astatc-*
        arbitrage locks) and Apex (tc-temp-* weather) -- their positions
        are none of this bot's concern and must not trigger a halt here.
        Apex is trivially excluded by slug prefix (tc-temp-* never matches
        _game_key), but the ladder bot trades the SAME astatc-mlb-*/
        astatc-nfl-* family this bot does -- slug shape alone can't tell
        them apart, so the shared account's currently-held slugs, as
        published every scan by the ladder bot's own routine position read
        (see bot.py's HELD_SLUGS_PATH -- despite the name, covers anything
        on the account, not just the ladder bot's own trades), are also
        subtracted before anything is treated as stray. Real incident
        2026-08-17: this exact gap halted the bot on 6 of the ladder bot's
        own lock legs, confirmed after the fact via the exchange's own
        manualOrderIndicator=AUTOMATIC on those fills.

        Also scoped to allowed_sports: since 2026-08-19 MLB and NFL each run
        as their own SportsPropsBot instance with their own ledger, sharing
        this same account. A position in the OTHER instance's sport is real
        and tracked -- just not by this one -- so it must never register as
        stray here either, the same way the ladder bot's own holdings don't."""
        try:
            live_positions = self.client.get_positions()
        except Exception as exc:
            logger.warning("[sports_props] position reconciliation check failed: %s", exc)
            return
        live_slugs = set()
        for p in live_positions:
            if abs(p.position) <= FILL_EPSILON:
                continue
            meta = _parse_any_prop_slug(p.market_slug)
            if meta is None or meta["sport"] not in self.allowed_sports:
                continue
            live_slugs.add(p.market_slug)
        tracked_slugs = {t["market_slug"] for t in self.memory.open_trades()}
        ladder_slugs = _ladder_bot_held_slugs()

        stray = live_slugs - tracked_slugs - ladder_slugs
        if stray:
            self.memory.halt(
                f"live account holds untracked sports-prop position(s) not in this bot's ledger: {sorted(stray)}"
            )
        attributed = (live_slugs - tracked_slugs) & ladder_slugs
        if attributed:
            logger.info(
                "[sports_props] %d live position(s) excluded from the stray check -- "
                "attributed to the ladder bot's own published holdings: %s",
                len(attributed),
                sorted(attributed),
            )

        missing = tracked_slugs - live_slugs
        if missing:
            logger.warning(
                "[sports_props] memory shows open position(s) the live account no longer holds: %s "
                "(may be a settlement not yet reconciled)",
                sorted(missing),
            )

    # ── Bucket selection ─────────────────────────────────────────────────────

    def _best_bucket_for_game(self, buckets: list) -> tuple | None:
        """Highest-edge tradeable bucket across every stat/player in one
        game, or None. Priced off the bulk scan's quotes as a coarse
        filter; the caller re-quotes the chosen candidate before sizing."""
        best = None  # (edge, side, slug, meta)
        for slug, meta, market in buckets:
            if market.yes_ask is not None:
                grounding = self.calibration.evaluate(meta["sport"], meta["stat"], "yes", market.yes_ask)
                if grounding.calibration_trustworthy:
                    edge = risk.edge_fraction(_model_prob(grounding), market.yes_ask)
                    if best is None or edge > best[0]:
                        best = (edge, "yes", slug, meta, grounding)
            if market.yes_bid is not None:
                no_price = 100 - market.yes_bid
                grounding = self.calibration.evaluate(meta["sport"], meta["stat"], "no", no_price)
                if grounding.calibration_trustworthy:
                    edge = risk.edge_fraction(_model_prob(grounding), no_price)
                    if best is None or edge > best[0]:
                        best = (edge, "no", slug, meta, grounding)
        return best

    def _requote_bucket(self, slug: str, side: str) -> tuple | None:
        try:
            fresh = self.client.get_market(slug)
        except Exception as exc:
            logger.warning("[sports_props] re-quote failed for %s: %s", slug, exc)
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
            logger.warning("[sports_props] %s order rejected on %s: %s", side.upper(), slug, exc)
            raw = self._position_after_order(slug, before) - before
            filled = raw if side == "yes" else -raw
            if abs(filled) <= FILL_EPSILON:
                return 0
            logger.error(
                "[sports_props] CRITICAL: %s order on %s reported an error but left %s contracts -- recording it",
                side.upper(),
                slug,
                _fmt_qty(filled),
            )
        else:
            raw = self._position_after_order(slug, before) - before
            filled = raw if side == "yes" else -raw
            if abs(filled) <= FILL_EPSILON:
                logger.info("[sports_props] %s leg on %s did not fill (IOC) -- no position taken", side.upper(), slug)
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

        grounding = self.calibration.evaluate(meta["sport"], meta["stat"], side, cost_price_cents)
        rationale = llm.generate_rationale(
            {
                "action": "buy",
                "market_slug": slug,
                "sport": meta["sport"],
                "stat": meta["stat"],
                "player": meta["player"],
                "model_prob": model_prob_for_side,
                "market_implied_prob": market_implied,
                "edge_pct": decision.edge_fraction,
                "calibration_note": grounding.calibration_note,
                "stake_cents": cost_cents,
                "count": filled,
                "reason": decision.reason,
                "bankroll_cents": self.memory.bankroll_cents,
                "recent_lessons": self.memory.recent_lessons(3),
            }
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
            rationale=rationale["rationale"],
        )
        logger.info(
            "[sports_props] %s %sx %s @ %d¢ (+%d¢ fee)  edge=%+.1f%%  cost=$%.2f  %s",
            side.upper(),
            _fmt_qty(filled),
            slug,
            cost_price_cents,
            fee_cents,
            decision.edge_fraction * 100,
            cost_cents / 100,
            rationale["rationale"],
        )
        return cost_cents

    # ── Scan loop ────────────────────────────────────────────────────────────

    def _scan_once(self) -> None:
        self.memory.roll_daily_window()
        self._reconcile_settlements()

        if self.memory.is_terminated():
            logger.error(
                "[sports_props] TERMINATED -- %s. Re-fund and reset the ledger by hand to resume.",
                self.memory.terminated_reason,
            )
            return

        self._reconcile_live_positions()
        if self.memory.is_halted():
            logger.error(
                "[sports_props] HALTED -- %s. Resolve by hand and clear anomaly_halted_reason "
                "in the ledger file to resume.",
                self.memory.anomaly_halted_reason,
            )
            return
        if self.memory.is_halted_today():
            logger.info("[sports_props] Halted for today -- %s. Observing only.", self.memory.daily_halted_reason)
            return

        # Reload before checking staleness -- same ordering bug this fixed
        # for Apex applies here identically (see calibration engine's own
        # reload/data_is_stale pairing).
        self.calibration.reload()
        if self.calibration.data_is_stale():
            logger.warning(
                "[sports_props] Calibration data is stale or missing -- is "
                "sports_prop_quote_logger.py running? Observing only this scan."
            )
            return

        try:
            markets = self.client.list_all_markets(active=True, closed=False)
        except Exception as exc:
            logger.warning("[sports_props] market scan failed: %s", exc)
            return

        buckets_by_game: dict[tuple, list] = {}
        for m in markets:
            if not m.slug.startswith("astatc-"):
                continue
            meta = _parse_any_prop_slug(m.slug)
            if meta is None or meta["sport"] not in self.allowed_sports:
                continue
            key = (meta["sport"], meta["matchup"], meta["date"])
            buckets_by_game.setdefault(key, []).append((m.slug, meta, m))
        if not buckets_by_game:
            logger.info("[sports_props] no open player-prop markets found this scan")
            return

        held_games = {_game_key(t["market_slug"]) for t in self.memory.open_trades()}
        held_buckets = {k for k in (_trade_bucket_key(t) for t in self.memory.open_trades()) if k is not None}

        try:
            balance_cents = self.client.get_balance()
        except Exception as exc:
            logger.warning("[sports_props] balance check failed: %s", exc)
            return
        self.memory.check_external_balance(balance_cents)
        if self.memory.is_terminated():
            logger.error("[sports_props] TERMINATED -- %s", self.memory.terminated_reason)
            return
        cash_cents = min(self.memory.cash_cents, balance_cents)

        for game, buckets in buckets_by_game.items():
            if game in held_games:
                continue
            candidate = self._best_bucket_for_game(buckets)
            if candidate is None:
                continue
            _coarse_edge, side, slug, meta, _stale_grounding = candidate

            requote = self._requote_bucket(slug, side)
            if requote is None:
                continue
            cost_price_cents, order_yes_price, depth = requote

            # Re-evaluate at the fresh requoted price, not the bulk-scan
            # price _best_bucket_for_game used -- calibration_actual_rate is
            # scoped to a 10-cent decile, so a price that moved deciles
            # between the scan and this requote would otherwise size a real
            # trade off a probability measured for a decile it's no longer
            # actually buying into (Apex doesn't have this exposure --
            # model_prob there comes from the NOAA forecast, independent of
            # price -- but this bot's model_prob IS the price's own decile
            # rate, so staleness here is a real, not just cosmetic, gap).
            grounding = self.calibration.evaluate(meta["sport"], meta["stat"], side, cost_price_cents)
            if not grounding.calibration_trustworthy:
                logger.debug(
                    "[sports_props] PASS %s (%s side): no longer trustworthy at the requoted price (%s)",
                    slug,
                    side,
                    grounding.calibration_note,
                )
                continue

            bucket_key = _bucket_key(meta["sport"], meta["stat"], side, cost_price_cents)
            if bucket_key in held_buckets:
                logger.info(
                    "[sports_props] PASS %s (%s side): already holding a position drawing on the same "
                    "(sport, stat, side, decile) signal %s -- not an independent bet",
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
                logger.debug("[sports_props] PASS %s (%s side): %s", slug, side, decision.reason)
                continue

            spent = self._place_decision(
                side, slug, meta, order_yes_price, cost_price_cents, model_prob_for_side, decision
            )
            cash_cents -= spent
            if spent > 0:
                # Same-cycle guard: a second, different game landing in this
                # exact bucket later in this same loop must also be blocked,
                # not just ones already open from a prior scan.
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
                logger.warning("[sports_props] scan error: %s", exc)
            if stop_event is not None:
                if stop_event.wait(timeout=self.poll_interval):
                    break
            else:
                time.sleep(self.poll_interval)
