# Kalshi Bot Operating Manual

The rules that govern whether the Kalshi bots may trade, how much they may
risk, and when they must stop. This is the specification the code in
`src/trading/kalshi/` implements — where a rule here has a numeric value,
that value is enforced in code and covered by a test, not left to
discretion at 2am.

Read §1 before anything else. This is not a greenfield project: these bots
existed, traded real money, and were switched off on purpose.

---

## 1. Status and history

**Current status: no Kalshi bot is live. Nothing here is armed by default.**

The prior generation of Kalshi bots (`src/trading/kalshi/*`, `run_all.py`,
`run_kalshi_weather.py`) was deleted in commit `a55b95d` — "chore: remove
the Kalshi and Polymarket-international trading bots". The commit message
records that the scheduled tasks had already been unregistered "per
explicit request" and that only research code was deliberately kept:

- **Kept:** `src/analysis/kalshi/*` (research scripts) and
  `src/indexers/kalshi/*` (the read-only data collector). Neither places
  trades.
- **Removed:** everything that could send an order.

That decision is the baseline. Rebuilding execution code does **not**
re-authorize live trading; going live is a separate, explicit act gated by
§6.

### Why this matters for how the rebuild is written

The removed code was not theoretical — it ran against a funded account and
left a documented trail of ways this specific strategy loses money for
reasons that have nothing to do with being wrong about the world. Those are
catalogued in §4 and each one maps to a guardrail in §5.

---

## 2. What the bots actually do

Two distinct edge theses, which must be judged separately because their
evidence bases are completely different:

### 2a. Calibration mispricing (`strategy.py`)

Build a table of historical taker win rate by price from the local parquet
dataset (`data/kalshi/trades` joined to resolved `data/kalshi/markets`),
then buy any live market whose ask is enough cents below the historical win
rate at that price.

The claim is: *at price P, contracts of this type have historically
resolved YES more often than P% of the time, so paying P is +EV.*

### 2b. Structural locks (`find_locks`)

Buy both legs when `yes_ask + no_ask < 100`. If both fill, the pair pays
exactly $1 and cost less than $1 — profit is arithmetic, not forecast.

These are not the same kind of bet, and §3 grades them very differently.

---

## 3. Proven vs. Assumption

### Proven

- **Locks are real arithmetic.** If both legs fill at the quoted prices,
  `100 - (yes_ask + no_ask)` is guaranteed gross profit per pair. No
  calibration, no model, no forecast. The mechanism cannot be wrong.
- **Kalshi's fee formula is published and deterministic.** Fees are a
  known function of price and count, so breakeven is computable exactly
  rather than estimated.
- **Prediction markets are measurably miscalibrated at the extremes.**
  The favourite-longshot bias — cheap contracts resolving YES less often
  than their price implies — is one of the most replicated findings in the
  literature on betting and prediction markets, and
  `src/analysis/kalshi/win_rate_by_price.py` measures it directly on this
  dataset. *That the bias exists is proven.*

### Assumption

- **That a measured historical win-rate table predicts future win rates
  for markets you are about to trade.** This is the load-bearing
  assumption of the entire mispricing bot and it is *not* established.
  A calibration curve is a description of a past sample; trading it
  assumes stationarity that nobody has demonstrated here.
- **That the edge survives fees.** A 5pp raw edge is not a 5pp net edge.
  Fees are charged on both entry and (if any) exit, and at the prices
  where measured edges look largest the fee is proportionally largest too.
- **That a lock's second leg fills.** The arithmetic in §2b holds only if
  *both* legs fill at the quoted prices. A filled first leg plus an
  unfilled second leg is not an arbitrage — it is an unhedged directional
  bet that was sized as though it were riskless. This is the single most
  dangerous gap between "provably profitable" and "actually profitable"
  in this system.
- **That measured edge is not an artifact of thin samples.** Documented
  in `apex/risk.py`: every longshot bucket the sibling bots actually
  traded was also among the thinnest-sampled (8, 19, 29 historical rows —
  at or below the sample floor). Cheap prices are exactly where a handful
  of extra outcomes swings the measured rate by double digits.
- **That any of this is net-positive after the operator's time.** No
  claim is made here that these bots earn more than the hours spent
  babysitting them.

### Known false — do not rebuild these

- **A pooled (not per-side) win-rate table.** Verified against the live
  paper record: pooled rates mispredicted NO bets by roughly 2-3x (10-29¢
  NO bets won 9-12% of the time against a pooled prediction of 22-29%).
  YES-takers and NO-takers at the same price do not win at the same rate.
  Tables are per-side, always.
- **Inverting the YES table via `100 - no_ask` to price NO.** Same defect,
  same evidence.
- **Longshot bets.** Explicit operator decision, 2026-08-19: *"i dont want
  long shot strat, thats just setting myself up to fail."* The tradeable
  band is 30-70¢. This is a standing preference, not a tunable.

---

## 4. Incident ledger

Every one of these actually happened on a funded account. Each is now a
guardrail, not a lesson to re-learn.

| # | Incident | Root cause | Guardrail |
|---|---|---|---|
| 1 | Cancellations silently no-op'd for an entire live session | `cancel_order` used a deprecated v1 path returning 410; the error was swallowed. Orders stayed resting on the real book and later filled at stale prices. | v2 endpoint; cancel failures are logged and surfaced, never swallowed. Positions re-derived from the exchange every scan. |
| 2 | Bets logged as settled that were never actually at risk | Resting maker orders were recorded as bets on acceptance, not on fill. P&L was recorded for money that never moved. | A bet is recorded only on **confirmed fill**. Pending fills persist to disk across restarts. |
| 3 | Four adjacent YES brackets on one BTC close hour, plus contradictory YES/NO threshold bets on the same close | Per-scan dedup picked one bracket per event, but nothing stopped the *next* scan adding an adjacent, mutually exclusive bracket once prices drifted. | Event-level exposure blocking that persists across scans, plus `(normalized_series, close_time)` keying so sibling series (KXBTC / KXBTCD / KXBTC15M) collapse to one exposure. |
| 4 | Live OOM crash, 2026-08-15 | Three bot threads each independently ran the same unfiltered full-table win-rate scan on an ~85M-row dataset. | Process-wide cached global rates behind a lock; one scan, shared. |
| 5 | Locks spending ~2x the intended budget | Sized off `max(yes_ask, no_ask)` instead of the true pair cost `yes_ask + no_ask`. | Pair cost is the sizing basis, asserted in tests. |
| 6 | One platform going live silently arming another | A single global `PAPER_TRADING` flag shared across platforms. | Per-platform overrides (`KALSHI_PAPER_TRADING`); unset or unrecognized always means paper. |

---

## 5. Capital and risk rules

All values are enforced in `src/trading/kalshi/risk.py`.

### Position level

| Rule | Value | Rationale |
|---|---|---|
| Tradeable price band | 30-70¢ | §3; no longshots. Also the most stable ground on the calibration curve. |
| Minimum after-fee edge | 4pp | Below this the edge is inside the noise of the calibration estimate. |
| Kelly fraction | Quarter-Kelly (0.25×) | Full Kelly assumes the probability estimate is correct. Ours is an assumption (§3). |
| Max single position | 5% of bankroll | One market can never be more than a flesh wound. |
| Min stake | 25¢ | Below this the fee and ledger footprint exceed the point of the trade. |
| Min historical sample | 500 rows per (side, price) bucket | Incident-driven: the thin-sample buckets were the ones that looked most attractive. |

### Portfolio level

| Rule | Value | Rationale |
|---|---|---|
| Cash reserve | ≥40% of bankroll | Ensures the bot can never be fully deployed into a correlated cluster. |
| Max exposure per event | Capped per day, per event key | Incident #3. One real-world event, one position. |
| Daily loss limit | 10% of starting-day bankroll | Trips a same-day halt. |
| Drawdown kill switch | 25% from peak bankroll | Trips a **hard stop** requiring manual re-arm. Not a pause. |

### Category circuit breakers

| Trigger | Action |
|---|---|
| 4 consecutive losses in a series | Block that series |
| ≥15 settled bets at a negative return rate | Block that series as proven-negative |
| 3 days with no new settlement on a blocked series | Allow one probe (a block otherwise can never lift, since blocked series stop generating data) |

Position size is additionally scaled by realized track record — asymmetric
by design: a bad record cuts size to 0.75×, a good one raises it to at most
1.20×. Punish harder than you reward.

---

## 6. Go-live gate

Paper mode is the default and stays the default. **Every** condition below
must hold before `KALSHI_PAPER_TRADING=false`:

1. **Data**: `data/kalshi/{trades,markets}` present and current. The
   calibration table is the strategy; a stale table is a stale strategy.
   With no data the bot finds zero opportunities by construction — this is
   correct, not a bug to work around.
2. **Paper record**: ≥30 days continuous paper operation, ≥200 settled
   paper bets, positive net-of-fees return.
3. **Fee model verified**: modelled fees reconcile exactly against real
   fills from the account's own settlement history. Not the published
   formula alone — the observed charges. (This is exactly how the sibling
   platform's `THETA_TAKER` was pinned down; do the same here rather than
   inheriting its 0.06, which is *polymarket.us's* number, not Kalshi's.)
4. **Lock fill discipline proven**: measured single-leg-fill rate on paper,
   with the unhedged-leg handling exercised. If leg two routinely misses,
   the lock strategy is not an arbitrage and must not be sized as one.
5. **Capital is disposable**: the funded amount is money whose total loss
   changes nothing else. Per `docs/passive-income-os.md` §7, this stream is
   tuition, not income, until it has months of live evidence.
6. **Kill switch tested**: trip the drawdown halt deliberately in paper and
   confirm the bot stops and stays stopped.

A "yes" to all six authorizes a **small** live allocation, not a full one.

---

## 7. Operating rhythm

- **Daily**: check the halt state and the settlement log. A bot that
  halted and was not noticed for a week is the same as no bot.
- **Weekly**: net-of-fees P&L by series; predicted vs actual win rate per
  bucket. Divergence between them is the leading indicator that the
  calibration assumption (§3) is breaking down.
- **Monthly**: re-run the calibration analyses in `src/analysis/kalshi/`
  against fresh data and compare against the table the bot is trading on.
- **On any halt**: root-cause before re-arming. Re-arming without a cause
  is how incident #1 ran for a full session.

### Stop conditions — turn it off and mean it

- Drawdown kill switch trips (hard stop by definition).
- Predicted vs actual win rate diverges by >10pp across ≥100 settled bets:
  the calibration assumption has failed. This is the whole thesis breaking,
  not a bad week.
- Any incident in §4 recurs.
- You find yourself wanting to widen the price band, lower the edge floor,
  or raise the Kelly fraction after a losing stretch. That impulse is the
  signal to stop, not to tune.

---

## 8. Where each rule lives

| Concern | Module |
|---|---|
| Fee model (Kalshi's own schedule) | `src/trading/kalshi/fees.py` |
| Sizing, edge floor, price band, Kelly caps, cash reserve | `src/trading/kalshi/risk.py` |
| Daily loss limit, drawdown kill switch, halt state | `src/trading/kalshi/risk.py` (`RiskLedger`) |
| Calibration table, opportunity + lock discovery | `src/trading/kalshi/strategy.py` |
| Bet records, circuit breakers, confidence sizing | `src/trading/kalshi/learning.py` |
| Authenticated orders, cancels, positions | `src/trading/kalshi/trading_client.py` |
| Simulated fills and settlement | `src/trading/kalshi/paper_client.py` |
| Scan loop, exposure control, halt enforcement | `src/trading/kalshi/bot.py` |
| Launcher (paper by default) | `run_kalshi.py` |

---

## 9. The honest summary

The locks strategy rests on arithmetic and is sound *if* both legs fill —
which is an execution question, not a modelling one, and is the thing to
measure first.

The mispricing strategy rests on an assumption that a historical
calibration table predicts future outcomes. That assumption is unproven,
was traded on before with a documented history of losing to execution bugs
rather than to being wrong about the world, and is the reason every number
in §5 is conservative.

Treat the whole system as an instrument for *measuring* whether an edge
exists, that happens to be able to place orders — not as an income stream.
Size it accordingly, and let §6 rather than optimism decide when it trades
real money.
