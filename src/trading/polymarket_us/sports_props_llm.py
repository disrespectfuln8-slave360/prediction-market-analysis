"""Optional LLM-authored rationale for an already-decided sports-props trade.

Same role and same guarantee as apex/llm.py, kept as a separate module
rather than reused directly: that one's template and system prompt
describe the NOAA forecast model by name, which would be a real, false
claim about what grounds a sports-props bet (see sports_prop_calibration.py
-- there is no independent forecast here, only the market's own historical
price-level accuracy). Everything else about the shape is identical:
called AFTER risk.evaluate() has already fixed the probability, edge, and
size; return value only ever fills display fields; never raises.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

RATIONALE_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "description": "1-3 sentence explanation of the decision, cold and probability-focused.",
        },
        "residual_risks": {
            "type": "string",
            "description": "1 sentence naming the main way this position could still lose.",
        },
    },
    "required": ["rationale", "residual_risks"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are an autonomous prediction-market trading agent for MLB/NFL player-prop "
    "markets. The trade decision below was already computed by deterministic code -- "
    "the probability, edge, and position size are fixed facts, not yours to "
    "second-guess, recompute, or restate as your own judgment. Your only job is a "
    "short, cold, probability-focused explanation of the decision: no emotion, no "
    "hedging filler, no repeating the numbers back verbatim. Reference what actually "
    "grounds the probability estimate -- this market's own historical price-level "
    "accuracy in this decile, not an independent forecast (there isn't one) and not "
    "player stats or matchup vibes."
)


def _template_rationale(context: dict) -> dict:
    if context["action"] == "buy":
        text = (
            f"Historical calibration gives this decile a {context['model_prob']:.0%} actual rate "
            f"({context['calibration_note']}); market implies {context['market_implied_prob']:.0%}. "
            f"Edge clears the floor after fees, sized at quarter-Kelly capped at the per-trade limit."
        )
    else:
        text = f"No trade: {context['reason']}."
    return {
        "rationale": text,
        "residual_risks": (
            "This is calibrated against the market's own past pricing, not an independent "
            "signal -- a real shift in how this category is priced going forward, or a small "
            "sample that hasn't yet revealed a bias, could still turn this into a loss."
        ),
    }


def generate_rationale(context: dict) -> dict:
    """context keys: action, market_slug, sport, stat, player, model_prob,
    market_implied_prob, edge_pct, calibration_note, stake_cents, count, reason,
    bankroll_cents, recent_lessons (list[str]).

    Returns {"rationale": str, "residual_risks": str}. Never raises -- same
    guarantee as apex/llm.py's generate_rationale, for the same reason (called
    after a real fill is already confirmed, before it's recorded in the ledger).
    """
    try:
        return _generate_rationale_inner(context)
    except Exception as exc:
        logger.warning("[sports_props] rationale generation failed unexpectedly (%s)", exc)
        return {
            "rationale": "(rationale unavailable due to an internal error -- see logs)",
            "residual_risks": "(unavailable)",
        }


def _generate_rationale_inner(context: dict) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _template_rationale(context)

    try:
        import anthropic
    except ImportError:
        logger.info("[sports_props] anthropic package not installed; using template rationale")
        return _template_rationale(context)

    user_content = (
        f"Market: {context['market_slug']} ({context['sport'].upper()} {context['stat']}, "
        f"player {context['player']})\n"
        f"Action: {context['action'].upper()}\n"
        f"Model probability: {context['model_prob']:.1%} ({context['calibration_note']})\n"
        f"Market-implied probability: {context['market_implied_prob']:.1%}\n"
        f"Edge after fees: {context['edge_pct']:+.1%}\n"
        f"Position size: ${context['stake_cents'] / 100:.2f} ({context['count']:g} contracts)\n"
        f"Bankroll: ${context['bankroll_cents'] / 100:.2f}\n"
        f"Reason/gate: {context['reason']}\n"
        f"Recent lessons: {'; '.join(context.get('recent_lessons') or []) or 'none logged yet'}"
    )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            thinking={"type": "disabled"},
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": RATIONALE_SCHEMA},
            },
            messages=[{"role": "user", "content": user_content}],
        )
        if response.stop_reason == "refusal":
            logger.warning("[sports_props] LLM rationale call refused, using template")
            return _template_rationale(context)
        text = next(b.text for b in response.content if b.type == "text")
        parsed = json.loads(text)
        return {"rationale": parsed["rationale"], "residual_risks": parsed["residual_risks"]}
    except Exception as exc:
        logger.warning("[sports_props] LLM rationale call failed (%s), using template", exc)
        return _template_rationale(context)
