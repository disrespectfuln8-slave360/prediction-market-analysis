"""Optional LLM-authored rationale for an already-decided Apex trade.

This is the one place an LLM touches this bot, and its role is narrow on
purpose: risk.evaluate() has already computed the probability, edge, and
position size with deterministic code (see risk.py's module docstring for
why). This module is called AFTER that decision is final. Its return value
fills the display "Rationale"/"Residual risks" fields only -- it is never
parsed back into the trade/pass decision or the sizing math. If
ANTHROPIC_API_KEY isn't set, the `anthropic` package isn't installed, or the
call fails for any reason, a deterministic template renders the same fields
from the same facts, so Apex runs end-to-end without ever requiring this
credential.
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
            "description": "1-3 sentence explanation of the decision, in Apex's cold, probability-focused voice.",
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
    "You are Apex, an autonomous prediction-market trading agent. The trade decision "
    "below was already computed by deterministic code -- the probability, edge, and "
    "position size are fixed facts, not yours to second-guess, recompute, or restate "
    "as your own judgment. Your only job is a short, cold, probability-focused "
    "explanation of the decision in Apex's voice: no emotion, no hedging filler, no "
    "repeating the numbers back verbatim. Reference what actually grounds the "
    "probability estimate (the NOAA forecast model and its measured calibration "
    "against real settlements), not vibes."
)


def _template_rationale(context: dict) -> dict:
    if context["action"] == "buy":
        text = (
            f"NOAA-forecast model gives {context['model_prob']:.0%} on this bucket "
            f"({context['calibration_note']}); market implies {context['market_implied_prob']:.0%}. "
            f"Edge clears the floor after fees, sized at quarter-Kelly capped at the per-trade limit."
        )
    else:
        text = f"No trade: {context['reason']}."
    return {
        "rationale": text,
        "residual_risks": (
            "Forecast error, a late model update, or a resolution dispute could still turn this into a loss."
        ),
    }


def generate_rationale(context: dict) -> dict:
    """context keys: action, market_slug, city, low, high, model_prob,
    market_implied_prob, edge_pct, calibration_note, stake_cents, count, reason,
    bankroll_cents, recent_lessons (list[str]).

    Returns {"rationale": str, "residual_risks": str}. Never raises -- this
    is called after a real fill is already confirmed (see bot.py's
    _place_decision) and before that fill is recorded in memory.py's
    ledger, so an exception here must never be able to strand an
    already-filled position unrecorded. The wrapper is the actual
    guarantee; _generate_rationale_inner is allowed to raise (e.g. a
    missing context key hitting _template_rationale) without breaking it.
    """
    try:
        return _generate_rationale_inner(context)
    except Exception as exc:
        logger.warning("[apex] rationale generation failed unexpectedly (%s)", exc)
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
        logger.info("[apex] anthropic package not installed; using template rationale")
        return _template_rationale(context)

    user_content = (
        f"Market: {context['market_slug']} ({context['city']} daily high, bucket "
        f"{context['low']}-{context['high']}F)\n"
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
            logger.warning("[apex] LLM rationale call refused, using template")
            return _template_rationale(context)
        text = next(b.text for b in response.content if b.type == "text")
        parsed = json.loads(text)
        return {"rationale": parsed["rationale"], "residual_risks": parsed["residual_risks"]}
    except Exception as exc:
        logger.warning("[apex] LLM rationale call failed (%s), using template", exc)
        return _template_rationale(context)
