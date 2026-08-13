"""Apex — an autonomous, edge-gated trading agent for polymarket.us weather markets.

See src/trading/polymarket_us/apex/bot.py for the module-level design notes:
why this only trades markets this repo has a real calibrated model for, why
the LLM step (if enabled) never touches sizing or the buy/pass decision, and
how the bankroll/risk rules are enforced.
"""
