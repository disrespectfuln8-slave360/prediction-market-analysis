"""Trading mode selection — defaults to paper (safe) unless explicitly overridden.

Set PAPER_TRADING=false in the environment/.env to place real orders. Any
other value (unset, "true", a typo) keeps the system in paper mode — a
misconfigured env var must never silently enable live trading.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_paper_mode() -> bool:
    return os.environ.get("PAPER_TRADING", "true").strip().lower() != "false"


def log_mode_banner(bot_name: str) -> None:
    if is_paper_mode():
        logger.info(
            "[%s] PAPER MODE — simulated fills only, no real orders will be placed. "
            "Set PAPER_TRADING=false to go live.",
            bot_name,
        )
    else:
        logger.warning(
            "[%s] *** LIVE MODE — REAL ORDERS WILL BE PLACED WITH REAL MONEY ***",
            bot_name,
        )
