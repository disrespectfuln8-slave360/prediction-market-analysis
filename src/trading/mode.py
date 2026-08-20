"""Trading mode selection — defaults to paper (safe) unless explicitly overridden.

Three independent platforms (Kalshi, Polymarket international, polymarket.us)
share this module but have separate real-money accounts with very different
funding states at any given time. A single global flag would mean flipping
one platform live also silently arms any other platform that happens to read
the same flag — confirmed almost happening in practice (Kalshi went live
while polymarket.us, which already held real funds, would have too). Each
platform gets its own override, checked before the global default, so going
live on one never implicitly goes live on another.

Set PLATFORM_PAPER_TRADING=false (e.g. KALSHI_PAPER_TRADING=false) to go live
on just that platform. The bare PAPER_TRADING flag is the fallback default
for any platform without its own override, and for callers that don't pass a
platform at all. Any unset/unrecognized value keeps that platform in paper
mode — a misconfigured env var must never silently enable live trading.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_paper_mode(platform: str | None = None) -> bool:
    if platform:
        override = os.environ.get(f"{platform.upper()}_PAPER_TRADING")
        if override is not None:
            return override.strip().lower() != "false"
    return os.environ.get("PAPER_TRADING", "true").strip().lower() != "false"


def log_mode_banner(bot_name: str, platform: str | None = None) -> None:
    if is_paper_mode(platform):
        logger.info(
            "[%s] PAPER MODE — simulated fills only, no real orders will be placed. "
            "Set %s_PAPER_TRADING=false to go live.",
            bot_name,
            (platform or "").upper() or "PAPER",
        )
    else:
        logger.warning(
            "[%s] *** LIVE MODE — REAL ORDERS WILL BE PLACED WITH REAL MONEY ***",
            bot_name,
        )
