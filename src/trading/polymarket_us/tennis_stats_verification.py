"""Independent tennis settlement verification against ESPN's public tennis
scoreboard/summary endpoints (site.api.espn.com, no key required) -- NOT
polymarket.us's own settlement determination.

Scope: TOTALS ONLY (atp/wta_{game,set}_totals). Confirmed live 2026-08-18
against a real, fully-verified match (Alex de Minaur bt Arthur Fery 7-5
7-6(7-3), Cincinnati Open): the real total (25 games, 2 sets) matches the
recorded settlement ladder exactly (tg-22.5->True, tg-25.5->False). Spread
markets (gs/ss) do NOT verify cleanly against the same real match under any
sign convention tried (player1-relative, favorite-relative, winner-
relative, home/away-relative) -- either a genuine settlement-accuracy
problem in spreads specifically (plausible, given the confirmed MLB
pattern) or a market structure this module doesn't understand without a
title to reference (these markets have empty titles, unlike MLB/NFL, so
the sign convention can't be confirmed the way it was there). Per explicit
user decision 2026-08-18: this module and tennis_props_bot.py both
deliberately do not cover spreads until that's resolved -- verify_settlement
returns None (inconclusive, caller falls back to trusting the exchange) for
any non-totals category rather than guess.

Player-name decoding: unlike MLB/NFL, polymarket.us's own asc-*/tsc-*
tennis markets have EMPTY titles (confirmed live), so client.get_market()
can't supply a real name the way it did there. Verification instead
reverse-matches the slug's two player codes against ESPN's own real
athlete names for candidate matches on that date -- see player_code()'s
docstring for the confirmed encoding (first token + last token, first 3
characters each, lowercased), validated against real examples: "juacer"
matches "Juan[...] Cerundolo", "felaug" matches "Felix Auger-Aliassime",
"artfer" matches "Arthur Fery".

READ-ONLY: only httpx GETs to site.api.espn.com are ever used. Never
places, cancels, or touches an order.
"""
from __future__ import annotations

from typing import Optional

import httpx

TOUR_PATHS = {"atp": "atp", "wta": "wta"}
_HTTP_TIMEOUT = 20


def player_code(full_name: str) -> str:
    """First 3 chars of the first name token + first 3 chars of the last
    name token, lowercased -- confirmed against real polymarket.us slugs
    cross-referenced with real player names (see module docstring). Names
    with a middle name/multiple tokens use the FIRST and LAST tokens only,
    skipping any middle ones (confirmed: "Juan Manuel Cerundolo" -> "jua" +
    "cer" = "juacer", not "jua"+"man")."""
    parts = full_name.split()
    if len(parts) < 2:
        return ""
    return (parts[0][:3] + parts[-1][:3]).lower()


def get_events_for(date: str, tour: str, cache: dict) -> list:
    """Raw ESPN scoreboard events for `tour` ("atp"/"wta") on `date`. Caller
    owns `cache` (keyed by (date, tour)), same reasoning as the MLB/NFL
    verification modules' caches."""
    key = (date, tour)
    if key not in cache:
        yyyymmdd = date.replace("-", "")
        r = httpx.get(f"https://site.api.espn.com/apis/site/v2/sports/tennis/{TOUR_PATHS[tour]}/scoreboard",
                       params={"dates": yyyymmdd}, timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
        cache[key] = r.json().get("events", [])
    return cache[key]


def find_match(date: str, tour: str, code1: str, code2: str, cache: dict) -> Optional[dict]:
    """The ESPN competition (match) dict on `date` whose two competitors'
    real names encode to {code1, code2}, or None if no such match is found
    (wrong date, player not in this tour's draw that day, or a code
    collision this simple scheme can't resolve -- all treated the same:
    unverifiable, caller falls back to trusting the exchange)."""
    wanted = {code1, code2}
    for event in get_events_for(date, tour, cache):
        for grouping in event.get("groupings", []):
            for comp in grouping.get("competitions", []):
                competitors = comp.get("competitors", [])
                if len(competitors) != 2:
                    continue
                codes = {player_code(c.get("athlete", {}).get("fullName", "")) for c in competitors}
                if codes == wanted:
                    return comp
    return None


def real_total_games(match: dict) -> Optional[int]:
    competitors = match.get("competitors", [])
    if len(competitors) != 2:
        return None
    total = 0
    for c in competitors:
        for ls in c.get("linescores", []):
            val = ls.get("value")
            if val is None:
                return None
            total += int(val)
    return total


def real_total_sets(match: dict) -> Optional[int]:
    competitors = match.get("competitors", [])
    if len(competitors) != 2:
        return None
    counts = {len(c.get("linescores", [])) for c in competitors}
    if len(counts) != 1:
        return None  # competitors disagree on set count -- data looks malformed, don't guess
    return counts.pop()


def verify_settlement(client, slug: str, meta: dict, cache: dict) -> Optional[bool]:
    """The REAL outcome_yes for a tennis TOTALS bucket (category ending in
    "_game_totals" or "_set_totals"), verified against ESPN's real match
    data, or None if it can't be determined -- including, deliberately, for
    any non-totals category (spreads), which this module does not attempt
    to verify at all (see module docstring). `client` is accepted only to
    match the same (client, slug, meta, cache) signature every other
    verify_settlement in this codebase uses -- unused here since player
    names come from ESPN's own data, not polymarket.us's (empty) titles."""
    category = meta.get("category", "")
    if category.endswith("_game_totals"):
        threshold_kind = "game"
    elif category.endswith("_set_totals"):
        threshold_kind = "set"
    else:
        return None  # spreads or unrecognized -- not covered, see module docstring

    tour = "atp" if category.startswith("atp_") else "wta" if category.startswith("wta_") else None
    if tour is None:
        return None

    matchup = meta.get("matchup")
    date = meta.get("date")
    if not matchup or not date:
        return None
    codes = matchup.split("-")
    if len(codes) != 2:
        return None

    try:
        match = find_match(date, tour, codes[0], codes[1], cache)
    except Exception:
        return None
    if match is None:
        return None

    real_val = real_total_games(match) if threshold_kind == "game" else real_total_sets(match)
    if real_val is None:
        return None
    threshold = meta.get("threshold")
    if threshold is None:
        return None
    return real_val >= threshold
