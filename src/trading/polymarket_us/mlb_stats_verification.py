"""Independent MLB settlement verification against the real, official MLB
Stats API (statsapi.mlb.com, no key required) -- NOT polymarket.us's own
settlement determination.

Why this exists
----------------
Confirmed live 2026-08-18: a 150-row random sample of astatc-mlb-* YES
settlements, checked against real MLB box scores, found 21 wrong (14.2%),
every one in the same direction (real outcome NO, recorded YES). 12 were
players who never appeared in the game at all (e.g. Nick Loftin, isOnBench
per the real box score, zero at-bats); 9 were players who played and
genuinely fell short of the threshold (e.g. Luis Robert Jr., 0 real hits on
a 2+ prop) but were still settled YES regardless. This is a bug in
polymarket.us's own settlement pipeline, not in this codebase or its data
collection -- there is nothing to patch on our end except to stop blindly
trusting it. sports_prop_quote_logger.py's _check_settlements() calls
verify_settlement() below before writing a row to sports_prop_settlements.
jsonl; when it disagrees with the exchange's own settlement, the verified
value wins (with a loud warning logged, never a silent override).

A separate live audit (also 2026-08-18) of the exact 3 calibration buckets
behind that day's 6 real trades found the errors did NOT flip any bucket's
trustworthiness determination -- so this fixes a real, ongoing source of
data corruption, not (by itself) whatever caused that specific day's
losses. Both problems are real and both needed fixing separately.

Stat definitions (confirmed 2026-08-17 against real box scores, one example
each): tb=totalBases, hits=hits, hr=homeRuns (all batting.*) -- hrr=hits+
runs+rbi combined (confirmed via Ke'Bryan Hayes: 1 hit+0 runs+0 rbi=1,
matching a recorded gte2->NO) -- outs=innings pitched*3 (pitching.
inningsPitched, e.g. "6.1"->19) -- k/er/ha/wa=pitching.{strikeOuts,
earnedRuns,hits,baseOnBalls}.

Player-name decoding: polymarket.us's own market title IS the player's real
full name even for closed/settled markets (confirmed live -- get_market()
on a long-closed slug still returns e.g. "Jake Cronenworth"), so this never
needs to guess at the {first3}{last3} slug-code scheme itself.

READ-ONLY: only httpx GETs to statsapi.mlb.com and the caller's own
client.get_market() are ever used. Never places, cancels, or touches an
order.
"""

from __future__ import annotations

import httpx

TEAM_CODE_TO_NAME = {
    "ari": "Diamondbacks",
    "az": "Diamondbacks",
    "atl": "Braves",
    "bal": "Orioles",
    "bos": "Red Sox",
    "chc": "Cubs",
    "cws": "White Sox",
    "cin": "Reds",
    "cle": "Guardians",
    "col": "Rockies",
    "det": "Tigers",
    "hou": "Astros",
    "kc": "Royals",
    "laa": "Angels",
    "lad": "Dodgers",
    "mia": "Marlins",
    "mil": "Brewers",
    "min": "Twins",
    "nym": "Mets",
    "nyy": "Yankees",
    "ath": "Athletics",
    "oak": "Athletics",
    "phi": "Phillies",
    "pit": "Pirates",
    "sd": "Padres",
    "sf": "Giants",
    "sea": "Mariners",
    "stl": "Cardinals",
    "tb": "Rays",
    "tex": "Rangers",
    "tor": "Blue Jays",
    "was": "Nationals",
    "wsh": "Nationals",
}

BATTER_STATS = {"tb": "totalBases", "hits": "hits", "hr": "homeRuns"}
PITCHER_STATS = {"k": "strikeOuts", "er": "earnedRuns", "ha": "hits", "wa": "baseOnBalls"}

_HTTP_TIMEOUT = 20


def get_boxscore_for(date: str, team_hint_names: list, schedule_cache: dict, boxscore_cache: dict):
    """(boxscore_json, away_name, home_name) for the first scheduled game on
    `date` involving any of `team_hint_names`, or (None, None, None). Caller
    owns the two cache dicts (keyed by date / gamePk) so a caller processing
    many rows for the same handful of games only pays for the API calls once
    -- there is no module-level cache since that would leak across callers
    with no way to bound it."""
    if date not in schedule_cache:
        r = httpx.get(
            "https://statsapi.mlb.com/api/v1/schedule", params={"sportId": 1, "date": date}, timeout=_HTTP_TIMEOUT
        )
        r.raise_for_status()
        schedule_cache[date] = r.json()
    data = schedule_cache[date]
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away = g["teams"]["away"]["team"]["name"]
            home = g["teams"]["home"]["team"]["name"]
            if any(h.lower() in away.lower() or h.lower() in home.lower() for h in team_hint_names):
                game_pk = g["gamePk"]
                if game_pk not in boxscore_cache:
                    rb = httpx.get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=_HTTP_TIMEOUT)
                    rb.raise_for_status()
                    boxscore_cache[game_pk] = rb.json()
                return boxscore_cache[game_pk], away, home
    return None, None, None


def find_player(boxscore: dict | None, full_name: str) -> dict | None:
    if boxscore is None:
        return None
    for side in ("away", "home"):
        for _pid, p in boxscore["teams"][side]["players"].items():
            if p.get("person", {}).get("fullName", "").lower() == full_name.lower():
                return p
    return None


def real_stat_value(player_entry: dict, stat_key: str) -> int | None:
    """The real value for `stat_key` from this game's box score entry, or
    None if the stat key isn't one this module knows how to read. An empty
    batting/pitching dict on a player who IS in the box score (rostered but
    didn't record any plate appearance or inning) correctly returns 0, not
    None -- that's real signal (they recorded zero of everything that
    game), not missing data. See module docstring: this exact distinction
    is what polymarket.us's own settlement pipeline gets wrong."""
    if stat_key in BATTER_STATS:
        b = player_entry.get("stats", {}).get("batting", {})
        return b.get(BATTER_STATS[stat_key], 0 if not b else None)
    if stat_key == "hrr":
        b = player_entry.get("stats", {}).get("batting", {})
        if not b:
            return 0
        return b.get("hits", 0) + b.get("runs", 0) + b.get("rbi", 0)
    if stat_key == "outs":
        p = player_entry.get("stats", {}).get("pitching", {})
        if not p:
            return 0
        ip = p.get("inningsPitched")
        if ip is None:
            return 0
        whole, _, frac = ip.partition(".")
        return int(whole) * 3 + int(frac or 0)
    if stat_key in PITCHER_STATS:
        p = player_entry.get("stats", {}).get("pitching", {})
        return p.get(PITCHER_STATS[stat_key], 0 if not p else None)
    return None


def verify_settlement(client, slug: str, meta: dict, schedule_cache: dict, boxscore_cache: dict) -> bool | None:
    """The REAL outcome_yes for an astatc-mlb-* player-prop bucket, verified
    against the official MLB Stats API, or None if it can't be determined
    (game not found, player not found in the box score at all, or the stat
    key is unrecognized) -- callers must fall back to trusting the
    exchange's own settlement in that case, not assume either direction.

    `meta` is the parsed slug dict (sport/matchup/date/stat/player/
    threshold, matching parse_prop_slug's shape); `client` is any object
    with a get_market(slug) -> Market method (used only for its .title,
    which is the player's real full name even on a closed market)."""
    try:
        full_name = client.get_market(slug).title
    except Exception:
        return None
    hints = [TEAM_CODE_TO_NAME.get(tc, tc) for tc in meta["matchup"].split("-")]
    try:
        box, _away, _home = get_boxscore_for(meta["date"], hints, schedule_cache, boxscore_cache)
    except Exception:
        return None
    player_entry = find_player(box, full_name)
    if player_entry is None:
        return None
    real_val = real_stat_value(player_entry, meta["stat"])
    if real_val is None:
        return None
    return real_val >= meta["threshold"]
