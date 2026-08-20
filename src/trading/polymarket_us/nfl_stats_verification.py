"""Independent NFL settlement verification against ESPN's public scoreboard/
summary endpoints (site.api.espn.com, no key required) -- NOT polymarket.us's
own settlement determination.

Same motivation as mlb_stats_verification.py (see that module's docstring
for the confirmed 14.2% MLB error rate this pattern exists to guard
against). A parallel 20-row spot-check of NFL settlements on 2026-08-18
found 18/20 explicitly verified correct and 0 confirmed errors -- NFL's
settlement pipeline looks meaningfully cleaner than MLB's on the sample
checked, but this is wired in anyway for the same reason a smoke detector
goes in every room, not just the one that already caught fire: nfl_prop_
quote_logger.py's own module docstring already flags this shared exchange
as the single common point of failure, and NFL's own historical sample size
(116 settlements at the time of that check) is too small to rule out a
lower-but-nonzero error rate with real confidence.

Stat definitions used by nfl_prop_quote_logger.py's tracked shape: ptd=
passing touchdowns, pyd=passing yards (both from ESPN's "passing" category),
recyd=receiving yards ("receiving"), ryd=rushing yards ("rushing"), td=any
touchdown scored by this player (rushing + receiving, explicitly excluding
the "passing" category's own TD column, which counts a QB's passing TDs,
not this player's own scores).

READ-ONLY: only httpx GETs to site.api.espn.com and the caller's own
client.get_market() are ever used. Never places, cancels, or touches an
order.
"""

from __future__ import annotations

import httpx

TEAM_CODE_TO_ABBR = {
    "den": "DEN",
    "atl": "ATL",
    "tb": "TB",
    "nyj": "NYJ",
    "mia": "MIA",
    "was": "WSH",
    "wsh": "WSH",
    "dal": "DAL",
    "sea": "SEA",
    "phi": "PHI",
    "bal": "BAL",
    "min": "MIN",
    "nyg": "NYG",
}

_HTTP_TIMEOUT = 20


def get_boxscore_for(date: str, team_codes: list, scoreboard_cache: dict):
    """The ESPN boxscore summary JSON for the first scoreboard event on
    `date` involving any of `team_codes`, or None. Caller owns
    `scoreboard_cache` (keyed by date), same reasoning as the MLB module's
    caches."""
    if date not in scoreboard_cache:
        yyyymmdd = date.replace("-", "")
        r = httpx.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
            params={"dates": yyyymmdd},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        scoreboard_cache[date] = r.json()
    data = scoreboard_cache[date]
    wanted = {TEAM_CODE_TO_ABBR.get(tc, tc.upper()) for tc in team_codes}
    for event in data.get("events", []):
        abbrs = {
            c.get("team", {}).get("abbreviation") for c in event.get("competitions", [{}])[0].get("competitors", [])
        }
        if wanted & abbrs:
            r2 = httpx.get(
                "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary",
                params={"event": event["id"]},
                timeout=_HTTP_TIMEOUT,
            )
            r2.raise_for_status()
            return r2.json()
    return None


def real_stat_value(boxscore: dict | None, full_name: str, stat_key: str) -> int | None:
    """The real value for `stat_key` from this game's box score, or None if
    the player can't be found in any relevant statistical category at all
    (unlike the MLB module, ESPN's boxscore only lists athletes who
    recorded at least one category's worth of stats -- there is no
    "rostered but empty" entry to distinguish DNP from not-found, so a
    genuine non-participant and a name-decode failure look the same here;
    callers must fall back to the exchange's settlement in both cases)."""
    if boxscore is None:
        return None
    players = boxscore.get("boxscore", {}).get("players", [])
    if stat_key == "td":
        total = 0
        found = False
        for team in players:
            for category in team.get("statistics", []):
                cat_name = category.get("name")
                labels = category.get("labels", [])
                for athlete in category.get("athletes", []):
                    if athlete.get("athlete", {}).get("displayName", "").lower() != full_name.lower():
                        continue
                    # The player appearing in ANY category (passing included)
                    # is enough to mark them found -- a passing-only player
                    # (e.g. a QB with no rush/catch that game) must still
                    # resolve to a real 0, not "undeterminable". Only the
                    # passing category's own TD column is excluded from the
                    # total: a QB's passing TDs aren't this player's "td" prop.
                    found = True
                    if cat_name == "passing":
                        continue
                    stats = dict(zip(labels, athlete.get("stats", [])))
                    for key in ("TD", "REC TD", "RUSH TD"):
                        if key in stats:
                            try:
                                total += int(stats[key])
                            except (ValueError, TypeError):
                                pass
        return total if found else None

    category_for_stat = {"ptd": "passing", "pyd": "passing", "recyd": "receiving", "ryd": "rushing"}
    label_for_stat = {"ptd": "TD", "pyd": "YDS", "recyd": "YDS", "ryd": "YDS"}
    wanted_category = category_for_stat.get(stat_key)
    if wanted_category is None:
        return None
    for team in players:
        for category in team.get("statistics", []):
            if category.get("name") != wanted_category:
                continue
            labels = category.get("labels", [])
            for athlete in category.get("athletes", []):
                if athlete.get("athlete", {}).get("displayName", "").lower() != full_name.lower():
                    continue
                stats = dict(zip(labels, athlete.get("stats", [])))
                val = stats.get(label_for_stat[stat_key])
                if val is None:
                    return None
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return None
    return None


def verify_settlement(client, slug: str, meta: dict, scoreboard_cache: dict) -> bool | None:
    """The REAL outcome_yes for an astatc-nfl-* per-game matchup prop,
    verified against ESPN's real box score, or None if undeterminable --
    callers must fall back to trusting the exchange's own settlement.
    Same `client`/`meta` contract as mlb_stats_verification.verify_settlement."""
    try:
        full_name = client.get_market(slug).title
    except Exception:
        return None
    try:
        box = get_boxscore_for(meta["date"], meta["matchup"].split("-"), scoreboard_cache)
    except Exception:
        return None
    real_val = real_stat_value(box, full_name, meta["stat"])
    if real_val is None:
        return None
    return real_val >= meta["threshold"]
