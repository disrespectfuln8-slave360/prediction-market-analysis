"""Map Kalshi tickers to equivalent Polymarket markets.

Kalshi ticker format:
  KXATPMATCH-26JUL03SINBRO-SIN   → Sinner wins vs Brown, Jul 3
  KXWTAMATCH-26JUL04SNIKRU-SNI   → Swiatek wins vs Krejcikova, Jul 4
  KXMLBGAME-26JUL042210SDLAD-SD  → SD wins vs LAD at 22:10 Jul 4

Polymarket question format:
  "Will Jannik Sinner be the 2026 Men's Wimbledon winner?"
  "MLB: Who will win San Diego Padres v. Los Angeles Dodgers?"

Matching strategy:
  ATP/WTA: extract 3-letter player code from Kalshi → search Polymarket by
           player name using the code→name table below.
  MLB:     extract team code → search Polymarket by team name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.trading.polymarket.client import PolyMarket

# ── Player code tables ─────────────────────────────────────────────────────────

# Kalshi 3-letter code → (last_name, full_name_fragment)
ATP_CODES: dict[str, tuple[str, str]] = {
    "SIN": ("Sinner", "Jannik Sinner"),
    "DJO": ("Djokovic", "Novak Djokovic"),
    "ALC": ("Alcaraz", "Carlos Alcaraz"),
    "ZVE": ("Zverev", "Alexander Zverev"),
    "MED": ("Medvedev", "Daniil Medvedev"),
    "RUB": ("Rublev", "Andrey Rublev"),
    "FRI": ("Fritz", "Taylor Fritz"),
    "PAU": ("Paul", "Tommy Paul"),
    "HUB": ("Hurkacz", "Hubert Hurkacz"),
    "RIN": ("Rune", "Holger Rune"),
    "BER": ("Berrettini", "Matteo Berrettini"),
    "FON": ("Fonseca", "Joao Fonseca"),
    "MEN": ("Mensik", "Jakub Mensik"),
    "STR": ("Stricker", "Stricker"),
    "DIM": ("Dimitrov", "Grigor Dimitrov"),
    "FIL": ("Fils", "Arthur Fils"),
    "FER": ("Fearnley", "Fearnley"),
    "CHO": ("Cho", "Cho"),
    "COB": ("Cob", "Cobolli"),
    "DUC": ("Duckworth", "Duckworth"),
    "AUG": ("Augusiak", "Augusiak"),
    "ZHE": ("Zhu", "Zhu"),
    "MOC": ("Mochizuki", "Mochizuki"),
    "JOD": ("Jodi", "Jodi"),
    "DAV": ("Davidovich", "Davidovich"),
    "FUC": ("Fucsovics", "Fucsovics"),
    "SAF": ("Safiullin", "Safiullin"),
    "HUR": ("Hurschitt", "Hurschitt"),
    "LEH": ("Lehman", "Lehman"),
    "MOL": ("Moller", "Moller"),
    "TIA": ("Tiago", "Tiago"),
    "BRO": ("Brown", "Brown"),
}

WTA_CODES: dict[str, tuple[str, str]] = {
    "SWI": ("Swiatek", "Iga Swiatek"),
    "SAB": ("Sabalenka", "Aryna Sabalenka"),
    "GAU": ("Gauff", "Coco Gauff"),
    "RYB": ("Rybakina", "Elena Rybakina"),
    "PEG": ("Pegula", "Jessica Pegula"),
    "OST": ("Ostapenko", "Ostapenko"),
    "ANI": ("Anisimova", "Amanda Anisimova"),
    "KRE": ("Krejcikova", "Barbora Krejcikova"),
    "KRU": ("Krueger", "Madison Krueger"),
    "SHE": ("Sherif", "Sherif"),
    "EAL": ("Ealham", "Ealham"),
    "SNI": ("Snigur", "Snigur"),
    "SAM": ("Samir", "Samir"),
    "BOU": ("Bouchard", "Bouchard"),
    "JOV": ("Jovic", "Jovic"),
    "ALE": ("Alexandrova", "Alexandrova"),
    "BAR": ("Barthel", "Barthel"),
}

# MLB 2-3 letter code → full team name fragment for Polymarket search
MLB_CODES: dict[str, str] = {
    "SD": "San Diego Padres", "LAD": "Los Angeles Dodgers",
    "NYY": "New York Yankees", "BOS": "Boston Red Sox",
    "HOU": "Houston Astros",  "LAA": "Los Angeles Angels",
    "MIL": "Milwaukee Brewers", "AZ": "Arizona Diamondbacks",
    "MIA": "Miami Marlins",   "ATH": "Oakland Athletics",
    "NYM": "New York Mets",   "ATL": "Atlanta Braves",
    "PHI": "Philadelphia Phillies", "KC": "Kansas City Royals",
    "SF": "San Francisco Giants", "COL": "Colorado Rockies",
    "STL": "St. Louis Cardinals", "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox", "CLE": "Cleveland Guardians",
    "DET": "Detroit Tigers",  "MIN": "Minnesota Twins",
    "TOR": "Toronto Blue Jays", "BAL": "Baltimore Orioles",
    "TB": "Tampa Bay Rays",   "SEA": "Seattle Mariners",
    "OAK": "Oakland Athletics", "TEX": "Texas Rangers",
    "CIN": "Cincinnati Reds", "PIT": "Pittsburgh Pirates",
    "WSH": "Washington Nationals",
}


@dataclass
class MarketPair:
    """A matched (Kalshi, Polymarket) event pair."""
    kalshi_ticker: str
    kalshi_side: str          # "yes" = Kalshi team/player A wins
    poly_market: PolyMarket
    poly_side: str            # "yes" or "no" on Polymarket = same outcome
    label: str                # human description e.g. "Sinner beats Brown"


def parse_kalshi_ticker(ticker: str) -> Optional[dict]:
    """Extract structured info from a Kalshi sports ticker.

    Returns dict with keys: series, side_code, opp_code (ATP/WTA)
    or: series, team_code, opp_team (MLB).
    Returns None if ticker can't be parsed.
    """
    # KXATPMATCH-26JUL03SINBRO-SIN
    m = re.match(r"(KXATPMATCH|KXWTAMATCH)-(\d+\w+?)-([A-Z]{2,4})$", ticker)
    if m:
        series, date_match, side_code = m.groups()
        # Extract both player codes from date_match part
        # e.g. "26JUL03SINBRO" → middle part "SINBRO" = SIN + BRO
        inner = re.sub(r"^\d+\w{5}", "", date_match)  # strip leading date "26JUL03"
        codes = [inner[:3], inner[3:6]] if len(inner) >= 6 else [inner, "???"]
        opp_code = codes[1] if codes[0] == side_code else codes[0]
        return {
            "series": series,
            "side_code": side_code,
            "opp_code": opp_code,
            "sport": "ATP" if "ATP" in series else "WTA",
        }

    # KXMLBGAME-26JUL042210SDLAD-SD
    m2 = re.match(r"KXMLBGAME-\d+\w+?-([A-Z]{2,4})$", ticker)
    if m2:
        team_code = m2.group(1)
        # Extract both teams from the ticker
        inner = re.sub(r"^KXMLBGAME-\d+\w{8}", "", ticker)  # strip series+date
        opp_code = inner.replace(f"-{team_code}", "").replace("-", "")[-3:].upper()
        return {
            "series": "KXMLBGAME",
            "team_code": team_code,
            "opp_code": opp_code,
            "sport": "MLB",
        }

    return None


def find_polymarket_match(
    kalshi_ticker: str,
    poly_markets: list[PolyMarket],
) -> Optional[MarketPair]:
    """Find the best Polymarket market matching a Kalshi ticker.

    For tennis: Polymarket only has tournament-winner markets ("Will X win
    Wimbledon?"), NOT individual match markets.  The match market and the
    tournament-winner market are different events UNTIL the final, at which
    point "wins the tournament" == "wins this last match".  We return a match
    whenever the player names line up, but the caller should check whether
    the Kalshi market close time is within 24 h of the Polymarket end date
    (i.e. it's the final) before treating this as guaranteed arb.

    Returns None if no match found or the match has no active prices.
    """
    info = parse_kalshi_ticker(kalshi_ticker)
    if not info:
        return None

    sport = info.get("sport", "")

    if sport in ("ATP", "WTA"):
        codes = ATP_CODES if sport == "ATP" else WTA_CODES
        side_code = info.get("side_code", "")
        player_info = codes.get(side_code)
        if not player_info:
            return None
        last_name = player_info[0]

        # Look for a Polymarket market with this player's name
        for pm in poly_markets:
            q_lower = pm.question.lower()
            if last_name.lower() in q_lower and pm.yes_ask is not None and pm.yes_ask > 0:
                # Polymarket "Will X win?" → YES = X wins tournament/match
                return MarketPair(
                    kalshi_ticker=kalshi_ticker,
                    kalshi_side="yes",
                    poly_market=pm,
                    poly_side="yes",
                    label=f"{last_name} wins",
                )

    elif sport == "MLB":
        team_code = info.get("team_code", "")
        opp_code = info.get("opp_code", "")
        team_name = MLB_CODES.get(team_code, "")
        opp_name = MLB_CODES.get(opp_code, "")
        if not team_name or not opp_name:
            return None

        # Require BOTH teams' full names to appear — matching on city alone
        # (e.g. "los" for both the Dodgers and Angels, "san" for both the
        # Padres and Giants) can pair the wrong game.
        for pm in poly_markets:
            q_lower = pm.question.lower()
            if (
                team_name.lower() in q_lower
                and opp_name.lower() in q_lower
                and "mlb" in q_lower
                and pm.yes_ask is not None
                and pm.yes_ask > 0
            ):
                return MarketPair(
                    kalshi_ticker=kalshi_ticker,
                    kalshi_side="yes",
                    poly_market=pm,
                    poly_side="yes",
                    label=f"MLB {team_code} wins",
                )

    return None
