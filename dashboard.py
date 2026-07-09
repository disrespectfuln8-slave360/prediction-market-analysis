"""Live trading terminal — http://localhost:5001"""
from __future__ import annotations
import json, re, time
from datetime import datetime
from zoneinfo import ZoneInfo
_PT = ZoneInfo("America/Los_Angeles")
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, Response, render_template_string
load_dotenv()
from src.trading.kalshi.trading_client import KalshiTradingClient

app = Flask(__name__)
BASE = Path(__file__).parent
LOG  = BASE / "bot_err.log"

# ── Backend helpers ────────────────────────────────────────────────────────────

def parse_placed():
    if not LOG.exists(): return []
    out, pat = [], re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Placed (YES|NO) on (\S+): (\d+)x @ (\d+).*edge=([\d.]+)pp")
    for l in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(l)
        if m:
            out.append({"time": m.group(1)[11:], "side": m.group(2), "ticker": m.group(3),
                        "count": int(m.group(4)), "price": int(m.group(5)), "edge": float(m.group(6))})
    return out[-200:]

def parse_balance_history():
    if not LOG.exists(): return []
    out, pat, seen = [], re.compile(r"(\d{2}:\d{2}:\d{2}).*balance=\$([\d.]+)"), set()
    for l in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(l)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append({"t": m.group(1), "v": float(m.group(2))})
    return out[-300:]

def parse_activity(n=60):
    """Rich activity feed from bot log."""
    if not LOG.exists(): return []
    out = []
    pat_placed = re.compile(r"(\d{2}:\d{2}:\d{2}).*Placed (YES|NO) on (\S+): (\d+)x @ (\d+).*edge=([\d.]+)pp")
    pat_lock   = re.compile(r"(\d{2}:\d{2}:\d{2}).*LOCK on (\S+): (\d+)x YES@(\d+)¢ \+ (\d+)x NO@(\d+)¢.*profit=\+(\d+)¢")
    pat_scan   = re.compile(r"(\d{2}:\d{2}:\d{2}).*\[(crypto|macro)\].*balance=\$([\d.]+).*markets=(\d+).*opps=(\d+).*slots=(\d+)")
    pat_reload = re.compile(r"(\d{2}:\d{2}:\d{2}).*Win rates loaded: (\d+) price")
    pat_err    = re.compile(r"(\d{2}:\d{2}:\d{2}).*ERROR.*")
    pat_cancel = re.compile(r"(\d{2}:\d{2}:\d{2}).*Cancelling stale order.*on (\S+)")
    for l in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat_lock.search(l)
        if m:
            out.append({"t": m.group(1), "type": "lock",
                        "msg": f"🔒 LOCK {m.group(2).split('-')[-1]} YES@{m.group(4)}¢ + NO@{m.group(6)}¢ = +{m.group(7)}¢ GUARANTEED"})
            continue
        m = pat_placed.search(l)
        if m:
            out.append({"t": m.group(1), "type": "bet",
                        "msg": f"BET {m.group(2)} {m.group(3).split('-')[-1]} {m.group(4)}x@{m.group(5)}¢ edge+{m.group(6)}pp"})
            continue
        m = pat_scan.search(l)
        if m:
            out.append({"t": m.group(1), "type": "scan",
                        "msg": f"[{m.group(2).upper()}] ${m.group(3)} · {m.group(5)} opps · {m.group(4)} mkts · {m.group(6)} slots"})
            continue
        m = pat_reload.search(l)
        if m:
            out.append({"t": m.group(1), "type": "model", "msg": f"MODEL RELOADED — {m.group(2)} price pts"})
            continue
        m = pat_cancel.search(l)
        if m:
            out.append({"t": m.group(1), "type": "cancel", "msg": f"CANCEL stale order on {m.group(2).split('-')[-1]}"})
            continue
        if "ERROR" in l:
            out.append({"t": l[11:19], "type": "error", "msg": l[50:120]})
    return out[-n:]

def parse_scan_stats():
    """Latest scan stats per bot."""
    if not LOG.exists(): return {}
    pat = re.compile(r"(\d{2}:\d{2}:\d{2}).*\[(\d{2}:\d{2}:\d{2})\] balance=\$([\d.]+) \(budget=\$([\d.]+)\)  markets=(\d+)  opps=(\d+)  positions=(\d+)  slots=(\d+)")
    # separate crypto (30s) and macro (90s)
    crypto_pat = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[(\d{2}:\d{2}:\d{2})\] balance=\$([\d.]+) \(budget=\$([\d.]+)\)  markets=(\d+)  opps=(\d+)  positions=(\d+)  slots=(\d+)")
    lines = LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
    last = {}
    for l in lines:
        m = crypto_pat.search(l)
        if m:
            last["scan"] = {"time": m.group(2), "bal": float(m.group(3)),
                            "budget": float(m.group(4)), "markets": int(m.group(5)),
                            "opps": int(m.group(6)), "positions": int(m.group(7)), "slots": int(m.group(8))}
    return last

def load_win_rates():
    try:
        import duckdb
        trades_dir  = BASE / "data" / "kalshi" / "trades"
        markets_dir = BASE / "data" / "kalshi" / "markets"
        if not list(trades_dir.glob("*.parquet")): return []
        with duckdb.connect() as con:
            df = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result FROM '{markets_dir}/*.parquet'
                    WHERE status='finalized' AND result IN ('yes','no')
                ),
                tp AS (
                    SELECT CASE WHEN t.taker_side='yes' THEN t.yes_price ELSE t.no_price END AS price,
                           CASE WHEN t.taker_side=m.result THEN 1 ELSE 0 END AS won
                    FROM '{trades_dir}/*.parquet' t
                    INNER JOIN resolved m ON t.ticker=m.ticker
                )
                SELECT price, ROUND(100.0*SUM(won)/COUNT(*),1) AS wr, COUNT(*) AS n
                FROM tp WHERE price BETWEEN 1 AND 99
                GROUP BY price HAVING COUNT(*)>=30 ORDER BY price
            """).df()
        return [{"p": int(r.price), "w": float(r.wr), "n": int(r.n)} for _, r in df.iterrows()]
    except Exception:
        return []

def fetch_state() -> dict:
    try:
        with KalshiTradingClient() as c:
            bal  = c.get_balance()
            pos  = c.get_positions()
            ords = c.get_open_orders()
            mkts = c.list_liquid_open_markets()
    except Exception as e:
        return {"error": str(e)}

    mkt_map   = {m.ticker: m for m in mkts}
    placed    = parse_placed()
    bal_hist  = parse_balance_history()
    activity  = parse_activity()
    scan      = parse_scan_stats()
    yes_ct    = sum(1 for b in placed if b["side"] == "YES")
    start_bal = bal_hist[0]["v"] if bal_hist else bal / 100

    positions_out = []
    for p in pos:
        m = mkt_map.get(p.ticker)
        positions_out.append({
            "ticker": p.ticker,
            "side":   "YES" if p.position > 0 else "NO",
            "qty":    abs(p.position),
            "close_ts": int(m.close_time.timestamp()) if m and m.close_time else None,
            "close_str": m.close_time.astimezone(_PT).strftime("%H:%M PST") if m and m.close_time else "—",
            "yes_bid": m.yes_bid if m else None, "yes_ask": m.yes_ask if m else None,
        })

    return {
        "ts":        datetime.now(_PT).strftime("%H:%M:%S PST"),
        "ts_epoch":  int(time.time()),
        "balance":   bal / 100,
        "pnl":       round(bal / 100 - start_bal, 2),
        "positions": positions_out,
        "open_orders": [{"ticker": o.ticker, "side": o.side.upper(), "count": o.count, "price": o.yes_price} for o in ords],
        "markets":   sorted([{"ticker": m.ticker, "yes_bid": m.yes_bid, "yes_ask": m.yes_ask,
                               "no_ask": m.no_ask, "vol": m.volume_24h,
                               "close": m.close_time.astimezone(_PT).strftime("%H:%M") if m.close_time else "—"} for m in mkts],
                            key=lambda x: x["vol"] or 0, reverse=True)[:40],
        "placed":    placed,
        "bal_hist":  bal_hist,
        "yes_count": yes_ct,
        "no_count":  len(placed) - yes_ct,
        "activity":  activity,
        "scan":      scan,
        "win_rates": load_win_rates(),
    }

# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KALSHI BOT TERMINAL</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#020406;--bg2:#060c10;--bg3:#0a141a;
  --border:#0d1e28;--border2:#163044;
  --green:#00ff88;--green2:#00cc66;--green3:#002a18;
  --red:#ff2d55;--red2:#cc2244;--red3:#2a0010;
  --amber:#f5c518;--amber2:#c49a10;--amber3:#1e1800;
  --blue:#00aaff;--blue2:#0077cc;--blue3:#001830;
  --purple:#bc8cff;--cyan:#00e5ff;
  --muted:#2a4a5a;--muted2:#4a7a92;--text:#c0dce8;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:12px;overflow-x:hidden}
body::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.06) 2px,rgba(0,0,0,.06) 4px);pointer-events:none;z-index:9999}

/* ── TAPE ── */
.tape{height:22px;background:#000;border-bottom:1px solid #001a08;overflow:hidden;display:flex;align-items:center}
.tape-inner{display:flex;white-space:nowrap;animation:tape 60s linear infinite}
.tape-inner:hover{animation-play-state:paused}
@keyframes tape{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.ti{display:inline-flex;gap:6px;padding:0 16px;border-right:1px solid #0d1e28;font-size:10px}
.ti-y{color:var(--green)} .ti-n{color:var(--red)} .ti-e{color:var(--amber)}

/* ── HEADER ── */
.hdr{display:grid;grid-template-columns:repeat(7,1fr);gap:1px;background:var(--border);border-bottom:2px solid var(--border2)}
.hc{background:var(--bg2);padding:8px 12px;display:flex;flex-direction:column;gap:1px}
.hv{font-size:22px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.hl{font-size:9px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted2);margin-top:1px}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:5px}
.live{background:var(--green);box-shadow:0 0 10px var(--green);animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}

/* ── GRID ── */
.main{display:grid;gap:1px;background:var(--border)}
.row{display:grid;gap:1px;background:var(--border)}
.r2{grid-template-columns:3fr 2fr}
.r2b{grid-template-columns:2fr 3fr}
.r3{grid-template-columns:1fr 1fr 1fr}
.r4{grid-template-columns:1fr 1fr 1fr 1fr}
.panel{background:var(--bg2);padding:12px;position:relative;overflow:hidden}
.ph{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid var(--border2)}
.ph h2{font-size:9px;text-transform:uppercase;letter-spacing:.15em;color:var(--muted2);font-weight:600}
.badge{font-size:10px;padding:1px 8px;border-radius:3px;border:1px solid}
.bg{color:var(--green);background:var(--green3);border-color:var(--green2)}
.br{color:var(--red);background:var(--red3);border-color:var(--red2)}
.ba{color:var(--amber);background:var(--amber3);border-color:var(--amber2)}
.bb{color:var(--blue);background:var(--blue3);border-color:var(--blue2)}

/* ── CHARTS ── */
canvas{display:block}

/* ── POSITIONS ── */
.pos-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:6px}
.pos-card{border-radius:5px;padding:8px 10px;border:1px solid;position:relative;overflow:hidden}
.pos-card.yes{background:linear-gradient(135deg,#002a18,#001a10);border-color:var(--green2)}
.pos-card.no{background:linear-gradient(135deg,#2a0010,#1a0008);border-color:var(--red2)}
.pos-ticker{font-size:9px;opacity:.6;margin-bottom:3px}
.pos-side{font-size:13px;font-weight:800}
.pos-qty{font-size:10px;opacity:.7;margin-top:1px}
.pos-timer{font-size:11px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}
.pos-card.yes .pos-side{color:var(--green)}
.pos-card.no  .pos-side{color:var(--red)}
.pos-card.yes .pos-timer{color:var(--green)}
.pos-card.no  .pos-timer{color:var(--red)}
.pos-glow-yes{position:absolute;inset:0;background:radial-gradient(ellipse at 0% 50%,rgba(0,255,136,.08),transparent 70%);pointer-events:none}
.pos-glow-no{position:absolute;inset:0;background:radial-gradient(ellipse at 0% 50%,rgba(255,45,85,.08),transparent 70%);pointer-events:none}

/* ── ACTIVITY FEED ── */
.feed{height:320px;overflow-y:auto;display:flex;flex-direction:column-reverse;gap:1px}
.feed-item{padding:4px 8px;border-radius:3px;font-size:10px;display:flex;gap:8px;align-items:baseline;border-left:2px solid transparent;flex-shrink:0}
.feed-item.bet{background:rgba(0,255,136,.06);border-color:var(--green2);color:var(--green)}
.feed-item.lock{background:rgba(188,140,255,.12);border-color:var(--purple);color:var(--purple);font-weight:700}
.feed-item.scan{background:rgba(0,170,255,.04);border-color:var(--blue2);color:var(--muted2)}
.feed-item.model{background:rgba(245,197,24,.06);border-color:var(--amber2);color:var(--amber)}
.feed-item.cancel{background:rgba(255,45,85,.04);border-color:var(--red2);color:var(--red)}
.feed-item.error{background:rgba(255,45,85,.1);border-color:var(--red);color:var(--red)}
.feed-time{color:var(--muted2);font-size:9px;flex-shrink:0;width:54px}
.feed-msg{flex:1;line-height:1.5}

/* ── SCANNER ── */
.scanner{display:flex;flex-direction:column;gap:8px;margin-top:6px}
.sc-row{display:flex;justify-content:space-between;align-items:center;padding:6px 10px;background:var(--bg3);border-radius:4px;border:1px solid var(--border2)}
.sc-label{font-size:9px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted2)}
.sc-val{font-size:16px;font-weight:800;font-variant-numeric:tabular-nums}
.sc-bar-wrap{height:3px;background:var(--border2);border-radius:2px;margin-top:4px;overflow:hidden}
.sc-bar{height:100%;border-radius:2px;transition:width .6s ease}

/* ── TABLES ── */
.tbl-wrap{overflow-y:auto;max-height:200px}
table{width:100%;border-collapse:collapse}
th{color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.08em;padding:3px 6px;border-bottom:1px solid var(--border2);position:sticky;top:0;background:var(--bg2)}
td{padding:4px 6px;border-bottom:1px solid #080e12;font-size:11px}
tr:last-child td{border:none}
tr:hover td{background:rgba(0,255,136,.03)}
.cy{color:var(--green);font-weight:700} .cn{color:var(--red);font-weight:700}
.ca{color:var(--amber)} .cm{color:var(--muted2)} .cb{color:var(--blue)}

/* ── WIN RATE ── */
.wr-wrap{position:relative;height:160px}

/* ── FILLS ── */
.fills-bar{height:14px;border-radius:3px;overflow:hidden;display:flex;border:1px solid var(--border2);margin:6px 0}
.fills-y{background:var(--green2);transition:width .6s}
.fills-n{background:var(--red2);transition:width .6s}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:6px}
.stat{background:var(--bg3);border:1px solid var(--border2);border-radius:4px;padding:7px 10px;text-align:center}
.stat .v{font-size:18px;font-weight:800} .stat .l{font-size:8px;color:var(--muted2);text-transform:uppercase;letter-spacing:.08em;margin-top:1px}

/* ── MESH ── */
#meshCanvas{width:100%;display:block}
.mesh-wrap{background:var(--bg);border-top:1px solid var(--border);border-bottom:1px solid var(--border)}
.mesh-lbl{display:flex;justify-content:space-around;font-size:8px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;padding:3px 0;background:var(--bg)}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}

/* ── GLOW ── */
.gg{text-shadow:0 0 14px var(--green)} .gr{text-shadow:0 0 14px var(--red)} .ga{text-shadow:0 0 14px var(--amber)}
.cy-g{color:var(--green)} .cy-r{color:var(--red)} .cy-a{color:var(--amber)} .cy-b{color:var(--blue)}

/* ── BLINK NEW ── */
@keyframes newbet{0%{background:rgba(0,255,136,.25)}100%{background:rgba(0,255,136,.06)}}
.bet-new{animation:newbet .8s ease-out}
</style>
</head>
<body>

<!-- TAPE -->
<div class="tape"><div class="tape-inner" id="tape"></div></div>

<!-- HEADER -->
<div class="hdr">
  <div class="hc">
    <div class="hv cy-g gg" id="h-bal">$—</div>
    <div class="hl"><span class="dot live"></span>CASH BALANCE</div>
  </div>
  <div class="hc">
    <div class="hv" id="h-pnl">—</div>
    <div class="hl">ALL-TIME P&amp;L</div>
  </div>
  <div class="hc">
    <div class="hv cy-b" id="h-bets">—</div>
    <div class="hl">TOTAL BETS</div>
  </div>
  <div class="hc">
    <div class="hv cy-a" id="h-opps">—</div>
    <div class="hl">LIVE OPPS</div>
  </div>
  <div class="hc">
    <div class="hv" id="h-pos">—</div>
    <div class="hl">OPEN POSITIONS</div>
  </div>
  <div class="hc">
    <div class="hv cy-a" id="h-edge">—</div>
    <div class="hl">AVG EDGE</div>
  </div>
  <div class="hc">
    <div class="hv cm" id="h-ts">—</div>
    <div class="hl">LAST UPDATE</div>
  </div>
</div>

<div class="main">

  <!-- ROW 1: balance chart + activity feed -->
  <div class="row r2">
    <div class="panel">
      <div class="ph">
        <h2>Portfolio Balance &amp; P&amp;L</h2>
        <div style="display:flex;gap:6px">
          <span class="badge ba" id="b-range">—</span>
          <span class="badge" id="b-pnl-badge">—</span>
        </div>
      </div>
      <div style="height:220px;position:relative"><canvas id="balChart"></canvas></div>
    </div>
    <div class="panel">
      <div class="ph">
        <h2>Live Activity Feed</h2>
        <span class="badge bb" id="b-act-ct">—</span>
      </div>
      <div class="feed" id="feed"></div>
    </div>
  </div>

  <!-- ROW 2: neural mesh -->
  <div style="background:var(--bg2)">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:5px 14px;border-bottom:1px solid var(--border2)">
      <span style="font-size:9px;text-transform:uppercase;letter-spacing:.14em;color:var(--muted2)">Fair-Value Signal Mesh</span>
      <span style="font-size:9px;color:var(--muted)">MARKET FEED → WIN-RATE MODEL → EDGE CALC → ORDER OUTPUT</span>
      <span id="eval-sec" style="font-size:9px;color:var(--amber)">EVAL/SEC: —</span>
    </div>
    <div class="mesh-wrap"><canvas id="meshCanvas" height="200"></canvas></div>
    <div class="mesh-lbl">
      <span>L1 · INPUTS<br><span style="color:var(--amber)">MARKET FEED</span></span>
      <span>L2 · MODEL<br><span style="color:var(--amber)">WIN-RATE P(UP)</span></span>
      <span>L3 · EDGE<br><span style="color:var(--amber)">FAIR-VALUE GAP</span></span>
      <span>L4 · OUTPUT<br><span style="color:var(--amber)">ORDER SIGNAL</span></span>
    </div>
  </div>

  <!-- ROW 3: positions + scanner + win rate -->
  <div class="row r3">
    <div class="panel">
      <div class="ph"><h2>Open Positions</h2><span class="badge ba" id="b-pos-ct">0</span></div>
      <div class="pos-grid" id="pos-grid">
        <div style="color:var(--muted);font-size:10px;grid-column:1/-1;text-align:center;padding:20px 0">No open positions</div>
      </div>
    </div>
    <div class="panel">
      <div class="ph"><h2>Bot Scanner</h2><span class="badge bb" id="b-scan-time">—</span></div>
      <div class="scanner">
        <div class="sc-row">
          <div><div class="sc-label">Opportunities Found</div>
          <div class="sc-val cy-g" id="sc-opps">—</div>
          <div class="sc-bar-wrap"><div class="sc-bar" id="sc-opps-bar" style="background:var(--green);width:0%"></div></div></div>
        </div>
        <div class="sc-row">
          <div><div class="sc-label">Markets Scanned</div>
          <div class="sc-val cy-b" id="sc-mkts">—</div>
          <div class="sc-bar-wrap"><div class="sc-bar" id="sc-mkts-bar" style="background:var(--blue);width:0%"></div></div></div>
        </div>
        <div class="sc-row">
          <div><div class="sc-label">Open Slots</div>
          <div class="sc-val cy-a" id="sc-slots">—</div>
          <div class="sc-bar-wrap"><div class="sc-bar" id="sc-slots-bar" style="background:var(--amber);width:0%"></div></div></div>
        </div>
        <div class="sc-row">
          <div><div class="sc-label">Budget Available</div>
          <div class="sc-val" id="sc-budget">—</div></div>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="ph">
        <h2>Win-Rate Model</h2>
        <span class="badge bb" id="b-wr-pts">— pts</span>
      </div>
      <div class="wr-wrap"><canvas id="wrChart"></canvas></div>
    </div>
  </div>

  <!-- ROW 4: fills + bets table + market feed -->
  <div class="row r2b">
    <div class="panel">
      <div class="ph"><h2>Fills Breakdown</h2></div>
      <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px">
        <span class="cy-g" id="yes-pct">YES —%</span>
        <span class="cy-r" id="no-pct">NO —%</span>
      </div>
      <div class="fills-bar">
        <div class="fills-y" id="fy" style="width:50%"></div>
        <div class="fills-n" id="fn" style="width:50%"></div>
      </div>
      <div class="stat-grid">
        <div class="stat"><div class="v cy-g" id="s-yes">—</div><div class="l">YES bets</div></div>
        <div class="stat"><div class="v cy-r" id="s-no">—</div><div class="l">NO bets</div></div>
        <div class="stat"><div class="v cy-a" id="s-edge">—</div><div class="l">Avg edge</div></div>
        <div class="stat"><div class="v cy-b" id="s-price">—</div><div class="l">Avg price</div></div>
      </div>
      <div style="margin-top:10px">
        <div class="ph"><h2>Recent Bets</h2><span class="badge ba" id="b-bets">0</span></div>
        <div class="tbl-wrap">
          <table id="bets-tbl">
            <thead><tr><th>Time</th><th>Market</th><th>Side</th><th>Qty</th><th>Price</th><th>Edge</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="ph"><h2>Live Market Feed</h2><span class="badge ba" id="b-mkt-ct">—</span></div>
      <div style="height:180px;position:relative"><canvas id="mktChart"></canvas></div>
      <div class="tbl-wrap" style="max-height:165px;margin-top:8px">
        <table id="mkt-tbl">
          <thead><tr><th>Market</th><th>YES Bid</th><th>YES Ask</th><th>NO Ask</th><th>Spread</th><th>Vol 24h</th><th>Close</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

</div>

<script>
// ── CHARTS ────────────────────────────────────────────────────────────────────
const balCtx=document.getElementById('balChart').getContext('2d');
const balGrad=balCtx.createLinearGradient(0,0,0,220);
balGrad.addColorStop(0,'rgba(0,255,136,.35)');balGrad.addColorStop(1,'rgba(0,255,136,0)');
const pnlGrad=balCtx.createLinearGradient(0,0,0,220);
pnlGrad.addColorStop(0,'rgba(245,197,24,.2)');pnlGrad.addColorStop(1,'rgba(245,197,24,0)');
const balChart=new Chart(balCtx,{
  type:'line',
  data:{labels:[],datasets:[
    {label:'Balance',data:[],borderColor:'#00ff88',backgroundColor:balGrad,borderWidth:2.5,tension:.4,pointRadius:0,fill:true,yAxisID:'y'},
    {label:'P&L',data:[],borderColor:'#f5c518',backgroundColor:pnlGrad,borderWidth:1.5,tension:.4,pointRadius:0,fill:true,yAxisID:'y2',borderDash:[4,3]},
  ]},
  options:{responsive:true,maintainAspectRatio:false,
    animation:{duration:500,easing:'easeInOutCubic'},
    interaction:{mode:'index',intersect:false},
    plugins:{legend:{labels:{color:'#4a7a92',font:{size:9},boxWidth:10,padding:8}},
             tooltip:{backgroundColor:'#060c10',borderColor:'#0d1e28',borderWidth:1,callbacks:{label:c=>' $'+c.parsed.y.toFixed(2)}}},
    scales:{
      x:{ticks:{maxTicksLimit:8,color:'#2a4a5a',font:{size:9}},grid:{color:'#060c10'},border:{color:'#0d1e28'}},
      y:{ticks:{color:'#2a4a5a',font:{size:9},callback:v=>'$'+v.toFixed(2)},grid:{color:'#06101a'},border:{color:'#0d1e28'},position:'left'},
      y2:{ticks:{color:'#4a3a0a',font:{size:9},callback:v=>(v>=0?'+':'')+v.toFixed(2)},grid:{display:false},border:{color:'#0d1e28'},position:'right'}
    }
  }
});

const wrCtx=document.getElementById('wrChart').getContext('2d');
const wrGrad=wrCtx.createLinearGradient(0,0,0,160);
wrGrad.addColorStop(0,'rgba(0,170,255,.3)');wrGrad.addColorStop(1,'rgba(0,170,255,0)');
const wrChart=new Chart(wrCtx,{
  type:'line',
  data:{labels:[],datasets:[
    {label:'Win Rate',data:[],borderColor:'#00aaff',backgroundColor:wrGrad,borderWidth:2,tension:.3,pointRadius:0,fill:true},
    {label:'Fair Value',data:[],borderColor:'rgba(245,197,24,.5)',borderWidth:1,tension:0,pointRadius:0,fill:false,borderDash:[3,3]},
  ]},
  options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
    interaction:{mode:'index',intersect:false},
    plugins:{legend:{labels:{color:'#4a7a92',font:{size:9},boxWidth:10,padding:6}},
             tooltip:{backgroundColor:'#060c10',borderColor:'#0d1e28',borderWidth:1,callbacks:{label:c=>` ${c.dataset.label}: ${c.parsed.y.toFixed(1)}%`}}},
    scales:{
      x:{ticks:{maxTicksLimit:10,color:'#2a4a5a',font:{size:8}},grid:{color:'#060c10'}},
      y:{ticks:{color:'#2a4a5a',font:{size:9},callback:v=>v+'%'},grid:{color:'#06101a'},min:0,max:100}
    }
  }
});

const mktCtx=document.getElementById('mktChart').getContext('2d');
const mktChart=new Chart(mktCtx,{
  type:'bar',
  data:{labels:[],datasets:[
    {label:'YES Ask',data:[],backgroundColor:'rgba(0,255,136,.45)',borderColor:'#00ff88',borderWidth:1},
    {label:'NO Ask', data:[],backgroundColor:'rgba(255,45,85,.45)', borderColor:'#ff2d55',borderWidth:1},
  ]},
  options:{responsive:true,maintainAspectRatio:false,animation:{duration:300},
    plugins:{legend:{labels:{color:'#4a7a92',font:{size:9},boxWidth:8}},tooltip:{backgroundColor:'#060c10',borderColor:'#0d1e28',borderWidth:1}},
    scales:{
      x:{ticks:{maxTicksLimit:12,color:'#2a4a5a',font:{size:8},maxRotation:45},grid:{color:'#060c10'}},
      y:{ticks:{color:'#2a4a5a',font:{size:9},callback:v=>v+'¢'},grid:{color:'#06101a'},max:100,min:0}
    }
  }
});

// ── NEURAL MESH ───────────────────────────────────────────────────────────────
const mesh=document.getElementById('meshCanvas');
const mCtx=mesh.getContext('2d');
let meshW,meshH,nodes,particles=[],lastBetCount=0;
const LAYERS=[
  {keys:['KXBTC','KXETH','KXSOL','KXDOGE','KXINX','FED','CPI','JOBS']},
  {keys:['p10','p20','p30','p40','p50','p60','p70','p80','p90','p95']},
  {keys:['e+5','e+3','e+2','e+1','e-0','e-2']},
  {keys:['YES BUY','NO BUY','HOLD']},
];
function buildNodes(){
  meshW=mesh.width=mesh.offsetWidth;meshH=mesh.height=200;
  nodes=LAYERS.map((layer,li)=>{
    const x=meshW*(.1+li*.27);
    return layer.keys.map((k,ni)=>{
      const y=meshH*(.1+ni*(0.82/(layer.keys.length-1)));
      return{x,y,label:k,li,active:0};
    });
  });
}
window.addEventListener('resize',buildNodes);buildNodes();
function spawnP(fl,fi,tl,ti,col){
  const f=nodes[fl][fi%nodes[fl].length],t=nodes[tl][ti%nodes[tl].length];
  if(!f||!t)return;
  particles.push({x:f.x,y:f.y,tx:t.x,ty:t.y,col,t:0,speed:.012+Math.random()*.012});
}
function drawMesh(){
  requestAnimationFrame(drawMesh);
  if(!nodes)return;
  mCtx.clearRect(0,0,meshW,meshH);
  for(let li=0;li<LAYERS.length-1;li++)
    nodes[li].forEach(n=>nodes[li+1].forEach(m=>{
      mCtx.beginPath();mCtx.moveTo(n.x,n.y);mCtx.lineTo(m.x,m.y);
      const a=n.active>0||m.active>0;
      mCtx.strokeStyle=a?'rgba(0,255,136,.15)':'rgba(10,25,35,.8)';
      mCtx.lineWidth=a?1:.4;mCtx.stroke();
    }));
  if(Math.random()<.2){const li=Math.floor(Math.random()*3),fi=Math.floor(Math.random()*nodes[li].length),ti=Math.floor(Math.random()*nodes[li+1].length);spawnP(li,fi,li+1,ti,'rgba(0,170,255,0.4)');}
  particles=particles.filter(p=>{
    p.t=Math.min(1,p.t+p.speed);
    const ex=p.x+(p.tx-p.x)*p.t,ey=p.y+(p.ty-p.y)*p.t,al=(1-Math.abs(p.t-.5)*2)*.9+.1;
    mCtx.beginPath();mCtx.arc(ex,ey,2.5,0,Math.PI*2);
    mCtx.fillStyle=p.col.replace(/[\d.]+\)$/,`${al})`);
    mCtx.shadowColor=p.col;mCtx.shadowBlur=5;mCtx.fill();mCtx.shadowBlur=0;
    return p.t<1;
  });
  nodes.forEach(layer=>layer.forEach(n=>{
    const r=n.active>0?6:3.5,col=n.active>0?'#00ff88':n.li===3?'#f5c518':'#0a2030';
    mCtx.beginPath();mCtx.arc(n.x,n.y,r,0,Math.PI*2);
    mCtx.fillStyle=col;
    if(n.active>0){mCtx.shadowColor='#00ff88';mCtx.shadowBlur=14;}
    mCtx.fill();mCtx.shadowBlur=0;
    mCtx.beginPath();mCtx.arc(n.x,n.y,r+1.5,0,Math.PI*2);
    mCtx.strokeStyle=n.active>0?'#00ff88':n.li===3?'#f5c518':'#163044';
    mCtx.lineWidth=1;mCtx.stroke();
    if(n.active>0)n.active-=.025;
    mCtx.fillStyle=n.active>0?'#00ff88':'#2a4a5a';
    mCtx.font=`${n.active>0?'700':'400'} 8px 'JetBrains Mono'`;
    mCtx.textAlign=n.li<2?'right':'left';
    mCtx.fillText(n.label,n.li<2?n.x-9:n.x+9,n.y+3);
  }));
}
drawMesh();

// ── COUNTDOWN TIMERS ─────────────────────────────────────────────────────────
let positionTimestamps={};
setInterval(()=>{
  Object.entries(positionTimestamps).forEach(([ticker,closeTs])=>{
    const el=document.getElementById('timer-'+ticker.replace(/[^a-zA-Z0-9]/g,'-'));
    if(!el)return;
    const secs=Math.max(0,closeTs-Math.floor(Date.now()/1000));
    const h=Math.floor(secs/3600),m=Math.floor((secs%3600)/60),s=secs%60;
    el.textContent=h>0?`${h}h ${m}m ${s}s`:`${m}m ${s}s`;
    if(secs<300)el.style.color='var(--red)';
    else if(secs<600)el.style.color='var(--amber)';
  });
},1000);

// ── HELPERS ───────────────────────────────────────────────────────────────────
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')}
let evalCt=0,lastEval=performance.now();

// ── RENDER ────────────────────────────────────────────────────────────────────
let lastActivity=[];
function render(d){
  if(d.error){document.getElementById('h-ts').textContent='ERR: '+d.error;return;}
  evalCt++;const now=performance.now();
  if(now-lastEval>1000){document.getElementById('eval-sec').textContent='EVAL/SEC: '+evalCt;evalCt=0;lastEval=now;}

  // Header
  const pos=d.pnl>=0;
  document.getElementById('h-bal').textContent='$'+d.balance.toFixed(2);
  const pe=document.getElementById('h-pnl');
  pe.textContent=(pos?'+':'')+' $'+d.pnl.toFixed(2);
  pe.className='hv '+(pos?'cy-g gg':'cy-r gr');
  document.getElementById('h-bets').textContent=d.placed.length;
  const scan=d.scan.scan||{};
  document.getElementById('h-opps').textContent=scan.opps!=null?scan.opps:'—';
  document.getElementById('h-pos').textContent=d.positions.length;
  const avgEdge=d.placed.length?d.placed.reduce((a,b)=>a+b.edge,0)/d.placed.length:0;
  document.getElementById('h-edge').textContent=avgEdge.toFixed(1)+'pp';
  document.getElementById('h-ts').textContent=d.ts;

  // Balance chart
  if(d.bal_hist.length){
    const base=d.bal_hist[0].v;
    balChart.data.labels=d.bal_hist.map(x=>x.t);
    balChart.data.datasets[0].data=d.bal_hist.map(x=>x.v);
    balChart.data.datasets[1].data=d.bal_hist.map(x=>parseFloat((x.v-base).toFixed(2)));
    const mn=Math.min(...d.bal_hist.map(x=>x.v)),mx=Math.max(...d.bal_hist.map(x=>x.v));
    document.getElementById('b-range').textContent='$'+mn.toFixed(2)+' – $'+mx.toFixed(2);
    const pb=document.getElementById('b-pnl-badge');
    pb.textContent=(d.pnl>=0?'+':'')+' $'+d.pnl.toFixed(2)+' P&L';
    pb.className='badge '+(d.pnl>=0?'bg':'br');
    balChart.update('none');
  }

  // Activity feed — only re-render if changed
  if(JSON.stringify(d.activity)!==JSON.stringify(lastActivity)){
    lastActivity=d.activity;
    const feed=document.getElementById('feed');
    feed.innerHTML=[...d.activity].reverse().map(a=>`
      <div class="feed-item ${a.type} ${a.type==='bet'?'bet-new':''}">
        <span class="feed-time">${esc(a.t)}</span>
        <span class="feed-msg">${esc(a.msg)}</span>
      </div>`).join('');
    document.getElementById('b-act-ct').textContent=d.activity.length+' events';
  }

  // Scanner stats
  if(scan.opps!=null){
    document.getElementById('b-scan-time').textContent=scan.time||'—';
    document.getElementById('sc-opps').textContent=scan.opps;
    document.getElementById('sc-opps-bar').style.width=Math.min(100,scan.opps*4)+'%';
    document.getElementById('sc-mkts').textContent=scan.markets;
    document.getElementById('sc-mkts-bar').style.width=Math.min(100,scan.markets/2)+'%';
    document.getElementById('sc-slots').textContent=scan.slots;
    document.getElementById('sc-slots-bar').style.width=(scan.slots/15*100)+'%';
    document.getElementById('sc-budget').textContent='$'+scan.budget.toFixed(2);
  }

  // Open positions with countdown timers
  positionTimestamps={};
  document.getElementById('b-pos-ct').textContent=d.positions.length;
  const pg=document.getElementById('pos-grid');
  if(!d.positions.length){
    pg.innerHTML='<div style="color:var(--muted);font-size:10px;grid-column:1/-1;text-align:center;padding:20px 0">No open positions</div>';
  } else {
    pg.innerHTML=d.positions.map(p=>{
      const id='timer-'+p.ticker.replace(/[^a-zA-Z0-9]/g,'-');
      if(p.close_ts) positionTimestamps[p.ticker]=p.close_ts;
      const label=p.ticker.split('-').slice(-2).join('-');
      return `<div class="pos-card ${p.side.toLowerCase()}">
        <div class="${p.side==='YES'?'pos-glow-yes':'pos-glow-no'}"></div>
        <div class="pos-ticker">${esc(p.ticker)}</div>
        <div class="pos-side">${p.side}</div>
        <div class="pos-qty">${p.qty} contracts</div>
        <div class="pos-timer" id="${id}">${p.close_str}</div>
      </div>`;
    }).join('');
  }

  // Win rate chart
  if(d.win_rates&&d.win_rates.length){
    wrChart.data.labels=d.win_rates.map(x=>x.p+'¢');
    wrChart.data.datasets[0].data=d.win_rates.map(x=>x.w);
    wrChart.data.datasets[1].data=d.win_rates.map(x=>x.p);
    document.getElementById('b-wr-pts').textContent=d.win_rates.length+' pts';
    wrChart.update('none');
  }

  // Market chart + table
  const top=d.markets.slice(0,15);
  mktChart.data.labels=top.map(m=>m.ticker.split('-').pop());
  mktChart.data.datasets[0].data=top.map(m=>m.yes_ask||0);
  mktChart.data.datasets[1].data=top.map(m=>m.no_ask||0);
  mktChart.update('none');
  document.getElementById('b-mkt-ct').textContent=d.markets.length+' live';
  document.querySelector('#mkt-tbl tbody').innerHTML=d.markets.slice(0,20).map(m=>{
    const sp=(m.yes_ask&&m.yes_bid)?m.yes_ask-m.yes_bid:null;
    const sc=sp==null?'var(--muted)':sp<=5?'var(--green)':sp<=15?'var(--amber)':'var(--red)';
    return `<tr><td class="cm" style="font-size:10px">${esc(m.ticker)}</td><td class="cy">${m.yes_bid??'—'}¢</td><td class="cy">${m.yes_ask??'—'}¢</td><td class="cn">${m.no_ask??'—'}¢</td><td style="color:${sc}">${sp!=null?sp+'¢':'—'}</td><td class="cm">${(m.vol||0).toLocaleString()}</td><td class="cm">${esc(m.close)}</td></tr>`;
  }).join('');

  // Mesh activation on new bet
  if(nodes&&d.placed.length>lastBetCount){
    d.placed.slice(lastBetCount).forEach(b=>{
      const col=b.side==='YES'?'rgba(0,255,136,.9)':'rgba(255,45,85,.9)';
      const ii=['KXBTC','KXETH','KXSOL','KXDOGE','KXINX','FED','CPI','JOBS'].findIndex(s=>b.ticker.includes(s.replace('KX','')));
      if(nodes[0][ii>=0?ii:0])nodes[0][ii>=0?ii:0].active=4;
      for(let li=0;li<3;li++){spawnP(li,Math.floor(Math.random()*nodes[li].length),li+1,Math.floor(Math.random()*nodes[li+1].length),col);spawnP(li,0,li+1,b.side==='YES'?0:1,col);}
      if(nodes[3])nodes[3][b.side==='YES'?0:1].active=6;
    });
    lastBetCount=d.placed.length;
  }

  // Tape
  if(d.placed.length){
    const items=[...d.placed].reverse().slice(0,30);
    document.getElementById('tape').innerHTML=(items.concat(items)).map(b=>
      `<span class="ti"><span class="ti-${b.side==='YES'?'y':'n'}">${b.side}</span><span style="color:var(--muted2)">${b.ticker.split('-').slice(-2).join('-')}</span><span class="ti-e">+${b.edge.toFixed(1)}pp</span><span style="color:var(--muted)">${b.price}¢</span></span>`
    ).join('');
  }

  // Fills
  const tot=d.yes_count+d.no_count||1;
  const yp=Math.round(d.yes_count/tot*100);
  document.getElementById('fy').style.width=yp+'%';
  document.getElementById('fn').style.width=(100-yp)+'%';
  document.getElementById('yes-pct').textContent='YES '+yp+'%';
  document.getElementById('no-pct').textContent='NO '+(100-yp)+'%';
  document.getElementById('s-yes').textContent=d.yes_count;
  document.getElementById('s-no').textContent=d.no_count;
  document.getElementById('s-edge').textContent=avgEdge.toFixed(1)+'pp';
  document.getElementById('s-price').textContent=d.placed.length?(d.placed.reduce((a,b)=>a+b.price,0)/d.placed.length).toFixed(0)+'¢':'—';

  // Bets table
  document.getElementById('b-bets').textContent=d.placed.length;
  document.querySelector('#bets-tbl tbody').innerHTML=[...d.placed].reverse().slice(0,25).map(b=>`
    <tr><td class="cm">${esc(b.time)}</td><td style="font-size:10px">${esc(b.ticker.split('-').slice(-2).join('-'))}</td>
    <td>${b.side==='YES'?'<span class="cy">▲ YES</span>':'<span class="cn">▼ NO</span>'}</td>
    <td>${b.count}</td><td>${b.price}¢</td><td class="ca">+${b.edge.toFixed(1)}</td></tr>`).join('')
    ||'<tr><td colspan=6 style="text-align:center;color:var(--muted);padding:12px">No bets yet</td></tr>';
}

// ── SSE ───────────────────────────────────────────────────────────────────────
function connect(){
  const es=new EventSource('/stream');
  es.onmessage=e=>render(JSON.parse(e.data));
  es.onerror=()=>{document.getElementById('h-ts').textContent='RECONNECTING…';es.close();setTimeout(connect,3000);};
}
connect();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/stream")
def stream():
    def gen():
        while True:
            yield f"data: {json.dumps(fetch_state())}\n\n"
            time.sleep(3)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    print("Terminal: http://localhost:5001")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
