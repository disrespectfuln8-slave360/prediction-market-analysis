import json
from pathlib import Path

path = Path("data/kalshi/bot_learning.json")
if not path.exists():
    print("No learning file yet")
else:
    data = json.loads(path.read_text())
    bets = data.get("bets", [])
    settled = [b for b in bets if b["result"]]
    pending = [b for b in bets if not b["result"]]
    print(f"Total recorded bets: {len(bets)}")
    print(f"Settled: {len(settled)}  Pending: {len(pending)}")
    if settled:
        wins = sum(1 for b in settled if b["result"] == "win")
        losses = sum(1 for b in settled if b["result"] == "loss")
        total_pnl = sum(b["profit_cents"] or 0 for b in settled)
        print(f"Wins: {wins}  Losses: {losses}  Win rate: {100*wins/len(settled):.1f}%")
        print(f"Total P&L from settled: {total_pnl:+d} cents  (${total_pnl/100:+.2f})")
        print()
        print("All settled bets:")
        for b in settled:
            print(f"  {b['ticker'][-28:]:<28}  {b['side'].upper():<3}  @{b['entry_price']}c  pred={b['predicted_win_rate']:.1f}%  -> {b['result'].upper()}  P&L={b['profit_cents']:+d}c")
