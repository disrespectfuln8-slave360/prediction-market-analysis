from dotenv import load_dotenv
load_dotenv()
from src.trading.kalshi.trading_client import KalshiTradingClient

candidates = [
    'KXMLBGAME','KXMLBTOTAL','KXMLBSPREAD',
    'KXNBAGAME','KXNBATOTAL','KXNBASPREAD',
    'KXNHLGAME','KXNHLTOTAL',
    'KXUFCFIGHT','KXUFCMOV',
    'KXATPMATCH','KXWTAMATCH',
    'KXBTC15M','KXETH15M','KXSOL15M',
    'KXBTCD',
    'KXLOLGAME','KXCS2GAME',
    'KXEPLGAME','KXUCLGAME','KXLALIGAGAME',
]

with KalshiTradingClient() as c:
    for series in candidates:
        try:
            data = c.http.get('/markets', params={'limit': 5, 'status': 'open', 'series_ticker': series})
            markets = data.get('markets', [])
            count = len(markets)
            if markets:
                ticker = markets[0]['ticker'][-25:]
                print(f'{series:<25}  open={count:3d}  sample={ticker}')
            else:
                print(f'{series:<25}  open=  0')
        except Exception as e:
            print(f'{series:<25}  ERROR: {e}')
