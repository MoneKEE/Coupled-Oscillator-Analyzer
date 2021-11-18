import ccxt
import config
import pandas as pd
from datetime import datetime as dt

exch = ccxt.coinbasepro(
    {'api_key':config.API_KEY,
    'secret':config.API_SECRET
    })

# exch = ccxt.binance()

exch.set_sandbox_mode(True)

t = exch.fetch_ohlcv('BTC-USD',timeframe='1m')
s = pd.DataFrame(data=t,columns=['dt','o','h','l','c','v'])
s.dt = pd.to_datetime(s.dt,unit='ms')
pass