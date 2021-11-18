###################################################
#TEST BED
###################################################
import pandas as pd
import frequencies as freq
import nonlinear as nl
import plots
from datetime import datetime as dt
from datetime import timedelta as de
import modes
import warnings
import numpy as np
import ccxt
import config
import time 

pd.plotting.register_matplotlib_converters()

def warn(*args,**kwargs):
    pass

warnings.warn=warn

def testbed(asset='BTC-USD',start=dt(2021,10,10),stop=dt.now().replace(second=0,microsecond=0)-de(seconds=3600),hrm='even',interval='1minute',F=368896,mode='dump',obv=['c','v'],m=1):

    asset = 'ETH-USD'
    hrm='even'
    interval='1minute'
    F=368896
    mode='dump'
    obv=['v','c']
    m=1

    df  = pd.read_csv('test_data.csv')
    df.dt=pd.to_datetime(df.dt,format='%Y-%m-%d %H:%M:%S')
    df.set_index('dt',drop=True,inplace=True)

    idx=df.index[0]+de(days=30*1)

    data = modes.dump(data=df.loc[:idx],asset='ETH-USD',hrm='even',m=1,F=368896)

    breakpoint()

    print('- Dump Complete...')
    return data


if __name__ == '__main__':
    testbed() 
