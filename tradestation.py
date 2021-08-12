'''
EM Wave Theory Inspired Price Action Model
The Mission: Treat an asset price time series as if it were a electro magnetic wave system.
The Method: Model the system based on the 1st, 2nd or 3rd difference of the price P(t) and volume V(t)
where dP(t) = P(t) - P(t-n) and dV(t) = V(t) - V(t-n)

'''
from datetime import datetime as dt
import frequencies as freq
import datacapture as dc
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import modes
import code
import plots

pd.plotting.register_matplotlib_converters()

def main():
    # PARAMETERS
    harms       = 3
    sr          = 1
    alpha       = 1
    Fs          = round(1/sr,2)
    diff_offset = 1
    diff        = 1
    refresh     = 0.02
    bt          = False
    windows     = [24,24*7,24*30]
    start       = dt(2018,1,1,0,0,0); stop = dt(2021,8,1,00,00,00)
    asset       = 'ETH-USD'
    obv         = ['dv1','dc1']
    interval    = 'hours'
    mode        = 'stream' if bt else 'dump'
    df1cols = ['dc1','dc2','dc3','sys_r','sys_pwr','dv3','dv2','dv1']
    df2cols = ['dv1','dv2','m1','mx1d2','mx2d2','m2','dc2','dc1']

    comp        = freq.harmonics(harms=harms,alpha=alpha)
    df_master   = dc.get_data_span( asset=asset
                                ,start=start
                                ,stop=stop
                                ,interval=interval
                                ,mode=mode
                                ) 
    params      = pd.DataFrame( [mode,harms,obv[0],obv[1],Fs,asset,interval,len(df_master)]
                                ,index=['Mode','n Harmonics','obv1','obv2','Fs','Asset','Interval','n Points']
                                ,columns=['parameters']
                                )

    print(params)

    # Determine the run mode ['backtest','dump','sweep']
    # Run backtest ['roll','expand']
    if bt:
        df = modes.stream(  data=df_master
                            ,comp=comp
                            ,diff_offset=diff_offset
                            ,diff=diff
                            ,obv=obv
                            ,harms=harms
                            ,Fs=Fs
                            ,refresh=refresh
                            ,df1cols=df1cols
                            ,df2cols=df2cols
                            ,bt=bt
                            ,windows=windows
                            )

    # Run dump
    else:
        df,dn = modes.dump( data=df_master
                        ,comp=comp
                        ,diff_offset=diff_offset
                        ,diff=diff
                        ,Fs=Fs
                        ,windows=windows
                        ,k1=1
                        ,k2=1
                        ,obv=obv
                        )


if __name__ == '__main__':
    main() 