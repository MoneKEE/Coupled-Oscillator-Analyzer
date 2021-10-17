import nonlinear as nl
import misc
from datetime import timedelta
import numpy as np
import models as mod
import frequencies as freq
import plots
import mlmod
import pandas as pd

def stream_r(data,F,mode,asset,refresh,hrm,obv=['v','c'],diff_offset=1,diff=1,m=1,N=7):
    # At the start it is assumed that test
    # N days have already been processed tst
    for row in data.index:
        try:
            print(f'\n - Processing Frame #{row.value}: start:{min(data.index)} end:{max(data.index)}\n')
        except KeyboardInterrupt:
            break

        data_i = data.loc[row:row+timedelta(days=N),:]

        data_p = mod.point_sys(data_i,size=3)
        data_d = mod.ddm(   data=data_p
                            )
        data_o,qw = nl.dualosc2(data=data_d
                                ,F=F
                                ,m=m
                                ,hrm=hrm
                                )
        data_n = misc.normalizedf(data_o,'std')

        endog = data_n.loc[:,'Pe':'Spd']
        exog  = data_n.pos
        try: 
            clf=clf.partial_fit(endog,exog)
            ypred = clf.predict(endog)
            posa = posa.shift(-1)
            posa.iloc[-1] = ypred[-1]
            posa.fillna(0,inplace=True)
        except UnboundLocalError:
            clf = mlmod.MLP(endog,exog,asset='ETH-USD')
            ypred = clf.predict(endog)
            posa = pd.Series(ypred)
            
        data_n['posa'] = posa.reset_index(drop=True)
        plots.showplots(df1=data_n,m=m,caller='stream',qw=qw,asset=asset,F=F,obv=obv,refresh=refresh,hrm=hrm)        

    print('- rolling backtest complete...')

    return data_o


def stream_e(data,harms,F,mode,refresh,hrm,obv=['v','c'],diff_offset=1,diff=1,m=1,N=7):
    # At the start it is assumed that 
    # N days have already been processed
    data_s = data[:N].copy()

    for t in range(N,len(data)):
        try:
            print(f'\n- Processing Frame #{t}: start:{min(data_s.index)} end:{max(data_s.index)}\n')
        except KeyboardInterrupt:
            break

        data_n = misc.normalizedf(data_s)

        data_p = mod.point_sys( data=data_n
                                ,size=3
                                )
        data_c  = freq.complex_coords(data_p,x=[data_p.dv1dt1nm,data_p.dc1dt1nm])
        data_m  = mod.ddm(  data=data_c
                            )
        data_o = nl.dualosc2(data=data_m
                                    ,m=m
                                    ,obv=obv
                                    ,F=F
                                    )
        data_n = misc.normalizedf(data_o,'std')
        plots.showplots(df1=data_n,caller='stream',F=F,obv=obv,refresh=refresh)  

        data_s = data_o.append(data.iloc[t])        

    print('- expanding backtest complete...')

    return data_s
   
def dump(data,harms,F,mode,asset,refresh,hrm,obv=['v','c'],diff_offset=1,diff=1,m=1):
    data_p = mod.point_sys( data=data
                            ,size=3
                            )
    data_m  = mod.ddm(  data=data_p
                        )
    data_o = nl.dualosc2(data=data_m
                                ,F=F
                                ,m=m
                                ,hrm=hrm
                                )
    data_n = misc.normalizedf(data_o,'std')
    plots.showplots(df1=data_n,m=m,caller='dump',asset=asset,F=F,obv=obv,refresh=refresh,hrm=hrm)  
    breakpoint()
    print('- Dump Complete...')
    return data_n