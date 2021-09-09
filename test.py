###################################################
#TEST BED
###################################################
import pandas as pd
import frequencies as freq
import nonlinear as nl
import plots
from datetime import datetime as dt
import datacapture as dc
import models as mod

def testbed(asset='ETH-USD',start=dt(2020,8,1),stop=dt(2021,8,1),Fs=8,interval='1hour',mode='dump',windows=[24,24*7,24*30],obv=['v','c'],m=1,refresh=0.5):

    df_master   = dc.get_data_span( asset=asset
                                ,start=start
                                ,stop=stop
                                ,interval=interval
                                ,mode=mode
                                ) 
    data_p = mod.point_sys(df_master,size=3)
    data_d = mod.ddm(   data=data_p
                        ,obv=obv
                        ,windows=windows
                        )
    data_o = nl.dual_oscillator(data=data_d
                                ,Fs=Fs
                                ,m=m
                                ,obv=obv
                                )
    data_f,alpha  = freq.fourier_analysis(Fs
                                    ,obv
                                    ,data_o
                                    )

    plots.showplots(data_f,alpha=alpha,caller='dump',Fs=Fs,obv=obv,refresh=refresh) 

    print('- Dump Complete...')
    return data_f

if __name__ == '__main__':
    testbed() 