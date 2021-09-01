###################################################
#TEST BED
###################################################

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import frequencies as freq
import nonlinear as nl
import time
from datetime import datetime as dt
import datacapture as dc
import models as mod
import misc

def testbed(asset='ETH-USD',start=dt(2019,1,1),stop=dt(2021,1,1),Fs=2,interval='1hour',mode='dump',windows=[24,24*7,24*30],obv=['v','c']):

    df_master   = dc.get_data_span( asset=asset
                                ,start=start
                                ,stop=stop
                                ,interval=interval
                                ,mode=mode
                                ) 
    data_n      = misc.normalizedf(data=df_master.copy())
    data_p      = mod.point_sys(data_n,size=3)
    data_m      = mod.ddm(  data=data_p
                            ,diff=1
                            ,diff_offset=1
                            ,obv=obv
                            ,windows=windows
                            )
    comp        = freq.harmonics(harms=9
                                ,alpha=1
                                ,type='harm_mlt'
                                )
    data_f      = freq.fourier_analysis( comp
                                        ,Fs
                                        ,obv
                                        ,data_m
                                        )
    data_o      = nl.dual_oscillator(data=data_f
                                    ,m=1
                                    ,obv=obv
                                    ,Fs=Fs
                                    )
    return data_o

if __name__ == '__main__':
    testbed() 