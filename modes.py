import nonlinear as nl
import misc
from datetime import timedelta
import models as mod
import numpy as np
import frequencies as freq
import pandas as pd
import plots
import matplotlib.pyplot as plt
from numpy import fft
import time

def stream_r(data,comp,harms,Fs,windows,mode,refresh,alpha,obv=['v','c'],diff_offset=1,diff=1,m=1,N=7):
    # At the start it is assumed that test
    # N days have already been processed tst

    data_n = misc.normalizedf(data)
    data_s = data_n.copy()

    for row in data_s.index:
        try:
            print(f'\n - Processing Frame #{row.value}: start:{min(data_s.index)} end:{max(data_s.index)}\n')
        except KeyboardInterrupt:
            break

        data_i = data_s.loc[row:row+timedelta(days=N),:]

        data_p = mod.point_sys( data=data_i
                                ,size=3
                                )
        data_m  = mod.ddm(  data=data_p
                            ,diff_offset=diff_offset
                            ,obv=obv
                            ,diff=diff
                            ,windows=windows
                            )
        data_o = nl.dual_oscillator(data=data_m
                                    ,m=m
                                    ,obv=obv
                                    ,Fs=Fs
                                    )
        data_f = freq.fourier_analysis( comp
                                ,Fs
                                ,obv
                                ,data_o
                                )
        plots.showplots(df1=data_f,caller='stream',alpha=alpha,Fs=Fs,obv=obv,refresh=refresh)        

    print('- rolling backtest complete...')

    return data_s


def stream_e(data,comp,harms,Fs,windows,mode,alpha,refresh,obv=['v','c'],diff_offset=1,diff=1,m=1,N=7):
    # At the start it is assumed that 
    # N days have already been processed

    data_n = misc.normalizedf(data)
    data_s = data_n[:N].copy()

    for t in range(N,len(data)):
        try:
            print(f'\n- Processing Frame #{t}: start:{min(data_s.index)} end:{max(data_s.index)}\n')
        except KeyboardInterrupt:
            break

        data_p = mod.point_sys( data_s
                                ,size=3
                                )
        data_m  = mod.ddm(  data=data_p
                            ,diff_offset=diff_offset
                            ,obv=obv
                            ,diff=diff
                            ,windows=windows
                            )
        data_f = freq.fourier_analysis( comp
                                        ,Fs
                                        ,obv
                                        ,data_m
                                        )
        data_o = nl.dual_oscillator(data=data_f
                                    ,m=m
                                    ,obv=obv
                                    ,Fs=Fs
                                    )
        plots.showplots(df1=data_o,caller='stream',alpha=alpha,Fs=Fs,obv=obv,refresh=refresh)  

        data_s = data_f.append(data_n.iloc[t])        

    print('- expanding backtest complete...')

    return data_s
   
def dump(data,comp,Fs,windows,refresh,alpha,m=1,obv=['v','c'],diff_offset=1,diff=1): 

    data_n = misc.normalizedf(data)

    data_p = mod.point_sys(data_n,size=3)

    data_d = mod.ddm(   data=data_p
                        ,diff_offset=diff_offset
                        ,obv=obv
                        ,diff=diff
                        ,windows=windows
                        )   
    data_f  = freq.fourier_analysis(comp
                                    ,Fs
                                    ,obv
                                    ,data_d
                                    )
    data_o = nl.dual_oscillator(data=data_f
                                ,Fs=Fs
                                ,m=m
                                ,obv=obv
                                )
    plots.showplots(df1=data_o,alpha=alpha,caller='dump',Fs=Fs,obv=obv,refresh=refresh) 

    print('- Dump Complete...')

    breakpoint()

    return data_o