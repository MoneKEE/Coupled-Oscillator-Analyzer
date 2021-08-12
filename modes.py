import misc
import models as mod
import frequencies as freq
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import trader
import plots
import frequencies
import nonlinear as nl
import time
import inspect
    
def stream(data,comp,harms,Fs,windows,bt,refresh,df1cols,df2cols,obv=['dv1','dc1'],diff_offset=1,diff=1,k1=1,k2=1):
    # At the start it is assumed that 
    # N days have already been processed
    N = 24*7

    data_norm = misc.normalizedf(data)
    data_s = data_norm[:N].copy()

    try:
        while True:
            for t in range(N,len(data)):
                try:
                    print('='*80)
                    print(f'- Processing Frame #{t}: start:{min(data_s.index)} end:{max(data_s.index)}\n')
                except:
                    break

                data_s  = mod.ddm(data=data_s,obv=obv,diff_offset=diff_offset,windows=windows)
                #data_s  = mod.nonlinear(data=data_s)

                for col in obv:
                    data_s  = freq.get_fft( data=data_s
                                            ,col=col
                                            ,Fs=Fs
                                            )

                    data_s  = freq.get_ifft( data=data_s
                                            ,col=col
                                            ,comp=comp
                                            )

                data_n = nl.dual_oscillator(data=data_s,k1=k1,k2=k2,obv=obv)
                
                figcols = ['dv1','dc1','dv2','dc2','dv3','dc3']
                pltdta = data_s[figcols]
                plots.showplots(df1=pltdta,caller='stream') 
                
                data_s = data_s.append(data.iloc[t])


            print('- Backtest Complete...')

            return data_s
            
    except KeyboardInterrupt:
        pass
   
def dump(data,comp,Fs,windows,obv=['dv1','dc2'],diff_offset=1,diff=1,k1=1,k2=1): 

    data_norm = misc.normalizedf(data)

    data_d = mod.ddm(   data=data_norm
                        ,diff_offset=diff_offset
                        ,obv=obv
                        ,diff=diff
                        ,windows=windows
                        )
    for col in obv:

        data_d  = freq.get_fft( data=data_d
                                ,col=col
                                ,Fs=Fs
                                )
        data_d  = freq.get_ifft(data=data_d
                                ,comp=comp
                                ,col=col
                                )
        data_n = nl.dual_oscillator(data=data_d,k1=k1,k2=k2,obv=obv)

    print('- Dump Complete...')
    fig1cols = ['dv1','dc1','dv2','dc2','dv3','dc3','dc1_ifft','dv1_ifft']
    pltdta = data_d[fig1cols]
    plots.showplots(df1=pltdta,caller='dump') 

    return data_d,data_n