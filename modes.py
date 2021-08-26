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

def stream_r(data,comp,harms,Fs,windows,mode,figcols,refresh,obv=['v','c'],diff_offset=1,diff=1,m=1,N=7):
    # At the start it is assumed that test
    # N days have already been processed tst

    data_n = misc.normalizedf(data)
    data_s = data_n.copy()

    for row in data_s.index:
        try:
            print(f'\n - Processing Frame #{row.value}: start:{min(data_s.index)} end:{max(data_s.index)}\n')
        except KeyboardInterrupt:
            break

        data_i = data_s.loc[row:row+timedelta(days=7),:]

        T = len(data_i)/Fs
        df = round(1/T,3)
        dw = round((2*np.pi)/T,3)
        ny = round((dw*len(data_i))/2,3)

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
                                    )
        data_f = freq.fourier_analysis( comp
                                ,Fs
                                ,obv
                                ,data_o
                                )

        peaks = pd.DataFrame(data=[[np.nan for x in range(len(obv))] for y in range(3000)],columns=obv,index=[x for x in range(3000)])
        for i in obv:
            pks,props = freq.peak_analysis(data_f,col=i)
            pksl = list(pks)
            if len(pksl)==0:peaks[i] = np.zeros(len(peaks))
            else: peaks[i]=np.nan;peaks[i][:len(pksl)] = pksl
        peaks.dropna(how='all',inplace=True)
        peaks.fillna(0,inplace=True)
        pks_c = np.int0(peaks.c[peaks.c!=0])
        pks_v = np.int0(peaks.v[peaks.v!=0])

        plots.showplots(df1=data_f,caller='stream',m=m,Fs=Fs,figcols=figcols,obv=obv,pks_v=pks_v,pks_c=pks_c,refresh=refresh)        

    print('- rolling backtest complete...')

    return data_s


def stream_e(data,comp,harms,Fs,windows,mode,figcols,refresh,obv=['v','c'],diff_offset=1,diff=1,m=1,N=7):
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
                                    )
        
        peaks = pd.DataFrame(data=[[np.nan for x in range(len(obv))] for y in range(3000)],columns=obv,index=[x for x in range(3000)])
        for i in obv:
            pks,props = freq.peak_analysis(data_o,col=i)
            pksl = list(pks)
            if len(pksl)==0:peaks[i] = np.zeros(len(peaks))
            else: peaks[i]=np.nan;peaks[i][:len(pksl)] = pksl
        peaks.dropna(how='all',inplace=True)
        peaks.fillna(0,inplace=True)
        pks_c = np.int0(peaks.c[peaks.c!=0])
        pks_v = np.int0(peaks.v[peaks.v!=0])

        plots.showplots(df1=data_o,caller='stream',m=m,Fs=Fs,figcols=figcols,obv=obv,pks_v=pks_v,pks_c=pks_c,refresh=refresh)  

        data_s = data_f.append(data_n.iloc[t])        

    print('- expanding backtest complete...')

    return data_s
   
def dump(data,comp,Fs,windows,refresh,figcols,m=1,obv=['v','c'],diff_offset=1,diff=1): 

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
                                ,m=m
                                ,obv=obv
                                ) 

    peaks = pd.DataFrame(data=[[np.nan for x in range(len(obv))] for y in range(3000)],columns=obv,index=[x for x in range(3000)])
    for i in obv:
        pks,props = freq.peak_analysis(data_o,col=i)
        pksl = list(pks)
        if len(pksl)==0:peaks[i] = np.zeros(len(peaks))
        else: peaks[i]=np.nan;peaks[i][:len(pksl)] = pksl
    peaks.dropna(how='all',inplace=True)
    peaks.fillna(0,inplace=True)
    pks_c = np.int0(peaks.c[peaks.c!=0])
    pks_v = np.int0(peaks.v[peaks.v!=0])
    
    print('- Dump Complete...')
    plots.showplots(df1=data_o,caller='dump',m=m,Fs=Fs,figcols=figcols,obv=obv,pks_v=pks_v,pks_c=pks_c,refresh=refresh) 

    breakpoint()

    return data_o