import nonlinear as nl
import misc
import models as mod
import numpy as np
import frequencies as freq
import pandas as pd
import plots
import time



def stream(data,comp,harms,Fs,windows,mode,figcols,refresh,obv=['v','c'],diff_offset=1,diff=1,k1=1,k2=1,N=7):
    # At the start it is assumed that 
    # N days have already been processed

    data_n = misc.normalizedf(data)
    data_s = data_n[:N].copy()

    for t in range(N,len(data)):
        try:
            print('='*80)
            print(f'- Processing Frame #{t}: start:{min(data_s.index)} end:{max(data_s.index)}\n')
        except:
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
        data_o = nl.dual_oscillator(data=data_m
                                    ,k1=k1
                                    ,k2=k2
                                    ,obv=obv
                                    )
        data_f = freq.fourier_analysis( comp
                                        ,Fs
                                        ,obv
                                        ,data_o
                                        )

        peaks = pd.DataFrame(data=[[np.nan for x in range(len(obv))] for y in range(300)],columns=obv,index=[x for x in range(300)])
        for i in obv:
            pks,props = freq.get_angfreq_peaks(data_f,col=i)
            pksl = list(pks)
            if len(pksl)==0:peaks[i] = np.zeros(len(peaks))
            else: peaks[i]=np.nan;peaks[i][:len(pksl)] = pksl 
        peaks.dropna(how='all',inplace=True)
        peaks.fillna(0,inplace=True)

        plots.showplots(df1=data_f,caller='stream',figcols=figcols,obv=obv,pks=peaks,refresh=refresh) 

        data_s = data_f.append(data_n.iloc[t])        

    print('- Backtest Complete...')

    return data_s
   
def dump(data,comp,Fs,windows,refresh,figcols,obv=['v','c'],diff_offset=1,diff=1,k1=1,k2=1): 

    data_n = misc.normalizedf(data)

    data_p = mod.point_sys(data_n,size=3)

    data_d = mod.ddm(   data=data_p
                        ,diff_offset=diff_offset
                        ,obv=obv
                        ,diff=diff
                        ,windows=windows
                        )
    data_o = nl.dual_oscillator(data=data_d
                                ,k1=k1
                                ,k2=k2
                                ,obv=obv
                                )    
    data_f  = freq.fourier_analysis(comp
                                    ,Fs
                                    ,obv
                                    ,data_o
                                    )

    peaks = pd.DataFrame(data=[[np.nan for x in range(len(obv))] for y in range(300)],columns=obv,index=[x for x in range(300)])
    for i in obv:
        pks,props = freq.get_angfreq_peaks(data_f,col=i)
        pksl = list(pks)
        if len(pksl)==0:peaks[i] = np.zeros(len(peaks))
        else: peaks[i]=np.nan;peaks[i][:len(pksl)] = pksl 
    peaks.dropna(how='all',inplace=True)
    peaks.fillna(0,inplace=True)
    
    print('- Dump Complete...')
    plots.showplots(df1=data_f,caller='dump',figcols=figcols,pks=peaks,obv=obv,refresh=refresh) 

    return data_f