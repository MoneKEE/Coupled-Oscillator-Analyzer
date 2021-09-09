import nonlinear as nl
import misc
from datetime import timedelta
import models as mod
import frequencies as freq
import plots

def stream_r(data,harms,Fs,windows,mode,refresh,obv=['v','c'],diff_offset=1,diff=1,m=1,N=7):
    # At the start it is assumed that test
    # N days have already been processed tst
    for row in data.index:
        try:
            print(f'\n - Processing Frame #{row.value}: start:{min(data.index)} end:{max(data.index)}\n')
        except KeyboardInterrupt:
            break

        data_i = data.loc[row:row+timedelta(days=N),:]

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
        data_f,alpha = freq.fourier_analysis(Fs
                                ,obv
                                ,data_o
                                )
        plots.showplots(df1=data_f,caller='stream',alpha=alpha,Fs=Fs,obv=obv,refresh=refresh)        

    print('- rolling backtest complete...')

    return data_f


def stream_e(data,harms,Fs,windows,mode,refresh,obv=['v','c'],diff_offset=1,diff=1,m=1,N=7):
    # At the start it is assumed that 
    # N days have already been processed
    data_s = data[:N].copy()

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
        data_o = nl.dual_oscillator(data=data_m
                                    ,m=m
                                    ,obv=obv
                                    ,Fs=Fs
                                    )
        data_f,alpha = freq.fourier_analysis( Fs
                                        ,obv
                                        ,data_o
                                        )
        plots.showplots(df1=data_f,caller='stream',alpha=alpha,Fs=Fs,obv=obv,refresh=refresh)  

        data_s = data_f.append(data.iloc[t])        

    print('- expanding backtest complete...')

    return data_s
   
def dump(data,Fs,windows,refresh,m=1,obv=['v','c'],diff_offset=1,diff=1):
    data_p = mod.point_sys(data,size=3)
    data_d = mod.ddm(   data=data_p
                        ,diff_offset=diff_offset
                        ,obv=obv
                        ,diff=diff
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

    plots.showplots(df1=data_f,alpha=alpha,caller='dump',Fs=Fs,obv=obv,refresh=refresh) 

    print('- Dump Complete...')

    breakpoint()

    return data_f