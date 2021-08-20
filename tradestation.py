'''
EM Wave Theory Inspired Price Action Model
The Mission: Treat an asset price time series as if it were a electro magnetic wave system.
The Method: Model the system based on the 1st, 2nd or 3rd difference of the price P(t) and volume V(t)
where dP(t) = P(t) - P(t-n) and dV(t) = V(t) - V(t-n)

'''
from datetime import datetime as dt

from numpy import NaN
import frequencies as freq
import datacapture as dc
import pandas as pd
import modes
import sys, getopt

pd.plotting.register_matplotlib_converters()

import warnings
warnings.filterwarnings('ignore')

def main(argv):
    # PARAMETERS
    harms       = 9
    sr          = 0.001
    alpha       = 3
    N           = 7
    Fs          = round(1/sr,3)
    diff_offset = 1
    diff        = 1
    k1          = 0.1
    k2          = 0.1
    refresh     = 0.01
    windows     = [24,24*7,24*30]
    start       = dt(2018,1,1,0,0,0); stop = dt(2018,2,1,00,00,00)
    asset       = 'ETH-USD'
    interval    = '15minutes'
    mode        = 'stream_r'
    figcols     = [ 'v_sig','c_sig'
                    ,'dv1t_0','dc1t_o'
                    ,'dv2t_oo','dc2t_oo'
                    ,'dv3t_ooo','dc3t_ooo'
                    ,'vc_pwr'
                    ,'vc_r','vc_c'
                    ,'vf_rad','cf_rad'
                    ,'vf_w','cf_w'
                    ,'vf_t','cf_t'
                    ,'fft_freq'
                    ,'m1','m2'
                    ,'ma1','ma2'
                    ,'f1','f2']

    obv         = ['v','c']


    try:
        opts,args = getopt.getopt(argv,'hm:f:t:i:h:s:a:k1:k2:',['mode=','from=','thru=','interval=','harms=','sr=','alpha=','k1=','k2='])
    except getopt.GetoptError:
        print('tradestation.py -m <mode> -i <interval> -h <harmonics> -s <sampling rate> -a alpha -1 <damping ratio 1> -2 <damping ratio 2>')
        sys.exit(2)
  
    for opt, arg in opts:
        if opt == '-h':
            print('tradestation.py -m <mode> -hr <harmonics> -sr <sampling rate> -a <alpha> -k1 <damping ratio 1> -k2 <damping ratio 2>')
            sys.exit()
        elif opt in ('-m','mode'):
            mode = arg
        elif opt in ('-i','int'):
            interval = arg
        elif opt in ('-h','harms'):
            harms = arg
        elif opt in ('-s', 'sr'):
            sr = arg
            Fs = round(1/float(sr),3)
        elif opt in ('-a','alpha'):
            alpha = arg
        elif opt in ('-1', 'k2'):
            k1 = arg
        elif opt in ('-2', 'k1'):
            k2 = arg
        elif opt in ('-f', 'from'):
            start = pd.to_datetime(arg)
        elif opt in ('-t', 'thru'):
            stop = pd.to_datetime(arg)

    #breakpoint()
    comp        = freq.harmonics(harms=harms
                                ,alpha=alpha
                                ,type='harm_mlt'
                                )
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

    if mode=='stream_e':
        ds = modes.stream_e(  data=df_master
                            ,comp=comp
                            ,diff_offset=diff_offset
                            ,diff=diff
                            ,obv=obv
                            ,harms=harms
                            ,Fs=Fs
                            ,refresh=refresh
                            ,figcols=figcols
                            ,k1=1
                            ,k2=1
                            ,mode=mode
                            ,windows=windows
                            ,N=N
                            )
    elif mode=='stream_r':
        ds = modes.stream_r(  data=df_master
                            ,comp=comp
                            ,diff_offset=diff_offset
                            ,diff=diff
                            ,obv=obv
                            ,harms=harms
                            ,Fs=Fs
                            ,refresh=refresh
                            ,figcols=figcols
                            ,k1=1
                            ,k2=1
                            ,mode=mode
                            ,windows=windows
                            ,N=N
                            )
    # Run dump
    else:
        dd = modes.dump( data=df_master
                            ,comp=comp
                            ,diff_offset=diff_offset
                            ,diff=diff
                            ,Fs=Fs
                            ,windows=windows
                            ,refresh=refresh
                            ,k1=1
                            ,k2=1
                            ,obv=obv
                            ,figcols=figcols
                            )


if __name__ == '__main__':
    main(sys.argv[1:]) 