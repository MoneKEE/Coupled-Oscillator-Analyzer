'''
EM Wave Theory Inspired Price Action Model
The Mission: Treat an asset price time series as if it were a electro magnetic wave system.
The Method: Model the system based on the 1st, 2nd or 3rd difference of the price P(t) and volume V(t)
where dP(t) = P(t) - P(t-n) and dV(t) = V(t) - V(t-n)

'''
from datetime import datetime as dt
import datacapture as dc
import pandas as pd
import misc
import modes
import sys, getopt

pd.plotting.register_matplotlib_converters()

import warnings
warnings.filterwarnings('ignore')

def main(argv):
    # PARAMETERS
    harms       = 9
    alpha       = 1
    N           = 7
    F          = 2
    diff_offset = 1
    diff        = 1
    m           = 1
    refresh     = 0.04
    start       = dt(2020,1,1,0,0,0); stop = dt(2021,1,1,00,00,00)
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
        opts,args = getopt.getopt(argv,'hm:f:t:i:s:w:n:r:',['mode=','from=','thru=','interval=','fs=','mass=','win=','ref='])
    except getopt.GetoptError:
        print('tradestation.py -m <mode> -f <from> -t <thru> -i <interval> -s <sampling freq> -w <system mass> -n <window> -r <refresh>')
        sys.exit(2)
  
    for opt, arg in opts:
        if opt == '-h':
            print('tradestation.py -m <mode> -f <from> -t <thru> -i <interval> -s <sampling freq> -w <system mass> -n <window> -r <refresh>')
            sys.exit()
        elif opt in ('-m','mode'):
            mode = arg
        elif opt in ('-i','int'):
            interval = arg
        elif opt in ('-s', 'sr'):
            t = float(arg)
            F = t
        elif opt in ('-w', 'mass'):
            m = float(arg)
        elif opt in ('-f', 'from'):
            start = pd.to_datetime(arg)
        elif opt in ('-t', 'thru'):
            stop = pd.to_datetime(arg)
        elif opt in ('-n', 'win'):
            N = int(arg)
        elif opt in ('-r', 'ref'):
            refresh = float(arg)

    df_master   = dc.get_data_span( asset=asset
                                ,start=start
                                ,stop=stop
                                ,interval=interval
                                ,mode=mode
                                ) 
    df_master = misc.normalizedf(df_master,'std')

    params      = pd.DataFrame( [mode,obv[0],obv[1],F,asset,interval,len(df_master)]
                                ,index=['Mode','obv1','obv2','F','Asset','Interval','n Points']
                                ,columns=['parameters']
                                )
    print(params)

    if mode=='stream_e':
        ds = modes.stream_e(  data=df_master
                            ,diff_offset=diff_offset
                            ,diff=diff
                            ,obv=obv
                            ,harms=harms
                            ,F=F
                            ,refresh=refresh
                            ,m=m
                            ,mode=mode
                            ,N=N
                            )
    elif mode=='stream_r':
        ds = modes.stream_r(  data=df_master
                            ,diff_offset=diff_offset
                            ,diff=diff
                            ,obv=obv
                            ,harms=harms
                            ,F=F
                            ,refresh=refresh
                            ,m=m
                            ,mode=mode
                            ,N=N
                            )
    # Run dump
    else:
        dd = modes.dump( data=df_master
                            ,diff_offset=diff_offset
                            ,diff=diff
                            ,F=F
                            ,refresh=refresh
                            ,m=m
                            ,obv=obv
                            )

if __name__ == '__main__':
    main(sys.argv[1:]) 