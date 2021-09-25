###################################################
#TEST BED
###################################################
import pandas as pd
import frequencies as freq
import nonlinear as nl
import plots
import matplotlib.pyplot as plt
from datetime import datetime as dt
import datacapture as dc
import models as mod
import misc
import warnings

pd.plotting.register_matplotlib_converters()

def warn(*args,**kwargs):
    pass

warnings.warn=warn

def testbed(asset='ETH-USD',start=dt(2020,8,1),stop=dt(2020,8,20),interval='5minutes',F=2,mode='dump',obv=['c','v'],m=1,refresh=0.5):
    
    df_master   = dc.get_data_span( asset=asset
                                ,start=start
                                ,stop=stop
                                ,interval=interval
                                ,mode=mode
                                ) 
    data_p = mod.point_sys(df_master,size=3)
    data_d = mod.ddm(   data=data_p
                        )
    data_n = misc.normalizedf(data_d)
    data_c = freq.complex_coords(data_n,[data_n.x1nm,data_n.x2nm])
    data_o = nl.dual_oscillator(data=data_c
                                ,F=F
                                ,m=m
                                ,obv=obv
                                )
    idx = data_o.columns.get_loc('al1')
    data_n = misc.normalizedf(data_o,s=idx)
    plots.showplots(data_n,caller='dump',F=F,obv=obv,refresh=refresh) 
    data_n[['x2nm','x2prnm','x2gnnm']].plot()
    plt.show()
    breakpoint()

    print('- Dump Complete...')
    return data_n

if __name__ == '__main__':
    testbed() 