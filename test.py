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

def testbed(asset='ETH-USD',start=dt(2020,9,1),stop=dt(2020,9,15),hrm='all',interval='1minute',F=131072,mode='dump',obv=['c','v'],m=1,refresh=0.5):
    
    df_master   = dc.get_data_span( asset=asset
                                ,start=start
                                ,stop=stop
                                ,interval=interval
                                ,mode=mode
                                ) 
    data_p = mod.point_sys(df_master,size=3)
    data_d = mod.ddm(   data=data_p
                        )
    data_o = nl.dualosc2(data=data_d
                            ,F=F
                            ,m=m
                            ,hrm=hrm
                            )
    data_n = misc.normalizedf(data_o,'std')
    #plots.showplots2(df1=data_n,caller='stream',F=F,m=m,obv=obv,refresh=refresh)  
    # data_n[['Trq']].plot()
    # plt.show()
    breakpoint()

    print('- Dump Complete...')
    return data_n

if __name__ == '__main__':
    testbed() 