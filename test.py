###################################################
#TEST BED
###################################################
import pandas as pd
import frequencies as freq
import nonlinear as nl
import plots
# import matplotlib.pyplot as plt
from datetime import datetime as dt
import datacapture as dc
import models as mod
import misc
import warnings
import stats as st
import mlmod
import numpy as np

pd.plotting.register_matplotlib_converters()

def warn(*args,**kwargs):
    pass

warnings.warn=warn

def testbed(asset='ETH-USD',start=dt(2020,9,1),stop=dt(2020,9,10),hrm='fnd',interval='1minute',F=368896,mode='dump',obv=['c','v'],m=1,refresh=0.5):
    
    df_master   = dc.get_data_span( asset=asset
                                ,start=start
                                ,stop=stop
                                ,interval=interval
                                ,mode=mode
                                ) 
    data_p = mod.point_sys(df_master,size=3)
    data_d = mod.ddm(   data=data_p
                        )
    data_o,qw = nl.dualosc2(data=data_d
                            ,F=F
                            ,m=m
                            ,hrm=hrm
                            )
    
    data_n = misc.normalizedf(data_o,'plot')
    # data_n = misc.normalizedf(data_o,None)
    # data_p = st.proba(data_n)

    endog = data_n.loc[:,'Pe':'Spd']
    exog  = data_n.pos

    clf = mlmod.MLP(endog[:-1],exog[:-1],asset='ETH-USD')
    
    ypred = clf.predict(endog)

    data_n['post'] = np.zeros(len(endog))
    data_n.post = ypred

    plots.showplots(data_n,F=F,obv=obv,m=m,hrm=hrm,qw=qw)

    breakpoint()

    print('- Dump Complete...')
    return data_n

if __name__ == '__main__':
    testbed() 