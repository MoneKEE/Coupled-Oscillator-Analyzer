###################################################
#TEST BED
###################################################

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from datetime import datetime as dt
#import datacapture as dc#
import models as mod
import misc

def testbed(asset='ETH-USD',start=dt(2019,1,1),stop=dt(2021,1,1),interval='hours',mode='dump',windows=[24,24*7,24*30],obv=['dv1','dc1']):

    df_master   = dc.get_data_span( asset=asset
                                ,start=start
                                ,stop=stop
                                ,interval=interval
                                ,mode=mode
                                ) 
    data_m      = mod.ddm(  data=df_master
                        ,windows=windows
                        ,obv=obv
                        )
    data_n      = misc.normalizedf(data=data_m)
    data_o      = data_n.copy()
    return data_o