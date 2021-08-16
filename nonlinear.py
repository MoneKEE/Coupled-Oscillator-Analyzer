import pandas as pd
import numpy as np
import scipy as sy
from scipy import signal as sig
import matplotlib.pyplot as plt
from datetime import datetime as dt

import misc 
import models

# The dual oscillator attempts to model price action through
# the motion of a two axis simple oscillator. It accepts
# two parameters X1 and X2
def dual_oscillator(data,obv=['v','c'],k1=1,k2=1):

    data_o = data.copy()
    e1 = [[1],[0]]
    e2 = [0,1]

    x1 = f'd{obv[0]}t_o'
    x2 = f'd{obv[1]}t_o'

    #These values represent the x and y coords
    #of point 1 and point 2.
    l1 = data_o[x1] - data_o[x1].shift(1)
    l2 = data_o[x2] - data_o[x2].shift(1)

    r1 = pd.DataFrame({'e1':data_o[x1].add(l1),'e2':data_o[x2]})
    r2 = pd.DataFrame({'e1':data_o[x1],'e2':data_o[x2].add(l2)})

    r1mag = np.sqrt(r1['e1']**2 + r1['e2']**2)
    r2mag = np.sqrt(r2['e1']**2 + r2['e2']**2)

    er1 = r1.divide(r1mag,axis=0)
    er2 = r2.divide(r2mag,axis=0)

    del1 = r1mag.sub(l1)
    del2 = r2mag.sub(l2)

    f1 = -k1*er1.multiply(del1,axis=0)
    f2 = -k2*er2.multiply(del2,axis=0)

    # # #F=MA this is the 
    ma1 = (f1.add(f2)).dot(e1)
    ma2 = (f1.add(f2)).dot(e2)

    data_o['ma1'] = ma1
    data_o['ma2'] = ma2
    
    m1 = ma1.divide(data_o.dvt_oo,axis=0)
    m2 = ma2.divide(data_o.dct_oo,axis=0)

    data_o['m1'] = m1
    data_o['m2'] = m2
    data_o.fillna(0,inplace=True)

    for col in data_o.columns:
        data_o[col] = data_o[col].replace([np.inf, -np.inf], np.nan)
        data_o[col] = data_o[col].fillna(np.abs(data_o[col]).max())
 
    data_n = misc.normalizedf(data_o)

    return data_n

################
#TEST BED
################
# import test
# data_o = test.testbed(start=dt(2020,1,1),stop=dt(2020,2,1))
# viz = showplots(data_o)
# viz.show()
# pass