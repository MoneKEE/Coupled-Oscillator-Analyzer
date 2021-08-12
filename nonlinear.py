import pandas as pd
import numpy as np
import scipy as sy
import datacapture as dc
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy import signal as sig
import misc 
import models

# The dual oscillator attempts to model price action through
# the motion of a two axis simple oscillator. It accepts
# two parameters X1 and X2
def dual_oscillator(data,obv,k1=1,k2=1):

    data_o = data.copy()
    e1 = [1,0]
    e2 = [0,1]

    if len(obv[0]) == 1:
        x1 = obv[0]
        x2 = obv[1]

        l1 = f'd{x1}1'
        l2 = f'd{x2}1'
    else:
        x1 = obv[0]
        x2 = obv[1]

        l1 = f'd{x1[1]}2'
        l2 = f'd{x2[1]}2'

    p = data_o[[x1,x2]]
    r1 = data_o[[l1,x2]]
    r2 = data_o[[x1,l2]]

    r1mag = np.sqrt((p[x1]+r1[l1])**2+p[x2]**2)
    r2mag = np.sqrt((p[x2]+r2[l2])**2+p[x1]**2)

    er1 = r1.divide(r1mag,axis=0)
    er2 = r2.divide(r2mag,axis=0)

    del1 = r1mag - r1[l1]
    del2 = r2mag - r2[l2]

    f1 = -k1*er1.multiply(del1,axis=0)
    f2 = -k2*er2.multiply(del2,axis=0)

    mx1d2 = pd.DataFrame({'s1':f1.iloc[:,0]+f2.iloc[:,0],'s2':f1.iloc[:,1]+f2.iloc[:,1]}).dot(e1)
    mx2d2 = pd.DataFrame({'s1':f1.iloc[:,0]+f2.iloc[:,0],'s2':f1.iloc[:,1]+f2.iloc[:,1]}).dot(e2)
    
    m1 = mx1d2.divide(data_o.dv2)
    m2 = mx2d2.divide(data_o.dc2)
    msum = m1 + m2
    result = pd.DataFrame(
                            {
                                'c':data_o.c
                                ,'v':data_o.v
                                ,'dc1':data_o.dc1
                                ,'dv1':data_o.dv1
                                ,'dc2':data_o.dc2
                                ,'dv2':data_o.dv2
                                ,'dc3':data_o.dc3
                                ,'dv3':data_o.dv3
                                ,'sys_r':data_o.sys_r
                                ,'sys_pwr':data_o.sys_pwr
                                ,'mx1d2':mx1d2
                                ,'mx2d2':mx2d2
                                ,'m1':m1
                                ,'m2':m2
                                ,'msum':msum
                            }
                        )

    result.fillna(0,inplace=True)
    
    for col in result.columns:
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)
        result[col] = result[col].fillna(np.abs(result[col]).max())

    result_n = misc.normalizedf(result)

    return result_n

################
#TEST BED
################
# import test
# data_o = test.testbed(start=dt(2020,1,1),stop=dt(2020,2,1))
# viz = showplots(data_o)
# viz.show()
# pass