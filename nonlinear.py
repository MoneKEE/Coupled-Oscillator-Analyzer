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

    e1 = [1,0]
    e2 = [0,1]

    x1 = f'd{obv[0]}1t_o'
    x2 = f'd{obv[1]}1t_o'

    dotx1 = f'd{obv[0]}2t_oo'
    dotx2 = f'd{obv[1]}2t_oo'

    l1 = data_o[x1] - data_o[x1].shift(1)
    l2 = data_o[x2] - data_o[x2].shift(1)

    r1 = pd.DataFrame({'e1':data_o[x1].add(l1),'e2':data_o[x2]})
    r2 = pd.DataFrame({'e1':data_o[x1],'e2':data_o[x2].add(l2)})

    r1mag = np.sqrt(r1['e1']**2 + r1['e2']**2)
    r2mag = np.sqrt(r2['e1']**2 + r2['e2']**2)

    c1 = 1 - (l1/r1mag)
    c2 = 1 - (l2/r2mag)

    data_o['c1'] = c1
    data_o['c2'] = c2

    cmag = np.sqrt(c1**2 + c2**2)
    c_theta = np.arcsin(c2/cmag)

    # Force vectors and sum
    f1 = data_o[dotx1].multiply(c1,axis=0) + data_o[x1].multiply(k1,axis=0) 
    f2 = data_o[dotx2].multiply(c2,axis=0) + data_o[x2].multiply(k2,axis=0)

    fmag = np.sqrt(f1**2 + f2**2)
    f_theta = np.arcsin(f2/fmag)

    ma1 = f1
    ma2 = f2

    data_o['ma1'] = ma1
    data_o['ma2'] = ma2

    # mass vectors and sum
    m1 = f1.divide(data_o[f'd{obv[0]}3t_ooo'].add(data_o[f'd{obv[1]}3t_ooo']))
    m2 = f2.divide(data_o[f'd{obv[0]}3t_ooo'].add(data_o[f'd{obv[1]}3t_ooo']))

    mmag = np.sqrt(m1**2 + m2**2)
    m_theta = np.arcsin(m2/mmag)

    data_o['m1'] = m1
    data_o['m2'] = m2
    
    # # Momentum calculations based on the derived masses m1 and m2
    p1 = m1.multiply(data_o[f'd{obv[0]}2t_oo'],axis=0)
    p2 = m2.multiply(data_o[f'd{obv[1]}2t_oo'],axis=0)

    data_o['p1'] = p1
    data_o['p2'] = p2

    pmag = np.sqrt(p1**2 + p2**2)
    p_theta = np.arcsin(p2/pmag)

    dp1dt = p1-p1.shift(1)
    dp2dt = p2-p2.shift(1) 

    data_o['dp1t_o'] = dp1dt
    data_o['dp2t_o'] = dp2dt

    dp1dv = (p1-p1.shift(1)).divide(data_o[f'd{obv[0]}2t_oo']-data_o[f'd{obv[0]}2t_oo'].shift(1),axis=0)
    dp2dv = (p2-p2.shift(1)).divide(data_o[f'd{obv[1]}2t_oo']-data_o[f'd{obv[1]}2t_oo'].shift(1),axis=0)

    data_o['dp1v_o'] = dp1dv
    data_o['dp2v_o'] = dp2dv 

    # angular natural freq
    w1_0 = np.sqrt(k1/m1)
    w2_0 = np.sqrt(k2/m2)

    data_o['w1_0'] = w1_0
    data_o['w2_0'] = w2_0 

    # temporal natural freq
    f1_0 = w1_0/(2*np.pi)
    f2_0 = w2_0/(2*np.pi)

    data_o['f1_0'] = f1_0
    data_o['f2_0'] = f2_0

    # damping ratio
    dr1 = c1.divide(2*np.sqrt(m1*k1),axis=0)
    dr2 = c2.divide(2*np.sqrt(m2*k2),axis=0)

    data_o['dr1'] = dr1
    data_o['dr2'] = dr2

    # anguluar freq
    w1 = w1_0.multiply(np.sqrt(1-dr1),axis=0)
    w2 = w2_0.multiply(np.sqrt(1-dr2),axis=0)

    data_o['w1'] = w1
    data_o['w2'] = w2

    # temporal freq
    f1 = w1/(2*np.pi)
    f2 = w2/(2*np.pi)

    data_o['f1'] = f1
    data_o['f2'] = f2

    # decay rate
    lmda1 = w1_0.multiply(dr1,axis=0)
    lmda2 = w2_0.multiply(dr2,axis=0)

    data_o['lmda1'] = lmda1
    data_o['lmda2'] = lmda2

    # Q factor
    q1 = (2*dr1)**-1
    q2 = (2*dr2)**-1

    data_o['q1'] = q1
    data_o['q2'] = q2

    data_o.fillna(0,inplace=True)

    for col in data_o.columns:
        data_o[col] = data_o[col].replace([np.inf, -np.inf], np.nan)
        data_o[col] = data_o[col].fillna(np.abs(data_o[col]).max())
 
    data_n = misc.normalizedf(data_o)

    return data_n