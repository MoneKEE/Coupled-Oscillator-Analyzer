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
def dual_oscillator(data,Fs,obv=['v','c'],m=1):

    data_o = data.copy()

    # e1 = np.array([1,0])
    # e2 = np.array([0,1])

    x1 = f'd{obv[0]}1t_o'
    x2 = f'd{obv[1]}1t_o'

    dotx1 = f'd{obv[0]}2t_oo'
    dotx2 = f'd{obv[1]}2t_oo'

    ddotx1 = f'd{obv[0]}3t_ooo'
    ddotx2 = f'd{obv[1]}3t_ooo'

    l1 = -data_o[x1].shift(1)
    l2 = -data_o[x2].shift(1)

    r1 = pd.DataFrame({'e1':data_o[x1].add(l1),'e2':data_o[x2]})
    r2 = pd.DataFrame({'e1':data_o[x1],'e2':data_o[x2].add(l2)})

    r1mag = np.sqrt((r1['e1']**2)+(r1['e2'])**2)
    r2mag = np.sqrt((r2['e1']**2)+(r2['e2'])**2)

    # er1 = r1.divide(r1mag,axis=0)
    # er2 = r2.divide(r2mag,axis=0)

    del1 = r1mag - l1
    del2 = r2mag - l2

    al1 = 1 - (l1/r1mag)
    al2 = 1 - (l2/r2mag)

    data_o['al1'] = al1
    data_o['al2'] = al2

    spec,freq,line = plt.magnitude_spectrum(data_o[x1],Fs=Fs)
    w1_n = freq[spec==spec.max()][0]
    # w1_n = data_o.fft_freq[data_o.vf_w==data_o.vf_w]

    spec,freq,line = plt.magnitude_spectrum(data_o[x2],Fs=Fs)
    w2_n = freq[spec==spec.max()][0]
    # w2_n = data_o[dotx2]/data_o[x2]

    k1 = m*w1_n**2
    k2 = m*w2_n**2

    # ma1 = -(al1*k1)*data_o[dotx1] - al2*k2*data_o[x1]
    # ma2 = -(al2*k2)*data_o[dotx2] - al1*k1*data_o[x2]

    # ft1 = m*data_o[ddotx1] + al1*k1*data_o[dotx1] + al2*k2*data_o[x1]
    # ft2 = m*data_o[ddotx2] + al2*k2*data_o[dotx2] + al1*k1*data_o[x2]

    ma1 = -(al1*k1)*(data_o[x1]+l1) - al2*k2*data_o[x1]
    ma2 = -(al2*k2)*(data_o[x2]+l2) - al1*k1*data_o[x2]

    mam = np.sqrt(ma1**2 + ma2**2)
    maa = np.arctan(ma2/ma1)

    ft1 = m*data_o[ddotx1] + al1*k1*(data_o[x1]+l1) + al2*k2*data_o[x1]
    ft2 = m*data_o[ddotx2] + al2*k2*(data_o[x2]+l2) + al1*k1*data_o[x2]

    ftm = np.sqrt(ft1**2+ft2**2)
    fta = np.arctan(ft2/ft1)

    data_o['ft1'] = ft1
    data_o['ft2'] = ft2

    data_o['ftm'] = ftm
    data_o['fta'] = fta

    ac1 = ma1/m
    ac2 = ma2/m

    data_o['k1'] = k1
    data_o['k2'] = k2

    a1k1 = al1*k1
    a2k2 = al2*k2

    data_o['a1k1'] = a1k1
    data_o['a2k2'] = a2k2

    data_o['ac1'] = ac1
    data_o['ac2'] = ac2

    data_o['ma1'] = ma1
    data_o['ma2'] = ma2

    # # Momentum calculations based on the derived masses m1 and m2
    p1 = m*data_o[dotx1]
    p2 = m*data_o[dotx2]

    data_o['p1'] = p1
    data_o['p2'] = p2

    pm = np.sqrt(p1**2 + p2**2)
    pa = np.arcsin(p2/pm)

    dp1dt = p1-p1.shift(1)
    dp2dt = p2-p2.shift(1) 

    data_o['dp1t_o'] = dp1dt
    data_o['dp2t_o'] = dp2dt

    dp1dv = (p1-p1.shift(1)).divide(data_o[ddotx1],axis=0)
    dp2dv = (p2-p2.shift(1)).divide(data_o[ddotx2],axis=0)

    data_o['dp1v_o'] = dp1dv
    data_o['dp2v_o'] = dp2dv 

    data_o['w1_n'] = w1_n
    data_o['w2_n'] = w2_n

    # Energy Profile
    PE1 = 0.5*k1*(del1**2)
    KE1 = 0.5*m*(data_o[dotx1]**2+data_o[dotx2]**2)
    TE1 = KE1-PE1
    
    data_o['PE1'] = PE1
    data_o['KE1'] = KE1
    data_o['TE1'] = TE1

    PE2 = 0.5*k2*(del2**2)
    KE2 = 0.5*m*(data_o[dotx1]**2+data_o[dotx2]**2)
    TE2 = KE2-PE2
    
    data_o['PE2'] = PE2
    data_o['KE2'] = KE2
    data_o['TE2'] = TE2

    # temporal natural freq
    fr1_n = w1_n/(2*np.pi)
    fr2_n = w2_n/(2*np.pi)

    data_o['fr1_n'] = fr1_n
    data_o['fr2_n'] = fr2_n

    # damping ratio
    dr1 = a1k1.divide(2*np.sqrt(m*k1),axis=0)
    dr2 = a2k2.divide(2*np.sqrt(m*k2),axis=0)

    data_o['dr1'] = dr1
    data_o['dr2'] = dr2

    # anguluar freq
    w1 = np.sqrt(1-dr1).multiply(w1_n,axis=0)
    w2 = np.sqrt(1-dr2).multiply(w2_n,axis=0)

    data_o['w1'] = w1
    data_o['w2'] = w2

    # temporal freq
    fr1 = w1/(2*np.pi)
    fr2 = w2/(2*np.pi)

    data_o['fr1'] = fr1
    data_o['fr2'] = fr2

    # exponential decay
    lmda1 = dr1.multiply(w1_n,axis=0)
    lmda2 = dr2.multiply(w2_n,axis=0)

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