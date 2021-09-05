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

    l1 = data_o[x1].shift(1)
    l2 = data_o[x2].shift(1)

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

    f,s = sig.periodogram(data_o[x1],fs=Fs)

    fr1_n   = f[s == s.max()][0]
    w1_n    = 2*np.pi*fr1_n

    f,s = sig.periodogram(data_o[x2],fs=Fs)

    fr2_n   = f[s == s.max()][0]
    w2_n    = 2*np.pi*fr2_n

    k1 = m*w1_n**2
    k2 = m*w2_n**2

    ## DERIVE GENERAL SOLUTION AND FREE BODY FORCES
    x1_0 = data_o[x1][0]
    x2_0 = data_o[x2][0]

    dotx1_0 = data_o[x1][data_o[x1].abs()>0][0]-data_o[x1][0]
    dotx2_0 = data_o[x2][data_o[x2].abs()>0][0]-data_o[x2][0]

    t = np.arange(0,len(data_o))

    x1_gn = pd.DataFrame({'val':x1_0*np.cos(w1_n*t)+dotx1_0*np.sin(w1_n*t)})
    x2_gn = pd.DataFrame({'val':x2_0*np.cos(w2_n*t)+dotx2_0*np.sin(w2_n*t)})

    x1_gn.set_index(data_o.index,inplace=True)
    x2_gn.set_index(data_o.index,inplace=True)

    data_o['x1_gn'] = x1_gn
    data_o['x2_gn'] = x2_gn

    ## THE PARTICULAR SOLUTION XP=X-XG
    x1_pr = data_o[x1].sub(x1_gn.val)
    x2_pr = data_o[x2].sub(x2_gn.val)

    data_o['x1_pr'] = x1_pr
    data_o['x2_pr'] = x2_pr

    l1_n = x1_gn.shift(1)
    l2_n = x2_gn.shift(1)

    r1_n = pd.DataFrame({'e1':x1_gn.add(l1_n).iloc[:,0],'e2':x2_gn.iloc[:,0]}).set_index(data_o.index)
    r2_n = pd.DataFrame({'e1':x1_gn.iloc[:,0],'e2':x2_gn.add(l2_n).iloc[:,0]}).set_index(data_o.index)

    r1magn = np.sqrt((r1_n['e1']**2)+(r1_n['e2'])**2)
    r2magn = np.sqrt((r2_n['e1']**2)+(r2_n['e2'])**2)

    al1n = 1-(l1_n.div(r1magn,axis=0))
    al2n = 1-(l2_n.div(r2magn,axis=0))

    ma1 = -(x1_gn+l1_n).multiply(al1n*k1,axis=0) - x1_gn.multiply(al2n*k2,axis=0)
    ma2 = -(x2_gn+l2_n).multiply(al2n*k2,axis=0) - x2_gn.multiply(al1n*k1,axis=0)

    mam = np.sqrt(ma1**2 + ma2**2)
    maa = np.arctan(ma2/ma1)

    data_o['mam'] = mam
    data_o['maa'] = maa

    ## FORCED SYSTEM FORCES
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

    data_o['w1_n'] = w1_n
    data_o['w2_n'] = w2_n

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

    data_o.fillna(0,inplace=True)

    for col in data_o.columns:
        data_o[col] = data_o[col].replace([np.inf, -np.inf], np.nan)
        data_o[col] = data_o[col].fillna(np.abs(data_o[col]).max())
 
    data_n = misc.normalizedf(data_o)

    return data_n