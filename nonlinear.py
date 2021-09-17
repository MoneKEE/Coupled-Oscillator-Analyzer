import pandas as pd
import numpy as np
from scipy import signal as sig
from scipy import stats

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

    er1 = r1.divide(r1mag,axis=0)
    er2 = r2.divide(r2mag,axis=0)

    del1 = r1mag - l1
    del2 = r2mag - l2

    al1 = 1 - (l1/r1mag)
    al2 = 1 - (l2/r2mag)

    data_o['al1'] = al1
    data_o['al2'] = al2

    # APPROX THE NATURAL FREQ USING ALL PEAKS
    # RESIDING A CERTAIN DISTANCE ABOVE THE MEAN
    f,s = sig.periodogram(data_o[x1],fs=Fs)

    freq   = f[s == s.max()][0]
    w1_n   = 2*np.pi*freq

    f,s = sig.periodogram(data_o[x2],fs=Fs)

    freq   = f[s == s.max()][0]
    w2_n   = 2*np.pi*freq

    # SPRING CONSTANTS
    k1 = m*w1_n**2 
    k2 = m*w2_n**2

    #  DERIVE GENERAL SOLUTION AND FREE BODY FORCES
    x1_0 = 0
    x2_0 = 0

    vals = data_o[x1].value_counts()
    dotx1_0 = (data_o[dotx1].describe()['75%'] - data_o[dotx1].describe()['25%'])/2

    vals = data_o[x2].value_counts()
    dotx2_0 = (data_o[dotx1].describe()['75%'] - data_o[dotx1].describe()['25%'])/2

    t = np.arange(0,len(data_o))

    x1_gn = pd.DataFrame({'val':x1_0*np.cos(w1_n*t)+dotx1_0*np.sin(w1_n*t)})
    x2_gn = pd.DataFrame({'val':x2_0*np.cos(w2_n*t)+dotx2_0*np.sin(w2_n*t)})

    x1_gn.set_index(data_o.index,inplace=True)
    x2_gn.set_index(data_o.index,inplace=True)

    data_o['x1_gn'] = x1_gn
    data_o['x2_gn'] = x2_gn

    data_o.fillna(0,inplace=True)

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

    data_o['al1n'] = al1n
    data_o['al2n'] = al2n

    ma1 = -(x1_gn+l1_n).multiply(al1n*k1,axis=0) - x1_gn.multiply(al2n*k2,axis=0)
    ma2 = -(x2_gn+l2_n).multiply(al2n*k2,axis=0) - x2_gn.multiply(al1n*k1,axis=0)

    data_o.fillna(0,inplace=True)

    ## FORCED SYSTEM FORCES
    ft1 = ma1.val + al1*k1*(data_o[x1]+l1) + al2*k2*data_o[x1]
    ft2 = ma2.val + al2*k2*(data_o[x2]+l2) + al1*k1*data_o[x2]

    ff1 = ft1.subtract(ma1.val,axis=0)
    ff2 = ft2.subtract(ma2.val,axis=0)

    data_o['ft1'] = ft1
    data_o['ft2'] = ft2

    data_o['ff1'] = ff1
    data_o['ff2'] = ff2

    ac1 = ma1/m
    ac2 = ma2/m

    data_o['k1'] = k1
    data_o['k2'] = k2

    data_o['ac1'] = ac1
    data_o['ac2'] = ac2

    data_o['ma1'] = ma1
    data_o['ma2'] = ma2

    data_o['w1_n'] = w1_n
    data_o['w2_n'] = w2_n

    # Part to Gen Ratio
    Xpr1 = x1_pr/data_o[x1]
    Xgr1 = x2_gn.val/data_o[x1]

    Xpr2 = x2_pr/data_o[x2]
    Xgr2 = x2_gn.val/data_o[x2]

    data_o['Xpr1'] = Xpr1
    data_o['Xgr1'] = Xgr1

    data_o['Xpr2'] = Xpr2
    data_o['Xgr2'] = Xgr2

    # damping ratio
    dr1 = (al1*k1).divide(2*np.sqrt(m*k1),axis=0)
    dr2 = (al2*k2).divide(2*np.sqrt(m*k2),axis=0)

    data_o['dr1'] = dr1
    data_o['dr2'] = dr2

    # anguluar freq
    w1 = np.sqrt(1-dr1).multiply(w1_n,axis=0)
    w2 = np.sqrt(1-dr2).multiply(w2_n,axis=0)

    data_o['w1'] = w1
    data_o['w2'] = w2

    # Resonance Factor
    data_o['rf1'] = w1/w1_n
    data_o['rf2'] = w2/w2_n

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
    PE1 = (del1**2)*k1*0.5
    KE1 = 0.5*m*(data_o[dotx1]**2+data_o[dotx2]**2)
    TE1 = KE1-PE1
    
    data_o['PE1'] = PE1
    data_o['KE1'] = KE1
    data_o['TE1'] = TE1

    PE2 = (del2**2)*k2*0.5
    KE2 = 0.5*m*(data_o[dotx1]**2+data_o[dotx2]**2)
    TE2 = KE2-PE2
    
    data_o['PE2'] = PE2
    data_o['KE2'] = KE2
    data_o['TE2'] = TE2

    Kr1 = KE1/TE1 
    Pr1 = PE1/TE1

    TEr = TE1/TE2

    data_o['Kr1'] = Kr1
    data_o['Pr1'] = Pr1

    Kr2 = KE2/TE2 
    Pr2 = PE2/TE2

    data_o['Kr2'] = Kr2
    data_o['Pr2'] = Pr2

    # Work Profile
    wrk1 = ft1*data_o[x1]
    wrk2 = ft2*data_o[x2]

    data_o['wrk1'] = wrk1
    data_o['wrk2'] = wrk2

    # Power
    pwr1 = wrk1-wrk1.shift(1)
    pwr2 = wrk2-wrk2.shift(1)

    data_o['pwr1'] = pwr1
    data_o['pwr2'] = pwr2

    data_o.fillna(0,inplace=True)

    # TOTAL SYSTEM METRICS 
    # Position
    Pxm = np.sqrt(data_o[x1]**2+data_o[x2]**2)
    Pxa = np.arctan2(data_o[x2],data_o[x1])
    Pxt = (Pxm*np.sin(Pxa))+(Pxm*np.cos(Pxa))

    data_o['Pxm'] = Pxm
    data_o['Pxa'] = Pxa
    data_o['Pxt'] = Pxt 

    # Total Force
    Ftm = np.sqrt(ft1**2+ft2**2)
    Fta = np.arctan2(ft2,ft1)
    Ftt = Ftm*np.sin(Fta)+Ftm*np.cos(Fta)

    data_o['Ftm'] = Ftm
    data_o['Fta'] = Fta
    data_o['Ftt'] = Ftt

    # Free Body Force
    Mam = np.sqrt(ma1**2 + ma2**2)
    Maa = np.arctan2(ma2,ma1)
    Mat = Mam*np.sin(Maa)+Mam*np.cos(Maa)

    data_o['Mam'] = Mam
    data_o['Maa'] = Maa
    data_o['Mat'] = Mat

    # Forcing Function
    Ffm = np.sqrt(ff1**2 + ff2**2)
    Ffa = np.arctan2(ff2,ff1)
    Fft = Ffm*np.sin(Ffa)+Ffm*np.cos(Ffa)

    data_o['Ffm'] = Ffm
    data_o['Ffa'] = Ffa
    data_o['Fft'] = Fft

    # Qf Factor
    Qfm = np.sqrt(q1**2+q2**2)
    Qfa = np.arctan2(q2,q1)
    Qft = Qfm*np.sin(Qfa)+Qfm*np.cos(Qfa)

    data_o['Qfm'] = Qfm
    data_o['Qfa'] = Qfa
    data_o['Qft'] = Qft

    # # Lambda
    # data_o['Lmm'] = np.sqrt(lmda1**2 + lmda2**2)
    # data_o['Lma'] = np.arctan(lmda2/lmda1)

    # # Damping Ratio
    Drm = np.sqrt(dr1**2+dr2**2)
    Dra = np.arctan2(dr2,dr1)
    Drt = Drm*np.cos(Dra)+Drm*np.sin(Dra)

    data_o['Drm'] = Drm
    data_o['Dra'] = Dra
    data_o['Drt'] = Drt

    # Work
    Wkm = np.sqrt(wrk1**2 + wrk2**2)
    Wka = np.arctan2(wrk2,wrk1)
    Wkt = Wkm*np.sin(Wka) + Wkm*np.cos(Wka)

    data_o['Wkm'] = np.sqrt(wrk1**2+wrk2**2)
    data_o['Wka'] = np.arctan2(wrk2,wrk1) 
    data_o['Wkt'] = Wkt

    # Power
    Pwm = np.sqrt(pwr1**2 + pwr2**2)
    Pwa = np.arctan2(pwr2,pwr1)
    Pwt = Pwm*np.sin(Pwa) + Pwm*np.cos(Pwa)

    data_o['Pwm'] = np.sqrt(pwr1**2+pwr2**2)
    data_o['Pwa'] = np.arctan2(pwr2,pwr1) 
    data_o['Pwt'] = Pwt

    # Torque
    # Still trying to figure this part out
    r=[[data_o[x1][i],data_o[x2][i]] for i in range(len(data_o))]
    f=[[data_o['ft1'][i],data_o['ft2'][i]] for i in range(len(data_o))]

    Trq = np.cross(r,f)
    data_o['Trq'] = Trq
    data_o.fillna(0,inplace=True)

    return data_o
    