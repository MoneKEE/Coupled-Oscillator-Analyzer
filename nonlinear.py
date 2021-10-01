import pandas as pd
import numpy as np
from scipy import signal as sig
from scipy.optimize import curve_fit
# from scipy import stats
# import matplotlib.pyplot as plt
# import cmath
# import misc

# The dual oscillator attempts to model price action through
# the motion of a two axis simple oscillator. It accepts
# two parameters X1 and X2 
def dual_oscillator(data,F,obv=['v','c'],m=1):

    N = len(data)
    sr = 0.0001
    Fs = 1/sr
    T = N/Fs
    df = 1/T
    dw = 2*np.pi*df
    ny = Fs/2

    data_o = data.copy()

    # e1 = np.array([1,0])
    # e2 = np.array([0,1])

    x1 = data_o[f'x1nm']
    x2 = data_o[f'x2nm']

    dotx1 = x1-x1.shift(1)
    dotx2 = x2-x2.shift(1)

    l1 = x1.shift(1)
    l2 = x2.shift(1)

    r1 = pd.DataFrame({'e1':x1.add(l1),'e2':x2})
    r2 = pd.DataFrame({'e1':x1,'e2':x2.add(l2)})

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

    # x1 freq info
    f1,s1 = sig.periodogram(x1,fs=Fs)
    b1 = round(len(f1)/3)

    print('sr:',sr,'Fs:',Fs,'T:',T,'df:',df,'dw:',dw)

    f2,s2 = sig.periodogram(x2,fs=Fs)
    b2 = round(len(f2)/3)

    # Return max freqs from bands l,m,h
    sl1 = [s1[:b1] ==s1[:b1].max()]
    sm1 = [s1[b1:-b1] ==s1[b1:-b1].max()]
    sh1 = [s1[-b1:] ==s1[-b1:].max()]

    freql1   = f1[:b1][sl1][0]
    freqm1   = f1[b1:-b1][sm1][0]
    freqh1   = f1[-b1:][sh1][0]

    sl1 = s1[:b1][s1[:b1] ==s1[:b1].max()][0]
    sm1 = s1[b1:-b1][s1[b1:-b1] ==s1[b1:-b1].max()][0]
    sh1 = s1[-b1:][s1[-b1:] ==s1[-b1:].max()][0]

    wl1   = 2*np.pi*freql1
    wm1   = 2*np.pi*freqm1
    wh1   = 2*np.pi*freqh1

    w1n = np.abs(np.abs(wh1-wm1) - wl1)
    data_o['w1n'] = w1n

    sl2 = [s2[:b2] ==s2[:b2].max()]
    sm2 = [s2[b2:-b2] ==s2[b2:-b2].max()]
    sh2 = [s2[-b2:] ==s2[-b2:].max()]

    freql2   = f2[:b2][sl2][0]
    freqm2   = f2[b2:-b2][sm2][0]
    freqh2   = f2[-b2:][sh2][0]

    sl2 = s2[:b2][s2[:b2] ==s2[:b2].max()][0]
    sm2 = s2[b2:-b2][s2[b2:-b2] ==s2[b2:-b2].max()][0]
    sh2 = s2[-b2:][s2[-b2:] ==s2[-b2:].max()][0]

    wl2   = 2*np.pi*freql2
    wm2   = 2*np.pi*freqm2
    wh2   = 2*np.pi*freqh2

    w2n = np.abs(np.abs(wh2-wm2) - wl2)
    data_o['w2n'] = w2n

    prof = pd.DataFrame(index=['x1','x2']
                        ,data=[[sl1,sm1,sh1,freql1,freqm1,freqh1,wl1,wm1,wh1],[sl2,sm2,sh2,freql2,freqm2,freqh2,wl2,wm2,wh2]]
                        ,columns=['sl','sm','sh','fl','fm','fh','wl','wm','wh'])
    
    # SPRING CONSTANTS
    k1 = m*w1n**2
    k2 = m*w2n**2

    data_o.fillna(0,inplace=True)

    # damping ratio
    dr1 = (al1*k1).divide(2*np.sqrt(np.abs(m*k1)),axis=0)
    dr2 = (al2*k2).divide(2*np.sqrt(np.abs(m*k2)),axis=0)

    data_o['dr1'] = dr1
    data_o['dr2'] = dr2

    # anguluar freq
    w1 = w1n*np.sqrt(np.abs(1-(2*(dr1**2))))
    w2 = w2n*np.sqrt(np.abs(1-(2*(dr2**2))))

    data_o['w1'] = np.round(w1,5)
    data_o['w2'] = np.round(w2,5)

    data_o.fillna(0,inplace=True)

    # DERIVE GENERAL SOLUTION AND FREE BODY FORCES
    x1_0 = x1[0]
    x2_0 = x2[0]

    t = np.arange(0,len(data_o))

    x1gn = pd.DataFrame({'val':x1_0*np.cos(prof[:1].wh.values*t)-prof[:1].sh.values*(prof[:1].wh.values**-1)*np.sin(prof[:1].wh.values*t)})
    x1gn += pd.DataFrame({'val':x1_0*np.cos(prof[:1].wm.values*t)-prof[:1].sm.values*(prof[:1].wm.values**-1)*np.sin(prof[:1].wm.values*t)})
    x1gn += pd.DataFrame({'val':x1_0*np.cos(prof[:1].wl.values*t)-prof[:1].sl.values*(prof[:1].wl.values**-1)*np.sin(prof[:1].wl.values*t)})

    x2gn = pd.DataFrame({'val':x2_0*np.cos(prof[-1:].wh.values*t)-prof[-1:].sh.values*(prof[-1:].wh.values**-1)*np.sin(prof[-1:].wh.values*t)})
    x2gn += pd.DataFrame({'val':x2_0*np.cos(prof[-1:].wm.values*t)-prof[-1:].sm.values*(prof[-1:].wm.values**-1)*np.sin(prof[-1:].wm.values*t)})
    x2gn += pd.DataFrame({'val':x2_0*np.cos(prof[-1:].wl.values*t)-prof[-1:].sl.values*(prof[-1:].wl.values**-1)*np.sin(prof[-1:].wl.values*t)})

    x1gn.set_index(data_o.index,inplace=True)
    x2gn.set_index(data_o.index,inplace=True)

    data_o['x1gn'] = x1gn
    data_o['x2gn'] = x2gn

    data_o['x1gnm'] = np.abs(np.fft.fft(x1gn))
    data_o['x1gna'] = np.angle(np.fft.fft(x1gn))

    data_o['x2gnm'] = np.abs(np.fft.fft(x2gn))
    data_o['x2gna'] = np.angle(np.fft.fft(x2gn))

    data_o.fillna(0,inplace=True)

    # THE PARTICULAR SOLUTION XP=X-XG
    x1pr = x1.sub(x1gn.val)
    x2pr = x2.sub(x2gn.val)

    data_o['x1pr'] = x1pr
    data_o['x2pr'] = x2pr

    data_o['x1prm'] = np.abs(np.fft.fft(x1pr))
    data_o['x1pra'] = np.angle(np.fft.fft(x1pr))

    data_o['x2prm'] = np.abs(np.fft.fft(x2pr))
    data_o['x2pra'] = np.angle(np.fft.fft(x2pr))

    ma1 = -(x1gn.add(l1,axis=0)).multiply(al1*k1,axis=0) - x1gn.multiply(al2*k2,axis=0)
    ma2 = -(x2gn.add(l2,axis=0)).multiply(al2*k2,axis=0) - x2gn.multiply(al1*k1,axis=0)

    data_o.fillna(0,inplace=True)

    # FORCED SYSTEM FORCES
    ft1 = ma1.val + al1*k1*(x1+l1) + al2*k2*x1
    ft2 = ma2.val + al2*k2*(x2+l2) + al1*k1*x2

    ff1 = ft1.subtract(ma1.val,axis=0)
    ff2 = ft2.subtract(ma2.val,axis=0)

    data_o['ft1'] = ft1
    data_o['ft2'] = ft2

    data_o['ff1'] = ff1
    data_o['ff2'] = ff2

    data_o['k1'] = k1
    data_o['k2'] = k2

    data_o['ma1'] = ma1
    data_o['ma2'] = ma2

    # Part to Whole Ratio
    x1ptr = x1pr/x1
    x2ptr = x2pr/x2

    data_o['x1ptr'] = x1ptr
    data_o['x2ptr'] = x2ptr

    # Resonance Factor
    resf1 = w1/w1n
    resf2 = w2/w2n

    data_o['resf1'] = resf1
    data_o['resf2'] = resf2

    # exponential decay
    lmda1 = dr1.multiply(w1n,axis=0)
    lmda2 = dr2.multiply(w2n,axis=0)

    data_o['lmda1'] = lmda1
    data_o['lmda2'] = lmda2

    # Q factor
    q1 = (2*dr1)**-1
    q2 = (2*dr2)**-1

    data_o['q1'] = q1
    data_o['q2'] = q2

    # Energy Profile
    x1pe = (del1**2)*k1*0.5
    x1ke = 0.5*m*(dotx1**2+dotx2**2)
    x1te = x1ke-x1pe
    
    data_o['x1pe'] = x1pe
    data_o['x1ke'] = x1ke
    data_o['x1te'] = x1te

    x2pe = (del2**2)*k2*0.5
    x2ke = 0.5*m*(dotx1**2+dotx2**2)
    x2te = x2ke-x2pe
    
    data_o['x2pe'] = x2pe
    data_o['x2ke'] = x2ke
    data_o['x2te'] = x2te

    x1ktr = x1ke/x1te 
    x1ptr = x1pe/x1te

    data_o['x1ktr'] = x1ktr
    data_o['x1ptr'] = x1ptr

    x2ktr = x2ke/x2te 
    x2ptr = x2pe/x2te

    data_o['x2ktr'] = x2ktr
    data_o['x2ptr'] = x2ptr

    # Work Profile
    x1wrk = ma1.multiply(r1mag,axis=0)
    x2wrk = ma2.multiply(r2mag,axis=0)

    data_o['x1wrk'] = x1wrk
    data_o['x2wrk'] = x2wrk

    # Power
    x1pwr = x1wrk-x1wrk.shift(1)
    x2pwr = x2wrk-x2wrk.shift(1)

    data_o['x1pwr'] = x1pwr
    data_o['x2pwr'] = x2pwr

    data_o.fillna(0,inplace=True)

    # TOTAL SYSTEM METRICS 
    # Position
    Pxm = np.sqrt(x1**2+x2**2)
    Pxa = np.arctan2(x2,x1)
    Pxt = (Pxm*np.sin(Pxa))+(Pxm*np.cos(Pxa))

    data_o['Pxm'] = Pxm
    data_o['Pxa'] = Pxa
    data_o['Pxt'] = Pxt 

    # General
    Xgm = np.sqrt((x1gn**2)+(x2gn**2))
    Xga = np.arctan2(x2gn,x1gn)
    Xgt = (Xgm*np.sin(Xga))+(Xgm*np.cos(Xga))

    data_o['Xgm'] = Xgm
    data_o['Xga'] = Xga
    data_o['Xgt'] = Xgt

    # Particular
    Xpm = np.sqrt((x1pr**2)+(x2pr**2))
    Xpa = np.arctan2(x2pr,x1pr)
    Xpt = (Xpm*np.sin(Xpa))+(Xpm*np.cos(Xpa))

    data_o['Xpm'] = Xpm
    data_o['Xpa'] = Xpa
    data_o['Xpt'] = Xpt

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

    # Damping Ratio
    Drm = np.sqrt(dr1**2+dr2**2)
    Dra = np.arctan2(dr2,dr1)
    Drt = Drm*np.cos(Dra)+Drm*np.sin(Dra)

    data_o['Drm'] = Drm
    data_o['Dra'] = Dra
    data_o['Drt'] = Drt

    # Work
    Wkm = np.sqrt(x1wrk**2 + x2wrk**2)
    Wka = np.arctan2(x2wrk,x1wrk)
    Wkt = Wkm*np.sin(Wka) + Wkm*np.cos(Wka)

    data_o['Wkm'] = np.sqrt(x1wrk**2+x2wrk**2)
    data_o['Wka'] = np.arctan2(x2wrk,x1wrk) 
    data_o['Wkt'] = Wkt

    # Energy component ratios
    TEr = x1te/x2te
    data_o['TEr'] = TEr

    # Power
    Pwm = np.sqrt(x1pwr**2 + x2pwr**2)
    Pwa = np.arctan2(x2pwr,x1pwr)
    Pwt = Pwm*np.sin(Pwa) + Pwm*np.cos(Pwa)

    data_o['Pwm'] = np.sqrt(x1pwr**2+x2pwr**2)
    data_o['Pwa'] = np.arctan2(x2pwr,x1pwr) 
    data_o['Pwt'] = Pwt

    data_o['d1dotPwdt'] = (data_o.Pwt-data_o.Pwt.shift(1))
    data_o['d2dotPwdt'] = (data_o.d1dotPwdt-data_o.d1dotPwdt.shift(1))

    # Impedence
    Rst = data_o.Pwt/x1**2
    data_o['Rst'] = Rst

    # Torque
    r=[[x1[i],x2[i]] for i in range(len(data_o))]
    f=[[data_o['ft1'][i],data_o['ft2'][i]] for i in range(len(data_o))]

    Trq = np.cross(r,f)
    data_o['Trq'] = Trq

    data_o.fillna(0,inplace=True)

    return data_o


def dualosc2(data,F,m=1):
    N = len(data)
    Fs = F
    sr = 1/Fs
    T = N/Fs
    df = 1/T
    dw = 2*np.pi*df
    ny = Fs/2

    data_o = data.copy()

    x1 = data_o.x1.round(5).reset_index(drop=True)
    dotx1 = data_o.d1dotx1.round(5).reset_index(drop=True)
    ddotx1 = data_o.d2dotx1.round(5).reset_index(drop=True)

    x2 = data_o.x2.round(5).reset_index(drop=True)
    dotx2 = data_o.d1dotx2.round(5).reset_index(drop=True)
    ddotx2 = data_o.d2dotx2.round(5).reset_index(drop=True)

    lin,w1o,a1o = signal_nat(x1,9)
    lin,w2o,a2o = signal_nat(x2,9)

    l1 = x1.shift(1)
    r1mag = np.sqrt((x1+l1)**2+x2**2)
    a1 = 1-(l1/r1mag)
    xg1=np.zeros(len(data_o))

    l2 = x2.shift(1)
    r2mag = np.sqrt((x2+l2)**2+x1**2)
    a2 = 1-(l2/r2mag) 
    xg2=np.zeros(len(data_o))

    fib = [1,2,3,5,8]
    prim = [1,3,5,7]

    even = [2,4,6,8]
    odd = [1,3,5,9]

    low = [1,2,3,4,5]
    high = [5,6,7,8,9]

    all = [1,2,3,4,5,6,7,8,9]

    L = round(512*T)

    for i in np.arange(len(w1o)):
        if i in all:
            w10 = w1o[i]
            w20 = w2o[i]

            b1=-1*np.abs(a1o[i])*w10**-1
            b2=-1*np.abs(a2o[i])*w20**-1
            p01 = [1,b1,w10]
            p02 = [1,b2,w20]

            res1 = curve_fit(fseries2,np.arange(0,len(data_o[:L])),x1[:L],p0=p01,maxfev=20000)
            res2 = curve_fit(fseries2,np.arange(0,len(data_o[:L])),x2[:L],p0=p02,maxfev=20000)

            xg1 += fseries2(np.arange(0,len(data_o)),res1[0][0],res1[0][1],res1[0][2])
            xg2 += fseries2(np.arange(0,len(data_o)),res2[0][0],res2[0][1],res2[0][2])

    xp1 = x1-xg1
    xp2 = x2-xg2

    k1 = m*w10**2
    k2 = m*w20**2

    ma1 = -(xg1 + l1)*a1*k1 - xg1*a2*k2
    ma2 = -(xg2 + l2)*a2*k2 - xg2*a1*k1

    dr1 = (a1*k1)/(2*np.sqrt(np.abs(m*k1)))
    dr2 = (a2*k2)/(2*np.sqrt(np.abs(m*k2)))

    w1 = w10*np.sqrt(np.abs(1-(2*(dr1**2))))
    w2 = w20*np.sqrt(np.abs(1-(2*(dr2**2))))

    rf1 = w10/w1
    rf2 = w20/w2

    q1 = (2*dr1)**-1
    q2 = (2*dr2)**-1

    lmda1 = dr1 *w10
    lmda2 = dr2 *w20

    del1 = r1mag - l1
    del2 = r2mag - l2

    ft1 = ma1 + a1*k1*(x1+l1) + a2*k2*x1
    ft2 = ma2 + a2*k2*(x2+l2) + a1*k1*x2

    ff1 = ft1 - ma1
    ff2 = ft2 - ma2

    pe1 = (del1**2)*k1*0.5
    ke1 = 0.5*m*(dotx1**2+dotx2**2)
    te1 = ke1-pe1

    pe2 = (del2**2)*k2*0.5
    ke2 = 0.5*m*(dotx1**2+dotx2**2)
    te2 = ke2-pe2

    wrk1 = ma1*r1mag
    wrk2 = ma2*r2mag

    pwr1 = wrk1 - wrk1.shift()
    pwr2 = wrk2 - wrk2.shift()

    psr1 = xp1/x1
    psr2 = xp2/x2

    # TOTAL SYSTEM FORMS
    Pxm = np.sqrt(x1**2+x2**2)
    Pxa = np.arctan2(x2,x1)

    Mam = np.sqrt(ma1**2+ma2**2)
    Maa = np.arctan2(ma2,ma1) 

    Ftm = np.sqrt(ft1**2+ft2**2)
    Fta = np.arctan2(ft2,ft1) 

    Ffm = np.sqrt(ff1**2+ff2**2)
    Ffa = np.arctan2(ff2,ff1) 

    Pwm = np.sqrt(pwr1**2+pwr2**2)
    Pwa = np.arctan2(pwr2,pwr1) 

    Wkm = np.sqrt(wrk1**2+wrk2**2)
    Wka = np.arctan2(wrk2,wrk1) 

    sys = pd.DataFrame({'Pxm':Pxm,'Pxa':Pxa,'Mam':Mam,'Maa':Maa,'Ftm':Ftm,'Fta':Fta,'Ffm':Ffm,'Ffa':Ffa,'Pwm':Pwm,'Pwa':Pwa,'Wkm':Wkm,'Wka':Wka})

    x1sln = pd.DataFrame({'x1':x1,'dotx1':dotx1,'ddotx1':ddotx1,'xg1':xg1,'xp1':xp1,'ma1':ma1,'ft1':ft1,'ff1':ff1,'te1':te1
                        ,'pe1':pe1,'ke1':ke1,'wrk1':wrk1,'pwr1':pwr1,'w1o':w1o[0],'w1':w1,'a1':a1,'k1':k1,'dr1':dr1,'lmda1':lmda1,'q1':q1,'rf1':rf1,'psr1':psr1})
                        
    x2sln = pd.DataFrame({'x2':x2,'dotx2':dotx2,'ddotx2':ddotx2,'xg2':xg2,'xp2':xp2,'ma2':ma2,'ft2':ft2,'ff2':ff2,'te2':te2
                            ,'pe2':pe2,'ke2':ke2,'wrk2':wrk2,'pwr2':pwr2,'w2o':w2o[0],'w2':w2,'a2':a2,'k2':k2,'dr2':dr2,'lmda2':lmda2,'q2':q2,'rf2':rf2,'psr2':psr2})

    x1sln = x1sln.fillna(0)
    x2sln = x2sln.fillna(0)

    x1sln = x1sln.apply(lambda x: x.replace(np.inf, x.quantile(.99)))
    x1sln = x1sln.apply(lambda x: x.replace(-np.inf, x.quantile(.01)))

    x2sln = x2sln.apply(lambda x: x.replace(np.inf, x.quantile(.99)))
    x2sln = x2sln.apply(lambda x: x.replace(-np.inf, x.quantile(.01)))

    sln = x1sln.join(x2sln).join(sys)

    pos = pd.DataFrame()

    pos['pos'] = np.where(ddotx2>0,1,np.where(ddotx2<0,-1,0))

    x1slnp = x1sln.join(pos.pos).fillna(0)
    x2slnp = x2sln.join(pos.pos).fillna(0)
    slnp = sln.join(pos.pos).fillna(0)

    return slnp


def signal_nat(x,bins=3):
    x1fft = np.fft.fft(x)
    x1frq = np.fft.fftshift(np.fft.fftfreq(len(x),1/(bins*2)))
    wo = [] ;ao=[]
    
    mask = x1frq >= 0
    
    for i in np.arange(0,bins):
        maskb = (x1frq > i) & (x1frq<=(i+1))
        try:
            wo.append(x1frq[maskb][x1fft[maskb] == x1fft[maskb].max()][0])
            ao.append(x1fft[maskb].max())
        except:
            pass
    flt1 = x1fft
    flt1[np.isin(x1frq,wo,invert=True)] = 0

    lin = np.fft.ifft(flt1)

    return lin, [x*2*np.pi for x in wo], ao
    
def fseries(x,a0,a1,b1,w):
    f = a0+a1*np.cos(w*x)+b1*np.sin(w*x)
    return f

def fseries2(x,a1,b1,w):
    f = a1*np.cos(w*x)+b1*np.sin(w*x)
    return f