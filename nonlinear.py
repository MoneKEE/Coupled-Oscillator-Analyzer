import pandas as pd
import numpy as np
from scipy import signal as sig
from scipy.optimize import curve_fit
# from scipy import stats
import matplotlib.pyplot as plt
# import cmath
# import misc

# The dual oscillator attempts to model price action through
# the motion of a two axis simple oscillator. It accepts
# two parameters X1 and X2 
def dualosc2(data,F,hrm,m=1):
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

    high = [5,6,7,8,9]
    mid = [3,4,5,6,7]
    low = [1,2,3,4,5]

    all = [1,2,3,4,5,6,7,8,9]

    L = len(x1)//2

    for i in np.arange(len(w1o)):
        if i+1 in locals()[hrm]:
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
    tes1 = ke1+pe1

    pe2 = (del2**2)*k2*0.5
    ke2 = 0.5*m*(dotx1**2+dotx2**2)
    te2 = ke2-pe2
    tes2 = ke2+pe2

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

    Sen = (te1-te2)/(te1+te2)

    # Torque
    r=[[r1mag[i],r2mag[i]] for i in range(len(x1))]
    f=[[ft1[i],ft2[i]] for i in range(len(x1))]

    Trq = np.cross(r,f)

    sys = pd.DataFrame({'Sen':Sen,'Pxm':Pxm,'Pxa':Pxa,'Mam':Mam,'Maa':Maa,'Ftm':Ftm,'Fta':Fta,'Ffm':Ffm,'Ffa':Ffa,'Pwm':Pwm,'Pwa':Pwa,'Wkm':Wkm,'Wka':Wka,'Trq':Trq})

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