import numpy as np
import pandas as pd
from scipy import stats

def ddm(data,tgt=0.8):
    print('\t- Starting DDM...\n')
    x1 = data.x1
    x2 = data.x2
    
    data_m = data.copy()

    data_m['pctchg'] = data_m.c.pct_change()
    data_m['logret'] = np.log(1+data_m.pctchg)
    
    data_m['x1pol'] = np.where(x1>0,1,np.where(x1<0,-1,0))
    data_m['x2pol'] = np.where(x2>0,1,np.where(x2<0,-1,0))
    data_m['quad_abs'] = np.where((x2>0) & (x1>0),1,np.where((x2>0) & (x1<0),2,np.where((x2<0) & (x1>0),3,4)))
    
    data_m.fillna(0,inplace=True)

    return data_m

def point_sys(data,obv=['v','c'],size=3,dt=1):
    data_p = data.copy()

    if len(data_p) >= size:
        print(f'\t- Starting point system order:{size}...\n')
        for k,o in enumerate(obv,1):
            for i in range(1,size+1):
                diff1 = data_p[o] - data_p[o].shift(1)
                data_p[f'x{k}'] = diff1.divide(dt)

                if i > 1:
                    diff = diff1 - diff1.shift(1)
                    ndiff = diff1.shift(-1) - diff1
                    data_p[f'dotx{k}'] = diff.divide(dt*1)
                    data_p[f'-dotx{k}'] = ndiff.divide(dt*1)
                if i > 2:
                    diff = diff1 - diff1.shift(2)
                    ndiff = diff1.shift(-2) - diff1
                    data_p[f'ddotx{k}'] = diff.divide(dt*2)
                    data_p[f'-ddotx{k}'] = ndiff.divide(dt*2)
                if i > 3:
                    diff = diff1 - diff1.shift(3)
                    ndiff = diff1.shift(-3) - diff1
                    data_p[f'd3dotx{k}'] = diff.divide(dt*3)
                    data_p[f'-d3dotx{k}'] = ndiff.divide(dt*3)
                if i > 4:
                    diff = diff1 - diff1.shift(4)
                    ndiff = diff1.shift(-4) - diff1
                    data_p[f'd4dotx{k}'] = diff.divide(dt*4)
                    data_p[f'-d4dotx{k}'] = ndiff.divide(dt*4)

                # if i > 1:
                #     diff2 = diff1-diff1.shift(1)
                #     data_p[f'd{i-1}dotx{k}'] = diff2.divide(dt*1)
                # if i > 2:
                #     diff3 = diff1-diff1.shift(2)
                #     data_p[f'd{i-1}dotx{k}'] = diff3.divide(dt*2)
                # if i > 3:
                #     diff4 = diff1-diff1.shift(3)
                #     data_p[f'd{i-1}dotx{k}'] = diff4.divide(dt*3)
                # if i > 4:
                #     diff5 = diff1-diff1.shift(4)
                #     data_p[f'd{i-1}dotx{k}'] = diff5.divide(dt*4)

    data_p.fillna(0,inplace=True)
    return data_p
