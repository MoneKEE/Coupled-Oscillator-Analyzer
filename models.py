import numpy as np
import pandas as pd
from scipy import stats

def ddm(data,tgt=0.8):
    x1 = data.x1
    x2 = data.x2

    dotx1 = data.d1dotx1
    dotx2 = data.d1dotx2
    
    data_m = data.copy()

    print('- Creating addtional features...\n')
    
    data_m['x1pol'] = np.where(x1>0,1,np.where(x1<0,-1,0))
    data_m['x2pol'] = np.where(x2>0,1,np.where(x2<0,-1,0))

    data_m['quad_abs'] = np.where((x2>0) & (x1>0),1,np.where((x2>0) & (x1<0),2,np.where((x2<0) & (x1>0),3,4)))
    data_m['pos'] = np.where(dotx2>0,1,0)

    print('Data modelling complete...\n')

    data_m.fillna(0,inplace=True)

    return data_m

def point_sys(data,obv=['v','c'],size=3,dt=1):
    data_p = data.copy()

    if len(data_p) >= size:
        print(f'\nCreating Parameter system of order:{size}...\n')
        for k,o in enumerate(obv,1):
            for i in range(1,size+1):
                diff1 = data_p[o] - data_p[o].shift(1)
                data_p[f'x{k}'] = diff1.divide(dt)

                if i > 1:
                    diff2 = diff1 - diff1.shift(1)
                    data_p[f'd{i-1}dotx{k}'] = diff2.divide(dt)
                if i > 2:
                    diff3 = diff2 - diff2.shift(1)
                    data_p[f'd{i-1}dotx{k}'] = diff3.divide(dt)
                if i > 3:
                    diff4 = diff3 - diff3.shift(1)
                    data_p[f'd{i-1}dotx{k}'] = diff4.divide(dt)
                if i > 4:
                    diff5 = diff4 - diff4.shift(1)
                    data_p[f'd{i-1}dotx{k}'] = diff5.divide(dt)

        print('Parameters are ready...\n')

    data_p.fillna(0,inplace=True)
    return data_p
