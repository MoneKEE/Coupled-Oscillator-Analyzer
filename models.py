import numpy as np
import pandas as pd
from scipy import stats

def ddm(data,tgt=0.8):
    x1 = f'x1'
    x2 = f'x2'
    dotx2 = f'd1dotx2'
    
    data_m = data.copy()

    print('- Creating addtional features...\n')
    
    data_m['x1pol'] = np.where(data_m[x1]>0,1,np.where(data_m[x1]<0,-1,0))
    data_m['x2pol'] = np.where(data_m[x2]>0,1,np.where(data_m[x2]<0,-1,0))

    data_m['quad_abs'] = np.where((data_m[x2]>0) & (data_m[x1]>0),1,np.where((data_m[x2]>0) & (data_m[x1]<0),2,np.where((data_m[x2]<0) & (data_m[x1]>0),3,4)))
    data_m['idpos1'] = np.where((data_m[x2].shift(-1) >= 1.5*data_m[x2])
                                        ,1
                                        ,0
                                        )
    
    # data_m[f'{x1}_avg_cum'] = data[x1].expanding().mean()
    # data_m[f'{x2}_avg_cum'] = data[x2].expanding().mean()

    # data_m[f'{x1}_med_cum'] = data[x1].expanding().median()
    # data_m[f'{x2}_med_cum'] = data[x2].expanding().median()

    # data_m[f'{x1}_var_cum'] = data[x1].expanding().var()
    # data_m[f'{x2}_var_cum'] = data[x2].expanding().var()

    # data_m[f'{x1}_min_cum'] = data[x1].expanding().min()
    # data_m[f'{x2}_min_cum'] = data[x2].expanding().min()
    
    # data_m[f'{x1}_max_cum'] = data[x1].expanding().max()
    # data_m[f'{x2}_max_cum'] = data[x2].expanding().max()

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
