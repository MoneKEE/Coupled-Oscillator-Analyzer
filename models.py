import numpy as np
import pandas as pd
from datetime import datetime as dt

def ddm(data,windows,obv=['c','v'],diff_offset=1,diff=1):
    data_m = data.copy()

    print('- Creating addtional features...\n')

    for i in range(1,len(obv)):
        # Create power, conductivity and resistance metrics
        # Create ideal position signal
        data_m[f'{obv[0]}{obv[i]}_pwr'] = data_m[f'd{obv[0]}t_o'] * data_m[f'd{obv[i]}t_o']
        data_m[f'{obv[0]}{obv[i]}_r'] = data_m[f'd{obv[0]}t_o'].divide(data_m[f'd{obv[i]}t_o'])
        data_m[f'{obv[0]}{obv[i]}_c'] = data_m[f'd{obv[i]}t_o'].divide(data_m[f'd{obv[0]}t_o'])
        data_m[f'idpos{obv[i]}'] = np.where(data_m[f'd{obv[i]}t_o'] > 0,1,0)

        data_m.fillna(0,inplace=True)

        # Remove nans and infs
        if np.isinf(data_m[f'{obv[0]}{obv[i]}_c'].max()):
            data_m.replace([np.inf, -np.inf], np.nan,inplace=True)
            data_m.fillna(data_m.data_m[f'{obv[0]}{obv[i]}_c'].max(),inplace=True)
        
    data_m.fillna(0,inplace=True)

    print('Data modelling complete...\n')
    return data_m

def point_sys(data,obv=['c','v'],size=3):
    data_p = data.copy()

    if len(data_p) > size:
        print(f'\nCreating Parameter system of order:{size}...\n')
        for o in obv:
            for i in range(1,size+1):
                data_p[f'{o}t_{i}'] = data_p[o] - data_p[o].shift(i)
                data_p[f'd{o}t_'+'o'*i] = data_p[f'{o}t_{i}'].divide(i)
                data_p[f'{o}rt_{i}'] = np.sqrt(data_p[f'{o}t_{i}']**2 + i**2)
                data_p[f'{o}angt_{i}'] = np.rad2deg(np.arcsin(data_p[f'{o}t_{i}'].divide(data_p[f'{o}rt_{i}'])))
        data_p.fillna(0,inplace=True)

        print('Parameters are ready...\n')
    return data_p
