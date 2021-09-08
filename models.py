import numpy as np
import pandas as pd

def ddm(data,windows,obv=['c','v'],diff_offset=1,diff=1):
    data_m = data.copy()

    print('- Creating addtional features...\n')

    for i in range(1,len(obv)):
        # Create power, conductivity and resistance metrics
        # Create ideal position signal
        data_m[f'{obv[0]}{obv[i]}_pwr'] = data_m[f'd{obv[0]}1t_o'] * data_m[f'd{obv[i]}1t_o']
        data_m[f'{obv[0]}{obv[i]}_r'] = data_m[f'd{obv[0]}1t_o'].divide(data_m[f'd{obv[i]}1t_o'])
        data_m[f'{obv[0]}{obv[i]}_c'] = data_m[f'd{obv[i]}1t_o'].divide(data_m[f'd{obv[0]}1t_o'])

        data_m[f'idpos{obv[i]}'] = np.where(data_m[f'd{obv[i]}1t_o'].shift(0) > data_m[f'd{obv[i]}1t_o'].mean()+(1.5*data_m[f'd{obv[i]}1t_o']).std(),1,0)

        data_m.fillna(0,inplace=True)

    for o in obv:
        data_m[f'{o}_mean'] = data_m[f'd{o}1t_o'].mean()
        data_m[f'{o}_median'] = data_m[f'd{o}1t_o'].median()
        data_m[f'{o}_mode'] = data_m[f'd{o}1t_o'].mode()
        data_m[f'{o}_std'] = data_m[f'd{o}1t_o'].std()

        # Remove nans and infs
        if np.isinf(data_m[f'{obv[0]}{obv[i]}_c'].max()):
            data_m.replace([np.inf, -np.inf], np.nan,inplace=True)
            data_m.fillna(data_m[f'{obv[0]}{obv[i]}_c'].max(),inplace=True)
        
    data_m.fillna(0,inplace=True)

    print('Data modelling complete...\n')
    return data_m

def point_sys(data,obv=['c','v'],size=3):
    data_p = data.copy()

    if len(data_p) > size:
        print(f'\nCreating Parameter system of order:{size}...\n')
        for o in obv:
            for i in range(1,size+1):
                data_p[f'{o}{i}t_1'] = data_p[o] - data_p[o].shift(1)
                data_p[f'd{o}{i}t_o'] = data_p[f'{o}{i}t_1']
                data_p[f'{o}{i}rt_1'] = np.sqrt(data_p[f'{o}{i}t_1']**2 + i**2)
                data_p[f'{o}{i}angt_1'] = np.arcsin(data_p[f'{o}{i}t_1'].divide(data_p[f'{o}{i}rt_1']))
                if i > 1:
                    data_p[f'{o}{i}to_1'] = data_p[f'd{o}{i}t_o'] - data_p[f'd{o}{i}t_o'].shift(1)
                    data_p[f'd{o}{i}t_oo'] = data_p[f'{o}{i}to_1'].divide(i)
                    data_p[f'd{o}{i}angt_o'] = data_p[f'{o}{i}angt_1'] - data_p[f'{o}{i}angt_1'].shift(1)
                if i > 2:
                    data_p[f'{o}{i}too_1'] = data_p[f'd{o}{i}t_oo'] - data_p[f'd{o}{i}t_oo'].shift(1)
                    data_p[f'd{o}{i}t_ooo'] = data_p[f'{o}{i}too_1'].divide(i)
        data_p.fillna(0,inplace=True)

        for col in data_p.columns:
            data_p[col] = data_p[col]/np.abs(data_p[col]).max()
            data_p[col] = data_p[col].replace([np.inf, -np.inf], np.nan)
            data_p[col] = data_p[col].fillna(np.abs(data_p[col]).max())

        print('Parameters are ready...\n')
    return data_p
