import numpy as np
import pandas as pd

def ddm(data,windows,obv=['c','v'],diff_offset=1,diff=1,tgt=0.8):
    data_m = data.copy()

    print('- Creating addtional features...\n')

    data_m[f'idpos1'] = np.where((data_m.dc1t_o > data_m.dc1t_o.quantile(tgt))
                                        ,1
                                        ,0
                                        )
    data_m[f'idpos2'] = np.where((data_m.dc1t_o > data_m.dc1t_o.quantile(tgt)) &(data_m.dv1t_o.abs()<data_m.dv1t_o.quantile(tgt))
                                        ,1
                                        ,0
                                        )
    data_m.fillna(0,inplace=True)

    print('Data modelling complete...\n')
    return data_m

def point_sys(data,obv=['c','v'],size=3):
    data_p = data.copy()

    if len(data_p) >= size:
        print(f'\nCreating Parameter system of order:{size}...\n')
        for o in obv:
            for i in range(1,size+1):
                data_p[f'{o}{i}t_1'] = data_p[o] - data_p[o].shift(1)
                data_p[f'd{o}{i}t_o'] = data_p[f'{o}{i}t_1']
                if i > 1:
                    data_p[f'{o}{i}to_1'] = data_p[f'd{o}{i}t_o'] - data_p[f'd{o}{i}t_o'].shift(1)
                    data_p[f'd{o}{i}t_oo'] = data_p[f'{o}{i}to_1'].divide(i)
                if i > 2:
                    data_p[f'{o}{i}too_1'] = data_p[f'd{o}{i}t_oo'] - data_p[f'd{o}{i}t_oo'].shift(1)
                    data_p[f'd{o}{i}t_ooo'] = data_p[f'{o}{i}too_1'].divide(i)
                if i > 3:
                    data_p[f'{o}{i}tooo_1'] = data_p[f'd{o}{i}t_ooo'] - data_p[f'd{o}{i}t_ooo'].shift(1)
                    data_p[f'd{o}{i}t_oooo'] = data_p[f'{o}{i}tooo_1'].divide(i)
        data_p.fillna(0,inplace=True)

        print('Parameters are ready...\n')
    return data_p
