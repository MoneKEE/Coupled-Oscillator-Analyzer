import numpy as np
import pandas as pd

def ddm(data,windows,obv,diff_offset=1,diff=1):
    # Create the model metrics
    # 1. Difference Signal ->dP(t) = P(t)-P(t-n)
    # 2. Polarity -> d(t) = (1,0,-1)
    # 3. Amplitude -> A(t) = abs(P(t)-P(t-n))
    # 4. Spread -> S(t) = H(t) - L(t) for the window
    # 5. Mean, Median, Std, Sum for the window
    # 6. dsum -> Sum(d(t)) for the window
    # 7. H, L, O for the window

    data_m = data.copy()

    print('- Processing data for past features...\n')

    # Create the difference signal for both price and volume
    print(f'- Creating difference {diff} with {diff_offset} offset...\n')

    dc1 = pd.DataFrame(columns=['val'])
    do1 = pd.DataFrame(columns=['val'])
    dh1 = pd.DataFrame(columns=['val'])
    dl1 = pd.DataFrame(columns=['val'])
    dv1 = pd.DataFrame(columns=['val'])

    dc1['val'] = (data_m['c'] - data_m['c'].shift(diff_offset))
    do1['val'] = (data_m['o'] - data_m['o'].shift(diff_offset))
    dh1['val'] = (data_m['h'] - data_m['h'].shift(diff_offset))
    dl1['val'] = (data_m['l'] - data_m['l'].shift(diff_offset))
    dv1['val'] = (data_m['v'] - data_m['v'].shift(diff_offset))

    dc2 = dc1 - dc1.shift(diff_offset)
    do2 = dc1 - do1.shift(diff_offset)
    dh2 = dh1 - dh1.shift(diff_offset)
    dl2 = dl1 - dl1.shift(diff_offset)
    dv2 = (dv1 - dv1.shift(diff_offset))/np.abs(dv1 - dv1.shift(diff_offset)).max()

    dc3 = dc2 - dc2.shift(diff_offset)
    do3 = dc2 - do2.shift(diff_offset)
    dh3 = dh2 - dh2.shift(diff_offset)
    dl3 = dl2 - dl2.shift(diff_offset)
    dv3 = dv2 - dv2.shift(diff_offset)

    for j in data_m.columns[:5]:
        for i in range(1,4):
            data_m[f'd{j}{i}'] = locals()[f'd{j}{i}']

    # Create the Amplitude metrics
    amps = pd.DataFrame(columns = [f'{obv[0]}',f'{obv[1]}'])
    amps[f'{obv[0]}'] = np.abs(data_m[obv[0]])
    amps[f'{obv[1]}'] = np.abs(data_m[obv[1]])

    # Create power and resistance metrics
    data_m['sys_pwr'] = data_m[obv[0]] * data_m[obv[1]]
    #data_m[f'sys_psd'] = np.abs(data_m['sys_pwr'])**2
    data_m['sys_r'] = (amps[f'{obv[1]}']/amps[f'{obv[0]}'])
    data_m.fillna(0,inplace=True)
    data_m['sys_c'] = amps[f'{obv[0]}']/amps[f'{obv[1]}']

    data_m.fillna(0,inplace=True)

    if np.isinf(data_m.sys_c.max()):
        data_m.replace([np.inf, -np.inf], np.nan,inplace=True)
        data_m.fillna(data_m.sys_c.max(),inplace=True)

    data_m['sys_r'] = data_m['sys_r']/data_m['sys_r'].max()
    data_m['sys_c'] = data_m['sys_c']/data_m['sys_c'].max()

    # set ideal position
    for i in ['c','o','h','l','v']:
        data_m[f'idpos{i}'] = np.where(data_m[f'd{i}1'] > 0,1,0)

    data_m.fillna(0,inplace=True)

    print('Field Stats')
    print('-'*30)

    return data_m
