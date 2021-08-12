import pandas as pd
import cbpro
import math as mt
from datetime import timedelta
from datetime import datetime as dt

def get_data_span(asset,start,stop,interval,mode):

    pc      = cbpro.PublicClient()
    data    = pd.DataFrame()

    intervals = {'days':86400,
                'hours':3600,
                'minutes':60}

    print('='*80)
    print(  f'Starting {mode} run for date range {start} - {stop}'
            ,'\n'
            )
    print(  '='*80
            ,'\n'
            )

    diff = stop - start
    d_s = diff.total_seconds()
    terms = mt.ceil(diff.days/300)
    if interval == 'minutes':
        terms = mt.ceil((d_s/60)/300)
    elif interval == 'hours':
        terms = mt.ceil((d_s/3600)/300)

    for term in range(terms):
        if interval == 'minutes':
            strt = timedelta(minutes=300*term)
            end = timedelta(minutes=300*(term+1))
        elif interval == 'hours':
            strt = timedelta(hours=300*term)
            end = timedelta(hours=300*(term+1))
        else:
            strt = timedelta(days=300*term)
            end = timedelta(days=300*(term+1))

        raw_data = pc.get_product_historic_rates(asset
                                                ,start + strt
                                                ,start + end
                                                ,intervals[interval]
                                                )

        data = data.append(raw_data)

    columnlist = {0:'t',1:'l',2:'h',3:'o',4:'c',5:'v'}
    data.rename(columns = columnlist,inplace=True)

    data['dt'] = pd.to_datetime(data['t'],unit='s')
    #data['dt'] = data['dt'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d %H:%M'))
    data.sort_values('dt', ascending=True, inplace=True)
    data.set_index(data.dt,inplace=True)
    data.drop(['t','dt'],inplace=True,axis=1)

    data['v']    = data['v'].round(3)
    data         = data.iloc[:data.index.get_loc(str(stop))]
    #print(f'\nCoinbase Data Pull| start:{data.index[0]} stop:{data.index[-1]} interval:{interval} diff:{len(data)} terms:{terms}\n')

    return data.drop_duplicates()