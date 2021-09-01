import pandas as pd
import cbpro
import math as mt
from datetime import timedelta
from datetime import datetime as dt

def get_data_span(asset,start,stop,interval,mode):

    pc      = cbpro.PublicClient()
    data    = pd.DataFrame()

    intervals = {'1day':86400,
                '6hours':21600,
                '1hour':3600,
                '15minutes':900,
                '5minutes':300,
                '1minute':60}

    print(f'\nStarting {mode} run for date range {start} - {stop} @interval={interval}\n')

    diff = stop - start
    d_s = diff.total_seconds()
    terms = mt.ceil((d_s/intervals[interval])/300)


    for term in range(terms):
        if interval == '1minute':
            strt = timedelta(minutes=300*term)
            end = timedelta(minutes=300*(term+1))
        elif interval == '5minutes':
            strt = timedelta(minutes=300*term*5)
            end = timedelta(minutes=300*(term+1)*5)
        elif interval == '1hour':
            strt = timedelta(hours=300*term)
            end = timedelta(hours=300*(term+1))
        elif interval == '1day':
            strt = timedelta(hours=300*term*24)
            end = timedelta(hours=300*(term+1)*24)
        elif interval == '15minutes':
            strt = timedelta(minutes=300*term*15)
            end = timedelta(minutes=300*(term+1)*15)
        else:
            break

        raw_data = pc.get_product_historic_rates(asset
                                                ,start + strt
                                                ,start + end
                                                ,intervals[interval]
                                                )

        data = data.append(raw_data)

    columnlist = {0:'t',1:'l',2:'h',3:'o',4:'c',5:'v'}
    data.rename(columns = columnlist,inplace=True)

    data['dt'] = pd.to_datetime(data['t'],unit='s')
    data.sort_values('dt', ascending=True, inplace=True)
    data.set_index(data.dt,inplace=True)
    data.drop(['t','dt'],inplace=True,axis=1)

    data['v']    = data['v'].round(3)
    data         = data.iloc[:data.index.get_loc(stop),:]
    
    print(f'\nData acquired. Total data points:{data.shape[0]}')
    return data.drop_duplicates()