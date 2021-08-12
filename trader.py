import inspect
import numpy as np
import pandas as pd

def positions(data,signal,cross,obv=['dv1','dc1'],bt=False):

    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    # Skip printing this message if called by harmonic sweeper
    if calframe[1][3] != 'harmonic_sweep':
          print('- Determining ideal and strategy positions...\n')
    else:
        pass

    try:
        if bt:
            data.iloc[-1,data.columns.get_loc('position')] = np.where(data.iloc[-1,data.columns.get_loc(f'ifft_{str(int(signal))}')] > data.iloc[-1,data.columns.get_loc(f'ifft_{str(int(cross))}')], 1, 0)
        else:
            data['position'] = np.where(data[f'ifft_{str(int(signal))}'] > data[f'ifft_{str(int(cross))}'], 1, 0)
    except:
        data['position'] = 0

    #STRATEGY
    data['id_position'] = np.where(data[obv[1]] > 0,1,0)
    
    buy_signals     = ((data.position == 1) 
                        & (data.position.shift(1) == 0)) | ((data.position == 0) 
                        & (data.position.shift(-1) == 1)
                        )
    id_buy_signals  = ((data.id_position == 1) 
                        & (data.id_position.shift(1) == 0)) | ((data.id_position == 0) 
                        & (data.id_position.shift(-1) == 1)
                        )

    sell_signals    = ((data.position == 0) 
                        & (data.position.shift(1) == 1)) | ((data.position == 1) 
                        & (data.position.shift(-1) == 0)
                        )
    id_sell_signals = ((data.id_position == 0) 
                        & (data.id_position.shift(1) == 1)) | ((data.id_position == 1) 
                        & (data.id_position.shift(-1) == 0)
                        )

    #buy_marker = df.sma100 * buy_signals - (df.sma100.max()*0.3)
    #buy_marker = buy_marker[buy_signals]
    #buy_dates = df.index[buy_signals]

    #sell_marker = df.sma100 * sell_signals + (df.sma100.max()*0.3)
    #sell_marker = sell_marker[sell_signals]
    #sell_dates = df.index[sell_signals]

    data.fillna(0,inplace=True)

    return data, buy_signals, sell_signals, id_buy_signals, id_sell_signals

def print_results(data):

    result = pd.DataFrame([round(data.strategy*100,2),round(data.ideal*100,2),round(data.hold*100,2)]
                            ,index=['Strategy','Ideal','Hold']
                            ,columns=['Return %']
                        )

    print('\n','vrojected Returns')
    print('-'*30)
    print(result,'\n')

def returns(data):
    # Hold returns are calculated by finding the value of the log of the ratio between the current close and the previous
    # close.
    # Strategy returns are found by 
    # Skip printing this message if called by harmonic sweeper
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    if calframe[1][3] != 'harmonic_sweep':
          print('- Calculating returns...\n')
    else:
        pass

    data['hold']        = np.log(data.close/data.close.shift(1))
    data['ideal']       = data.id_position * data.hold
    data['strategy']    = data.position * data.hold

    results = np.exp(data[['hold','strategy','ideal']].sum())-1
    #n_days = (data.index[-1] - data.index[0]).days
    #returns_ann = 365/n_days * returns

    data.fillna(0,inplace=True)

    if calframe[1][3] != 'harmonic_sweep':
          print_results(data=results)
    else:
        pass

    return data, results
