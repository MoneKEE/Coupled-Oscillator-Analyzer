'''
testing

'''
import cbpro
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
from sklearn.metrics import mean_squared_error as mse
from cmath import phase
import matplotlib.pyplot as plt
import matplotlib.dates as mpdates
import math as mt
pd.plotting.register_matplotlib_converters()
#import talib

def plot_fft(data,comp,fft_col,p_col='close',Fs=1):
    mask = data.fft_freq > 0

    with plt.style.context(style='ggplot'):
        plt.figure(1)

        dims = (2,3)
        r=1
        c=1
        ax1 = plt.subplot2grid(dims,(0,0),rowspan=r,colspan=c)
        ax2 = plt.subplot2grid(dims,(0,1),rowspan=r,colspan=c,polar=True)
        ax3 = plt.subplot2grid(dims,(0,2),rowspan=r,colspan=c,sharex=ax1)
        ax4 = plt.subplot2grid(dims,(1,0),rowspan=r,colspan=3)

        #ax1.magnitude_spectrum(data[fft_col],Fs=Fs)
        ax1.stem(data.fft_freq[mask],np.abs(data['fft'][mask])/max(np.abs(data['fft'][mask])),linefmt='C0',markerfmt='',basefmt='C0')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Amplitude')
        ax1.title.set_text(f'magnitude @ Fs:{Fs} Size:{len(data)}')

        ax2.plot(data.fft_ang[mask],(np.abs(data.fft[mask])/max(np.abs(data.fft[mask]))))
        ax2.title.set_text(f'angle @ Fs:{Fs} Size:{len(data)}')

        ax3.phase_spectrum(data[fft_col],Fs=Fs)
        ax3.title.set_text(f'phase @ Fs:{Fs} Size:{len(data)}')

        ax4.plot(data.index,0.2*data[fft_col],linewidth=1, linestyle='dotted',label=f'{fft_col}')
        for num_ in comp:
            ax4.plot(data.index,data[f'ifft_{num_}'],linewidth=2, label=f'ifft_{num_}')
        ax4.legend(loc='lower right')
        plt.draw()

def plot_time_series(data):
    with plt.style.context(style='ggplot'):
        plt.figure(2)

        dims = (27,7)
        r = 4
        c = 5
        ax1 = plt.subplot2grid(dims,(0,0),rowspan=r,colspan=c)
        ax2 = plt.subplot2grid(dims,(6,0),rowspan=r,colspan=c,sharex=ax1)
        ax3 = plt.subplot2grid(dims,(12,0),rowspan=r,colspan=c,sharex=ax1)
        ax4 = plt.subplot2grid(dims,(18,0),rowspan=r,colspan=c,sharex=ax1)
        ax5 = plt.subplot2grid(dims,(24,0),rowspan=r,colspan=c,sharex=ax1)

        r = 4
        c = 3
        ax6 = plt.subplot2grid(dims,(0,5),rowspan=r,colspan=c)
        ax7 = plt.subplot2grid(dims,(6,5),rowspan=r,colspan=c)
        ax8 = plt.subplot2grid(dims,(12,5),rowspan=r,colspan=c)
        ax9 = plt.subplot2grid(dims,(18,5),rowspan=r,colspan=c)
        ax10 = plt.subplot2grid(dims,(24,5),rowspan=r,colspan=c)

        ax1.plot(data.index,data['close'])
        ax1.title.set_text(f'Close')

        ax2.plot(data.index,data['close_1d'])
        ax2.title.set_text(f'Close 1st Difference')
        ax3.plot(data.index,data['close_1d2'])
        ax3.title.set_text(f'Close 2nd Difference')
        ax4.plot(data.index,data['close_1d3'])
        ax4.title.set_text(f'Close 3rd Difference')
        ax5.bar(data.index,data['volume'])
        ax5.title.set_text(f'Volume')

        ax6.boxplot(data.close,vert=False)
        ax7.boxplot(data.close_1d,vert=False)
        ax8.boxplot(data.close_1d2,vert=False)
        ax9.boxplot(data.close_1d3,vert=False)
        ax10.boxplot(data.volume,vert=False)
        plt.draw()

def plot_positions(data,comp,buy_signals,sell_signals,fft_col='close_1d'):
    with plt.style.context(style='ggplot'):
        plt.figure(3)

        ax1 = plt.subplot2grid((20,1),(0,0),rowspan=5,colspan=1)
        ax2 = plt.subplot2grid((20,1),(7,0),rowspan=3,colspan=1,sharex=ax1)
        ax3 = plt.subplot2grid((20,1),(12,0),rowspan=5,colspan=1,sharex=ax1)
        ax4 = plt.subplot2grid((20,1),(18,0),rowspan=3,colspan=1,sharex=ax1)
        
        ax1.plot(data.index,data.close,linewidth=2,label='close')
        ax1.title.set_text('close')

        ax2.plot(data.index,data.id_position,drawstyle='steps', linewidth=1)
        ax2.plot(data.index,data.position,drawstyle='steps',linestyle='dotted',linewidth=3)
        ax2.title.set_text('positions')

        ax3.plot(data.index,data.close_1d,linewidth = 1.5, label=fft_col, linestyle='dotted')
        for num_ in comp:
            ax3.plot(data.index,data[f'ifft_{num_}'],linewidth=2, label=f'ifft_{num_}')
        ax3.title.set_text('Harmonic triplet')
        ax3.legend(loc='lower right')

        ax4.bar(data.index, data.volume, linewidth = 2, label = 'volume')
        plt.draw()

def get_data_span(asset,start,stop,interval):
    pc      = cbpro.PublicClient()
    data    = pd.DataFrame()

    intervals = {'days':86400,
                'hours':3600,
                'minutes':60}

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

    columnlist = {0:'time',1:'low',2:'high',3:'open',4:'close',5:'volume'}
    data.rename(columns = columnlist,inplace=True)

    data['datetime'] = pd.to_datetime(data['time'],unit='s')
    #data['datetime'].dt.strftime("%y-%m-%d %H:%M:%S")

    data.sort_values('datetime', ascending=True, inplace=True)
    data.set_index(data.datetime,inplace=True)
    data.drop(['time','datetime'],inplace=True,axis=1)

    data.volume     = data['volume'].round(3)
    data            = data.loc[:stop]
    #print(f'\nCoinbase Data Pull| start:{data.index[0]} stop:{data.index[-1]} interval:{interval} diff:{len(data)} terms:{terms}\n')

    return data.drop_duplicates()

def fourier(data,fft_col='close_1d',n_harms=10,Fs=1):
    print(f'- Deriving Fourier Transform...\n')

    close_fft       = np.fft.fft(np.asarray(data[fft_col].tolist()))
    data['fft']     = close_fft
    data['fft_ang'] = np.angle(close_fft)
    
    fft_freq = np.fft.fftfreq(len(data),Fs)

    data['fft_freq'] = fft_freq

    return data

def filter_freqs(data,harms):
    #Return a single freq from each of the three bands
    #f_h,f_m and f_l.
    f_list  = []
    mag     = np.abs(data['fft']).reset_index().copy()
    mag.drop(columns=['datetime'],inplace=True)

    # Determines the segment length for each frequency band.
    len_f       = len(data.fft_freq[freq>0])
    bandwidth   = mt.floor(len_f/harms)

    f = data.fft_freq.copy()

    for i in range(harms):
        mag_pseg    = mag[bandwidth*i:bandwidth*(i+1)]
        mag_nseg    = mag.iloc[::-1][bandwidth*i:bandwidth*(i+1)]
        mag_all     = mag_pseg.append(mag_nseg)
        mag_max     = mag_all.fft.values.argmax()
        mag_min     = mag_all.fft.values.argmin()

        # Return the first freq in each band. Since the series are sorted in
        # descending order, the returned index corresponds to the frequence 
        # of greatest magnitude.d

        f_list.append([mag_max,mag_all.fft.iloc[mag_max]])

    return f_list

def get_harms(data,f_list):
    print('- Converting frequency to periods...\n')

    comp = []
    for x in f_list:
        test = mt.ceil(len(data)/x[0])
        if test not in comp:
            comp.append(test)
    comp.sort()

    return comp

def get_ifft(data,comp):
    print('- Determining component sinusoids and extracting desired harmonics...\n')
    fft_list = np.asarray(data.fft.tolist())

    _num_ = 0
    for num_ in comp:
        bnd                     = num_
        fft_listm10             = np.copy(fft_list)
        fft_listm10[bnd:-bnd]   = 0
        data['ifft_'+str(bnd)]  =np.real(np.fft.ifft(fft_listm10))
        _num_                   = bnd

    return data

def returns(data):
    # Hold returns are calculated by finding the value of the log of the ratio between the current close and the previous
    # close.
    # Strategy returns are found by 

    print('- Calculating returns...\n')

    data['hold']        = np.log(data.close/data.close.shift(1))
    data['ideal']       = data.id_position * data.hold
    data['strategy']    = data.position * data.hold

    results = np.exp(data[['hold','strategy','ideal']].sum())-1
    #n_days = (data.index[-1] - data.index[0]).days
    #returns_ann = 365/n_days * returns

    data.fillna(0,inplace=True)

    print_results(data=results)

    return data, results

def positions(data,signal,cross,fft_col,bt=False):
    print('- Determining ideal and strategy positions...\n')

    try:
        if bt:
            data.iloc[-1,data.columns.get_loc('position')] = np.where(data.iloc[-1,data.columns.get_loc(f'ifft_{str(int(signal))}')] > data.iloc[-1,data.columns.get_loc(f'ifft_{str(int(cross))}')], 1, 0)
        else:
            data['position'] = np.where(data[f'ifft_{str(int(signal))}'] > data[f'ifft_{str(int(cross))}'], 1, 0)
    except:
        data['position'] = 0

    #STRATEGY
    data['id_position'] = np.where(data[fft_col] > 0,1,0)
    
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

def benchmark(data,comp,fft_col='close_1d'):
    t_results = []
    for i in comp:
        for j in comp:
            data, buy_signals, sell_signals, id_buy_signals, id_sell_signals = positions(data=data,signal=i,cross=j,fft_col=fft_col)
            data,results = returns(data)
            if i != j:
                t_results.append([i
                                ,j
                                ,results['strategy']
                                ,results['hold']
                                ,results['ideal']])

    final_res = pd.DataFrame(t_results,columns=['signal','cross','strategy','hold','ideal'])

    return final_res

def optimize(data,fft_col,results):
    idx = results['strategy'].values.argmax()
    s_res = results.iloc[idx,:3]
    h_res = results['hold'][0]
    i_res = results['ideal'][0]

    print(f'- Optimizing...\n')

    data, buy_signals, sell_signals, id_buy_signals, id_sell_signals = positions(data=data,signal=s_res.signal,cross=s_res.cross,fft_col=fft_col)

    results['rank'] = results.strategy.rank(ascending=False)
    results.sort_values('rank',ascending=True,inplace=True)
    results.reset_index(drop=True,inplace=True)
    results.drop(columns='rank',inplace=True)

    same    = len(data[(data.id_position == data.position)])
    diff    = len(data[(data.id_position != data.position)])
    acc     = round(100*same/(same + diff),2)

    print(f'Returns Optimized Strategy| ifft_{str(int(s_res.signal))} > ifft_{str(int(s_res.cross))}')
    print(f'returns| total: {round(s_res.strategy*100,2)}%\thodl: {round(h_res*100,2)}%\tideal: {round(i_res*100,2)}%')
    print(f'position accuracy| {acc}%\n')

    buys        = data[buy_signals]
    sells       = data[sell_signals]
    id_buys     = data[id_buy_signals]
    id_sells    = data[id_sell_signals]

    print(f'Spreads\nstrategy -> buys:{len(buys[buys.position==1])} sells:{len(sells[sells.position==0])}')
    print(f'ideal -> buys:{len(id_buys[id_buys.id_position==1])} sells:{len(id_sells[id_sells.id_position==0])}\n')

    return data,buy_signals, sell_signals, id_buy_signals, id_sell_signals, results

def process_data_for_labels_past(data,hm_days=1):
    print('- Processing data for past features...\n')

    for i in range(1,hm_days+1):
        data['close_{}d'.format(i)] = (data['close'] - data.close.shift(i))
        data['close_{}d2'.format(i)] = (data['close_{}d'.format(i)] - data['close_{}d'.format(i)].shift(i))
        data['close_{}d3'.format(i)] = (data['close_{}d2'.format(i)] - data['close_{}d2'.format(i)].shift(i))

    data.fillna(0,inplace=True)

    print('Field Stats')
    print('-'*30)
    print(data.describe(),'\n\n')

    return data

def resume_run(inc):
    plt.pause(inc)
    plt.clf()

def mean_squared_error(data,comp,fft_col='close_1d'):
    h_list = ['ifft_'+str(x) for x in comp]
    ifft_sum = data[h_list].sum(axis=1)
    signal = data[fft_col]
    val = mse(signal,ifft_sum)

    return val

def print_results(data):
    result = pd.DataFrame([round(data.strategy*100,2),round(data.ideal*100,2),round(data.hold*100,2)]
                            ,index=['Strategy','Ideal','Hold']
                            ,columns=['Return %']
                        )

    print('\n','Projected Returns')
    print('-'*30)
    print(result)
    
def backtest(data,comp,hm_days,fft_col,harms,Fs,refresh,bt):
    # At the start it is assumed that 
    # N days have already been processed
    N = 24*7
    data_inc = data[:N].copy()

    for t in range(N,len(data)):
        data_inc = data_inc.append(data[t:(t+1)])

        try:
            print('='*80)
            print(f'- Processing Frame #{t}: start:{min(data_inc.index)} end:{max(data_inc.index)}\n')
        except:
            break

        data_inc  = process_data_for_labels_past(data=data_inc,hm_days=hm_days)

        data_inc              = fourier(data=data_inc,fft_col=fft_col,n_harms=harms,Fs=Fs)
        data_inc              = get_ifft(data=data_inc,comp=comp)

        data_inc, buy_signals, sell_signals, id_buy_signals, id_sell_signals = positions(data=data_inc,signal=comp[-1],cross=comp[0],fft_col=fft_col,bt=bt)
        data_inc, results = returns(data=data_inc)

        buys        = data_inc[buy_signals]
        sells       = data_inc[sell_signals]
        id_buys     = data_inc[id_buy_signals]
        id_sells    = data_inc[id_sell_signals]

        plot_time_series(data_inc)
        plot_fft(data_inc,Fs=Fs,comp=comp,fft_col=fft_col)
        plot_positions(data_inc,comp=comp,buy_signals=buys,sell_signals=sells,fft_col=fft_col)

        if t == len(data_inc):
            plt.show()
        else:
            resume_run(refresh)

    return data_inc
    
def dump(data,comp,hm_days,fft_col,harms,Fs):
    data_d = process_data_for_labels_past(data=data,hm_days=hm_days)

    data_d           = fourier(data=data_d,fft_col=fft_col,n_harms=harms,Fs=Fs)
    # freq_filt       = filter_freqs(data=data_d,freq=fft_freq,harms=harms)
    # freq_2_period   = get_harms(data=data_d,f_list=freq_filt)
    data_d           = get_ifft(data=data_d,comp=comp)

    data_d, buy_signals, sell_signals, id_buy_signals, id_sell_signals = positions(data=data_d,signal=comp[-1],cross=comp[0],fft_col=fft_col)
    data_d, results = returns(data=data_d)

    buys        = data_d[buy_signals]
    sells       = data_d[sell_signals]
    id_buys     = data_d[id_buy_signals]
    id_sells    = data_d[id_sell_signals]

    plot_positions(data=data_d,comp=comp,buy_signals=buys,sell_signals=sells,fft_col=fft_col)
    plot_fft(data=data_d,Fs=Fs,comp=comp,fft_col=fft_col)
    plot_time_series(data=data_d)
    plt.show()

    return data_d

##############################################
# MAIN PROCEDURE
##############################################
def main():

    #################
    # PARAMETERS
    #################
    harms       = 3
    sr          = 1
    Fs          = round(1/sr,2)
    hm_days     = 1
    asset       = 'ETH-USD'
    fft_col     = 'close_1d'
    interval    = 'hours'
    start       = dt(2019,1,1,0,0,0); stop = dt(2019,1,31,23,59,59)
    refresh     = 0.07
    bt          = True
    mode        = 'backtest' if bt else 'dump'
    

    #################
    # HARMONICS
    #################
    fibo    = [1,1]
    alpha   = 1

    _124_   = [1,2,4]
    _135_   = [1,3,5]
    _369_   = [3,6,9]
    _obv_   = [1,2,5]

    fibo    = [fibo.append(fibo[k-1]+fibo[k-2]) for k in range(2,10)]
    prim    = [x for x in range(2, 30) if all(x % y != 0 for y in range(2, x))]
    unt     = [x for x in range(1,11)]
    sqr     = [x**2 for x in range(1,11)]
    cub     = [x**3 for x in range(1,11)]
    h_rng   = [[x,y,z] for x in range(1,10) for y in range(1,10) for z in range(1,10) if x < y if y < z]
    
    comp    = [alpha*x for x in prim[:harms+1]]
    
    ##################
    # GO
    #################
    df_master  = get_data_span(asset=asset,start=start,stop=stop,interval=interval) 

    params = pd.DataFrame([mode,harms,fft_col,Fs,asset,interval,len(df_master)]
                            ,index=['Mode','n Harmonics','FFT Col','Fs','Asset','Interval','n Points']
                            ,columns=['Parameters']
                        )

    print('='*80)
    print(f'STARTING NEW RUN FOR DATA RANGE FROM:{start} TO:{stop}','\n')
    print(params)
    print('='*80,'\n')

    if bt:
        df = backtest(data=df_master,comp=comp,hm_days=hm_days,fft_col=fft_col,harms=harms,Fs=Fs,refresh=refresh,bt=bt)
    else:
        df = dump(data=df_master,comp=comp,hm_days=hm_days,fft_col=fft_col,harms=harms,Fs=Fs)

    print(df)

if __name__ == '__main__':
    main()