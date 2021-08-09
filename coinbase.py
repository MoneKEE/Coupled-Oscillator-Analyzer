'''
EM Wave Theory Inspired Price Action Model
The Mission: Treat an asset price time series as if it were a electro magnetic wave system.
The Method: Model the system based on the 1st, 2nd or 3rd difference of the price P(t) and volume V(t)
where dP(t) = P(t) - P(t-n) and dV(t) = V(t) - V(t-n)

'''
import cbpro
from numpy.lib.function_base import diff
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
from sklearn.metrics import mean_squared_error as mse
from scipy.signal import find_peaks
from cmath import phase
from contextlib import suppress
import matplotlib.pyplot as plt
import matplotlib.dates as mpdates
import math as mt
import inspect
import progressbar
import time

from sklearn.metrics.regression import mean_squared_error
pd.plotting.register_matplotlib_converters()
#import talib

def plot_fft(data,comp,psig_col,p_col='c',Fs=1):

    mask_1 = data.fft_freq > 0
    mask_2 = np.abs(data[f'{psig_col}_fft'][mask_1]) > np.std(np.abs(data[f'{psig_col}_fft'][mask_1])) * 4

    with plt.style.context(style='ggplot'):
        plt.figure(1)

        dims = (2,3)
        r=1
        c=1
        ax1 = plt.subplot2grid(dims,(0,0),rowspan=r,colspan=c)
        ax2 = plt.subplot2grid(dims,(0,1),rowspan=r,colspan=c,polar=True)
        ax3 = plt.subplot2grid(dims,(0,2),rowspan=r,colspan=c,sharex=ax1)
        ax4 = plt.subplot2grid(dims,(1,0),rowspan=r,colspan=3)

        ax1.stem(data.fft_freq[mask_1],np.where(mask_2==False,np.abs(data[f'{psig_col}_fft'][mask_1])/max(np.abs(data[f'{psig_col}_fft'][mask_1])),0),linefmt='C0', markerfmt='C0',basefmt='C0')
        ax1.stem(data.fft_freq[mask_1],np.where(mask_2,np.abs(data[f'{psig_col}_fft'][mask_1])/max(np.abs(data[f'{psig_col}_fft'][mask_1])),0),linefmt='C1',markerfmt='C1',basefmt='C1')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Amplitude')
        ax1.title.set_text(f'magnitude @ Fs:{Fs} Size:{len(data)}')

        ax2.plot(data[f'{psig_col}_ang'],(np.abs(data[f'{psig_col}_fft'])/max(np.abs(data[f'{psig_col}_fft']))))
        ax2.title.set_text(f'angle @ Fs:{Fs} Size:{len(data)}')

        ax3.phase_spectrum(data[psig_col]/np.max(data[psig_col]),Fs=Fs)
        ax3.title.set_text(f'phase @ Fs:{Fs} Size:{len(data)}')

        ax4.plot(data.index,0.2*data[psig_col],linewidth=1, linestyle='dotted',label=f'{psig_col}')
        for num_ in comp:
            ax4.plot(data.index,data[f'ifft_{num_}'],linewidth=2, label=f'ifft_{num_}')
        ax4.legend(loc='lower right')
        ax4.set_ylabel('amplitude - c_1d')
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

        ax1.plot(data.index,data['c_1d1'])
        ax1.title.set_text('dc')

        # ax2.annotate(f'Hi There', xy=(3, 1),  xycoords='data',
        #     xytext=(0.8, 0.95), textcoords='axes fraction',
        #     arrowprops=dict(facecolor='black', shrink=0.05),
        #     horizontalalignment='right', verticalalignment='top',
        #     )

        ax2.plot(data.index,data['c_1d1_avgwnd7'])
        ax2.title.set_text(f'c avg 7')
        ax3.plot(data.index,data['v_1d1'])
        ax3.title.set_text(f'dv')
        ax4.plot(data.index,data['v_1d1_avgwnd7'])
        ax4.title.set_text(f'v avg 7')
        ax5.plot(data.index,data['sys_pwr'])
        ax5.title.set_text(f'power')

        ax6.boxplot(data.c,vert=False)
        ax7.boxplot(data.c_1d1,vert=False)
        ax8.boxplot(data.v_1d1,vert=False)
        ax9.boxplot(data.v_1d1_avgwnd7,vert=False)
        ax10.boxplot(data.sys_pwr,vert=False)
        plt.draw()

def plot_positions(data,comp,buy_signals,sell_signals,psig_col='c_1d1'):

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

        ax3.plot(data.index,data.c_1d1,linewidth = 1.5, label=psig_col, linestyle='dotted')
        for num_ in comp:
            ax3.plot(data.index,data[f'ifft_{num_}'],linewidth=2, label=f'ifft_{num_}')
        ax3.title.set_text('Harmonic triplet')
        ax3.legend(loc='lower right')

        ax4.bar(data.index, data.v, linewidth = 2, label = 'volume')
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

    columnlist = {0:'t',1:'l',2:'h',3:'o',4:'c',5:'v'}
    data.rename(columns = columnlist,inplace=True)

    data['dt'] = pd.to_datetime(data['t'],unit='s')
    #data['datetime'].dt.strftime("%y-%m-%d %H:%M:%S")

    data.sort_values('dt', ascending=True, inplace=True)
    data.set_index(data.dt,inplace=True)
    data.drop(['t','dt'],inplace=True,axis=1)

    data.volume     = data['v'].round(3)
    data            = data.loc[:stop]
    #print(f'\nCoinbase Data Pull| start:{data.index[0]} stop:{data.index[-1]} interval:{interval} diff:{len(data)} terms:{terms}\n')

    return data.drop_duplicates()

def get_fft(data,Fs,col):

    print(f'- Performing Fourier Transform for {col}..\n')

    c_fft       = np.fft.fft(np.asarray(data[col].tolist()))
    data[f'{col}_fft']     = c_fft
    data[f'{col}_ang'] = np.angle(c_fft)
    
    fft_freq = np.fft.fftfreq(len(data),d=1/Fs)

    data['fft_freq'] = fft_freq

    data.fillna(0,inplace=True)

    return data

def get_harms(data,f_list):

    print('- Finding Harmonics...\n')

    comp = []
    for x in f_list:
        test = mt.ceil(len(data)/x[0])
        if test not in comp:
            comp.append(test)
    comp.sort()

    return comp

def get_ifft(data,comp,col):

    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Skip printing this message if called by harmonic sweeper
    if calframe[1][3] != 'harmonic_sweep':
        print('- Determining component sinusoids and extracting desired harmonics...\n')
    else:
        pass

    fft_list = np.asarray(data[col].tolist())

    data['ifft'] = np.real(np.fft.ifft(np.copy(fft_list)))

    _num_ = 0
    for num_ in comp:
        bnd                     = num_
        fft_listm10             = np.copy(fft_list)
        fft_listm10[bnd:-bnd]   = 0
        data['ifft_'+str(bnd)]  =np.real(np.fft.ifft(fft_listm10))
        _num_                   = bnd

    data.fillna(0,inplace=True)

    return data

def get_fftpeaks(data,col='sys_psd'):
    data_fp = data.copy()
    pos = data['fft_freq'] > 0

    return find_peaks(10 * np.log10(data[col][pos]))

def positions(data,signal,cross,psig_col,bt=False):

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
    data['id_position'] = np.where(data[psig_col] > 0,1,0)
    
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

def resume_run(inc):

    plt.pause(inc)
    plt.clf()

def get_mean_squared_error(data,comp,col):

    h_list = ['ifft_'+str(x) for x in comp]
    ifft_sum = data[h_list].sum(axis=1)
    signal = data[col]
    val = mean_squared_error(signal,ifft_sum)

    return val

def print_results(data):

    result = pd.DataFrame([round(data.strategy*100,2),round(data.ideal*100,2),round(data.hold*100,2)]
                            ,index=['Strategy','Ideal','Hold']
                            ,columns=['Return %']
                        )

    print('\n','vrojected Returns')
    print('-'*30)
    print(result,'\n')

def progress_bar(x,load_text):
      
    widgets = [f'{load_text}: ', progressbar.AnimatedMarker(),' [',
         progressbar.Timer(),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    bar = progressbar.ProgressBar(widgets=widgets,maxval=x).start()
      
    return bar

def process_data_for_labels_past(data,windows,psig_col,vsig_col,diff_offset=1,diff=1):
    # Create the model metrics
    # 1. Difference Signal ->dP(t) = P(t)-P(t-n)
    # 2. Polarity -> d(t) = (1,0,-1)
    # 3. Amplitude -> A(t) = abs(P(t)-P(t-n))
    # 4. Spread -> S(t) = H(t) - L(t) for the window
    # 5. Mean, Median, Std, Sum for the window
    # 6. dsum -> Sum(d(t)) for the window
    # 7. H, L, O for the window

    data_p = data.copy()

    print('- Processing data for past features...\n')

    # Create the difference signal for both price and volume
    print(f'- Creating difference {diff} with {diff_offset} offset...\n')

    dp1 = (data_p['c'] - data_p['c'].shift(diff_offset))/np.abs(data_p['c'] - data_p['c'].shift(diff_offset)).max()
    dv1 = (data_p[f'v'] - data_p[f'v'].shift(diff_offset))/np.abs(data_p[f'v'] - data_p[f'v'].shift(diff_offset)).max()

    dp2 = dp1 - dp1.shift(diff_offset)
    dv2 = dv1 - dv1.shift(diff_offset)

    if diff == 1:
        data_p[psig_col] = dp1
        data_p[vsig_col] = dv1
    elif diff == 2:
        data_p[psig_col] = dp1 - dp1.shift(diff_offset)
        data_p[vsig_col] = dv1 - dv1.shift(diff_offset)
    else:
        data_p[psig_col] = dp2 - dp2.shift(diff_offset)
        data_p[vsig_col] = dv2 - dv2.shift(diff_offset)

    # Create the Amplitude and Polarity metrics
    data_p[f'{psig_col}_amp'] = np.abs(data_p[psig_col])
    data_p[f'{vsig_col}_amp'] = np.abs(data_p[vsig_col])

    data_p[f'{psig_col}_pol'] = np.where(-1 * data_p[psig_col] > 0,-1
                            ,np.where(-1 * data_p[psig_col] < 0,1,0) )
    
    data_p[f'{vsig_col}_pol'] = np.where(-1 * data_p[vsig_col] > 0,-1
                            ,np.where(-1 * data_p[vsig_col] < 0,1,0) )

    # Create power and resistance metrics
    data_p['sys_pwr'] = data_p[psig_col] * data_p[vsig_col]
    data_p[f'sys_psd'] = np.abs(data_p['sys_pwr'])**2
    data_p['sys_r'] = (data_p[f'{psig_col}_amp']/data_p[f'{vsig_col}_amp'])
    data_p.fillna(0,inplace=True)
    data_p['sys_c'] = data_p[f'{vsig_col}_amp']/data_p[f'{psig_col}_amp']
    data_p.fillna(0,inplace=True)
    if np.isinf(data_p.sys_c.max()):
        data_p.replace([np.inf, -np.inf], np.nan,inplace=True)
        data_p.fillna(data_p.sys_c.max(),inplace=True)

    data_p['sys_r'] = data_p['sys_r']/data_p['sys_r'].max()
    data_p['sys_c'] = data_p['sys_c']/data_p['sys_c'].max()
 
    # Create the Spread metrics
    if data_p.shape[1] == 15:
    # Initialize the columns
        data_p[f'{psig_col}_sprdwnd7']= None
        data_p[f'{vsig_col}_sprdwnd7']= None
        data_p[f'{psig_col}_shpwnd7']= None
        data_p[f'{vsig_col}_shpwnd7']= None
        data_p[f'{psig_col}_avgwnd7'] = None
        data_p[f'{psig_col}_medwnd7'] = None
        data_p[f'{psig_col}_modwnd7'] = None
        data_p[f'{psig_col}_stdwnd7'] = None
        data_p[f'{vsig_col}_avgwnd7'] = None
        data_p[f'{vsig_col}_medwnd7'] = None
        data_p[f'{vsig_col}_modwnd7'] = None
        data_p[f'{vsig_col}_stdwnd7'] = None

        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{psig_col}_sprdwnd7')] = data_p.iloc[-windows[1]:,data_p.columns.get_loc(psig_col)].max() - data_p.iloc[-windows[1]:,data_p.columns.get_loc(psig_col)].min()
        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{vsig_col}_sprdwnd7')] = data_p.iloc[-windows[1]:,data_p.columns.get_loc(vsig_col)].max() - data_p.iloc[-windows[1]:,data_p.columns.get_loc(vsig_col)].min()
    # Create the Shape Context metrics [1=L->H,0=H->L]
    try:
        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{psig_col}_shpwnd7')].iloc[-windows[1]:] = np.where(data_p.iloc[-windows[1]:,data_p.columns.get_loc(psig_col)].argmax()>data_p.iloc[-windows[1]:,data_p.columns.get_loc(psig_col)].argmin(),1,0)
        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{vsig_col}_shpwnd7')].iloc[-windows[1]:] = np.where(data_p.iloc[-windows[1]:,data_p.columns.get_loc(vsig_col)].argmax()>data_p.iloc[-windows[1]:,data_p.columns.get_loc(vsig_col)].argmin(),1,0)
    except:
        pass
    # Create Mean, Median and Standard Deviation metrics
    try:
        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{psig_col}_avgwnd7')] = data_p.iloc[-windows[1]:,data_p.columns.get_loc(psig_col)].mean()
        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{psig_col}_medwnd7')] = data_p.iloc[-windows[1]:,data_p.columns.get_loc(psig_col)].median()
        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{psig_col}_modwnd7')] = np.abs(data_p.iloc[-windows[1]:,data_p.columns.get_loc(psig_col)].mode()).max()
        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{psig_col}_stdwnd7')] = data_p.iloc[-windows[1]:,data_p.columns.get_loc(psig_col)].std()
    except:
        pass

    
    data_p[f'{psig_col}_avg'] = data_p[psig_col].mean()
    data_p[f'{psig_col}_med'] = data_p[psig_col].median()
    data_p[f'{psig_col}_mod'] = data_p[psig_col].mode()
    data_p[f'{psig_col}_std'] = data_p[psig_col].std()
    

    try:
        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{vsig_col}_avgwnd7')] = data_p.iloc[-windows[1]:,data_p.columns.get_loc(vsig_col)].mean()
        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{vsig_col}_medwnd7')] = data_p.iloc[-windows[1]:,data_p.columns.get_loc(vsig_col)].median()
        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{vsig_col}_modwnd7')] = np.abs(data_p.iloc[-windows[1]:,data_p.columns.get_loc(vsig_col)].mode()).max()
        data_p.iloc[-windows[1]:,data_p.columns.get_loc(f'{vsig_col}_stdwnd7')] = data_p.iloc[-windows[1]:,data_p.columns.get_loc(vsig_col)].std()
    except:
        pass
        
    data_p[f'{vsig_col}_avg'] = data_p[vsig_col].mean()
    data_p[f'{vsig_col}_med'] = data_p[vsig_col].median()
    data_p[f'{vsig_col}_mod'] = data_p[vsig_col].mode()
    data_p[f'{vsig_col}_std'] = data_p[vsig_col].std()

    data_p.fillna(0,inplace=True)

    print('Field Stats')
    print('-'*30)
    # print(data_p.describe(),'\n\n')
    
    try:
        data_p.drop(columns=['h','l','o'],inplace=True) 
    except:
        pass

    return data_p

def optimize(data,psig_col,results):
    pass
    return None

def harmonic_sweep(data,h_rng,Fs,diff_offset,psig_col='c_1d1'):

    results= []
    load_text = f'verforming {len(data)} runs'

    data_h = data.copy()
    data_h  = process_data_for_labels_past(data=data_h,diff_offset=diff_offset)
    data_h  = get_fft(data=data_h,psig_col=psig_col,Fs=Fs)

    bar = progress_bar(x=len(h_rng),load_text=load_text)

    for i,chord in enumerate(h_rng):
        bar.update(i)
        data_h  = get_ifft(data=data_h,comp=chord)
        chord_mse = mean_squared_error(data_h,chord,psig_col=psig_col)

        data_h, buy_signals, sell_signals, id_buy_signals, id_sell_signals = positions(data=data_h,signal=chord[-1],cross=chord[0],psig_col=psig_col)
        data_h, ret = returns(data=data_h)

        results.append([chord,chord_mse,ret.strategy])
        labels = ['ifft_' + str(x) for x in chord]
        data_h.drop(labels = labels,axis=1,inplace=True)
        
    bar.finish()
    return data_h, pd.DataFrame(results,columns=['chord','chord_mse','returns'])
    
def backtest(data,comp,psig_col,vsig_col,harms,Fs,windows,bt,refresh,diff_offset=1,diff=1):
    # At the start it is assumed that 
    # N days have already been processed
    N = 24*7
    data_inc = data[:N].copy()
    stats = pd.DataFrame()

    for t in range(N+1,len(data)):

        try:
            print('='*80)
            print(f'- Processing Frame #{t}: start:{min(data_inc.index)} end:{max(data_inc.index)}\n')
        except:
            break

        data_inc  = process_data_for_labels_past(data=data_inc,psig_col=psig_col,vsig_col=vsig_col,diff_offset=diff_offset,windows=windows)
        data_inc  = get_fft(data=data_inc,col=psig_col,Fs=Fs)
        data_inc  = get_ifft(data=data_inc,comp=comp,col=psig_col)

        # data_inc, buy_signals, sell_signals, id_buy_signals, id_sell_signals = positions(data=data_inc,signal=comp[-1],cross=comp[0],psig_col=psig_col,bt=bt)
        # data_inc, results = returns(data=data_inc)

        # #data_inc = calc_stats(data_inc,psig_col=psig_col,wnd=24)

        # buys        = data_inc[buy_signals]
        # sells       = data_inc[sell_signals]
        # id_buys     = data_inc[id_buy_signals]
        # id_sells    = data_inc[id_sell_signals]

        plot_time_series(data_inc)
        #plot_fft(data_inc,Fs=Fs,comp=comp,psig_col=psig_col)
        # plot_positions(data_inc,comp=comp,buy_signals=buys,sell_signals=sells,psig_col=psig_col)

        # data_inc.iloc[:,:10].plot(subplots=True)
        # data_inc.iloc[:,10:21].plot(subplots=True)
        # time.sleep(2)

        data_inc = data_inc.append(data[t:(t+1)])

        if t == len(data):
            plt.show()
        else:
           resume_run(refresh)

    return data_inc
    
def dump(data,comp,psig_col,vsig_col,harms,Fs,windows,diff_offset=1,diff=1): 

    stats = pd.DataFrame()
    data_d = process_data_for_labels_past(  data=data
                                            ,diff_offset=diff_offset
                                            ,psig_col=psig_col
                                            ,vsig_col=vsig_col
                                            ,diff=diff
                                            ,windows=windows
                                            )
    data_d  = get_fft(  data=data_d
                        ,col=psig_col
                        ,Fs=Fs
                        )
    data_d  = get_fft(  data=data_d
                        ,col=vsig_col
                        ,Fs=Fs
                        )
    ppeaks, pprops = get_fftpeaks(  data=data_d
                                    ,col='sys_psd'
                                    )
    vpeaks, vprops = get_fftpeaks(  data=data_d
                                    ,col='sys_psd'
                                    )
    # data_d  = get_ifft( data=data_d
    #                     ,comp=comp
    #                     )

    data_d.iloc[:,2:12].plot(subplots=True)
    pd.plotting.scatter_matrix(data_d[['c_1d1','sys_r','sys_c','v_1d1','sys_pwr']],grid=True,diagonal='hist')
    data_d.iloc[:,13:].plot(subplots=True)
    plt.show()

    return data_d

# MAIN PROCEDURE
def main():

    # PARAMETERS
    harms       = 3
    sr          = 0.3
    alpha       = 1
    Fs          = round(1/sr,2)
    diff_offset = 1
    diff        = 1
    refresh     = 0.07
    bt          = False
    windows     = [24,24*7,24*30]
    start       = dt(2019,1,1,0,0,0); stop = dt(2019,2,1,00,00,00)
    asset       = 'ETH-USD'
    psig_col    = f'c_{diff}d{diff_offset}'
    vsig_col    = f'v_{diff}d{diff_offset}'
    interval    = 'hours'
    mode        = 'backtest' if bt else 'dump'
   
    # HARMONICS
    prim        = [x for x in range(2, 30) if all(x % y != 0 for y in range(2, x))]
    harm_mlt    = [harms*x for x in range(1,11)]
    harm_exp    = [x**(harms+1) for x in range(1,11)]
    h_rng       = [[x,y,z] for x in range(2,10) for y in range(3,10) for z in range(4,10) if x < y if y < z if (y-x) >= 2 if (z-y) >= 4]
    
    comp        = [alpha*x for x in harm_exp[:harms]]
    
    # GO GO GO...
    df_master   = get_data_span( asset=asset
                                ,start=start
                                ,stop=stop
                                ,interval=interval
                                ) 
    params      = pd.DataFrame( [mode,harms,psig_col,vsig_col,Fs,asset,interval,len(df_master)]
                                ,index=['Mode','n Harmonics','psig_col','vsig_col','Fs','Asset','Interval','n Points']
                                ,columns=['parameters']
                                )

    print('='*80)
    print(  f'Starting {mode} run for date range {start} - {stop}'
            ,'\n'
            )
    print(params)
    print(  '='*80
            ,'\n'
            )

    # Determine the run mode ['backtest','dump','sweep']
    # Run backtest ['roll','expand']
    if bt:
        df = backtest(  data=df_master
                        ,comp=comp
                        ,diff_offset=diff_offset
                        ,diff=diff
                        ,psig_col=psig_col
                        ,vsig_col=vsig_col
                        ,harms=harms
                        ,Fs=Fs
                        ,refresh=refresh
                        ,bt=bt
                        ,windows=windows
                        )

        print('- Backtest Complete...')

    # Run dump
    elif (mode == 'dump') & (not bt):
        df = dump(  data=df_master
                    ,comp=comp
                    ,diff_offset=diff_offset
                    ,diff=diff
                    ,psig_col=psig_col
                    ,vsig_col=vsig_col
                    ,harms=harms
                    ,Fs=Fs
                    ,windows=windows
                    )

        print('- Dump Complete...')

    # Run sweep
    else:
        df,sweep_res = harmonic_sweep(  data=df_master
                                        ,psig_col=psig_col
                                        ,vsig_col=vsig_col
                                        ,Fs=Fs
                                        ,diff_offset=diff_offset
                                        ,h_rng=h_rng
                                        )

        print('- Sweep Complete...')

        print(  '\n'
                ,f'the winner:{sweep_res.iloc[sweep_res.chord_mse.argmin(),0]} with an MSE of {sweep_res.iloc[sweep_res.chord_mse.argmin(),1]}'
                ) 

    ppeaks, pprops = get_fftpeaks(data=df,col='sys_psd')
    vpeaks, vprops = get_fftpeaks(data=df,col='sys_psd')
    # Pause for interactive session   
    #pd.plotting.scatter_matrix(df.loc[:,[f'{vsig_col}',f'{vsig_col}_psd',f'{psig_col}_psd',f'{psig_col}','sig_power']])
    # df.loc[:,[f'{psig_col}',f'{vsig_col}_meanwnd7',f'{vsig_col}_modewnd7',f'{vsig_col}','sys_psd','pv_power']].plot(subplots=True)
    # plt.show() 
    pass


if __name__ == '__main__':
    main()