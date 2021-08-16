import numpy as np
import math as mt
import matplotlib.pyplot as plt
import inspect
from scipy.signal import find_peaks

def fourier_analysis(comp, Fs, obv, data_s):
    data_f = data_s.copy()

    for col in obv:
        data_af  = get_angfreq(  data=data_f
                                ,col=col
                                ,Fs=Fs
                                        )
        data_r  = get_tfreq( data=data_af
                                ,col=col
                                ,comp=comp
                                )
        data_f = data_r
    return data_f

def harmonics(harms,alpha,type='harm_exp'):
# HARMONICS
    prim        = [x for x in range(2, 30) if all(x % y != 0 for y in range(2, x))]
    harm_mlt    = [harms*x for x in range(1,10)]
    harm_exp    = [x**(harms+1) for x in range(1,11)]
    h_rng       = [[x,y,z] for x in range(2,10) for y in range(3,10) for z in range(4,10) if x < y if y < z if (y-x) >= 2 if (z-y) >= 4]

    comp        = [alpha*x for x in locals()[type][:harms]]

    return comp

def get_angfreq(data,Fs,col):
    data_f = data.copy()
    print(f'- Performing Fourier Transform for {col}..\n')

    c_fft    = np.fft.fft(np.asarray(data_f[f'd{col}t_o'].tolist()))
    data_f[f'{col}_w']     = c_fft
    data_f[f'{col}_theta'] = np.angle(c_fft)
    
    fft_freq = np.fft.fftfreq(len(data_f),d=1/Fs)

    data_f['fft_freq'] = fft_freq

    data_f.fillna(0,inplace=True)

    return data_f

def get_freq2period(data,f_list):
    data_h = data.copy()
    print('- Finding Harmonics...\n')

    comp = []
    for x in f_list:
        test = mt.ceil(len(data_h)/x[0])
        if test not in comp:
            comp.append(test)
    comp.sort()

    return comp

def get_tfreq(data,comp,col):
    data_i = data.copy()
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    fft_list = np.asarray(data_i[f'd{col}t_o'].tolist())

    data_i[f'{col}_f'] = np.real(np.fft.ifft(np.copy(fft_list)))

    _num_ = 0
    for num_ in comp:
        bnd                     = num_
        fft_listm10             = np.copy(fft_list)
        fft_listm10[bnd:-bnd]   = 0
        data[f'{col}_f'+str(bnd)]  =np.real(np.fft.ifft(fft_listm10))
        _num_                   = bnd

    data_i.fillna(0,inplace=True)

    return data_i

def get_angfreq_peaks(data,col,oneside=True):
    data_p = data.copy()
    if not oneside:
        pos = data_p['fft_freq'] > 0
    else:
        pos = data_p['fft_freq'] != 0

    return find_peaks(10 * np.log10(data_p[f'{col}_f'][pos]))