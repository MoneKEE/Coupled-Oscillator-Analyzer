import numpy as np
import math as mt
import inspect
from scipy.signal import find_peaks

def harmonics(harms,alpha,type='harm_exp'):
# HARMONICS
    prim        = [x for x in range(2, 30) if all(x % y != 0 for y in range(2, x))]
    harm_mlt    = [harms*x for x in range(1,11)]
    harm_exp    = [x**(harms+1) for x in range(1,11)]
    h_rng       = [[x,y,z] for x in range(2,10) for y in range(3,10) for z in range(4,10) if x < y if y < z if (y-x) >= 2 if (z-y) >= 4]

    comp        = [alpha*x for x in locals()[type][:harms]]

    return comp

def get_fft(data,Fs,col):
    data_f = data.copy()
    print(f'- Performing Fourier Transform for {col}..\n')

    c_fft       = np.fft.fft(np.asarray(data_f[col].tolist()))
    data_f[f'{col}_fft']     = c_fft
    data_f[f'{col}_ang'] = np.angle(c_fft)
    
    fft_freq = np.fft.fftfreq(len(data_f),d=1/Fs)

    data_f['fft_freq'] = fft_freq

    data_f.fillna(0,inplace=True)

    return data_f

def get_harms(data,f_list):
    data_h = data.copy()
    print('- Finding Harmonics...\n')

    comp = []
    for x in f_list:
        test = mt.ceil(len(data_h)/x[0])
        if test not in comp:
            comp.append(test)
    comp.sort()

    return comp

def get_ifft(data,comp,col):
    data_i = data.copy()
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Skip printing this message if called by harmonic sweeper
    if calframe[1][3] != 'harmonic_sweep':
        print('- Determining component sinusoids and extracting desired harmonics...\n')
    else:
        pass

    fft_list = np.asarray(data_i[col].tolist())

    data_i[f'{col}_ifft'] = np.real(np.fft.ifft(np.copy(fft_list)))

    _num_ = 0
    for num_ in comp:
        bnd                     = num_
        fft_listm10             = np.copy(fft_list)
        fft_listm10[bnd:-bnd]   = 0
        data[f'{col}_ifft_'+str(bnd)]  =np.real(np.fft.ifft(fft_listm10))
        _num_                   = bnd

    data_i.fillna(0,inplace=True)

    return data_i

def get_fftpeaks(data,col='sys_psd'):
    data_p = data.copy()
    pos = data_p['fft_freq'] > 0

    return find_peaks(10 * np.log10(data_p[col][pos]))