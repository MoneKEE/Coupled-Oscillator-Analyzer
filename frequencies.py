import numpy as np
from scipy.signal import periodogram as pe

def fourier_analysis(Fs, obv, data_s):
    data_f = data_s.copy()

    for col in obv:
        data_af  = get_angfreq(  data=data_f
                                ,col=col
                                ,Fs=Fs
                                        )
        data_r,alpha  = get_tfreq( data=data_af
                                ,col=col
                                ,Fs=Fs
                                )
        data_f = data_r
    return data_r, alpha

def harmonics(alpha,harms=9,type='harm_mlt'):
# HARMONICS
    prim        = [x for x in range(2, 30) if all(x % y != 0 for y in range(2, x))]
    harm_mlt    = [x for x in range(1,10)]
    harm_exp    = [x**(harms+1) for x in range(1,11)]
    h_rng       = [[x,y,z] for x in range(2,10) for y in range(3,10) for z in range(4,10) if x < y if y < z if (y-x) >= 2 if (z-y) >= 4]

    comp        = [alpha*x for x in locals()[type][:harms]]

    return comp

def get_angfreq(data,Fs,col):
    data_f = data.copy()
    print(f'- Performing Fourier Transform for {col}..\n')

    data_f[f'{col}_fft']    = np.fft.fft(np.asarray(data_f[f'd{col}1t_o'].tolist()))
    
    fft_freq = np.fft.fftfreq(len(data_f),d=1/Fs)
    data_f['fft_freq'] = fft_freq

    data_f.fillna(0,inplace=True)

    return data_f

def get_tfreq(data,col,Fs):
    data_i = data.copy()

    fft_list = np.asarray(data_i[f'd{col}1t_o'].tolist())


    f,s = pe(data_i.dv1t_o,fs=Fs)
    alpha = int(np.ceil(f[s == s.max()][0]))
    data_i[f'{col}f_t'] = np.fft.ifft(np.fft.fft(np.copy(fft_list)))
    _num_ = 0
    comp = [alpha*x for x in range(1,10)]
    for num_ in comp:
        bnd                     = num_
        fft_listm10             = np.copy(fft_list)
        fft_listm10[bnd:-bnd]   = 0
        data_i[f'{col}f_t'+str(bnd)]  =np.real(np.fft.ifft(fft_listm10))
        _num_                   = bnd

    data_i.fillna(0,inplace=True)

    return data_i, alpha