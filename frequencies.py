import numpy as np
from scipy.signal import periodogram as pe


def complex_coords(df,x):
    dfc = df.copy()
    for s,i in enumerate(x,start=1):
        dfc[f'x{s}fft'] = np.fft.fft(i)
        dfc[f'x{s}ifft'] = np.fft.ifft(dfc[f'x{s}fft'])

    return dfc


def get_tfreq(data,col,Fs):
    data_i = data.copy()

    fft_list = np.asarray(data_i[f'd{col}1t_o'].tolist())

    data_i[f'{col}f_t'] = np.fft.ifft(np.fft.fft(np.copy(fft_list)))
    _num_ = 0
    comp = []
    for num_ in comp:
        bnd                     = num_
        fft_listm10             = np.copy(fft_list)
        fft_listm10[bnd:-bnd]   = 0
        data_i[f'{col}f_t'+str(bnd)]  =np.real(np.fft.ifft(fft_listm10))
        _num_                   = bnd

    data_i.fillna(0,inplace=True)

    return data_i