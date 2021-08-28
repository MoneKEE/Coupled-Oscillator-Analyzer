import progressbar
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.metrics import mean_squared_error

def get_mean_squared_error(data,comp,col):

    h_list = ['ifft_'+str(x) for x in comp]
    ifft_sum = data[h_list].sum(axis=1)
    signal = data[col]
    val = mean_squared_error(signal,ifft_sum)

    return val
def iseven(num):
    if (int(num) % 2) == 0:
        return 1
    else:
        return 0

def progress_bar(x,load_text):
      
    widgets = [f'{load_text}: ', progressbar.AnimatedMarker(),' [',
         progressbar.Timer(),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    bar = progressbar.ProgressBar(widgets=widgets,maxval=x).start()
      
    return bar

def normalizedf(data):
    #Normalize the values
    for col in data.columns:
        data[col] = data[col]/np.max(np.abs(data[col]))
        data[col] = round(data[col],3)
    return data

def serial_todt(x,format='%y/%m/%d %H'):
    for i in range(len(x)):
        x[i] = pd.to_datetime(x[i],format)
    return x
