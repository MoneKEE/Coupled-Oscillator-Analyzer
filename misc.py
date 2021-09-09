import progressbar
import numpy as np
import pandas as pd
# from sklearn import preprocessing

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
    data=data
    for col in data.columns:
        if col !='idposc':
            # data[col] = (data[col]-data[col].mean())/data[col].mean()
            data[col] = data[col].div(np.abs(data[col].max()),axis=0)

    # data_a = data.drop('idposc',axis=1)
    # Scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    # data_t = pd.DataFrame(Scaler.fit_transform(data_a),index=data_a.index,columns=data_a.columns)

    data_t=data.copy()

    for col in data_t.columns:
        data_t[col] = data_t[col].replace([np.inf, -np.inf], np.nan)
        data_t[col] = data_t[col].fillna(np.abs(data_t[col]).max())

    return data_t

def serial_todt(x,format='%y/%m/%d %H'):
    for i in range(len(x)):
        x[i] = pd.to_datetime(x[i],format)
    return x
