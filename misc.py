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

def normalizedf(data,type='max'):
    #Normalize the values
    for col in data.columns:
        if col not in ['Pxa','Ffa','Fta','Maa','Qfa','Dra','Wka','Rfa']:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            data[col] = data[col].fillna(np.abs(data[col]).max())
            if type == 'max':
                data[col] = data[col]/np.abs(data[col]).max()
            else:
                data[col] = (data[col]-data[col].mean())/data[col].std()

    return data

def serial_todt(x,format='%y/%m/%d %H'):
    for i in range(len(x)):
        x[i] = pd.to_datetime(x[i],format)
    return x
