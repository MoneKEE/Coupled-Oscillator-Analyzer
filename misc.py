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

def normalizedf(data,rtype='max'):
    datac = data.copy()

    #Normalize the values
    for col in datac.columns:
        if col not in ['quad_abs','x1pol','x2pol','w1o','w1','w2o','w2','pos','Pxa','Pwa','Maa','Fta','Ffa']:
            # datac[col] = datac[col].replace([np.inf, -np.inf], np.nan)
            # datac[col] = datac[col].fillna(np.abs(datac[col]).max())
            if rtype == 'max':
                datac[col] = datac[col]/np.max(datac[col].abs())
            else:
                datac[col] = (datac[col]-datac[col].mean())/datac[col].std()

    return datac

def get_roots(inp):
    dcply = np.polynomial.Polynomial.fit(np.arange(0,len(inp)),inp.values,deg=2)

    return dcply