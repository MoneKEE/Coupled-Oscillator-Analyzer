# import progressbar
import numpy as np
import pandas as pd
# from sklearn import preprocessing

def iseven(num):
    if (int(num) % 2) == 0:
        return 1
    else:
        return 0

# def progress_bar(x,load_text):
      
#     widgets = [f'{load_text}: ', progressbar.AnimatedMarker(),' [',
#          progressbar.Timer(),
#          '] ',
#            progressbar.Bar('*'),' (',
#            progressbar.ETA(), ') ',
#           ]
#     bar = progressbar.ProgressBar(widgets=widgets,maxval=x).start()
      
#     return bar

def normalizedf(data,rtype='plot'):
    print('\t- Normalizing data...\n')
    datac = data.copy()

    #Normalize the values
    for col in datac.columns:
        # datac[col] = datac[col].replace([np.inf, -np.inf], np.nan)
        # datac[col] = datac[col].fillna(np.abs(datac[col]).max())
        if rtype == 'plot':
            if col not in ['quad_abs','x1pol','x2pol','w1o','w1','w2o','w2','pos','ddpos','dpos','Pxa','Pwa','Maa','Fta','Ffa']:
                datac[col] = datac[col]/np.max(np.abs(datac[col]))
        else:
            if col not in ['str','orc','bnh']:
                datac[col] = datac[col]/np.max(np.abs(datac[col]))
    return datac

def get_roots(inp):
    dcply = np.polynomial.Polynomial.fit(np.arange(0,len(inp)),inp.values,deg=2)
    
    return dcply