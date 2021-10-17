import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import misc

def proba(data):
    data_p = data.copy()
    q = 0.69
    sys = data_p.iloc[:,data_p.columns.get_loc('Pe'):]
    sig1 = data_p.iloc[:,:data_p.columns.get_loc('x2')].join(data_p.pos)
    sig2 = data_p.iloc[:,data_p.columns.get_loc('x2'):data_p.columns.get_loc('Pe')].join(data_p.pos)

    cols = sys.corrwith(sys.pos.shift(-1))[np.abs(sys.corrwith(sys.pos.shift(-1)))>=sys.corrwith(sys.pos.shift(-1)).quantile(q)].sort_values(ascending=False).index.values
    
    for col in cols:

        if col not in ['pos','posa','w1o','w2o']:
            sys[f'{col}_gt0'] = np.where(sys[col]>0,1,0)
            sys[f'{col}_lt0'] = np.where(sys[col]<=0,1,0)

            sys[f'{col}_gta'] = np.where(sys[col]>sys[col].mean(),1,0)
            sys[f'{col}_lta'] = np.where(sys[col]<=sys[col].mean(),1,0)

            sys[f'{col}_gtq1'] = np.where(sys[col]>sys[col].quantile(0.25),1,0)
            sys[f'{col}_ltq1'] = np.where(sys[col]<=sys[col].quantile(0.25),1,0)

            sys[f'{col}_gtq2'] = np.where(sys[col]>sys[col].quantile(0.5),1,0)
            sys[f'{col}_ltq2'] = np.where(sys[col]<=sys[col].quantile(0.5),1,0)

            sys[f'{col}_gtq3'] = np.where(sys[col]>sys[col].quantile(0.75),1,0)
            sys[f'{col}_ltq3'] = np.where(sys[col]<=sys[col].quantile(0.75),1,0)

    return sig1.drop('pos',axis=1).join(sig2.drop('pos',axis=1)).join(sys)