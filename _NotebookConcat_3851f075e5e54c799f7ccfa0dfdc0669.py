for col in data_o.columns:
        if col not in ['idpos1']:
            data_o[col] = data_o[col].replace([np.inf, -np.inf], np.nan)
            data_o[col] = data_o[col].fillna(np.abs(data_o[col]).max())
            if type == 'max':
                data_o[col] = (data_o[col]-data_o[col].mean())/data_o[col].std()
                data_o[col] = data_o[col]/np.abs(data_o[col]).max()
            else:
                data_o[col] = (data_o[col]-data_o[col].mean())/data_o[col].std()