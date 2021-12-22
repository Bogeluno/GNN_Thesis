import pandas as pd
import numpy as np
import pickle

samples_per_day = pd.read_csv('data/processed/SimpleNNData.csv', index_col=0, parse_dates = [1]).time.dt.date.value_counts().sort_index()

def sample_function(sample_series):
    no_cs = []
    cs = []
    cum_sum = sample_series.cumsum().shift(1, fill_value = 0)
    for day in sample_series.index:
        perm = np.random.permutation(np.arange(sample_series.loc[day]))
        perm_cs = perm + cum_sum.loc[day]

        split5 = np.array_split(perm,5)
        split5_cs = np.array_split(perm_cs,5)

        if len(split5[3]) > len(split5[4]):
            np.append(split5[2],split5[3][-1])
            split5[3] = split5[3][:-1]

            np.append(split5_cs[2],split5_cs[3][-1])
            split5_cs[3] = split5_cs[3][:-1]


        no_cs.append([np.concatenate(split5[:3]), split5[3], split5[4]])
        cs.append([np.concatenate(split5_cs[:3]), split5_cs[3], split5_cs[4]])
    
    return no_cs, cs

nc, c = sample_function(samples_per_day)
cc = (np.concatenate([x[0] for x in c]),np.concatenate([x[1] for x in c]),np.concatenate([x[2] for x in c]))

with open('data/processed/Sample_NC', 'wb') as handle:
    pickle.dump(nc, handle, pickle.HIGHEST_PROTOCOL)

with open('data/processed/Sample_CC', 'wb') as handle:
    pickle.dump(cc, handle, pickle.HIGHEST_PROTOCOL)
