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


'''
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from scipy import sparse
import torch
import torch.nn.functional as F
from datetime import date, timedelta
import torch.optim as optim
import torch.nn as nn
from torch_geometric.nn import Sequential, GCNConv , Linear, GraphNorm
from torch_geometric import utils, data
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from torch_scatter import scatter
import subprocess
import time

def make_PTG(graph, zones, Weather_Scale):
    attr, adj = graph

    # Filter out 
    if (attr.time_to_reservation.values[-1] >= 48) or ~attr.next_customer[-1]:
        return None
    
    if attr.leave_zone[-1] not in zones:
        return None

    # Slice
    _, labels = sparse.csgraph.connected_components(csgraph=adj, directed=False, return_labels=True)
    newl = labels[-1]
    indices = labels == newl   

    attr = attr[indices]
    adj = adj[indices,:].tocsc()[:,indices].tocsr()

    # drop
    attr.drop(columns=['park_location_lat', 'park_location_long', 'leave_location_lat', 'leave_location_long', 'park_fuel', 'park_zone', 'moved', 'movedTF', 'time', 'prev_customer', 'next_customer', 'action'], inplace = True)

    # One hot encoding
    attr['leave_zone'] = pd.Categorical(attr['leave_zone'], categories=zones)
    attr = pd.get_dummies(attr, columns = ['leave_zone'], prefix='lz')

    attr['engine']= pd.Categorical(attr['engine'], categories=['118I', 'I3', 'COOPER', 'X1'])
    attr = pd.get_dummies(attr, columns = ['engine'], prefix='eng')

    # Add degree
    attr['degree'] = np.squeeze(np.asarray(adj.sum(axis=1)))/50

    # Normalize fuel, weahter and dist 
    attr['leave_fuel'] = attr['leave_fuel']/100
    attr['dist_to_station'] = attr['dist_to_station']/5000
    attr[Weather_Scale.index] = (attr[Weather_Scale.index] - Weather_Scale['Min'])/Weather_Scale['diff']

    # Get edges
    edge_index, edge_weight = utils.convert.from_scipy_sparse_matrix(adj)

    # Make pytorch data type
    d = data.Data(x = torch.tensor(attr.drop(columns = ['time_to_reservation']).to_numpy(dtype = 'float')).float(), edge_index=edge_index, edge_attr=edge_weight.float(), y = torch.tensor(attr.time_to_reservation.values).float())

    return d


zones = [int(z[3:]) for z in pd.read_csv('data/processed/SimpleNNData.csv', index_col=0).filter(regex = 'lz').columns]
sdate = date(2019, 9, 1) # start date
delta = timedelta(days=10)
files = ['data/processed/Graphs/'+(sdate + timedelta(days=i)).strftime("%Y%m%d")+'.pickle' for i in range(delta.days + 1)]

dataset = []

with open(files[0], 'rb') as f:
    graph_collection = pickle.load(f)

for g in graph_collection.values():
    res = make_PTG(g,zones, Weather_Scale)
    if res:
        dataset.append(res)

train_data = [dataset[i] for i in nc[0][0]]
val_data = [dataset[i] for i in nc[0][1]]
test_data = [dataset[i] for i in nc[0][2]]

for file, slicer in tqdm(zip(files[1:4], nc[1:4])):
    dataset = []
    with open(file, 'rb') as f:
        graph_collection = pickle.load(f)

    for g in graph_collection.values():
        res = make_PTG(g,zones, Weather_Scale)
        if res:
            dataset.append(res)

    train_data = torch.utils.data.ConcatDataset([train_data,[dataset[i] for i in slicer[0]]])
    val_data = torch.utils.data.ConcatDataset([val_data,[dataset[i] for i in slicer[1]]])
    test_data = torch.utils.data.ConcatDataset([test_data,[dataset[i] for i in slicer[2]]])

del dataset, zones
'''