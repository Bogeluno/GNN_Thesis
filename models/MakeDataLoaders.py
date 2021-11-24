import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from scipy import sparse
import torch
from datetime import date, timedelta
from torch_geometric import utils, data
pd.set_option('mode.chained_assignment',None)

# Get zones
zones = [int(z[3:]) for z in pd.read_csv('data/processed/SimpleNNData.csv', index_col=0).filter(regex = 'lz').columns]

# Make datasets for PTG
def make_PTG(graph, zones):
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

    # Time variables
    attr['weekend'] = attr.time.dt.weekday//5

    def circle_transform(col, max_val=86400):
        tot_sec = ((col - col.dt.normalize()) / pd.Timedelta('1 second')).astype(int)
        cos_val = np.cos(2*np.pi*tot_sec/max_val)
        sin_val = np.sin(2*np.pi*tot_sec/max_val)
        return cos_val, sin_val

    attr['Time_Cos'], attr['Time_Sin'] = [x.values for x in circle_transform(attr.time)]

    # drop
    attr.drop(columns=['park_location_lat', 'park_location_long', 'leave_location_lat', 'leave_location_long', 'park_fuel', 'park_zone', 'moved', 'movedTF', 'time', 'prev_customer', 'next_customer'], inplace = True)

    # One hot encoding
    attr['leave_zone'] = pd.Categorical(attr['leave_zone'], categories=zones)
    attr = pd.get_dummies(attr, columns = ['leave_zone'], prefix='lz')

    attr['engine']= pd.Categorical(attr['engine'], categories=['118I', 'I3', 'COOPER', 'X1'])
    attr = pd.get_dummies(attr, columns = ['engine'], prefix='eng')

    # Get edges
    edge_index, edge_weight = utils.convert.from_scipy_sparse_matrix(adj)

    # Make pytorch data type
    d = data.Data(x = torch.tensor(attr.drop(columns = ['time_to_reservation']).to_numpy(dtype = 'float')).float(), edge_index=edge_index, edge_attr=edge_weight.float(), y = torch.tensor(attr.time_to_reservation.values).float())

    return d



# Get files
sdate = date(2019, 8, 15)   # start date
edate = date(2019, 12, 18)   # end date
delta = edate - sdate       # as timedelta
files = ['data/processed/graphs/'+(sdate + timedelta(days=i)).strftime("%Y%m%d")+'.pickle' for i in range(delta.days + 1)]


# load them
dataset = []

for file in tqdm(files):
    with open(file, 'rb') as f:
        graph_collection = pickle.load(f)

    for g in graph_collection.values():
        res = make_PTG(g,zones)
        if res:
            dataset.append(res)


# Creat train, val, test
train_val_size = int(0.8 * len(dataset))
val_test_size = len(dataset)-train_val_size
train_val_data, test_data = torch.utils.data.random_split(dataset, [train_val_size, val_test_size])
train_size = train_val_size-val_test_size
train_data, val_data = torch.utils.data.random_split(train_val_data, [train_size, val_test_size])
del dataset, zones, files,

# Save them
print('Saving files')
with open('data/processed/GNNDatasets/Train.pickle', 'wb') as handle:
    pickle.dump(train_data, handle, pickle.HIGHEST_PROTOCOL)
del train_data
print('Train done')

with open('data/processed/GNNDatasets/Val.pickle', 'wb') as handle:
    pickle.dump(val_data, handle, pickle.HIGHEST_PROTOCOL)
del val_data
print('Val done')

with open('data/processed/GNNDatasets/Test.pickle', 'wb') as handle:
    pickle.dump(test_data, handle, pickle.HIGHEST_PROTOCOL)
del test_data

print('Test done')