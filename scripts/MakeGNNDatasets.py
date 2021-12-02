import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import glob
from scipy import sparse
import torch
import gc
import subprocess
from torch_geometric import utils, data
pd.set_option('mode.chained_assignment',None)

# Get zones
zones = [int(z[3:]) for z in pd.read_csv('SimpleNNData.csv', index_col=0).filter(regex = 'lz').columns]

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

    # Normalize fuel and dist 
    attr['leave_fuel'] = attr['leave_fuel']/100
    #df['dist_to_station'] = df['dist_to_station']/5320

    # Get edges
    edge_index, edge_weight = utils.convert.from_scipy_sparse_matrix(adj)

    # Make pytorch data type
    d = data.Data(x = torch.tensor(attr.drop(columns = ['time_to_reservation']).to_numpy(dtype = 'float')).float(), edge_index=edge_index, edge_attr=edge_weight.float(), y = torch.tensor(attr.time_to_reservation.values).float())

    return d

# Get files
files = glob.glob("Graphs/*")

dataset = []
with open(files[0], 'rb') as f:
    graph_collection = pickle.load(f)

for g in graph_collection.values():
    res = make_PTG(g,zones)
    if res:
        dataset.append(res)

train_val_size = int(0.8 * len(dataset))
val_test_size = len(dataset)-train_val_size
train_val_data, test_data = torch.utils.data.random_split(dataset, [train_val_size, val_test_size])
train_size = train_val_size-val_test_size
train_data, val_data = torch.utils.data.random_split(train_val_data, [train_size, val_test_size])
del train_val_data

for file in tqdm(files[1:]):
    dataset = []
    with open(file, 'rb') as f:
        graph_collection = pickle.load(f)

    for g in graph_collection.values():
        res = make_PTG(g,zones)
        if res:
            dataset.append(res)

    train_val_size = int(0.8 * len(dataset))
    val_test_size = len(dataset)-train_val_size
    train_val_data_tmp, test_data_tmp = torch.utils.data.random_split(dataset, [train_val_size, val_test_size])
    train_size = train_val_size-val_test_size
    train_data_tmp, val_data_tmp = torch.utils.data.random_split(train_val_data_tmp, [train_size, val_test_size])

    train_data = torch.utils.data.ConcatDataset([train_data,train_data_tmp])
    val_data = torch.utils.data.ConcatDataset([val_data,val_data_tmp])
    test_data = torch.utils.data.ConcatDataset([test_data,test_data_tmp])

print(subprocess.run(['free', '-m'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
print('Deleting other variables')
del zones, files, dataset, res, train_val_size, val_test_size, train_size, train_val_data_tmp, train_data_tmp, val_data_tmp, test_data_tmp
gc.collect()
print(subprocess.run(['free', '-m'], stdout=subprocess.PIPE).stdout.decode('utf-8'))


with open(f'GNNDatasets/Val_data.pickle', 'wb') as handle:
    pickle.dump(val_data, handle, pickle.HIGHEST_PROTOCOL)
print('Val dumped\n')
del val_data
gc.collect()
print(subprocess.run(['free', '-m'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

with open(f'GNNDatasets/Test_data.pickle', 'wb') as handle:
    pickle.dump(test_data, handle, pickle.HIGHEST_PROTOCOL)
print('Test dumped\n')
del test_data
gc.collect()
print(subprocess.run(['free', '-m'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

with open(f'GNNDatasets/Train_data.pickle', 'wb') as handle:
    pickle.dump(train_data, handle, pickle.HIGHEST_PROTOCOL)
print('Train dumped')