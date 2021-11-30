import pandas as pd
import networkx as nx
import numpy as np
import glob
from tqdm import tqdm
import pickle
from scipy import sparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datetime import date, timedelta
import torch.optim as optim
import torch.nn as nn
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from torch_geometric.nn import GCNConv
from torch_geometric import utils, data
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
import time
pd.set_option('mode.chained_assignment',None)
print(device)

t = time.time()

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

zones = [int(z[3:]) for z in pd.read_csv('GNNDatasets/SimpleNNData.csv', index_col=0).filter(regex = 'lz').columns]


sdate = date(2019, 8, 15)   # start date
edate = date(2019, 8, 31)   # end date
delta = edate - sdate       # as timedelta
files = ['Graphs/'+(sdate + timedelta(days=i)).strftime("%Y%m%d")+'.pickle' for i in range(delta.days + 1)]

dataset = []

print('Loading Files')
for file in tqdm(files):
    with open(file, 'rb') as f:
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
del dataset

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(269, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p = 0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p = 0.3, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)

        return x.squeeze()

GNN = GCN().to(device)
print(GNN, sum(p.numel() for p in GNN.parameters()))
print('Start learning')

optimizer = optim.Adam(GNN.parameters(), lr=0.01, weight_decay = 0.0001) #Chaged to Adam and learning + regulariztion rate set
criterion = nn.MSELoss(reduction = 'mean')



# Set number of epochs
num_epochs = 10

# Set up lists for loss/R2
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    ### Train
    cur_loss_train = []
    GNN.train()
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        out = GNN(batch)
        batch_loss = criterion(out[batch.ptr[1:]-1],batch.y[batch.ptr[1:]-1])
        batch_loss.backward()
        optimizer.step()

        cur_loss_train.append(batch_loss.item())
    
    train_losses.append(np.mean(cur_loss_train))
    


    ### Evaluate training
    with torch.no_grad():
        GNN.eval()
        train_preds, train_targs = [], []
        for batch in train_loader:
            target_mask = batch.ptr[1:]-1
            batch.to(device)
            preds = GNN(batch)
            train_targs += list(batch.y.cpu().numpy()[target_mask])
            train_preds += list(preds.cpu().detach().numpy()[target_mask])


    ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = []
        for batch in val_loader:
            batch.to(device)
            preds = GNN(batch)[batch.ptr[1:]-1]
            y_val = batch.y[batch.ptr[1:]-1]
            val_targs += list(y_val.cpu().numpy())
            val_preds += list(preds.cpu().detach().numpy())
            cur_loss_val.append(criterion(preds, y_val).item())

        val_losses.append(np.mean(cur_loss_val))


    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    print("Epoch %2i: Train Loss %f , Valid Loss %f ,Train R2 %f, Valid R2 %f" % (
                epoch+1, train_losses[-1], val_losses[-1],train_r2_cur, valid_r2_cur))


    
print(time.time()-t)
