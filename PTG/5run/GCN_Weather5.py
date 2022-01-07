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
from torch_geometric.nn import Sequential, GCNConv, Linear
from torch_geometric import utils, data
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, classification_report
from torch_scatter import scatter
import subprocess
import time
t = time.time()
pd.set_option('mode.chained_assignment',None)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
no_days = int(sys.argv[1])
num_epochs = int(sys.argv[2])
print(device)
name = "GCN_Weather5"
sys.stdout = open("Results/"+name+".txt", "w")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter % 5 == 0:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return -r2


batch_size = 512

# Load slicing
with open("Data/Sample_NC", "rb") as fp: 
    nc = pickle.load(fp)

with open("Data/Sample_CC", "rb") as fp: 
    cc = pickle.load(fp)

# For classification
df_full = pd.read_csv('Data/SimpleNNData.csv', index_col=0, parse_dates = [1]).sort_values(by = 'time')
Clas_Coef = dict(pd.concat([df_full.time.dt.hour.iloc[np.concatenate(cc[:2])],df_full.time_to_reservation.iloc[np.concatenate(cc[:2])]], axis = 1).groupby('time')['time_to_reservation'].mean()*2)
df_clas = pd.concat([df_full.time.dt.hour.iloc[cc[2]],df_full.time_to_reservation.iloc[cc[2]]], axis = 1)
df_clas['Cut'] = df_clas.time.map(dict(Clas_Coef))
df_clas = df_clas.iloc[:sum([len(x[2]) for x in nc[:(no_days+1)]])]
zones = [int(z[3:]) for z in df_full.filter(regex = 'lz').columns]
del df_full, cc

# Load weather
Weather_Scale = pd.read_csv('Data/MinMaxWeather.csv', index_col=0)


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
    attr.drop(columns = ['leave_zone'], inplace = True)
    # One hot encoding
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


# Load files
sdate = date(2019, 9, 1) # start date
delta = timedelta(days=no_days)
files = ['Graphs/'+(sdate + timedelta(days=i)).strftime("%Y%m%d")+'.pickle' for i in range(delta.days + 1)]

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


for file, slicer in tqdm(zip(files[1:], nc[1:len(files)]), total = len(files)-1):
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


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 4)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False, num_workers = 4)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convM = Sequential('x, edge_index, edge_weight', [
        (GCNConv(17,32, aggr = 'max'),'x, edge_index, edge_weight -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.1), 'x -> x')
        ])

        self.convA = Sequential('x, edge_index, edge_weight', [
        (GCNConv(17,32, aggr = 'add'),'x, edge_index, edge_weight -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.1), 'x -> x')
        ])

        self.linS = Sequential('x', [
        (Linear(17,32),'x -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.1), 'x -> x')
        ])

        self.seq = Sequential('x', [
            (Linear(96,48),'x -> x'),
            nn.ReLU(inplace = True),
            (nn.Dropout(0.2), 'x -> x'),
            (Linear(48,1),'x -> x')
        ])


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        xConvM = self.convM(x, edge_index, edge_weight)
        xConvA = self.convA(x, edge_index, edge_weight)
        xLin = self.linS(x)

        x = torch.cat([xConvM,xConvA,xLin], axis = 1)

        x = self.seq(x)

        return x.squeeze()



print(GCN(), sum(p.numel() for p in GCN().parameters()))


for _ in range(5):
    GNN = GCN().to(device)
    optimizer = optim.Adam(GNN.parameters(), lr=0.001, weight_decay = 0.00001) #Chaged to Adam and learning + regulariztion rate set

    # Set up lists for loss/R2
    train_r2, train_loss = [], []
    valid_r2, valid_loss = [], []
    cur_loss = 0
    train_losses = []
    val_losses = []

    early_stopping = EarlyStopping(patience=20, verbose=False, path = 'Checkpoints/'+name+'.pt')

    no_train = len(train_loader)
    no_val = len(val_loader)

    for epoch in tqdm(range(num_epochs)):
        ### Train
        cur_loss_train = 0
        GNN.train()
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            out = GNN(batch)
            batch_loss = r2_loss(out[batch.ptr[1:]-1],batch.y[batch.ptr[1:]-1])
            batch_loss.backward()
            optimizer.step()

            cur_loss_train += batch_loss.item()
        
        train_losses.append(cur_loss_train/no_train)

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
            cur_loss_val = 0
            for batch in val_loader:
                batch.to(device)
                preds = GNN(batch)[batch.ptr[1:]-1]
                y_val = batch.y[batch.ptr[1:]-1]
                val_targs += list(y_val.cpu().numpy())
                val_preds += list(preds.cpu().detach().numpy())
                cur_loss_val += r2_loss(preds, y_val)

            val_losses.append(cur_loss_val/no_val)


        train_r2_cur = r2_score(train_targs, train_preds)
        valid_r2_cur = r2_score(val_targs, val_preds)
        
        train_r2.append(train_r2_cur)
        valid_r2.append(valid_r2_cur)

        # EarlyStopping
        early_stopping(val_losses[-1], GNN)
        if early_stopping.early_stop:
            print("Early stopping")
            print("Epoch %2i: Train Loss %f , Valid Loss %f , Train R2 %f, Valid R2 %f" % (
                epoch+1, train_losses[-1], val_losses[-1], train_r2_cur, valid_r2_cur))
            break

        print("Epoch %2i: Train Loss %f, Valid Loss %f, Train R2 %f, Valid R2 %f" % (
                    epoch+1, train_losses[-1], val_losses[-1],train_r2_cur, valid_r2_cur))


    # Load best model
    GNN.load_state_dict(torch.load('Checkpoints/'+name+'.pt'))
    GNN.eval()
    print('-----------------------------------')
    print(f'Best val R2: {max(valid_r2)}')
    GNN.to(torch.device('cpu'))
    df_clas['Targets'] = [obs.y[-1].numpy().item() for obs in test_loader.dataset]
    df_clas['Preds'] = [GNN(b).detach().numpy().item(-1) for b in test_loader]
    print(f'Test score: {r2_score(df_clas.Targets,df_clas.Preds)}')
    print('F1-score:',classification_report(df_clas.Targets > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
    print(f'Time Spent: {time.time()-t}')
    print('\n')
    print('\n')

sys.stdout.close()