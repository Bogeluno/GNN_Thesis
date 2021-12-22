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
from torch_geometric.nn import Sequential, GCNConv, GATConv, TopKPooling, Linear, MessagePassing
from torch_geometric import utils, data
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from torch_scatter import scatter
import subprocess
import time



#####################
#####################
#####################
#####################
#####################
from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
class GATv2Conv(MessagePassing):
  
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, torch.Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[torch.Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights: bool = None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, torch.Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, torch.Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, torch.Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, edge_attr: OptTensor,
                index: torch.Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x += edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

###################
###################
###################
###################
###################


sys.stdout = open("PTG_GATV2.txt", "w")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
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
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter % 5 == 0:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        self.val_loss_min = val_loss

def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return -r2

no_days = int(sys.argv[1])
batch_size = int(sys.argv[2])

if no_days > 60:
    sys.exit('No days can not be larger than 60')

print(f'Number of days: {no_days}')
print(f'Batch_size: {batch_size}')
pd.set_option('mode.chained_assignment',None)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
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

def pool_neighbor(x, edge_index, num_nodes, reduce = 'mean'):
    r"""Average pools neighboring node features, where each feature in
    :obj:`data.x` is replaced by the average feature values from the central
    node and its neighbors.
    """

    row, col = edge_index
    row, col = (row, col)

    x += scatter(x[row], col, dim=0, dim_size=num_nodes, reduce=reduce)
    return x



print(f'Time spent: {time.time()-t}')
# Load datasets
if no_days == 0:
    with open('GNNDatasets/Train_data.pickle', 'rb') as f:
        train_data = pickle.load(f)
    print('Train data loaded')

    with open('GNNDatasets/Val_data.pickle', 'rb') as f:
        val_data = pickle.load(f)
    print('Validation data loaded')

    with open('GNNDatasets/Test_data.pickle', 'rb') as f:
        test_data = pickle.load(f)
    print('Test data loaded')

# Generate datasets
else:
    zones = [int(z[3:]) for z in pd.read_csv('SimpleNNData.csv', index_col=0).filter(regex = 'lz').columns]

    sdate = date(2019, 9, 1) # start date
    delta = timedelta(days=no_days)
    files = ['Graphs/'+(sdate + timedelta(days=i)).strftime("%Y%m%d")+'.pickle' for i in range(delta.days + 1)]

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

    del train_val_data_tmp, test_data_tmp, train_data_tmp, val_data_tmp, dataset, zones
print(f'Time spent: {time.time()-t}')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 4)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False, num_workers = 4)
del train_data, val_data, test_data
print(subprocess.run(['free', '-m'], stdout=subprocess.PIPE).stdout.decode('utf-8'))


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convM = Sequential('x, edge_index, edge_weight', [
        (GATv2Conv(264,48, edge_dim = 1, aggr = 'max'),'x, edge_index, edge_weight -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.25), 'x -> x')
        ])

        self.convA = Sequential('x, edge_index, edge_weight', [
        (GATv2Conv(264,48, edge_dim = 1, aggr = 'add'),'x, edge_index, edge_weight -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.25), 'x -> x')
        ])

        self.linS = Sequential('x', [
        (Linear(264,48),'x -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.25), 'x -> x')
        ])

        self.seq = Sequential('x', [
            (Linear(144,64),'x -> x'),
            nn.ReLU(inplace = True),
            (nn.Dropout(0.25), 'x -> x'),
            (Linear(64,1),'x -> x')
        ])


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        xConvM = self.convM(x, edge_index, edge_weight)
        xConvA = self.convA(x, edge_index, edge_weight)
        xLin = self.linS(x)

        x = torch.cat([xConvM,xConvA,xLin], axis = 1)

        x = self.seq(x)

        return x.squeeze()


GNN = GCN().to(device)
print(GNN, sum(p.numel() for p in GNN.parameters()))
print('Start learning')

optimizer = optim.Adam(GNN.parameters(), lr=0.001, weight_decay = 0.00001) #Chaged to Adam and learning + regulariztion rate set

# Set number of epochs
num_epochs = int(sys.argv[3])

# Set up lists for loss/R2
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

early_stopping = EarlyStopping(patience=10, verbose=False)

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
    early_stopping(val_losses[-1], GCN)
    if early_stopping.early_stop:
        print("Early stopping")
        print("Epoch %2i: Train Loss %f , Valid Loss %f , Train R2 %f, Valid R2 %f" % (
            epoch+1, train_losses[-1], val_losses[-1], train_r2_cur, valid_r2_cur))
        break

    print("Epoch %2i: Train Loss %f, Valid Loss %f, Train R2 %f, Valid R2 %f" % (
                epoch+1, train_losses[-1], val_losses[-1],train_r2_cur, valid_r2_cur))


print(f'Best val R2: {max(valid_r2)}')

GNN.to(torch.device('cpu'))
test_preds = [GNN(b).detach().numpy().item(-1) for b in test_loader]
test_targest = [obs.y[-1].numpy().item() for obs in test_loader.dataset]
print(f'Test score: {r2_score(test_targest,test_preds)}')
print(f'Time Spent: {time.time()-t}')

#############
print('\n')
print('\n')


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convM = Sequential('x, edge_index, edge_weight', [
        (GATv2Conv(264,48, edge_dim=1, aggr = 'max'),'x, edge_index, edge_weight -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.25), 'x -> x')
        ])

        self.convA = Sequential('x, edge_index, edge_weight', [
        (GATv2Conv(264,48, edge_dim=1, aggr = 'add'),'x, edge_index, edge_weight -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.25), 'x -> x')
        ])

        self.seq = Sequential('x', [
            (Linear(96,64),'x -> x'),
            nn.ReLU(inplace = True),
            (nn.Dropout(0.25), 'x -> x'),
            (Linear(64,1),'x -> x')
        ])


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        xConvM = self.convM(x, edge_index, edge_weight)
        xConvA = self.convA(x, edge_index, edge_weight)

        x = torch.cat([xConvM,xConvA], axis = 1)

        x = self.seq(x)

        return x.squeeze()


GNN = GCN().to(device)
print(GNN, sum(p.numel() for p in GNN.parameters()))
print('Start learning')

optimizer = optim.Adam(GNN.parameters(), lr=0.001, weight_decay = 0.00001) #Chaged to Adam and learning + regulariztion rate set

#print(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

# Set number of epochs
num_epochs = int(sys.argv[3])

# Set up lists for loss/R2
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

early_stopping = EarlyStopping(patience=10, verbose=False)

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
    early_stopping(val_losses[-1], GCN)
    if early_stopping.early_stop:
        print("Early stopping")
        print("Epoch %2i: Train Loss %f , Valid Loss %f , Train R2 %f, Valid R2 %f" % (
            epoch+1, train_losses[-1], val_losses[-1], train_r2_cur, valid_r2_cur))
        break

    print("Epoch %2i: Train Loss %f, Valid Loss %f, Train R2 %f, Valid R2 %f" % (
                epoch+1, train_losses[-1], val_losses[-1],train_r2_cur, valid_r2_cur))


print(f'Best val R2: {max(valid_r2)}')

GNN.to(torch.device('cpu'))
test_preds = [GNN(b).detach().numpy().item(-1) for b in test_loader]
test_targest = [obs.y[-1].numpy().item() for obs in test_loader.dataset]
print(f'Test score: {r2_score(test_targest,test_preds)}')
print(f'Time Spent: {time.time()-t}')

#############
print('\n')
print('\n')


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convA = Sequential('x, edge_index, edge_weight', [
        (GATv2Conv(264,48, edge_dim=1, aggr = 'add'),'x, edge_index, edge_weight -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.25), 'x -> x')
        ])

        self.linS = Sequential('x', [
        (Linear(264,48),'x -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.25), 'x -> x')
        ])

        self.seq = Sequential('x', [
            (Linear(96,64),'x -> x'),
            nn.ReLU(inplace = True),
            (nn.Dropout(0.25), 'x -> x'),
            (Linear(64,1),'x -> x')
        ])


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        xConvA = self.convA(x, edge_index, edge_weight)
        xLin = self.linS(x)

        x = torch.cat([xConvA,xLin], axis = 1)

        x = self.seq(x)

        return x.squeeze()


GNN = GCN().to(device)
print(GNN, sum(p.numel() for p in GNN.parameters()))
print('Start learning')

optimizer = optim.Adam(GNN.parameters(), lr=0.001, weight_decay = 0.00001) #Chaged to Adam and learning + regulariztion rate set

#print(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

# Set number of epochs
num_epochs = int(sys.argv[3])

# Set up lists for loss/R2
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

early_stopping = EarlyStopping(patience=10, verbose=False)

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
    early_stopping(val_losses[-1], GCN)
    if early_stopping.early_stop:
        print("Early stopping")
        print("Epoch %2i: Train Loss %f , Valid Loss %f , Train R2 %f, Valid R2 %f" % (
            epoch+1, train_losses[-1], val_losses[-1], train_r2_cur, valid_r2_cur))
        break

    print("Epoch %2i: Train Loss %f, Valid Loss %f, Train R2 %f, Valid R2 %f" % (
                epoch+1, train_losses[-1], val_losses[-1],train_r2_cur, valid_r2_cur))


print(f'Best val R2: {max(valid_r2)}')

GNN.to(torch.device('cpu'))
test_preds = [GNN(b).detach().numpy().item(-1) for b in test_loader]
test_targest = [obs.y[-1].numpy().item() for obs in test_loader.dataset]
print(f'Test score: {r2_score(test_targest,test_preds)}')
print(f'Time Spent: {time.time()-t}')

#############
print('\n')
print('\n')



class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convM = Sequential('x, edge_index, edge_weight', [
        (GATv2Conv(264,48, edge_dim=1, aggr = 'max'),'x, edge_index, edge_weight -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.25), 'x -> x')
        ])

        self.linS = Sequential('x', [
        (Linear(264,48),'x -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.25), 'x -> x')
        ])

        self.seq = Sequential('x', [
            (Linear(96,64),'x -> x'),
            nn.ReLU(inplace = True),
            (nn.Dropout(0.25), 'x -> x'),
            (Linear(64,1),'x -> x')
        ])


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        xConvM = self.convM(x, edge_index, edge_weight)
        xLin = self.linS(x)

        x = torch.cat([xConvM,xLin], axis = 1)

        x = self.seq(x)

        return x.squeeze()


GNN = GCN().to(device)
print(GNN, sum(p.numel() for p in GNN.parameters()))
print('Start learning')

optimizer = optim.Adam(GNN.parameters(), lr=0.001, weight_decay = 0.00001) #Chaged to Adam and learning + regulariztion rate set

#print(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

# Set number of epochs
num_epochs = int(sys.argv[3])

# Set up lists for loss/R2
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

early_stopping = EarlyStopping(patience=10, verbose=False)

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
    early_stopping(val_losses[-1], GCN)
    if early_stopping.early_stop:
        print("Early stopping")
        print("Epoch %2i: Train Loss %f , Valid Loss %f , Train R2 %f, Valid R2 %f" % (
            epoch+1, train_losses[-1], val_losses[-1], train_r2_cur, valid_r2_cur))
        break

    print("Epoch %2i: Train Loss %f, Valid Loss %f, Train R2 %f, Valid R2 %f" % (
                epoch+1, train_losses[-1], val_losses[-1],train_r2_cur, valid_r2_cur))


print(f'Best val R2: {max(valid_r2)}')

GNN.to(torch.device('cpu'))
test_preds = [GNN(b).detach().numpy().item(-1) for b in test_loader]
test_targest = [obs.y[-1].numpy().item() for obs in test_loader.dataset]
print(f'Test score: {r2_score(test_targest,test_preds)}')
print(f'Time Spent: {time.time()-t}')


#############
print('\n')
print('\n')


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convM1 = Sequential('x, edge_index, edge_weight', [
        (GATv2Conv(264,64, edge_dim=1, aggr = 'max'),'x, edge_index, edge_weight -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.25), 'x -> x')
        ])

        self.convM2 = Sequential('x, edge_index, edge_weight', [
        (GATv2Conv(64,64, edge_dim=1, aggr = 'max'),'x, edge_index, edge_weight -> x'),
        nn.ReLU(inplace = True),
        (nn.Dropout(0.25), 'x -> x')
        ])

        self.seq = Sequential('x', [
            (Linear(64,64),'x -> x'),
            nn.ReLU(inplace = True),
            (nn.Dropout(0.25), 'x -> x'),
            (Linear(64,1),'x -> x')
        ])


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.convM1(x, edge_index, edge_weight)
        x = self.convM2(x, edge_index, edge_weight)

        x = self.seq(x)

        return x.squeeze()


GNN = GCN().to(device)
print(GNN, sum(p.numel() for p in GNN.parameters()))
print('Start learning')

optimizer = optim.Adam(GNN.parameters(), lr=0.001, weight_decay = 0.00001) #Chaged to Adam and learning + regulariztion rate set

#print(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

# Set number of epochs
num_epochs = int(sys.argv[3])

# Set up lists for loss/R2
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

early_stopping = EarlyStopping(patience=10, verbose=False)

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
    early_stopping(val_losses[-1], GCN)
    if early_stopping.early_stop:
        print("Early stopping")
        print("Epoch %2i: Train Loss %f , Valid Loss %f , Train R2 %f, Valid R2 %f" % (
            epoch+1, train_losses[-1], val_losses[-1], train_r2_cur, valid_r2_cur))
        break

    print("Epoch %2i: Train Loss %f, Valid Loss %f, Train R2 %f, Valid R2 %f" % (
                epoch+1, train_losses[-1], val_losses[-1],train_r2_cur, valid_r2_cur))


print(f'Best val R2: {max(valid_r2)}')

GNN.to(torch.device('cpu'))
test_preds = [GNN(b).detach().numpy().item(-1) for b in test_loader]
test_targest = [obs.y[-1].numpy().item() for obs in test_loader.dataset]
print(f'Test score: {r2_score(test_targest,test_preds)}')
print(f'Time Spent: {time.time()-t}')

sys.stdout.close()