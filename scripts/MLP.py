import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import pickle
import time
from sklearn.metrics import r2_score, classification_report

pd.set_option('mode.chained_assignment',None)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
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
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
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


# Load full Data
df_full = pd.read_csv('Data/SimpleNNData.csv', index_col=0, parse_dates = [1]).sort_values(by = 'time')
y = df_full.time_to_reservation
df_full.drop(columns=['time_to_reservation', 'hour_index'], inplace=True)

# Load weather
Weather_Scale = pd.read_csv('Data/MinMaxWeather.csv', index_col=0)
weather_var = list(Weather_Scale.index)

# Load slicing
with open("Data/Sample_CC", "rb") as fp: 
    cc = pickle.load(fp)

# For classification
Clas_Coef = dict(pd.concat([df_full.time.dt.hour.iloc[np.concatenate(cc[:2])],y.iloc[np.concatenate(cc[:2])]], axis = 1).groupby('time')['time_to_reservation'].mean()*2)
df_clas = pd.concat([df_full.time.dt.hour.iloc[cc[2]],y.iloc[cc[2]]], axis = 1)
df_clas['Cut'] = df_clas.time.map(dict(Clas_Coef))

# Common setting
batch_size = 512
num_epochs = 1500

# Set up print
time_start = time.time()
sys.stdout = open("Results/MLP_3Sizes_Results.txt", "w")

'''
##################################
### NO ZONES
##################################
print('----------------------------------------------')
print('---NO ZONES')
print('----------------------------------------------')

# Prep data
df = df_full.drop(columns = list(df_full.filter(regex = 'lz').columns) + weather_var + ['dist_to_station','time'])
df['leave_fuel'] = df['leave_fuel']/100
df['degree'] = df['degree']/50

X_train = torch.tensor(df.iloc[cc[0]].to_numpy(dtype = 'float')).float()
y_train = torch.tensor(y.iloc[cc[0]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_val = torch.tensor(df.iloc[cc[1]].to_numpy(dtype = 'float')).float()
y_val = torch.tensor(y.iloc[cc[1]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_test = torch.tensor(df.iloc[cc[2]].to_numpy(dtype = 'float')).float()
y_test = torch.tensor(y.iloc[cc[2]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)

# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(9,32),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(32,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))

optimizer = optim.Adam(net.parameters(), lr=0.002, weight_decay = 0.0001) #Chaged to Adam and learning + regulariztion rate set

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/NoZones.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    
    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')


# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(9,32),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(16,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))

optimizer = optim.Adam(net.parameters(), lr=0.002, weight_decay = 0.0001) #Chaged to Adam and learning + regulariztion rate set

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/NoZones.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
    
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')

# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(9,128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))

optimizer = optim.Adam(net.parameters(), lr=0.002, weight_decay = 0.0001) #Chaged to Adam and learning + regulariztion rate set

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/NoZones.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')
print(f'Time spent: {time.time()-time_start}')
print('\n\n')


##################################
### ADD ZONES
##################################
print('----------------------------------------------')
print('---ADD ZONES')
print('----------------------------------------------')

# Prep data
df = df_full.drop(columns = weather_var + ['dist_to_station','time'])
df['leave_fuel'] = df['leave_fuel']/100
df['degree'] = df['degree']/50

X_train = torch.tensor(df.iloc[cc[0]].to_numpy(dtype = 'float')).float()
y_train = torch.tensor(y.iloc[cc[0]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_val = torch.tensor(df.iloc[cc[1]].to_numpy(dtype = 'float')).float()
y_val = torch.tensor(y.iloc[cc[1]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_test = torch.tensor(df.iloc[cc[2]].to_numpy(dtype = 'float')).float()
y_test = torch.tensor(y.iloc[cc[2]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)

# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(265,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))


optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/WithZones.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    
    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')

# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(265,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))


optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/WithZones.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    
    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')

# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(265,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))


optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/WithZones.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')


print(f'Time spent: {time.time()-time_start}')
print('\n\n')



##################################
### ADD ENCODED ZONES
##################################
print('----------------------------------------------')
print('---ADD ENCODED ZONES')
print('----------------------------------------------')

# Prep data
df = df_full.drop(columns = weather_var + ['dist_to_station','time'])
df['leave_fuel'] = df['leave_fuel']/100
df['degree'] = df['degree']/50
Mean_Zone_Times = dict(pd.DataFrame({'Zone': df.iloc[np.concatenate(cc[:2])].filter(regex = 'lz').idxmax(axis = 1).values, 'Time':y.iloc[np.concatenate(cc[:2])].values}).groupby('Zone').mean().squeeze())
df['Zone_E'] = df.filter(regex = 'lz').idxmax(1).map(Mean_Zone_Times)
df.drop(columns =  df.filter(regex = 'lz'), inplace = True)

X_train = torch.tensor(df.iloc[cc[0]].to_numpy(dtype = 'float')).float()
y_train = torch.tensor(y.iloc[cc[0]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_val = torch.tensor(df.iloc[cc[1]].to_numpy(dtype = 'float')).float()
y_val = torch.tensor(y.iloc[cc[1]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_test = torch.tensor(df.iloc[cc[2]].to_numpy(dtype = 'float')).float()
y_test = torch.tensor(y.iloc[cc[2]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)

# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(10,32),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))


optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/WithZonesEncoded.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('(\n')


# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(10,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))


optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/WithZonesEncoded.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('(\n')


# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(10,128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))


optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/WithZonesEncoded.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')

print(f'Time spent: {time.time()-time_start}')
print('\n\n')
'''

##################################
### ADD WEATHER AND DIST
##################################
print('----------------------------------------------')
print('---ADD WEATHER AND DIST')
print('----------------------------------------------')
# Prep data
df = df_full.drop(columns = list(df_full.filter(regex = 'lz').columns) + ['time'])
df['leave_fuel'] = df['leave_fuel']/100
df['degree'] = df['degree']/50
df['dist_to_station'] = df['dist_to_station']/5000
df[Weather_Scale.index] = (df[Weather_Scale.index] - Weather_Scale['Min'])/Weather_Scale['diff']

X_train = torch.tensor(df.iloc[cc[0]].to_numpy(dtype = 'float')).float()
y_train = torch.tensor(y.iloc[cc[0]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_val = torch.tensor(df.iloc[cc[1]].to_numpy(dtype = 'float')).float()
y_val = torch.tensor(y.iloc[cc[1]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_test = torch.tensor(df.iloc[cc[2]].to_numpy(dtype = 'float')).float()
y_test = torch.tensor(y.iloc[cc[2]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)

# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(17,32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))

optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/WithWeather.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')

# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(17,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))

optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/WithWeather.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')

# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(17,128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))

optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/WithWeather.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')

print(f'Time spent: {time.time()-time_start}')
print('\n\n')

##################################
### With all
##################################
print('----------------------------------------------')
print('---WITH ALL')
print('----------------------------------------------')
# Prep data
df = df_full.drop(columns = ['time'])
df['leave_fuel'] = df['leave_fuel']/100
df['degree'] = df['degree']/50
df['dist_to_station'] = df['dist_to_station']/5000
df[Weather_Scale.index] = (df[Weather_Scale.index] - Weather_Scale['Min'])/Weather_Scale['diff']

X_train = torch.tensor(df.iloc[cc[0]].to_numpy(dtype = 'float')).float()
y_train = torch.tensor(y.iloc[cc[0]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_val = torch.tensor(df.iloc[cc[1]].to_numpy(dtype = 'float')).float()
y_val = torch.tensor(y.iloc[cc[1]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_test = torch.tensor(df.iloc[cc[2]].to_numpy(dtype = 'float')).float()
y_test = torch.tensor(y.iloc[cc[2]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)



# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(273,16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))


optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/Full.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')


# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(273,32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))


optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/Full.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')


# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(273,128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))


optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/Full.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')

print(f'Time spent: {time.time()-time_start}')
print('\n\n')

##################################
### With all and encoded
##################################
print('----------------------------------------------')
print('---WITH ALL AND ENCODED')
print('----------------------------------------------')
# Prep data
df = df_full.drop(columns = ['time'])
df['leave_fuel'] = df['leave_fuel']/100
df['degree'] = df['degree']/50
df['dist_to_station'] = df['dist_to_station']/5000
df[Weather_Scale.index] = (df[Weather_Scale.index] - Weather_Scale['Min'])/Weather_Scale['diff']
Mean_Zone_Times = dict(pd.DataFrame({'Zone': df.iloc[np.concatenate(cc[:2])].filter(regex = 'lz').idxmax(axis = 1).values, 'Time':y.iloc[np.concatenate(cc[:2])].values}).groupby('Zone').mean().squeeze())
df['Zone_E'] = df.filter(regex = 'lz').idxmax(1).map(Mean_Zone_Times)
df.drop(columns =  df.filter(regex = 'lz'), inplace = True)

X_train = torch.tensor(df.iloc[cc[0]].to_numpy(dtype = 'float')).float()
y_train = torch.tensor(y.iloc[cc[0]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_val = torch.tensor(df.iloc[cc[1]].to_numpy(dtype = 'float')).float()
y_val = torch.tensor(y.iloc[cc[1]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)
X_test = torch.tensor(df.iloc[cc[2]].to_numpy(dtype = 'float')).float()
y_test = torch.tensor(y.iloc[cc[2]].to_numpy(dtype = 'float')).float().unsqueeze(dim = 1)

# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(18,16),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(16,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))


optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/FullEncoded.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')



# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(18,32),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(16,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))

optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/FullEncoded.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')


# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.seq = nn.Sequential(
            nn.Linear(18,128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,1),
        )

        self.seq.apply(init_weights)
        

    def forward(self, x):
        x = self.seq(x)
        return x

net = Net()
print(net, sum(p.numel() for p in net.parameters()))

optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001)

num_samples_train = X_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = X_val.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_r2, train_loss = [], []
valid_r2, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
train_losses = []
val_losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

path = 'Checkpoints/FullEncoded.pt'
early_stopping = EarlyStopping(patience=20, verbose=False, path = path)

for epoch in tqdm(range(num_epochs)):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss_train = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(X_train[slce])
        
        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = r2_loss(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss_train += batch_loss
    train_losses.append(cur_loss_train/num_batches_train)

    ### Evaluate training
    with torch.no_grad():
        net.eval()
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = net(X_train[slce])
            
            preds = output
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())


        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss_val = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = net(X_val[slce])
            preds = output
            val_targs += list(y_val[slce].numpy())
            val_preds += list(preds.data.numpy())

            cur_loss_val += r2_loss(output, y_val[slce])

        val_losses.append(cur_loss_val/num_batches_valid)

    train_r2_cur = r2_score(train_targs, train_preds)
    valid_r2_cur = r2_score(val_targs, val_preds)
    
    train_r2.append(train_r2_cur)
    valid_r2.append(valid_r2_cur)

    # EarlyStopping
    early_stopping(val_losses[-1], net)
    if early_stopping.early_stop:
        break


# Load best model
net.load_state_dict(torch.load(path))
net.eval()

print('Test R2:',r2_score(y_test.detach().numpy()[:,0],net.forward(X_test).detach().numpy()[:,0]))
df_clas['Preds'] = net.forward(X_test).detach().numpy()[:,0]
print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
print('\n')
print(f'Time spent: {time.time()-time_start}')
print('\n\n')

##################################
### End and exit stdout
##################################
sys.stdout.close()