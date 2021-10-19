# Load data
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle
from scipy import sparse
import matplotlib.pyplot as plt
import torch
#from torch_geometric import utils, data

def haversine_start(df, car1, car2, max_dist = 1500):
    def _edge_weight(x, max_dist):
        return max((max_dist-x)/max_dist,0)
    dfc1 = df[df.car == car1]
    dfc2 = df[df.car == car2]

    point1 = dfc1[['leave_location_lat','leave_location_long']].values[0]
    point2 = dfc2[['leave_location_lat','leave_location_long']].values[0]
    #return point1

    # convert decimal degrees to radians
    lat1, lon1 = map(np.radians, point1)
    lat2, lon2 = map(np.radians, point2)

    # Deltas
    delta_lon = lon2 - lon1 
    delta_lat = lat2 - lat1 
    
    # haversine formula 
    a = np.sin(delta_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371000 # Radius of earth in m
    return _edge_weight(c * r, max_dist)

def haversine_add(current_locs, to_add, max_dist = 1500):
    def _edge_weight(x, max_dist):
        return max((max_dist-x)/max_dist,0)
    def _haversine(point1, lat2, lon2):
        
         # convert decimal degrees to radians
        lat1, lon1 = map(np.radians, point1)

        # Deltas
        delta_lon = lon2 - lon1 
        delta_lat = lat2 - lat1 
        
        # haversine formula 
        a = np.sin(delta_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        r = 6371000 # Radius of earth in m
        return c * r

    leave_lat_add, leave_long_add = to_add.leave_location_lat, to_add.leave_location_long

    lat2, lon2 = map(np.radians, [leave_lat_add,leave_long_add])

    new_weights = [_edge_weight(_haversine(loc, lat2, lon2), max_dist) for loc in current_locs]

    return new_weights


def delete_rc(mat, i):
    # row
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

    # col
    mat = mat.tocsc()
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0], mat._shape[1]-1)

    return mat.tocsr()


# Load data
df = pd.read_csv('VacancySplit.csv', index_col=0, parse_dates = [2]).astype({'time_to_reservation': 'float32', 'park_location_lat': 'float32', 'park_location_long': 'float32', 'leave_location_lat': 'float32', 'leave_location_long': 'float32', 'park_zone': 'int32', 'leave_zone': 'int32', 'park_fuel': 'int8', 'leave_fuel': 'int8', 'moved': 'float32', 'movedTF': 'bool'})
df.info()

# Create init graph
Start_Time = pd.Timestamp('2018-07-09 14:27:00')
start_df = df[df.time <= Start_Time]
propegate_df = df[df.time > Start_Time]

# Start
CarID_dict_start = dict(iter(start_df.groupby('car')))
Start_Garph_data = []

for sub_df in CarID_dict_start.values():
    last_obs = sub_df.iloc[-1]
    if last_obs.action: # True is park
        Start_Garph_data.append(last_obs)

start_df_graph = pd.DataFrame(Start_Garph_data).iloc[:,:-1]


# Adj matrix
max_dist = 1500
def _edge_weight(x, max_dist):
        return max((max_dist-x)/max_dist,0)
A = pd.DataFrame(data = [[haversine_start(start_df_graph, car1, car2) for car1 in start_df_graph.car] for car2 in tqdm(start_df_graph.car)], index = start_df_graph.car, columns=start_df_graph.car, dtype='float16')

# And make it sparse
As = sparse.csr_matrix(A.values)

# Populate
Graph_dict = {pd.Timestamp('2018-07-09 14:27:00'): (start_df_graph ,As)}
node_data = start_df_graph.set_index('car')

time_to_next = propegate_df.time.diff().shift(-1)
positive_time = (time_to_next > pd.Timedelta(0,'s'))

new_day = (propegate_df.time.dt.date.diff().shift(-1) > pd.Timedelta(0,'s'))


for idx, next_row in tqdm(propegate_df.iterrows(), total = propegate_df.shape[0]):
    if next_row.action: # True is park
        # Get current locs
        locs = [[attr['leave_location_lat'], attr['leave_location_long']] for _, attr in node_data.iterrows()]
        
        # Add to node data
        node_data = node_data.append(next_row.rename(index = next_row['car']).iloc[1:16], verify_integrity = True)

        # Calculate new weights
        new_weights = haversine_add(locs, next_row, max_dist = 1500)

        # Add new weights to adjacency
        As = sparse.hstack([sparse.vstack([As,sparse.csr_matrix(new_weights)]).tocsc(), sparse.csc_matrix(new_weights+[1]).T]).tocsr()

    else:
        # Getindex
        idx_to_drop = np.where(node_data.index == next_row.car)[0][0]

        # Drop it
        As = delete_rc(As, idx_to_drop)

        # Drop from feature-matrix
        node_data.drop(index = next_row.car, inplace=True)
        
    # Save graph if new time 
    if positive_time[idx]:
        Graph_dict[next_row.time] = (node_data, As)

    # Save file every day on last obs
    if new_day[idx]:
        f_name = next_row.time.strftime('graphs/%Y%m%d')+'.pickle'
        with open(f_name, 'wb') as handle:
            pickle.dump(Graph_dict, handle, pickle.HIGHEST_PROTOCOL)

        # Clear memory
        Graph_dict = {}

