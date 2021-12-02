import pandas as pd
import glob
from tqdm import tqdm
import pickle

files = glob.glob("data/processed/Graphs/*")

cols = ['time', 'time_to_reservation', 'park_location_lat','park_location_long', 'leave_location_lat', 'leave_location_long', 'park_zone', 'leave_zone', 'park_fuel', 'leave_fuel', 'engine', 'moved', 'prev_customer', 'next_customer', 'movedTF', 'degree']
df = pd.DataFrame(columns = cols)
i = 0

for f in tqdm(files):
    with open(f, 'rb') as day_file:
        day = pickle.load(day_file)

    for attr, adj in day.values():
        df.loc[i] = list(attr.iloc[-1].values)+[adj.getrow(-1).sum()]
        i += 1

df.to_csv('data/processed/Fall2019P.csv')