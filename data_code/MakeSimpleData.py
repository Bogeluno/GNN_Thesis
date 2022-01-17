# Package imports
import pandas as pd
import glob
import pickle
from tqdm import tqdm


# Create data 
files = glob.glob("data/processed/Graphs/*")
cols = ['time', 'time_to_reservation', 'park_location_lat', 'park_location_long', 'leave_location_lat', 'leave_location_long','park_zone', 'leave_zone', 'park_fuel', 'leave_fuel', 'engine', 'moved','prev_customer', 'next_customer',
        'movedTF', 'action', 'weekend', 'Time_Cos', 'Time_Sin', 'mean_temp', 'mean_wind_speed', 'acc_precip', 'bright_sunshine', 'mean_pressure', 'mean_relative_hum', 'mean_cloud_cover', 'dist_to_station', 'degree']
df = pd.DataFrame(columns = cols)

i = 0
for f in tqdm(files):
    with open(f, 'rb') as day_file:
        day = pickle.load(day_file)

    for attr, adj in day.values():
        df.loc[i] = list(attr.iloc[-1].values)+[adj.getrow(-1).sum()]
        i += 1


df['timeh'] = df.time.round('h').dt.hour
trafic_index = dict(df.groupby('timeh').mean()['time_to_reservation'])
df.drop(columns = ['timeh'], inplace=True)
df['hour_index'] = df.time.dt.hour.map(trafic_index)

# Only to customers
df = df[df.next_customer]
df.drop(columns=['prev_customer', 'next_customer'], inplace = True)

# No more than 2 days
df = df[df.time_to_reservation < 48]

# Remove zones with too little support
df = df[~df.leave_zone.isin((df.leave_zone.value_counts() < 30).index[df.leave_zone.value_counts() < 30])]

# One hot encoding
df = pd.get_dummies(df, columns = ['engine','leave_zone'], prefix=['eng','lz'])

# Drop those far away
df = df[df['dist_to_station'] <= 7000]

# Drop some columns
df.drop(columns=['park_location_lat', 'park_location_long', 'leave_location_lat', 'leave_location_long', 'park_fuel', 'park_zone', 'moved', 'movedTF', 'action'], inplace = True)

# Reset index
df = df.reset_index(drop=True)

# Save
df.to_csv('data/processed/SimpleNNData.csv')