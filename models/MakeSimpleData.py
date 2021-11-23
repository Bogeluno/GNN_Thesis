# Package imports
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


# Get trafic index
#tdf = pd.read_csv('data/processed/Vacancy_new.csv', parse_dates=[2,3])
#tdf['park_time'] = tdf.park_time.round('h').dt.hour
#trafic_index = dict(tdf.groupby('park_time').mean()['time_to_reservation'])
#del tdf



# Load data 
df = pd.read_csv('data/processed/Fall2019P.csv', index_col=0, parse_dates=[1])

df['timeh'] = df.time.round('h').dt.hour
trafic_index = dict(df.groupby('timeh').mean()['time_to_reservation'])
df.drop(columns = ['timeh'], inplace=True)

# Only to customers
df = df[df.next_customer]
df.drop(columns=['prev_customer', 'next_customer'], inplace = True)

# Time variables
df['weekend'] = df.time.dt.weekday//5

def circle_transform(col, max_val=86400):
    tot_sec = ((col - col.dt.normalize()) / pd.Timedelta('1 second')).astype(int)
    cos_val = np.cos(2*np.pi*tot_sec/max_val)
    sin_val = np.sin(2*np.pi*tot_sec/max_val)
    return cos_val, sin_val

df['Time_Cos'], df['Time_Sin'] = [x.values for x in circle_transform(df.time)]
df['hour_index'] = df.time.dt.hour.map(trafic_index)

# No more than 2 days
df = df[df.time_to_reservation < 48]

# Remove zones with too little support
df = df[~df.leave_zone.isin((df.leave_zone.value_counts() < 30).index[df.leave_zone.value_counts() < 30])]

# One hot encoding
df = pd.get_dummies(df, columns = ['engine','leave_zone'], prefix=['eng','lz'])

# Join weather
df_weather = pd.read_csv('data/processed/weather.csv', index_col=0, parse_dates=[0])
df.time = df.time.round('H')
df = df.set_index('time').join(df_weather).reset_index()

# Average weather
df['park_time'] = df['index'].round('H')
df['reserve_time'] = (df['index']+pd.to_timedelta(df.time_to_reservation,'h')).round('H')

weather_average1 = pd.DataFrame(data=[df_weather.loc[
    pd.to_datetime(np.linspace(row.park_time.value, row.reserve_time.value, int((row.reserve_time-row.park_time)/ np.timedelta64(1, 'h'))+1)) 
    ].mean().values for _, row in tqdm(df.iterrows(), total = len(df))
], columns = ['Avg_'+x for x in df_weather.columns], index = df.index)

df = pd.concat([df ,weather_average1], axis = 1)

# Average Weather Index
df['park_time'] = df['index'].round('H')

weather_average2 = pd.DataFrame(data=[df_weather.loc[
    pd.to_datetime(np.linspace(row.park_time.value, row.park_time.value+pd.to_timedelta(round(row.hour_index), 'h').value, int((row.park_time+pd.to_timedelta(round(row.hour_index), 'h')-row.park_time)/ np.timedelta64(1, 'h'))+1))
    ].mean().values for _, row in tqdm(df.iterrows(), total = len(df))
], columns = ['Avg_Index_'+x for x in df_weather.columns], index = df.index)

df = pd.concat([df ,weather_average2], axis = 1)

# Add dist to station
with open('data/processed/Train_stations.pickle', 'rb') as handle:
    Stations = pickle.load(handle)

# Haversine function
def haversine(point1, point2):
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
    return c * r

df['dist_to_station'] = [min({k:haversine(v,r[1].values) for k,v in Stations.items()}.values()) for r in tqdm(df[['park_location_lat',	'park_location_long']].iterrows(), total = len(df))]

# Drop those far away
df = df[df['dist_to_station'] <= 7000]

# Drop some columns
df.drop(columns=['park_location_lat', 'park_location_long', 'leave_location_lat', 'leave_location_long', 'park_fuel', 'park_zone', 'moved', 'movedTF', 'park_time', 'reserve_time'], inplace = True)

# Save
df.to_csv('data/processed/SimpleNNData.csv')