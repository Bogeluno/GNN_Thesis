# Load packages
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import glob
import numpy as np
import networkx as nx
import tqdm
import datetime
import geopandas as gpd
import rtree
import time

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

t = time.time()
print('Loading data')
# Load data
files = glob.glob("data/raw/SNData/*.csv")

dfs = []
for f in files:
    dfs.append(pd.read_csv(f, header=0, sep=";"))

Full_data = pd.concat(dfs,ignore_index=True) # Save this to interim
Full_data.to_csv('data/interim/Full_data.csv')
print('Data loaded! Initial cleaning started')
print(f'Time elapsed: {time.time()-t}')

# Drop 53 rows with na values
df = Full_data.dropna()

# Rename Columns to English
df. columns = ['Customer_Group', 'CustomerID', 'CarID', 'Engine', 'Rental_flag', 'RentalID', 'Rental_Usage_Type', 'Reservation_Time', 'End_Time', 'Revenue', 'Distance', 'Drives', 'Reservation_Minutes','Fuel_Start','Fuel_End','Start_Lat', 'Start_Long', 'End_Lat', 'End_Long']

# Fix type
df = df.astype({'CustomerID': 'int32', 'RentalID': 'int64'})

# Drop drives as it has no info (only ones)
df.drop(columns = 'Drives', inplace=True)

# Remove all rows with a CarID as it can not be used
df = df[df.CarID != '0']

# Remoce DK from CarID so the same car does not have two id's
df['CarID'] = df['CarID'].str.replace('DK','')

# Fix engine
# Engine has two types of missing values that is alligned
df["Engine"].replace({" ": '0'}, inplace=True)
# If a CarID already has an engine type assign that to the missing ones
Engine_dict = {c: df[df.CarID == c].Engine.nunique() for c in df[df.Engine == '0'].CarID.unique()}
for car, engine in Engine_dict.items():
    if engine == 1:
        continue
    True_Engine = [x for x in df[df.CarID == car].Engine.unique() if x!= '0'][0]
    df.loc[(df.CarID == car) & (df.Engine == '0'), 'Engine'] = True_Engine

# Populate the rest manual based on ID
df.loc[(df.CarID == 'WBA1R5104J7B14310') & (df.Engine == '0'), 'Engine'] = '118I'
df.loc[(df.CarID == 'WBA1R5104J5K58061') & (df.Engine == '0'), 'Engine'] = '118I'
df.loc[(df.CarID == 'WBA1R5103K7D66678') & (df.Engine == '0'), 'Engine'] = '118I'
df.loc[(df.CarID == 'WBY8P2105K7D70350') & (df.Engine == '0'), 'Engine'] = 'I3 120'
df.loc[(df.CarID == 'WBY8P2102K7D70287') & (df.Engine == '0'), 'Engine'] = 'I3 120'

# Convert time types
df['Reservation_Time'] = pd.to_datetime(df['Reservation_Time'], format="%d.%m.%Y %H:%M:%S")
df['End_Time'] = pd.to_datetime(df['End_Time'], format="%d.%m.%Y %H:%M:%S")

print('Enignes fixed. Merging trips based on the same customer using the same car.')
print(f'Time elapsed: {time.time()-t}')

## Fix trips where same user use same car
# Split data on Car level
CarID_dict = dict(iter(df.groupby('CarID')))

def fix_merges(dataframe, max_time_diff = 60):
    dataframe = dataframe.sort_values(by = 'Reservation_Time')
    # Get index where same customer uses the same car back to back
    diff0_iloc = [dataframe.index.get_loc(x) for x in dataframe.index[(dataframe.CustomerID.diff() == 0).tolist()]]

    # Find paris to be merged
    merge_pairs = [(idx-1,idx) for idx in diff0_iloc if dataframe.iloc[idx-1].End_Time+pd.to_timedelta(max_time_diff+dataframe.iloc[idx].Reservation_Minutes,'m') > dataframe.iloc[idx].Reservation_Time]

    # Model as graph to get cc
    graph_model = nx.Graph(merge_pairs)
    groups = [(min(cc),max(cc)) for cc in list(nx.connected_components(graph_model))]

    # Populate 
    for pair in groups:
        dataframe.loc[dataframe.index[pair[0]],['End_Time', 'Fuel_End', 'End_Lat', 'End_Long']] = dataframe.loc[dataframe.index[pair[1]],['End_Time', 'Fuel_End', 'End_Lat', 'End_Long']]


    # Delete now unwanted rows
    rows_to_delete = [x[1] for x in merge_pairs]
    dataframe.drop(index = [dataframe.index[x] for x in rows_to_delete], inplace = True)

    # Return fixed dataframe
    return dataframe

# Merge new datasets
dfs = []
for sub_df in tqdm.tqdm(CarID_dict.values()):
    dfs.append(fix_merges(sub_df))

df = pd.concat(dfs,ignore_index=False).sort_values(by = 'RentalID')

print('Merged! Now fixing wierd times')
print(f'Time elapsed: {time.time()-t}')
# Fixing wierd times
# Winter Time
WinterTimeIndex = df[(df.Reservation_Time > df.End_Time) & (df.End_Time.apply(lambda x: x.month) == 10) & (df.End_Time.apply(lambda x: x.hour) < 4)].index
WinterTimeIndexBack = [2179859, 1683947, 1683948]
WinterTimeIndexForward = [x for x in WinterTimeIndex if x not in WinterTimeIndexBack]
df.loc[WinterTimeIndexBack, 'Reservation_Time'] = df.loc[WinterTimeIndexBack, 'Reservation_Time'] - pd.to_timedelta(1,'h')
df.loc[WinterTimeIndexForward, 'End_Time'] = df.loc[WinterTimeIndexForward, 'End_Time'] + pd.to_timedelta(1,'h')

# Remove remaining 50 observations as they will not introduce more vacancy time
df.drop(index = df[df.Reservation_Time > df.End_Time].index, inplace = True)

print('Merging non-customers...')
print(f'Time elapsed: {time.time()-t}')
# Split data on Car level
CarID_dict = dict(iter(df.groupby('CarID')))

def merge_NC(dataframe):
    dataframe = dataframe.sort_values(by = 'Reservation_Time')
    # Get index where non_customer
    is_NC = dataframe.Customer_Group == 'Non_Customer'

    # Find paris to be merged
    merge_pairs = [(is_NC.index.get_loc(k1),is_NC.index.get_loc(k2)) for (k1, v1),(k2,v2) in zip(is_NC.iloc[:-1].iteritems(),is_NC.iloc[1:].iteritems()) if v1&v2]

    # Model as graph to get cc
    graph_model = nx.Graph(merge_pairs)
    groups = [(min(cc),max(cc)) for cc in list(nx.connected_components(graph_model))]

    # Populate 
    for pair in groups:
        dataframe.loc[dataframe.index[pair[0]],['End_Time', 'Fuel_End', 'End_Lat', 'End_Long']] = dataframe.loc[dataframe.index[pair[1]],['End_Time', 'Fuel_End', 'End_Lat', 'End_Long']]

    # Delete now unwanted rows
    rows_to_delete = [x[1] for x in merge_pairs]
    dataframe.drop(index = [dataframe.index[x] for x in rows_to_delete], inplace = True)

    # Return fixed dataframe
    return dataframe

# Merge new datasets
dfs = []
for sub_df in CarID_dict.values():
    dfs.append(merge_NC(sub_df))

df = pd.concat(dfs,ignore_index=False).sort_values(by = 'Reservation_Time')

print('Fixing overlapping times...')

# Overlapping trips
CarID_dict = dict(iter(df.groupby('CarID')))
tat = []
endtat0 = []
endtat1 = []
endtat2 = []
endtat3 = []

# Generate dataset with rows to be fixed
for car,dataf in CarID_dict.items():
    dataf = dataf.sort_values(by = 'Reservation_Time')
    tap = list( zip( dataf.iloc[np.where(dataf.Reservation_Time.iloc[1:].values<dataf.End_Time.iloc[:-1].values)[0]].Customer_Group.values, dataf.iloc[np.where(dataf.Reservation_Time.iloc[1:].values<dataf.End_Time.iloc[:-1].values)[0]+1].Customer_Group.values ) )
    tat.extend( tap )

    endtat0.extend( dataf.iloc[np.where(dataf.Reservation_Time.iloc[1:].values<dataf.End_Time.iloc[:-1].values)[0]].index )
    endtat1.extend( dataf.iloc[np.where(dataf.Reservation_Time.iloc[1:].values<dataf.End_Time.iloc[:-1].values)[0]].Customer_Group )
    endtat2.extend( dataf.iloc[np.where(dataf.Reservation_Time.iloc[1:].values<dataf.End_Time.iloc[:-1].values)[0]+1].Customer_Group )
    endtat3.extend( dataf.iloc[np.where(dataf.Reservation_Time.iloc[1:].values<dataf.End_Time.iloc[:-1].values)[0]].End_Lat )

overlap_df = pd.DataFrame(data=[endtat0,endtat1,endtat2,endtat3]).T

# Fix those with bad end_loc
fix_idx0 = overlap_df[(overlap_df[1] == 'Customer') & (overlap_df[3] < 1)][0].values
df.loc[fix_idx0, 'End_Time'] = df.loc[fix_idx0, 'Reservation_Time'].values + pd.to_timedelta(1,'m')

# Fix the other C-C to average of the two reservation times
fix_idxP = overlap_df[(overlap_df[1] == 'Customer') & (overlap_df[3] > 1)][0].values
    
for fix_idx in fix_idxP:
    # Get sub_df
    tmp_car_df = df[df.CarID == df.loc[fix_idx].CarID]
    
    # Get iloc in sub_df of to be fixed
    fix_iloc = tmp_car_df.index.get_loc(fix_idx)

    # Get end loc of curent and start of next
    end_loc = tmp_car_df.loc[fix_idx, ['End_Lat', 'End_Long']].values
    start_loc = tmp_car_df.loc[tmp_car_df.index[fix_iloc+1], ['Start_Lat', 'Start_Long']].values

    # If parked at same place adjust
    if haversine(end_loc, start_loc) < 100:
        avg_time = df.loc[fix_idx,'Reservation_Time'] + (df.loc[tmp_car_df.index[fix_iloc+1],'Reservation_Time'] - df.loc[fix_idx,'Reservation_Time']) / 2
        df.loc[fix_idx,'End_Time'] = avg_time

# Manual fixes/guestimates
df.loc[51903,'End_Time'] = pd.Timestamp("2016-11-03 20:00:00")
df.loc[661452,'End_Time'] = pd.Timestamp("2017-12-01 17:00:00")
df.loc[52806,'End_Time'] = pd.Timestamp("2016-11-05 08:00:10")
df.loc[2376045,'Reservation_Time'] = pd.Timestamp("2016-08-05 12:49:38")
df.loc[661513,'End_Time'] = pd.Timestamp("2017-12-02 16:16:24")
df.loc[784104,'End_Time'] = pd.Timestamp("2017-10-04 12:20:10")

df.drop(index = [22088, 25828, 809192, 664080, 1137264, 713741, 1604116, 2470015, 404202, 661521, 404308], inplace = True)

# Customer to non customer
fix_idxCNC = overlap_df[(overlap_df[1]=='Customer') & (overlap_df[2]=='Non_Customer')][0].values
for fix_idx in fix_idxCNC:
    # Get sub_df
    tmp_car_df = df[df.CarID == df.loc[fix_idx].CarID]
    
    # Get iloc in sub_df of to be fixed
    fix_iloc = tmp_car_df.index.get_loc(fix_idx)

    # Replace values
    df.loc[fix_idx,['End_Time', 'Fuel_End', 'End_Lat', 'End_Long']] = df.loc[tmp_car_df.index[fix_iloc+1],['End_Time', 'Fuel_End', 'End_Lat', 'End_Long']]

    # Drop the old NC row
    df.drop(tmp_car_df.index[fix_iloc+1], inplace = True)

# Non customer with broken end loc
fix_idx_NC0 = overlap_df[(overlap_df[1] ==  'Non_Customer') & (overlap_df[3] < 1)][0].values
to_drop = []

for fix_idx in fix_idx_NC0:
    # Get sub_df
    try:
        tmp_car_df = df[df.CarID == df.loc[fix_idx].CarID].sort_values(by = 'Reservation_Time')
    except:
        continue
    
    # Get iloc in sub_df of to be fixed
    fix_iloc = tmp_car_df.index.get_loc(fix_idx)

    # Get the two start locs
    start_loc0 = tmp_car_df.loc[tmp_car_df.index[fix_iloc-1], ['Start_Lat', 'Start_Long']].values
    start_loc1 = tmp_car_df.loc[fix_idx, ['Start_Lat', 'Start_Long']].values
    start_loc2 = tmp_car_df.loc[tmp_car_df.index[fix_iloc+1], ['Start_Lat', 'Start_Long']].values

    # If left same spot then drop
    if haversine(start_loc0, start_loc1) < 100:
        to_drop.append(fix_idx)
    if haversine(start_loc1, start_loc2) < 100:
        to_drop.append(fix_idx)
        
df.drop(index = to_drop, inplace = True)

# Fix last by using reservation minutes
fix_idx_RM = [x for x in overlap_df[(overlap_df[1] == 'Non_Customer' ) & (overlap_df[3] < 1)][0].values if x in df.index]

for fix_idx in fix_idx_RM:
    df.loc[fix_idx, 'End_Time'] = df.loc[fix_idx,'Reservation_Time']+pd.to_timedelta(df.loc[fix_idx,'Reservation_Minutes'], 'm')


# Fix trips with 0 time
idx_to_drop = df[df.Reservation_Time == df.End_Time].index
df.drop(index = idx_to_drop, inplace = True)

print('Overlaps fixed! Slicing out 2018-2019')
print(f'Time elapsed: {time.time()-t}')

# 2018+2019
df1819 = df[df.Reservation_Time >= pd.Timestamp("2018-01-01")]
df1819['TripDist'] = df1819.apply(lambda x: haversine([x['Start_Lat'], x['Start_Long']], [x['End_Lat'], x['End_Long']]), axis = 1)

print('Fixing wierd fuel')
# Manuel -1 fixes. Remaining -1 are start so no prob there
df1819.loc[516417,'Fuel_End'], df1819.loc[516674,'Fuel_Start'] = 78,78
df1819.loc[1423849,'Fuel_End'], df1819.loc[1424064,'Fuel_Start'] = 92,92

# Fix single missing
CarID_dict = CarID_dict = dict(iter(df1819.groupby('CarID')))
for sub_df in CarID_dict.values():
    sub_df = sub_df.sort_values(by = 'Reservation_Time')
    idx_fix_start = [sub_df.index.get_loc(x) for x in sub_df[sub_df.Fuel_Start == 0].index]

    if len(idx_fix_start) == 0:
        continue

    # Ensure the ones are lone
    idx_fix_start = [x for x in idx_fix_start if x-1 not in idx_fix_start and x+1 not in idx_fix_start]
    #idx_fix_end =[x-1 for x in idx_fix_start]

    for idx in idx_fix_start:
        # Get average
        replace_val = (sub_df.iloc[idx].Fuel_End+sub_df.iloc[idx-1].Fuel_Start)//2

        # Replace
        df1819.loc[sub_df.index[idx], 'Fuel_Start'] = replace_val
        df1819.loc[sub_df.index[idx-1], 'Fuel_End'] = replace_val

df1819.drop(index = df1819[(df1819.Fuel_Start <= 0) & ((df1819.TripDist <= 0.1) | ((df1819.TripDist > 5000000)))].index, inplace = True)
# Manual fixes
df1819.drop(index = [221640, 223431, 224544, 227363], inplace = True)
df1819.loc[2108824,'Fuel_Start'] = 48
df1819.loc[208967,'Fuel_End'] = 74
df1819.loc[241387,'Fuel_Start'] = 74
df1819.loc[241387,'Fuel_End'] = 69
df1819.loc[241859,'Fuel_Start'] = 69
df1819.loc[241859,'Fuel_End'] = 62
df1819.loc[1853849,'Fuel_Start'] = 100

df1819.loc[1614655,'Fuel_Start'] = 100

# Car WBY1Z21020V307871
df1819.loc[2194695, 'Fuel_Start'] = 79
df1819.loc[2195546, 'Fuel_Start'] = 73
df1819.loc[2195720, 'Fuel_Start'] = 67
df1819.loc[2195864, 'Fuel_Start'] = 54
df1819.loc[2195905, 'Fuel_Start'] = 51
df1819.loc[2195934, 'Fuel_Start'] = 48
df1819.loc[2196040, 'Fuel_Start'] = 46
df1819.loc[2196073, 'Fuel_Start'] = 44

# Car WBY8P2105K7D70350
df1819.loc[1810440, 'Fuel_Start'] = 100
df1819.loc[1811631, 'Fuel_Start'] = 84
df1819.loc[1812237, 'Fuel_Start'] = 73
df1819.loc[1813020, 'Fuel_Start'] = 70
df1819.loc[1814957, 'Fuel_Start'] = 68
df1819.loc[1815464, 'Fuel_Start'] = 63
df1819.loc[1818056, 'Fuel_Start'] = 61
df1819.loc[1818503, 'Fuel_Start'] = 55
df1819.loc[1818835, 'Fuel_Start'] = 54
df1819.loc[1821416, 'Fuel_Start'] = 51
df1819.loc[1822755, 'Fuel_Start'] = 50

df1819.loc[1993339, 'Fuel_Start'] = 15

# Merge NC again due to some being dropped
# Split data on Car level
CarID_dict = dict(iter(df1819.groupby('CarID')))

def merge_NC(dataframe):
    dataframe = dataframe.sort_values(by = 'Reservation_Time')
    # Get index where non_customer
    is_NC = dataframe.Customer_Group == 'Non_Customer'

    # Find paris to be merged
    merge_pairs = [(is_NC.index.get_loc(k1),is_NC.index.get_loc(k2)) for (k1, v1),(k2,v2) in zip(is_NC.iloc[:-1].iteritems(),is_NC.iloc[1:].iteritems()) if v1&v2]

    # Model as graph to get cc
    graph_model = nx.Graph(merge_pairs)
    groups = [(min(cc),max(cc)) for cc in list(nx.connected_components(graph_model))]

    # Populate 
    for pair in groups:
        dataframe.loc[dataframe.index[pair[0]],['End_Time', 'Fuel_End', 'End_Lat', 'End_Long']] = dataframe.loc[dataframe.index[pair[1]],['End_Time', 'Fuel_End', 'End_Lat', 'End_Long']]

    # Delete now unwanted rows
    rows_to_delete = [x[1] for x in merge_pairs]
    dataframe.drop(index = [dataframe.index[x] for x in rows_to_delete], inplace = True)

    # Return fixed dataframe
    return dataframe

# Merge new datasets
dfs = []
for sub_df in CarID_dict.values():
    dfs.append(merge_NC(sub_df))

df1819 = pd.concat(dfs,ignore_index=False).sort_values(by = 'Reservation_Time')

# Fix missing where start is available
CarID_dict = dict(iter(df1819.groupby('CarID')))
for sub_df in CarID_dict.values():
    sub_df = sub_df.sort_values(by = 'Reservation_Time')
    idx_fix_start = [sub_df.index.get_loc(x) for x in sub_df[sub_df.Fuel_Start <= 0].index if x > 0]

    if len(idx_fix_start) == 0:
        continue

    idx_fix_start = [x for x in idx_fix_start if sub_df.loc[sub_df.index[x-1],'Fuel_End']>0]

    for idx in idx_fix_start:
        df1819.loc[sub_df.index[idx], 'Fuel_Start'] = df1819.loc[sub_df.index[idx-1], 'Fuel_End']

# Fix ones where end-fuel is available from previous trip
for car in df1819[df1819.Fuel_Start <= 0].CarID.value_counts().keys():
    sub_df = df1819[df1819.CarID == car]
    idx_fix = [sub_df.index.get_loc(x) for x in sub_df[sub_df.Fuel_Start <= 0].index][0]
    
    if df1819.loc[sub_df.index[idx_fix],'Start_Lat'] == df1819.loc[sub_df.index[idx_fix-1],'Start_Lat']:
        df1819.loc[sub_df.index[idx_fix],'Fuel_Start'] = df1819.loc[sub_df.index[idx_fix-1],'Fuel_Start']

        df1819.drop(index = sub_df.index[idx_fix-1], inplace=True)

# Last manual fixes
df1819.loc[2096334, 'Fuel_Start'] = 88
df1819.loc[2013446, 'Fuel_Start'] = 100
df1819.loc[2544201, 'Fuel_Start'] = 15
df1819.loc[192176, 'Fuel_Start'] = 95
df1819.loc[2111646, 'Fuel_Start'] = 22
df1819.loc[2184762, 'Fuel_Start'] = 10
df1819.loc[2474381, 'Fuel_Start'] = 5
df1819.loc[2499013, 'Fuel_Start'] = 10
df1819.loc[1798819, 'Fuel_Start'] = 6
df1819.loc[518233, 'Fuel_Start'] = 97
df1819.loc[585544, 'Fuel_Start'] = 90

print('Fuel fixed. Now fixing 0,0 locations.')
print(f'Time elapsed: {time.time()-t}')


## Fix 0,0
# Start
for i, row in df1819[(df1819.Start_Lat < 5)].iterrows():
    # Skip if first instance as it will unaffect vacancy
    sub_df = df1819[df1819.CarID == row.CarID].sort_values('RentalID')
    err_index = sub_df.index.get_loc(i)
    if err_index == 0:
        continue

    # Populate based on previous end 
    df1819.loc[i, ['Start_Lat', 'Start_Long']] = sub_df.iloc[err_index-1].loc[['End_Lat','End_Long']].values

# End
for i, row in df1819[(df1819.End_Lat < 5)].iterrows():
    sub_df = df1819[df1819.CarID == row.CarID].sort_values('RentalID')
    err_index = sub_df.index.get_loc(i)

    # Will fail if last index
    try:
        df1819.loc[i, ['End_Lat', 'End_Long']] = sub_df.iloc[err_index+1].loc[['Start_Lat','Start_Long']].values
    except:
        continue
    

print('Next up is adding zones')
print(f'Time elapsed: {time.time()-t}')

## Add zones
# Load shapefile and set projection
shapefile = gpd.read_file("../Zonekort/LTM_Zone3/zones_level3.shp")
shapefile = shapefile.to_crs(epsg=4326)

# Create a geoDF with geometry as starting point
gdf_start = gpd.GeoDataFrame(df1819, geometry= gpd.points_from_xy(df1819.Start_Long, df1819.Start_Lat))

# Set projection
gdf_start = gdf_start.set_crs(epsg=4326)

# Populate zones based on which zone they are within
gdpj_start  = gpd.sjoin(gdf_start, shapefile, op='within')
df1819['Start_Zone'] = gdpj_start.zoneid

# Populate the rest based on which zone they are closest too
Start_zone_filler = {x: shapefile.zoneid[shapefile.distance(df1819.loc[x].geometry).sort_values().index[0]] for x in df1819.index[df1819['Start_Zone'].isna()]}
df1819['Start_Zone'] = df1819['Start_Zone'].fillna(Start_zone_filler)

# Create a geoDF with geometry as end point
gdf_end = gpd.GeoDataFrame(df1819, geometry= gpd.points_from_xy(df1819.End_Long, df1819.End_Lat))

# Set projection
gdf_end = gdf_end.set_crs(epsg=4326)

# Populate zones based on which zone they are within
gdpj_end  = gpd.sjoin(gdf_end, shapefile, op='within')
df1819['End_Zone'] = gdpj_end.zoneid

# Populate the rest based on which zone they are closest too
End_zone_filler = {x: shapefile.zoneid[shapefile.distance(df1819.loc[x].geometry).sort_values().index[0]] for x in df1819.index[df1819['End_Zone'].isna()]}
df1819['End_Zone'] = df1819['End_Zone'].fillna(End_zone_filler)

# Remove geomery type and make IDs int columns
df1819.drop(columns = 'geometry', inplace = True)
df1819 = df1819.astype({'CustomerID': 'int32', 'RentalID': 'int64', 'Start_Zone': 'int32','End_Zone': 'int32'})

# Sort dataset
df_sorted = df1819.sort_values("Reservation_Time")
df_sorted.CarID.nunique()
df_sorted.to_csv('data/processed/Full_data_set.csv')

print('Zones added. Now creating vacancy dataframe (takes around 20 minutes)...')
print(f'Time elapsed: {time.time()-t}')

CarID_dict = CarID_dict = dict(iter(df1819.groupby('CarID')))

data = []
for car, sub_df in tqdm.tqdm(CarID_dict.items()):
    for (_, row1), (_, row2) in zip(sub_df[:-1].iterrows(),sub_df[1:].iterrows()):
        park_time = row1['End_Time']
        reservation_time = row2['Reservation_Time']
        #start_time = row2['Start_Time']
        time_to_reservation = (row2['Reservation_Time']-row1['End_Time']).total_seconds()/3600
        #time_to_start = (row2['Start_Time']-row1['End_Time']).total_seconds()/3600
        park_location_lat = row1['End_Lat']
        park_location_long = row1['End_Long']
        leave_location_lat = row2['Start_Lat']
        leave_location_long = row2['Start_Long']
        park_zone = row1['End_Zone']
        leave_zone = row2['Start_Zone']
        park_fuel = row1['Fuel_End']
        leave_fuel = row2['Fuel_Start']
        engine = row1['Engine']
        moved = haversine(row1.loc[['End_Lat','End_Long']].values, row2.loc[['Start_Lat','Start_Long']].values) 
        prev_customer = row1['Customer_Group']
        next_customer = row2['Customer_Group']
        data.append([car, park_time,reservation_time, time_to_reservation, park_location_lat, park_location_long, leave_location_lat, leave_location_long, park_zone, leave_zone, park_fuel, leave_fuel, engine, moved, prev_customer, next_customer])

# Create new df
df_vacancy = pd.DataFrame(data = data, columns = ['car', 'park_time', 'reservation_time', 'time_to_reservation', 'park_location_lat', 'park_location_long', 'leave_location_lat', 'leave_location_long', 'park_zone', 'leave_zone', 'park_fuel', 'leave_fuel', 'engine', 'moved', 'prev_customer', 'next_customer'])

# Infer types
df_vacancy = df_vacancy.convert_dtypes()

# Save
df_vacancy.to_csv('data/processed/Vacancy_new.csv')
print(f'Done! Total time:{time.time()-t}')