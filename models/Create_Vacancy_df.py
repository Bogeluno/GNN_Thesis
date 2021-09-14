import pandas as pd
import numpy as np

# Load data
df = pd.read_excel("data/raw/DTU - data til case_LTMZones1.xlsx")

# Convert dates to datetime format
df["Reservationstidspunkt"] = pd.to_datetime(df["Reservationstidspunkt"], format="%Y-%m-%d %H:%M:%S")
df["Start tidspunkt"] = pd.to_datetime(df["Start tidspunkt"], format="%Y-%m-%d %H:%M:%S")
df["Slut tidspunkt"] = pd.to_datetime(df["Slut tidspunkt"], format="%Y-%m-%d %H:%M:%S")
df["date"] = df["Start tidspunkt"].dt.date

# All English headers (and fix swap on reservation and start)
df.columns = ['TripID', 'CarID', 'PersonID', 'Latitude_Start', 'Longitude_Start', 'Latitude_End', 'Longitude_End', 'Start_Time', 'Reservation_Time', 'End_Time', 'Driver_Age', 'Driver_Gender', 'Battery_Start', 'Battery_End', 'KM_driven', 'Zone_Start','Zone_End','Date']

# Delete columns with "-"
df[df.CarID != "-"]

# Set manual index
df = df.set_index('TripID')

# Sort by start time
df_sorted = df.sort_values("Start_Time")

# Save interim
df_sorted.to_csv('data/interim/Sorted_data.csv')

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


data = []
for i, car in enumerate(df_sorted.CarID.unique()):
    if car == '-':
        continue
    car_sub_df = df_sorted[df_sorted.CarID == car]
    if not i%10:
        print(f'{i} cars processed' ,end="\r")
    for (_, row1), (_, row2) in zip(car_sub_df[:-1].iterrows(),car_sub_df[1:].iterrows()):
        park_time = row1['End_Time']
        reservation_time = row2['Reservation_Time']
        start_time = row2['Start_Time']
        time_to_reservation = (row2['Reservation_Time']-row1['End_Time']).total_seconds()/3600
        time_to_start = (row2['Start_Time']-row1['End_Time']).total_seconds()/3600
        park_location_lat = row1['Latitude_End']
        park_location_long = row1['Longitude_End']
        park_zone = row1['Zone_End']
        park_battery = row1['Battery_End']
        moved = haversine(row1.loc[['Latitude_End','Longitude_End']].values, row2.loc[['Latitude_Start','Longitude_Start']].values) 
        data.append([car, park_time,reservation_time, start_time, time_to_reservation, time_to_start, park_location_lat, park_location_long, park_zone, park_battery, moved])

# Create new df
df_vacancy = pd.DataFrame(data = data, columns = ['car', 'park_time', 'reservation_time', 'start_time','time_to_reservation', 'time_to_start', 'park_location_lat', 'park_location_long', 'park_zone', 'park_battery', 'moved'])

# Infer types
df_vacancy = df_vacancy.convert_dtypes()

# Save it
df_vacancy.to_csv('data/processed/Vacancy.csv')