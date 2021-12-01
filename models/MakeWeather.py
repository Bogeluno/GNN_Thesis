# Load packages
import requests
from operator import itemgetter
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd

# Define API-key
Key = "037922f9-5511-4d01-8287-cf26dd4a8645"


# Get dates
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

start_date = datetime(2019,8,31)
end_date = datetime(2019,11,1)

dates = [single_date.strftime("%Y-%m-%d") for single_date in daterange(start_date, end_date)]


# Function to get weather data
def get_weather(param, dates, key):
    mt = []
    for day in dates:
        mt.append([itemgetter('from','value')(x.get('properties')) for x in requests.get(f'https://dmigw.govcloud.dk/v2/climateData/collections/municipalityValue/items?municipalityId=0101&datetime={day}T00:00:00Z/{day}T23:59:00Z&timeResolution=hour&parameterId={param}&api-key={key}').json().get('features')])

    All_days = [item for sublist in mt for item in sublist]

    idx, val = zip(*All_days)
    return pd.Series(val, idx).sort_index()

# Create dataframe
param_list = ['mean_temp','mean_wind_speed','acc_precip','bright_sunshine','mean_pressure','mean_relative_hum','mean_cloud_cover']
ss = [get_weather(param, dates, Key) for param in tqdm(param_list)]
weather_df = pd.concat(ss, axis=1, keys = param_list)

# Fix index
weather_df.index = pd.to_datetime(weather_df.index).tz_convert('EET').tz_localize(None)

weather_df = weather_df.iloc[21:-1]

# Save dataframe
weather_df.to_csv('data/processed/weather.csv')