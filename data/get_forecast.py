import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

locations = ["Fire Headquarters", "Whitehorn", "Southland", "Hillhurst",
			 "Glenmore","Corp. Warehouse", "Richmond", "Bearspaw",
			 "CFD Firehall #7", "Manchester"]

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/gem"
params = {
	"latitude": [51.01654, 51.085017, 50.961578, 51.05731, 51.002592, 51.073194, 51.029909, 51.104156, 51.076243, 51.029167],
	"longitude": [-114.036713, -113.982745, -114.10772, -114.092517, -114.099734, -114.005373, -114.116951, -114.255363, -114.071349, -114.047147],
	"hourly": ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "shortwave_radiation"],
	"timezone": "America/Denver",
	"models": "best_match"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
# response = responses[0]
df_list = []
for i,response in enumerate(responses):
	print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Elevation {response.Elevation()} m asl")
	print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
	hourly_rain = hourly.Variables(2).ValuesAsNumpy()
	hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
	hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
	hourly_cloud_cover_low = hourly.Variables(5).ValuesAsNumpy()
	hourly_cloud_cover_mid = hourly.Variables(6).ValuesAsNumpy()
	hourly_cloud_cover_high = hourly.Variables(7).ValuesAsNumpy()
	hourly_shortwave_radiation = hourly.Variables(8).ValuesAsNumpy()

	# First, convert the timestamps to UTC pandas datetime objects
	utc_datetimes = pd.date_range(
		start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
		end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
		freq=pd.Timedelta(seconds=hourly.Interval()),
		inclusive="left"
	)

	# Then convert from UTC to the local timezone (America/Denver)
	local_datetimes = utc_datetimes.tz_convert('America/Denver')

	hourly_data = {"date": local_datetimes}

	hourly_data["temperature_2m"] = hourly_temperature_2m
	hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
	hourly_data["rain"] = hourly_rain
	hourly_data["snowfall"] = hourly_snowfall
	hourly_data["cloud_cover"] = hourly_cloud_cover
	hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
	hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
	hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
	hourly_data["shortwave_radiation"] = hourly_shortwave_radiation

	df = pd.DataFrame(data = hourly_data)
	df['location'] = locations[i]
	df_list.append(df)

df_out = pd.concat(df_list)
# added to get naive datetime objects
# df_out['date'] = df_out['date'].dt.tz_localize(None)

df_out.to_csv('forecast_hrdps_continental.csv')