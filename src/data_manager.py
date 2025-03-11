import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

class DataManager:

    def __init__(self, latitude = None, longitude = None, hourly_vars = None,
                 models = None):
        self.latitude = latitude
        self.longitude = longitude
        self.hourly_vars = hourly_vars
        self.models = models

        self.locations = ["Fire Headquarters", "Whitehorn", "Southland", "Hillhurst",
			 "Glenmore","Corp. Warehouse", "Richmond", "Bearspaw",
			 "CFD Firehall #7", "Manchester"]


    def get_data(self, mode = 'forecast', start_date = "2018-01-01",
                     end_date = "2024-12-31"):
        """Retrieve data from the open meteo API, based on initialization arguments of the class.

        Parameters
        ------------
        mode: str
            Either 'forecast' or 'historical'.
            For 'forecast', a 7-day forecast is returned.
            For 'historical', data from 'start_date' to 'end_date' is returned.
        start_date: str
            Start date of time series for historical data in format YYYY-MM-DD.
        end_date: str
            End date of time series for historical data in format YYYY-MM-DD.

        Returns
        ------------
        pandas.Dataframe
        
        """
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        if mode == 'forecast':
            url = "https://api.open-meteo.com/v1/gem"
        elif mode == 'historical':
            url = "https://archive-api.open-meteo.com/v1/archive"
        else:
            raise Exception("Invalid argument for parameter 'mode', only 'forecast' and 'historical' allowed. ")

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": self.hourly_vars,
            "timezone": "America/Denver",
            "models": self.models
        }

        if mode == 'historical':
            params["start_date"] = start_date
            params["end_date"] = end_date

        responses = openmeteo.weather_api(url, params=params)

        df_list = []
        for i, response in enumerate(responses):
            hourly = response.Hourly()

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

            for v, var in enumerate(self.hourly_vars):
                hourly_data[var] = hourly.Variables(v).ValuesAsNumpy()

            df = pd.DataFrame(data=hourly_data)
            df['location'] = self.locations[i]
            df_list.append(df)

        return pd.concat(df_list)
