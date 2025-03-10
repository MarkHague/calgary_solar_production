import pandas as pd


class Preprocessing:

    def __init__(self, df_production=None, df_weather=None):
        self.df_production = df_production
        self.df_weather = df_weather

    def slice_by_date(self, df=None, start_date='20180101',
                   end_date='20241231'):
        """Slice a pandas dataframe based on start and end dates.
        Parameters
        ------------
        df: pandas.DataFrame
            Input dataframe to slice.
        start_date: str
            Starting date string in format 'YYYYMMDD'.
        end_date: str
            End date string in format 'YYYYMMDD'.
        """

        return df[start_date:end_date]

    def merge(self, df_left = None, df_right = None, merge_dim ='date',
              join_method ='inner'):
        """Merges 2 pandas dataframes along merge_dim.
           "merge_dim" becomes the new index of the merged dataframe.
        Parameters
        ------------
        df_left: pandas.DataFrame
            Left input DataFrame.
        df_right: pandas.DataFrame
            Right input DataFrame.
        merge_dim: str
            Dimension along which to perform the merge. See pandas docs.
        join_method: label or list
            Join type passed to pandas.DataFrame.merge. See pandas docs.
        """
        df_merged = df_left.merge(df_right, on=merge_dim, how=join_method)
        return df_merged

    def misc_production(self):
        """A purely organizational method to clean-up the production data.
        """

        self.df_production['date'] = pd.to_datetime(self.df_production['date'])
        # change the names of the locations to something easier to work with
        self.df_production['location'] = self.df_production['name'].replace({"Bearspaw Water Treatment Plant": "Bearspaw",
                                                                   "Calgary Fire Hall Headquarters": "Fire Headquarters",
                                                                   "City of Calgary North Corporate Warehouse": "Corp. Warehouse",
                                                                   "Glenmore Water Treatment Plant": "Glenmore",
                                                                   "Hillhurst Sunnyside Community Association": "Hillhurst",
                                                                   "Manchester Building M": "Manchester",
                                                                   "Richmond - Knob Hill Community Hall": "Richmond",
                                                                   "Southland Leisure Centre": "Southland",
                                                                   "Whitehorn Multi-Service Centre": "Whitehorn"})

        # remove Telus Spark location - not enough data
        self.df_production = self.df_production[self.df_production['name'] != "Telus Spark"]
        # drop redundant name column, sort data by date for later merge with weather data
        self.df_production = self.df_production.drop(columns=['name']).set_index('date')
        self.df_production.sort_index(inplace=True)

        # Data after 25 September 2023 changed to units of Watts instead of kW
        date_start = pd.to_datetime('2023-09-25')
        date_end = pd.to_datetime('2025-12-31')
        # Create mask for rows between start and end dates
        mask = (self.df_production.index >= date_start) & (self.df_production.index <= date_end)
        # Apply the conversion
        self.df_production.loc[mask, 'kWh'] = self.df_production.loc[mask, 'kWh'] / 1000

        return self.df_production

    def add_time_features(self, df):
        """Convenience method to add the following time related features:
        "day_of_year", "month","y" """
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        df['hour'] = df.index.hour

        return df

    def misc_weather(self):
        self.df_weather.drop(columns=['Unnamed: 0'], inplace=True)
        self.df_weather['date'] = pd.to_datetime(self.df_weather['date']).dt.tz_localize(None)
        self.df_weather.set_index('date', inplace=True)
        self.df_weather.sort_index(inplace=True)

        return self.df_weather