import pandas as pd


class Preprocessing:

    def __init__(self, df_production=None, df_weather=None):
        self.df_production = df_production
        self.df_weather = df_weather

    def slice_by_date(self, df=None, start_date='20180101',
                   end_date='20240630', date_name = 'date'):
        """Slice a pandas dataframe based on start and end dates.
        Parameters
        ------------
        df: pandas.DataFrame
            Input dataframe to slice.
        start_date: str
            Starting date string in format 'YYYYMMDD'.
        end_date: str
            End date string in format 'YYYYMMDD'.
        date_name: str
            Variable name storing the date information. Must be datetime64[ns] dtype.
        """

        df = df.sort_values(date_name).set_index(date_name)
        return df[start_date:end_date]

    def merge(self, df1, df2, merge_dim = 'date'):
        """Merges 2 pandas dataframes along merge_dim.
           "merge_dim" becomes the new index of the merged dataframe.
        """
        df_merged = pd.merge_asof(df1, df2, on=merge_dim)
        return df_merged.set_index(merge_dim)

