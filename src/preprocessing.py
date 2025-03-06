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
        join_method: str
            Join type passed to pandas.DataFrame.merge. See pandas docs.
        """
        df_merged = df_left.merge(df_right, on=merge_dim, how=join_method)
        return df_merged
