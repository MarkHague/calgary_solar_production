import pandas as pd
from src import preprocessing

df_production = pd.read_csv('Solar_Energy_Production_20250219.csv')
df_production['date'] = pd.to_datetime(df_production['date'])

df_weather = pd.read_csv('POWER_Point_Hourly_20180101_20241231_051d04N_0114d05W_LST.csv')
df_weather = df_weather.rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'})
df_weather['date'] = pd.to_datetime(df_weather[['year','month','day','hour']] )

preprocessing = preprocessing.Preprocessing(df_production=df_production,
                                            df_weather=df_weather)
# remove Telus Spark location - not enough data
df_production = df_production[df_production['name'] != "Telus Spark"]

# slice data to 2018 to end June 2024
df_production = preprocessing.slice_by_date(df=df_production)
df_weather = preprocessing.slice_by_date(df=df_weather)

df_merged = preprocessing.merge(df_production, df_weather)



