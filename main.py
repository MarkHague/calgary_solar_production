import pandas as pd
import numpy as np
import pickle
from src import preprocessing, model_xgboost, model_linear

# ------------------- PREPROCESSING ------------------------------------------- #
# production data from city of calgary
df_production = pd.read_csv('data/Solar_Energy_Production_20250219.csv')
# historical weather data
df_weather = pd.read_csv('data/hourly_weather_data_ERA5_all_locs.csv')


preprop = preprocessing.Preprocessing(df_production=df_production,
                                            df_weather=df_weather)
# cleans up the production data
df_production = preprop.misc_production()

# some cleaning of the weather data
df_weather = preprop.misc_weather()

# slice data to 2018 to end 2024
df_production = preprop.slice_by_date(df=df_production, end_date='20241231')
df_weather = preprop.slice_by_date(df=df_weather, end_date='20241231')

df_merged = preprop.merge(df_production, df_weather, merge_dim = ['date','location'] )

# some feature engineering
df_merged = preprop.add_time_features(df_merged)

# seems like very low temps and high humidity probably lead to ice, which hurts output
df_merged['temp_hum'] = df_merged['temperature_2m'].values * df_merged['relative_humidity_2m'].values

# ------------------- MODEL TRAINING ------------------------------------------- #

# features we want to use for training
feature_list = ['kWh','location', 'month','hour','day_of_year','temperature_2m',
                'relative_humidity_2m','rain','snowfall',
                'cloud_cover','cloud_cover_low','cloud_cover_mid',
                'cloud_cover_high','shortwave_radiation','temp_hum']
location_list = df_merged['location'].unique()

features = df_merged[feature_list].drop(columns=['kWh'])
target = df_merged[['kWh','location']]

#  ---------------  Linear Model ------------------------ #

# linear_model = model_linear.ModelLinear()
# models_linear = linear_model.train_loc_models(features=features, target=target,
#                                              loc_list=location_list,
#                                              num_features=np.arange(3,10),
#                                              n_poly_degree=np.arange(3,7),
#                                              alpha=[0.1,1,10])

# we now have a list of tuples (1 for each location)
# each tuple has 3 elements: the model itself, x_test and y_test
# with this we can compute the test error for each location

#  ---------------  XGBoost ------------------------ #
# initialise an instance of the sklearn version of XGBoost tree ensemble
xgb_model = model_xgboost.ModelXGBoost()
# we train a separate ensemble for each location
# by default we use grid search to find the best combination of hyperparams

trained_model_xgboost, x_test_xgb, y_test_xgb = xgb_model.train_loc_models(
                                     features = features,
                                     target = target,
                                     loc_list= location_list,
                                     max_depth = [4,6,8,10,12],
                                     n_estimators = [30,50,100,150])

# we now have 3 lists
# each tuple has 3 elements: the model itself, x_test and y_test
# with this we can compute the test error for each location
with open('trained_model_data/trained_models_xgboost.pkl', 'wb') as f:
    pickle.dump(trained_model_xgboost, f)
pd.concat(x_test_xgb).to_csv('trained_model_data/x_test.csv')
pd.concat(y_test_xgb).to_csv('trained_model_data/y_test.csv')