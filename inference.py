import pandas as pd
import numpy as np
import pickle
from src import preprocessing, data_manager

datam = data_manager.DataManager(
    latitude=[51.01654, 51.085017, 50.961578, 51.05731, 51.002592, 51.073194, 51.029909, 51.104156, 51.076243, 51.029167],
    longitude=[-114.036713, -113.982745, -114.10772, -114.092517, -114.099734, -114.005373, -114.116951, -114.255363, -114.071349, -114.047147],
    hourly_vars=["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "shortwave_radiation"],
    models="best_match")

# load the max output at each location to fix bad predictions
df_max_out = pd.read_csv('data/max_output_each_location.csv')

# grab the latest 7-day forecast, clean it up to pass into model
df_weather = datam.get_data(mode = 'forecast')
preprop= preprocessing.Preprocessing(df_weather=df_weather)
df_weather = preprop.misc_weather()
df_weather = preprop.add_time_features(df_weather)

# load our trained models for each location
with open('trained_model_data/trained_models_xgboost.pkl', 'rb') as pickle_file:
    xgb_models = pickle.load(pickle_file)

feature_list = ['month','hour','day_of_year','temperature_2m',
                'relative_humidity_2m','rain','snowfall',
                'cloud_cover','cloud_cover_low','cloud_cover_mid',
                'cloud_cover_high','shortwave_radiation','temp_hum']

pred_df = pd.DataFrame(index = df_weather[df_weather['location'] == datam.locations[0]].index)

for loc in datam.locations:
    feature_order_model = xgb_models[loc].feature_names_in_

    df_bm_loc = df_weather[df_weather['location'] == loc]
    df_bm_pred = df_bm_loc.drop(columns=['location'])
    df_bm_pred['temp_hum'] = df_bm_pred['temperature_2m'].values * df_bm_pred['relative_humidity_2m'].values
    df_bm_pred = df_bm_pred[feature_list]

    predicted = xgb_models[loc].predict(df_bm_pred)

    # remove negative values
    predicted[predicted < 0] = 0
    # year-round night time should also be zero
    predicted = np.where(df_bm_pred['hour'].values < 6, 0.0, predicted)
    # values higher than the max ever recorded should be brought back down
    max_output = df_max_out[df_max_out['location'] == loc]['kWh'].values[0]
    predicted = np.where(predicted > max_output, max_output, predicted)
    pred_df[loc] = predicted

pred_df['sum'] = pred_df.sum(axis = 1)
pred_df.to_csv('data/prediction.csv')
