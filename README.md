# Calgary Solar Output Prediction

## Overview
The goal of this project is to predict the energy output of the 
City of Calgary's 10 solar installations (i.e. locations).
Having an accurate forecast of how much energy will be produced 
is crucial for grid operators to anticipate fluctuations and 
maintain grid stability. 

For a detailed overview of the whole project, take a look at the notebook in `jupyter_notebooks/walkthrough.ipynb`

## Code Structure

You'll find the most important file in `src`, which includes:
- `preprocessing.py` - performs all the data cleaning. 
- `data_manager.py` - gets weather data from the API (historical and forecast).
- `model_linear.py` - implements a linear regression model.
- `model_xgboost.py` - implements the model Boosted ensemble of trees.
- `plot.py` - some useful plotting methods.

There's also `training.py` and `inference.py` which calls the methods in `src`.

## Data Overview
The data from City of Calgary came with the following variables:
- **Energy production** in kWh
- **Date** of each measurement
- **Name** of each installation
- **Address** of each installation
- **Installation date**

These where available at hourly frequency, and from 2018 to present.

I then combined this with weather data from the [Open-meteo](https://open-meteo.com).
After some research I settled on the following features to model solar production:
- Shortwave radiation 
- Temperature
- Relative Humidity
- Rainfall amount
- Snowfall amount
- Cloudy cover percentage

Finally, I added some engineered features to help the model:
- Hour
- Day of the year
- Month

Please check out the walkthrough notebook linked above for more details!

## Data cleaning

For the production data (i.e. the energy output from the solar panel) I performed the following:
1. Indexed the data with a proper `datetime` index. 
2. Removed one of installations (called Telus Spark) since its data ended in March 2018.
3. Corrected a serious mistake where the units for the energy production suddenly changed from from *kilo*-watts to *watts* in late 2023.

For the weather data I only needed to harmonize its `datetime` index to the production data's.
I was then able to merge the two datasets on the time column, using the `pandas.DataFrame.merge`.
This ensures that the correct weather data is linked to corresponding solar output measurement.

## Model Building

When building the models to predict the solar output I first decided to model *each location* separately.
So I ended up with 10 different models for each installation, which are then summed to get the total for the city at inference.
Why not just model the total output? 
Well, the results are much better if model each location, since each has its own set of unique characteristics.
These include things like its aspect (which way the panel face), shading (from trees and buildings) and how dusty the panels get (more on that in the walkthrough notebook).

I then trained 2 models (for each location):
1. **Linear regression** with polynomial features.
2. Boosted ensemble of trees using **XGBoost**.

I estimated the optimal hyperparamters for these using `sklearn`'s `GridSearchCV`.

With this, I obtained the following metrics (averaged across all locations):
- Linear regression
  - MAE = 12.9
  - R<sup>2</sup> = 0.76
- XGBoost regressor
  - MAE = 11.7
  - R<sup>2</sup> = 0.79

In the future I hope to improve these results. Here I list some options I could look into:
- Higher resolution historical radiation data.
- Access to the tilt of the solar panels (for more accurate solar radiation data).
- Access to the cleaning schedule of the panels (create a "days since cleaning" feature).
- Create a "days since last rain" feature, which might help with the dirt issue. 
- Access to any (un)planned shutdown of the panels (e.g. for maintenance) 

## Data availability

The power output data from City of Calgary I used in training is included in this repo under `data/Solar_Energy_Production_20250219.csv`.
This data is updated regularly [here](https://data.calgary.ca/Environment/Solar-Energy-Production/ytdn-2qsp/about_data).


