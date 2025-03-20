import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import numpy as np

def scatter(dataframe = None, location = None,
                 x = 'shortwave_radiation', y = 'kWh',
                 color_var = None,
                 colormap = 'RdBu_r', color_fit = 'k'):
    """Scatter plot an optional third variable mapped to colormap."""

    if location is not None:
        dft = dataframe[dataframe['location'] == location]
    else:
        dft = dataframe


    if color_var is not None:
        ax = sns.scatterplot(data=dft, x=x, y=y, hue=color_var,
                             palette=colormap)
        rp = sns.regplot(data=dft, x=x, y=y, marker='',
                         color=color_fit)

        norm = plt.Normalize(dft[color_var].min(), dft[color_var].max())
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        cbar = ax.figure.colorbar(sm, ax=plt.gca())
        cbar.set_label(color_var)

    else:
        ax = sns.scatterplot(data=dft, x=x, y=y)
        rp = sns.regplot(data=dft, x=x, y=y, marker='', color=color_fit)


    # add r2 and p-value to upper left
    lin_reg = linregress(dft[x].values, dft[y].values)
    ax.text(.02, .95, '$R^2 = $' + '{0:.2f}'.format(lin_reg.rvalue ** 2), transform=ax.transAxes)
    ax.text(.02, .9, 'p = ' + '{0:.2f}'.format(lin_reg.pvalue), transform=ax.transAxes)

def bar_compare_years(df = None, loc = 'Bearspaw'):
    """Compare solar production with incoming energy from the sun for each year
    """

    df_loc = df[df['location'] == loc]
    dfp = df_loc.groupby(df_loc.index.year)[['kWh', 'shortwave_radiation']].mean()
    # dfp['shortwave_radiation'] = dfp['shortwave_radiation'] * (0.3)

    result_df = pd.DataFrame(columns=['power', 'source'])
    df1 = pd.DataFrame({
        'power': dfp['kWh'],
        'source': 'Solar Production'})

    df2 = pd.DataFrame({
        'power': dfp['shortwave_radiation'],
        'source': 'Incoming power from sun'})

    result_df = pd.concat([df1, df2], ignore_index=True)
    years = np.arange(2018, 2025)
    yr_ind = np.append(years, years)
    result_df['year'] = yr_ind
    result_df['power_norm'] = (result_df['power'] - result_df['power'].min()) / (
                result_df['power'].max() - result_df['power'].min())
    g = sns.catplot(
        data=result_df, kind="bar",
        x="year", y="power", hue="source",
        palette="dark", alpha=.6, height=6
    )

    g.set_axis_labels("", "Power")
    g.legend.set_title("")

def get_model_preds(x_test = None, y_test = None, model = None,
                        loc = None, df_max = None,
                        hue_var = 'shortwave_radiation'):
    """Get model predictions to compare test set labels.
    Parameters
    ------------
    x_test: pandas.DataFrame
    y_test: pandas.DataFrame
    model: sklearn model with predict method
    loc: str
    df_max: pandas.DataFrame
        Maximum output for each location
    hue_var: str
        Third variable to add to output - can be used in scatter method to shade points.

    Returns
    --------------
    pandas.DataFrame with predictions and labels as columns, with "hue_var" as additional column
    """

    x_test_linear = x_test[x_test['location'] == loc].drop(columns=['date', 'location'])
    preds = model[loc].predict(x_test_linear)
    # remove bad predictions
    max_output = df_max[df_max['location'] == loc]['kWh'].values[0]
    preds[preds < 0.0] = 0
    preds = np.where(preds > max_output, max_output, preds)
    preds = np.where(x_test_linear['hour'].values < 6, 0.0, preds)

    dfp = pd.DataFrame()
    dfp['Prediction'] = preds
    dfp['Actual'] = y_test[y_test['location'] == loc]['kWh'].values
    dfp[hue_var] = x_test[x_test['location'] == loc][hue_var]

    return dfp



