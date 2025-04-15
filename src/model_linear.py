from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd

class ModelLinear:

    def __init__(self, n_features = 4, n_poly = 2, alpha = .1):
        """Initializes a linear regression model based on sklean.linear_model.PolynomialFeatures.
        Parameters
        ------------
        n_features: int
            Number of input features to use in sklearn "SelectKBest".
        n_poly: int
            Degree polynomial to fit
        alpha: float
            Regularization parameter L2
        """
        self.N_FEATURES = n_features
        self.N_POLY = n_poly
        self.ALPHA = alpha

    def linear_pipeline(self, fit_int = True):
        """Construct a standard pipeline used to train a linear model.
            Consists of feature selection, feature scaling, polynomial terms and regularization.
        """
        steps = [
            ('feature_selection', SelectKBest(r_regression, k=self.N_FEATURES)),
            ('scalar', StandardScaler()),
            ('poly', PolynomialFeatures(degree=self.N_POLY)),
            ('model', Ridge(alpha=self.ALPHA, fit_intercept=fit_int))
        ]
        return Pipeline(steps)

    def train_model(self, features = None, target = None):
        """Train a linear regression model with input feature and target.
           Performs feature scaling and regularization using sklearn.pipeline.Pipeline.

        Parameters
        ------------
        features: pandas.DataFrame, array
            Input data (target must be removed).
        target: pandas.DataFrame, array
            Target variable.

        Returns
        ------------
        sklearn.pipeline.Pipeline, x_test, y_test
        """

        x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=0,
                                                           test_size=0.2)

        pipeline = self.linear_pipeline()

        pipeline.fit(x_train, y_train)
        return pipeline, x_test, y_test

    def get_kbest(self, features = None, target = None, k = 5):
        """Get only the features which the model is trained on when SelectKBest method is used.
         Returns a dataframe with only the features used.
         """
        # Create and fit selector
        selector = SelectKBest(r_regression, k=k)
        selector.fit(features, target)
        # Get columns to keep and create new dataframe with those only
        cols_idxs = selector.get_support(indices=True)
        features_df_new = features.iloc[:, cols_idxs]
        return features_df_new

    def grid_search(self, features = None, target = None,
                    num_features = np.arange(2, 12),
                    n_poly_degree = np.arange(2,7),
                    alpha = np.array([0.1, 1, 10, 50]),
                    eval_metric = 'neg_mean_absolute_error'):
        """Use grid search to estimate optimal hyperparmeters for a linear model.
        Parameters
        ------------
        features: pandas.DataFrame, array
            Input data (target must be removed).
        target: pandas.DataFrame, array
            Target variable.
        num_features: list, array of int
            Potential number of features to try.
        n_poly_degree: list, array of int
            Potential polunomial degree to fit.
        alpha: list, array of int
            Potential regularization terms to try.
        eval_metric: str
            Metric used to compare different hyperparam combinations.
            Refer to sklearn.model_selection.GridSearchCV "scoring" param.

        Returns
        ------------
        GridSearchCV object, x_test, y_test
        The .predict method will use the best performing model.
        Refer to sklearn.model_selection.GridSearchCV docs for full description.
        Test set is also returned for final model evaluation.
        """

        x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=0,
                                                            test_size=0.2)
        pipeline = self.linear_pipeline()

        param_grid = {
            'feature_selection__k': num_features,
            'poly__degree': n_poly_degree,
            'model__alpha': alpha
        }

        grid = GridSearchCV(pipeline, param_grid, n_jobs=4, scoring=eval_metric)
        return grid.fit(x_train, y_train), x_test, y_test


    def train_loc_models(self, features = None, target = None,
                         loc_list = None, use_grid_search = True,
                         num_features=None, n_poly_degree=None, alpha = None,
                         eval_metric='neg_mean_absolute_error'
                         ):
        """Train a separate model for each location.
        Parameters
        ------------
        features: pandas.DataFrame
            Input data (target must be removed).
        target: pandas.DataFrame
            Target variable, needs to have "location" column in this case.
        loc_list: list of str
            List of locations to fit a model to.
            These must all be present in "features" and "target" DataFrames.

        use_grid_search: bool
            If True, run grid search for each location in loc_list.
            All params below are only used in this case, otherwise they have no effect.
        num_features: list, array of int
            Potential number of features to try.
        n_poly_degree: list, array of int
            Potential polunomial degree to fit.
        alpha: list, array of int
            Potential regularization terms to try.
        eval_metric: str
            Metric used to compare different hyperparam combinations.
            Refer to sklearn.model_selection.GridSearchCV "scoring" param.

        Returns
        ------------
        A dict containing xgboost.sklearn.XGBRegressor objects.
        If `use_grid_search` is True, then the test set pandas.DataFrame for each location is also included in the list.
        """

        model_list = {}
        x_test_list = []
        y_test_list = []


        for loc in loc_list:
            print('LOCATION:', loc)
            features_loc = features[features['location'] == loc].drop(columns=['location'])
            target_loc = target[target['location'] == loc].drop(columns=['location']).values.ravel()

            if use_grid_search:
                model,x_test,y_test = self.grid_search(features = features_loc, target = target_loc,
                                 num_features=num_features,
                                n_poly_degree=n_poly_degree,
                                alpha = alpha,
                                 eval_metric=eval_metric)
                model_list[loc] = model.best_estimator_
                x_test['location'] = loc
                x_test_list.append(x_test)
                dff = pd.DataFrame(target_loc, columns=['kWh'])
                dff.set_index(features_loc.index, inplace=True)
                dff['location'] = loc
                y_test_list.append(dff)
            else:
                model = self.train_model(features = features_loc, target = target_loc)
                model_list[loc] = model.best_estimator_
        if use_grid_search:
            return model_list,x_test_list,y_test_list
        else:
            return model_list