from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression, f_regression
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

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

    def train_model(self, x_train = None, y_train = None, fit_int = True):
        """Train a linear regression model with input x_train and target y_train.
           Performs feature scaling and regularization."""

        pipeline = self.linear_pipeline()

        pipeline.fit(x_train, y_train)
        return pipeline

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
                    num_features = np.arange(2, 10),
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
            Metric used to compare different models. Refer to sklearn.model_selection.GridSearchCV "scoring" param.

        Returns
        ------------
        GridSearchCV object.
        The .predict method will use the best performing model.
        Refer to sklearn.model_selection.GridSearchCV docs for full description.
        """

        x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=1,
                                                            test_size=0.2)
        pipeline = self.linear_pipeline()

        param_grid = {
            'feature_selection__k': num_features,
            'poly__degree': n_poly_degree,
            'model__alpha': alpha
        }

        grid = GridSearchCV(pipeline, param_grid, n_jobs=2, scoring=eval_metric)
        return grid.fit(x_train, y_train)